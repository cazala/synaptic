import { invariant } from "./errors.js";
import {
  type ActivationId,
  type ConnectionId,
  type ConnectionSpec,
  type Delay,
  type GateSpec,
  type InitializerSpec,
  type ModelDefinition,
  type ParameterId,
  type ParameterSpec,
  type Precision,
  type TopologySpec,
  type UnitId,
  type UnitSpec,
} from "./types.js";
import { validateDefinition } from "./validate.js";

export interface Port {
  readonly units: readonly UnitId[];
  readonly shape: readonly number[];
  readonly outboundGate?: readonly UnitId[];
}

interface UnitOptions {
  readonly activation?: ActivationId;
  readonly stage?: number;
  readonly constant?: number;
  readonly label?: string;
}

interface ParameterOptions {
  readonly trainable?: boolean;
  readonly initializer?: InitializerSpec;
  readonly label?: string;
}

interface ConnectionOptions extends ParameterOptions {
  readonly delay?: Delay;
  readonly parameter?: ParameterId;
}

interface GateOptions {
  readonly delay?: Delay;
  readonly mode?: "all" | "one-to-one-target";
}

const DEFAULT_INITIALIZER: InitializerSpec = {
  kind: "uniform",
  min: -0.1,
  max: 0.1,
};

export class GraphBuilder {
  readonly #units: UnitSpec[] = [];
  readonly #connections: ConnectionSpec[] = [];
  readonly #gates: GateSpec[] = [];
  readonly #parameters: ParameterSpec[] = [];
  #currentStage = 0;
  #built = false;

  get currentStage(): number {
    return this.#currentStage;
  }

  stage(stage: number): this {
    this.#assertMutable();
    invariant(Number.isInteger(stage) && stage >= 0, "Stage must be a non-negative integer", "INVALID_STAGE");
    this.#currentStage = stage;
    return this;
  }

  nextStage(): number {
    this.#assertMutable();
    this.#currentStage += 1;
    return this.#currentStage;
  }

  units(count: number, options: UnitOptions = {}): UnitId[] {
    this.#assertMutable();
    invariant(Number.isInteger(count) && count > 0, "Unit count must be a positive integer", "INVALID_UNIT_COUNT");
    const stage = options.stage ?? this.#currentStage;
    const result: UnitId[] = [];
    for (let index = 0; index < count; index += 1) {
      const id = this.#units.length;
      this.#units.push({
        id,
        stage,
        activation: options.activation ?? "logistic",
        ...(options.constant === undefined ? {} : { constant: options.constant }),
        ...(options.label === undefined ? {} : { label: `${options.label}:${index}` }),
      });
      result.push(id);
    }
    return result;
  }

  input(size: number, shape: readonly number[] = [size]): Port {
    invariant(this.#units.length === 0, "Inputs must be declared before other units", "INPUT_ORDER");
    const units = this.units(size, { activation: "identity", stage: 0, label: "input" });
    return { units, shape: [...shape] };
  }

  constant(value: number, stage = 0): UnitId {
    const unit = this.units(1, {
      activation: "identity",
      constant: value,
      label: "constant",
      stage,
    })[0];
    invariant(unit !== undefined, "Failed to allocate constant unit", "INTERNAL_ALLOCATION_ERROR");
    return unit;
  }

  parameter(options: ParameterOptions = {}): ParameterId {
    this.#assertMutable();
    const id = this.#parameters.length;
    this.#parameters.push({
      id,
      trainable: options.trainable ?? true,
      initializer: options.initializer ?? DEFAULT_INITIALIZER,
      ...(options.label === undefined ? {} : { label: options.label }),
    });
    return id;
  }

  connect(
    from: readonly UnitId[],
    to: readonly UnitId[],
    mode: "all-to-all" | "one-to-one" = "all-to-all",
    options: ConnectionOptions = {},
  ): ConnectionId[] {
    this.#assertMutable();
    invariant(from.length > 0 && to.length > 0, "Connections require non-empty endpoints", "EMPTY_CONNECTION_GROUP");
    if (mode === "one-to-one") {
      invariant(from.length === to.length, "One-to-one groups must have equal sizes", "CONNECTION_SIZE_MISMATCH");
    }

    const pairs: Array<readonly [UnitId, UnitId]> = [];
    if (mode === "one-to-one") {
      for (let index = 0; index < from.length; index += 1) {
        const source = from[index];
        const target = to[index];
        invariant(source !== undefined && target !== undefined, "Invalid one-to-one endpoint", "UNKNOWN_UNIT");
        pairs.push([source, target]);
      }
    } else {
      for (const source of from) {
        for (const target of to) {
          pairs.push([source, target]);
        }
      }
    }

    return pairs.map(([source, target]) => {
      const sourceUnit = this.#units[source];
      const targetUnit = this.#units[target];
      invariant(sourceUnit && targetUnit, "Connection references an unknown unit", "UNKNOWN_UNIT");
      const delay = options.delay ?? (sourceUnit.stage < targetUnit.stage ? 0 : 1);
      const parameter = options.parameter ?? this.parameter(options);
      const id = this.#connections.length;
      this.#connections.push({ id, from: source, to: target, parameter, delay });
      return id;
    });
  }

  gate(
    gaters: readonly UnitId[],
    connections: readonly ConnectionId[],
    options: GateOptions = {},
  ): void {
    this.#assertMutable();
    invariant(gaters.length > 0 && connections.length > 0, "Gates require non-empty inputs", "EMPTY_GATE_GROUP");
    const mode = options.mode ?? "all";
    if (mode === "all") {
      invariant(gaters.length === 1, "All-mode gating requires exactly one gater", "GATE_SIZE_MISMATCH");
    }

    const targetOrder = [...new Set(connections.map((id) => this.#connections[id]?.to))];
    if (mode === "one-to-one-target") {
      invariant(gaters.length === targetOrder.length, "Gaters must match unique connection targets", "GATE_SIZE_MISMATCH");
    }

    for (const connectionId of connections) {
      const connection = this.#connections[connectionId];
      invariant(connection, "Gate references an unknown connection", "UNKNOWN_CONNECTION");
      const targetIndex = targetOrder.indexOf(connection.to);
      const gaterId = mode === "all" ? gaters[0] : gaters[targetIndex];
      const gater = gaterId === undefined ? undefined : this.#units[gaterId];
      const target = this.#units[connection.to];
      invariant(gater && target, "Gate references an unknown unit", "UNKNOWN_UNIT");
      const delay = options.delay ?? (gater.stage < target.stage ? 0 : 1);
      this.#gates.push({ connection: connectionId, gater: gater.id, delay });
    }
  }

  build(options: {
    readonly inputs: Port | readonly UnitId[];
    readonly outputs: Port | readonly UnitId[];
    readonly precision?: Precision;
    readonly metadata?: Readonly<Record<string, unknown>>;
  }): ModelDefinition {
    this.#assertMutable();
    this.#built = true;
    const inputs = "units" in options.inputs ? options.inputs.units : options.inputs;
    const outputs = "units" in options.outputs ? options.outputs.units : options.outputs;
    const topology: TopologySpec = {
      version: 1,
      units: this.#units.map((unit) => Object.freeze(unit)),
      connections: this.#connections.map((connection) => Object.freeze(connection)),
      gates: this.#gates.map((gate) => Object.freeze(gate)),
      parameters: this.#parameters.map((parameter) => Object.freeze(parameter)),
      inputs: [...inputs],
      outputs: [...outputs],
      ...(options.metadata === undefined ? {} : { metadata: options.metadata }),
    };
    const definition: ModelDefinition = {
      format: "synaptic-model",
      formatVersion: 1,
      algorithm: "lstm-g",
      algorithmVersion: 1,
      precision: options.precision ?? "f32",
      topology,
      ...(options.metadata === undefined ? {} : { metadata: options.metadata }),
    };
    validateDefinition(definition);
    return Object.freeze(definition);
  }

  #assertMutable(): void {
    invariant(!this.#built, "A graph cannot change after build()", "GRAPH_ALREADY_BUILT");
  }
}
