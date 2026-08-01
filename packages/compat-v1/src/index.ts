import {
  SynapticError,
  compilePlan,
  createRuntimeState,
  invariant,
  validateDefinition,
  type ActivationId,
  type ConnectionSpec,
  type GateSpec,
  type ModelCheckpoint,
  type ModelDefinition,
  type ModelSnapshot,
  type ParameterSpec,
  type UnitSpec,
} from "@synaptic/core";

export type LegacyFormat = "synaptic-v1";

export interface LegacyImportWarning {
  readonly code: string;
  readonly message: string;
  readonly source?: Readonly<Record<string, unknown>>;
}

export interface LegacyImportResult<T> {
  readonly value: T;
  readonly warnings: readonly LegacyImportWarning[];
}

export interface LegacyImportedModel {
  readonly format: LegacyFormat;
  readonly definition: ModelDefinition;
  readonly snapshot: ModelSnapshot;
  readonly checkpoint: ModelCheckpoint;
}

interface LegacyV1Neuron {
  readonly state?: number;
  readonly old?: number;
  readonly activation?: number;
  readonly bias?: number;
  readonly layer: "input" | "output" | number;
  readonly squash?: string;
  readonly trace?: {
    readonly elegibility?: Readonly<Record<string, number>>;
    readonly extended?: Readonly<Record<string, Readonly<Record<string, number>>>>;
  };
}

interface LegacyV1Connection {
  readonly from: number;
  readonly to: number;
  readonly weight: number;
  readonly gater?: number | null;
}

interface LegacyV1Network {
  readonly neurons: readonly LegacyV1Neuron[];
  readonly connections: readonly LegacyV1Connection[];
}

interface LegacyConnectionEndpoints {
  readonly from: number;
  readonly to: number;
}

interface LegacyGate extends LegacyConnectionEndpoints {
  readonly gater: number;
}

interface PortableParts {
  readonly definition: ModelDefinition;
  readonly parameters: Float32Array;
}

const V1_ACTIVATIONS: Readonly<Record<string, ActivationId>> = {
  HLIM: "step",
  IDENTITY: "identity",
  LOGISTIC: "logistic",
  RELU: "relu",
  TANH: "tanh",
};

function finite(value: number | undefined, fallback = 0): number {
  return value !== undefined && Number.isFinite(value) ? value : fallback;
}

function endpointKey(from: number, to: number): string {
  return `${from}:${to}`;
}

function assignStages(
  groups: readonly (readonly number[])[],
  connections: readonly LegacyConnectionEndpoints[],
  warnings: LegacyImportWarning[],
): Uint32Array {
  const highestUnit = Math.max(-1, ...groups.flat());
  const stages = new Uint32Array(highestUnit + 1);
  let stage = 0;
  for (let groupIndex = 0; groupIndex < groups.length; groupIndex += 1) {
    const group = groups[groupIndex] ?? [];
    if (groupIndex === 0) {
      for (const unit of group) {
        stages[unit] = 0;
      }
      continue;
    }
    stage += 1;
    const members = new Set(group);
    const orderSensitive = connections.some(
      (connection) =>
        connection.from !== connection.to
        && members.has(connection.from)
        && members.has(connection.to),
    );
    for (const unit of group) {
      stages[unit] = stage;
      if (orderSensitive) {
        stage += 1;
      }
    }
    if (orderSensitive) {
      warnings.push({
        code: "LEGACY_SERIAL_LAYER_ORDER",
        message: `Layer ${groupIndex} contains intra-layer connections and was expanded into serial stages`,
      });
      stage -= 1;
    }
  }
  return stages;
}

function buildPortable(
  units: readonly UnitSpec[],
  connections: readonly {
    readonly from: number;
    readonly to: number;
    readonly weight: number;
    readonly trainable?: boolean;
  }[],
  gates: readonly LegacyGate[],
  inputs: readonly number[],
  outputs: readonly number[],
  metadata: Readonly<Record<string, unknown>>,
): PortableParts {
  const parameters: ParameterSpec[] = [];
  const connectionSpecs: ConnectionSpec[] = [];
  const values = new Float32Array(connections.length);
  const connectionByEndpoints = new Map<string, number>();

  for (const connection of connections) {
    const id = connectionSpecs.length;
    const parameter = parameters.length;
    const from = units[connection.from];
    const to = units[connection.to];
    invariant(from !== undefined && to !== undefined, "Legacy connection references an unknown unit", "INVALID_LEGACY_CONNECTION");
    parameters.push({
      id: parameter,
      trainable: connection.trainable ?? connection.from !== connection.to,
      initializer: { kind: "constant", value: connection.weight },
    });
    values[parameter] = connection.weight;
    connectionSpecs.push({
      id,
      from: connection.from,
      to: connection.to,
      parameter,
      delay: from.stage < to.stage ? 0 : 1,
    });
    connectionByEndpoints.set(endpointKey(connection.from, connection.to), id);
  }

  const gateSpecs: GateSpec[] = gates.map((gate) => {
    const connection = connectionByEndpoints.get(endpointKey(gate.from, gate.to));
    invariant(connection !== undefined, "Legacy gate has no matching connection", "INVALID_LEGACY_GATE");
    const target = units[gate.to];
    const gater = units[gate.gater];
    invariant(target !== undefined && gater !== undefined, "Legacy gate references an unknown unit", "INVALID_LEGACY_GATE");
    return {
      connection,
      gater: gate.gater,
      delay: gater.stage < target.stage ? 0 : 1,
    };
  });

  const definition: ModelDefinition = {
    format: "synaptic-model",
    formatVersion: 1,
    algorithm: "lstm-g",
    algorithmVersion: 1,
    precision: "f32",
    topology: {
      version: 1,
      units,
      connections: connectionSpecs,
      gates: gateSpecs,
      parameters,
      inputs,
      outputs,
      metadata,
    },
    metadata,
  };
  validateDefinition(definition);
  return { definition, parameters: values };
}

function checkpointFromArrays(
  parts: PortableParts,
  source: {
    readonly state?: readonly number[];
    readonly activation?: readonly number[];
    readonly previousActivation?: readonly number[];
    readonly constants?: ReadonlyMap<number, number>;
  },
): ModelCheckpoint {
  const plan = compilePlan(parts.definition);
  const runtime = createRuntimeState(plan, parts.definition.precision);
  for (let unit = 0; unit < plan.unitCount; unit += 1) {
    runtime.state[unit] = finite(source.state?.[unit]);
    runtime.activation[unit] = finite(source.activation?.[unit]);
    runtime.previousActivation[unit] = finite(source.previousActivation?.[unit]);
    const constant = source.constants?.get(unit);
    if (constant !== undefined) {
      runtime.state[unit] = constant;
      runtime.activation[unit] = constant;
    }
  }
  return {
    definition: parts.definition,
    parameters: parts.parameters.slice(),
    runtime,
  };
}

export function importV1Network(
  source: LegacyV1Network,
): LegacyImportResult<LegacyImportedModel> {
  invariant(Array.isArray(source.neurons) && source.neurons.length > 0, "Legacy v1 JSON has no neurons", "INVALID_LEGACY_ARTIFACT");
  invariant(Array.isArray(source.connections), "Legacy v1 JSON has no connection list", "INVALID_LEGACY_ARTIFACT");
  const warnings: LegacyImportWarning[] = [];
  const inputIds = source.neurons
    .map((neuron, id) => ({ neuron, id }))
    .filter(({ neuron }) => neuron.layer === "input")
    .map(({ id }) => id);
  const outputIds = source.neurons
    .map((neuron, id) => ({ neuron, id }))
    .filter(({ neuron }) => neuron.layer === "output")
    .map(({ id }) => id);
  invariant(inputIds.length > 0 && outputIds.length > 0, "Legacy v1 JSON is missing input or output neurons", "INVALID_LEGACY_ARTIFACT");

  const hiddenLayers = [...new Set(
    source.neurons
      .map((neuron) => neuron.layer)
      .filter((layer): layer is number => typeof layer === "number"),
  )].sort((left, right) => left - right);
  const groups: number[][] = [
    inputIds,
    ...hiddenLayers.map((layer) =>
      source.neurons
        .map((neuron, id) => ({ neuron, id }))
        .filter(({ neuron }) => neuron.layer === layer)
        .map(({ id }) => id)),
    outputIds,
  ];
  const stages = assignStages(groups, source.connections, warnings);
  const units: UnitSpec[] = source.neurons.map((neuron, id) => {
    const activation = V1_ACTIVATIONS[neuron.squash ?? "LOGISTIC"];
    invariant(activation !== undefined, `Unsupported v1 squash function: ${neuron.squash}`, "UNSUPPORTED_LEGACY_ACTIVATION");
    return {
      id,
      stage: stages[id] ?? 0,
      activation,
      label: `v1:${String(neuron.layer)}:${id}`,
    };
  });

  const biasUnit = units.length;
  units.push({
    id: biasUnit,
    stage: 0,
    activation: "identity",
    constant: 1,
    label: "v1:bias",
  });
  const importedConnections = source.connections.map((connection) => ({
    from: connection.from,
    to: connection.to,
    weight: finite(connection.weight),
    trainable: connection.from !== connection.to,
  }));
  for (let unit = 0; unit < source.neurons.length; unit += 1) {
    if (!inputIds.includes(unit)) {
      importedConnections.push({
        from: biasUnit,
        to: unit,
        weight: finite(source.neurons[unit]?.bias),
        trainable: true,
      });
    }
  }
  const gates: LegacyGate[] = source.connections.flatMap((connection) =>
    connection.gater === null || connection.gater === undefined
      ? []
      : [{ from: connection.from, to: connection.to, gater: connection.gater }]);
  const metadata = { sourceFormat: "synaptic-v1" as const };
  const parts = buildPortable(
    units,
    importedConnections,
    gates,
    inputIds,
    outputIds,
    metadata,
  );
  const constants = new Map([[biasUnit, 1]]);
  const checkpoint = checkpointFromArrays(parts, {
    state: source.neurons.map((neuron) => finite(neuron.state)),
    activation: source.neurons.map((neuron) => finite(neuron.activation)),
    previousActivation: source.neurons.map((neuron) => finite(neuron.old)),
    constants,
  });
  if (source.neurons.some((neuron) =>
    Object.keys(neuron.trace?.elegibility ?? {}).length > 0
    || Object.keys(neuron.trace?.extended ?? {}).length > 0)) {
    warnings.push({
      code: "LEGACY_TRACE_KEYS_DROPPED",
      message: "v1 trace dictionaries use runtime connection IDs that are not present in Network.toJSON()",
    });
  }
  const snapshot: ModelSnapshot = {
    definition: parts.definition,
    parameters: parts.parameters.slice(),
  };
  return {
    value: {
      format: "synaptic-v1",
      definition: parts.definition,
      snapshot,
      checkpoint,
    },
    warnings,
  };
}

export function detectLegacyFormat(source: unknown): LegacyFormat {
  invariant(typeof source === "object" && source !== null, "Legacy artifact must be an object", "INVALID_LEGACY_ARTIFACT");
  if ("neurons" in source) {
    return "synaptic-v1";
  }
  throw new SynapticError("Unknown legacy artifact format", "UNKNOWN_LEGACY_FORMAT");
}

export function importLegacy(
  source: unknown,
): LegacyImportResult<LegacyImportedModel> {
  detectLegacyFormat(source);
  return importV1Network(source as LegacyV1Network);
}
