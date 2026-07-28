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
  type Precision,
  type UnitSpec,
} from "@synaptic/core";

export type LegacyFormat = "synaptic-v1" | "synaptic2-engine";

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

interface Synaptic2Connection {
  readonly from: number;
  readonly to: number;
}

interface Synaptic2Gate extends Synaptic2Connection {
  readonly gater: number;
}

interface Synaptic2Engine {
  readonly size: number;
  readonly layers: readonly (readonly number[])[];
  readonly activationFunction: readonly number[];
  readonly connections: readonly Synaptic2Connection[];
  readonly gates?: readonly Synaptic2Gate[];
  readonly weight: readonly (readonly number[])[];
  readonly state?: readonly number[];
  readonly activation?: readonly number[];
  readonly derivative?: readonly number[];
  readonly elegibilityTrace?: readonly (readonly number[])[];
  readonly extendedElegibilityTrace?: readonly (readonly (readonly number[])[])[];
  readonly errorResponsibility?: readonly number[];
  readonly projectedErrorResponsibility?: readonly number[];
  readonly gatedErrorResponsibility?: readonly number[];
  readonly biasUnit?: number | null;
}

interface PortableParts {
  readonly definition: ModelDefinition;
  readonly parameters: Float32Array;
  readonly connectionByEndpoints: ReadonlyMap<string, number>;
}

const V1_ACTIVATIONS: Readonly<Record<string, ActivationId>> = {
  HLIM: "step",
  IDENTITY: "identity",
  LOGISTIC: "logistic",
  RELU: "relu",
  TANH: "tanh",
};

const SYNAPTIC2_ACTIVATIONS: Readonly<Record<number, ActivationId>> = {
  0: "logistic",
  1: "tanh",
  2: "relu",
  5: "identity",
};

function finite(value: number | undefined, fallback = 0): number {
  return value !== undefined && Number.isFinite(value) ? value : fallback;
}

function endpointKey(from: number, to: number): string {
  return `${from}:${to}`;
}

function assignStages(
  groups: readonly (readonly number[])[],
  connections: readonly Synaptic2Connection[],
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
  gates: readonly Synaptic2Gate[],
  inputs: readonly number[],
  outputs: readonly number[],
  precision: Precision,
  metadata: Readonly<Record<string, unknown>>,
): PortableParts {
  const parameters: ParameterSpec[] = [];
  const connectionSpecs: ConnectionSpec[] = [];
  const values = precision === "f64"
    ? new Float64Array(connections.length)
    : new Float32Array(connections.length);
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
    precision,
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
  return { definition, parameters: values as Float32Array, connectionByEndpoints };
}

function checkpointFromArrays(
  parts: PortableParts,
  source: {
    readonly state?: readonly number[];
    readonly activation?: readonly number[];
    readonly previousActivation?: readonly number[];
    readonly derivative?: readonly number[];
    readonly eligibility?: (connection: number, from: number, to: number) => number;
    readonly extended?: (connection: number, from: number, to: number, target: number) => number;
    readonly projectedError?: readonly number[];
    readonly gatedError?: readonly number[];
    readonly error?: readonly number[];
    readonly constants?: ReadonlyMap<number, number>;
  },
): ModelCheckpoint {
  const plan = compilePlan(parts.definition);
  const runtime = createRuntimeState(plan, parts.definition.precision);
  for (let unit = 0; unit < plan.unitCount; unit += 1) {
    runtime.state[unit] = finite(source.state?.[unit]);
    runtime.activation[unit] = finite(source.activation?.[unit]);
    runtime.previousActivation[unit] = finite(source.previousActivation?.[unit]);
    runtime.derivative[unit] = finite(source.derivative?.[unit]);
    const constant = source.constants?.get(unit);
    if (constant !== undefined) {
      runtime.state[unit] = constant;
      runtime.activation[unit] = constant;
    }
  }
  for (let connection = 0; connection < plan.connectionCount; connection += 1) {
    runtime.eligibilityTrace[connection] = finite(source.eligibility?.(
      connection,
      plan.connectionFrom[connection] ?? 0,
      plan.connectionTo[connection] ?? 0,
    ));
  }
  for (let trace = 0; trace < plan.extendedTraceTarget.length; trace += 1) {
    const connection = plan.extendedTraceConnection[trace] ?? 0;
    runtime.extendedEligibilityTrace[trace] = finite(source.extended?.(
      connection,
      plan.connectionFrom[connection] ?? 0,
      plan.connectionTo[connection] ?? 0,
      plan.extendedTraceTarget[trace] ?? 0,
    ));
  }
  for (let unit = 0; unit < plan.unitCount; unit += 1) {
    runtime.projectedError[unit] = finite(source.projectedError?.[unit]);
    runtime.gatedError[unit] = finite(source.gatedError?.[unit]);
    runtime.error[unit] = finite(source.error?.[unit]);
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
  const gates: Synaptic2Gate[] = source.connections.flatMap((connection) =>
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
    "f32",
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

export function importSynaptic2Engine(
  source: Synaptic2Engine,
): LegacyImportResult<LegacyImportedModel> {
  invariant(Number.isInteger(source.size) && source.size > 0, "synaptic2 engine has an invalid size", "INVALID_LEGACY_ARTIFACT");
  invariant(source.layers.length >= 2, "synaptic2 engine needs input and output layers", "INVALID_LEGACY_ARTIFACT");
  const warnings: LegacyImportWarning[] = [];
  const stages = assignStages(source.layers, source.connections, warnings);
  const layerMembers = new Set(source.layers.flat());
  const units: UnitSpec[] = [];
  for (let id = 0; id < source.size; id += 1) {
    const activationCode = source.activationFunction[id] ?? 0;
    const activation = SYNAPTIC2_ACTIVATIONS[activationCode];
    invariant(activation !== undefined, `Unsupported synaptic2 activation code: ${activationCode}`, "UNSUPPORTED_LEGACY_ACTIVATION");
    const constant = id === source.biasUnit ? 1 : undefined;
    units.push({
      id,
      stage: layerMembers.has(id) ? (stages[id] ?? 0) : 0,
      activation,
      ...(constant === undefined ? {} : { constant }),
      label: id === source.biasUnit ? "synaptic2:bias" : `synaptic2:${id}`,
    });
  }
  const importedConnections = source.connections.map((connection) => ({
    from: connection.from,
    to: connection.to,
    weight: finite(source.weight[connection.to]?.[connection.from]),
    trainable: connection.from !== connection.to,
  }));
  const metadata = { sourceFormat: "synaptic2-engine" as const };
  const parts = buildPortable(
    units,
    importedConnections,
    source.gates ?? [],
    source.layers[0] ?? [],
    source.layers[source.layers.length - 1] ?? [],
    "f32",
    metadata,
  );
  const constants = source.biasUnit === null || source.biasUnit === undefined
    ? new Map<number, number>()
    : new Map([[source.biasUnit, 1]]);
  const checkpoint = checkpointFromArrays(parts, {
    state: source.state,
    activation: source.activation,
    derivative: source.derivative,
    projectedError: source.projectedErrorResponsibility,
    gatedError: source.gatedErrorResponsibility,
    error: source.errorResponsibility,
    constants,
    eligibility: (_connection, from, to) =>
      finite(source.elegibilityTrace?.[to]?.[from]),
    extended: (_connection, from, to, target) =>
      finite(source.extendedElegibilityTrace?.[to]?.[from]?.[target]),
  });
  const snapshot: ModelSnapshot = {
    definition: parts.definition,
    parameters: parts.parameters.slice(),
  };
  return {
    value: {
      format: "synaptic2-engine",
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
  if ("activationFunction" in source && "layers" in source) {
    return "synaptic2-engine";
  }
  throw new SynapticError("Unknown legacy artifact format", "UNKNOWN_LEGACY_FORMAT");
}

export function importLegacy(
  source: unknown,
): LegacyImportResult<LegacyImportedModel> {
  return detectLegacyFormat(source) === "synaptic-v1"
    ? importV1Network(source as LegacyV1Network)
    : importSynaptic2Engine(source as Synaptic2Engine);
}
