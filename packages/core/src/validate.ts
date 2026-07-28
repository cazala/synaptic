import { invariant } from "./errors.js";
import {
  ACTIVATION_CODES,
  type ConnectionSpec,
  type GateSpec,
  type ModelDefinition,
  type TopologySpec,
} from "./types.js";

function validateContiguousIds(
  values: readonly { readonly id: number }[],
  kind: string,
): void {
  for (let index = 0; index < values.length; index += 1) {
    invariant(
      values[index]?.id === index,
      `${kind} IDs must be contiguous and ordered`,
      "INVALID_ID_SEQUENCE",
      { expected: index, received: values[index]?.id, kind },
    );
  }
}

function validateDelay(
  topology: TopologySpec,
  connection: ConnectionSpec,
): void {
  const from = topology.units[connection.from];
  const to = topology.units[connection.to];
  invariant(from && to, "Connection references an unknown unit", "UNKNOWN_UNIT", {
    connection: connection.id,
  });
  if (connection.delay === 0) {
    invariant(
      from.stage < to.stage,
      "Zero-delay connections must move to a later stage",
      "INVALID_ZERO_DELAY_CONNECTION",
      { connection: connection.id, fromStage: from.stage, toStage: to.stage },
    );
  }
}

function validateGate(topology: TopologySpec, gate: GateSpec): void {
  const connection = topology.connections[gate.connection];
  const gater = topology.units[gate.gater];
  invariant(connection && gater, "Gate references an unknown graph element", "UNKNOWN_GATE_REFERENCE", {
    connection: gate.connection,
    gater: gate.gater,
  });
  const target = topology.units[connection.to];
  invariant(target, "Gated connection has an unknown target", "UNKNOWN_UNIT");
  if (gate.delay === 0) {
    invariant(
      gater.stage < target.stage,
      "Zero-delay gates must be evaluated before the gated target",
      "INVALID_ZERO_DELAY_GATE",
      { connection: gate.connection, gater: gate.gater },
    );
  }
}

export function validateTopology(topology: TopologySpec): void {
  invariant(topology.version === 1, "Unsupported topology version", "UNSUPPORTED_TOPOLOGY_VERSION");
  invariant(topology.units.length > 0, "A model needs at least one unit", "EMPTY_TOPOLOGY");
  invariant(topology.inputs.length > 0, "A model needs at least one input", "MISSING_INPUT");
  invariant(topology.outputs.length > 0, "A model needs at least one output", "MISSING_OUTPUT");

  validateContiguousIds(topology.units, "Unit");
  validateContiguousIds(topology.connections, "Connection");
  validateContiguousIds(topology.parameters, "Parameter");

  const stages = new Set(topology.units.map((unit) => unit.stage));
  const highestStage = Math.max(...stages);
  for (let stage = 0; stage <= highestStage; stage += 1) {
    invariant(stages.has(stage), "Stages must be contiguous", "INVALID_STAGE_SEQUENCE", { stage });
  }

  for (const unit of topology.units) {
    invariant(
      Number.isInteger(unit.stage) && unit.stage >= 0,
      "Unit stages must be non-negative integers",
      "INVALID_STAGE",
      { unit: unit.id, stage: unit.stage },
    );
    invariant(unit.activation in ACTIVATION_CODES, "Unknown activation", "UNKNOWN_ACTIVATION", {
      unit: unit.id,
      activation: unit.activation,
    });
  }

  const inputSet = new Set(topology.inputs);
  const outputSet = new Set(topology.outputs);
  for (const id of [...inputSet, ...outputSet]) {
    invariant(topology.units[id], "Input or output references an unknown unit", "UNKNOWN_PORT_UNIT", { id });
  }

  for (const connection of topology.connections) {
    invariant(
      topology.parameters[connection.parameter],
      "Connection references an unknown parameter",
      "UNKNOWN_PARAMETER",
      { connection: connection.id, parameter: connection.parameter },
    );
    validateDelay(topology, connection);
  }

  const gatedConnections = new Set<number>();
  for (const gate of topology.gates) {
    invariant(
      !gatedConnections.has(gate.connection),
      "A connection can have only one gater",
      "DUPLICATE_GATE",
      { connection: gate.connection },
    );
    gatedConnections.add(gate.connection);
    validateGate(topology, gate);
  }
}

export function validateDefinition(definition: ModelDefinition): void {
  invariant(definition.format === "synaptic-model", "Unknown model format", "UNKNOWN_MODEL_FORMAT");
  invariant(definition.formatVersion === 1, "Unsupported model format version", "UNSUPPORTED_MODEL_VERSION");
  invariant(definition.algorithm === "lstm-g", "Unsupported training algorithm", "UNSUPPORTED_ALGORITHM");
  invariant(definition.algorithmVersion === 1, "Unsupported algorithm version", "UNSUPPORTED_ALGORITHM_VERSION");
  invariant(
    definition.precision === "f32" || definition.precision === "f64",
    "Unsupported precision",
    "UNSUPPORTED_PRECISION",
  );
  validateTopology(definition.topology);
}
