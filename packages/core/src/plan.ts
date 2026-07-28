import { invariant } from "./errors.js";
import { ACTIVATION_CODES, type ModelDefinition } from "./types.js";
import { validateDefinition } from "./validate.js";

export interface ExecutionPlan {
  readonly definition: ModelDefinition;
  readonly unitCount: number;
  readonly connectionCount: number;
  readonly parameterCount: number;
  readonly stageOffsets: Uint32Array;
  readonly stageUnits: Uint32Array;
  readonly unitStage: Uint32Array;
  readonly unitActivation: Uint32Array;
  readonly unitConstant: Float32Array;
  readonly connectionFrom: Uint32Array;
  readonly connectionTo: Uint32Array;
  readonly connectionParameter: Uint32Array;
  readonly connectionDelay: Uint32Array;
  readonly connectionGater: Int32Array;
  readonly gateDelay: Uint32Array;
  readonly incomingOffsets: Uint32Array;
  readonly incomingConnections: Uint32Array;
  readonly outgoingOffsets: Uint32Array;
  readonly outgoingConnections: Uint32Array;
  readonly projectionOffsets: Uint32Array;
  readonly projectionTargets: Uint32Array;
  readonly gatedTargetOffsets: Uint32Array;
  readonly gatedTargets: Uint32Array;
  readonly extendedTraceConnection: Uint32Array;
  readonly extendedTraceTarget: Uint32Array;
  readonly extendedTraceOffsets: Uint32Array;
  readonly inputs: Uint32Array;
  readonly outputs: Uint32Array;
}

function buildAdjacency(
  unitCount: number,
  connections: readonly { readonly from: number; readonly to: number }[],
  endpoint: "from" | "to",
): readonly [Uint32Array, Uint32Array] {
  const buckets = Array.from({ length: unitCount }, () => [] as number[]);
  for (let id = 0; id < connections.length; id += 1) {
    const connection = connections[id];
    invariant(connection, "Missing connection while building adjacency", "INVALID_CONNECTION");
    buckets[connection[endpoint]]?.push(id);
  }
  const offsets = new Uint32Array(unitCount + 1);
  const values = new Uint32Array(connections.length);
  let cursor = 0;
  for (let unit = 0; unit < unitCount; unit += 1) {
    offsets[unit] = cursor;
    const bucket = buckets[unit] ?? [];
    values.set(bucket, cursor);
    cursor += bucket.length;
  }
  offsets[unitCount] = cursor;
  return [offsets, values];
}

function flattenSets(sets: readonly ReadonlySet<number>[]): readonly [Uint32Array, Uint32Array] {
  const offsets = new Uint32Array(sets.length + 1);
  const flattened: number[] = [];
  for (let index = 0; index < sets.length; index += 1) {
    offsets[index] = flattened.length;
    flattened.push(...[...(sets[index] ?? [])].sort((left, right) => left - right));
  }
  offsets[sets.length] = flattened.length;
  return [offsets, Uint32Array.from(flattened)];
}

export function compilePlan(definition: ModelDefinition): ExecutionPlan {
  validateDefinition(definition);
  const topology = definition.topology;
  const unitCount = topology.units.length;
  const connectionCount = topology.connections.length;
  const highestStage = Math.max(...topology.units.map((unit) => unit.stage));
  const stageBuckets = Array.from({ length: highestStage + 1 }, () => [] as number[]);

  const unitStage = new Uint32Array(unitCount);
  const unitActivation = new Uint32Array(unitCount);
  const unitConstant = new Float32Array(unitCount);
  unitConstant.fill(Number.NaN);
  for (const unit of topology.units) {
    unitStage[unit.id] = unit.stage;
    unitActivation[unit.id] = ACTIVATION_CODES[unit.activation];
    if (unit.constant !== undefined) {
      unitConstant[unit.id] = unit.constant;
    }
    stageBuckets[unit.stage]?.push(unit.id);
  }

  const stageOffsets = new Uint32Array(stageBuckets.length + 1);
  const stageUnits = new Uint32Array(unitCount);
  let stageCursor = 0;
  for (let stage = 0; stage < stageBuckets.length; stage += 1) {
    stageOffsets[stage] = stageCursor;
    const bucket = stageBuckets[stage] ?? [];
    stageUnits.set(bucket, stageCursor);
    stageCursor += bucket.length;
  }
  stageOffsets[stageBuckets.length] = stageCursor;

  const connectionFrom = new Uint32Array(connectionCount);
  const connectionTo = new Uint32Array(connectionCount);
  const connectionParameter = new Uint32Array(connectionCount);
  const connectionDelay = new Uint32Array(connectionCount);
  const connectionGater = new Int32Array(connectionCount);
  const gateDelay = new Uint32Array(connectionCount);
  connectionGater.fill(-1);

  for (const connection of topology.connections) {
    connectionFrom[connection.id] = connection.from;
    connectionTo[connection.id] = connection.to;
    connectionParameter[connection.id] = connection.parameter;
    connectionDelay[connection.id] = connection.delay;
  }
  for (const gate of topology.gates) {
    connectionGater[gate.connection] = gate.gater;
    gateDelay[gate.connection] = gate.delay;
  }

  const [incomingOffsets, incomingConnections] = buildAdjacency(unitCount, topology.connections, "to");
  const [outgoingOffsets, outgoingConnections] = buildAdjacency(unitCount, topology.connections, "from");

  const projectionSets = Array.from({ length: unitCount }, () => new Set<number>());
  const gatedTargetSets = Array.from({ length: unitCount }, () => new Set<number>());
  for (const connection of topology.connections) {
    if (connection.delay === 0) {
      projectionSets[connection.from]?.add(connection.to);
    }
  }
  for (const gate of topology.gates) {
    if (gate.delay === 0) {
      const connection = topology.connections[gate.connection];
      if (connection) {
        gatedTargetSets[gate.gater]?.add(connection.to);
      }
    }
  }
  const [projectionOffsets, projectionTargets] = flattenSets(projectionSets);
  const [gatedTargetOffsets, gatedTargets] = flattenSets(gatedTargetSets);

  const extendedTraceConnection: number[] = [];
  const extendedTraceTarget: number[] = [];
  const extendedTraceOffsets = new Uint32Array(connectionCount + 1);
  for (const connection of topology.connections) {
    extendedTraceOffsets[connection.id] = extendedTraceConnection.length;
    if (connection.from === connection.to) {
      continue;
    }
    for (const target of gatedTargetSets[connection.to] ?? []) {
      extendedTraceConnection.push(connection.id);
      extendedTraceTarget.push(target);
    }
  }
  extendedTraceOffsets[connectionCount] = extendedTraceConnection.length;

  return Object.freeze({
    definition,
    unitCount,
    connectionCount,
    parameterCount: topology.parameters.length,
    stageOffsets,
    stageUnits,
    unitStage,
    unitActivation,
    unitConstant,
    connectionFrom,
    connectionTo,
    connectionParameter,
    connectionDelay,
    connectionGater,
    gateDelay,
    incomingOffsets,
    incomingConnections,
    outgoingOffsets,
    outgoingConnections,
    projectionOffsets,
    projectionTargets,
    gatedTargetOffsets,
    gatedTargets,
    extendedTraceConnection: Uint32Array.from(extendedTraceConnection),
    extendedTraceTarget: Uint32Array.from(extendedTraceTarget),
    extendedTraceOffsets,
    inputs: Uint32Array.from(topology.inputs),
    outputs: Uint32Array.from(topology.outputs),
  });
}
