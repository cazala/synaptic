import {
  invariant,
  type ExecutionPlan,
  type ModelSnapshot,
} from "@synaptic/core";

export type HeapSectionName =
  | "stageUnits"
  | "unitActivation"
  | "unitConstant"
  | "unitConstantMask"
  | "inputSlot"
  | "connectionFrom"
  | "connectionTo"
  | "connectionParameter"
  | "parameterOffsets"
  | "parameterConnections"
  | "connectionDelay"
  | "connectionGater"
  | "gateDelay"
  | "incomingOffsets"
  | "incomingConnections"
  | "outgoingOffsets"
  | "outgoingConnections"
  | "gatedTargetOffsets"
  | "gatedTargets"
  | "extendedTraceTarget"
  | "extendedTraceOffsets"
  | "outputs"
  | "outputSlot"
  | "parameterTrainable"
  | "parameters"
  | "state"
  | "activation"
  | "previousActivation"
  | "derivative"
  | "eligibilityTrace"
  | "extendedEligibilityTrace"
  | "projectedError"
  | "gatedError"
  | "error"
  | "connectionGradient"
  | "input"
  | "target"
  | "learningRate"
  | "output";

export interface HeapSection {
  readonly offset: number;
  readonly wordOffset: number;
  readonly length: number;
  readonly byteLength: number;
}

export interface WebGpuHeapLayout {
  readonly byteLength: number;
  readonly sections: Readonly<Record<HeapSectionName, HeapSection>>;
}

export interface WebGpuHeap {
  readonly layout: WebGpuHeapLayout;
  readonly bytes: Uint8Array;
}

const WORD_SIZE = 4;

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

export function buildHeapLayout(plan: ExecutionPlan): WebGpuHeapLayout {
  const lengths: Readonly<Record<HeapSectionName, number>> = {
    stageUnits: plan.stageUnits.length,
    unitActivation: plan.unitActivation.length,
    unitConstant: plan.unitConstant.length,
    unitConstantMask: plan.unitCount,
    inputSlot: plan.unitCount,
    connectionFrom: plan.connectionFrom.length,
    connectionTo: plan.connectionTo.length,
    connectionParameter: plan.connectionParameter.length,
    parameterOffsets: plan.parameterCount + 1,
    parameterConnections: plan.connectionCount,
    connectionDelay: plan.connectionDelay.length,
    connectionGater: plan.connectionGater.length,
    gateDelay: plan.gateDelay.length,
    incomingOffsets: plan.incomingOffsets.length,
    incomingConnections: plan.incomingConnections.length,
    outgoingOffsets: plan.outgoingOffsets.length,
    outgoingConnections: plan.outgoingConnections.length,
    gatedTargetOffsets: plan.gatedTargetOffsets.length,
    gatedTargets: plan.gatedTargets.length,
    extendedTraceTarget: plan.extendedTraceTarget.length,
    extendedTraceOffsets: plan.extendedTraceOffsets.length,
    outputs: plan.outputs.length,
    outputSlot: plan.unitCount,
    parameterTrainable: plan.parameterCount,
    parameters: plan.parameterCount,
    state: plan.unitCount,
    activation: plan.unitCount,
    previousActivation: plan.unitCount,
    derivative: plan.unitCount,
    eligibilityTrace: plan.connectionCount,
    extendedEligibilityTrace: plan.extendedTraceTarget.length,
    projectedError: plan.unitCount,
    gatedError: plan.unitCount,
    error: plan.unitCount,
    connectionGradient: plan.connectionCount,
    input: plan.inputs.length,
    target: plan.outputs.length,
    learningRate: 1,
    output: plan.outputs.length,
  };
  const sections = {} as Record<HeapSectionName, HeapSection>;
  let cursor = 0;
  for (const name of Object.keys(lengths) as HeapSectionName[]) {
    cursor = align(cursor, WORD_SIZE);
    const length = lengths[name];
    sections[name] = {
      offset: cursor,
      wordOffset: cursor / WORD_SIZE,
      length,
      byteLength: length * WORD_SIZE,
    };
    cursor += length * WORD_SIZE;
  }
  return Object.freeze({
    byteLength: align(cursor, WORD_SIZE),
    sections: Object.freeze(sections),
  });
}

export function buildWebGpuHeap(
  plan: ExecutionPlan,
  snapshot: ModelSnapshot,
): WebGpuHeap {
  invariant(snapshot.parameters instanceof Float32Array, "WebGPU parameters must use f32 storage", "PRECISION_MISMATCH");
  const layout = buildHeapLayout(plan);
  const buffer = new ArrayBuffer(layout.byteLength);

  const writeU32 = (name: HeapSectionName, values: ArrayLike<number>): void => {
    const section = layout.sections[name];
    invariant(values.length === section.length, `Heap section ${name} has the wrong length`, "INVALID_HEAP_LAYOUT");
    new Uint32Array(buffer, section.offset, section.length).set(values);
  };
  const writeI32 = (name: HeapSectionName, values: ArrayLike<number>): void => {
    const section = layout.sections[name];
    invariant(values.length === section.length, `Heap section ${name} has the wrong length`, "INVALID_HEAP_LAYOUT");
    new Int32Array(buffer, section.offset, section.length).set(values);
  };
  const writeF32 = (name: HeapSectionName, values: ArrayLike<number>): void => {
    const section = layout.sections[name];
    invariant(values.length === section.length, `Heap section ${name} has the wrong length`, "INVALID_HEAP_LAYOUT");
    new Float32Array(buffer, section.offset, section.length).set(values);
  };

  const inputSlot = new Int32Array(plan.unitCount);
  const outputSlot = new Int32Array(plan.unitCount);
  const unitConstantMask = Uint32Array.from(
    plan.definition.topology.units,
    (unit) => Number(unit.constant !== undefined),
  );
  inputSlot.fill(-1);
  outputSlot.fill(-1);
  plan.inputs.forEach((unit, slot) => {
    inputSlot[unit] = slot;
  });
  plan.outputs.forEach((unit, slot) => {
    outputSlot[unit] = slot;
  });
  const parameterTrainable = Uint32Array.from(
    plan.definition.topology.parameters,
    (parameter) => Number(parameter.trainable),
  );
  const parameterOffsets = new Uint32Array(plan.parameterCount + 1);
  for (const parameter of plan.connectionParameter) {
    parameterOffsets[parameter + 1] = (parameterOffsets[parameter + 1] ?? 0) + 1;
  }
  for (let parameter = 0; parameter < plan.parameterCount; parameter += 1) {
    parameterOffsets[parameter + 1] =
      (parameterOffsets[parameter + 1] ?? 0) + (parameterOffsets[parameter] ?? 0);
  }
  const parameterCursor = parameterOffsets.slice(0, plan.parameterCount);
  const parameterConnections = new Uint32Array(plan.connectionCount);
  for (let connection = 0; connection < plan.connectionCount; connection += 1) {
    const parameter = plan.connectionParameter[connection];
    invariant(parameter !== undefined, "Connection has no parameter", "INVALID_PLAN");
    const cursor = parameterCursor[parameter];
    invariant(cursor !== undefined, "Parameter adjacency is invalid", "INVALID_PLAN");
    parameterConnections[cursor] = connection;
    parameterCursor[parameter] = cursor + 1;
  }

  writeU32("stageUnits", plan.stageUnits);
  writeU32("unitActivation", plan.unitActivation);
  writeF32("unitConstant", plan.unitConstant);
  writeU32("unitConstantMask", unitConstantMask);
  writeI32("inputSlot", inputSlot);
  writeU32("connectionFrom", plan.connectionFrom);
  writeU32("connectionTo", plan.connectionTo);
  writeU32("connectionParameter", plan.connectionParameter);
  writeU32("parameterOffsets", parameterOffsets);
  writeU32("parameterConnections", parameterConnections);
  writeU32("connectionDelay", plan.connectionDelay);
  writeI32("connectionGater", plan.connectionGater);
  writeU32("gateDelay", plan.gateDelay);
  writeU32("incomingOffsets", plan.incomingOffsets);
  writeU32("incomingConnections", plan.incomingConnections);
  writeU32("outgoingOffsets", plan.outgoingOffsets);
  writeU32("outgoingConnections", plan.outgoingConnections);
  writeU32("gatedTargetOffsets", plan.gatedTargetOffsets);
  writeU32("gatedTargets", plan.gatedTargets);
  writeU32("extendedTraceTarget", plan.extendedTraceTarget);
  writeU32("extendedTraceOffsets", plan.extendedTraceOffsets);
  writeU32("outputs", plan.outputs);
  writeI32("outputSlot", outputSlot);
  writeU32("parameterTrainable", parameterTrainable);
  writeF32("parameters", snapshot.parameters);

  return { layout, bytes: new Uint8Array(buffer) };
}

export const WEBGPU_UNIFORM_STRIDE = 256;
export const WEBGPU_UNIFORM_WORDS = 48;

export function buildUniformRecords(
  plan: ExecutionPlan,
  layout: WebGpuHeapLayout,
): Uint8Array {
  const stageCount = plan.stageOffsets.length - 1;
  const buffer = new ArrayBuffer(Math.max(1, stageCount) * WEBGPU_UNIFORM_STRIDE);
  for (let stage = 0; stage < stageCount; stage += 1) {
    const words = new Uint32Array(
      buffer,
      stage * WEBGPU_UNIFORM_STRIDE,
      WEBGPU_UNIFORM_WORDS,
    );
    words.set([
      plan.unitCount,
      plan.stageOffsets[stage] ?? 0,
      plan.stageOffsets[stage + 1] ?? 0,
      layout.sections.stageUnits.wordOffset,
      layout.sections.unitActivation.wordOffset,
      layout.sections.unitConstant.wordOffset,
      layout.sections.unitConstantMask.wordOffset,
      layout.sections.inputSlot.wordOffset,
      layout.sections.connectionFrom.wordOffset,
      layout.sections.connectionParameter.wordOffset,
      layout.sections.connectionDelay.wordOffset,
      layout.sections.connectionGater.wordOffset,
      layout.sections.gateDelay.wordOffset,
      layout.sections.incomingOffsets.wordOffset,
      layout.sections.incomingConnections.wordOffset,
      layout.sections.parameters.wordOffset,
      layout.sections.state.wordOffset,
      layout.sections.activation.wordOffset,
      layout.sections.previousActivation.wordOffset,
      layout.sections.input.wordOffset,
      layout.sections.outputs.wordOffset,
      layout.sections.output.wordOffset,
      plan.outputs.length,
      layout.sections.connectionTo.wordOffset,
      layout.sections.outgoingOffsets.wordOffset,
      layout.sections.outgoingConnections.wordOffset,
      layout.sections.gatedTargetOffsets.wordOffset,
      layout.sections.gatedTargets.wordOffset,
      layout.sections.extendedTraceTarget.wordOffset,
      layout.sections.extendedTraceOffsets.wordOffset,
      layout.sections.parameterTrainable.wordOffset,
      layout.sections.outputSlot.wordOffset,
      layout.sections.derivative.wordOffset,
      layout.sections.eligibilityTrace.wordOffset,
      layout.sections.extendedEligibilityTrace.wordOffset,
      layout.sections.projectedError.wordOffset,
      layout.sections.gatedError.wordOffset,
      layout.sections.error.wordOffset,
      layout.sections.connectionGradient.wordOffset,
      layout.sections.target.wordOffset,
      layout.sections.learningRate.wordOffset,
      plan.connectionCount,
      plan.parameterCount,
      layout.sections.parameterOffsets.wordOffset,
      layout.sections.parameterConnections.wordOffset,
      0,
      0,
      0,
    ]);
  }
  return new Uint8Array(buffer);
}
