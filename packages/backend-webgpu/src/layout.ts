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
  | "connectionParameter"
  | "connectionDelay"
  | "connectionGater"
  | "gateDelay"
  | "incomingOffsets"
  | "incomingConnections"
  | "outputs"
  | "parameters"
  | "state"
  | "activation"
  | "previousActivation"
  | "input"
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
    connectionParameter: plan.connectionParameter.length,
    connectionDelay: plan.connectionDelay.length,
    connectionGater: plan.connectionGater.length,
    gateDelay: plan.gateDelay.length,
    incomingOffsets: plan.incomingOffsets.length,
    incomingConnections: plan.incomingConnections.length,
    outputs: plan.outputs.length,
    parameters: plan.parameterCount,
    state: plan.unitCount,
    activation: plan.unitCount,
    previousActivation: plan.unitCount,
    input: plan.inputs.length,
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
  const unitConstantMask = Uint32Array.from(
    plan.definition.topology.units,
    (unit) => Number(unit.constant !== undefined),
  );
  inputSlot.fill(-1);
  plan.inputs.forEach((unit, slot) => {
    inputSlot[unit] = slot;
  });

  writeU32("stageUnits", plan.stageUnits);
  writeU32("unitActivation", plan.unitActivation);
  writeF32("unitConstant", plan.unitConstant);
  writeU32("unitConstantMask", unitConstantMask);
  writeI32("inputSlot", inputSlot);
  writeU32("connectionFrom", plan.connectionFrom);
  writeU32("connectionParameter", plan.connectionParameter);
  writeU32("connectionDelay", plan.connectionDelay);
  writeI32("connectionGater", plan.connectionGater);
  writeU32("gateDelay", plan.gateDelay);
  writeU32("incomingOffsets", plan.incomingOffsets);
  writeU32("incomingConnections", plan.incomingConnections);
  writeU32("outputs", plan.outputs);
  writeF32("parameters", snapshot.parameters);

  return { layout, bytes: new Uint8Array(buffer) };
}

export const WEBGPU_UNIFORM_STRIDE = 256;
export const WEBGPU_UNIFORM_WORDS = 28;

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
      0,
      0,
      0,
      0,
      0,
    ]);
  }
  return new Uint8Array(buffer);
}
