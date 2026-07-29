import { invariant } from "./errors.js";
import type { ExecutionPlan } from "./plan.js";
import type {
  ActivationId,
  ModelCheckpoint,
  ModelSnapshot,
  Precision,
  RuntimeState,
  TensorLike,
} from "./types.js";

export type NumericArray = Float32Array | Float64Array;

export interface MutableRuntimeState {
  state: NumericArray;
  activation: NumericArray;
  previousActivation: NumericArray;
  derivative: NumericArray;
  eligibilityTrace: NumericArray;
  extendedEligibilityTrace: NumericArray;
  projectedError: NumericArray;
  gatedError: NumericArray;
  error: NumericArray;
  step: number;
  randomSeed: number;
  randomCounter: number;
}

export function createNumericArray(precision: Precision, length: number): NumericArray {
  return precision === "f64" ? new Float64Array(length) : new Float32Array(length);
}

export function createRuntimeState(
  plan: ExecutionPlan,
  precision: Precision,
  randomSeed = 0,
): MutableRuntimeState {
  return {
    state: createNumericArray(precision, plan.unitCount),
    activation: createNumericArray(precision, plan.unitCount),
    previousActivation: createNumericArray(precision, plan.unitCount),
    derivative: createNumericArray(precision, plan.unitCount),
    eligibilityTrace: createNumericArray(precision, plan.connectionCount),
    extendedEligibilityTrace: createNumericArray(precision, plan.extendedTraceTarget.length),
    projectedError: createNumericArray(precision, plan.unitCount),
    gatedError: createNumericArray(precision, plan.unitCount),
    error: createNumericArray(precision, plan.unitCount),
    step: 0,
    randomSeed: randomSeed >>> 0,
    randomCounter: 0,
  };
}

export function copyRuntimeState(runtime: MutableRuntimeState): RuntimeState {
  return {
    state: runtime.state.slice(),
    activation: runtime.activation.slice(),
    previousActivation: runtime.previousActivation.slice(),
    derivative: runtime.derivative.slice(),
    eligibilityTrace: runtime.eligibilityTrace.slice(),
    extendedEligibilityTrace: runtime.extendedEligibilityTrace.slice(),
    projectedError: runtime.projectedError.slice(),
    gatedError: runtime.gatedError.slice(),
    error: runtime.error.slice(),
    step: runtime.step,
    randomSeed: runtime.randomSeed,
    randomCounter: runtime.randomCounter,
  };
}

export function createCheckpoint(
  snapshot: ModelSnapshot,
  runtime: MutableRuntimeState,
  optimizer?: ModelCheckpoint["optimizer"],
): ModelCheckpoint {
  return {
    definition: snapshot.definition,
    parameters: snapshot.parameters.slice(),
    runtime: copyRuntimeState(runtime),
    ...(optimizer === undefined
      ? {}
      : { optimizer: copyOptimizerState(optimizer) }),
  };
}

export function copyOptimizerState(
  optimizer: NonNullable<ModelCheckpoint["optimizer"]>,
): Readonly<Record<string, NumericArray>> {
  return Object.fromEntries(
    Object.entries(optimizer).map(([name, values]) => [name, values.slice()]),
  );
}

export function validateSnapshot(plan: ExecutionPlan, snapshot: ModelSnapshot): void {
  invariant(
    snapshot.definition === plan.definition
      || JSON.stringify(snapshot.definition) === JSON.stringify(plan.definition),
    "Snapshot definition does not match the execution plan",
    "SNAPSHOT_DEFINITION_MISMATCH",
  );
  invariant(
    snapshot.parameters.length === plan.parameterCount,
    "Snapshot parameter count does not match the execution plan",
    "PARAMETER_COUNT_MISMATCH",
  );
  invariant(
    snapshot.definition.precision === "f32"
      ? snapshot.parameters instanceof Float32Array
      : snapshot.parameters instanceof Float64Array,
    "Snapshot parameter storage does not match its declared precision",
    "PRECISION_MISMATCH",
  );
}

export function validateCheckpoint(
  plan: ExecutionPlan,
  checkpoint: ModelCheckpoint,
): void {
  validateSnapshot(plan, checkpoint);
  const expectedLengths: Readonly<Record<keyof Omit<RuntimeState, "step" | "randomSeed" | "randomCounter">, number>> = {
    state: plan.unitCount,
    activation: plan.unitCount,
    previousActivation: plan.unitCount,
    derivative: plan.unitCount,
    eligibilityTrace: plan.connectionCount,
    extendedEligibilityTrace: plan.extendedTraceTarget.length,
    projectedError: plan.unitCount,
    gatedError: plan.unitCount,
    error: plan.unitCount,
  };
  for (const [name, expected] of Object.entries(expectedLengths)) {
    const values = checkpoint.runtime[
      name as keyof typeof expectedLengths
    ];
    invariant(
      values.length === expected,
      `Checkpoint runtime field ${name} has the wrong length`,
      "CHECKPOINT_SIZE_MISMATCH",
      { actual: values.length, expected, name },
    );
    invariant(
      checkpoint.definition.precision === "f32"
        ? values instanceof Float32Array
        : values instanceof Float64Array,
      `Checkpoint runtime field ${name} has the wrong precision`,
      "PRECISION_MISMATCH",
      { name },
    );
    for (let index = 0; index < values.length; index += 1) {
      invariant(
        Number.isFinite(values[index]),
        `Checkpoint runtime field ${name} contains a non-finite value`,
        "NON_FINITE_CHECKPOINT",
        { index, name },
      );
    }
  }
  invariant(
    Number.isSafeInteger(checkpoint.runtime.step) && checkpoint.runtime.step >= 0,
    "Checkpoint step must be a non-negative safe integer",
    "INVALID_CHECKPOINT_COUNTER",
  );
  invariant(
    Number.isSafeInteger(checkpoint.runtime.randomSeed)
      && checkpoint.runtime.randomSeed >= 0
      && checkpoint.runtime.randomSeed <= 0xffff_ffff,
    "Checkpoint random seed must be an unsigned 32-bit integer",
    "INVALID_CHECKPOINT_COUNTER",
  );
  invariant(
    Number.isSafeInteger(checkpoint.runtime.randomCounter)
      && checkpoint.runtime.randomCounter >= 0,
    "Checkpoint random counter must be a non-negative safe integer",
    "INVALID_CHECKPOINT_COUNTER",
  );
  for (const [name, values] of Object.entries(checkpoint.optimizer ?? {})) {
    invariant(name.length > 0, "Optimizer buffer names cannot be empty", "INVALID_BUFFER_NAME");
    invariant(
      checkpoint.definition.precision === "f32"
        ? values instanceof Float32Array
        : values instanceof Float64Array,
      `Optimizer buffer ${name} has the wrong precision`,
      "PRECISION_MISMATCH",
    );
    for (let index = 0; index < values.length; index += 1) {
      invariant(
        Number.isFinite(values[index]),
        `Optimizer buffer ${name} contains a non-finite value`,
        "NON_FINITE_CHECKPOINT",
        { index, name },
      );
    }
  }
}

export function restoreRuntimeState(
  target: MutableRuntimeState,
  source: RuntimeState,
): void {
  target.state.set(source.state);
  target.activation.set(source.activation);
  target.previousActivation.set(source.previousActivation);
  target.derivative.set(source.derivative);
  target.eligibilityTrace.set(source.eligibilityTrace);
  target.extendedEligibilityTrace.set(source.extendedEligibilityTrace);
  target.projectedError.set(source.projectedError);
  target.gatedError.set(source.gatedError);
  target.error.set(source.error);
  target.step = source.step;
  target.randomSeed = source.randomSeed >>> 0;
  target.randomCounter = source.randomCounter;
}

export function readTensor(values: TensorLike, expectedLength: number, name: string): Float32Array {
  invariant(values.length === expectedLength, `${name} length does not match the model port`, "TENSOR_SHAPE_MISMATCH", {
    actual: values.length,
    expected: expectedLength,
    name,
  });
  const result = values instanceof Float32Array ? values : Float32Array.from(values);
  for (let index = 0; index < result.length; index += 1) {
    invariant(Number.isFinite(result[index]), `${name} contains a non-finite value`, "NON_FINITE_TENSOR", {
      index,
      name,
    });
  }
  return result;
}

export function activate(id: ActivationId, value: number): number {
  switch (id) {
    case "identity":
      return value;
    case "logistic":
      return 1 / (1 + Math.exp(-value));
    case "tanh":
      return Math.tanh(value);
    case "relu":
      return Math.max(0, value);
    case "step":
      return value > 0 ? 1 : 0;
  }
}

export function derivative(id: ActivationId, state: number, activation: number): number {
  switch (id) {
    case "identity":
      return 1;
    case "logistic":
      return activation * (1 - activation);
    case "tanh":
      return 1 - activation * activation;
    case "relu":
      return state > 0 ? 1 : 0;
    case "step":
      return 0;
  }
}
