import type {
  Metrics,
  ModelCheckpoint,
  ModelSnapshot,
  Tensor,
  TensorLike,
  TrainingBatch,
} from "./types.js";
import type { ExecutionPlan } from "./plan.js";
import { invariant } from "./errors.js";

export interface CompileOptions {
  readonly training?: boolean;
  readonly fallback?: readonly string[];
}

export interface SupportIssue {
  readonly code: string;
  readonly message: string;
}

export interface SupportReport {
  readonly supported: boolean;
  readonly issues: readonly SupportIssue[];
}

export interface TrainStepOptions {
  readonly learningRate?: number;
}

export interface TrainingSequenceStep extends TrainingBatch {
  readonly learningRate?: number;
}

export interface SequenceOptions {
  /**
   * Default learning rate for steps that do not provide their own value.
   */
  readonly learningRate?: number;
  /**
   * `none` lets an accelerated backend avoid reading predictions back merely
   * to calculate a loss that the caller does not use.
   */
  readonly metrics?: "last" | "none";
}

export interface SequenceMetrics {
  readonly steps: number;
  readonly step: number;
  readonly loss?: number;
}

export interface Session {
  readonly backend: string;
  forward(input: TensorLike): Promise<Tensor>;
  /**
   * Runs inputs in order while preserving recurrent state between them.
   */
  forwardSequence(inputs: readonly TensorLike[]): Promise<readonly Tensor[]>;
  trainStep(batch: TrainingBatch, options?: TrainStepOptions): Promise<Metrics>;
  /**
   * Trains ordered steps while preserving recurrent state and updating
   * parameters after each step.
   */
  trainSequence(
    sequence: readonly TrainingSequenceStep[],
    options?: SequenceOptions,
  ): Promise<SequenceMetrics>;
  resetState(): Promise<void>;
  restore(checkpoint: ModelCheckpoint): Promise<void>;
  snapshot(): Promise<ModelSnapshot>;
  checkpoint(): Promise<ModelCheckpoint>;
  dispose(): void;
}

export interface Backend {
  readonly id: string;
  inspect(plan: ExecutionPlan, options?: CompileOptions): SupportReport;
  compile(
    plan: ExecutionPlan,
    snapshot: ModelSnapshot,
    options?: CompileOptions,
  ): Promise<Session>;
}

/**
 * Portable implementation for backends that do not specialize sequence
 * execution. Backends may replace it with a single-submission implementation.
 */
export async function runForwardSequence(
  session: Pick<Session, "forward">,
  inputs: readonly TensorLike[],
): Promise<readonly Tensor[]> {
  const outputs: Tensor[] = [];
  for (const input of inputs) {
    outputs.push(await session.forward(input));
  }
  return outputs;
}

/**
 * Portable ordered-training implementation. This deliberately retains online
 * update semantics: every step sees the parameters produced by the prior one.
 */
export async function runTrainingSequence(
  session: Pick<Session, "trainStep">,
  sequence: readonly TrainingSequenceStep[],
  options: SequenceOptions = {},
): Promise<SequenceMetrics> {
  invariant(
    sequence.length > 0,
    "A training sequence must contain at least one step",
    "EMPTY_TRAINING_SEQUENCE",
  );
  let last: Metrics | undefined;
  for (const trainingStep of sequence) {
    const learningRate = trainingStep.learningRate ?? options.learningRate;
    last = await session.trainStep(
      trainingStep,
      learningRate === undefined ? undefined : { learningRate },
    );
  }
  invariant(last !== undefined, "Training sequence did not execute", "EMPTY_TRAINING_SEQUENCE");
  return {
    steps: sequence.length,
    step: last.step,
    ...(options.metrics === "none" ? {} : { loss: last.loss }),
  };
}
