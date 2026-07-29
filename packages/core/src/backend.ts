import type {
  Metrics,
  ModelCheckpoint,
  ModelSnapshot,
  Tensor,
  TensorLike,
  TrainingBatch,
} from "./types.js";
import type { ExecutionPlan } from "./plan.js";

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

export interface Session {
  readonly backend: string;
  forward(input: TensorLike): Promise<Tensor>;
  trainStep(batch: TrainingBatch, options?: TrainStepOptions): Promise<Metrics>;
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
