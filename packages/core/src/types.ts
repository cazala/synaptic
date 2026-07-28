export const ACTIVATION_CODES = {
  identity: 0,
  logistic: 1,
  tanh: 2,
  relu: 3,
  step: 4,
} as const;

export type ActivationId = keyof typeof ACTIVATION_CODES;
export type UnitId = number;
export type ConnectionId = number;
export type ParameterId = number;
export type StageId = number;
export type Delay = 0 | 1;
export type Precision = "f32" | "f64";

export type InitializerSpec =
  | { readonly kind: "constant"; readonly value: number }
  | { readonly kind: "uniform"; readonly min: number; readonly max: number };

export interface UnitSpec {
  readonly id: UnitId;
  readonly stage: StageId;
  readonly activation: ActivationId;
  readonly constant?: number;
  readonly label?: string;
}

export interface ConnectionSpec {
  readonly id: ConnectionId;
  readonly from: UnitId;
  readonly to: UnitId;
  readonly parameter: ParameterId;
  readonly delay: Delay;
}

export interface GateSpec {
  readonly connection: ConnectionId;
  readonly gater: UnitId;
  readonly delay: Delay;
}

export interface ParameterSpec {
  readonly id: ParameterId;
  readonly trainable: boolean;
  readonly initializer: InitializerSpec;
  readonly label?: string;
}

export interface TopologySpec {
  readonly version: 1;
  readonly units: readonly UnitSpec[];
  readonly connections: readonly ConnectionSpec[];
  readonly gates: readonly GateSpec[];
  readonly parameters: readonly ParameterSpec[];
  readonly inputs: readonly UnitId[];
  readonly outputs: readonly UnitId[];
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface ModelDefinition {
  readonly format: "synaptic-model";
  readonly formatVersion: 1;
  readonly algorithm: "lstm-g";
  readonly algorithmVersion: 1;
  readonly precision: Precision;
  readonly topology: TopologySpec;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface ModelSnapshot {
  readonly definition: ModelDefinition;
  readonly parameters: Float32Array | Float64Array;
}

export interface RuntimeState {
  readonly state: Float32Array | Float64Array;
  readonly activation: Float32Array | Float64Array;
  readonly previousActivation: Float32Array | Float64Array;
  readonly derivative: Float32Array | Float64Array;
  readonly eligibilityTrace: Float32Array | Float64Array;
  readonly extendedEligibilityTrace: Float32Array | Float64Array;
  readonly projectedError: Float32Array | Float64Array;
  readonly gatedError: Float32Array | Float64Array;
  readonly error: Float32Array | Float64Array;
  readonly step: number;
  readonly randomSeed: number;
  readonly randomCounter: number;
}

export interface ModelCheckpoint extends ModelSnapshot {
  readonly runtime: RuntimeState;
  readonly optimizer?: Readonly<Record<string, Float32Array | Float64Array>>;
}

export interface Tensor {
  readonly data: Float32Array;
  readonly shape: readonly number[];
}

export type TensorLike = readonly number[] | Float32Array;

export interface TrainingBatch {
  readonly input: TensorLike;
  readonly target: TensorLike;
}

export interface Metrics {
  readonly loss: number;
  readonly step: number;
}
