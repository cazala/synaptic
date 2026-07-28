declare namespace __AdaptedExports {
  /** Exported memory */
  export const memory: WebAssembly.Memory;
  // Exported runtime interface
  export function __new(size: number, id: number): number;
  export function __pin(ptr: number): number;
  export function __unpin(ptr: number): void;
  export function __collect(): void;
  export const __rtti_base: number;
  /** packages/backend-wasm/assembly/runtime/FLOAT32ARRAY_ID */
  export const FLOAT32ARRAY_ID: {
    /** @type `u32` */
    get value(): number
  };
  /** packages/backend-wasm/assembly/runtime/UINT32ARRAY_ID */
  export const UINT32ARRAY_ID: {
    /** @type `u32` */
    get value(): number
  };
  /** packages/backend-wasm/assembly/runtime/INT32ARRAY_ID */
  export const INT32ARRAY_ID: {
    /** @type `u32` */
    get value(): number
  };
  /**
   * packages/backend-wasm/assembly/runtime/initialize
   * @param nextUnitCount `i32`
   * @param nextConnectionCount `i32`
   * @param nextParameterCount `i32`
   * @param nextStageOffsets `~lib/typedarray/Uint32Array`
   * @param nextStageUnits `~lib/typedarray/Uint32Array`
   * @param nextUnitActivation `~lib/typedarray/Uint32Array`
   * @param nextUnitConstant `~lib/typedarray/Float32Array`
   * @param nextConnectionFrom `~lib/typedarray/Uint32Array`
   * @param nextConnectionTo `~lib/typedarray/Uint32Array`
   * @param nextConnectionParameter `~lib/typedarray/Uint32Array`
   * @param nextConnectionDelay `~lib/typedarray/Uint32Array`
   * @param nextConnectionGater `~lib/typedarray/Int32Array`
   * @param nextGateDelay `~lib/typedarray/Uint32Array`
   * @param nextIncomingOffsets `~lib/typedarray/Uint32Array`
   * @param nextIncomingConnections `~lib/typedarray/Uint32Array`
   * @param nextOutgoingOffsets `~lib/typedarray/Uint32Array`
   * @param nextOutgoingConnections `~lib/typedarray/Uint32Array`
   * @param nextGatedTargetOffsets `~lib/typedarray/Uint32Array`
   * @param nextGatedTargets `~lib/typedarray/Uint32Array`
   * @param nextExtendedTraceTarget `~lib/typedarray/Uint32Array`
   * @param nextExtendedTraceOffsets `~lib/typedarray/Uint32Array`
   * @param nextInputs `~lib/typedarray/Uint32Array`
   * @param nextOutputs `~lib/typedarray/Uint32Array`
   * @param nextInputSlot `~lib/typedarray/Int32Array`
   * @param nextOutputSlot `~lib/typedarray/Int32Array`
   * @param nextParameterTrainable `~lib/typedarray/Uint32Array`
   * @param nextParameters `~lib/typedarray/Float32Array`
   */
  export function initialize(nextUnitCount: number, nextConnectionCount: number, nextParameterCount: number, nextStageOffsets: Uint32Array, nextStageUnits: Uint32Array, nextUnitActivation: Uint32Array, nextUnitConstant: Float32Array, nextConnectionFrom: Uint32Array, nextConnectionTo: Uint32Array, nextConnectionParameter: Uint32Array, nextConnectionDelay: Uint32Array, nextConnectionGater: Int32Array, nextGateDelay: Uint32Array, nextIncomingOffsets: Uint32Array, nextIncomingConnections: Uint32Array, nextOutgoingOffsets: Uint32Array, nextOutgoingConnections: Uint32Array, nextGatedTargetOffsets: Uint32Array, nextGatedTargets: Uint32Array, nextExtendedTraceTarget: Uint32Array, nextExtendedTraceOffsets: Uint32Array, nextInputs: Uint32Array, nextOutputs: Uint32Array, nextInputSlot: Int32Array, nextOutputSlot: Int32Array, nextParameterTrainable: Uint32Array, nextParameters: Float32Array): void;
  /**
   * packages/backend-wasm/assembly/runtime/forward
   * @param input `~lib/typedarray/Float32Array`
   * @returns `~lib/typedarray/Float32Array`
   */
  export function forward(input: Float32Array): Float32Array;
  /**
   * packages/backend-wasm/assembly/runtime/forwardBuffered
   */
  export function forwardBuffered(): void;
  /**
   * packages/backend-wasm/assembly/runtime/trainStep
   * @param input `~lib/typedarray/Float32Array`
   * @param target `~lib/typedarray/Float32Array`
   * @param learningRate `f32`
   * @returns `f32`
   */
  export function trainStep(input: Float32Array, target: Float32Array, learningRate: number): number;
  /**
   * packages/backend-wasm/assembly/runtime/trainStepBuffered
   * @param learningRate `f32`
   * @returns `f32`
   */
  export function trainStepBuffered(learningRate: number): number;
  /**
   * packages/backend-wasm/assembly/runtime/resetState
   */
  export function resetState(): void;
  /**
   * packages/backend-wasm/assembly/runtime/getParameters
   * @returns `~lib/typedarray/Float32Array`
   */
  export function getParameters(): Float32Array;
  /**
   * packages/backend-wasm/assembly/runtime/getState
   * @returns `~lib/typedarray/Float32Array`
   */
  export function getState(): Float32Array;
  /**
   * packages/backend-wasm/assembly/runtime/getActivation
   * @returns `~lib/typedarray/Float32Array`
   */
  export function getActivation(): Float32Array;
  /**
   * packages/backend-wasm/assembly/runtime/getPreviousActivation
   * @returns `~lib/typedarray/Float32Array`
   */
  export function getPreviousActivation(): Float32Array;
  /**
   * packages/backend-wasm/assembly/runtime/getDerivative
   * @returns `~lib/typedarray/Float32Array`
   */
  export function getDerivative(): Float32Array;
  /**
   * packages/backend-wasm/assembly/runtime/getEligibilityTrace
   * @returns `~lib/typedarray/Float32Array`
   */
  export function getEligibilityTrace(): Float32Array;
  /**
   * packages/backend-wasm/assembly/runtime/getExtendedEligibilityTrace
   * @returns `~lib/typedarray/Float32Array`
   */
  export function getExtendedEligibilityTrace(): Float32Array;
  /**
   * packages/backend-wasm/assembly/runtime/getProjectedError
   * @returns `~lib/typedarray/Float32Array`
   */
  export function getProjectedError(): Float32Array;
  /**
   * packages/backend-wasm/assembly/runtime/getGatedError
   * @returns `~lib/typedarray/Float32Array`
   */
  export function getGatedError(): Float32Array;
  /**
   * packages/backend-wasm/assembly/runtime/getError
   * @returns `~lib/typedarray/Float32Array`
   */
  export function getError(): Float32Array;
  /**
   * packages/backend-wasm/assembly/runtime/getStep
   * @returns `i32`
   */
  export function getStep(): number;
  /**
   * packages/backend-wasm/assembly/runtime/getInputDataStart
   * @returns `usize`
   */
  export function getInputDataStart(): number;
  /**
   * packages/backend-wasm/assembly/runtime/getTargetDataStart
   * @returns `usize`
   */
  export function getTargetDataStart(): number;
  /**
   * packages/backend-wasm/assembly/runtime/getOutputDataStart
   * @returns `usize`
   */
  export function getOutputDataStart(): number;
}
/** Instantiates the compiled WebAssembly module with the given imports. */
export declare function instantiate(module: WebAssembly.Module, imports: {
  env: unknown,
}): Promise<typeof __AdaptedExports>;
