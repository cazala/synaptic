import {
  SynapticError,
  compilePlan,
  invariant,
  readTensor,
  validateSnapshot,
  type Backend,
  type CompileOptions,
  type ExecutionPlan,
  type Metrics,
  type ModelCheckpoint,
  type ModelSnapshot,
  type RuntimeState,
  type Session,
  type SupportReport,
  type Tensor,
  type TensorLike,
  type TrainingBatch,
  type TrainStepOptions,
} from "@synaptic/core";
import { instantiate as instantiateRuntime } from "../wasm/runtime.js";

const DEFAULT_LEARNING_RATE = 0.1;
type RuntimeExports = Awaited<ReturnType<typeof instantiateRuntime>>;
export type WasmVariant = "scalar" | "simd";

export interface WasmSelectionPolicy {
  readonly enabled: boolean;
  readonly minimumUnits: number;
  readonly minimumConnectionSteps: number;
}

export const DEFAULT_WASM_SELECTION_POLICY: WasmSelectionPolicy = Object.freeze({
  // Generic sparse LSTM-g plans currently benchmark faster on the JITed CPU
  // interpreter. Explicit Wasm remains available while specialization evolves.
  enabled: false,
  minimumUnits: 32,
  minimumConnectionSteps: 8_192,
});

export function prefersWasm(
  plan: ExecutionPlan,
  expectedSteps = 1,
  policy: WasmSelectionPolicy = DEFAULT_WASM_SELECTION_POLICY,
): boolean {
  return policy.enabled
    && plan.unitCount >= policy.minimumUnits
    && plan.connectionCount * expectedSteps >= policy.minimumConnectionSteps;
}

const moduleCache = new Map<WasmVariant, Promise<WebAssembly.Module>>();

async function readModule(variant: WasmVariant): Promise<WebAssembly.Module> {
  const cached = moduleCache.get(variant);
  if (cached) {
    return cached;
  }
  const loading = (async () => {
    invariant(typeof WebAssembly !== "undefined", "WebAssembly is not available", "WASM_UNAVAILABLE");
    const filename = variant === "simd" ? "runtime-simd.wasm" : "runtime.wasm";
    const url = new URL(`../wasm/${filename}`, import.meta.url);
    if (url.protocol === "file:") {
      const nodeModule = "node:fs/promises";
      const fileSystem = await import(nodeModule);
      return WebAssembly.compile(await fileSystem.readFile(url));
    }
    const response = await fetch(url);
    invariant(response.ok, `Failed to load ${filename}`, "WASM_LOAD_FAILED", {
      status: response.status,
      url: url.href,
    });
    return WebAssembly.compileStreaming(response);
  })();
  moduleCache.set(variant, loading);
  try {
    return await loading;
  } catch (error) {
    moduleCache.delete(variant);
    throw error;
  }
}

async function instantiateBestRuntime(): Promise<{
  readonly exports: RuntimeExports;
  readonly variant: WasmVariant;
}> {
  try {
    const module = await readModule("simd");
    return { exports: await instantiateRuntime(module, { env: {} }), variant: "simd" };
  } catch (simdError) {
    try {
      const module = await readModule("scalar");
      return { exports: await instantiateRuntime(module, { env: {} }), variant: "scalar" };
    } catch (scalarError) {
      throw new SynapticError("Unable to instantiate a WebAssembly runtime", "WASM_INSTANTIATION_FAILED", {
        scalarError,
        simdError,
      });
    }
  }
}

export class WasmBackend implements Backend {
  readonly id = "wasm";

  inspect(plan: ExecutionPlan, options: CompileOptions = {}): SupportReport {
    const issues = [];
    if (typeof WebAssembly === "undefined") {
      issues.push({ code: "WASM_UNAVAILABLE", message: "WebAssembly is not available in this runtime" });
    }
    if (plan.definition.precision !== "f32") {
      issues.push({
        code: "UNSUPPORTED_PRECISION",
        message: "The WebAssembly backend uses portable f32 precision",
      });
    }
    if (options.training) {
      for (let connection = 0; connection < plan.connectionCount; connection += 1) {
        if (
          plan.connectionFrom[connection] === plan.connectionTo[connection]
          && plan.definition.topology.parameters[plan.connectionParameter[connection] ?? -1]?.trainable
        ) {
          issues.push({
            code: "TRAINABLE_SELF_CONNECTION",
            message: "LSTM-g treats recurrent self-connections as fixed memory coefficients",
          });
          break;
        }
      }
    }
    return { supported: issues.length === 0, issues };
  }

  async compile(
    plan: ExecutionPlan,
    snapshot: ModelSnapshot,
    options: CompileOptions = {},
  ): Promise<Session> {
    validateSnapshot(plan, snapshot);
    const report = this.inspect(plan, options);
    if (!report.supported) {
      throw new SynapticError("The WebAssembly backend cannot compile this plan", "UNSUPPORTED_PLAN", {
        issues: report.issues,
      });
    }
    invariant(snapshot.parameters instanceof Float32Array, "Wasm parameters must use f32 storage", "PRECISION_MISMATCH");
    const runtime = await instantiateBestRuntime();
    return new WasmSession(
      plan,
      { definition: snapshot.definition, parameters: snapshot.parameters },
      runtime.exports,
      runtime.variant,
    );
  }
}

export class WasmSession implements Session {
  readonly backend = "wasm";
  readonly variant: WasmVariant;
  readonly #plan: ExecutionPlan;
  readonly #definition: ModelSnapshot["definition"];
  readonly #inputView: Float32Array;
  readonly #targetView: Float32Array;
  readonly #outputView: Float32Array;
  #runtime: RuntimeExports | undefined;

  constructor(
    plan: ExecutionPlan,
    snapshot: ModelSnapshot & { readonly parameters: Float32Array },
    runtime: RuntimeExports,
    variant: WasmVariant,
  ) {
    this.#plan = plan;
    this.#definition = snapshot.definition;
    this.#runtime = runtime;
    this.variant = variant;

    const inputSlot = new Int32Array(plan.unitCount);
    const outputSlot = new Int32Array(plan.unitCount);
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
    runtime.initialize(
      plan.unitCount,
      plan.connectionCount,
      plan.parameterCount,
      plan.stageOffsets,
      plan.stageUnits,
      plan.unitActivation,
      plan.unitConstant,
      plan.connectionFrom,
      plan.connectionTo,
      plan.connectionParameter,
      plan.connectionDelay,
      plan.connectionGater,
      plan.gateDelay,
      plan.incomingOffsets,
      plan.incomingConnections,
      plan.outgoingOffsets,
      plan.outgoingConnections,
      plan.gatedTargetOffsets,
      plan.gatedTargets,
      plan.extendedTraceTarget,
      plan.extendedTraceOffsets,
      plan.inputs,
      plan.outputs,
      inputSlot,
      outputSlot,
      parameterTrainable,
      snapshot.parameters,
    );
    this.#inputView = new Float32Array(
      runtime.memory.buffer,
      runtime.getInputDataStart(),
      plan.inputs.length,
    );
    this.#targetView = new Float32Array(
      runtime.memory.buffer,
      runtime.getTargetDataStart(),
      plan.outputs.length,
    );
    this.#outputView = new Float32Array(
      runtime.memory.buffer,
      runtime.getOutputDataStart(),
      plan.outputs.length,
    );
  }

  async forward(input: TensorLike): Promise<Tensor> {
    const runtime = this.#assertActive();
    const values = readTensor(input, this.#plan.inputs.length, "Input");
    this.#inputView.set(values);
    runtime.forwardBuffered();
    const output = this.#outputView.slice();
    return { data: output, shape: [output.length] };
  }

  async trainStep(
    batch: TrainingBatch,
    options: TrainStepOptions = {},
  ): Promise<Metrics> {
    const runtime = this.#assertActive();
    const input = readTensor(batch.input, this.#plan.inputs.length, "Input");
    const target = readTensor(batch.target, this.#plan.outputs.length, "Target");
    const learningRate = options.learningRate ?? DEFAULT_LEARNING_RATE;
    invariant(
      Number.isFinite(learningRate) && learningRate >= 0,
      "Learning rate must be a finite non-negative number",
      "INVALID_LEARNING_RATE",
    );
    this.#inputView.set(input);
    this.#targetView.set(target);
    return {
      loss: runtime.trainStepBuffered(learningRate),
      step: runtime.getStep(),
    };
  }

  async resetState(): Promise<void> {
    this.#assertActive().resetState();
  }

  async snapshot(): Promise<ModelSnapshot> {
    const runtime = this.#assertActive();
    return {
      definition: this.#definition,
      parameters: runtime.getParameters(),
    };
  }

  async checkpoint(): Promise<ModelCheckpoint> {
    const runtime = this.#assertActive();
    return {
      definition: this.#definition,
      parameters: runtime.getParameters(),
      runtime: this.#runtimeState(runtime),
    };
  }

  dispose(): void {
    this.#runtime = undefined;
  }

  #runtimeState(runtime: RuntimeExports): RuntimeState {
    return {
      state: runtime.getState(),
      activation: runtime.getActivation(),
      previousActivation: runtime.getPreviousActivation(),
      derivative: runtime.getDerivative(),
      eligibilityTrace: runtime.getEligibilityTrace(),
      extendedEligibilityTrace: runtime.getExtendedEligibilityTrace(),
      projectedError: runtime.getProjectedError(),
      gatedError: runtime.getGatedError(),
      error: runtime.getError(),
      step: runtime.getStep(),
      randomSeed: 0,
      randomCounter: 0,
    };
  }

  #assertActive(): RuntimeExports {
    invariant(this.#runtime !== undefined, "The session has been disposed", "SESSION_DISPOSED");
    return this.#runtime;
  }
}

export function compileWasm(
  definition: ModelSnapshot["definition"],
  snapshot: ModelSnapshot,
  options?: CompileOptions,
): Promise<Session> {
  return new WasmBackend().compile(compilePlan(definition), snapshot, options);
}
