import { PaperBackend } from "@synaptic/backend-paper";
import { CpuBackend } from "@synaptic/backend-cpu";
import {
  DEFAULT_WASM_SELECTION_POLICY,
  WasmBackend,
  prefersWasm,
} from "@synaptic/backend-wasm";
import { WebGpuBackend } from "@synaptic/backend-webgpu";
import {
  compilePlan,
  createSnapshot,
  type Backend,
  type CompileOptions,
  type ExecutionPlan,
  type ModelDefinition,
  type ModelSnapshot,
  type Session,
} from "@synaptic/core";

export * from "@synaptic/core";
export * from "@synaptic/layers";
export * from "@synaptic/compat-v1";
export { PaperBackend } from "@synaptic/backend-paper";
export { CpuBackend } from "@synaptic/backend-cpu";
export {
  DEFAULT_WASM_SELECTION_POLICY,
  WasmBackend,
  prefersWasm,
} from "@synaptic/backend-wasm";
export {
  GROWING_NEURAL_CA_FORMAT,
  GROWING_NEURAL_CA_VERSION,
  GrowingNeuralCaCurriculum,
  GrowingNeuralCaTrainer,
  WebGpuBackend,
  buildGrowingNeuralCaHeapLayout,
  createGrowingNeuralCaWeights,
} from "@synaptic/backend-webgpu";
export type {
  GrowingNeuralCaArtifact,
  GrowingNeuralCaBatchQuality,
  GrowingNeuralCaCurriculumPhase,
  GrowingNeuralCaCurriculumPhaseKey,
  GrowingNeuralCaCurriculumSnapshot,
  GrowingNeuralCaCurriculumTrainingOptions,
  GrowingNeuralCaHeapLayout,
  GrowingNeuralCaHeapSection,
  GrowingNeuralCaHeapSectionName,
  GrowingNeuralCaMetrics,
  GrowingNeuralCaQualityMetrics,
  GrowingNeuralCaTrainerOptions,
  GrowingNeuralCaTrainOptions,
} from "@synaptic/backend-webgpu";

export type BackendId = "auto" | "paper" | "cpu" | "wasm" | "webgpu";

export interface CompileModelOptions extends CompileOptions {
  readonly backend?: BackendId | Backend;
  readonly seed?: number;
  readonly expectedSteps?: number;
}

export interface AutoSelection {
  readonly backend: "cpu" | "wasm" | "webgpu";
  readonly reason: string;
}

export function selectBackend(
  plan: ExecutionPlan,
  options: CompileModelOptions = {},
): AutoSelection {
  const expectedSteps = options.expectedSteps ?? 1;
  const webGpu = new WebGpuBackend();
  const webGpuWork = plan.connectionCount * expectedSteps;
  const webGpuSupported = webGpu.inspect(plan, options).supported;
  if (
    webGpuSupported
    && plan.unitCount >= 64
    && webGpuWork >= 32_768
  ) {
    return {
      backend: "webgpu",
      reason: "The model has enough repeated work to amortize GPU setup and readback",
    };
  }
  if (prefersWasm(plan, expectedSteps, DEFAULT_WASM_SELECTION_POLICY)) {
    return {
      backend: "wasm",
      reason: "The configured Wasm policy selects this model size",
    };
  }
  return {
    backend: "cpu",
    reason: webGpuSupported
      ? "The CPU avoids accelerator setup for this workload"
      : "The CPU is the portable available production backend",
  };
}

function backendById(id: Exclude<BackendId, "auto">): Backend {
  switch (id) {
    case "paper":
      return new PaperBackend();
    case "cpu":
      return new CpuBackend();
    case "wasm":
      return new WasmBackend();
    case "webgpu":
      return new WebGpuBackend();
  }
}

export async function compileModel(
  model: ModelDefinition | ModelSnapshot,
  options: CompileModelOptions = {},
): Promise<Session> {
  const snapshot = "parameters" in model
    ? model
    : createSnapshot(model, options.seed);
  const plan = compilePlan(snapshot.definition);
  const requested = options.backend ?? "auto";
  const backend = typeof requested === "string"
    ? backendById(
      requested === "auto"
        ? selectBackend(plan, options).backend
        : requested,
    )
    : requested;
  return backend.compile(plan, snapshot, options);
}

export class Model {
  readonly definition: ModelDefinition;

  constructor(definition: ModelDefinition) {
    this.definition = definition;
  }

  snapshot(seed?: number): ModelSnapshot {
    return createSnapshot(this.definition, seed);
  }

  compile(options?: CompileModelOptions): Promise<Session> {
    return compileModel(this.definition, options);
  }
}
