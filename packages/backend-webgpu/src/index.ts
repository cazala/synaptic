import { CpuBackend } from "@synaptic/backend-cpu";
import { WasmBackend } from "@synaptic/backend-wasm";
import {
  SynapticError,
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
  type SupportIssue,
  type SupportReport,
  type Tensor,
  type TensorLike,
  type TrainingBatch,
  type TrainStepOptions,
} from "@synaptic/core";
import {
  WEBGPU_UNIFORM_STRIDE,
  WEBGPU_UNIFORM_WORDS,
  buildUniformRecords,
  buildWebGpuHeap,
  type WebGpuHeap,
} from "./layout.js";
import { FORWARD_WGSL } from "./forward-shader.js";

export * from "./layout.js";
export * from "./forward-shader.js";

const WORKGROUP_SIZE = 64;
const BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;
const MAP_READ = 0x0001;
const COMPUTE_STAGE = 0x0004;

export interface WebGpuBackendOptions {
  readonly gpu?: GPU | null;
  readonly powerPreference?: GPUPowerPreference;
  readonly fallbackBackends?: readonly Backend[];
}

function installedGpu(): GPU | undefined {
  return typeof navigator !== "undefined" && "gpu" in navigator
    ? navigator.gpu
    : undefined;
}

export class WebGpuBackend implements Backend {
  readonly id = "webgpu";
  readonly #gpu: GPU | undefined;
  readonly #powerPreference: GPUPowerPreference | undefined;
  readonly #fallbackBackends: readonly Backend[];

  constructor(options: WebGpuBackendOptions = {}) {
    this.#gpu = options.gpu === undefined ? installedGpu() : (options.gpu ?? undefined);
    this.#powerPreference = options.powerPreference;
    this.#fallbackBackends = options.fallbackBackends ?? [
      new WasmBackend(),
      new CpuBackend(),
    ];
  }

  inspect(plan: ExecutionPlan, options: CompileOptions = {}): SupportReport {
    const issues: SupportIssue[] = [];
    if (this.#gpu === undefined) {
      issues.push({
        code: "WEBGPU_UNAVAILABLE",
        message: "WebGPU is not available in this runtime",
      });
    }
    if (plan.definition.precision !== "f32") {
      issues.push({
        code: "UNSUPPORTED_PRECISION",
        message: "WebGPU supports the portable f32 model precision",
      });
    }
    if (options.training) {
      issues.push({
        code: "WEBGPU_TRAINING_UNAVAILABLE",
        message: "This WebGPU milestone implements forward execution only",
      });
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
    if (report.supported && this.#gpu !== undefined) {
      try {
        const adapter = await this.#gpu.requestAdapter({
          ...(this.#powerPreference === undefined
            ? {}
            : { powerPreference: this.#powerPreference }),
        });
        invariant(adapter !== null, "No suitable WebGPU adapter is available", "WEBGPU_ADAPTER_UNAVAILABLE");
        const device = await adapter.requestDevice();
        return await WebGpuSession.create(plan, snapshot, device);
      } catch (error) {
        return this.#compileFallback(plan, snapshot, options, [
          ...report.issues,
          {
            code: "WEBGPU_COMPILE_FAILED",
            message: error instanceof Error ? error.message : String(error),
          },
        ]);
      }
    }
    return this.#compileFallback(plan, snapshot, options, report.issues);
  }

  async #compileFallback(
    plan: ExecutionPlan,
    snapshot: ModelSnapshot,
    options: CompileOptions,
    webGpuIssues: readonly SupportIssue[],
  ): Promise<Session> {
    const requested = options.fallback;
    const candidates = requested === undefined
      ? this.#fallbackBackends
      : requested.flatMap((id) => this.#fallbackBackends.filter((backend) => backend.id === id));
    const fallbackIssues: SupportIssue[] = [...webGpuIssues];
    for (const backend of candidates) {
      const report = backend.inspect(plan, options);
      if (!report.supported) {
        fallbackIssues.push(...report.issues);
        continue;
      }
      try {
        return await backend.compile(plan, snapshot, options);
      } catch (error) {
        fallbackIssues.push({
          code: `${backend.id.toUpperCase()}_COMPILE_FAILED`,
          message: error instanceof Error ? error.message : String(error),
        });
      }
    }
    throw new SynapticError(
      "WebGPU is unsupported and no requested fallback could compile the plan",
      "NO_SUPPORTED_BACKEND",
      { issues: fallbackIssues },
    );
  }
}

interface WebGpuPipelines {
  readonly prepare: GPUComputePipeline;
  readonly forward: GPUComputePipeline;
  readonly gather: GPUComputePipeline;
}

export class WebGpuSession implements Session {
  readonly backend = "webgpu";
  readonly #plan: ExecutionPlan;
  readonly #definition: ModelSnapshot["definition"];
  readonly #parameters: Float32Array;
  readonly #device: GPUDevice;
  readonly #heap: WebGpuHeap;
  readonly #heapBuffer: GPUBuffer;
  readonly #uniformBuffer: GPUBuffer;
  readonly #readbackBuffer: GPUBuffer;
  readonly #bindGroup: GPUBindGroup;
  readonly #pipelines: WebGpuPipelines;
  readonly #uncapturedErrorHandler: EventListener;
  #step = 0;
  #busy = false;
  #disposed = false;
  #lostError: SynapticError | undefined;

  private constructor(
    plan: ExecutionPlan,
    snapshot: ModelSnapshot & { readonly parameters: Float32Array },
    device: GPUDevice,
    heap: WebGpuHeap,
    heapBuffer: GPUBuffer,
    uniformBuffer: GPUBuffer,
    readbackBuffer: GPUBuffer,
    bindGroup: GPUBindGroup,
    pipelines: WebGpuPipelines,
  ) {
    this.#plan = plan;
    this.#definition = snapshot.definition;
    this.#parameters = snapshot.parameters.slice();
    this.#device = device;
    this.#heap = heap;
    this.#heapBuffer = heapBuffer;
    this.#uniformBuffer = uniformBuffer;
    this.#readbackBuffer = readbackBuffer;
    this.#bindGroup = bindGroup;
    this.#pipelines = pipelines;
    this.#uncapturedErrorHandler = ((event: GPUUncapturedErrorEvent) => {
      this.#lostError = new SynapticError(
        event.error.message,
        "WEBGPU_UNCAPTURED_ERROR",
      );
    }) as EventListener;
    device.addEventListener("uncapturederror", this.#uncapturedErrorHandler);
    void device.lost.then((info) => {
      if (!this.#disposed) {
        this.#lostError = new SynapticError(
          `WebGPU device lost: ${info.message}`,
          "WEBGPU_DEVICE_LOST",
          { reason: info.reason },
        );
      }
    });
  }

  static async create(
    plan: ExecutionPlan,
    snapshot: ModelSnapshot,
    device: GPUDevice,
  ): Promise<WebGpuSession> {
    invariant(snapshot.parameters instanceof Float32Array, "WebGPU parameters must use f32 storage", "PRECISION_MISMATCH");
    const heap = buildWebGpuHeap(plan, snapshot);
    validateDeviceLimits(device, plan, heap);
    const uniformRecords = buildUniformRecords(plan, heap.layout);

    device.pushErrorScope("validation");
    const allocatedBuffers: GPUBuffer[] = [];
    let sessionResources:
      | {
        readonly heapBuffer: GPUBuffer;
        readonly uniformBuffer: GPUBuffer;
        readonly readbackBuffer: GPUBuffer;
        readonly bindGroup: GPUBindGroup;
        readonly pipelines: WebGpuPipelines;
      }
      | undefined;
    try {
      const heapBuffer = device.createBuffer({
        label: "synaptic heap",
        size: heap.layout.byteLength,
        usage: BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST | BUFFER_USAGE.COPY_SRC,
      });
      allocatedBuffers.push(heapBuffer);
      const uniformBuffer = device.createBuffer({
        label: "synaptic stage uniforms",
        size: uniformRecords.byteLength,
        usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
      });
      allocatedBuffers.push(uniformBuffer);
      const outputSection = heap.layout.sections.output;
      const readbackBuffer = device.createBuffer({
        label: "synaptic output readback",
        size: outputSection.byteLength,
        usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST,
      });
      allocatedBuffers.push(readbackBuffer);
      device.queue.writeBuffer(heapBuffer, 0, heap.bytes);
      device.queue.writeBuffer(uniformBuffer, 0, uniformRecords);

      const shader = device.createShaderModule({
        label: "synaptic forward shader",
        code: FORWARD_WGSL,
      });
      const compilation = await shader.getCompilationInfo();
      const shaderErrors = compilation.messages.filter((message) => message.type === "error");
      if (shaderErrors.length > 0) {
        throw new SynapticError(
          "WebGPU shader compilation failed",
          "WGSL_COMPILATION_FAILED",
          {
            messages: shaderErrors.map((message) =>
              `${message.lineNum}:${message.linePos} ${message.message}`),
          },
        );
      }

      const bindGroupLayout = device.createBindGroupLayout({
        label: "synaptic forward bindings",
        entries: [
          {
            binding: 0,
            visibility: COMPUTE_STAGE,
            buffer: { type: "storage" },
          },
          {
            binding: 1,
            visibility: COMPUTE_STAGE,
            buffer: {
              type: "uniform",
              hasDynamicOffset: true,
              minBindingSize: WEBGPU_UNIFORM_WORDS * 4,
            },
          },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({
        label: "synaptic forward pipeline layout",
        bindGroupLayouts: [bindGroupLayout],
      });
      const [prepare, forward, gather] = await Promise.all([
        device.createComputePipelineAsync({
          label: "synaptic prepare",
          layout: pipelineLayout,
          compute: { module: shader, entryPoint: "prepare" },
        }),
        device.createComputePipelineAsync({
          label: "synaptic forward stage",
          layout: pipelineLayout,
          compute: { module: shader, entryPoint: "forwardStage" },
        }),
        device.createComputePipelineAsync({
          label: "synaptic gather output",
          layout: pipelineLayout,
          compute: { module: shader, entryPoint: "gather" },
        }),
      ]);
      const bindGroup = device.createBindGroup({
        label: "synaptic forward bind group",
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: heapBuffer } },
          {
            binding: 1,
            resource: {
              buffer: uniformBuffer,
              offset: 0,
              size: WEBGPU_UNIFORM_WORDS * 4,
            },
          },
        ],
      });
      sessionResources = {
        heapBuffer,
        uniformBuffer,
        readbackBuffer,
        bindGroup,
        pipelines: { prepare, forward, gather },
      };
    } catch (error) {
      for (const buffer of allocatedBuffers) {
        buffer.destroy();
      }
      throw error;
    } finally {
      const validationError = await device.popErrorScope();
      if (validationError !== null) {
        sessionResources?.heapBuffer.destroy();
        sessionResources?.uniformBuffer.destroy();
        sessionResources?.readbackBuffer.destroy();
        throw new SynapticError(
          validationError.message,
          "WEBGPU_VALIDATION_ERROR",
        );
      }
    }
    invariant(sessionResources !== undefined, "Failed to initialize WebGPU resources", "WEBGPU_INITIALIZATION_FAILED");
    return new WebGpuSession(
      plan,
      { definition: snapshot.definition, parameters: snapshot.parameters },
      device,
      heap,
      sessionResources.heapBuffer,
      sessionResources.uniformBuffer,
      sessionResources.readbackBuffer,
      sessionResources.bindGroup,
      sessionResources.pipelines,
    );
  }

  async forward(input: TensorLike): Promise<Tensor> {
    this.#assertAvailable();
    invariant(!this.#busy, "WebGPU session operations must be awaited", "SESSION_BUSY");
    this.#busy = true;
    try {
      const values = readTensor(input, this.#plan.inputs.length, "Input");
      const inputSection = this.#heap.layout.sections.input;
      this.#device.queue.writeBuffer(
        this.#heapBuffer,
        inputSection.offset,
        values,
      );
      const encoder = this.#device.createCommandEncoder({
        label: `synaptic forward ${this.#step + 1}`,
      });

      this.#encodePass(
        encoder,
        this.#pipelines.prepare,
        0,
        dispatchCount(this.#plan.unitCount),
        "prepare recurrent context",
      );
      const stageCount = this.#plan.stageOffsets.length - 1;
      for (let stage = 0; stage < stageCount; stage += 1) {
        const length = (this.#plan.stageOffsets[stage + 1] ?? 0)
          - (this.#plan.stageOffsets[stage] ?? 0);
        this.#encodePass(
          encoder,
          this.#pipelines.forward,
          stage * WEBGPU_UNIFORM_STRIDE,
          dispatchCount(length),
          `forward stage ${stage}`,
        );
      }
      this.#encodePass(
        encoder,
        this.#pipelines.gather,
        0,
        dispatchCount(this.#plan.outputs.length),
        "gather output",
      );
      const outputSection = this.#heap.layout.sections.output;
      encoder.copyBufferToBuffer(
        this.#heapBuffer,
        outputSection.offset,
        this.#readbackBuffer,
        0,
        outputSection.byteLength,
      );
      this.#device.queue.submit([encoder.finish()]);
      await this.#readbackBuffer.mapAsync(MAP_READ, 0, outputSection.byteLength);
      const output = new Float32Array(
        this.#readbackBuffer.getMappedRange(0, outputSection.byteLength).slice(0),
      );
      this.#readbackBuffer.unmap();
      this.#step += 1;
      return { data: output, shape: [output.length] };
    } finally {
      this.#busy = false;
    }
  }

  async trainStep(
    _batch: TrainingBatch,
    _options?: TrainStepOptions,
  ): Promise<Metrics> {
    this.#assertAvailable();
    throw new SynapticError(
      "WebGPU training is not available in the forward-only milestone",
      "WEBGPU_TRAINING_UNAVAILABLE",
    );
  }

  async resetState(): Promise<void> {
    this.#assertAvailable();
    invariant(!this.#busy, "WebGPU session operations must be awaited", "SESSION_BUSY");
    const zeroes = new Float32Array(this.#plan.unitCount);
    for (const name of ["state", "activation", "previousActivation"] as const) {
      this.#device.queue.writeBuffer(
        this.#heapBuffer,
        this.#heap.layout.sections[name].offset,
        zeroes,
      );
    }
    this.#step = 0;
  }

  async snapshot(): Promise<ModelSnapshot> {
    this.#assertAvailable();
    return {
      definition: this.#definition,
      parameters: this.#parameters.slice(),
    };
  }

  async checkpoint(): Promise<ModelCheckpoint> {
    this.#assertAvailable();
    invariant(!this.#busy, "WebGPU session operations must be awaited", "SESSION_BUSY");
    this.#busy = true;
    const readback = this.#device.createBuffer({
      label: "synaptic checkpoint readback",
      size: this.#heap.layout.byteLength,
      usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST,
    });
    try {
      const encoder = this.#device.createCommandEncoder({
        label: "synaptic checkpoint",
      });
      encoder.copyBufferToBuffer(
        this.#heapBuffer,
        0,
        readback,
        0,
        this.#heap.layout.byteLength,
      );
      this.#device.queue.submit([encoder.finish()]);
      await readback.mapAsync(MAP_READ, 0, this.#heap.layout.byteLength);
      const bytes = readback.getMappedRange(0, this.#heap.layout.byteLength);
      const copySection = (name: "state" | "activation" | "previousActivation"): Float32Array => {
        const section = this.#heap.layout.sections[name];
        return new Float32Array(bytes.slice(section.offset, section.offset + section.byteLength));
      };
      const runtime: RuntimeState = {
        state: copySection("state"),
        activation: copySection("activation"),
        previousActivation: copySection("previousActivation"),
        derivative: new Float32Array(this.#plan.unitCount),
        eligibilityTrace: new Float32Array(this.#plan.connectionCount),
        extendedEligibilityTrace: new Float32Array(this.#plan.extendedTraceTarget.length),
        projectedError: new Float32Array(this.#plan.unitCount),
        gatedError: new Float32Array(this.#plan.unitCount),
        error: new Float32Array(this.#plan.unitCount),
        step: this.#step,
        randomSeed: 0,
        randomCounter: 0,
      };
      readback.unmap();
      return {
        definition: this.#definition,
        parameters: this.#parameters.slice(),
        runtime,
      };
    } finally {
      readback.destroy();
      this.#busy = false;
    }
  }

  dispose(): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.#device.removeEventListener("uncapturederror", this.#uncapturedErrorHandler);
    this.#heapBuffer.destroy();
    this.#uniformBuffer.destroy();
    this.#readbackBuffer.destroy();
    this.#device.destroy();
  }

  #encodePass(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    dynamicOffset: number,
    workgroups: number,
    label: string,
  ): void {
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.#bindGroup, [dynamicOffset]);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
  }

  #assertAvailable(): void {
    invariant(!this.#disposed, "The session has been disposed", "SESSION_DISPOSED");
    if (this.#lostError !== undefined) {
      throw this.#lostError;
    }
  }
}

function dispatchCount(length: number): number {
  return Math.max(1, Math.ceil(length / WORKGROUP_SIZE));
}

function validateDeviceLimits(
  device: GPUDevice,
  plan: ExecutionPlan,
  heap: WebGpuHeap,
): void {
  invariant(
    heap.layout.byteLength <= device.limits.maxStorageBufferBindingSize,
    "The compiled model exceeds maxStorageBufferBindingSize",
    "WEBGPU_STORAGE_LIMIT",
    {
      actual: heap.layout.byteLength,
      maximum: device.limits.maxStorageBufferBindingSize,
    },
  );
  invariant(
    heap.layout.byteLength <= device.limits.maxBufferSize,
    "The compiled model exceeds maxBufferSize",
    "WEBGPU_BUFFER_LIMIT",
    {
      actual: heap.layout.byteLength,
      maximum: device.limits.maxBufferSize,
    },
  );
  const largestStage = plan.stageOffsets.reduce((largest, offset, index, offsets) => {
    if (index === 0) {
      return largest;
    }
    return Math.max(largest, offset - (offsets[index - 1] ?? 0));
  }, 0);
  invariant(
    dispatchCount(Math.max(plan.unitCount, largestStage, plan.outputs.length))
      <= device.limits.maxComputeWorkgroupsPerDimension,
    "The compiled model exceeds maxComputeWorkgroupsPerDimension",
    "WEBGPU_DISPATCH_LIMIT",
  );
  invariant(
    device.limits.maxStorageBuffersPerShaderStage >= 1
      && device.limits.maxUniformBuffersPerShaderStage >= 1,
    "The device does not expose the required buffer bindings",
    "WEBGPU_BINDING_LIMIT",
  );
}
