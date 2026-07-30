import { CpuBackend } from "@synaptic/backend-cpu";
import { WasmBackend } from "@synaptic/backend-wasm";
import {
  SynapticError,
  copyOptimizerState,
  invariant,
  readTensor,
  validateCheckpoint,
  validateSnapshot,
  type Backend,
  type CompileOptions,
  type ExecutionPlan,
  type Metrics,
  type ModelCheckpoint,
  type ModelSnapshot,
  type RuntimeState,
  type SequenceMetrics,
  type SequenceOptions,
  type Session,
  type SupportIssue,
  type SupportReport,
  type Tensor,
  type TensorLike,
  type TrainingBatch,
  type TrainingSequenceStep,
  type TrainStepOptions,
} from "@synaptic/core";
import {
  WEBGPU_UNIFORM_STRIDE,
  WEBGPU_UNIFORM_WORDS,
  buildUniformRecords,
  buildWebGpuHeap,
  type HeapSectionName,
  type WebGpuHeap,
} from "./layout.js";
import { FORWARD_WGSL } from "./forward-shader.js";

export * from "./layout.js";
export * from "./forward-shader.js";
export * from "./growing-neural-ca.js";

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
    if ("runtime" in snapshot) {
      validateCheckpoint(plan, snapshot as ModelCheckpoint);
    }
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
        const session = await WebGpuSession.create(plan, snapshot, device);
        try {
          if ("runtime" in snapshot) {
            await session.restore(snapshot as ModelCheckpoint);
          }
          return session;
        } catch (error) {
          session.dispose();
          throw error;
        }
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
  readonly clearTraining: GPUComputePipeline;
  readonly backward: GPUComputePipeline;
  readonly updateParameters: GPUComputePipeline;
}

export class WebGpuSession implements Session {
  readonly backend = "webgpu";
  readonly #plan: ExecutionPlan;
  readonly #definition: ModelSnapshot["definition"];
  readonly #device: GPUDevice;
  readonly #heap: WebGpuHeap;
  readonly #heapBuffer: GPUBuffer;
  readonly #uniformBuffer: GPUBuffer;
  readonly #readbackBuffer: GPUBuffer;
  readonly #bindGroup: GPUBindGroup;
  readonly #pipelines: WebGpuPipelines;
  readonly #uncapturedErrorHandler: EventListener;
  readonly #stateResetBytes: Uint8Array;
  readonly #sequenceBuffers: GPUBuffer[] = [];
  #sequenceUploadBuffer:
    | { readonly buffer: GPUBuffer; readonly capacity: number }
    | undefined;
  #sequenceReadbackBuffer:
    | { readonly buffer: GPUBuffer; readonly capacity: number }
    | undefined;
  #step = 0;
  #randomSeed = 0;
  #randomCounter = 0;
  #optimizer: ModelCheckpoint["optimizer"];
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
    this.#device = device;
    this.#heap = heap;
    this.#heapBuffer = heapBuffer;
    this.#uniformBuffer = uniformBuffer;
    this.#readbackBuffer = readbackBuffer;
    this.#bindGroup = bindGroup;
    this.#pipelines = pipelines;
    const stateStart = heap.layout.sections.state.offset;
    const stateEnd = heap.layout.sections.connectionGradient.offset
      + heap.layout.sections.connectionGradient.byteLength;
    this.#stateResetBytes = new Uint8Array(stateEnd - stateStart);
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
      const [
        prepare,
        forward,
        gather,
        clearTraining,
        backward,
        updateParameters,
      ] = await Promise.all([
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
        device.createComputePipelineAsync({
          label: "synaptic clear training state",
          layout: pipelineLayout,
          compute: { module: shader, entryPoint: "clearTraining" },
        }),
        device.createComputePipelineAsync({
          label: "synaptic backward stage",
          layout: pipelineLayout,
          compute: { module: shader, entryPoint: "backwardStage" },
        }),
        device.createComputePipelineAsync({
          label: "synaptic update parameters",
          layout: pipelineLayout,
          compute: { module: shader, entryPoint: "updateParameters" },
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
        pipelines: {
          prepare,
          forward,
          gather,
          clearTraining,
          backward,
          updateParameters,
        },
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
    const outputs = await this.forwardSequence([input]);
    const output = outputs[0];
    invariant(output !== undefined, "Forward execution produced no output", "WEBGPU_EXECUTION_FAILED");
    return output;
  }

  async forwardSequence(inputs: readonly TensorLike[]): Promise<readonly Tensor[]> {
    this.#assertAvailable();
    invariant(!this.#busy, "WebGPU session operations must be awaited", "SESSION_BUSY");
    if (inputs.length === 0) {
      return [];
    }
    this.#busy = true;
    try {
      const inputLength = this.#plan.inputs.length;
      const inputByteLength = inputLength * Float32Array.BYTES_PER_ELEMENT;
      const packed = new Float32Array(inputs.length * inputLength);
      for (let index = 0; index < inputs.length; index += 1) {
        packed.set(
          readTensor(inputs[index] ?? [], inputLength, `Input ${index}`),
          index * inputLength,
        );
      }
      const upload = this.#uploadBuffer(packed.byteLength);
      this.#device.queue.writeBuffer(upload, 0, packed);

      const output = this.#heap.layout.sections.output;
      const outputByteLength = output.byteLength * inputs.length;
      const readback = this.#sequenceOutputBuffer(outputByteLength);
      const encoder = this.#device.createCommandEncoder({
        label: `synaptic forward sequence (${inputs.length} steps)`,
      });
      for (let index = 0; index < inputs.length; index += 1) {
        encoder.copyBufferToBuffer(
          upload,
          index * inputByteLength,
          this.#heapBuffer,
          this.#heap.layout.sections.input.offset,
          inputByteLength,
        );
        this.#encodeForward(encoder);
        encoder.copyBufferToBuffer(
          this.#heapBuffer,
          output.offset,
          readback,
          index * output.byteLength,
          output.byteLength,
        );
      }
      this.#device.queue.submit([encoder.finish()]);
      await readback.mapAsync(MAP_READ, 0, outputByteLength);
      let values: Float32Array;
      try {
        values = new Float32Array(
          readback.getMappedRange(0, outputByteLength),
        ).slice();
      } finally {
        readback.unmap();
      }
      this.#step += inputs.length;
      return inputs.map((_, index) => {
        const start = index * output.length;
        const data = values.slice(start, start + output.length);
        return { data, shape: [data.length] };
      });
    } finally {
      this.#busy = false;
    }
  }

  async trainStep(
    batch: TrainingBatch,
    options: TrainStepOptions = {},
  ): Promise<Metrics> {
    const metrics = await this.trainSequence(
      [{
        input: batch.input,
        target: batch.target,
        ...(options.learningRate === undefined
          ? {}
          : { learningRate: options.learningRate }),
      }],
    );
    invariant(metrics.loss !== undefined, "Training step produced no loss", "WEBGPU_EXECUTION_FAILED");
    return { loss: metrics.loss, step: metrics.step };
  }

  async trainSequence(
    sequence: readonly TrainingSequenceStep[],
    options: SequenceOptions = {},
  ): Promise<SequenceMetrics> {
    this.#assertAvailable();
    invariant(!this.#busy, "WebGPU session operations must be awaited", "SESSION_BUSY");
    invariant(
      sequence.length > 0,
      "A training sequence must contain at least one step",
      "EMPTY_TRAINING_SEQUENCE",
    );
    this.#busy = true;
    try {
      const inputLength = this.#plan.inputs.length;
      const targetLength = this.#plan.outputs.length;
      const recordLength = inputLength + targetLength + 1;
      const recordByteLength = recordLength * Float32Array.BYTES_PER_ELEMENT;
      const packed = new Float32Array(sequence.length * recordLength);
      let lastTarget: Float32Array<ArrayBufferLike> = new Float32Array(targetLength);
      for (let index = 0; index < sequence.length; index += 1) {
        const trainingStep = sequence[index];
        invariant(trainingStep !== undefined, "Training sequence is sparse", "INVALID_TRAINING_SEQUENCE");
        const input = readTensor(trainingStep.input, inputLength, `Input ${index}`);
        const target = readTensor(trainingStep.target, targetLength, `Target ${index}`);
        const learningRate = trainingStep.learningRate ?? options.learningRate ?? 0.1;
        invariant(
          Number.isFinite(learningRate) && learningRate >= 0,
          "Learning rate must be a finite non-negative number",
          "INVALID_LEARNING_RATE",
        );
        const recordStart = index * recordLength;
        packed.set(input, recordStart);
        packed.set(target, recordStart + inputLength);
        packed[recordStart + inputLength + targetLength] = learningRate;
        if (index === sequence.length - 1) {
          lastTarget = target;
        }
      }

      const upload = this.#uploadBuffer(packed.byteLength);
      this.#device.queue.writeBuffer(upload, 0, packed);
      const includeMetrics = options.metrics !== "none";
      const encoder = this.#device.createCommandEncoder({
        label: `synaptic training sequence (${sequence.length} steps)`,
      });
      for (let index = 0; index < sequence.length; index += 1) {
        const recordOffset = index * recordByteLength;
        encoder.copyBufferToBuffer(
          upload,
          recordOffset,
          this.#heapBuffer,
          this.#heap.layout.sections.input.offset,
          inputLength * Float32Array.BYTES_PER_ELEMENT,
        );
        encoder.copyBufferToBuffer(
          upload,
          recordOffset + inputLength * Float32Array.BYTES_PER_ELEMENT,
          this.#heapBuffer,
          this.#heap.layout.sections.target.offset,
          targetLength * Float32Array.BYTES_PER_ELEMENT,
        );
        encoder.copyBufferToBuffer(
          upload,
          recordOffset + (inputLength + targetLength) * Float32Array.BYTES_PER_ELEMENT,
          this.#heapBuffer,
          this.#heap.layout.sections.learningRate.offset,
          Float32Array.BYTES_PER_ELEMENT,
        );
        const gatherOutput = includeMetrics && index === sequence.length - 1;
        this.#encodeForward(encoder, gatherOutput);
        this.#encodeTraining(encoder);
      }
      if (includeMetrics) {
        this.#encodeOutputCopy(encoder);
      }
      this.#device.queue.submit([encoder.finish()]);

      let loss: number | undefined;
      if (includeMetrics) {
        const output = await this.#readOutput();
        loss = 0;
        for (let index = 0; index < lastTarget.length; index += 1) {
          const difference = (lastTarget[index] ?? 0) - (output[index] ?? 0);
          loss += difference * difference;
        }
        loss /= lastTarget.length;
      }
      this.#step += sequence.length;
      return {
        steps: sequence.length,
        step: this.#step,
        ...(loss === undefined ? {} : { loss }),
      };
    } finally {
      this.#busy = false;
    }
  }

  async resetState(): Promise<void> {
    this.#assertAvailable();
    invariant(!this.#busy, "WebGPU session operations must be awaited", "SESSION_BUSY");
    if (this.#stateResetBytes.byteLength > 0) {
      this.#device.queue.writeBuffer(
        this.#heapBuffer,
        this.#heap.layout.sections.state.offset,
        this.#stateResetBytes,
      );
    }
    this.#step = 0;
    this.#randomCounter = 0;
  }

  async restore(checkpoint: ModelCheckpoint): Promise<void> {
    this.#assertAvailable();
    invariant(!this.#busy, "WebGPU session operations must be awaited", "SESSION_BUSY");
    validateCheckpoint(this.#plan, checkpoint);
    invariant(
      checkpoint.parameters instanceof Float32Array,
      "WebGPU checkpoints must use f32 storage",
      "PRECISION_MISMATCH",
    );
    const values = {
      parameters: checkpoint.parameters,
      state: checkpoint.runtime.state,
      activation: checkpoint.runtime.activation,
      previousActivation: checkpoint.runtime.previousActivation,
      derivative: checkpoint.runtime.derivative,
      eligibilityTrace: checkpoint.runtime.eligibilityTrace,
      extendedEligibilityTrace: checkpoint.runtime.extendedEligibilityTrace,
      projectedError: checkpoint.runtime.projectedError,
      gatedError: checkpoint.runtime.gatedError,
      error: checkpoint.runtime.error,
    } as const;
    for (const [name, data] of Object.entries(values) as [
      keyof typeof values,
      Float32Array,
    ][]) {
      const section = this.#heap.layout.sections[name];
      if (section.byteLength > 0) {
        this.#device.queue.writeBuffer(this.#heapBuffer, section.offset, data);
      }
    }
    this.#step = checkpoint.runtime.step;
    this.#randomSeed = checkpoint.runtime.randomSeed;
    this.#randomCounter = checkpoint.runtime.randomCounter;
    this.#optimizer = checkpoint.optimizer === undefined
      ? undefined
      : copyOptimizerState(checkpoint.optimizer);
  }

  async snapshot(): Promise<ModelSnapshot> {
    this.#assertAvailable();
    invariant(!this.#busy, "WebGPU session operations must be awaited", "SESSION_BUSY");
    this.#busy = true;
    try {
      return {
        definition: this.#definition,
        parameters: await this.#readHeapSection("parameters"),
      };
    } finally {
      this.#busy = false;
    }
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
      const bytes = new Uint8Array(
        readback.getMappedRange(0, this.#heap.layout.byteLength),
      ).slice();
      readback.unmap();
      const copySection = (
        name:
          | "parameters"
          | "state"
          | "activation"
          | "previousActivation"
          | "derivative"
          | "eligibilityTrace"
          | "extendedEligibilityTrace"
          | "projectedError"
          | "gatedError"
          | "error",
      ): Float32Array => {
        const section = this.#heap.layout.sections[name];
        return new Float32Array(
          bytes.buffer.slice(section.offset, section.offset + section.byteLength),
        );
      };
      const runtime: RuntimeState = {
        state: copySection("state"),
        activation: copySection("activation"),
        previousActivation: copySection("previousActivation"),
        derivative: copySection("derivative"),
        eligibilityTrace: copySection("eligibilityTrace"),
        extendedEligibilityTrace: copySection("extendedEligibilityTrace"),
        projectedError: copySection("projectedError"),
        gatedError: copySection("gatedError"),
        error: copySection("error"),
        step: this.#step,
        randomSeed: this.#randomSeed,
        randomCounter: this.#randomCounter,
      };
      return {
        definition: this.#definition,
        parameters: copySection("parameters"),
        runtime,
        ...(this.#optimizer === undefined
          ? {}
          : { optimizer: copyOptimizerState(this.#optimizer) }),
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
    for (const buffer of this.#sequenceBuffers) {
      buffer.destroy();
    }
    this.#device.destroy();
  }

  #encodeForward(
    encoder: GPUCommandEncoder,
    gatherOutput = true,
  ): void {
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
    if (gatherOutput) {
      this.#encodePass(
        encoder,
        this.#pipelines.gather,
        0,
        dispatchCount(this.#plan.outputs.length),
        "gather output",
      );
    }
  }

  #encodeTraining(encoder: GPUCommandEncoder): void {
    this.#encodePass(
      encoder,
      this.#pipelines.clearTraining,
      0,
      dispatchCount(Math.max(this.#plan.unitCount, this.#plan.connectionCount)),
      "clear training state",
    );
    const stageCount = this.#plan.stageOffsets.length - 1;
    for (let stage = stageCount - 1; stage > 0; stage -= 1) {
      const length = (this.#plan.stageOffsets[stage + 1] ?? 0)
        - (this.#plan.stageOffsets[stage] ?? 0);
      this.#encodePass(
        encoder,
        this.#pipelines.backward,
        stage * WEBGPU_UNIFORM_STRIDE,
        dispatchCount(length),
        `backward stage ${stage}`,
      );
    }
    this.#encodePass(
      encoder,
      this.#pipelines.updateParameters,
      0,
      dispatchCount(this.#plan.parameterCount),
      "update parameters",
    );
  }

  #encodeOutputCopy(encoder: GPUCommandEncoder): void {
    const output = this.#heap.layout.sections.output;
    encoder.copyBufferToBuffer(
      this.#heapBuffer,
      output.offset,
      this.#readbackBuffer,
      0,
      output.byteLength,
    );
  }

  async #readOutput(): Promise<Float32Array> {
    const output = this.#heap.layout.sections.output;
    await this.#readbackBuffer.mapAsync(MAP_READ, 0, output.byteLength);
    const bytes = new Uint8Array(
      this.#readbackBuffer.getMappedRange(0, output.byteLength),
    ).slice();
    this.#readbackBuffer.unmap();
    return new Float32Array(bytes.buffer);
  }

  #uploadBuffer(byteLength: number): GPUBuffer {
    const current = this.#sequenceUploadBuffer;
    if (current !== undefined && current.capacity >= byteLength) {
      return current.buffer;
    }
    const capacity = alignedBufferSize(byteLength);
    this.#assertBufferSize(capacity);
    const buffer = this.#device.createBuffer({
      label: "synaptic sequence upload",
      size: capacity,
      usage: BUFFER_USAGE.COPY_SRC | BUFFER_USAGE.COPY_DST,
    });
    this.#sequenceBuffers.push(buffer);
    this.#sequenceUploadBuffer = { buffer, capacity };
    return buffer;
  }

  #sequenceOutputBuffer(byteLength: number): GPUBuffer {
    const current = this.#sequenceReadbackBuffer;
    if (current !== undefined && current.capacity >= byteLength) {
      return current.buffer;
    }
    const capacity = alignedBufferSize(byteLength);
    this.#assertBufferSize(capacity);
    const buffer = this.#device.createBuffer({
      label: "synaptic sequence output readback",
      size: capacity,
      usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST,
    });
    this.#sequenceBuffers.push(buffer);
    this.#sequenceReadbackBuffer = { buffer, capacity };
    return buffer;
  }

  #assertBufferSize(byteLength: number): void {
    invariant(
      byteLength <= this.#device.limits.maxBufferSize,
      `WebGPU sequence buffer requires ${byteLength} bytes, exceeding the device limit`,
      "WEBGPU_SEQUENCE_TOO_LARGE",
      {
        byteLength,
        maxBufferSize: this.#device.limits.maxBufferSize,
      },
    );
  }

  async #readHeapSection(name: HeapSectionName): Promise<Float32Array> {
    const section = this.#heap.layout.sections[name];
    if (section.byteLength === 0) {
      return new Float32Array();
    }
    const readback = this.#device.createBuffer({
      label: `synaptic ${name} readback`,
      size: section.byteLength,
      usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST,
    });
    try {
      const encoder = this.#device.createCommandEncoder({
        label: `synaptic ${name} snapshot`,
      });
      encoder.copyBufferToBuffer(
        this.#heapBuffer,
        section.offset,
        readback,
        0,
        section.byteLength,
      );
      this.#device.queue.submit([encoder.finish()]);
      await readback.mapAsync(MAP_READ, 0, section.byteLength);
      const bytes = new Uint8Array(
        readback.getMappedRange(0, section.byteLength),
      ).slice();
      readback.unmap();
      return new Float32Array(bytes.buffer);
    } finally {
      readback.destroy();
    }
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

function alignedBufferSize(byteLength: number): number {
  return Math.max(
    Float32Array.BYTES_PER_ELEMENT,
    Math.ceil(byteLength / Float32Array.BYTES_PER_ELEMENT)
      * Float32Array.BYTES_PER_ELEMENT,
  );
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
    dispatchCount(Math.max(
      plan.unitCount,
      plan.connectionCount,
      plan.parameterCount,
      largestStage,
      plan.outputs.length,
    ))
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
