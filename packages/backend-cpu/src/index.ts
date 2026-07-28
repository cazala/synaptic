import {
  SynapticError,
  compilePlan,
  createCheckpoint,
  createRuntimeState,
  invariant,
  readTensor,
  validateSnapshot,
  type Backend,
  type CompileOptions,
  type DenseStageSpecialization,
  type ExecutionPlan,
  type Metrics,
  type ModelCheckpoint,
  type ModelSnapshot,
  type MutableRuntimeState,
  type Session,
  type SupportReport,
  type Tensor,
  type TensorLike,
  type TrainingBatch,
  type TrainStepOptions,
} from "@synaptic/core";

const DEFAULT_LEARNING_RATE = 0.1;

export class CpuBackend implements Backend {
  readonly id = "cpu";

  inspect(plan: ExecutionPlan, options: CompileOptions = {}): SupportReport {
    const issues = [];
    if (plan.definition.precision !== "f32") {
      issues.push({
        code: "UNSUPPORTED_PRECISION",
        message: "The CPU production backend uses portable f32 precision",
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
      throw new SynapticError("The CPU backend cannot compile this plan", "UNSUPPORTED_PLAN", {
        issues: report.issues,
      });
    }
    invariant(snapshot.parameters instanceof Float32Array, "CPU parameters must use f32 storage", "PRECISION_MISMATCH");
    return new CpuSession(plan, snapshot);
  }
}

export class CpuSession implements Session {
  readonly backend = "cpu";
  readonly #plan: ExecutionPlan;
  readonly #definition: ModelSnapshot["definition"];
  readonly #parameters: Float32Array;
  readonly #runtime: MutableRuntimeState;
  readonly #inputSlot: Int32Array;
  readonly #outputSlot: Int32Array;
  readonly #denseStages: ReadonlyMap<number, DenseStageSpecialization>;
  #disposed = false;

  constructor(plan: ExecutionPlan, snapshot: ModelSnapshot) {
    this.#plan = plan;
    this.#definition = snapshot.definition;
    this.#parameters = new Float32Array(snapshot.parameters);
    this.#runtime = createRuntimeState(plan, "f32");
    this.#inputSlot = new Int32Array(plan.unitCount);
    this.#outputSlot = new Int32Array(plan.unitCount);
    this.#inputSlot.fill(-1);
    this.#outputSlot.fill(-1);
    plan.inputs.forEach((unit, slot) => {
      this.#inputSlot[unit] = slot;
    });
    plan.outputs.forEach((unit, slot) => {
      this.#outputSlot[unit] = slot;
    });
    this.#denseStages = new Map(
      plan.specializations.denseStages.map((specialization) => [
        specialization.stage,
        specialization,
      ]),
    );
  }

  async forward(input: TensorLike): Promise<Tensor> {
    this.#assertActive();
    const values = readTensor(input, this.#plan.inputs.length, "Input");
    const runtime = this.#runtime;
    runtime.previousActivation.set(runtime.activation);
    const stageCount = this.#plan.stageOffsets.length - 1;

    for (let stage = 0; stage < stageCount; stage += 1) {
      const start = this.#plan.stageOffsets[stage] ?? 0;
      const end = this.#plan.stageOffsets[stage + 1] ?? start;
      const dense = this.#denseStages.get(stage);
      if (dense !== undefined) {
        this.#activateDenseStage(dense);
        continue;
      }
      for (let cursor = start; cursor < end; cursor += 1) {
        const unit = this.#plan.stageUnits[cursor];
        invariant(unit !== undefined, "Stage references an unknown unit", "INVALID_PLAN");
        const inputSlot = this.#inputSlot[unit] ?? -1;
        const constant = this.#plan.unitConstant[unit] ?? Number.NaN;
        if (inputSlot >= 0) {
          const value = values[inputSlot] ?? 0;
          runtime.state[unit] = value;
          runtime.activation[unit] = value;
          runtime.derivative[unit] = 0;
        } else if (!Number.isNaN(constant)) {
          runtime.state[unit] = constant;
          runtime.activation[unit] = constant;
          runtime.derivative[unit] = 0;
        } else {
          this.#activateUnit(unit);
        }
      }
    }

    runtime.step += 1;
    const output = new Float32Array(this.#plan.outputs.length);
    this.#plan.outputs.forEach((unit, slot) => {
      output[slot] = runtime.activation[unit] ?? 0;
    });
    return { data: output, shape: [output.length] };
  }

  async trainStep(
    batch: TrainingBatch,
    options: TrainStepOptions = {},
  ): Promise<Metrics> {
    const predicted = await this.forward(batch.input);
    const target = readTensor(batch.target, this.#plan.outputs.length, "Target");
    const learningRate = options.learningRate ?? DEFAULT_LEARNING_RATE;
    invariant(
      Number.isFinite(learningRate) && learningRate >= 0,
      "Learning rate must be a finite non-negative number",
      "INVALID_LEARNING_RATE",
    );
    this.#propagate(target, learningRate);

    let loss = 0;
    for (let index = 0; index < target.length; index += 1) {
      const difference = (target[index] ?? 0) - (predicted.data[index] ?? 0);
      loss = Math.fround(loss + Math.fround(difference * difference));
    }
    return { loss: loss / target.length, step: this.#runtime.step };
  }

  async resetState(): Promise<void> {
    this.#assertActive();
    const runtime = this.#runtime;
    runtime.state.fill(0);
    runtime.activation.fill(0);
    runtime.previousActivation.fill(0);
    runtime.derivative.fill(0);
    runtime.eligibilityTrace.fill(0);
    runtime.extendedEligibilityTrace.fill(0);
    runtime.projectedError.fill(0);
    runtime.gatedError.fill(0);
    runtime.error.fill(0);
    runtime.step = 0;
    runtime.randomCounter = 0;
  }

  async snapshot(): Promise<ModelSnapshot> {
    this.#assertActive();
    return { definition: this.#definition, parameters: this.#parameters.slice() };
  }

  async checkpoint(): Promise<ModelCheckpoint> {
    this.#assertActive();
    return createCheckpoint(
      { definition: this.#definition, parameters: this.#parameters },
      this.#runtime,
    );
  }

  dispose(): void {
    this.#disposed = true;
  }

  #activateUnit(unit: number): void {
    const runtime = this.#runtime;
    const incomingStart = this.#plan.incomingOffsets[unit] ?? 0;
    const incomingEnd = this.#plan.incomingOffsets[unit + 1] ?? incomingStart;
    const previousState = runtime.state[unit] ?? 0;
    let recurrence = 0;
    let state = 0;

    for (let cursor = incomingStart; cursor < incomingEnd; cursor += 1) {
      const connection = this.#plan.incomingConnections[cursor];
      invariant(connection !== undefined, "Incoming adjacency is invalid", "INVALID_PLAN");
      const gain = this.#gain(connection);
      const parameter = this.#plan.connectionParameter[connection] ?? 0;
      const weight = this.#parameters[parameter] ?? 0;
      if (this.#plan.connectionFrom[connection] === unit) {
        recurrence = Math.fround(recurrence + Math.fround(gain * weight));
        state = Math.fround(state + Math.fround(gain * weight * previousState));
      } else {
        state = Math.fround(state + Math.fround(gain * weight * this.#sourceValue(connection)));
      }
    }

    runtime.state[unit] = state;
    const code = this.#plan.unitActivation[unit] ?? 0;
    const [activation, derivative] = activationAndDerivative(code, state);
    runtime.activation[unit] = activation;
    runtime.derivative[unit] = derivative;

    for (let cursor = incomingStart; cursor < incomingEnd; cursor += 1) {
      const connection = this.#plan.incomingConnections[cursor];
      invariant(connection !== undefined, "Incoming adjacency is invalid", "INVALID_PLAN");
      if (this.#plan.connectionFrom[connection] === unit) {
        continue;
      }
      const eligibility = Math.fround(
        Math.fround(recurrence * (runtime.eligibilityTrace[connection] ?? 0))
        + Math.fround(this.#gain(connection) * this.#sourceValue(connection)),
      );
      runtime.eligibilityTrace[connection] = eligibility;
      const traceStart = this.#plan.extendedTraceOffsets[connection] ?? 0;
      const traceEnd = this.#plan.extendedTraceOffsets[connection + 1] ?? traceStart;
      for (let trace = traceStart; trace < traceEnd; trace += 1) {
        const target = this.#plan.extendedTraceTarget[trace];
        invariant(target !== undefined, "Extended trace has no target", "INVALID_PLAN");
        runtime.extendedEligibilityTrace[trace] = Math.fround(
          Math.fround(this.#selfFactor(target) * (runtime.extendedEligibilityTrace[trace] ?? 0))
          + Math.fround(derivative * eligibility * this.#bigParenthesis(target, unit)),
        );
      }
    }
  }

  #activateDenseStage(specialization: DenseStageSpecialization): void {
    const runtime = this.#runtime;
    const width = specialization.sources.length;
    for (let row = 0; row < specialization.units.length; row += 1) {
      const unit = specialization.units[row];
      invariant(unit !== undefined, "Dense specialization has an unknown unit", "INVALID_PLAN");
      let state = 0;
      for (let column = 0; column < width; column += 1) {
        const source = specialization.sources[column];
        const connection = specialization.connections[row * width + column];
        invariant(
          source !== undefined && connection !== undefined,
          "Dense specialization matrix is incomplete",
          "INVALID_PLAN",
        );
        const parameter = this.#plan.connectionParameter[connection] ?? 0;
        const sourceActivation = runtime.activation[source] ?? 0;
        state = Math.fround(
          state + Math.fround((this.#parameters[parameter] ?? 0) * sourceActivation),
        );
        runtime.eligibilityTrace[connection] = sourceActivation;
      }
      runtime.state[unit] = state;
      const [activation, derivative] = activationAndDerivative(
        this.#plan.unitActivation[unit] ?? 0,
        state,
      );
      runtime.activation[unit] = activation;
      runtime.derivative[unit] = derivative;

      for (let column = 0; column < width; column += 1) {
        const connection = specialization.connections[row * width + column];
        invariant(connection !== undefined, "Dense specialization matrix is incomplete", "INVALID_PLAN");
        const traceStart = this.#plan.extendedTraceOffsets[connection] ?? 0;
        const traceEnd = this.#plan.extendedTraceOffsets[connection + 1] ?? traceStart;
        for (let trace = traceStart; trace < traceEnd; trace += 1) {
          const target = this.#plan.extendedTraceTarget[trace];
          invariant(target !== undefined, "Extended trace has no target", "INVALID_PLAN");
          runtime.extendedEligibilityTrace[trace] = Math.fround(
            Math.fround(this.#selfFactor(target) * (runtime.extendedEligibilityTrace[trace] ?? 0))
            + Math.fround(
              derivative
              * (runtime.eligibilityTrace[connection] ?? 0)
              * this.#bigParenthesis(target, unit),
            ),
          );
        }
      }
    }
  }

  #propagate(target: Float32Array, learningRate: number): void {
    const runtime = this.#runtime;
    runtime.projectedError.fill(0);
    runtime.gatedError.fill(0);
    runtime.error.fill(0);
    const gradients = new Float32Array(this.#plan.parameterCount);
    const stageCount = this.#plan.stageOffsets.length - 1;

    for (let stage = stageCount - 1; stage > 0; stage -= 1) {
      const start = this.#plan.stageOffsets[stage] ?? 0;
      const end = this.#plan.stageOffsets[stage + 1] ?? start;
      for (let cursor = end - 1; cursor >= start; cursor -= 1) {
        const unit = this.#plan.stageUnits[cursor];
        invariant(unit !== undefined, "Stage references an unknown unit", "INVALID_PLAN");
        const outputSlot = this.#outputSlot[unit] ?? -1;
        if (outputSlot >= 0) {
          runtime.projectedError[unit] = Math.fround(
            (target[outputSlot] ?? 0) - (runtime.activation[unit] ?? 0),
          );
          runtime.gatedError[unit] = 0;
        } else {
          let projected = 0;
          const outgoingStart = this.#plan.outgoingOffsets[unit] ?? 0;
          const outgoingEnd = this.#plan.outgoingOffsets[unit + 1] ?? outgoingStart;
          for (let outgoing = outgoingStart; outgoing < outgoingEnd; outgoing += 1) {
            const connection = this.#plan.outgoingConnections[outgoing];
            if (connection === undefined || this.#plan.connectionDelay[connection] !== 0) {
              continue;
            }
            const targetUnit = this.#plan.connectionTo[connection] ?? 0;
            const parameter = this.#plan.connectionParameter[connection] ?? 0;
            projected = Math.fround(
              projected
              + Math.fround(
                (runtime.error[targetUnit] ?? 0)
                * this.#gain(connection)
                * (this.#parameters[parameter] ?? 0),
              ),
            );
          }
          runtime.projectedError[unit] = Math.fround((runtime.derivative[unit] ?? 0) * projected);

          let gated = 0;
          const gatedStart = this.#plan.gatedTargetOffsets[unit] ?? 0;
          const gatedEnd = this.#plan.gatedTargetOffsets[unit + 1] ?? gatedStart;
          for (let gatedCursor = gatedStart; gatedCursor < gatedEnd; gatedCursor += 1) {
            const targetUnit = this.#plan.gatedTargets[gatedCursor];
            invariant(targetUnit !== undefined, "Gated adjacency is invalid", "INVALID_PLAN");
            gated = Math.fround(
              gated
              + Math.fround((runtime.error[targetUnit] ?? 0) * this.#bigParenthesis(targetUnit, unit)),
            );
          }
          runtime.gatedError[unit] = Math.fround((runtime.derivative[unit] ?? 0) * gated);
        }
        runtime.error[unit] = Math.fround(
          (runtime.projectedError[unit] ?? 0) + (runtime.gatedError[unit] ?? 0),
        );

        const incomingStart = this.#plan.incomingOffsets[unit] ?? 0;
        const incomingEnd = this.#plan.incomingOffsets[unit + 1] ?? incomingStart;
        for (let incoming = incomingStart; incoming < incomingEnd; incoming += 1) {
          const connection = this.#plan.incomingConnections[incoming];
          if (connection === undefined || this.#plan.connectionFrom[connection] === unit) {
            continue;
          }
          let gradient = Math.fround(
            (runtime.projectedError[unit] ?? 0) * (runtime.eligibilityTrace[connection] ?? 0),
          );
          const traceStart = this.#plan.extendedTraceOffsets[connection] ?? 0;
          const traceEnd = this.#plan.extendedTraceOffsets[connection + 1] ?? traceStart;
          for (let trace = traceStart; trace < traceEnd; trace += 1) {
            const targetUnit = this.#plan.extendedTraceTarget[trace] ?? 0;
            gradient = Math.fround(
              gradient
              + Math.fround(
                (runtime.error[targetUnit] ?? 0)
                * (runtime.extendedEligibilityTrace[trace] ?? 0),
              ),
            );
          }
          const parameter = this.#plan.connectionParameter[connection] ?? 0;
          gradients[parameter] = Math.fround((gradients[parameter] ?? 0) + gradient);
        }
      }
    }

    for (const parameter of this.#plan.definition.topology.parameters) {
      if (parameter.trainable) {
        this.#parameters[parameter.id] = Math.fround(
          (this.#parameters[parameter.id] ?? 0)
          + Math.fround(learningRate * (gradients[parameter.id] ?? 0)),
        );
      }
    }
  }

  #gain(connection: number): number {
    const gater = this.#plan.connectionGater[connection] ?? -1;
    if (gater < 0) {
      return 1;
    }
    return this.#plan.gateDelay[connection] === 0
      ? (this.#runtime.activation[gater] ?? 0)
      : (this.#runtime.previousActivation[gater] ?? 0);
  }

  #sourceValue(connection: number): number {
    const source = this.#plan.connectionFrom[connection] ?? 0;
    return this.#plan.connectionDelay[connection] === 0
      ? (this.#runtime.activation[source] ?? 0)
      : (this.#runtime.previousActivation[source] ?? 0);
  }

  #selfFactor(unit: number): number {
    let result = 0;
    const start = this.#plan.incomingOffsets[unit] ?? 0;
    const end = this.#plan.incomingOffsets[unit + 1] ?? start;
    for (let cursor = start; cursor < end; cursor += 1) {
      const connection = this.#plan.incomingConnections[cursor];
      if (connection !== undefined && this.#plan.connectionFrom[connection] === unit) {
        const parameter = this.#plan.connectionParameter[connection] ?? 0;
        result = Math.fround(result + Math.fround(this.#gain(connection) * (this.#parameters[parameter] ?? 0)));
      }
    }
    return result;
  }

  #bigParenthesis(target: number, gater: number): number {
    let result = 0;
    const start = this.#plan.incomingOffsets[target] ?? 0;
    const end = this.#plan.incomingOffsets[target + 1] ?? start;
    for (let cursor = start; cursor < end; cursor += 1) {
      const connection = this.#plan.incomingConnections[cursor];
      if (
        connection === undefined
        || this.#plan.connectionGater[connection] !== gater
        || this.#plan.gateDelay[connection] !== 0
      ) {
        continue;
      }
      const parameter = this.#plan.connectionParameter[connection] ?? 0;
      const value = this.#plan.connectionFrom[connection] === target
        ? (this.#runtime.state[target] ?? 0)
        : this.#sourceValue(connection);
      result = Math.fround(result + Math.fround((this.#parameters[parameter] ?? 0) * value));
    }
    return result;
  }

  #assertActive(): void {
    invariant(!this.#disposed, "The session has been disposed", "SESSION_DISPOSED");
  }
}

function activationAndDerivative(code: number, state: number): readonly [number, number] {
  let activation: number;
  switch (code) {
    case 1:
      activation = Math.fround(1 / (1 + Math.exp(-state)));
      return [activation, Math.fround(activation * (1 - activation))];
    case 2:
      activation = Math.fround(Math.tanh(state));
      return [activation, Math.fround(1 - activation * activation)];
    case 3:
      activation = Math.fround(Math.max(0, state));
      return [activation, state > 0 ? 1 : 0];
    case 4:
      return [state > 0 ? 1 : 0, 0];
    default:
      return [state, 1];
  }
}

export function compileCpu(
  definition: ModelSnapshot["definition"],
  snapshot: ModelSnapshot,
  options?: CompileOptions,
): Promise<Session> {
  return new CpuBackend().compile(compilePlan(definition), snapshot, options);
}
