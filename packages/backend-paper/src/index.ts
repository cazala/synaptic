import {
  SynapticError,
  activate,
  compilePlan,
  createCheckpoint,
  createRuntimeState,
  derivative,
  invariant,
  readTensor,
  validateSnapshot,
  type Backend,
  type CompileOptions,
  type ExecutionPlan,
  type Metrics,
  type ModelCheckpoint,
  type ModelSnapshot,
  type MutableRuntimeState,
  type NumericArray,
  type Session,
  type SupportReport,
  type Tensor,
  type TensorLike,
  type TrainingBatch,
  type TrainStepOptions,
} from "@synaptic/core";

const DEFAULT_LEARNING_RATE = 0.1;

/**
 * Equation-aligned implementation of Monner and Reggia's LSTM-g algorithm.
 *
 * This deliberately traverses the semantic topology rather than the compiled
 * adjacency arrays. It is the readable correctness oracle for optimized
 * backends, not their performance baseline.
 */
export class PaperBackend implements Backend {
  readonly id = "paper";

  inspect(plan: ExecutionPlan, options: CompileOptions = {}): SupportReport {
    const issues = [];
    if (options.training) {
      for (const connection of plan.definition.topology.connections) {
        if (
          connection.from === connection.to
          && plan.definition.topology.parameters[connection.parameter]?.trainable
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
      throw new SynapticError("The Paper backend cannot compile this plan", "UNSUPPORTED_PLAN", {
        issues: report.issues,
      });
    }
    return new PaperSession(plan, snapshot);
  }
}

export class PaperSession implements Session {
  readonly backend = "paper";
  readonly #plan: ExecutionPlan;
  readonly #snapshotDefinition: ModelSnapshot["definition"];
  readonly #parameters: NumericArray;
  readonly #runtime: MutableRuntimeState;
  readonly #inputSlot: Int32Array;
  readonly #outputSlot: Int32Array;
  readonly #gatedTargets: readonly number[][];
  #disposed = false;

  constructor(plan: ExecutionPlan, snapshot: ModelSnapshot) {
    this.#plan = plan;
    this.#snapshotDefinition = snapshot.definition;
    this.#parameters = snapshot.parameters.slice();
    this.#runtime = createRuntimeState(plan, snapshot.definition.precision);
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

    const targets = Array.from({ length: plan.unitCount }, () => new Set<number>());
    for (const gate of plan.definition.topology.gates) {
      if (gate.delay === 0) {
        const connection = plan.definition.topology.connections[gate.connection];
        if (connection) {
          targets[gate.gater]?.add(connection.to);
        }
      }
    }
    this.#gatedTargets = targets.map((set) => [...set].sort((left, right) => left - right));
  }

  async forward(input: TensorLike): Promise<Tensor> {
    this.#assertActive();
    const values = readTensor(input, this.#plan.inputs.length, "Input");
    const topology = this.#plan.definition.topology;
    const runtime = this.#runtime;
    runtime.previousActivation.set(runtime.activation);

    const stageCount = this.#plan.stageOffsets.length - 1;
    for (let stage = 0; stage < stageCount; stage += 1) {
      for (const unit of topology.units) {
        if (unit.stage !== stage) {
          continue;
        }
        const inputSlot = this.#inputSlot[unit.id] ?? -1;
        if (inputSlot >= 0) {
          const value = values[inputSlot] ?? 0;
          runtime.state[unit.id] = value;
          runtime.activation[unit.id] = value;
          runtime.derivative[unit.id] = 0;
        } else if (unit.constant !== undefined) {
          runtime.state[unit.id] = unit.constant;
          runtime.activation[unit.id] = unit.constant;
          runtime.derivative[unit.id] = 0;
        } else {
          this.#activateUnit(unit.id);
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
      loss += difference * difference;
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
    return {
      definition: this.#snapshotDefinition,
      parameters: this.#parameters.slice(),
    };
  }

  async checkpoint(): Promise<ModelCheckpoint> {
    this.#assertActive();
    return createCheckpoint(
      {
        definition: this.#snapshotDefinition,
        parameters: this.#parameters,
      },
      this.#runtime,
    );
  }

  dispose(): void {
    this.#disposed = true;
  }

  #activateUnit(j: number): void {
    const topology = this.#plan.definition.topology;
    const unit = topology.units[j];
    invariant(unit, "Execution plan references an unknown unit", "INVALID_PLAN");
    const runtime = this.#runtime;
    const previousState = runtime.state[j] ?? 0;
    let recurrence = 0;
    let state = 0;

    for (const connection of topology.connections) {
      if (connection.to !== j) {
        continue;
      }
      const gain = this.#gain(connection.id);
      const weight = this.#parameters[connection.parameter] ?? 0;
      if (connection.from === j) {
        recurrence += gain * weight;
        state += gain * weight * previousState;
      } else {
        state += gain * weight * this.#sourceValue(connection.id);
      }
    }

    runtime.state[j] = state;
    const activation = activate(unit.activation, state);
    runtime.activation[j] = activation;
    runtime.derivative[j] = derivative(unit.activation, state, activation);

    // Eq. 17 and 18: regular and extended eligibility traces.
    for (const connection of topology.connections) {
      if (connection.to !== j || connection.from === j) {
        continue;
      }
      const gain = this.#gain(connection.id);
      const eligibility = recurrence * (runtime.eligibilityTrace[connection.id] ?? 0)
        + gain * this.#sourceValue(connection.id);
      runtime.eligibilityTrace[connection.id] = eligibility;

      const start = this.#plan.extendedTraceOffsets[connection.id] ?? 0;
      const end = this.#plan.extendedTraceOffsets[connection.id + 1] ?? start;
      for (let trace = start; trace < end; trace += 1) {
        const target = this.#plan.extendedTraceTarget[trace];
        invariant(target !== undefined, "Extended trace has no target", "INVALID_PLAN");
        runtime.extendedEligibilityTrace[trace] =
          this.#selfFactor(target) * (runtime.extendedEligibilityTrace[trace] ?? 0)
          + (runtime.derivative[j] ?? 0) * eligibility * this.#bigParenthesis(target, j);
      }
    }
  }

  #propagate(target: Float32Array, learningRate: number): void {
    const topology = this.#plan.definition.topology;
    const runtime = this.#runtime;
    runtime.projectedError.fill(0);
    runtime.gatedError.fill(0);
    runtime.error.fill(0);
    const gradients = new Float64Array(this.#plan.parameterCount);
    const stageCount = this.#plan.stageOffsets.length - 1;

    for (let stage = stageCount - 1; stage > 0; stage -= 1) {
      for (let unitIndex = topology.units.length - 1; unitIndex >= 0; unitIndex -= 1) {
        const unit = topology.units[unitIndex];
        if (!unit || unit.stage !== stage) {
          continue;
        }
        const j = unit.id;
        const outputSlot = this.#outputSlot[j] ?? -1;
        if (outputSlot >= 0) {
          runtime.projectedError[j] = (target[outputSlot] ?? 0) - (runtime.activation[j] ?? 0);
          runtime.gatedError[j] = 0;
        } else {
          let projected = 0;
          for (const connection of topology.connections) {
            if (connection.from === j && connection.delay === 0) {
              projected += (runtime.error[connection.to] ?? 0)
                * this.#gain(connection.id)
                * (this.#parameters[connection.parameter] ?? 0);
            }
          }
          runtime.projectedError[j] = (runtime.derivative[j] ?? 0) * projected;

          let gated = 0;
          for (const targetUnit of this.#gatedTargets[j] ?? []) {
            gated += (runtime.error[targetUnit] ?? 0) * this.#bigParenthesis(targetUnit, j);
          }
          runtime.gatedError[j] = (runtime.derivative[j] ?? 0) * gated;
        }
        runtime.error[j] = (runtime.projectedError[j] ?? 0) + (runtime.gatedError[j] ?? 0);

        for (const connection of topology.connections) {
          if (connection.to !== j || connection.from === j) {
            continue;
          }
          let gradient = (runtime.projectedError[j] ?? 0)
            * (runtime.eligibilityTrace[connection.id] ?? 0);
          const start = this.#plan.extendedTraceOffsets[connection.id] ?? 0;
          const end = this.#plan.extendedTraceOffsets[connection.id + 1] ?? start;
          for (let trace = start; trace < end; trace += 1) {
            const targetUnit = this.#plan.extendedTraceTarget[trace] ?? 0;
            gradient += (runtime.error[targetUnit] ?? 0)
              * (runtime.extendedEligibilityTrace[trace] ?? 0);
          }
          gradients[connection.parameter] = (gradients[connection.parameter] ?? 0) + gradient;
        }
      }
    }

    for (const parameter of topology.parameters) {
      if (parameter.trainable) {
        this.#parameters[parameter.id] =
          (this.#parameters[parameter.id] ?? 0) + learningRate * (gradients[parameter.id] ?? 0);
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
    let factor = 0;
    const topology = this.#plan.definition.topology;
    for (const connection of topology.connections) {
      if (connection.from === unit && connection.to === unit) {
        factor += this.#gain(connection.id) * (this.#parameters[connection.parameter] ?? 0);
      }
    }
    return factor;
  }

  #bigParenthesis(target: number, gater: number): number {
    let result = 0;
    const topology = this.#plan.definition.topology;
    for (const connection of topology.connections) {
      if (
        connection.to !== target
        || this.#plan.connectionGater[connection.id] !== gater
        || this.#plan.gateDelay[connection.id] !== 0
      ) {
        continue;
      }
      const weight = this.#parameters[connection.parameter] ?? 0;
      result += connection.from === target
        ? weight * (this.#runtime.state[target] ?? 0)
        : weight * this.#sourceValue(connection.id);
    }
    return result;
  }

  #assertActive(): void {
    invariant(!this.#disposed, "The session has been disposed", "SESSION_DISPOSED");
  }
}

export function compilePaper(
  definition: ModelSnapshot["definition"],
  snapshot: ModelSnapshot,
  options?: CompileOptions,
): Promise<Session> {
  return new PaperBackend().compile(compilePlan(definition), snapshot, options);
}
