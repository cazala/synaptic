import type {
  GrowingNeuralCaBatchQuality,
  GrowingNeuralCaMetrics,
} from "./growing-neural-ca.js";

export type GrowingNeuralCaCurriculumPhaseKey =
  | "growth"
  | "stability"
  | "regeneration";

export interface GrowingNeuralCaCurriculumPhase {
  readonly index: number;
  readonly key: GrowingNeuralCaCurriculumPhaseKey;
  readonly label: string;
  readonly detail: string;
  readonly criterion: string;
  readonly progress: number;
  readonly state: "complete" | "active" | "pending";
  readonly stateLabel: "complete" | "active" | "waiting" | "mastered";
}

export interface GrowingNeuralCaCurriculumSnapshot {
  readonly index: number;
  readonly key: GrowingNeuralCaCurriculumPhaseKey;
  readonly label: string;
  readonly detail: string;
  readonly progress: number;
  readonly mastered: boolean;
  readonly phases: readonly GrowingNeuralCaCurriculumPhase[];
}

export interface GrowingNeuralCaCurriculumTrainingOptions {
  readonly learningRate: number;
  readonly damageProbability: number;
  readonly useSamplePool: boolean;
}

interface PhaseDefinition {
  readonly key: GrowingNeuralCaCurriculumPhaseKey;
  readonly label: string;
  readonly detail: string;
  readonly criterion: string;
  readonly relativeLossTarget: number;
  readonly minimumAliveRatio: number;
  readonly maximumAliveRatio: number;
  readonly observations: number;
}

interface Tracker {
  readonly scores: number[];
  readonly observations: number;
}

const PROMOTION_CONFIDENCE = 0.92;
const PHASES: readonly PhaseDefinition[] = Object.freeze([
  Object.freeze({
    key: "growth",
    label: "growth",
    detail: "Testing fresh seed rollouts for shape and living-cell coverage.",
    criterion: "seed → target · repeated quality checks",
    relativeLossTarget: 0.08,
    minimumAliveRatio: 0.6,
    maximumAliveRatio: 1.6,
    observations: 16,
  }),
  Object.freeze({
    key: "stability",
    label: "stability",
    detail: "Testing retained states after another randomized rollout.",
    criterion: "retained states · repeated quality checks",
    relativeLossTarget: 0.06,
    minimumAliveRatio: 0.7,
    maximumAliveRatio: 1.45,
    observations: 32,
  }),
  Object.freeze({
    key: "regeneration",
    label: "regeneration",
    detail: "Testing damaged states for recovered shape and coverage.",
    criterion: "damaged states · repeated quality checks",
    relativeLossTarget: 0.08,
    minimumAliveRatio: 0.6,
    maximumAliveRatio: 1.6,
    observations: 48,
  }),
]);

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export class GrowingNeuralCaCurriculum {
  readonly #targetEnergy: number;
  readonly #targetAliveCells: number;
  readonly #trackers: readonly Tracker[];
  #phaseIndex = 0;
  #regenerationMastered = false;

  constructor(target: ArrayLike<number>) {
    if (target.length === 0 || target.length % 4 !== 0) {
      throw new RangeError("Growing Neural CA target must contain RGBA cells");
    }
    let energy = 0;
    let aliveCells = 0;
    for (let index = 0; index < target.length; index += 1) {
      const value = target[index];
      if (value === undefined || !Number.isFinite(value)) {
        throw new RangeError("Growing Neural CA target must be finite");
      }
      energy += value * value;
      if (index % 4 === 3 && value > 0.1) {
        aliveCells += 1;
      }
    }
    if (energy === 0 || aliveCells === 0) {
      throw new RangeError(
        "Growing Neural CA target must contain living, visible cells",
      );
    }
    this.#targetEnergy = energy / target.length;
    this.#targetAliveCells = aliveCells;
    this.#trackers = PHASES.map(({ observations }) => ({
      scores: [],
      observations,
    }));
  }

  update(metrics: GrowingNeuralCaMetrics): GrowingNeuralCaCurriculumSnapshot {
    this.#observe(0, metrics.quality.seed);
    this.#observe(1, metrics.quality.persistent);
    this.#observe(2, metrics.quality.damaged);

    if (this.#phaseIndex === PHASES.length - 1) {
      this.#regenerationMastered = this.#phaseReady(this.#phaseIndex);
    } else if (this.#phaseReady(this.#phaseIndex)) {
      this.#phaseIndex += 1;
    }
    return this.snapshot();
  }

  snapshot(): GrowingNeuralCaCurriculumSnapshot {
    const definition = PHASES[this.#phaseIndex]!;
    const progress = this.#regenerationMastered
      ? 1
      : this.#phaseProgress(this.#phaseIndex);
    const phases = PHASES.map((phase, index) => {
      const complete =
        index < this.#phaseIndex ||
        (index === 2 && this.#regenerationMastered);
      return Object.freeze({
        index,
        key: phase.key,
        label: phase.label,
        detail: phase.detail,
        criterion: phase.criterion,
        progress: complete
          ? 1
          : index === this.#phaseIndex
          ? progress
          : 0,
        state: complete
          ? "complete"
          : index === this.#phaseIndex
          ? "active"
          : "pending",
        stateLabel: index === 2 && this.#regenerationMastered
          ? "mastered"
          : complete
          ? "complete"
          : index === this.#phaseIndex
          ? "active"
          : "waiting",
      } satisfies GrowingNeuralCaCurriculumPhase);
    });
    return Object.freeze({
      index: this.#phaseIndex,
      key: definition.key,
      label: definition.label,
      detail: this.#regenerationMastered
        ? "Regeneration learned; continuing maintenance training."
        : definition.detail,
      progress,
      mastered: this.#regenerationMastered,
      phases: Object.freeze(phases),
    });
  }

  trainingOptions(): GrowingNeuralCaCurriculumTrainingOptions {
    if (this.#phaseIndex === 0) {
      return Object.freeze({
        learningRate: 0.002,
        damageProbability: 0,
        useSamplePool: false,
      });
    }
    if (this.#phaseIndex === 1) {
      return Object.freeze({
        learningRate: 0.0005,
        damageProbability: 0,
        useSamplePool: true,
      });
    }
    const progress = this.#regenerationMastered
      ? 1
      : this.#phaseProgress(2);
    return Object.freeze({
      learningRate: this.#regenerationMastered ? 0.0003 : 0.0005,
      damageProbability: 0.25 + progress * 0.75,
      useSamplePool: true,
    });
  }

  #observe(
    phaseIndex: number,
    quality: GrowingNeuralCaBatchQuality | undefined,
  ): void {
    if (quality === undefined || quality.samples === 0) {
      return;
    }
    const definition = PHASES[phaseIndex]!;
    const relativeLoss = quality.meanLoss / this.#targetEnergy;
    const lossScore = relativeLoss === 0
      ? 1
      : clamp(definition.relativeLossTarget / relativeLoss);
    const aliveRatio = quality.meanAliveCells / this.#targetAliveCells;
    const aliveScore = aliveRatio < definition.minimumAliveRatio
      ? clamp(aliveRatio / definition.minimumAliveRatio)
      : aliveRatio > definition.maximumAliveRatio
      ? clamp(definition.maximumAliveRatio / aliveRatio)
      : 1;
    const tracker = this.#trackers[phaseIndex]!;
    tracker.scores.push(Math.min(lossScore, aliveScore));
    if (tracker.scores.length > tracker.observations) {
      tracker.scores.shift();
    }
  }

  #trackerConfidence(phaseIndex: number): number {
    const tracker = this.#trackers[phaseIndex]!;
    if (tracker.scores.length === 0) {
      return 0;
    }
    const average =
      tracker.scores.reduce((sum, score) => sum + score, 0) /
      tracker.scores.length;
    const evidence = Math.min(
      1,
      tracker.scores.length / tracker.observations,
    );
    return average * evidence;
  }

  #phaseProgress(phaseIndex: number): number {
    let confidence = 1;
    for (let index = 0; index <= phaseIndex; index += 1) {
      confidence = Math.min(confidence, this.#trackerConfidence(index));
    }
    return clamp(confidence);
  }

  #phaseReady(phaseIndex: number): boolean {
    for (let index = 0; index <= phaseIndex; index += 1) {
      const tracker = this.#trackers[index]!;
      if (
        tracker.scores.length < tracker.observations ||
        this.#trackerConfidence(index) < PROMOTION_CONFIDENCE
      ) {
        return false;
      }
    }
    return true;
  }
}
