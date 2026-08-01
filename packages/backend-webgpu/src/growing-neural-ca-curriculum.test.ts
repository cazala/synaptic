import { describe, expect, it } from "vitest";
import {
  GrowingNeuralCaCurriculum,
  type GrowingNeuralCaCurriculumSnapshot,
} from "./growing-neural-ca-curriculum.js";
import type {
  GrowingNeuralCaBatchQuality,
  GrowingNeuralCaMetrics,
} from "./growing-neural-ca.js";

function target(): Float32Array {
  const value = new Float32Array(4 * 4 * 4);
  for (let cell = 0; cell < 8; cell += 1) {
    value[cell * 4] = 0.8;
    value[cell * 4 + 1] = 0.2;
    value[cell * 4 + 3] = 1;
  }
  return value;
}

function quality(
  meanLoss = 0.0001,
  meanAliveCells = 8,
  overrides: Partial<GrowingNeuralCaBatchQuality> = {},
): GrowingNeuralCaBatchQuality {
  return {
    samples: 4,
    meanLoss,
    maximumLoss: meanLoss,
    meanAliveCells,
    minimumAliveCells: meanAliveCells,
    maximumAliveCells: meanAliveCells,
    minimumTargetCoverage: 1,
    ...overrides,
  };
}

function metrics(
  seed: GrowingNeuralCaBatchQuality | undefined,
  persistent?: GrowingNeuralCaBatchQuality,
  damaged?: GrowingNeuralCaBatchQuality,
): GrowingNeuralCaMetrics {
  return {
    iteration: 1,
    loss: 0,
    rolloutSteps: 64,
    durationMs: 1,
    damagedSamples: damaged?.samples ?? 0,
    quality: { seed, persistent, damaged },
  };
}

function advance(
  curriculum: GrowingNeuralCaCurriculum,
  count: number,
  value: GrowingNeuralCaMetrics,
): GrowingNeuralCaCurriculumSnapshot {
  let snapshot = curriculum.snapshot();
  for (let index = 0; index < count; index += 1) {
    snapshot = curriculum.update(value);
  }
  return snapshot;
}

describe("Growing Neural CA adaptive curriculum", () => {
  it("does not promote a phase just because many updates elapsed", () => {
    const curriculum = new GrowingNeuralCaCurriculum(target());
    const snapshot = advance(
      curriculum,
      1_000,
      metrics(quality(1, 0)),
    );

    expect(snapshot).toMatchObject({
      key: "growth",
      progress: 0,
      mastered: false,
    });
    expect(curriculum.trainingOptions()).toEqual({
      learningRate: 0.002,
      damageProbability: 0,
      useSamplePool: false,
    });
  });

  it("promotes measured seed growth before enabling the sample pool", () => {
    const curriculum = new GrowingNeuralCaCurriculum(target());
    const before = advance(curriculum, 15, metrics(quality()));

    expect(before.key).toBe("growth");
    expect(before.progress).toBeGreaterThan(0.9);

    const after = curriculum.update(metrics(quality()));
    expect(after.key).toBe("stability");
    expect(after.phases[0]).toMatchObject({
      progress: 1,
      state: "complete",
      stateLabel: "complete",
    });
    expect(curriculum.trainingOptions()).toEqual({
      learningRate: 0.0005,
      damageProbability: 0,
      useSamplePool: true,
    });
  });

  it("requires retained and damaged states to preserve earlier skills", () => {
    const curriculum = new GrowingNeuralCaCurriculum(target());
    advance(curriculum, 16, metrics(quality()));
    const stable = advance(
      curriculum,
      32,
      metrics(quality(), quality()),
    );

    expect(stable.key).toBe("regeneration");
    expect(curriculum.trainingOptions()).toMatchObject({
      damageProbability: 0.5,
      useSamplePool: true,
    });

    const repairing = advance(
      curriculum,
      95,
      metrics(quality(), quality(), quality()),
    );
    expect(repairing.key).toBe("regeneration");
    expect(repairing.progress).toBeGreaterThan(0.9);
    expect(repairing.mastered).toBe(false);

    const mastered = curriculum.update(
      metrics(quality(), quality(), quality()),
    );
    expect(mastered).toMatchObject({
      key: "regeneration",
      progress: 1,
      mastered: true,
    });
    expect(mastered.phases[2]?.stateLabel).toBe("mastered");
    expect(curriculum.trainingOptions()).toEqual({
      learningRate: 0.0003,
      damageProbability: 1,
      useSamplePool: true,
    });

    const regressed = advance(
      curriculum,
      6,
      metrics(quality(1, 0), quality(1, 0), quality(1, 0)),
    );
    expect(regressed.mastered).toBe(false);
    expect(regressed.progress).toBeLessThan(0.92);
    expect(regressed.phases[2]?.stateLabel).toBe("active");
    expect(curriculum.trainingOptions().learningRate).toBe(0.0005);
  });

  it("does not let good averages hide unreliable target repair", () => {
    const curriculum = new GrowingNeuralCaCurriculum(target());
    advance(curriculum, 16, metrics(quality()));
    advance(curriculum, 32, metrics(quality(), quality()));

    const almostCovered = quality(0.0001, 8, {
      // The mean looks excellent, but at least one repaired sample misses
      // enough target cells to fail the strict regeneration criterion.
      minimumTargetCoverage: 0.895,
    });
    const unreliable = advance(
      curriculum,
      96,
      metrics(quality(), quality(), almostCovered),
    );

    expect(unreliable.key).toBe("regeneration");
    expect(unreliable.progress).toBeGreaterThan(0.97);
    expect(unreliable.mastered).toBe(false);

    const reliable = advance(
      curriculum,
      96,
      metrics(quality(), quality(), quality()),
    );
    expect(reliable.mastered).toBe(true);
  });

  it("rejects empty targets that could falsely satisfy image loss", () => {
    expect(
      () => new GrowingNeuralCaCurriculum(new Float32Array(16)),
    ).toThrow(/living, visible cells/);
  });
});
