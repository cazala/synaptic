import { describe, expect, it, vi } from "vitest";
import {
  runForwardSequence,
  runTrainingSequence,
  type Metrics,
  type TensorLike,
  type TrainingBatch,
  type TrainStepOptions,
} from "./index.js";

describe("portable sequence execution", () => {
  it("forwards inputs in order", async () => {
    const seen: TensorLike[] = [];
    const forward = vi.fn(async (input: TensorLike) => {
      seen.push(input);
      const data = Float32Array.from(input);
      return { data, shape: [data.length] };
    });
    const inputs = [[1], [2], [3]] as const;

    const outputs = await runForwardSequence({ forward }, inputs);

    expect(seen).toEqual(inputs);
    expect(outputs.map((output) => [...output.data])).toEqual([[1], [2], [3]]);
  });

  it("trains in order with per-step learning-rate overrides", async () => {
    let step = 0;
    const calls: Array<{ batch: TrainingBatch; options?: TrainStepOptions }> = [];
    const trainStep = vi.fn(
      async (
        batch: TrainingBatch,
        options?: TrainStepOptions,
      ): Promise<Metrics> => {
        calls.push({ batch, ...(options === undefined ? {} : { options }) });
        step += 1;
        return { loss: step / 10, step };
      },
    );
    const sequence = [
      { input: [0], target: [1] },
      { input: [1], target: [0], learningRate: 0.25 },
    ] as const;

    const metrics = await runTrainingSequence(
      { trainStep },
      sequence,
      { learningRate: 0.5 },
    );

    expect(calls.map((call) => call.options?.learningRate)).toEqual([0.5, 0.25]);
    expect(metrics).toEqual({ loss: 0.2, step: 2, steps: 2 });
  });

  it("can suppress loss metrics and rejects empty training sequences", async () => {
    const trainStep = vi.fn(async (): Promise<Metrics> => ({ loss: 1, step: 7 }));

    await expect(
      runTrainingSequence(
        { trainStep },
        [{ input: [0], target: [0] }],
        { metrics: "none" },
      ),
    ).resolves.toEqual({ step: 7, steps: 1 });
    await expect(runTrainingSequence({ trainStep }, [])).rejects.toMatchObject({
      code: "EMPTY_TRAINING_SEQUENCE",
    });
  });
});
