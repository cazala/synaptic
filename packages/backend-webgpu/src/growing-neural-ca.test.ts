import { describe, expect, it } from "vitest";
import {
  GROWING_NEURAL_CA_FORMAT,
  GROWING_NEURAL_CA_VERSION,
  buildGrowingNeuralCaHeapLayout,
  createGrowingNeuralCaWeights,
} from "./growing-neural-ca.js";

describe("Growing Neural CA trainer substrate", () => {
  it("uses the Automata artifact identity and reference initialization", () => {
    const channels = 16;
    const hidden = 32;
    const perception = channels * 3;
    const weights = createGrowingNeuralCaWeights({
      channels,
      hidden,
      seed: 0x6e63_6102,
    });
    const hiddenBias = hidden * perception;
    const hiddenToOutput = hiddenBias + hidden;

    expect(GROWING_NEURAL_CA_FORMAT).toBe(
      "@cazala/automata/growing-neural-ca",
    );
    expect(GROWING_NEURAL_CA_VERSION).toBe(1);
    expect(weights).toHaveLength(
      hidden * perception + hidden + channels * hidden + channels,
    );
    expect(
      weights.slice(0, hiddenBias).some((value) => value !== 0),
    ).toBe(true);
    expect(
      weights.slice(hiddenBias, hiddenToOutput).every((value) => value === 0),
    ).toBe(true);
    expect(
      weights.slice(hiddenToOutput).every((value) => value === 0),
    ).toBe(true);
  });

  it("allocates a non-overlapping tape and backward scratch heap", () => {
    const layout = buildGrowingNeuralCaHeapLayout({
      size: 24,
      channels: 16,
      hidden: 32,
      batchSize: 2,
      rolloutSteps: 24,
    });
    const sections = Object.values(layout.sections);

    expect(layout.byteLength).toBeLessThan(8 * 1024 * 1024);
    expect(layout.sections.stateTape.length).toBe(
      25 * 2 * 24 * 24 * 16,
    );
    expect(layout.sections.perception.length).toBe(
      2 * 2 * 24 * 24 * 16 * 3,
    );
    for (let index = 1; index < sections.length; index += 1) {
      const previous = sections[index - 1];
      const current = sections[index];
      expect(previous).toBeDefined();
      expect(current).toBeDefined();
      expect(current!.byteOffset).toBeGreaterThanOrEqual(
        previous!.byteOffset + previous!.byteLength,
      );
    }
    const last = sections.at(-1);
    expect(last).toBeDefined();
    expect(last!.byteOffset + last!.byteLength).toBe(layout.byteLength);
  });

  it("rejects dimensions outside the portable shader limits", () => {
    expect(() =>
      buildGrowingNeuralCaHeapLayout({
        size: 7,
        channels: 16,
        hidden: 32,
        batchSize: 2,
        rolloutSteps: 24,
      }),
    ).toThrow(/size must be an integer between 8 and 64/);
    expect(() =>
      createGrowingNeuralCaWeights({ channels: 3 }),
    ).toThrow(/channels must be an integer between 4 and 32/);
  });
});
