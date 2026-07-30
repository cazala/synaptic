import { describe, expect, it } from "vitest";
import { dense, input, sequential } from "@synaptic/layers";
import {
  GROWING_NEURAL_CA_FORMAT,
  GrowingNeuralCaCurriculum,
  Model,
  compileModel,
  compilePlan,
  createGrowingNeuralCaWeights,
  selectBackend,
} from "./index.js";

describe("synaptic facade", () => {
  const definition = sequential(
    input({ size: 2 }),
    dense({ units: 1, activation: "logistic" }),
  );

  it("selects the low-overhead CPU path for small auto workloads", () => {
    expect(selectBackend(compilePlan(definition))).toMatchObject({
      backend: "cpu",
    });
  });

  it("compiles explicit and auto backends through one async API", async () => {
    const explicit = await compileModel(definition, {
      backend: "paper",
      seed: 5,
    });
    const automatic = await new Model(definition).compile({ seed: 5 });
    expect(explicit.backend).toBe("paper");
    expect(automatic.backend).toBe("cpu");
    expect((await explicit.forward([1, 0])).data).toHaveLength(1);
    expect((await automatic.forward([1, 0])).data).toHaveLength(1);
    explicit.dispose();
    automatic.dispose();
  });

  it("re-exports the portable Growing Neural CA authoring boundary", () => {
    expect(GROWING_NEURAL_CA_FORMAT).toBe(
      "@cazala/automata/growing-neural-ca",
    );
    expect(
      createGrowingNeuralCaWeights({ channels: 4, hidden: 2 }),
    ).toHaveLength(2 * 12 + 2 + 4 * 2 + 4);
    expect(
      new GrowingNeuralCaCurriculum(
        new Float32Array([0, 0, 0, 1]),
      ).snapshot().key,
    ).toBe("growth");
  });
});
