import { describe, expect, it } from "vitest";
import {
  GraphBuilder,
  compilePlan,
  createSnapshot,
  type ModelDefinition,
} from "@synaptic/core";
import { PaperBackend } from "./index.js";

function identityDefinition(precision: "f32" | "f64" = "f32"): ModelDefinition {
  const graph = new GraphBuilder();
  const input = graph.input(1);
  graph.stage(1);
  const output = graph.units(1, { activation: "identity" });
  graph.connect(input.units, output, "one-to-one", {
    initializer: { kind: "constant", value: 0.5 },
  });
  return graph.build({ inputs: input, outputs: output, precision });
}

describe("PaperBackend equations", () => {
  it("matches a hand-calculated Eq. 17 and Eq. 24 update", async () => {
    const definition = identityDefinition();
    const session = await new PaperBackend().compile(
      compilePlan(definition),
      createSnapshot(definition),
      { training: true },
    );

    const metrics = await session.trainStep(
      { input: [2], target: [3] },
      { learningRate: 0.1 },
    );
    const checkpoint = await session.checkpoint();

    expect(metrics.loss).toBeCloseTo(4, 7);
    expect(checkpoint.runtime.eligibilityTrace[0]).toBeCloseTo(2, 7);
    expect(checkpoint.runtime.projectedError[1]).toBeCloseTo(2, 7);
    expect(checkpoint.parameters[0]).toBeCloseTo(0.9, 6);
    expect((await session.forward([2])).data[0]).toBeCloseTo(1.8, 6);
  });

  it("supports diagnostic f64 execution and enforces session disposal", async () => {
    const definition = identityDefinition("f64");
    const session = await new PaperBackend().compile(
      compilePlan(definition),
      createSnapshot(definition),
    );
    expect((await session.forward([0.25])).data[0]).toBeCloseTo(0.125, 7);
    session.dispose();
    await expect(session.forward([0.25])).rejects.toThrow("disposed");
  });
});
