import { describe, expect, it } from "vitest";
import { GraphBuilder } from "./graph-builder.js";
import { compilePlan } from "./plan.js";

describe("GraphBuilder", () => {
  it("freezes an explicitly staged recurrent graph into a compact plan", () => {
    const graph = new GraphBuilder();
    const input = graph.input(2);
    graph.stage(1);
    const hidden = graph.units(2, { activation: "tanh", label: "hidden" });
    graph.stage(2);
    const output = graph.units(1, { activation: "logistic", label: "output" });

    graph.connect(input.units, hidden);
    graph.connect(hidden, hidden, "one-to-one", { delay: 1 });
    const projected = graph.connect(hidden, output);
    graph.gate([hidden[0]!], projected, { mode: "all" });

    const definition = graph.build({ inputs: input, outputs: output });
    const plan = compilePlan(definition);

    expect(plan.unitCount).toBe(5);
    expect(plan.stageOffsets).toEqual(new Uint32Array([0, 2, 4, 5]));
    expect(plan.connectionCount).toBe(8);
    expect(plan.connectionGater.filter((value) => value >= 0)).toHaveLength(2);
    expect(plan.extendedTraceTarget.length).toBeGreaterThan(0);
  });

  it("rejects a zero-delay connection within one parallel stage", () => {
    const graph = new GraphBuilder();
    const input = graph.input(1);
    const peer = graph.units(1, { activation: "tanh", stage: 0 });
    graph.connect(input.units, peer, "one-to-one", { delay: 0 });

    expect(() => graph.build({ inputs: input, outputs: peer })).toThrow(
      "Zero-delay connections must move to a later stage",
    );
  });

  it("cannot be mutated after build", () => {
    const graph = new GraphBuilder();
    const input = graph.input(1);
    graph.stage(1);
    const output = graph.units(1);
    graph.connect(input.units, output);
    graph.build({ inputs: input, outputs: output });
    expect(() => graph.units(1)).toThrow("cannot change");
  });
});
