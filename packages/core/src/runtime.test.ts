import { describe, expect, it } from "vitest";
import { GraphBuilder } from "./graph-builder.js";
import { createSnapshot } from "./model.js";
import { compilePlan } from "./plan.js";
import {
  activate,
  createCheckpoint,
  createNumericArray,
  createRuntimeState,
  derivative,
  readTensor,
  restoreRuntimeState,
  validateCheckpoint,
} from "./runtime.js";

function fixture() {
  const graph = new GraphBuilder();
  const input = graph.input(1);
  graph.nextStage();
  const output = graph.units(1, { activation: "identity" });
  graph.connect(input.units, output, "one-to-one", {
    initializer: { kind: "constant", value: 0.5 },
  });
  const definition = graph.build({ inputs: input, outputs: output });
  const plan = compilePlan(definition);
  const snapshot = createSnapshot(definition, 1);
  return { plan, snapshot };
}

describe("runtime numeric semantics", () => {
  it("implements every portable activation and derivative", () => {
    expect(activate("identity", -2)).toBe(-2);
    expect(derivative("identity", -2, -2)).toBe(1);
    expect(activate("logistic", 0)).toBe(0.5);
    expect(derivative("logistic", 0, 0.5)).toBe(0.25);
    expect(activate("tanh", 0.5)).toBeCloseTo(Math.tanh(0.5), 12);
    expect(derivative("tanh", 0.5, 0.5)).toBe(0.75);
    expect(activate("relu", -1)).toBe(0);
    expect(activate("relu", 2)).toBe(2);
    expect(derivative("relu", 0, 0)).toBe(0);
    expect(derivative("relu", 2, 2)).toBe(1);
    expect(activate("step", 0)).toBe(0);
    expect(activate("step", 0.01)).toBe(1);
    expect(derivative("step", 1, 1)).toBe(0);
  });

  it("creates storage matching the declared precision", () => {
    expect(createNumericArray("f32", 2)).toBeInstanceOf(Float32Array);
    expect(createNumericArray("f64", 2)).toBeInstanceOf(Float64Array);
  });

  it("normalizes array input and rejects malformed tensors", () => {
    expect(readTensor([1, 2], 2, "Input")).toEqual(Float32Array.of(1, 2));
    expect(() => readTensor([1], 2, "Input")).toThrow("length");
    expect(() => readTensor([Number.NaN], 1, "Input")).toThrow("non-finite");
    expect(() => readTensor([Number.POSITIVE_INFINITY], 1, "Input"))
      .toThrow("non-finite");
  });
});

describe("checkpoint validation and restoration", () => {
  it("copies every mutable runtime field and counter", () => {
    const { plan } = fixture();
    const source = createRuntimeState(plan, "f32", 123);
    source.state.fill(1);
    source.activation.fill(2);
    source.previousActivation.fill(3);
    source.derivative.fill(4);
    source.eligibilityTrace.fill(5);
    source.extendedEligibilityTrace.fill(6);
    source.projectedError.fill(7);
    source.gatedError.fill(8);
    source.error.fill(9);
    source.step = 10;
    source.randomCounter = 11;
    const target = createRuntimeState(plan, "f32");

    restoreRuntimeState(target, source);

    expect(target).toEqual(source);
    expect(target.state).not.toBe(source.state);
  });

  it("rejects runtime buffers with the wrong length", () => {
    const { plan, snapshot } = fixture();
    const checkpoint = createCheckpoint(
      snapshot,
      createRuntimeState(plan, "f32"),
    );
    const invalid = {
      ...checkpoint,
      runtime: {
        ...checkpoint.runtime,
        state: new Float32Array(),
      },
    };
    expect(() => validateCheckpoint(plan, invalid)).toThrow("wrong length");
  });

  it("rejects runtime buffers with the wrong precision", () => {
    const { plan, snapshot } = fixture();
    const checkpoint = createCheckpoint(
      snapshot,
      createRuntimeState(plan, "f32"),
    );
    const invalid = {
      ...checkpoint,
      runtime: {
        ...checkpoint.runtime,
        state: new Float64Array(checkpoint.runtime.state.length),
      },
    };
    expect(() => validateCheckpoint(plan, invalid)).toThrow("wrong precision");
  });

  it("rejects non-finite runtime values", () => {
    const { plan, snapshot } = fixture();
    const runtime = createRuntimeState(plan, "f32");
    runtime.state[0] = Number.NaN;
    expect(() => validateCheckpoint(
      plan,
      createCheckpoint(snapshot, runtime),
    )).toThrow("non-finite");
  });

  it("rejects invalid logical and PRNG counters", () => {
    const { plan, snapshot } = fixture();
    const checkpoint = createCheckpoint(
      snapshot,
      createRuntimeState(plan, "f32"),
    );
    expect(() => validateCheckpoint(plan, {
      ...checkpoint,
      runtime: { ...checkpoint.runtime, step: -1 },
    })).toThrow("step");
    expect(() => validateCheckpoint(plan, {
      ...checkpoint,
      runtime: { ...checkpoint.runtime, randomSeed: 0x1_0000_0000 },
    })).toThrow("seed");
    expect(() => validateCheckpoint(plan, {
      ...checkpoint,
      runtime: { ...checkpoint.runtime, randomCounter: 0.5 },
    })).toThrow("random counter");
  });
});
