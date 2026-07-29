import { describe, expect, it } from "vitest";
import { CpuBackend } from "@synaptic/backend-cpu";
import { PaperBackend } from "@synaptic/backend-paper";
import { WasmBackend, WasmSession } from "@synaptic/backend-wasm";
import { compilePlan } from "@synaptic/core";
import {
  feedForwardFixture,
  recurrentFixture,
  runConformance,
  sharedParameterFixture,
  type ConformanceResult,
} from "./index.js";

function expectArrayClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  precision = 5,
): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? 0, precision);
  }
}

function expectResultClose(actual: ConformanceResult, expected: ConformanceResult): void {
  expect(actual.outputs).toHaveLength(expected.outputs.length);
  actual.outputs.forEach((output, index) => {
    expectArrayClose(output, expected.outputs[index] ?? [], 5);
  });
  expectArrayClose(actual.losses, expected.losses, 5);
  expectArrayClose(actual.checkpoint.parameters, expected.checkpoint.parameters, 5);
  expectArrayClose(actual.checkpoint.runtime.state, expected.checkpoint.runtime.state, 5);
  expectArrayClose(actual.checkpoint.runtime.eligibilityTrace, expected.checkpoint.runtime.eligibilityTrace, 5);
  expectArrayClose(
    actual.checkpoint.runtime.extendedEligibilityTrace,
    expected.checkpoint.runtime.extendedEligibilityTrace,
    5,
  );
  expectArrayClose(actual.checkpoint.runtime.error, expected.checkpoint.runtime.error, 5);
}

describe("Paper/CPU conformance", () => {
  it("matches single-step feed-forward inference and training", async () => {
    const fixture = feedForwardFixture();
    const [paper, cpu] = await Promise.all([
      runConformance(new PaperBackend(), fixture, { train: true, learningRate: 0.08 }),
      runConformance(new CpuBackend(), fixture, { train: true, learningRate: 0.08 }),
    ]);
    expectResultClose(cpu, paper);
  });

  it("matches recurrent state, gating, traces, and updates over a sequence", async () => {
    const fixture = recurrentFixture();
    const [paper, cpu] = await Promise.all([
      runConformance(new PaperBackend(), fixture, { train: true, learningRate: 0.04 }),
      runConformance(new CpuBackend(), fixture, { train: true, learningRate: 0.04 }),
    ]);
    expectResultClose(cpu, paper);
  });

  it("matches recurrent inference without mutating parameters", async () => {
    const fixture = recurrentFixture();
    const [paper, cpu] = await Promise.all([
      runConformance(new PaperBackend(), fixture),
      runConformance(new CpuBackend(), fixture),
    ]);
    expectResultClose(cpu, paper);
    expectArrayClose(cpu.checkpoint.parameters, fixture.snapshot.parameters, 7);
  });
});

describe("Wasm conformance", () => {
  it("matches CPU recurrent inference and training state", async () => {
    const fixture = recurrentFixture();
    const [wasm, cpu] = await Promise.all([
      runConformance(new WasmBackend(), fixture, { train: true, learningRate: 0.04 }),
      runConformance(new CpuBackend(), fixture, { train: true, learningRate: 0.04 }),
    ]);
    expectResultClose(wasm, cpu);
  });

  it("selects a compiled variant and releases disposed sessions", async () => {
    const fixture = feedForwardFixture();
    const session = await new WasmBackend().compile(
      compilePlan(fixture.definition),
      fixture.snapshot,
    );
    expect(session).toBeInstanceOf(WasmSession);
    expect((session as WasmSession).variant).toMatch(/^(scalar|simd)$/);
    session.dispose();
    await expect(session.forward([0, 0])).rejects.toThrow("disposed");
  });
});

describe("shared parameter conformance", () => {
  it("reduces every tied connection into one deterministic update", async () => {
    const fixture = sharedParameterFixture();
    const results = await Promise.all([
      runConformance(new PaperBackend(), fixture, { train: true, learningRate: 0.1 }),
      runConformance(new CpuBackend(), fixture, { train: true, learningRate: 0.1 }),
      runConformance(new WasmBackend(), fixture, { train: true, learningRate: 0.1 }),
    ]);
    expectResultClose(results[1]!, results[0]!);
    expectResultClose(results[2]!, results[0]!);
    expect(results[0]?.checkpoint.parameters).toHaveLength(1);
    expect(results[0]?.checkpoint.parameters[0]).toBeCloseTo(-1, 6);
  });
});

describe("portable checkpoint restoration", () => {
  it("resumes recurrent execution identically across production and oracle backends", async () => {
    const fixture = recurrentFixture();
    const plan = compilePlan(fixture.definition);
    const source = await new CpuBackend().compile(
      plan,
      fixture.snapshot,
      { training: true },
    );
    await source.trainStep(
      fixture.sequence[0]!,
      { learningRate: 0.04 },
    );
    await source.trainStep(
      fixture.sequence[1]!,
      { learningRate: 0.04 },
    );
    const checkpoint = {
      ...await source.checkpoint(),
      optimizer: {
        momentum: new Float32Array(plan.parameterCount).fill(0.125),
      },
    };
    source.dispose();

    const sessions = await Promise.all([
      new PaperBackend().compile(plan, checkpoint, { training: true }),
      new CpuBackend().compile(plan, checkpoint, { training: true }),
      new WasmBackend().compile(plan, checkpoint, { training: true }),
    ]);
    const outputs = await Promise.all(
      sessions.map((session) => session.forward(fixture.sequence[2]!.input)),
    );
    expectArrayClose(outputs[1]!.data, outputs[0]!.data, 5);
    expectArrayClose(outputs[2]!.data, outputs[0]!.data, 5);

    const restored = await Promise.all(
      sessions.map((session) => session.checkpoint()),
    );
    expectResultClose(
      { backend: "cpu", outputs: [], losses: [], checkpoint: restored[1]! },
      { backend: "paper", outputs: [], losses: [], checkpoint: restored[0]! },
    );
    expectResultClose(
      { backend: "wasm", outputs: [], losses: [], checkpoint: restored[2]! },
      { backend: "paper", outputs: [], losses: [], checkpoint: restored[0]! },
    );
    expect(restored[0]!.runtime.step).toBe(checkpoint.runtime.step + 1);
    restored.forEach((value) => {
      expect(value.optimizer?.momentum).toEqual(checkpoint.optimizer.momentum);
      expect(value.optimizer?.momentum).not.toBe(checkpoint.optimizer.momentum);
    });
    sessions.forEach((session) => session.dispose());
  });

  it("restores an existing session after reset", async () => {
    const fixture = recurrentFixture();
    const session = await new CpuBackend().compile(
      compilePlan(fixture.definition),
      fixture.snapshot,
    );
    await session.forward(fixture.sequence[0]!.input);
    const checkpoint = await session.checkpoint();
    const expected = await session.forward(fixture.sequence[1]!.input);
    await session.resetState();
    await session.restore(checkpoint);
    const actual = await session.forward(fixture.sequence[1]!.input);
    expectArrayClose(actual.data, expected.data, 7);
    session.dispose();
  });
});
