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
