import { describe, expect, it } from "vitest";
import { CpuBackend } from "@synaptic/backend-cpu";
import { PaperBackend } from "@synaptic/backend-paper";
import { WasmBackend } from "@synaptic/backend-wasm";
import type { Backend } from "@synaptic/core";
// The deliberately small legacy dataset package does not publish TypeScript types.
// @ts-expect-error -- mnist@1.1.0 is CommonJS without a declaration file.
import mnist from "mnist";
import {
  createMnistDataset,
  runDsrLearning,
  runMnistLearning,
  runXorLearning,
  type MnistSource,
} from "./learning.js";

const dataset = createMnistDataset(mnist as MnistSource);

const backends: readonly [
  name: string,
  create: () => Backend,
][] = [
  ["Paper", () => new PaperBackend()],
  ["CPU", () => new CpuBackend()],
  ["Wasm", () => new WasmBackend()],
];

describe.each(backends)("%s backend learning", (_name, createBackend) => {
  it("solves XOR from a seeded initialization", async () => {
    const backend = createBackend();
    const result = await runXorLearning(backend);

    expect(result.backend).toBe(backend.id);
    expect(result.accuracy).toBe(1);
    expect(result.outputs[0]).toBeLessThan(0.1);
    expect(result.outputs[1]).toBeGreaterThan(0.9);
    expect(result.outputs[2]).toBeGreaterThan(0.9);
    expect(result.outputs[3]).toBeLessThan(0.1);
  }, 30_000);

  it("learns all ten MNIST digit classes and generalizes to held-out samples", async () => {
    const backend = createBackend();
    const result = await runMnistLearning(backend, dataset);

    expect(result.backend).toBe(backend.id);
    expect(result.trainingAccuracy).toBeGreaterThanOrEqual(0.85);
    expect(result.testAccuracy).toBeGreaterThanOrEqual(0.8);
  }, 30_000);

  it("passes the discrete sequence recall task", async () => {
    const backend = createBackend();
    const result = await runDsrLearning(backend);

    expect(result.backend).toBe(backend.id);
    expect(result.distractorAccuracy).toBe(1);
    expect(result.promptAccuracy).toBeGreaterThanOrEqual(0.95);
    expect(result.sequenceAccuracy).toBeGreaterThanOrEqual(0.95);
  }, 60_000);
});
