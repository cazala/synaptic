import { afterAll, describe, expect, it } from "vitest";
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

interface WorkloadTiming {
  readonly backend: string;
  readonly durationMs: number;
  readonly result: string;
  readonly workload: string;
}

const timings: WorkloadTiming[] = [];

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

async function timed<T>(
  backend: string,
  workload: string,
  run: () => Promise<T>,
  summarize: (result: T) => string,
): Promise<T> {
  const start = performance.now();
  const result = await run();
  timings.push({
    backend,
    workload,
    durationMs: Math.round(performance.now() - start),
    result: summarize(result),
  });
  return result;
}

describe("cross-backend learning workloads", () => {
  for (const [backendName, createBackend] of backends) {
    it(`${backendName} / XOR: solves all four inputs`, async () => {
      const backend = createBackend();
      const result = await timed(
        backendName,
        "XOR",
        () => runXorLearning(backend),
        (value) => `accuracy ${percent(value.accuracy)}`,
      );

      expect(result.backend).toBe(backend.id);
      expect(result.accuracy).toBe(1);
      expect(result.outputs[0]).toBeLessThan(0.1);
      expect(result.outputs[1]).toBeGreaterThan(0.9);
      expect(result.outputs[2]).toBeGreaterThan(0.9);
      expect(result.outputs[3]).toBeLessThan(0.1);
    }, 30_000);

    it(`${backendName} / MNIST: learns all ten digit classes`, async () => {
      const backend = createBackend();
      const result = await timed(
        backendName,
        "MNIST",
        () => runMnistLearning(backend, dataset),
        (value) =>
          `train ${percent(value.trainingAccuracy)}, test ${percent(value.testAccuracy)}`,
      );

      expect(result.backend).toBe(backend.id);
      expect(result.trainingAccuracy).toBeGreaterThanOrEqual(0.85);
      expect(result.testAccuracy).toBeGreaterThanOrEqual(0.8);
    }, 30_000);

    it(`${backendName} / DSR: recalls both targets after distractors`, async () => {
      const backend = createBackend();
      const result = await timed(
        backendName,
        "DSR",
        () => runDsrLearning(backend),
        (value) =>
          `prompts ${percent(value.promptAccuracy)}, sequences ${percent(value.sequenceAccuracy)}`,
      );

      expect(result.backend).toBe(backend.id);
      expect(result.trainingTrials).toBe(9_625);
      expect(result.distractorAccuracy).toBe(1);
      expect(result.promptAccuracy).toBeGreaterThanOrEqual(0.95);
      expect(result.sequenceAccuracy).toBeGreaterThanOrEqual(0.95);
    }, 60_000);
  }

  afterAll(() => {
    console.log("\nCross-backend learning workload timings:");
    console.table(timings);
    console.log(
      "WebGPU runs separately, with fallback disabled, in "
      + "packages/backend-webgpu/test/learning-workloads.html.",
    );
  });
});
