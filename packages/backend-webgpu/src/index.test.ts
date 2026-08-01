import { describe, expect, it } from "vitest";
import {
  compilePlan,
  createSnapshot,
} from "@synaptic/core";
import { dense, input, sequential } from "@synaptic/layers";
import { WebGpuBackend, WebGpuSession } from "./index.js";

function fixture() {
  const definition = sequential(
    input({ size: 2 }),
    dense({ units: 2, activation: "tanh" }),
    dense({ units: 1 }),
  );
  return {
    definition,
    plan: compilePlan(definition),
    snapshot: createSnapshot(definition, 99),
  };
}

describe("WebGpuBackend", () => {
  it("falls back deterministically when WebGPU is unavailable", async () => {
    const { plan, snapshot } = fixture();
    const session = await new WebGpuBackend({ gpu: null }).compile(plan, snapshot);
    expect(["wasm", "cpu"]).toContain(session.backend);
    expect((await session.forward([0.25, -0.5])).data).toHaveLength(1);
    session.dispose();
  });

  it("reports an error when fallback is explicitly disabled", async () => {
    const { plan, snapshot } = fixture();
    await expect(
      new WebGpuBackend({ gpu: null }).compile(plan, snapshot, { fallback: [] }),
    ).rejects.toMatchObject({ code: "NO_SUPPORTED_BACKEND" });
  });

  it("falls back to a training-capable runtime when the GPU is unavailable", async () => {
    const { plan, snapshot } = fixture();
    const session = await new WebGpuBackend({ gpu: null }).compile(
      plan,
      snapshot,
      { training: true },
    );
    const metrics = await session.trainStep(
      { input: [0.25, -0.5], target: [0.75] },
      { learningRate: 0.05 },
    );
    expect(metrics.loss).toBeGreaterThanOrEqual(0);
    expect(["wasm", "cpu"]).toContain(session.backend);
    session.dispose();
  });

  it("owns and destroys resources created from an available device", async () => {
    const { plan, snapshot } = fixture();
    const destroyed: string[] = [];
    const listeners = new Map<string, EventListener>();
    const fakeDevice = {
      limits: {
        maxStorageBufferBindingSize: 1_000_000,
        maxBufferSize: 1_000_000,
        maxComputeWorkgroupsPerDimension: 65_535,
        maxStorageBuffersPerShaderStage: 8,
        maxUniformBuffersPerShaderStage: 8,
      },
      lost: new Promise(() => undefined),
      queue: {
        writeBuffer() {},
        submit() {},
      },
      pushErrorScope() {},
      async popErrorScope() { return null; },
      createBuffer(descriptor: { label?: string }) {
        return {
          label: descriptor.label,
          destroy() { destroyed.push(descriptor.label ?? "buffer"); },
        };
      },
      createShaderModule() {
        return {
          async getCompilationInfo() { return { messages: [] }; },
        };
      },
      createBindGroupLayout() { return {}; },
      createPipelineLayout() { return {}; },
      async createComputePipelineAsync() { return {}; },
      createBindGroup() { return {}; },
      addEventListener(name: string, listener: EventListener) {
        listeners.set(name, listener);
      },
      removeEventListener(name: string) {
        listeners.delete(name);
      },
      destroy() { destroyed.push("device"); },
    };
    const gpu = {
      async requestAdapter() {
        return {
          async requestDevice() { return fakeDevice; },
        };
      },
    } as unknown as GPU;

    const session = await new WebGpuBackend({
      gpu,
      fallbackBackends: [],
    }).compile(plan, snapshot);
    expect(session).toBeInstanceOf(WebGpuSession);
    expect(listeners.has("uncapturederror")).toBe(true);
    session.dispose();
    expect(destroyed).toEqual([
      "synaptic heap",
      "synaptic stage uniforms",
      "synaptic output readback",
      "device",
    ]);
    expect(listeners.has("uncapturederror")).toBe(false);
  });
});
