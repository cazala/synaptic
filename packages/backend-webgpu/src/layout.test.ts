import { describe, expect, it } from "vitest";
import {
  GraphBuilder,
  compilePlan,
  createSnapshot,
} from "@synaptic/core";
import { buildHeapLayout, buildWebGpuHeap } from "./layout.js";
import { FORWARD_WGSL } from "./forward-shader.js";

function fixture() {
  const graph = new GraphBuilder();
  const input = graph.input(2);
  graph.stage(1);
  const hidden = graph.units(2, { activation: "tanh" });
  graph.stage(2);
  const output = graph.units(1);
  graph.connect(input.units, hidden);
  graph.connect(hidden, hidden, "one-to-one", {
    delay: 1,
    trainable: false,
    initializer: { kind: "constant", value: 1 },
  });
  graph.connect(hidden, output);
  return graph.build({ inputs: input, outputs: output });
}

describe("WebGPU heap lowering", () => {
  it("packs every host/shader section deterministically on word boundaries", () => {
    const definition = fixture();
    const plan = compilePlan(definition);
    const first = buildHeapLayout(plan);
    const second = buildHeapLayout(plan);

    expect(second).toEqual(first);
    expect(first.byteLength % 4).toBe(0);
    for (const section of Object.values(first.sections)) {
      expect(section.offset % 4).toBe(0);
      expect(section.wordOffset).toBe(section.offset / 4);
      expect(section.byteLength).toBe(section.length * 4);
    }
  });

  it("copies plan IDs and parameters without numeric reinterpretation drift", () => {
    const definition = fixture();
    const plan = compilePlan(definition);
    const snapshot = createSnapshot(definition, 44);
    const heap = buildWebGpuHeap(plan, snapshot);
    const parameterSection = heap.layout.sections.parameters;
    const parameters = new Float32Array(
      heap.bytes.buffer,
      parameterSection.offset,
      parameterSection.length,
    );

    expect(parameters).toEqual(snapshot.parameters);
  });

  it("indexes connections by parameter without a quadratic shader scan", () => {
    const definition = fixture();
    const plan = compilePlan(definition);
    const heap = buildWebGpuHeap(plan, createSnapshot(definition, 44));
    const offsetsSection = heap.layout.sections.parameterOffsets;
    const connectionsSection = heap.layout.sections.parameterConnections;
    const offsets = new Uint32Array(
      heap.bytes.buffer,
      offsetsSection.offset,
      offsetsSection.length,
    );
    const connections = new Uint32Array(
      heap.bytes.buffer,
      connectionsSection.offset,
      connectionsSection.length,
    );

    expect(offsets[plan.parameterCount]).toBe(plan.connectionCount);
    expect([...connections].sort((left, right) => left - right)).toEqual(
      Array.from({ length: plan.connectionCount }, (_, index) => index),
    );
    for (let parameter = 0; parameter < plan.parameterCount; parameter += 1) {
      const start = offsets[parameter] ?? 0;
      const end = offsets[parameter + 1] ?? start;
      for (let cursor = start; cursor < end; cursor += 1) {
        expect(plan.connectionParameter[connections[cursor] ?? -1]).toBe(parameter);
      }
    }
    expect(FORWARD_WGSL).toContain("config.parameterConnections + cursor");
    expect(FORWARD_WGSL).not.toContain(
      "connection < config.connectionCount; connection += 1u",
    );
  });

  it("guards every over-dispatched entry point", () => {
    expect(FORWARD_WGSL).toContain("if (unit >= config.unitCount)");
    expect(FORWARD_WGSL).toContain("if (localIndex >= stageLength)");
    expect(FORWARD_WGSL).toContain("if (slot >= config.outputCount)");
    expect(FORWARD_WGSL).toContain("if (index < config.connectionCount)");
    expect(FORWARD_WGSL).toContain("if (parameter >= config.parameterCount)");
  });
});
