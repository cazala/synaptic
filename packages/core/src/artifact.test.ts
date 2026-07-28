import { describe, expect, it } from "vitest";
import { loadSnapshot, saveSnapshot } from "./artifact.js";
import { GraphBuilder } from "./graph-builder.js";
import { createSnapshot } from "./model.js";

function model() {
  const graph = new GraphBuilder();
  const input = graph.input(2);
  graph.stage(1);
  const output = graph.units(1);
  graph.connect(input.units, output);
  return graph.build({ inputs: input, outputs: output });
}

describe("portable artifacts", () => {
  it("round-trips a snapshot through little-endian binary sidecars", () => {
    const snapshot = createSnapshot(model(), 42);
    const loaded = loadSnapshot(saveSnapshot(snapshot));
    expect(loaded.definition).toEqual(snapshot.definition);
    expect(loaded.parameters).toEqual(snapshot.parameters);
  });

  it("rejects a modified sidecar", () => {
    const bundle = saveSnapshot(createSnapshot(model(), 42));
    const original = bundle.buffers.get("parameters.bin");
    expect(original).toBeDefined();
    const changed = original!.slice();
    changed[0] = changed[0]! ^ 0xff;
    const corrupted = {
      manifest: bundle.manifest,
      buffers: new Map([["parameters.bin", changed]]),
    };
    expect(() => loadSnapshot(corrupted)).toThrow("checksum");
  });
});
