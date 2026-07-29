import { describe, expect, it } from "vitest";
import {
  loadCheckpoint,
  loadSnapshot,
  saveCheckpoint,
  saveSnapshot,
} from "./artifact.js";
import { GraphBuilder } from "./graph-builder.js";
import { createSnapshot } from "./model.js";
import { compilePlan } from "./plan.js";
import { createCheckpoint, createRuntimeState } from "./runtime.js";

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

  it("round-trips runtime and optimizer state in a checkpoint", () => {
    const snapshot = createSnapshot(model(), 42);
    const runtime = createRuntimeState(
      compilePlan(snapshot.definition),
      snapshot.definition.precision,
      0xdecafbad,
    );
    runtime.state.fill(0.25);
    runtime.activation.fill(0.5);
    runtime.previousActivation.fill(-0.25);
    runtime.derivative.fill(0.75);
    runtime.eligibilityTrace.fill(1.25);
    runtime.projectedError.fill(-0.5);
    runtime.gatedError.fill(0.125);
    runtime.error.fill(-0.375);
    runtime.step = 17;
    runtime.randomCounter = 9;
    const checkpoint = {
      ...createCheckpoint(snapshot, runtime),
      optimizer: {
        momentum: new Float32Array(snapshot.parameters.length).fill(0.125),
      },
    };

    const loaded = loadCheckpoint(saveCheckpoint(checkpoint));

    expect(loaded.definition).toEqual(checkpoint.definition);
    expect(loaded.parameters).toEqual(checkpoint.parameters);
    expect(loaded.runtime).toEqual(checkpoint.runtime);
    expect(loaded.optimizer).toEqual(checkpoint.optimizer);
  });

  it("rejects corrupted checkpoint runtime sidecars", () => {
    const snapshot = createSnapshot(model(), 42);
    const checkpoint = createCheckpoint(
      snapshot,
      createRuntimeState(compilePlan(snapshot.definition), "f32"),
    );
    const bundle = saveCheckpoint(checkpoint);
    const id = "runtime/state.bin";
    const changed = bundle.buffers.get(id)?.slice();
    expect(changed).toBeDefined();
    changed![0] = changed![0]! ^ 0xff;

    expect(() => loadCheckpoint({
      manifest: bundle.manifest,
      buffers: new Map([...bundle.buffers, [id, changed!]]),
    })).toThrow("checksum");
  });

  it("does not mistake a model snapshot for a resumable checkpoint", () => {
    expect(() => loadCheckpoint(saveSnapshot(createSnapshot(model(), 42))))
      .toThrow("checkpoint metadata");
  });
});
