import { describe, expect, it } from "vitest";
import { CpuBackend } from "@synaptic/backend-cpu";
import { compilePlan, loadSnapshot, saveSnapshot } from "@synaptic/core";
import { detectLegacyFormat, importLegacy } from "./index.js";

async function readFixture(name: string): Promise<unknown> {
  const moduleName = "node:fs/promises";
  const { readFile } = await import(moduleName);
  return JSON.parse(
    await readFile(new URL(`../test/fixtures/${name}`, import.meta.url), "utf8"),
  ) as unknown;
}

describe("legacy import", () => {
  it("imports v1 weights, trainable biases, schedule, and recurrent context", async () => {
    const source = await readFixture("v1-network.json");
    expect(detectLegacyFormat(source)).toBe("synaptic-v1");
    const imported = importLegacy(source);
    const { definition, snapshot, checkpoint } = imported.value;

    expect(definition.topology.units).toHaveLength(3);
    expect(definition.topology.connections).toHaveLength(2);
    expect([...snapshot.parameters]).toEqual([0.5, 0.25]);
    expect(checkpoint.runtime.activation).toHaveLength(3);
    expect(loadSnapshot(saveSnapshot(snapshot)).parameters).toEqual(snapshot.parameters);

    const session = await new CpuBackend().compile(
      compilePlan(definition),
      snapshot,
    );
    expect((await session.forward([2])).data[0]).toBeCloseTo(
      1 / (1 + Math.exp(-1.25)),
      6,
    );
    session.dispose();
  });

  it("rejects unfinished synaptic2 engine artifacts", () => {
    expect(() => importLegacy({
      activationFunction: [5, 0],
      connections: [{ from: 0, to: 1 }],
      layers: [[0], [1]],
      size: 2,
    })).toThrow("Unknown legacy artifact format");
  });

  it("rejects unknown legacy artifacts", () => {
    expect(() => detectLegacyFormat({ version: 99 })).toThrow(
      "Unknown legacy artifact format",
    );
  });
});
