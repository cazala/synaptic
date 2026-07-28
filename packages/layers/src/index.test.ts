import { describe, expect, it } from "vitest";
import { compilePlan } from "@synaptic/core";
import { dense, input, lstm, sequential } from "./index.js";

describe("layer macros", () => {
  it("expands an LSTM into portable units, connections, and gates", () => {
    const definition = sequential(
      input({ size: 2 }),
      lstm({ units: 3 }),
      dense({ units: 1 }),
    );
    const plan = compilePlan(definition);

    expect(plan.inputs).toHaveLength(2);
    expect(plan.outputs).toHaveLength(1);
    expect(definition.topology.gates.length).toBeGreaterThan(0);
    expect(
      definition.topology.connections.some((connection) => connection.delay === 1),
    ).toBe(true);
    expect(
      definition.topology.parameters.some((parameter) => !parameter.trainable),
    ).toBe(true);
  });

  it("uses an LSTM output gate for connections created by a later layer", () => {
    const definition = sequential(
      input({ size: 1 }),
      lstm({ units: 2, peepholes: false }),
      dense({ units: 2 }),
    );
    expect(compilePlan(definition).specializations.denseStages.length).toBeGreaterThan(0);
    const outputGates = new Set(
      definition.topology.units
        .filter((unit) => unit.label?.startsWith("lstm.output-gate"))
        .map((unit) => unit.id),
    );
    const denseUnits = new Set(
      definition.topology.units
        .filter((unit) => unit.label?.startsWith("dense"))
        .map((unit) => unit.id),
    );

    expect(
      definition.topology.gates.some((gate) => {
        const connection = definition.topology.connections[gate.connection];
        return outputGates.has(gate.gater) && connection !== undefined && denseUnits.has(connection.to);
      }),
    ).toBe(true);
  });
});
