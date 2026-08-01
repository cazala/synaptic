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

  it("allows each LSTM gate bias initializer to be tuned", () => {
    const definition = sequential(
      input({ size: 1 }),
      lstm({
        units: 1,
        inputBias: -1,
        forgetBias: 3,
        memoryBias: 0.25,
        outputBias: 2,
      }),
      dense({ units: 1 }),
    );
    const initializers = Object.fromEntries(
      definition.topology.parameters
        .filter((parameter) => parameter.label?.includes("-bias"))
        .map((parameter) => [
          parameter.label,
          parameter.initializer.kind === "constant"
            ? parameter.initializer.value
            : undefined,
        ]),
    );

    expect(initializers).toMatchObject({
      "lstm.input-bias": -1,
      "lstm.forget-bias": 3,
      "lstm.memory-bias": 0.25,
      "lstm.output-bias": 2,
    });
  });
});
