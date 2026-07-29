// @ts-expect-error -- @cazala/automata 0.1.0's NodeNext declarations omit
// extension suffixes from its internal re-exports, although the bundled runtime
// exports Neural correctly.
import { Neural } from "@cazala/automata";
import {
  createSnapshot,
  type ModelDefinition,
  type ParameterId,
} from "@synaptic/core";
import { describe, expect, it } from "vitest";
// The browser demo is intentionally plain JavaScript so it can run without a
// bundler after the workspace build.
// @ts-expect-error -- browser demo modules do not publish declaration files.
import { AUTOMATA_KERNEL_INITIAL as rawKernelInitial, AUTOMATA_TANH_ACTIVATION as rawTanhActivation, createAutomataConvolutionDefinition as rawCreateDefinition } from "../test/demo-models.js";

interface ConvolutionDefinition {
  readonly definition: ModelDefinition;
  readonly kernelParameters: Readonly<{
    center: ParameterId;
    edge: ParameterId;
    corner: ParameterId;
  }>;
}

const AUTOMATA_KERNEL_INITIAL = rawKernelInitial as Readonly<{
  center: number;
  edge: number;
  corner: number;
}>;
const AUTOMATA_TANH_ACTIVATION = rawTanhActivation as 1;
const createAutomataConvolutionDefinition = rawCreateDefinition as (
  imageSize: number,
  options?: { readonly trainKernel?: boolean },
) => ConvolutionDefinition;

describe("MNIST to Neural CA demo topology", () => {
  it("builds a wrapped convolution with Automata's three tied parameters", () => {
    const { definition, kernelParameters } =
      createAutomataConvolutionDefinition(14);
    const convolution = definition.topology.units.filter(
      (unit) => unit.label?.startsWith("mnist-ca.convolution:"),
    );

    expect(definition.topology.inputs).toHaveLength(196);
    expect(definition.topology.outputs).toHaveLength(10);
    expect(convolution).toHaveLength(196);
    expect(convolution.every((unit) => unit.activation === "tanh")).toBe(true);

    const uses = new Map<number, number>();
    for (const connection of definition.topology.connections) {
      uses.set(connection.parameter, (uses.get(connection.parameter) ?? 0) + 1);
    }
    expect(uses.get(kernelParameters.center)).toBe(196);
    expect(uses.get(kernelParameters.edge)).toBe(196 * 4);
    expect(uses.get(kernelParameters.corner)).toBe(196 * 4);

    const topLeft = convolution[0];
    expect(topLeft).toBeDefined();
    const sources = definition.topology.connections
      .filter((connection) => connection.to === topLeft?.id)
      .map((connection) => connection.from);
    expect(sources).toEqual(
      expect.arrayContaining([195, 182, 183, 13, 0, 1, 27, 14, 15]),
    );
  });

  it("transfers its initial kernel and tanh activation without translation", () => {
    const { definition, kernelParameters } =
      createAutomataConvolutionDefinition(14);
    const snapshot = createSnapshot(definition, 0xca20_12ca);
    const transferred = {
      center: snapshot.parameters[kernelParameters.center],
      edge: snapshot.parameters[kernelParameters.edge],
      corner: snapshot.parameters[kernelParameters.corner],
    };
    const automaton = new Neural({
      mode: "direct",
      channels: 3,
      activation: AUTOMATA_TANH_ACTIVATION,
      kernel: transferred,
    });

    expect(transferred).toEqual(AUTOMATA_KERNEL_INITIAL);
    expect(automaton.getMode()).toBe("direct");
    expect(automaton.getActivation()).toBe(1);
    expect(automaton.getKernel()).toEqual(AUTOMATA_KERNEL_INITIAL);
  });

  it("can freeze only the transferable kernel during readout warm-up", () => {
    const { definition, kernelParameters } =
      createAutomataConvolutionDefinition(14, { trainKernel: false });
    const parameters = definition.topology.parameters;

    expect(parameters[kernelParameters.center]?.trainable).toBe(false);
    expect(parameters[kernelParameters.edge]?.trainable).toBe(false);
    expect(parameters[kernelParameters.corner]?.trainable).toBe(false);
    expect(
      parameters
        .filter((parameter) => !Object.values(kernelParameters).includes(parameter.id))
        .some((parameter) => parameter.trainable),
    ).toBe(true);
  });
});
