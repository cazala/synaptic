// @ts-expect-error -- @cazala/automata's NodeNext declarations omit
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
import { AUTOMATA_KERNEL_INITIAL as rawKernelInitial, AUTOMATA_NETWORK_CHANNELS as rawNetworkChannels, AUTOMATA_NETWORK_HIDDEN as rawNetworkHidden, AUTOMATA_TANH_ACTIVATION as rawTanhActivation, createAutomataConvolutionDefinition as rawCreateDefinition, createAutomataNetworkDefinition as rawCreateNetworkDefinition, extractAutomataNetworkWeights as rawExtractNetworkWeights } from "../test/demo-models.js";

interface ConvolutionDefinition {
  readonly definition: ModelDefinition;
  readonly kernelParameters: Readonly<{
    center: ParameterId;
    edge: ParameterId;
    corner: ParameterId;
  }>;
}

interface NetworkParameterMap {
  readonly channels: number;
  readonly hidden: number;
  readonly inputToHidden: readonly ParameterId[];
  readonly hiddenBias: readonly ParameterId[];
  readonly hiddenToOutput: readonly ParameterId[];
  readonly outputBias: readonly ParameterId[];
}

interface NetworkDefinition {
  readonly definition: ModelDefinition;
  readonly filterParameters: Readonly<{
    positiveEighth: ParameterId;
    negativeEighth: ParameterId;
    positiveQuarter: ParameterId;
    negativeQuarter: ParameterId;
    one: ParameterId;
  }>;
  readonly kernelParameters: ConvolutionDefinition["kernelParameters"];
  readonly networkParameters: NetworkParameterMap;
}

interface NetworkWeights {
  readonly channels: number;
  readonly hidden: number;
  readonly inputToHidden: Float32Array;
  readonly hiddenBias: Float32Array;
  readonly hiddenToOutput: Float32Array;
  readonly outputBias: Float32Array;
}

const AUTOMATA_KERNEL_INITIAL = rawKernelInitial as Readonly<{
  center: number;
  edge: number;
  corner: number;
}>;
const AUTOMATA_NETWORK_CHANNELS = rawNetworkChannels as 3;
const AUTOMATA_NETWORK_HIDDEN = rawNetworkHidden as 8;
const AUTOMATA_TANH_ACTIVATION = rawTanhActivation as 1;
const createAutomataConvolutionDefinition = rawCreateDefinition as (
  imageSize: number,
  options?: { readonly trainKernel?: boolean },
) => ConvolutionDefinition;
const createAutomataNetworkDefinition = rawCreateNetworkDefinition as (
  imageSize: number,
  options?: {
    readonly channels?: number;
    readonly hidden?: number;
    readonly trainLocalRule?: boolean;
  },
) => NetworkDefinition;
const extractAutomataNetworkWeights = rawExtractNetworkWeights as (
  snapshot: ReturnType<typeof createSnapshot>,
  parameters: NetworkParameterMap,
) => NetworkWeights;

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

describe("MNIST to Neural CA network-mode topology", () => {
  it("matches Automata's four perception blocks and shared MLP shapes", () => {
    const {
      definition,
      filterParameters,
      kernelParameters,
      networkParameters,
    } = createAutomataNetworkDefinition(14);
    const units = definition.topology.units;
    const parameters = definition.topology.parameters;
    const uses = new Map<number, number>();
    for (const connection of definition.topology.connections) {
      uses.set(connection.parameter, (uses.get(connection.parameter) ?? 0) + 1);
    }

    expect(definition.topology.inputs).toHaveLength(196);
    expect(definition.topology.outputs).toHaveLength(10);
    expect(
      units.filter((unit) =>
        unit.label?.startsWith("mnist-ca-network.perception.")
      ),
    ).toHaveLength(196 * AUTOMATA_NETWORK_CHANNELS * 4);
    expect(
      units.filter((unit) =>
        unit.label?.startsWith("mnist-ca-network.hidden.")
      ),
    ).toHaveLength(196 * AUTOMATA_NETWORK_HIDDEN);
    expect(
      units.filter((unit) =>
        unit.label?.startsWith("mnist-ca-network.delta.")
      ),
    ).toHaveLength(196 * AUTOMATA_NETWORK_CHANNELS);

    expect(networkParameters.inputToHidden).toHaveLength(
      AUTOMATA_NETWORK_HIDDEN * AUTOMATA_NETWORK_CHANNELS * 4,
    );
    expect(networkParameters.hiddenBias).toHaveLength(
      AUTOMATA_NETWORK_HIDDEN,
    );
    expect(networkParameters.hiddenToOutput).toHaveLength(
      AUTOMATA_NETWORK_CHANNELS * AUTOMATA_NETWORK_HIDDEN,
    );
    expect(networkParameters.outputBias).toHaveLength(
      AUTOMATA_NETWORK_CHANNELS,
    );
    for (const parameter of [
      ...networkParameters.inputToHidden,
      ...networkParameters.hiddenBias,
      ...networkParameters.hiddenToOutput,
      ...networkParameters.outputBias,
    ]) {
      expect(uses.get(parameter)).toBe(196);
    }

    expect(uses.get(kernelParameters.center)).toBe(196 * 3);
    expect(uses.get(kernelParameters.edge)).toBe(196 * 3 * 4);
    expect(uses.get(kernelParameters.corner)).toBe(196 * 3 * 4);
    expect(uses.get(filterParameters.one)).toBe(196 * 3);
    expect(uses.get(filterParameters.positiveEighth)).toBe(196 * 3 * 4);
    expect(uses.get(filterParameters.negativeEighth)).toBe(196 * 3 * 4);
    expect(uses.get(filterParameters.positiveQuarter)).toBe(196 * 3 * 2);
    expect(uses.get(filterParameters.negativeQuarter)).toBe(196 * 3 * 2);
    expect(
      Object.values(filterParameters).every(
        (parameter) => parameters[parameter]?.trainable === false,
      ),
    ).toBe(true);
  });

  it("uses Automata's block order and exact wrapped Sobel-x taps", () => {
    const { definition, filterParameters } =
      createAutomataNetworkDefinition(14);
    const sobelX = definition.topology.units.find((unit) =>
      unit.label === "mnist-ca-network.perception.sobel-x.channel-0:0"
    );
    expect(sobelX).toBeDefined();

    const incoming = definition.topology.connections
      .filter((connection) => connection.to === sobelX?.id)
      .map((connection) => ({
        from: connection.from,
        parameter: connection.parameter,
      }));
    expect(incoming).toEqual([
      { from: 183, parameter: filterParameters.positiveEighth },
      { from: 1, parameter: filterParameters.positiveQuarter },
      { from: 15, parameter: filterParameters.positiveEighth },
      { from: 195, parameter: filterParameters.negativeEighth },
      { from: 13, parameter: filterParameters.negativeQuarter },
      { from: 27, parameter: filterParameters.negativeEighth },
    ]);
  });

  it("exports a byte-for-byte network artifact accepted by Automata 0.2", () => {
    const { definition, kernelParameters, networkParameters } =
      createAutomataNetworkDefinition(14);
    const snapshot = createSnapshot(definition, 0x0ca2_0200);
    const weights = extractAutomataNetworkWeights(
      snapshot,
      networkParameters,
    );
    const kernel = {
      center: snapshot.parameters[kernelParameters.center] ?? 0,
      edge: snapshot.parameters[kernelParameters.edge] ?? 0,
      corner: snapshot.parameters[kernelParameters.corner] ?? 0,
    };
    const automaton = new Neural({
      mode: "network",
      activation: AUTOMATA_TANH_ACTIVATION,
      kernel,
      weights,
    });

    expect(automaton.getMode()).toBe("network");
    expect(automaton.getChannels()).toBe(AUTOMATA_NETWORK_CHANNELS);
    expect(automaton.getHidden()).toBe(AUTOMATA_NETWORK_HIDDEN);
    expect(automaton.getActivation()).toBe(AUTOMATA_TANH_ACTIVATION);
    expect(automaton.getKernel()).toEqual(AUTOMATA_KERNEL_INITIAL);
    expect(automaton.getNetworkWeights()).toEqual(weights);
  });

  it("freezes the entire transferable local rule during readout warm-up", () => {
    const { definition, kernelParameters, networkParameters } =
      createAutomataNetworkDefinition(14, { trainLocalRule: false });
    const localParameters = [
      ...Object.values(kernelParameters),
      ...networkParameters.inputToHidden,
      ...networkParameters.hiddenBias,
      ...networkParameters.hiddenToOutput,
      ...networkParameters.outputBias,
    ];

    expect(
      localParameters.every(
        (parameter) =>
          definition.topology.parameters[parameter]?.trainable === false,
      ),
    ).toBe(true);
    expect(
      definition.topology.parameters.some(
        (parameter) =>
          !localParameters.includes(parameter.id) && parameter.trainable,
      ),
    ).toBe(true);
  });
});
