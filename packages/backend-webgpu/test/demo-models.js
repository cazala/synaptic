import { WebGpuBackend } from "@synaptic/backend-webgpu";
import {
  compilePlan,
  createSnapshot,
  GraphBuilder,
} from "@synaptic/core";
import { dense, input, lstm, sequential } from "@synaptic/layers";

const createBackend = () => new WebGpuBackend({ fallbackBackends: [] });

const argmax = (values) => {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) {
    if ((values[index] ?? -Infinity) > (values[best] ?? -Infinity)) {
      best = index;
    }
  }
  return best;
};

export const XOR_SAMPLES = [
  { input: [0, 0], target: [0] },
  { input: [0, 1], target: [1] },
  { input: [1, 0], target: [1] },
  { input: [1, 1], target: [0] },
];

export async function trainXorDemo(onProgress) {
  const definition = sequential(
    input({ size: 2 }),
    dense({ units: 3, activation: "logistic", label: "xor.hidden" }),
    dense({ units: 1, activation: "logistic", label: "xor.output" }),
  );
  const session = await createBackend().compile(
    compilePlan(definition),
    createSnapshot(definition, 0xfeed_1234),
    { training: true },
  );
  const epochs = 2_500;
  const started = performance.now();
  try {
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const sequence = XOR_SAMPLES.map(
        (_, offset) => XOR_SAMPLES[(epoch + offset) % XOR_SAMPLES.length],
      );
      await session.trainSequence(sequence, {
        learningRate: .3,
        metrics: "none",
      });
      if ((epoch + 1) % 100 === 0 || epoch + 1 === epochs) {
        onProgress?.(epoch + 1, epochs);
      }
    }
    const outputs = await session.forwardSequence(
      XOR_SAMPLES.map(({ input: values }) => values),
    );
    return {
      backend: session.backend,
      durationMs: Math.round(performance.now() - started),
      epochs,
      outputs: outputs.map(({ data }) => data[0] ?? 0),
    };
  } finally {
    session.dispose();
  }
}

async function evaluateMnist(session, examples) {
  const outputs = await session.forwardSequence(
    examples.map(({ input: values }) => values),
  );
  let correct = 0;
  for (let index = 0; index < examples.length; index += 1) {
    correct += argmax(outputs[index]?.data ?? []) === examples[index]?.label
      ? 1
      : 0;
  }
  return correct / examples.length;
}

export async function trainMnistDemo(dataset, onProgress) {
  const definition = sequential(
    input({ size: dataset.imageSize * dataset.imageSize }),
    dense({ units: 128, activation: "logistic", label: "mnist.hidden" }),
    dense({ units: 10, activation: "logistic", label: "mnist.output" }),
  );
  const session = await createBackend().compile(
    compilePlan(definition),
    createSnapshot(definition, 0x6d6e_6973),
    { training: true },
  );
  const epochs = 30;
  const stride = 7;
  const started = performance.now();
  try {
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const start = (epoch * 13) % dataset.training.length;
      const sequence = Array.from(
        { length: dataset.training.length },
        (_, offset) =>
          dataset.training[(start + offset * stride) % dataset.training.length],
      );
      await session.trainSequence(sequence, {
        learningRate: .08,
        metrics: "none",
      });
      onProgress?.(epoch + 1, epochs);
    }
    const trainingAccuracy = await evaluateMnist(session, dataset.training);
    const testAccuracy = await evaluateMnist(session, dataset.test);
    return {
      backend: session.backend,
      durationMs: Math.round(performance.now() - started),
      epochs,
      session,
      testAccuracy,
      trainingAccuracy,
    };
  } catch (error) {
    session.dispose();
    throw error;
  }
}

export const AUTOMATA_TANH_ACTIVATION = 1;
export const AUTOMATA_NETWORK_CHANNELS = 3;
export const AUTOMATA_NETWORK_HIDDEN = 8;
export const AUTOMATA_KERNEL_INITIAL = Object.freeze({
  center: 1,
  edge: -.125,
  corner: .0625,
});

export function createAutomataConvolutionDefinition(
  imageSize,
  { trainKernel = true } = {},
) {
  if (!Number.isInteger(imageSize) || imageSize < 3) {
    throw new RangeError("The convolution image size must be an integer of at least 3");
  }

  const graph = new GraphBuilder();
  const inputs = graph.input(
    imageSize * imageSize,
    [imageSize, imageSize],
  );
  const kernelParameters = {
    center: graph.parameter({
      trainable: trainKernel,
      initializer: {
        kind: "constant",
        value: AUTOMATA_KERNEL_INITIAL.center,
      },
      label: "mnist-ca.kernel.center",
    }),
    edge: graph.parameter({
      trainable: trainKernel,
      initializer: {
        kind: "constant",
        value: AUTOMATA_KERNEL_INITIAL.edge,
      },
      label: "mnist-ca.kernel.edge",
    }),
    corner: graph.parameter({
      trainable: trainKernel,
      initializer: {
        kind: "constant",
        value: AUTOMATA_KERNEL_INITIAL.corner,
      },
      label: "mnist-ca.kernel.corner",
    }),
  };

  graph.stage(1);
  const convolution = graph.units(imageSize * imageSize, {
    activation: "tanh",
    label: "mnist-ca.convolution",
  });
  const taps = [
    [-1, -1, kernelParameters.corner],
    [0, -1, kernelParameters.edge],
    [1, -1, kernelParameters.corner],
    [-1, 0, kernelParameters.edge],
    [0, 0, kernelParameters.center],
    [1, 0, kernelParameters.edge],
    [-1, 1, kernelParameters.corner],
    [0, 1, kernelParameters.edge],
    [1, 1, kernelParameters.corner],
  ];
  for (let y = 0; y < imageSize; y += 1) {
    for (let x = 0; x < imageSize; x += 1) {
      const target = convolution[y * imageSize + x];
      for (const [offsetX, offsetY, parameter] of taps) {
        const sourceX = (x + offsetX + imageSize) % imageSize;
        const sourceY = (y + offsetY + imageSize) % imageSize;
        graph.connect(
          [inputs.units[sourceY * imageSize + sourceX]],
          [target],
          "one-to-one",
          { delay: 0, parameter },
        );
      }
    }
  }

  graph.stage(2);
  const outputs = graph.units(10, {
    activation: "logistic",
    label: "mnist-ca.output",
  });
  graph.connect(convolution, outputs, "all-to-all", {
    initializer: { kind: "uniform", min: -.1, max: .1 },
    label: "mnist-ca.readout",
  });
  const bias = graph.constant(1, 0);
  graph.connect([bias], outputs, "all-to-all", {
    initializer: { kind: "constant", value: 0 },
    label: "mnist-ca.output-bias",
  });

  return {
    definition: graph.build({
      inputs,
      outputs,
      metadata: {
        activation: "tanh",
        automataMode: "direct",
        convolution: {
          kernel: "symmetric-3x3",
          padding: "wrap",
          stride: 1,
        },
        name: "MNIST → Neural CA",
      },
    }),
    kernelParameters,
  };
}

export async function trainAutomataConvolutionDemo(dataset, onProgress) {
  const warmupEpochs = 30;
  const fineTuneEpochs = 12;
  const totalEpochs = warmupEpochs + fineTuneEpochs;
  const stride = 7;
  const started = performance.now();
  const orderedEpoch = (epoch) => {
    const start = (epoch * 13) % dataset.training.length;
    return Array.from(
      { length: dataset.training.length },
      (_, offset) =>
        dataset.training[(start + offset * stride) % dataset.training.length],
    );
  };

  const warmup = createAutomataConvolutionDefinition(
    dataset.imageSize,
    { trainKernel: false },
  );
  const warmupSession = await createBackend().compile(
    compilePlan(warmup.definition),
    createSnapshot(warmup.definition, 0xca20_12ca),
    { training: true },
  );
  let warmupSnapshot;
  try {
    for (let epoch = 0; epoch < warmupEpochs; epoch += 1) {
      await warmupSession.trainSequence(orderedEpoch(epoch), {
        learningRate: .08,
        metrics: "none",
      });
      onProgress?.(epoch + 1, totalEpochs, "readout");
    }
    warmupSnapshot = await warmupSession.snapshot();
  } finally {
    warmupSession.dispose();
  }

  const fineTune = createAutomataConvolutionDefinition(dataset.imageSize);
  const initial = createSnapshot(fineTune.definition, 0xca20_12ca);
  initial.parameters.set(warmupSnapshot.parameters);
  const session = await createBackend().compile(
    compilePlan(fineTune.definition),
    initial,
    { training: true },
  );
  try {
    for (let epoch = 0; epoch < fineTuneEpochs; epoch += 1) {
      await session.trainSequence(orderedEpoch(warmupEpochs + epoch), {
        learningRate: .00035,
        metrics: "none",
      });
      onProgress?.(
        warmupEpochs + epoch + 1,
        totalEpochs,
        "kernel + readout",
      );
    }
    const trainingAccuracy = await evaluateMnist(session, dataset.training);
    const testAccuracy = await evaluateMnist(session, dataset.test);
    const snapshot = await session.snapshot();
    const kernel = Object.fromEntries(
      Object.entries(fineTune.kernelParameters).map(([name, parameter]) => [
        name,
        snapshot.parameters[parameter] ?? 0,
      ]),
    );
    const examples = Array.from(
      { length: 10 },
      (_, label) => dataset.test.find((example) => example.label === label),
    ).filter(Boolean);
    const predictions = await session.forwardSequence(
      examples.map(({ input: values }) => values),
    );
    return {
      backend: session.backend,
      durationMs: Math.round(performance.now() - started),
      epochs: totalEpochs,
      examples: examples.map((example, index) => ({
        ...example,
        prediction: argmax(predictions[index]?.data ?? []),
      })),
      kernel,
      testAccuracy,
      trainingAccuracy,
    };
  } finally {
    session.dispose();
  }
}

const AUTOMATA_PERCEPTION_FILTERS = Object.freeze([
  "identity",
  "sobel-x",
  "sobel-y",
  "kernel",
]);

const fixedParameter = (graph, value, label) => graph.parameter({
  trainable: false,
  initializer: { kind: "constant", value },
  label,
});

const connectWrappedFilter = (
  graph,
  inputs,
  targets,
  imageSize,
  taps,
) => {
  for (let y = 0; y < imageSize; y += 1) {
    for (let x = 0; x < imageSize; x += 1) {
      const target = targets[y * imageSize + x];
      for (const [offsetX, offsetY, parameter] of taps) {
        const sourceX = (x + offsetX + imageSize) % imageSize;
        const sourceY = (y + offsetY + imageSize) % imageSize;
        graph.connect(
          [inputs.units[sourceY * imageSize + sourceX]],
          [target],
          "one-to-one",
          { delay: 0, parameter },
        );
      }
    }
  }
};

/**
 * Build the exact local substrate used by Automata's Neural network mode:
 * four perception blocks per channel, a shared tanh MLP, and a three-channel
 * linear delta. A Synaptic-only classifier reads every cell's delta.
 *
 * The MNIST input is copied conceptually into all three channels during
 * training. At runtime each Automata channel is perceived independently by the
 * same exported matrices, allowing the seed channels to develop differently.
 */
export function createAutomataNetworkDefinition(
  imageSize,
  {
    channels = AUTOMATA_NETWORK_CHANNELS,
    hidden = AUTOMATA_NETWORK_HIDDEN,
    trainLocalRule = true,
  } = {},
) {
  if (!Number.isInteger(imageSize) || imageSize < 3) {
    throw new RangeError("The network image size must be an integer of at least 3");
  }
  if (!Number.isInteger(channels) || channels < 1 || channels > 16) {
    throw new RangeError("Automata network channels must be between 1 and 16");
  }
  if (!Number.isInteger(hidden) || hidden < 1 || hidden > 64) {
    throw new RangeError("Automata network hidden units must be between 1 and 64");
  }

  const graph = new GraphBuilder();
  const cellCount = imageSize * imageSize;
  const inputs = graph.input(cellCount, [imageSize, imageSize]);
  const kernelParameters = {
    center: graph.parameter({
      trainable: trainLocalRule,
      initializer: {
        kind: "constant",
        value: AUTOMATA_KERNEL_INITIAL.center,
      },
      label: "mnist-ca-network.kernel.center",
    }),
    edge: graph.parameter({
      trainable: trainLocalRule,
      initializer: {
        kind: "constant",
        value: AUTOMATA_KERNEL_INITIAL.edge,
      },
      label: "mnist-ca-network.kernel.edge",
    }),
    corner: graph.parameter({
      trainable: trainLocalRule,
      initializer: {
        kind: "constant",
        value: AUTOMATA_KERNEL_INITIAL.corner,
      },
      label: "mnist-ca-network.kernel.corner",
    }),
  };
  const filterParameters = {
    positiveEighth: fixedParameter(
      graph,
      .125,
      "mnist-ca-network.filter.positive-eighth",
    ),
    negativeEighth: fixedParameter(
      graph,
      -.125,
      "mnist-ca-network.filter.negative-eighth",
    ),
    positiveQuarter: fixedParameter(
      graph,
      .25,
      "mnist-ca-network.filter.positive-quarter",
    ),
    negativeQuarter: fixedParameter(
      graph,
      -.25,
      "mnist-ca-network.filter.negative-quarter",
    ),
    one: fixedParameter(graph, 1, "mnist-ca-network.filter.one"),
  };

  graph.stage(1);
  const perception = Array.from(
    { length: AUTOMATA_PERCEPTION_FILTERS.length * channels },
    (_, slot) => {
      const block = Math.floor(slot / channels);
      const channel = slot % channels;
      return graph.units(cellCount, {
        activation: "identity",
        label:
          `mnist-ca-network.perception.${AUTOMATA_PERCEPTION_FILTERS[block]}.channel-${channel}`,
      });
    },
  );
  const identityTaps = [[0, 0, filterParameters.one]];
  const sobelXTaps = [
    [1, -1, filterParameters.positiveEighth],
    [1, 0, filterParameters.positiveQuarter],
    [1, 1, filterParameters.positiveEighth],
    [-1, -1, filterParameters.negativeEighth],
    [-1, 0, filterParameters.negativeQuarter],
    [-1, 1, filterParameters.negativeEighth],
  ];
  const sobelYTaps = [
    [-1, 1, filterParameters.positiveEighth],
    [0, 1, filterParameters.positiveQuarter],
    [1, 1, filterParameters.positiveEighth],
    [-1, -1, filterParameters.negativeEighth],
    [0, -1, filterParameters.negativeQuarter],
    [1, -1, filterParameters.negativeEighth],
  ];
  const kernelTaps = [
    [-1, -1, kernelParameters.corner],
    [0, -1, kernelParameters.edge],
    [1, -1, kernelParameters.corner],
    [-1, 0, kernelParameters.edge],
    [0, 0, kernelParameters.center],
    [1, 0, kernelParameters.edge],
    [-1, 1, kernelParameters.corner],
    [0, 1, kernelParameters.edge],
    [1, 1, kernelParameters.corner],
  ];
  for (let channel = 0; channel < channels; channel += 1) {
    connectWrappedFilter(
      graph,
      inputs,
      perception[channel],
      imageSize,
      identityTaps,
    );
    connectWrappedFilter(
      graph,
      inputs,
      perception[channels + channel],
      imageSize,
      sobelXTaps,
    );
    connectWrappedFilter(
      graph,
      inputs,
      perception[2 * channels + channel],
      imageSize,
      sobelYTaps,
    );
    connectWrappedFilter(
      graph,
      inputs,
      perception[3 * channels + channel],
      imageSize,
      kernelTaps,
    );
  }

  const perceptionSize = channels * AUTOMATA_PERCEPTION_FILTERS.length;
  const hiddenScale = Math.sqrt(6 / (perceptionSize + hidden));
  const outputScale = Math.sqrt(6 / (hidden + channels));
  const networkParameters = {
    channels,
    hidden,
    inputToHidden: [],
    hiddenBias: [],
    hiddenToOutput: [],
    outputBias: [],
  };
  graph.stage(2);
  const hiddenUnits = Array.from({ length: hidden }, (_, hiddenIndex) =>
    graph.units(cellCount, {
      activation: "tanh",
      label: `mnist-ca-network.hidden.${hiddenIndex}`,
    })
  );
  const bias = graph.constant(1, 0);
  for (let hiddenIndex = 0; hiddenIndex < hidden; hiddenIndex += 1) {
    for (
      let perceptionIndex = 0;
      perceptionIndex < perceptionSize;
      perceptionIndex += 1
    ) {
      const parameter = graph.parameter({
        trainable: trainLocalRule,
        initializer: {
          kind: "uniform",
          min: -hiddenScale,
          max: hiddenScale,
        },
        label:
          `mnist-ca-network.w1.${hiddenIndex}.${perceptionIndex}`,
      });
      networkParameters.inputToHidden.push(parameter);
      graph.connect(
        perception[perceptionIndex],
        hiddenUnits[hiddenIndex],
        "one-to-one",
        { delay: 0, parameter },
      );
    }
    const parameter = graph.parameter({
      trainable: trainLocalRule,
      initializer: { kind: "constant", value: 0 },
      label: `mnist-ca-network.b1.${hiddenIndex}`,
    });
    networkParameters.hiddenBias.push(parameter);
    graph.connect([bias], hiddenUnits[hiddenIndex], "all-to-all", {
      delay: 0,
      parameter,
    });
  }

  graph.stage(3);
  const deltaUnits = Array.from({ length: channels }, (_, channel) =>
    graph.units(cellCount, {
      activation: "identity",
      label: `mnist-ca-network.delta.channel-${channel}`,
    })
  );
  for (let channel = 0; channel < channels; channel += 1) {
    for (let hiddenIndex = 0; hiddenIndex < hidden; hiddenIndex += 1) {
      const parameter = graph.parameter({
        trainable: trainLocalRule,
        initializer: {
          kind: "uniform",
          min: -outputScale,
          max: outputScale,
        },
        label: `mnist-ca-network.w2.${channel}.${hiddenIndex}`,
      });
      networkParameters.hiddenToOutput.push(parameter);
      graph.connect(
        hiddenUnits[hiddenIndex],
        deltaUnits[channel],
        "one-to-one",
        { delay: 0, parameter },
      );
    }
    const parameter = graph.parameter({
      trainable: trainLocalRule,
      initializer: { kind: "constant", value: 0 },
      label: `mnist-ca-network.b2.${channel}`,
    });
    networkParameters.outputBias.push(parameter);
    graph.connect([bias], deltaUnits[channel], "all-to-all", {
      delay: 0,
      parameter,
    });
  }

  graph.stage(4);
  const outputs = graph.units(10, {
    activation: "logistic",
    label: "mnist-ca-network.readout",
  });
  graph.connect(deltaUnits.flat(), outputs, "all-to-all", {
    initializer: { kind: "uniform", min: -.075, max: .075 },
    label: "mnist-ca-network.classifier",
  });
  graph.connect([bias], outputs, "all-to-all", {
    initializer: { kind: "constant", value: 0 },
    label: "mnist-ca-network.classifier-bias",
  });

  return {
    definition: graph.build({
      inputs,
      outputs,
      metadata: {
        activation: "tanh",
        automataMode: "network",
        channels,
        hidden,
        perception: [
          "identity",
          "sobel-x",
          "sobel-y",
          "symmetric-3x3",
        ],
        name: "MNIST → Neural CA network",
      },
    }),
    filterParameters,
    kernelParameters,
    networkParameters,
  };
}

const valuesForParameters = (snapshot, parameters) =>
  Float32Array.from(
    parameters,
    (parameter) => snapshot.parameters[parameter] ?? 0,
  );

export function extractAutomataNetworkWeights(snapshot, networkParameters) {
  return {
    channels: networkParameters.channels,
    hidden: networkParameters.hidden,
    inputToHidden: valuesForParameters(
      snapshot,
      networkParameters.inputToHidden,
    ),
    hiddenBias: valuesForParameters(
      snapshot,
      networkParameters.hiddenBias,
    ),
    hiddenToOutput: valuesForParameters(
      snapshot,
      networkParameters.hiddenToOutput,
    ),
    outputBias: valuesForParameters(
      snapshot,
      networkParameters.outputBias,
    ),
  };
}

export async function trainAutomataNetworkDemo(dataset, onProgress) {
  const warmupEpochs = 24;
  const fineTuneEpochs = 12;
  const totalEpochs = warmupEpochs + fineTuneEpochs;
  const stride = 7;
  const started = performance.now();
  const orderedEpoch = (epoch) => {
    const start = (epoch * 13) % dataset.training.length;
    return Array.from(
      { length: dataset.training.length },
      (_, offset) =>
        dataset.training[(start + offset * stride) % dataset.training.length],
    );
  };

  const warmup = createAutomataNetworkDefinition(
    dataset.imageSize,
    { trainLocalRule: false },
  );
  const warmupSession = await createBackend().compile(
    compilePlan(warmup.definition),
    createSnapshot(warmup.definition, 0x0ca2_0200),
    { training: true },
  );
  let warmupSnapshot;
  try {
    for (let epoch = 0; epoch < warmupEpochs; epoch += 1) {
      await warmupSession.trainSequence(orderedEpoch(epoch), {
        learningRate: .07,
        metrics: "none",
      });
      onProgress?.(epoch + 1, totalEpochs, "network readout");
    }
    warmupSnapshot = await warmupSession.snapshot();
  } finally {
    warmupSession.dispose();
  }

  const fineTune = createAutomataNetworkDefinition(dataset.imageSize);
  const initial = createSnapshot(fineTune.definition, 0x0ca2_0200);
  initial.parameters.set(warmupSnapshot.parameters);
  const session = await createBackend().compile(
    compilePlan(fineTune.definition),
    initial,
    { training: true },
  );
  try {
    for (let epoch = 0; epoch < fineTuneEpochs; epoch += 1) {
      await session.trainSequence(orderedEpoch(warmupEpochs + epoch), {
        learningRate: .00012,
        metrics: "none",
      });
      onProgress?.(
        warmupEpochs + epoch + 1,
        totalEpochs,
        "MLP + readout",
      );
    }
    const trainingAccuracy = await evaluateMnist(session, dataset.training);
    const testAccuracy = await evaluateMnist(session, dataset.test);
    const snapshot = await session.snapshot();
    const kernel = Object.fromEntries(
      Object.entries(fineTune.kernelParameters).map(([name, parameter]) => [
        name,
        snapshot.parameters[parameter] ?? 0,
      ]),
    );
    const weights = extractAutomataNetworkWeights(
      snapshot,
      fineTune.networkParameters,
    );
    const examples = Array.from(
      { length: 10 },
      (_, label) => dataset.test.find((example) => example.label === label),
    ).filter(Boolean);
    const predictions = await session.forwardSequence(
      examples.map(({ input: values }) => values),
    );
    return {
      backend: session.backend,
      durationMs: Math.round(performance.now() - started),
      epochs: totalEpochs,
      examples: examples.map((example, index) => ({
        ...example,
        prediction: argmax(predictions[index]?.data ?? []),
      })),
      kernel,
      testAccuracy,
      trainingAccuracy,
      weights,
    };
  } finally {
    session.dispose();
  }
}

export const DSR_TARGETS = [2, 4];
export const DSR_DISTRACTORS = [3, 5];
export const DSR_PROMPTS = [0, 1];
const DSR_SYMBOL_COUNT =
  DSR_TARGETS.length + DSR_DISTRACTORS.length + DSR_PROMPTS.length;

const seededRandom = (seed) => {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
};

export function createDsrSequence(random = Math.random, length = 10) {
  const recallLength = length - DSR_PROMPTS.length;
  const symbols = Array.from(
    { length: recallLength },
    () => DSR_DISTRACTORS[
      Math.floor(random() * DSR_DISTRACTORS.length)
    ],
  );
  const targetIndexes = [];
  const positions = [];
  for (let index = 0; index < DSR_PROMPTS.length; index += 1) {
    targetIndexes.push(Math.floor(random() * DSR_TARGETS.length));
    let position = Math.floor(random() * recallLength);
    while (positions.includes(position)) {
      position = Math.floor(random() * recallLength);
    }
    positions.push(position);
  }
  positions.sort((left, right) => left - right);
  for (let index = 0; index < DSR_PROMPTS.length; index += 1) {
    symbols[positions[index]] = DSR_TARGETS[targetIndexes[index]];
    symbols.push(DSR_PROMPTS[index]);
  }
  return symbols.map((symbol, step) => {
    const inputValues = new Array(DSR_SYMBOL_COUNT).fill(0);
    const target = new Array(DSR_TARGETS.length).fill(0);
    inputValues[symbol] = 1;
    const targetIndex = step >= recallLength
      ? targetIndexes[step - recallLength]
      : undefined;
    if (targetIndex !== undefined) {
      target[targetIndex] = 1;
    }
    return {
      expectedSymbol:
        targetIndex === undefined ? undefined : DSR_TARGETS[targetIndex],
      input: inputValues,
      symbol,
      target,
    };
  });
}

export async function trainDsrDemo(onProgress) {
  const definition = sequential(
    input({ size: DSR_SYMBOL_COUNT }),
    lstm({
      units: 4,
      forgetBias: 3,
      label: "dsr.memory",
    }),
    dense({
      units: DSR_TARGETS.length,
      activation: "logistic",
      label: "dsr.output",
    }),
  );
  const snapshot = createSnapshot(definition, 1);
  for (const parameter of definition.topology.parameters) {
    if (parameter.initializer.kind === "uniform") {
      snapshot.parameters[parameter.id] =
        (snapshot.parameters[parameter.id] ?? 0) * 5;
    }
  }
  const session = await createBackend().compile(
    compilePlan(definition),
    snapshot,
    { training: true },
  );
  const curriculumTrials = 1_375;
  const trainingTrials = 9_625;
  const random = seededRandom(0x1234_abcd);
  const started = performance.now();
  for (let trial = 0; trial < trainingTrials; trial += 1) {
    await session.resetState();
    const length = 4 + Math.min(6, Math.floor(trial / curriculumTrials));
    const sequence = createDsrSequence(random, length);
    await session.trainSequence(
      sequence.map((sample, step) => ({
        input: sample.input,
        target: sample.target,
        learningRate: step < length - 2 ? .003 : .03,
      })),
      { metrics: "none" },
    );
    if ((trial + 1) % 125 === 0 || trial + 1 === trainingTrials) {
      onProgress?.(trial + 1, trainingTrials, length);
    }
  }
  await session.snapshot();
  return {
    backend: session.backend,
    durationMs: Math.round(performance.now() - started),
    session,
    trainingTrials,
  };
}
