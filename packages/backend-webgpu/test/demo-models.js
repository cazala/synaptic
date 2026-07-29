import { WebGpuBackend } from "@synaptic/backend-webgpu";
import { compilePlan, createSnapshot } from "@synaptic/core";
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
    dense({ units: 48, activation: "logistic", label: "mnist.hidden" }),
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
