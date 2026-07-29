import {
  compilePlan,
  createSnapshot,
  type Backend,
  type Session,
  type TrainingBatch,
} from "@synaptic/core";
import { dense, input, lstm, sequential } from "@synaptic/layers";

export interface LearningProgress {
  readonly completed: number;
  readonly phase: "evaluate" | "train";
  readonly total: number;
  readonly workload: "dsr" | "mnist" | "xor";
}

export type LearningProgressCallback = (progress: LearningProgress) => void;

export interface XorLearningOptions {
  readonly epochs?: number;
  readonly hiddenUnits?: number;
  readonly learningRate?: number;
  readonly onProgress?: LearningProgressCallback;
  readonly seed?: number;
}

export interface XorLearningResult {
  readonly accuracy: number;
  readonly backend: string;
  readonly epochs: number;
  readonly outputs: readonly number[];
}

export interface MnistDigitSource {
  readonly length: number;
  get(index: number): ArrayLike<number>;
}

export type MnistSource = readonly MnistDigitSource[];

export interface MnistExample extends TrainingBatch {
  readonly label: number;
}

export interface MnistDataset {
  readonly imageSize: number;
  readonly test: readonly MnistExample[];
  readonly training: readonly MnistExample[];
}

export interface MnistDatasetOptions {
  readonly imageSize?: number;
  readonly testPerDigit?: number;
  readonly trainingPerDigit?: number;
}

export interface MnistLearningOptions {
  readonly epochs?: number;
  readonly hiddenUnits?: number;
  readonly learningRate?: number;
  readonly onProgress?: LearningProgressCallback;
  readonly seed?: number;
}

export interface MnistLearningResult {
  readonly backend: string;
  readonly epochs: number;
  readonly testAccuracy: number;
  readonly trainingAccuracy: number;
}

export interface DsrLearningOptions {
  readonly curriculumTrialsPerLength?: number;
  readonly distractorLearningRate?: number;
  readonly forgetBias?: number;
  readonly hiddenUnits?: number;
  readonly initialWeightScale?: number;
  readonly learningRate?: number;
  readonly onProgress?: LearningProgressCallback;
  readonly seed?: number;
  readonly sequenceLength?: number;
  readonly trainingTrials?: number;
  readonly validationSeed?: number;
  readonly validationTrials?: number;
}

export interface DsrLearningResult {
  readonly backend: string;
  readonly distractorAccuracy: number;
  readonly promptAccuracy: number;
  readonly sequenceAccuracy: number;
  readonly stepAccuracy: number;
  readonly trainingTrials: number;
  readonly validationTrials: number;
}

const XOR_SAMPLES = [
  { input: [0, 0], target: [0] },
  { input: [0, 1], target: [1] },
  { input: [1, 0], target: [1] },
  { input: [1, 1], target: [0] },
] as const satisfies readonly TrainingBatch[];

const DSR_TARGETS = [2, 4] as const;
const DSR_DISTRACTORS = [3, 5] as const;
const DSR_PROMPTS = [0, 1] as const;
const DSR_SYMBOL_COUNT =
  DSR_TARGETS.length + DSR_DISTRACTORS.length + DSR_PROMPTS.length;

function notify(
  callback: LearningProgressCallback | undefined,
  workload: LearningProgress["workload"],
  phase: LearningProgress["phase"],
  completed: number,
  total: number,
): void {
  callback?.({ completed, phase, total, workload });
}

function greatestCommonDivisor(left: number, right: number): number {
  let a = Math.abs(left);
  let b = Math.abs(right);
  while (b !== 0) {
    const remainder = a % b;
    a = b;
    b = remainder;
  }
  return a;
}

function coprimeStride(length: number): number {
  let stride = Math.min(7, Math.max(1, length - 1));
  while (greatestCommonDivisor(stride, length) !== 1) {
    stride += 1;
  }
  return stride;
}

function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (
      (values[index] ?? Number.NEGATIVE_INFINITY)
      > (values[best] ?? Number.NEGATIVE_INFINITY)
    ) {
      best = index;
    }
  }
  return best;
}

function isRoundedMatch(
  output: ArrayLike<number>,
  target: ArrayLike<number>,
): boolean {
  if (output.length !== target.length) {
    return false;
  }
  for (let index = 0; index < output.length; index += 1) {
    if (Math.round(output[index] ?? 0) !== (target[index] ?? 0)) {
      return false;
    }
  }
  return true;
}

async function evaluateMnist(
  session: Session,
  examples: readonly MnistExample[],
): Promise<number> {
  let correct = 0;
  const outputs = await session.forwardSequence(
    examples.map(({ input }) => input),
  );
  for (let index = 0; index < examples.length; index += 1) {
    const example = examples[index];
    const output = outputs[index];
    if (example === undefined || output === undefined) {
      continue;
    }
    correct += argmax(output.data) === example.label ? 1 : 0;
  }
  return correct / examples.length;
}

function downsampleMnist(
  image: ArrayLike<number>,
  imageSize: number,
): number[] {
  if (image.length !== 28 * 28) {
    throw new RangeError(
      `MNIST images must contain 784 pixels, received ${image.length}`,
    );
  }
  if (!Number.isInteger(imageSize) || imageSize <= 0 || 28 % imageSize !== 0) {
    throw new RangeError("MNIST imageSize must be a positive divisor of 28");
  }
  const scale = 28 / imageSize;
  const result = new Array<number>(imageSize * imageSize).fill(0);
  for (let row = 0; row < imageSize; row += 1) {
    for (let column = 0; column < imageSize; column += 1) {
      let sum = 0;
      for (let y = 0; y < scale; y += 1) {
        for (let x = 0; x < scale; x += 1) {
          const source = (row * scale + y) * 28 + column * scale + x;
          sum += image[source] ?? 0;
        }
      }
      result[row * imageSize + column] = sum / (scale * scale);
    }
  }
  return result;
}

function createTarget(label: number): number[] {
  const target = new Array<number>(10).fill(0);
  target[label] = 1;
  return target;
}

function requirePositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive integer`);
  }
}

export function createMnistDataset(
  source: MnistSource,
  options: MnistDatasetOptions = {},
): MnistDataset {
  if (source.length !== 10) {
    throw new RangeError(
      `MNIST source must contain 10 digits, received ${source.length}`,
    );
  }
  const imageSize = options.imageSize ?? 7;
  const trainingPerDigit = options.trainingPerDigit ?? 15;
  const testPerDigit = options.testPerDigit ?? 5;
  requirePositiveInteger(trainingPerDigit, "trainingPerDigit");
  requirePositiveInteger(testPerDigit, "testPerDigit");

  const training: MnistExample[] = [];
  const test: MnistExample[] = [];
  source.forEach((digit, label) => {
    if (digit.length < trainingPerDigit + testPerDigit) {
      throw new RangeError(`MNIST digit ${label} does not contain enough samples`);
    }
    const target = createTarget(label);
    for (let index = 0; index < trainingPerDigit; index += 1) {
      training.push({
        input: downsampleMnist(digit.get(index), imageSize),
        label,
        target,
      });
    }
    for (let index = 0; index < testPerDigit; index += 1) {
      test.push({
        input: downsampleMnist(
          digit.get(trainingPerDigit + index),
          imageSize,
        ),
        label,
        target,
      });
    }
  });
  return { imageSize, test, training };
}

export async function runXorLearning(
  backend: Backend,
  options: XorLearningOptions = {},
): Promise<XorLearningResult> {
  const epochs = options.epochs ?? 2_500;
  const hiddenUnits = options.hiddenUnits ?? 3;
  const learningRate = options.learningRate ?? 0.3;
  requirePositiveInteger(epochs, "epochs");
  requirePositiveInteger(hiddenUnits, "hiddenUnits");
  const definition = sequential(
    input({ size: 2 }),
    dense({ units: hiddenUnits, activation: "logistic", label: "xor.hidden" }),
    dense({ units: 1, activation: "logistic", label: "xor.output" }),
  );
  const session = await backend.compile(
    compilePlan(definition),
    createSnapshot(definition, options.seed ?? 0xfeed_1234),
    { training: true },
  );
  try {
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const sequence = XOR_SAMPLES.map(
        (_, offset) => XOR_SAMPLES[(epoch + offset) % XOR_SAMPLES.length],
      ).filter((sample) => sample !== undefined);
      await session.trainSequence(sequence, {
        learningRate,
        metrics: "none",
      });
      if ((epoch + 1) % 100 === 0 || epoch + 1 === epochs) {
        notify(options.onProgress, "xor", "train", epoch + 1, epochs);
      }
    }

    const predicted = await session.forwardSequence(
      XOR_SAMPLES.map(({ input }) => input),
    );
    const outputs = predicted.map((output) => output.data[0] ?? 0);
    let correct = 0;
    for (let index = 0; index < XOR_SAMPLES.length; index += 1) {
      const sample = XOR_SAMPLES[index];
      const value = outputs[index] ?? 0;
      if (sample === undefined) {
        continue;
      }
      correct += Math.round(value) === sample.target[0] ? 1 : 0;
    }
    notify(
      options.onProgress,
      "xor",
      "evaluate",
      XOR_SAMPLES.length,
      XOR_SAMPLES.length,
    );
    return {
      accuracy: correct / XOR_SAMPLES.length,
      backend: session.backend,
      epochs,
      outputs,
    };
  } finally {
    session.dispose();
  }
}

export async function runMnistLearning(
  backend: Backend,
  dataset: MnistDataset,
  options: MnistLearningOptions = {},
): Promise<MnistLearningResult> {
  const epochs = options.epochs ?? 30;
  const hiddenUnits = options.hiddenUnits ?? 24;
  const learningRate = options.learningRate ?? 0.08;
  requirePositiveInteger(epochs, "epochs");
  requirePositiveInteger(hiddenUnits, "hiddenUnits");
  if (dataset.training.length === 0 || dataset.test.length === 0) {
    throw new RangeError("MNIST training and test sets must not be empty");
  }
  const definition = sequential(
    input({ size: dataset.imageSize * dataset.imageSize }),
    dense({ units: hiddenUnits, activation: "logistic", label: "mnist.hidden" }),
    dense({ units: 10, activation: "logistic", label: "mnist.output" }),
  );
  const session = await backend.compile(
    compilePlan(definition),
    createSnapshot(definition, options.seed ?? 0x6d6e_6973),
    { training: true },
  );
  try {
    const stride = coprimeStride(dataset.training.length);
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      const start = (epoch * 13) % dataset.training.length;
      const sequence: MnistExample[] = [];
      for (let offset = 0; offset < dataset.training.length; offset += 1) {
        const index = (start + offset * stride) % dataset.training.length;
        const example = dataset.training[index];
        if (example !== undefined) {
          sequence.push(example);
        }
      }
      await session.trainSequence(sequence, {
        learningRate,
        metrics: "none",
      });
      notify(options.onProgress, "mnist", "train", epoch + 1, epochs);
    }
    const trainingAccuracy = await evaluateMnist(session, dataset.training);
    const testAccuracy = await evaluateMnist(session, dataset.test);
    notify(
      options.onProgress,
      "mnist",
      "evaluate",
      dataset.training.length + dataset.test.length,
      dataset.training.length + dataset.test.length,
    );
    return {
      backend: session.backend,
      epochs,
      testAccuracy,
      trainingAccuracy,
    };
  } finally {
    session.dispose();
  }
}

class DsrRandom {
  #state: number;

  constructor(seed: number) {
    this.#state = seed >>> 0;
  }

  next(): number {
    this.#state =
      (Math.imul(this.#state, 1_664_525) + 1_013_904_223) >>> 0;
    return this.#state / 0x1_0000_0000;
  }
}

function createDsrSequence(
  random: DsrRandom,
  sequenceLength: number,
): readonly TrainingBatch[] {
  const recallLength = sequenceLength - DSR_PROMPTS.length;
  const symbols: number[] = Array.from(
    { length: recallLength },
    () => DSR_DISTRACTORS[Math.floor(random.next() * DSR_DISTRACTORS.length)] ?? 0,
  );
  const targetIndexes: number[] = [];
  const positions: number[] = [];
  for (let index = 0; index < DSR_PROMPTS.length; index += 1) {
    targetIndexes.push(Math.floor(random.next() * DSR_TARGETS.length));
    let position = Math.floor(random.next() * recallLength);
    while (positions.includes(position)) {
      position = Math.floor(random.next() * recallLength);
    }
    positions.push(position);
  }
  positions.sort((left, right) => left - right);
  for (let index = 0; index < DSR_PROMPTS.length; index += 1) {
    const position = positions[index];
    const targetIndex = targetIndexes[index];
    if (position !== undefined && targetIndex !== undefined) {
      symbols[position] = DSR_TARGETS[targetIndex] ?? 0;
    }
    symbols.push(DSR_PROMPTS[index] ?? 0);
  }

  return symbols.map((symbol, step) => {
    const input = new Array<number>(DSR_SYMBOL_COUNT).fill(0);
    const target = new Array<number>(DSR_TARGETS.length).fill(0);
    input[symbol] = 1;
    if (step >= recallLength) {
      const targetIndex = targetIndexes[step - recallLength];
      if (targetIndex !== undefined) {
        target[targetIndex] = 1;
      }
    }
    return { input, target };
  });
}

export async function runDsrLearning(
  backend: Backend,
  options: DsrLearningOptions = {},
): Promise<DsrLearningResult> {
  const sequenceLength = options.sequenceLength ?? 10;
  const trainingTrials = options.trainingTrials ?? 8_250;
  const validationTrials = options.validationTrials ?? 100;
  const curriculumTrialsPerLength =
    options.curriculumTrialsPerLength ?? 1_375;
  const hiddenUnits = options.hiddenUnits ?? 4;
  const learningRate = options.learningRate ?? 0.03;
  const distractorLearningRate = options.distractorLearningRate ?? 0.003;
  requirePositiveInteger(trainingTrials, "trainingTrials");
  requirePositiveInteger(validationTrials, "validationTrials");
  requirePositiveInteger(curriculumTrialsPerLength, "curriculumTrialsPerLength");
  requirePositiveInteger(hiddenUnits, "hiddenUnits");
  if (sequenceLength < 4) {
    throw new RangeError("DSR sequenceLength must be at least 4");
  }

  const definition = sequential(
    input({ size: DSR_SYMBOL_COUNT }),
    lstm({
      units: hiddenUnits,
      forgetBias: options.forgetBias ?? 3,
      label: "dsr.memory",
    }),
    dense({
      units: DSR_TARGETS.length,
      activation: "logistic",
      label: "dsr.output",
    }),
  );
  const snapshot = createSnapshot(definition, options.seed ?? 1);
  const initialWeightScale = options.initialWeightScale ?? 5;
  if (!Number.isFinite(initialWeightScale) || initialWeightScale <= 0) {
    throw new RangeError("initialWeightScale must be a positive finite number");
  }
  if (initialWeightScale !== 1) {
    for (const parameter of definition.topology.parameters) {
      if (parameter.initializer.kind === "uniform") {
        snapshot.parameters[parameter.id] =
          (snapshot.parameters[parameter.id] ?? 0) * initialWeightScale;
      }
    }
  }
  const session = await backend.compile(
    compilePlan(definition),
    snapshot,
    { training: true },
  );
  try {
    const trainingRandom = new DsrRandom(0x1234_abcd);
    for (let trial = 0; trial < trainingTrials; trial += 1) {
      await session.resetState();
      const currentLength =
        4 +
        Math.min(
          sequenceLength - 4,
          Math.floor(trial / curriculumTrialsPerLength),
        );
      const sequence = createDsrSequence(trainingRandom, currentLength);
      await session.trainSequence(
        sequence.map((sample, step) => ({
          ...sample,
          learningRate:
            step < currentLength - DSR_PROMPTS.length
              ? distractorLearningRate
              : learningRate,
        })),
        { metrics: "none" },
      );
      if ((trial + 1) % 250 === 0 || trial + 1 === trainingTrials) {
        notify(
          options.onProgress,
          "dsr",
          "train",
          trial + 1,
          trainingTrials,
        );
      }
    }

    const validationRandom = new DsrRandom(
      options.validationSeed ?? 0xdeca_fbad,
    );
    let correctDistractors = 0;
    let correctPrompts = 0;
    let correctSequences = 0;
    let correctSteps = 0;
    for (let trial = 0; trial < validationTrials; trial += 1) {
      await session.resetState();
      let sequenceCorrect = true;
      const sequence = createDsrSequence(validationRandom, sequenceLength);
      const outputs = await session.forwardSequence(
        sequence.map(({ input }) => input),
      );
      for (let step = 0; step < sequence.length; step += 1) {
        const sample = sequence[step];
        const output = outputs[step]?.data;
        if (sample === undefined || output === undefined) {
          continue;
        }
        const correct = isRoundedMatch(output, sample.target);
        correctSteps += correct ? 1 : 0;
        sequenceCorrect &&= correct;
        if (step < sequenceLength - DSR_PROMPTS.length) {
          correctDistractors += correct ? 1 : 0;
        } else {
          correctPrompts += argmax(output) === argmax(sample.target) ? 1 : 0;
        }
      }
      correctSequences += sequenceCorrect ? 1 : 0;
      if ((trial + 1) % 10 === 0 || trial + 1 === validationTrials) {
        notify(
          options.onProgress,
          "dsr",
          "evaluate",
          trial + 1,
          validationTrials,
        );
      }
    }
    const distractorSteps =
      validationTrials * (sequenceLength - DSR_PROMPTS.length);
    const promptSteps = validationTrials * DSR_PROMPTS.length;
    return {
      backend: session.backend,
      distractorAccuracy: correctDistractors / distractorSteps,
      promptAccuracy: correctPrompts / promptSteps,
      sequenceAccuracy: correctSequences / validationTrials,
      stepAccuracy: correctSteps / (validationTrials * sequenceLength),
      trainingTrials,
      validationTrials,
    };
  } finally {
    session.dispose();
  }
}
