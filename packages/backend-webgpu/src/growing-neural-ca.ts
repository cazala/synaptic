import { SynapticError, invariant } from "@synaptic/core";

export const GROWING_NEURAL_CA_FORMAT =
  "@cazala/automata/growing-neural-ca" as const;
export const GROWING_NEURAL_CA_VERSION = 1 as const;

const BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;
const MAP_READ = 0x0001;
const COMPUTE_STAGE = 0x0004;
const WORKGROUP_SIZE = 64;
const UNIFORM_STRIDE = 256;

export interface GrowingNeuralCaArtifact {
  readonly format: typeof GROWING_NEURAL_CA_FORMAT;
  readonly version: typeof GROWING_NEURAL_CA_VERSION;
  readonly channels: number;
  readonly hidden: number;
  readonly perception: readonly ["identity", "sobel-x", "sobel-y"];
  readonly activation: "relu";
  readonly fireRate: number;
  readonly stepSize: number;
  readonly boundary: "zero";
  readonly life: Readonly<{
    channel: 3;
    threshold: number;
    neighborhood: 3;
    preAndPost: true;
  }>;
  readonly weights: Readonly<{
    inputToHidden: readonly number[];
    hiddenBias: readonly number[];
    hiddenToOutput: readonly number[];
    outputBias: readonly number[];
  }>;
}

export interface GrowingNeuralCaTrainerOptions {
  readonly gpu?: GPU;
  readonly powerPreference?: GPUPowerPreference;
  readonly size?: number;
  readonly channels?: number;
  readonly hidden?: number;
  readonly batchSize?: number;
  /** Maximum BPTT horizon. */
  readonly rolloutSteps?: number;
  /** Minimum sampled BPTT horizon; defaults to `rolloutSteps`. */
  readonly minRolloutSteps?: number;
  /** Number of consecutive terminal states included in the image loss. */
  readonly stabilitySteps?: number;
  readonly fireRate?: number;
  readonly stepSize?: number;
  readonly aliveThreshold?: number;
  readonly learningRate?: number;
  readonly seed?: number;
  readonly poolSize?: number;
  readonly poolWarmupIterations?: number;
  readonly poolValueLimit?: number;
  readonly damageProbability?: number;
  readonly initialArtifact?: GrowingNeuralCaArtifact;
}

export interface GrowingNeuralCaTrainOptions {
  readonly learningRate?: number;
  readonly damageProbability?: number;
}

export interface GrowingNeuralCaMetrics {
  readonly iteration: number;
  readonly loss: number;
  readonly rolloutSteps: number;
  readonly durationMs: number;
  readonly damagedSamples: number;
}

export interface GrowingNeuralCaStateHealth {
  readonly loss: number;
  readonly maximumAbsoluteValue: number;
  readonly aliveCells: number;
  readonly nonFiniteValues: number;
}

export interface GrowingNeuralCaHeapSection {
  readonly offset: number;
  readonly length: number;
  readonly byteOffset: number;
  readonly byteLength: number;
}

export type GrowingNeuralCaHeapSectionName =
  | "weights"
  | "adamFirst"
  | "adamSecond"
  | "gradient"
  | "stateTape"
  | "candidate"
  | "life"
  | "gradientA"
  | "gradientB"
  | "perception"
  | "hidden"
  | "delta"
  | "hiddenGradient"
  | "candidateGradient"
  | "target"
  | "initial"
  | "loss"
  | "metrics";

export interface GrowingNeuralCaHeapLayout {
  readonly byteLength: number;
  readonly parameterCount: number;
  readonly sections: Readonly<
    Record<GrowingNeuralCaHeapSectionName, GrowingNeuralCaHeapSection>
  >;
}

interface ResolvedOptions {
  readonly gpu: GPU;
  readonly powerPreference: GPUPowerPreference | undefined;
  readonly size: number;
  readonly channels: number;
  readonly hidden: number;
  readonly batchSize: number;
  readonly rolloutSteps: number;
  readonly minRolloutSteps: number;
  readonly stabilitySteps: number;
  readonly fireRate: number;
  readonly stepSize: number;
  readonly aliveThreshold: number;
  readonly learningRate: number;
  readonly seed: number;
  readonly poolSize: number;
  readonly poolWarmupIterations: number;
  readonly poolValueLimit: number;
  readonly damageProbability: number;
  readonly initialArtifact: GrowingNeuralCaArtifact | undefined;
}

interface GrowingNeuralCaPipelines {
  readonly forwardCandidate: GPUComputePipeline;
  readonly forwardLife: GPUComputePipeline;
  readonly initializeLoss: GPUComputePipeline;
  readonly addStabilityLoss: GPUComputePipeline;
  readonly reduceLoss: GPUComputePipeline;
  readonly backwardMlp: GPUComputePipeline;
  readonly backwardPerception: GPUComputePipeline;
  readonly parameterGradient: GPUComputePipeline;
  readonly normalizeGradient: GPUComputePipeline;
  readonly updateAdam: GPUComputePipeline;
}

interface ParameterLayout {
  readonly inputToHidden: number;
  readonly hiddenBias: number;
  readonly hiddenToOutput: number;
  readonly outputBias: number;
  readonly count: number;
}

function integer(
  name: string,
  value: number,
  minimum: number,
  maximum: number,
): number {
  invariant(
    Number.isInteger(value) && value >= minimum && value <= maximum,
    `${name} must be an integer between ${minimum} and ${maximum}`,
    "INVALID_GROWING_CA_OPTION",
  );
  return value;
}

function finiteRange(
  name: string,
  value: number,
  minimum: number,
  maximum: number,
): number {
  invariant(
    Number.isFinite(value) && value >= minimum && value <= maximum,
    `${name} must be between ${minimum} and ${maximum}`,
    "INVALID_GROWING_CA_OPTION",
  );
  return value;
}

function align(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function parameterLayout(channels: number, hidden: number): ParameterLayout {
  const perception = channels * 3;
  const inputToHidden = 0;
  const hiddenBias = inputToHidden + hidden * perception;
  const hiddenToOutput = hiddenBias + hidden;
  const outputBias = hiddenToOutput + channels * hidden;
  return {
    inputToHidden,
    hiddenBias,
    hiddenToOutput,
    outputBias,
    count: outputBias + channels,
  };
}

export function buildGrowingNeuralCaHeapLayout(options: {
  readonly size: number;
  readonly channels: number;
  readonly hidden: number;
  readonly batchSize: number;
  readonly rolloutSteps: number;
}): GrowingNeuralCaHeapLayout {
  const size = integer("size", options.size, 8, 64);
  const channels = integer("channels", options.channels, 4, 32);
  const hidden = integer("hidden", options.hidden, 1, 128);
  const batch = integer("batchSize", options.batchSize, 1, 8);
  const steps = integer("rolloutSteps", options.rolloutSteps, 1, 96);
  const cells = size * size;
  const batchCells = batch * cells;
  const parameters = parameterLayout(channels, hidden).count;
  const perception = channels * 3;
  const lengths: Readonly<Record<GrowingNeuralCaHeapSectionName, number>> = {
    weights: parameters,
    adamFirst: parameters,
    adamSecond: parameters,
    gradient: parameters,
    stateTape: (steps + 1) * batchCells * channels,
    candidate: batchCells * channels,
    life: steps * batchCells,
    gradientA: batchCells * channels,
    gradientB: batchCells * channels,
    // Forward values and their gradients share one section.
    perception: batchCells * perception * 2,
    hidden: batchCells * hidden,
    delta: batchCells * channels,
    hiddenGradient: batchCells * hidden,
    candidateGradient: batchCells * channels,
    target: cells * 4,
    initial: batchCells * channels,
    loss: batchCells * 4,
    // Loss plus one gradient-normalization scale for each trainable tensor.
    metrics: 5,
  };
  const sections = {} as Record<
    GrowingNeuralCaHeapSectionName,
    GrowingNeuralCaHeapSection
  >;
  let cursor = 0;
  for (const name of Object.keys(lengths) as GrowingNeuralCaHeapSectionName[]) {
    cursor = align(cursor, 4);
    const length = lengths[name];
    sections[name] = {
      offset: cursor / 4,
      length,
      byteOffset: cursor,
      byteLength: length * 4,
    };
    cursor += length * 4;
  }
  return Object.freeze({
    byteLength: align(cursor, 4),
    parameterCount: parameters,
    sections: Object.freeze(sections),
  });
}

export function measureGrowingNeuralCaState(
  state: ArrayLike<number>,
  target: ArrayLike<number>,
  channels: number,
  aliveThreshold = 0.1,
): GrowingNeuralCaStateHealth {
  const channelCount = integer("channels", channels, 4, 32);
  finiteRange("aliveThreshold", aliveThreshold, 0, 1);
  invariant(
    target.length % 4 === 0 &&
      state.length === (target.length / 4) * channelCount,
    `State and target dimensions do not match ${channelCount} channels`,
    "INVALID_GROWING_CA_STATE",
  );
  let squaredError = 0;
  let maximumAbsoluteValue = 0;
  let aliveCells = 0;
  let nonFiniteValues = 0;
  for (let cell = 0; cell < target.length / 4; cell += 1) {
    const stateOffset = cell * channelCount;
    const targetOffset = cell * 4;
    const alpha = state[stateOffset + 3];
    if (alpha !== undefined && Number.isFinite(alpha) && alpha > aliveThreshold) {
      aliveCells += 1;
    }
    for (let channel = 0; channel < channelCount; channel += 1) {
      const value = state[stateOffset + channel];
      if (value === undefined || !Number.isFinite(value)) {
        nonFiniteValues += 1;
        continue;
      }
      maximumAbsoluteValue = Math.max(maximumAbsoluteValue, Math.abs(value));
      if (channel < 4) {
        const expected = target[targetOffset + channel];
        if (expected === undefined || !Number.isFinite(expected)) {
          nonFiniteValues += 1;
        } else {
          const difference = value - expected;
          squaredError += difference * difference;
        }
      }
    }
  }
  return Object.freeze({
    loss: nonFiniteValues === 0
      ? squaredError / target.length
      : Number.POSITIVE_INFINITY,
    maximumAbsoluteValue,
    aliveCells,
    nonFiniteValues,
  });
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function copyNumbers(
  name: string,
  values: ArrayLike<number>,
  expected: number,
): Float32Array {
  invariant(
    values.length === expected,
    `${name} has length ${values.length}; expected ${expected}`,
    "INVALID_GROWING_CA_ARTIFACT",
  );
  const copy = new Float32Array(expected);
  for (let index = 0; index < expected; index += 1) {
    const value = values[index];
    invariant(
      value !== undefined && Number.isFinite(value),
      `${name}[${index}] must be finite`,
      "INVALID_GROWING_CA_ARTIFACT",
    );
    copy[index] = value;
  }
  return copy;
}

function validateArtifact(
  artifact: GrowingNeuralCaArtifact,
  channels: number,
  hidden: number,
): Float32Array {
  invariant(
    artifact.format === GROWING_NEURAL_CA_FORMAT &&
      artifact.version === GROWING_NEURAL_CA_VERSION,
    "Unsupported Growing Neural CA artifact format or version",
    "INVALID_GROWING_CA_ARTIFACT",
  );
  invariant(
    artifact.channels === channels &&
      artifact.hidden === hidden &&
      artifact.activation === "relu" &&
      artifact.boundary === "zero" &&
      artifact.life.channel === 3 &&
      artifact.life.neighborhood === 3 &&
      artifact.life.preAndPost === true &&
      artifact.perception.join(",") === "identity,sobel-x,sobel-y",
    "Growing Neural CA artifact semantics or dimensions do not match the trainer",
    "INVALID_GROWING_CA_ARTIFACT",
  );
  const layout = parameterLayout(channels, hidden);
  const weights = new Float32Array(layout.count);
  weights.set(
    copyNumbers(
      "inputToHidden",
      artifact.weights.inputToHidden,
      hidden * channels * 3,
    ),
    layout.inputToHidden,
  );
  weights.set(
    copyNumbers("hiddenBias", artifact.weights.hiddenBias, hidden),
    layout.hiddenBias,
  );
  weights.set(
    copyNumbers(
      "hiddenToOutput",
      artifact.weights.hiddenToOutput,
      channels * hidden,
    ),
    layout.hiddenToOutput,
  );
  weights.set(
    copyNumbers("outputBias", artifact.weights.outputBias, channels),
    layout.outputBias,
  );
  return weights;
}

export function createGrowingNeuralCaWeights(options: {
  readonly channels?: number;
  readonly hidden?: number;
  readonly seed?: number;
} = {}): Float32Array {
  const channels = integer("channels", options.channels ?? 16, 4, 32);
  const hidden = integer("hidden", options.hidden ?? 64, 1, 128);
  const layout = parameterLayout(channels, hidden);
  const weights = new Float32Array(layout.count);
  const random = mulberry32(options.seed ?? 0x6e63_6102);
  const scale = Math.sqrt(6 / (channels * 3 + hidden));
  for (let index = 0; index < hidden * channels * 3; index += 1) {
    weights[layout.inputToHidden + index] = (random() * 2 - 1) * scale;
  }
  // The final layer and both biases deliberately start at zero, matching the
  // reference implementation's stable identity initialization.
  return weights;
}

function resolveOptions(options: GrowingNeuralCaTrainerOptions): ResolvedOptions {
  const installed =
    typeof navigator !== "undefined" && "gpu" in navigator
      ? navigator.gpu
      : undefined;
  const gpu = options.gpu ?? installed;
  invariant(
    gpu !== undefined,
    "WebGPU is required for Growing Neural CA training",
    "WEBGPU_UNAVAILABLE",
  );
  const artifact = options.initialArtifact;
  const channels = integer(
    "channels",
    options.channels ?? artifact?.channels ?? 16,
    4,
    32,
  );
  const hidden = integer(
    "hidden",
    options.hidden ?? artifact?.hidden ?? 64,
    1,
    128,
  );
  const rolloutSteps = integer(
    "rolloutSteps",
    options.rolloutSteps ?? 24,
    1,
    96,
  );
  const minRolloutSteps = integer(
    "minRolloutSteps",
    options.minRolloutSteps ?? rolloutSteps,
    1,
    rolloutSteps,
  );
  const stabilitySteps = integer(
    "stabilitySteps",
    options.stabilitySteps ?? 1,
    1,
    minRolloutSteps,
  );
  return {
    gpu,
    powerPreference: options.powerPreference,
    size: integer("size", options.size ?? 24, 8, 64),
    channels,
    hidden,
    batchSize: integer("batchSize", options.batchSize ?? 2, 1, 8),
    rolloutSteps,
    minRolloutSteps,
    stabilitySteps,
    fireRate: finiteRange(
      "fireRate",
      options.fireRate ?? artifact?.fireRate ?? 0.5,
      0,
      1,
    ),
    stepSize: finiteRange(
      "stepSize",
      options.stepSize ?? artifact?.stepSize ?? 1,
      0,
      2,
    ),
    aliveThreshold: finiteRange(
      "aliveThreshold",
      options.aliveThreshold ?? artifact?.life.threshold ?? 0.1,
      0,
      1,
    ),
    learningRate: finiteRange(
      "learningRate",
      options.learningRate ?? 0.002,
      0,
      1,
    ),
    seed: options.seed ?? 0x6e63_6102,
    poolSize: integer(
      "poolSize",
      options.poolSize ?? Math.max(8, (options.batchSize ?? 2) * 4),
      1,
      256,
    ),
    poolWarmupIterations: integer(
      "poolWarmupIterations",
      options.poolWarmupIterations ?? 128,
      0,
      10_000,
    ),
    poolValueLimit: finiteRange(
      "poolValueLimit",
      options.poolValueLimit ?? 256,
      1,
      1_000,
    ),
    damageProbability: finiteRange(
      "damageProbability",
      options.damageProbability ?? 0.35,
      0,
      1,
    ),
    initialArtifact: artifact,
  };
}

function dispatchCount(length: number): number {
  return Math.max(1, Math.ceil(length / WORKGROUP_SIZE));
}

function seedState(size: number, channels: number): Float32Array {
  const state = new Float32Array(size * size * channels);
  const center = (Math.floor(size / 2) * size + Math.floor(size / 2)) * channels;
  for (let channel = 3; channel < channels; channel += 1) {
    state[center + channel] = 1;
  }
  return state;
}

function shader(
  options: ResolvedOptions,
  heap: GrowingNeuralCaHeapLayout,
): string {
  const { size, channels, hidden, batchSize } = options;
  const cells = size * size;
  const batchCells = cells * batchSize;
  const perception = channels * 3;
  const parameters = parameterLayout(channels, hidden);
  const o = (name: GrowingNeuralCaHeapSectionName): number =>
    heap.sections[name].offset;
  return /* wgsl */ `
struct Dispatch {
  time: u32,
  iteration: u32,
  activeSteps: u32,
  learningRate: f32,
};

@group(0) @binding(0) var<uniform> dispatch: Dispatch;
@group(0) @binding(1) var<storage, read_write> heap: array<f32>;

const SIZE: u32 = ${size}u;
const CELLS: u32 = ${cells}u;
const CHANNELS: u32 = ${channels}u;
const HIDDEN: u32 = ${hidden}u;
const PERCEPTION: u32 = ${perception}u;
const BATCH: u32 = ${batchSize}u;
const BATCH_CELLS: u32 = ${batchCells}u;
const PARAMS: u32 = ${parameters.count}u;
const STABILITY_STEPS: u32 = ${options.stabilitySteps}u;
const FIRE_RATE: f32 = ${options.fireRate};
const STEP_SIZE: f32 = ${options.stepSize};
const ALIVE_THRESHOLD: f32 = ${options.aliveThreshold};

const WEIGHTS: u32 = ${o("weights")}u;
const W1: u32 = ${o("weights") + parameters.inputToHidden}u;
const B1: u32 = ${o("weights") + parameters.hiddenBias}u;
const W2: u32 = ${o("weights") + parameters.hiddenToOutput}u;
const B2: u32 = ${o("weights") + parameters.outputBias}u;
const ADAM_FIRST: u32 = ${o("adamFirst")}u;
const ADAM_SECOND: u32 = ${o("adamSecond")}u;
const GRADIENT: u32 = ${o("gradient")}u;
const STATE: u32 = ${o("stateTape")}u;
const CANDIDATE: u32 = ${o("candidate")}u;
const LIFE: u32 = ${o("life")}u;
const GRADIENT_A: u32 = ${o("gradientA")}u;
const GRADIENT_B: u32 = ${o("gradientB")}u;
const PERCEPTION_SCRATCH: u32 = ${o("perception")}u;
const HIDDEN_SCRATCH: u32 = ${o("hidden")}u;
const DELTA_SCRATCH: u32 = ${o("delta")}u;
const HIDDEN_GRADIENT: u32 = ${o("hiddenGradient")}u;
const CANDIDATE_GRADIENT: u32 = ${o("candidateGradient")}u;
const TARGET: u32 = ${o("target")}u;
const INITIAL: u32 = ${o("initial")}u;
const LOSS: u32 = ${o("loss")}u;
const METRICS: u32 = ${o("metrics")}u;

fn stateIndex(time: u32, batchCell: u32, channel: u32) -> u32 {
  return STATE + (time * BATCH_CELLS + batchCell) * CHANNELS + channel;
}

fn sampleState(
  time: u32,
  batch: u32,
  x: i32,
  y: i32,
  channel: u32,
) -> f32 {
  if (x < 0 || y < 0 || x >= i32(SIZE) || y >= i32(SIZE)) {
    return 0.0;
  }
  let cell = u32(y) * SIZE + u32(x);
  return heap[stateIndex(time, batch * CELLS + cell, channel)];
}

fn sampleCandidate(batch: u32, x: i32, y: i32, channel: u32) -> f32 {
  if (x < 0 || y < 0 || x >= i32(SIZE) || y >= i32(SIZE)) {
    return 0.0;
  }
  let cell = u32(y) * SIZE + u32(x);
  return heap[CANDIDATE + (batch * CELLS + cell) * CHANNELS + channel];
}

fn hash(value: u32) -> u32 {
  var result = value;
  result = result ^ (result >> 16u);
  result = result * 0x7feb352du;
  result = result ^ (result >> 15u);
  result = result * 0x846ca68bu;
  result = result ^ (result >> 16u);
  return result;
}

fn fires(batchCell: u32) -> f32 {
  let key =
    batchCell +
    dispatch.time * 0x9e3779b9u +
    dispatch.iteration * 0x85ebca6bu;
  let random = f32(hash(key)) / 4294967295.0;
  return select(0.0, 1.0, random <= FIRE_RATE);
}

fn fillPerception(
  time: u32,
  batch: u32,
  x: i32,
  y: i32,
  values: ptr<function, array<f32, ${perception}>>,
) {
  for (var channel = 0u; channel < CHANNELS; channel += 1u) {
    let s00 = sampleState(time, batch, x - 1, y - 1, channel);
    let s10 = sampleState(time, batch, x,     y - 1, channel);
    let s20 = sampleState(time, batch, x + 1, y - 1, channel);
    let s01 = sampleState(time, batch, x - 1, y,     channel);
    let s11 = sampleState(time, batch, x,     y,     channel);
    let s21 = sampleState(time, batch, x + 1, y,     channel);
    let s02 = sampleState(time, batch, x - 1, y + 1, channel);
    let s12 = sampleState(time, batch, x,     y + 1, channel);
    let s22 = sampleState(time, batch, x + 1, y + 1, channel);
    (*values)[channel * 3u] = s11;
    (*values)[channel * 3u + 1u] =
      (s20 + 2.0 * s21 + s22 - s00 - 2.0 * s01 - s02) / 8.0;
    (*values)[channel * 3u + 2u] =
      (s02 + 2.0 * s12 + s22 - s00 - 2.0 * s10 - s20) / 8.0;
  }
}

fn gradientInputBase() -> u32 {
  let reverseIndex = dispatch.activeSteps - 1u - dispatch.time;
  return select(GRADIENT_A, GRADIENT_B, (reverseIndex & 1u) == 1u);
}

fn gradientOutputBase() -> u32 {
  let reverseIndex = dispatch.activeSteps - 1u - dispatch.time;
  return select(GRADIENT_B, GRADIENT_A, (reverseIndex & 1u) == 1u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn forwardCandidate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batchCell = gid.x;
  if (batchCell >= BATCH_CELLS) {
    return;
  }
  let cell = batchCell % CELLS;
  let batch = batchCell / CELLS;
  let x = i32(cell % SIZE);
  let y = i32(cell / SIZE);

  var maxAlpha = 0.0;
  for (var oy = -1; oy <= 1; oy += 1) {
    for (var ox = -1; ox <= 1; ox += 1) {
      maxAlpha = max(
        maxAlpha,
        sampleState(dispatch.time, batch, x + ox, y + oy, 3u),
      );
    }
  }
  heap[LIFE + dispatch.time * BATCH_CELLS + batchCell] =
    select(0.0, 1.0, maxAlpha > ALIVE_THRESHOLD);

  var perception: array<f32, ${perception}>;
  fillPerception(dispatch.time, batch, x, y, &perception);
  var delta: array<f32, ${channels}>;
  for (var channel = 0u; channel < CHANNELS; channel += 1u) {
    delta[channel] = heap[B2 + channel];
  }
  for (var hidden = 0u; hidden < HIDDEN; hidden += 1u) {
    var value = heap[B1 + hidden];
    for (var input = 0u; input < PERCEPTION; input += 1u) {
      value += heap[W1 + hidden * PERCEPTION + input] * perception[input];
    }
    value = max(value, 0.0);
    for (var channel = 0u; channel < CHANNELS; channel += 1u) {
      delta[channel] += heap[W2 + channel * HIDDEN + hidden] * value;
    }
  }
  let update = fires(batchCell) * STEP_SIZE;
  for (var channel = 0u; channel < CHANNELS; channel += 1u) {
    let previous = heap[stateIndex(dispatch.time, batchCell, channel)];
    heap[CANDIDATE + batchCell * CHANNELS + channel] =
      previous + update * delta[channel];
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn forwardLife(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batchCell = gid.x;
  if (batchCell >= BATCH_CELLS) {
    return;
  }
  let cell = batchCell % CELLS;
  let batch = batchCell / CELLS;
  let x = i32(cell % SIZE);
  let y = i32(cell / SIZE);
  var maxAlpha = 0.0;
  for (var oy = -1; oy <= 1; oy += 1) {
    for (var ox = -1; ox <= 1; ox += 1) {
      maxAlpha = max(maxAlpha, sampleCandidate(batch, x + ox, y + oy, 3u));
    }
  }
  let maskIndex = LIFE + dispatch.time * BATCH_CELLS + batchCell;
  let alive =
    heap[maskIndex] > 0.5 && maxAlpha > ALIVE_THRESHOLD;
  heap[maskIndex] = select(0.0, 1.0, alive);
  for (var channel = 0u; channel < CHANNELS; channel += 1u) {
    let value = heap[CANDIDATE + batchCell * CHANNELS + channel];
    heap[stateIndex(dispatch.time + 1u, batchCell, channel)] =
      select(0.0, value, alive);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn initializeLoss(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= BATCH_CELLS * CHANNELS) {
    return;
  }
  let channel = index % CHANNELS;
  let batchCell = index / CHANNELS;
  heap[GRADIENT_A + index] = 0.0;
  heap[GRADIENT_B + index] = 0.0;
  if (channel < 4u) {
    let cell = batchCell % CELLS;
    let difference =
      heap[stateIndex(dispatch.activeSteps, batchCell, channel)] -
      heap[TARGET + cell * 4u + channel];
    let scale =
      1.0 / f32(BATCH_CELLS * 4u * STABILITY_STEPS);
    heap[GRADIENT_A + index] = 2.0 * difference * scale;
    heap[LOSS + batchCell * 4u + channel] =
      difference * difference / f32(STABILITY_STEPS);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn addStabilityLoss(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= BATCH_CELLS * 4u) {
    return;
  }
  let channel = index % 4u;
  let batchCell = index / 4u;
  let cell = batchCell % CELLS;
  let difference =
    heap[stateIndex(dispatch.time, batchCell, channel)] -
    heap[TARGET + cell * 4u + channel];
  let scale =
    1.0 / f32(BATCH_CELLS * 4u * STABILITY_STEPS);
  let gradientIndex =
    gradientOutputBase() + batchCell * CHANNELS + channel;
  heap[gradientIndex] += 2.0 * difference * scale;
  heap[LOSS + index] +=
    difference * difference / f32(STABILITY_STEPS);
}

@compute @workgroup_size(1)
fn reduceLoss() {
  var sum = 0.0;
  for (var index = 0u; index < BATCH_CELLS * 4u; index += 1u) {
    sum += heap[LOSS + index];
  }
  heap[METRICS] = sum / f32(BATCH_CELLS * 4u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn backwardMlp(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batchCell = gid.x;
  if (batchCell >= BATCH_CELLS) {
    return;
  }
  let cell = batchCell % CELLS;
  let batch = batchCell / CELLS;
  let x = i32(cell % SIZE);
  let y = i32(cell / SIZE);
  let life =
    heap[LIFE + dispatch.time * BATCH_CELLS + batchCell];
  let update = fires(batchCell) * STEP_SIZE;
  let gradientBase = gradientInputBase();

  var perception: array<f32, ${perception}>;
  fillPerception(dispatch.time, batch, x, y, &perception);
  for (var input = 0u; input < PERCEPTION; input += 1u) {
    heap[PERCEPTION_SCRATCH + batchCell * PERCEPTION + input] =
      perception[input];
  }

  var hiddenValues: array<f32, ${hidden}>;
  for (var hidden = 0u; hidden < HIDDEN; hidden += 1u) {
    var value = heap[B1 + hidden];
    for (var input = 0u; input < PERCEPTION; input += 1u) {
      value += heap[W1 + hidden * PERCEPTION + input] * perception[input];
    }
    hiddenValues[hidden] = max(value, 0.0);
    heap[HIDDEN_SCRATCH + batchCell * HIDDEN + hidden] =
      hiddenValues[hidden];
  }

  var deltaGradient: array<f32, ${channels}>;
  for (var channel = 0u; channel < CHANNELS; channel += 1u) {
    let candidateGradient =
      heap[gradientBase + batchCell * CHANNELS + channel] * life;
    heap[CANDIDATE_GRADIENT + batchCell * CHANNELS + channel] =
      candidateGradient;
    deltaGradient[channel] = candidateGradient * update;
    heap[DELTA_SCRATCH + batchCell * CHANNELS + channel] =
      deltaGradient[channel];
  }

  var perceptionGradient: array<f32, ${perception}>;
  for (var input = 0u; input < PERCEPTION; input += 1u) {
    perceptionGradient[input] = 0.0;
  }
  for (var hidden = 0u; hidden < HIDDEN; hidden += 1u) {
    var gradient = 0.0;
    for (var channel = 0u; channel < CHANNELS; channel += 1u) {
      gradient +=
        heap[W2 + channel * HIDDEN + hidden] * deltaGradient[channel];
    }
    gradient *= select(0.0, 1.0, hiddenValues[hidden] > 0.0);
    heap[HIDDEN_GRADIENT + batchCell * HIDDEN + hidden] = gradient;
    for (var input = 0u; input < PERCEPTION; input += 1u) {
      perceptionGradient[input] +=
        heap[W1 + hidden * PERCEPTION + input] * gradient;
    }
  }
  for (var input = 0u; input < PERCEPTION; input += 1u) {
    heap[PERCEPTION_SCRATCH + BATCH_CELLS * PERCEPTION +
      batchCell * PERCEPTION + input] = perceptionGradient[input];
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn backwardPerception(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= BATCH_CELLS * CHANNELS) {
    return;
  }
  let channel = index % CHANNELS;
  let batchCell = index / CHANNELS;
  let cell = batchCell % CELLS;
  let batch = batchCell / CELLS;
  let x = i32(cell % SIZE);
  let y = i32(cell / SIZE);
  let perceptionGradient =
    PERCEPTION_SCRATCH + BATCH_CELLS * PERCEPTION;
  var gradient =
    heap[CANDIDATE_GRADIENT + batchCell * CHANNELS + channel] +
    heap[perceptionGradient + batchCell * PERCEPTION + channel * 3u];

  for (var outputY = y - 1; outputY <= y + 1; outputY += 1) {
    for (var outputX = x - 1; outputX <= x + 1; outputX += 1) {
      if (
        outputX < 0 || outputY < 0 ||
        outputX >= i32(SIZE) || outputY >= i32(SIZE)
      ) {
        continue;
      }
      let dx = x - outputX;
      let dy = y - outputY;
      let outputCell = u32(outputY) * SIZE + u32(outputX);
      let outputBatchCell = batch * CELLS + outputCell;
      var coefficientX = 0.0;
      var coefficientY = 0.0;
      if (dx != 0) {
        coefficientX =
          f32(dx) * select(0.125, 0.25, dy == 0);
      }
      if (dy != 0) {
        coefficientY =
          f32(dy) * select(0.125, 0.25, dx == 0);
      }
      gradient +=
        heap[perceptionGradient +
          outputBatchCell * PERCEPTION + channel * 3u + 1u] *
          coefficientX;
      gradient +=
        heap[perceptionGradient +
          outputBatchCell * PERCEPTION + channel * 3u + 2u] *
          coefficientY;
    }
  }
  heap[gradientOutputBase() + index] = gradient;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn parameterGradient(@builtin(global_invocation_id) gid: vec3<u32>) {
  let parameter = gid.x;
  if (parameter >= PARAMS) {
    return;
  }
  var sum = 0.0;
  if (parameter < ${parameters.hiddenBias}u) {
    let relative = parameter - ${parameters.inputToHidden}u;
    let hidden = relative / PERCEPTION;
    let input = relative % PERCEPTION;
    for (var batchCell = 0u; batchCell < BATCH_CELLS; batchCell += 1u) {
      sum +=
        heap[HIDDEN_GRADIENT + batchCell * HIDDEN + hidden] *
        heap[PERCEPTION_SCRATCH + batchCell * PERCEPTION + input];
    }
  } else if (parameter < ${parameters.hiddenToOutput}u) {
    let hidden = parameter - ${parameters.hiddenBias}u;
    for (var batchCell = 0u; batchCell < BATCH_CELLS; batchCell += 1u) {
      sum += heap[HIDDEN_GRADIENT + batchCell * HIDDEN + hidden];
    }
  } else if (parameter < ${parameters.outputBias}u) {
    let relative = parameter - ${parameters.hiddenToOutput}u;
    let channel = relative / HIDDEN;
    let hidden = relative % HIDDEN;
    for (var batchCell = 0u; batchCell < BATCH_CELLS; batchCell += 1u) {
      sum +=
        heap[DELTA_SCRATCH + batchCell * CHANNELS + channel] *
        heap[HIDDEN_SCRATCH + batchCell * HIDDEN + hidden];
    }
  } else {
    let channel = parameter - ${parameters.outputBias}u;
    for (var batchCell = 0u; batchCell < BATCH_CELLS; batchCell += 1u) {
      sum += heap[DELTA_SCRATCH + batchCell * CHANNELS + channel];
    }
  }
  heap[GRADIENT + parameter] += sum;
}

@compute @workgroup_size(1)
fn normalizeGradient() {
  var w1Squared = 0.0;
  var b1Squared = 0.0;
  var w2Squared = 0.0;
  var b2Squared = 0.0;
  for (var parameter = 0u; parameter < ${parameters.hiddenBias}u; parameter += 1u) {
    let value = heap[GRADIENT + parameter];
    w1Squared += value * value;
  }
  for (
    var parameter = ${parameters.hiddenBias}u;
    parameter < ${parameters.hiddenToOutput}u;
    parameter += 1u
  ) {
    let value = heap[GRADIENT + parameter];
    b1Squared += value * value;
  }
  for (
    var parameter = ${parameters.hiddenToOutput}u;
    parameter < ${parameters.outputBias}u;
    parameter += 1u
  ) {
    let value = heap[GRADIENT + parameter];
    w2Squared += value * value;
  }
  for (
    var parameter = ${parameters.outputBias}u;
    parameter < PARAMS;
    parameter += 1u
  ) {
    let value = heap[GRADIENT + parameter];
    b2Squared += value * value;
  }
  heap[METRICS + 1u] = inverseSqrt(max(w1Squared, 1e-16));
  heap[METRICS + 2u] = inverseSqrt(max(b1Squared, 1e-16));
  heap[METRICS + 3u] = inverseSqrt(max(w2Squared, 1e-16));
  heap[METRICS + 4u] = inverseSqrt(max(b2Squared, 1e-16));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn updateAdam(@builtin(global_invocation_id) gid: vec3<u32>) {
  let parameter = gid.x;
  if (parameter >= PARAMS) {
    return;
  }
  let beta1 = 0.9;
  let beta2 = 0.999;
  var scale = heap[METRICS + 1u];
  if (parameter >= ${parameters.hiddenBias}u) {
    scale = heap[METRICS + 2u];
  }
  if (parameter >= ${parameters.hiddenToOutput}u) {
    scale = heap[METRICS + 3u];
  }
  if (parameter >= ${parameters.outputBias}u) {
    scale = heap[METRICS + 4u];
  }
  let gradient = heap[GRADIENT + parameter] * scale;
  let first =
    beta1 * heap[ADAM_FIRST + parameter] + (1.0 - beta1) * gradient;
  let second =
    beta2 * heap[ADAM_SECOND + parameter] +
    (1.0 - beta2) * gradient * gradient;
  heap[ADAM_FIRST + parameter] = first;
  heap[ADAM_SECOND + parameter] = second;
  let step = f32(dispatch.iteration + 1u);
  let correctedFirst = first / (1.0 - pow(beta1, step));
  let correctedSecond = second / (1.0 - pow(beta2, step));
  heap[WEIGHTS + parameter] -=
    dispatch.learningRate *
    correctedFirst /
    (sqrt(correctedSecond) + 1e-8);
}
`;
}

export class GrowingNeuralCaTrainer {
  readonly backend = "webgpu";
  readonly size: number;
  readonly channels: number;
  readonly hidden: number;
  readonly batchSize: number;
  readonly rolloutSteps: number;
  readonly minRolloutSteps: number;
  readonly stabilitySteps: number;

  readonly #options: ResolvedOptions;
  readonly #device: GPUDevice;
  readonly #layout: GrowingNeuralCaHeapLayout;
  readonly #heap: GPUBuffer;
  readonly #uniforms: GPUBuffer;
  readonly #bindGroup: GPUBindGroup;
  readonly #pipelines: GrowingNeuralCaPipelines;
  readonly #readback: GPUBuffer;
  readonly #seedState: Float32Array;
  readonly #pool: Float32Array[] = [];
  readonly #uncapturedErrorHandler: EventListener;
  #iteration = 0;
  #busy = false;
  #disposed = false;
  #lostError: SynapticError | undefined;

  private constructor(
    options: ResolvedOptions,
    device: GPUDevice,
    layout: GrowingNeuralCaHeapLayout,
    heap: GPUBuffer,
    uniforms: GPUBuffer,
    bindGroup: GPUBindGroup,
    pipelines: GrowingNeuralCaPipelines,
    readback: GPUBuffer,
  ) {
    this.#options = options;
    this.#device = device;
    this.#layout = layout;
    this.#heap = heap;
    this.#uniforms = uniforms;
    this.#bindGroup = bindGroup;
    this.#pipelines = pipelines;
    this.#readback = readback;
    this.size = options.size;
    this.channels = options.channels;
    this.hidden = options.hidden;
    this.batchSize = options.batchSize;
    this.rolloutSteps = options.rolloutSteps;
    this.minRolloutSteps = options.minRolloutSteps;
    this.stabilitySteps = options.stabilitySteps;
    this.#seedState = seedState(options.size, options.channels);
    this.#fillPoolWithSeeds();
    this.#uncapturedErrorHandler = ((event: GPUUncapturedErrorEvent) => {
      this.#lostError = new SynapticError(
        event.error.message,
        "WEBGPU_VALIDATION_ERROR",
      );
    }) as EventListener;
    device.addEventListener("uncapturederror", this.#uncapturedErrorHandler);
    void device.lost.then((info) => {
      if (!this.#disposed) {
        this.#lostError = new SynapticError(
          `WebGPU device lost: ${info.message}`,
          "WEBGPU_DEVICE_LOST",
          { reason: info.reason },
        );
      }
    });
  }

  static async create(
    options: GrowingNeuralCaTrainerOptions = {},
  ): Promise<GrowingNeuralCaTrainer> {
    const resolved = resolveOptions(options);
    const adapter = await resolved.gpu.requestAdapter({
      ...(resolved.powerPreference === undefined
        ? {}
        : { powerPreference: resolved.powerPreference }),
    });
    invariant(
      adapter !== null,
      "No suitable WebGPU adapter is available",
      "WEBGPU_ADAPTER_UNAVAILABLE",
    );
    const device = await adapter.requestDevice();
    try {
      const layout = buildGrowingNeuralCaHeapLayout(resolved);
      invariant(
        layout.byteLength <= device.limits.maxBufferSize &&
          layout.byteLength <= device.limits.maxStorageBufferBindingSize,
        `Growing Neural CA heap requires ${layout.byteLength} bytes, exceeding the device limit`,
        "WEBGPU_GROWING_CA_HEAP_TOO_LARGE",
        {
          byteLength: layout.byteLength,
          maxBufferSize: device.limits.maxBufferSize,
          maxStorageBufferBindingSize:
            device.limits.maxStorageBufferBindingSize,
        },
      );
      const heap = device.createBuffer({
        label: "synaptic growing neural ca heap",
        size: layout.byteLength,
        usage:
          BUFFER_USAGE.STORAGE |
          BUFFER_USAGE.COPY_SRC |
          BUFFER_USAGE.COPY_DST,
      });
      const uniformCount = resolved.rolloutSteps;
      const uniforms = device.createBuffer({
        label: "synaptic growing neural ca uniforms",
        size: Math.max(1, uniformCount) * UNIFORM_STRIDE,
        usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
      });
      const uniformData = new ArrayBuffer(
        Math.max(1, uniformCount) * UNIFORM_STRIDE,
      );
      for (let time = 0; time < uniformCount; time += 1) {
        new Uint32Array(uniformData, time * UNIFORM_STRIDE, 2).set([time, 0]);
      }
      device.queue.writeBuffer(uniforms, 0, uniformData);

      const readbackBytes =
        4 +
        resolved.batchSize *
          resolved.size *
          resolved.size *
          resolved.channels *
          4;
      const readback = device.createBuffer({
        label: "synaptic growing neural ca readback",
        size: readbackBytes,
        usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST,
      });
      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: COMPUTE_STAGE,
            buffer: {
              type: "uniform",
              hasDynamicOffset: true,
              minBindingSize: 16,
            },
          },
          {
            binding: 1,
            visibility: COMPUTE_STAGE,
            buffer: {
              type: "storage",
              hasDynamicOffset: false,
              minBindingSize: layout.byteLength,
            },
          },
        ],
      });
      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      });
      const module = device.createShaderModule({
        label: "synaptic growing neural ca",
        code: shader(resolved, layout),
      });
      const compilation = await module.getCompilationInfo();
      const errors = compilation.messages.filter(
        (message) => message.type === "error",
      );
      if (errors.length > 0) {
        throw new SynapticError(
          errors
            .map(
              (message) =>
                `${message.lineNum}:${message.linePos} ${message.message}`,
            )
            .join("\n"),
          "WEBGPU_SHADER_COMPILATION_FAILED",
        );
      }
      const entryPoints = [
        "forwardCandidate",
        "forwardLife",
        "initializeLoss",
        "addStabilityLoss",
        "reduceLoss",
        "backwardMlp",
        "backwardPerception",
        "parameterGradient",
        "normalizeGradient",
        "updateAdam",
      ] as const;
      const built = await Promise.all(
        entryPoints.map(async (entryPoint) => {
          device.pushErrorScope("validation");
          const pipeline = await device.createComputePipelineAsync({
            label: `synaptic growing neural ca ${entryPoint}`,
            layout: pipelineLayout,
            compute: { module, entryPoint },
          });
          const error = await device.popErrorScope();
          if (error !== null) {
            throw new SynapticError(
              `${entryPoint}: ${error.message}`,
              "WEBGPU_PIPELINE_CREATION_FAILED",
            );
          }
          return pipeline;
        }),
      );
      const pipelines = Object.fromEntries(
        entryPoints.map((name, index) => [name, built[index]]),
      ) as unknown as GrowingNeuralCaPipelines;
      const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: {
              buffer: uniforms,
              offset: 0,
              size: 16,
            },
          },
          {
            binding: 1,
            resource: {
              buffer: heap,
              offset: 0,
              size: layout.byteLength,
            },
          },
        ],
      });
      const trainer = new GrowingNeuralCaTrainer(
        resolved,
        device,
        layout,
        heap,
        uniforms,
        bindGroup,
        pipelines,
        readback,
      );
      trainer.#initializeWeights();
      return trainer;
    } catch (error) {
      device.destroy();
      throw error;
    }
  }

  get iteration(): number {
    return this.#iteration;
  }

  resetPool(): void {
    this.#assertAvailable();
    this.#fillPoolWithSeeds();
  }

  async trainStep(
    target: Float32Array,
    options: GrowingNeuralCaTrainOptions = {},
  ): Promise<GrowingNeuralCaMetrics> {
    this.#assertAvailable();
    invariant(!this.#busy, "Growing Neural CA trainer is busy", "SESSION_BUSY");
    invariant(
      target.length === this.size * this.size * 4,
      `Target has length ${target.length}; expected ${this.size * this.size * 4}`,
      "INVALID_GROWING_CA_TARGET",
    );
    for (let index = 0; index < target.length; index += 1) {
      invariant(
        Number.isFinite(target[index]),
        `Target value ${index} must be finite`,
        "INVALID_GROWING_CA_TARGET",
      );
    }
    const learningRate = finiteRange(
      "learningRate",
      options.learningRate ?? this.#options.learningRate,
      0,
      1,
    );
    const damageProbability = finiteRange(
      "damageProbability",
      options.damageProbability ?? this.#options.damageProbability,
      0,
      1,
    );
    this.#busy = true;
    const started = performance.now();
    try {
      const activeRolloutSteps = this.#activeRolloutSteps();
      const { initial, damagedSamples, poolIndices } =
        this.#trainingBatch(target, damageProbability);
      const sections = this.#layout.sections;
      this.#device.queue.writeBuffer(
        this.#heap,
        sections.initial.byteOffset,
        initial,
      );
      this.#device.queue.writeBuffer(
        this.#heap,
        sections.stateTape.byteOffset,
        initial,
      );
      this.#device.queue.writeBuffer(
        this.#heap,
        sections.target.byteOffset,
        target,
      );
      this.#device.queue.writeBuffer(
        this.#heap,
        sections.gradient.byteOffset,
        new Float32Array(sections.gradient.length),
      );
      this.#writeUniforms(learningRate, activeRolloutSteps);

      const encoder = this.#device.createCommandEncoder({
        label: `synaptic growing neural ca iteration ${this.#iteration}`,
      });
      for (let time = 0; time < activeRolloutSteps; time += 1) {
        this.#pass(
          encoder,
          this.#pipelines.forwardCandidate,
          time,
          dispatchCount(this.batchSize * this.size * this.size),
          `forward candidate ${time}`,
        );
        this.#pass(
          encoder,
          this.#pipelines.forwardLife,
          time,
          dispatchCount(this.batchSize * this.size * this.size),
          `forward life ${time}`,
        );
      }
      this.#pass(
        encoder,
        this.#pipelines.initializeLoss,
        0,
        dispatchCount(
          this.batchSize * this.size * this.size * this.channels,
        ),
        "initialize loss",
      );
      for (let time = activeRolloutSteps - 1; time >= 0; time -= 1) {
        this.#pass(
          encoder,
          this.#pipelines.backwardMlp,
          time,
          dispatchCount(this.batchSize * this.size * this.size),
          `backward mlp ${time}`,
        );
        this.#pass(
          encoder,
          this.#pipelines.parameterGradient,
          time,
          dispatchCount(this.#layout.parameterCount),
          `parameter gradient ${time}`,
        );
        this.#pass(
          encoder,
          this.#pipelines.backwardPerception,
          time,
          dispatchCount(
            this.batchSize * this.size * this.size * this.channels,
          ),
          `backward perception ${time}`,
        );
        if (
          time > 0 &&
          time >= activeRolloutSteps - this.stabilitySteps + 1
        ) {
          this.#pass(
            encoder,
            this.#pipelines.addStabilityLoss,
            time,
            dispatchCount(this.batchSize * this.size * this.size * 4),
            `stability loss ${time}`,
          );
        }
      }
      this.#pass(
        encoder,
        this.#pipelines.reduceLoss,
        0,
        1,
        "reduce loss",
      );
      this.#pass(
        encoder,
        this.#pipelines.normalizeGradient,
        0,
        1,
        "normalize gradient",
      );
      this.#pass(
        encoder,
        this.#pipelines.updateAdam,
        0,
        dispatchCount(this.#layout.parameterCount),
        "Adam update",
      );
      encoder.copyBufferToBuffer(
        this.#heap,
        sections.metrics.byteOffset,
        this.#readback,
        0,
        4,
      );
      const finalOffset =
        sections.stateTape.byteOffset +
        activeRolloutSteps *
          this.batchSize *
          this.size *
          this.size *
          this.channels *
          4;
      encoder.copyBufferToBuffer(
        this.#heap,
        finalOffset,
        this.#readback,
        4,
        this.batchSize * this.size * this.size * this.channels * 4,
      );
      this.#device.queue.submit([encoder.finish()]);

      await this.#readback.mapAsync(MAP_READ);
      const bytes = new Uint8Array(this.#readback.getMappedRange()).slice();
      this.#readback.unmap();
      const loss = new Float32Array(bytes.buffer, 0, 1)[0] ?? Number.NaN;
      const final = new Float32Array(
        bytes.buffer,
        4,
        this.batchSize * this.size * this.size * this.channels,
      ).slice();
      invariant(
        Number.isFinite(loss) && final.every(Number.isFinite),
        "Growing Neural CA training produced non-finite values",
        "NON_FINITE_TRAINING_STATE",
      );
      this.#updatePool(final, poolIndices);
      this.#iteration += 1;
      return {
        iteration: this.#iteration,
        loss,
        rolloutSteps: activeRolloutSteps,
        durationMs: performance.now() - started,
        damagedSamples,
      };
    } finally {
      this.#busy = false;
    }
  }

  async artifact(): Promise<GrowingNeuralCaArtifact> {
    this.#assertAvailable();
    invariant(!this.#busy, "Growing Neural CA trainer is busy", "SESSION_BUSY");
    const parameters = parameterLayout(this.channels, this.hidden);
    const section = this.#layout.sections.weights;
    const readback = this.#device.createBuffer({
      label: "synaptic growing neural ca weights readback",
      size: section.byteLength,
      usage: BUFFER_USAGE.MAP_READ | BUFFER_USAGE.COPY_DST,
    });
    try {
      const encoder = this.#device.createCommandEncoder({
        label: "synaptic growing neural ca artifact",
      });
      encoder.copyBufferToBuffer(
        this.#heap,
        section.byteOffset,
        readback,
        0,
        section.byteLength,
      );
      this.#device.queue.submit([encoder.finish()]);
      await readback.mapAsync(MAP_READ);
      const weights = new Float32Array(
        readback.getMappedRange().slice(0),
      );
      readback.unmap();
      return {
        format: GROWING_NEURAL_CA_FORMAT,
        version: GROWING_NEURAL_CA_VERSION,
        channels: this.channels,
        hidden: this.hidden,
        perception: ["identity", "sobel-x", "sobel-y"],
        activation: "relu",
        fireRate: this.#options.fireRate,
        stepSize: this.#options.stepSize,
        boundary: "zero",
        life: {
          channel: 3,
          threshold: this.#options.aliveThreshold,
          neighborhood: 3,
          preAndPost: true,
        },
        weights: {
          inputToHidden: Array.from(
            weights.slice(
              parameters.inputToHidden,
              parameters.hiddenBias,
            ),
          ),
          hiddenBias: Array.from(
            weights.slice(
              parameters.hiddenBias,
              parameters.hiddenToOutput,
            ),
          ),
          hiddenToOutput: Array.from(
            weights.slice(
              parameters.hiddenToOutput,
              parameters.outputBias,
            ),
          ),
          outputBias: Array.from(
            weights.slice(parameters.outputBias, parameters.count),
          ),
        },
      };
    } finally {
      readback.destroy();
    }
  }

  dispose(): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    this.#device.removeEventListener(
      "uncapturederror",
      this.#uncapturedErrorHandler,
    );
    this.#heap.destroy();
    this.#uniforms.destroy();
    this.#readback.destroy();
    this.#device.destroy();
  }

  #initializeWeights(): void {
    const section = this.#layout.sections.weights;
    const weights = this.#options.initialArtifact === undefined
      ? createGrowingNeuralCaWeights({
          channels: this.channels,
          hidden: this.hidden,
          seed: this.#options.seed,
        })
      : validateArtifact(
          this.#options.initialArtifact,
          this.channels,
          this.hidden,
        );
    this.#device.queue.writeBuffer(
      this.#heap,
      section.byteOffset,
      weights,
    );
  }

  #writeUniforms(learningRate: number, activeRolloutSteps: number): void {
    const records = new ArrayBuffer(this.rolloutSteps * UNIFORM_STRIDE);
    for (let time = 0; time < this.rolloutSteps; time += 1) {
      const offset = time * UNIFORM_STRIDE;
      new Uint32Array(records, offset, 3).set([
        time,
        this.#iteration,
        activeRolloutSteps,
      ]);
      new Float32Array(records, offset + 12, 1)[0] = learningRate;
    }
    this.#device.queue.writeBuffer(this.#uniforms, 0, records);
  }

  #pass(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    time: number,
    workgroups: number,
    label: string,
  ): void {
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.#bindGroup, [time * UNIFORM_STRIDE]);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
  }

  #activeRolloutSteps(): number {
    if (this.minRolloutSteps === this.rolloutSteps) {
      return this.rolloutSteps;
    }
    const random = mulberry32(
      this.#options.seed ^
        Math.imul(this.#iteration + 1, 0x7f4a_7c15) ^
        0x726f_6c6c,
    );
    return this.minRolloutSteps +
      Math.floor(
        random() * (this.rolloutSteps - this.minRolloutSteps + 1),
      );
  }

  #fillPoolWithSeeds(): void {
    this.#pool.length = 0;
    for (let index = 0; index < this.#options.poolSize; index += 1) {
      this.#pool.push(this.#seedState.slice());
    }
  }

  #trainingBatch(
    target: Float32Array,
    damageProbability: number,
  ): {
    readonly initial: Float32Array;
    readonly damagedSamples: number;
    readonly poolIndices: readonly number[];
  } {
    const stateLength = this.size * this.size * this.channels;
    const initial = new Float32Array(this.batchSize * stateLength);
    const random = mulberry32(
      this.#options.seed ^ Math.imul(this.#iteration + 1, 0x9e37_79b9),
    );
    const availablePoolIndices = Array.from(
      { length: this.#pool.length },
      (_, index) => index,
    );
    const poolIndices: number[] = [];
    const samples: Float32Array[] = [];
    for (let batch = 0; batch < this.batchSize; batch += 1) {
      const poolIndex = availablePoolIndices.length === 0
        ? Math.floor(random() * this.#pool.length)
        : availablePoolIndices.splice(
            Math.floor(random() * availablePoolIndices.length),
            1,
          )[0]!;
      poolIndices.push(poolIndex);
      samples.push((this.#pool[poolIndex] ?? this.#seedState).slice());
    }

    const warmingUp =
      this.#iteration < this.#options.poolWarmupIterations;
    const ranked = samples
      .map((sample, batch) => ({
        batch,
        loss: measureGrowingNeuralCaState(
          sample,
          target,
          this.channels,
          this.#options.aliveThreshold,
        ).loss,
      }))
      .sort((left, right) => left.loss - right.loss);
    const seedBatch = warmingUp
      ? -1
      : this.batchSize === 1
      ? (this.#iteration % 4 === 0 ? 0 : -1)
      : ranked.at(-1)!.batch;
    const damageCandidateCount = Math.max(
      1,
      Math.floor((this.batchSize - (seedBatch >= 0 ? 1 : 0)) / 2),
    );
    const damageCandidates = new Set(
      ranked
        .filter(({ batch }) => batch !== seedBatch)
        .slice(0, damageCandidateCount)
        .map(({ batch }) => batch),
    );

    let damagedSamples = 0;
    for (let batch = 0; batch < this.batchSize; batch += 1) {
      const forceSeed = warmingUp || batch === seedBatch;
      const sample = forceSeed
        ? this.#seedState.slice()
        : samples[batch]!;
      if (
        !forceSeed &&
        damageCandidates.has(batch) &&
        random() < damageProbability
      ) {
        // Match the broad random cuts used by the reference regenerating
        // curriculum. A fixed, small hole teaches local smoothing but does
        // not make the target a sufficiently wide repair attractor.
        const minimumRadius = Math.max(2, Math.floor(this.size * 0.1));
        const maximumRadius = Math.max(
          minimumRadius,
          Math.floor(this.size * 0.2),
        );
        const radius =
          minimumRadius +
          Math.floor(random() * (maximumRadius - minimumRadius + 1));
        const centerX =
          Math.floor(this.size / 2) +
          Math.floor((random() * 2 - 1) * this.size * 0.2);
        const centerY =
          Math.floor(this.size / 2) +
          Math.floor((random() * 2 - 1) * this.size * 0.2);
        for (let y = 0; y < this.size; y += 1) {
          for (let x = 0; x < this.size; x += 1) {
            if ((x - centerX) ** 2 + (y - centerY) ** 2 > radius ** 2) {
              continue;
            }
            sample.fill(
              0,
              (y * this.size + x) * this.channels,
              (y * this.size + x + 1) * this.channels,
            );
          }
        }
        damagedSamples += 1;
      }
      initial.set(sample, batch * stateLength);
    }
    return { initial, damagedSamples, poolIndices };
  }

  #updatePool(
    final: Float32Array,
    poolIndices: readonly number[],
  ): void {
    const stateLength = this.size * this.size * this.channels;
    for (let batch = 0; batch < this.batchSize; batch += 1) {
      const sample = final.slice(
        batch * stateLength,
        (batch + 1) * stateLength,
      );
      let maximum = 0;
      let finite = true;
      for (const value of sample) {
        finite &&= Number.isFinite(value);
        maximum = Math.max(maximum, Math.abs(value));
      }
      const poolIndex = poolIndices[batch];
      invariant(
        poolIndex !== undefined,
        "Growing Neural CA pool selection is incomplete",
        "INVALID_GROWING_CA_POOL",
      );
      this.#pool[poolIndex] =
        finite && maximum <= this.#options.poolValueLimit
          ? sample
          : this.#seedState.slice();
    }
  }

  #assertAvailable(): void {
    invariant(
      !this.#disposed,
      "Growing Neural CA trainer has been disposed",
      "SESSION_DISPOSED",
    );
    if (this.#lostError !== undefined) {
      throw this.#lostError;
    }
  }
}
