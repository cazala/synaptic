# Growing Neural Cellular Automata

`GrowingNeuralCaTrainer` is Synaptic's WebGPU specialization for the model in
[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/). It
trains a local rule in Synaptic and exports the same versioned artifact that
Automata runs for inference.

## Minimal training loop

```ts
import { GrowingNeuralCaTrainer } from "@synaptic/backend-webgpu";

const trainer = await GrowingNeuralCaTrainer.create({
  size: 24,
  channels: 16,
  hidden: 32,
  batchSize: 4,
  rolloutSteps: 48,
  fireRate: 0.5,
  learningRate: 0.001,
  poolWarmupIterations: 128,
});

// Row-major 24 × 24 premultiplied RGBA, with values normally in [0, 1].
const target = new Float32Array(24 * 24 * 4);

for (let iteration = 0; iteration < 8_000; iteration += 1) {
  const persistence = iteration >= 128;
  const metrics = await trainer.trainStep(target, {
    learningRate: persistence ? 0.0002 : 0.001,
    damageProbability: iteration >= 512 ? 0.4 : 0,
  });
  console.log(metrics.iteration, metrics.loss, metrics.durationMs);
}

const artifact = await trainer.artifact();
trainer.dispose();
```

The target's RGB components must already be multiplied by alpha. A Canvas
`ImageData` buffer is straight alpha, so convert it:

```ts
const alpha = pixels[index + 3] / 255;
target[index] = (pixels[index] / 255) * alpha;
target[index + 1] = (pixels[index + 1] / 255) * alpha;
target[index + 2] = (pixels[index + 2] / 255) * alpha;
target[index + 3] = alpha;
```

## Run the artifact in Automata

```ts
import { Engine, GrowingNeural } from "@cazala/automata";

const automaton = GrowingNeural.fromArtifact(artifact);
const engine = new Engine({
  canvas,
  automaton,
  grid: { width: 24, height: 24, wrap: false, maxCells: 24 },
  stepsPerSecond: 60,
  render: { colorBg: { r: 1, g: 1, b: 1, a: 1 } },
});

await engine.initialize();
engine.coverGrid();
engine.reset({ mode: "center" });
engine.play();
```

While training continues, avoid rebuilding the Automata pipelines:

```ts
const next = await trainer.artifact();
automaton.setWeights({
  channels: next.channels,
  hidden: next.hidden,
  ...next.weights,
});
```

## Exact model semantics

Every cell contains premultiplied RGB, alpha/liveness, and latent channels.
One generation performs:

1. identity, Sobel-x, and Sobel-y perception independently on every channel,
   with zero-valued samples beyond the grid;
2. a pointwise shared `channels × 3 → hidden → channels` MLP with ReLU;
3. a `stepSize` residual update under an independent stochastic fire mask;
4. the pre-update and post-update 3×3 alpha life tests.

The life comparisons and random mask are treated as stopped-gradient masks,
matching the useful derivative path in the TensorFlow reference. The
candidate and final life mask are separate compute phases so every cell sees a
complete neighboring candidate state.

## Training implementation

The trainer keeps all mutable arrays in one f32 WebGPU storage heap:

- the state tape for every batch and generation;
- candidate, life-mask, perception, hidden, and reverse-mode scratch;
- the four parameter tensors, accumulated gradients, and Adam slots;
- target, initial batch, loss, and metrics.

Backward perception uses a gathered 3×3 transpose convolution. Each invocation
owns one source cell and reads neighboring output gradients, so the portable
shader does not require f32 atomics. Gradients are normalized separately for
the two kernels and two biases before Adam, as in the reference training loop.

## Sample pool and damage curriculum

Training only from a seed can learn growth over the rollout horizon but usually
does not learn persistence. The trainer therefore stores evolved states and
uses them as later initial states. Recommended phases are:

1. seed-only growth warm-up;
2. persistent sample-pool rollouts at a lower learning rate;
3. damaged pool samples for regeneration.

One batch member regularly remains a fresh center seed. Rollouts whose absolute
state exceeds `poolValueLimit` are replaced by a seed, the compact equivalent
of replacing the worst-ranked sample in the reference pool. This prevents one
numerically runaway state from dominating a small browser batch.

The task is iterative, not an instant classifier. A small 24×24, 32-hidden
demo begins forming the target within hundreds of iterations and becomes more
stable over thousands. Larger hidden layers and 64–96 generation rollouts are
closer to the paper but cost proportionally more GPU work.

## Options

| Option | Default | Constraint | Meaning |
| --- | ---: | ---: | --- |
| `size` | 24 | 8–64 | square grid edge |
| `channels` | 16 | 4–32 | cell state width; channel 3 is alpha |
| `hidden` | 64 | 1–128 | shared MLP hidden width |
| `batchSize` | 2 | 1–8 | independent rollouts per update |
| `rolloutSteps` | 24 | 1–64 | BPTT generations |
| `fireRate` | 0.5 | 0–1 | per-cell update probability |
| `stepSize` | 1 | 0–2 | residual multiplier |
| `aliveThreshold` | 0.1 | 0–1 | alpha neighborhood threshold |
| `learningRate` | 0.002 | 0–1 | Adam learning rate |
| `poolSize` | max(8, batch × 4) | 1–64 | retained evolved states |
| `poolWarmupIterations` | 128 | 0–10,000 | seed-only updates |
| `poolValueLimit` | 8 | 1–1,000 | runaway-state replacement bound |
| `damageProbability` | 0.35 | 0–1 | chance to erase a sampled state |

The trainer requires WebGPU and has no fallback. This is deliberate: the
generation tape and spatial reverse kernels are a GPU specialization, not the
generic `Session` graph contract.
