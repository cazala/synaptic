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
  hidden: 128,
  batchSize: 8,
  minRolloutSteps: 64,
  rolloutSteps: 96,
  stabilitySteps: 16,
  fireRate: 0.5,
  learningRate: 0.002,
  poolSize: 128,
  poolWarmupIterations: 128,
});

// Row-major 24 × 24 premultiplied RGBA, with values normally in [0, 1].
const target = new Float32Array(24 * 24 * 4);

for (let iteration = 0; iteration < 20_000; iteration += 1) {
  const persistence = iteration >= 128;
  const regenerationProgress = Math.max(
    0,
    Math.min(1, (iteration - 512) / 512),
  );
  const metrics = await trainer.trainStep(target, {
    learningRate: !persistence
      ? 0.002
      : iteration < 2_048
      ? 0.0005
      : 0.0003,
    damageProbability: regenerationProgress,
  });
  console.log(
    metrics.iteration,
    metrics.rolloutSteps,
    metrics.loss,
    metrics.durationMs,
  );
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

The browser demo also rate-limits a small `engine.getCells()` readback and
passes it to `measureGrowingNeuralCaState(...)`. It uses a wider live-state
limit than the training pool, then reseeds inference when values become
non-finite, hidden channels exceed that range, living cells saturate the grid,
or a previously formed organism collapses. This guard is intentionally outside
the animation loop; production rendering should not perform a full state
readback per frame.

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
uses them as later initial states. Each update samples a horizon between
`minRolloutSteps` and `rolloutSteps`, preventing the target from becoming a
single transient checkpoint. `stabilitySteps` applies the target loss to
consecutive states at the end of that horizon, directly training the completed
organism to remain in its target basin. Recommended phases are:

1. seed-only growth warm-up;
2. persistent sample-pool rollouts at a lower learning rate;
3. a gradual ramp from undamaged to several damaged healthy samples per batch.

The demo uses the reference batch shape of eight samples: the worst sample is
reseeded and the healthiest three are damage candidates. It begins the damage
ramp at update 512, reaches three damaged samples per batch at update 1,024,
and retains the persistence learning rate through update 2,048 before decaying
it. This teaches repair during the same early window in which the flower first
becomes stable instead of postponing regeneration for a separate long phase.
The page presents those boundaries as three explicit curriculum states:
growth (updates 0–127), stability (128–511), and regeneration (512 onward).
Each state has its own progress indicator and live caption. Manual damage stays
locked through the first two phases so an accidental cut cannot be mistaken
for a regeneration test.

Selected outputs replace the same pool entries they came from, allowing a state
to accumulate a long trajectory over many short BPTT windows. After warm-up,
the selected state with the highest target loss is replaced by a fresh center
seed. Damage is sampled from the healthier selected states using random circles
whose radius ranges from roughly 10% to 20% of the grid width. This includes
the centered four-cell-radius cut made by the demo's Damage button. Rollouts
whose absolute state exceeds `poolValueLimit` are also replaced by a seed,
preventing one numerically runaway state from poisoning the pool.

The demo continues its bounded health readbacks during the fifteen-second
manual-damage grace period. Numerical divergence can still trigger a reset,
but normal target-loss drift cannot interrupt repair. Once the visible loss
returns near its pre-damage value, the page reports that regeneration
succeeded. Once the regeneration phase starts, the centered Damage button and
click-or-drag erasing on the live Automata canvas use the same four-cell-radius
cut that appears in training. Pointer coordinates are transformed through
Automata's camera and zoom, and drag segments are interpolated so fast movement
does not leave gaps.

## Custom targets

The demo includes a flower plus a small locally bundled set of
[Twemoji](https://github.com/twitter/twemoji) targets. The artwork is converted
in the browser to a centered 24×24 image, then its straight-alpha canvas pixels
are converted to the premultiplied RGBA tensor expected by the trainer. Twemoji
is licensed under CC BY 4.0; its bundled attribution is next to the demo assets.

The Upload image control accepts a PNG, JPEG, WebP, GIF, or SVG up to 12 MB.
The file is decoded and downsampled entirely in the browser and is never sent
to a server. Both a preset change and an upload create a fresh trainer because
an organism trained for one target is not a useful continuation point for an
unrelated target. The live Automata instance is retained: only its learned
weights and center seed are replaced.

The task is iterative, not an instant classifier. The 24×24, 128-hidden demo
begins forming the target within hundreds of iterations and becomes more
stable over thousands. Its 64–96 generation rollouts match the reference
training horizon but cost proportionally more GPU work than the compact
defaults.

## Options

| Option | Default | Constraint | Meaning |
| --- | ---: | ---: | --- |
| `size` | 24 | 8–64 | square grid edge |
| `channels` | 16 | 4–32 | cell state width; channel 3 is alpha |
| `hidden` | 64 | 1–128 | shared MLP hidden width |
| `batchSize` | 2 | 1–8 | independent rollouts per update |
| `rolloutSteps` | 24 | 1–96 | maximum BPTT generations |
| `minRolloutSteps` | `rolloutSteps` | 1–`rolloutSteps` | minimum sampled BPTT generations |
| `stabilitySteps` | 1 | 1–`minRolloutSteps` | consecutive terminal states included in loss |
| `fireRate` | 0.5 | 0–1 | per-cell update probability |
| `stepSize` | 1 | 0–2 | residual multiplier |
| `aliveThreshold` | 0.1 | 0–1 | alpha neighborhood threshold |
| `learningRate` | 0.002 | 0–1 | Adam learning rate |
| `poolSize` | max(8, batch × 4) | 1–256 | retained evolved states |
| `poolWarmupIterations` | 128 | 0–10,000 | seed-only updates |
| `poolValueLimit` | 256 | 1–1,000 | runaway-state replacement bound |
| `damageProbability` | 0.35 | 0–1 | chance to erase a healthy sampled state |

The trainer requires WebGPU and has no fallback. This is deliberate: the
generation tape and spatial reverse kernels are a GPU specialization, not the
generic `Session` graph contract.
