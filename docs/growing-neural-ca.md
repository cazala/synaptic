# Growing Neural Cellular Automata

`GrowingNeuralCaTrainer` is Synaptic's WebGPU specialization for the model in
[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/). It
trains a local rule in Synaptic and exports the same versioned artifact that
Automata runs for inference.

## Minimal training loop

```ts
import {
  GrowingNeuralCaCurriculum,
  GrowingNeuralCaTrainer,
} from "@synaptic/backend-webgpu";

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
});

// Row-major 24 × 24 premultiplied RGBA, with values normally in [0, 1].
const target = await loadPremultipliedTarget();
const curriculum = new GrowingNeuralCaCurriculum(target);

for (let iteration = 0; iteration < 20_000; iteration += 1) {
  const metrics = await trainer.trainStep(
    target,
    curriculum.trainingOptions(),
  );
  const state = curriculum.update(metrics);
  console.log(
    metrics.iteration,
    metrics.rolloutSteps,
    metrics.loss,
    metrics.durationMs,
    state.key,
    state.progress,
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

const inferenceSize = 48;
const automaton = GrowingNeural.fromArtifact(artifact);
const engine = new Engine({
  canvas,
  automaton,
  grid: {
    width: inferenceSize,
    height: inferenceSize,
    wrap: false,
    maxCells: inferenceSize,
  },
  stepsPerSecond: 60,
  render: { colorBg: { r: 1, g: 1, b: 1, a: 1 } },
});

await engine.initialize();
engine.coverGrid();
engine.reset({ mode: "center" });
engine.play();
```

The learned rule is local and does not encode a grid size. The browser demo
therefore keeps training at 24×24 but runs Automata on a centered 48×48 grid.
That gives the organism twice as much space in each direction and halves the
rendered cell size without changing the training tape or tensors.

While training continues, avoid rebuilding the Automata pipelines:

```ts
const next = await trainer.artifact();
automaton.setWeights({
  channels: next.channels,
  hidden: next.hidden,
  ...next.weights,
});
```

The browser demo also rate-limits a small `engine.getCells()` readback. It
center-pads the 24×24 target into the 48×48 inference grid before passing both
arrays to `measureGrowingNeuralCaState(...)`, then normalizes the padded loss
back to the training-area basis. It uses a wider live-state limit than the
training pool, then reseeds inference when values become non-finite, hidden
channels exceed that range, living cells saturate the grid, or a previously
formed organism collapses. This guard is intentionally outside the animation
loop; production rendering should not perform a full state readback per frame.

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
reseeded and the healthiest three are damage candidates. Unlike a fixed
iteration schedule, `GrowingNeuralCaCurriculum` advances from evidence in the
final states already returned by each training step:

1. growth measures fresh seed rollouts;
2. stability measures retained sample-pool states after another randomized
   64–96-generation rollout;
3. regeneration measures states after a broad random cut.

Every cohort reports mean visible loss and living-cell count. The curriculum
normalizes loss by the selected target's own RGBA energy and cell coverage, so
a sparse flower and a dense uploaded image are not judged against the same
absolute error. It requires a rolling set of successful observations rather
than accepting one lucky stochastic rollout. Regeneration confidence also
includes current growth and stability confidence, preventing repair from
passing after the model forgets an earlier skill.

The percentage and progress bar in the demo show this measured confidence.
Sample-pool training starts only after growth qualifies, damage probability
ramps from 25% to 100% as repair confidence rises, and the maintenance learning
rate begins only after regeneration is mastered. Manual damage stays locked
through the first two phases so an accidental cut cannot be mistaken for a
regeneration test. Mastery remains live rather than permanent: if the rolling
quality window regresses, confidence drops below 100% and the stronger repair
learning rate resumes until the organism qualifies again.

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
may qualify different targets at very different iterations. Its 64–96
generation rollouts match the reference training horizon but cost
proportionally more GPU work than the compact defaults.

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

`trainStep(...)` also accepts `useSamplePool` to let an adaptive controller
choose when persistence begins. Its `quality` metrics group final rollout
health into `seed`, `persistent`, and `damaged` cohorts without another GPU
readback; those final states were already mapped to update the sample pool.

The trainer requires WebGPU and has no fallback. This is deliberate: the
generation tape and spatial reverse kernels are a GPU specialization, not the
generic `Session` graph contract.
