# Testing

## Commands

```sh
npm test
npm run typecheck
npm run build
npm run test:coverage
npm run benchmark:wasm
```

`npm run check` runs type checking and the test suite. Production TypeScript
builds exclude `*.test.ts`; a separate no-emit test configuration type-checks
the tests against source aliases.

## Test layers

The suite covers:

- graph construction, stage validation, shared parameters, and plan indexes;
- deterministic PRNG and initialization;
- snapshot/checkpoint round trips, corruption, dimensions, and checksums;
- hand-calculated Paper equations;
- CPU/Paper/Wasm inference and training parity;
- recurrent state, gates, eligibility traces, extended traces, and errors;
- tied-parameter gradient reduction;
- cross-backend checkpoint restoration;
- portable ordered-sequence inference and training contracts;
- seeded XOR, MNIST, and discrete sequence recall learning;
- WebGPU heap offsets, parameter adjacency, and uniform records;
- Growing Neural CA tape/scratch layout, reference initialization, shader
  compilation, loss descent, and Automata artifact acceptance;
- fallback ordering, disabled fallback, ownership, and disposal;
- Synaptic v1 fixture import and unsupported-format rejection;
- public facade and auto-selection behavior.

## Conformance

`@synaptic/conformance` owns small deterministic fixtures. Each backend is
compared at semantic boundaries rather than only final output:

- output per step;
- mean squared loss;
- final parameters;
- state;
- eligibility and extended eligibility traces;
- projected, gated, and total error.

Paper is the reference. Production f32 comparisons use tolerances appropriate
for differences in legal floating-point operation ordering.

## Learning workloads

`@synaptic/conformance` contains three deterministic functional workloads. The
regular test suite runs each one from the same seeded initialization on Paper,
CPU, and Wasm:

- XOR trains a 2-3-1 perceptron and must classify all four inputs with outputs
  beyond the `0.1`/`0.9` margins.
- MNIST uses `mnist@1.1.0`, takes 15 training and 5 disjoint test images for
  each of all ten classes, downsamples them from 28×28 to 7×7, and requires at
  least 85% training and 80% held-out accuracy.
- Discrete sequence recall reproduces the `gh-pages` demo vocabulary: targets
  2 and 4, distractors 3 and 5, prompts 0 and 1, and length-10 sequences. A
  seeded 4→10 length curriculum trains a 6-4-2 LSTM for 1,375 trials at each
  length; 100 separately seeded length-10 validation sequences require at least
  95% prompt and whole-sequence accuracy.

These are bounded regression tests, not representative accuracy benchmarks.
The dataset split, ordering, initialization, and validation sequences are fixed
so a backend regression cannot hide behind favorable random sampling.
The default test reporter names the backend in every learning test and prints a
summary table with per-workload elapsed time and achieved accuracy. WebGPU is
intentionally absent from that Node table: its no-fallback browser page reports
its own per-workload timings alongside the hardware results.

## Browser WebGPU smoke tests

WebGPU needs a real adapter; Node-only CI tests validate layout, support
reporting, fallback, and resource ownership with no fake numerical GPU.

Build and serve a local index of the hardware tests:

```sh
npm run test:webgpu
```

Then open `http://127.0.0.1:4173/` and choose a test:

- `http://localhost:4173/packages/backend-webgpu/test/xor.html`
- `http://localhost:4173/packages/backend-webgpu/test/mnist.html`
- `http://localhost:4173/packages/backend-webgpu/test/mnist-automata.html`
- `http://localhost:4173/packages/backend-webgpu/test/dsr.html`
- `http://localhost:4173/packages/backend-webgpu/test/learn-to-paint.html`
- `http://localhost:4173/packages/backend-webgpu/test/growing-neural-ca.html`
- `http://localhost:4173/packages/backend-webgpu/test/growing-neural-ca-smoke.html`
- `http://localhost:4173/packages/backend-webgpu/test/shader-smoke.html`
- `http://localhost:4173/packages/backend-webgpu/test/forward-smoke.html`
- `http://localhost:4173/packages/backend-webgpu/test/training-parity.html`
- `http://localhost:4173/packages/backend-webgpu/test/learning-workloads.html`

The command builds every workspace first. To use another port:

```sh
npm run test:webgpu -- --port 8080
```

The pages compile every WGSL entry point and compare forward, recurrent,
training, ordered-sequence, trace, and shared-parameter behavior with CPU. The
learning page runs the same XOR, MNIST, and sequence-recall helpers as the Node
suite with fallback disabled, so its reported backend must be `webgpu`; it
reports end-to-end time for each workload, including model compilation.

The first six pages are polished interactive examples:

- XOR shows the 2→3→1 topology and all four predictions as training converges.
- MNIST trains a 196→128→10 classifier on 300 MNIST samples plus 80
  drawing-oriented examples, evaluates 100 held-out MNIST samples, and lets
  you draw a digit. Strokes are centered, scaled to 28×28, and averaged to the
  same 14×14 input representation used for training.
- MNIST to Neural CA trains and compares both Neural substrates from
  `@cazala/automata@0.2.0`. Direct mode uses a wrapped, symmetric 3×3
  convolution with only three tied parameters: corner, edge, and center.
  Network mode reproduces Automata's exact identity/Sobel-x/Sobel-y/symmetric
  perception order and trains a spatially shared 12→8→3 MLP. After frozen-rule
  readout warm-ups, WebGPU fine-tunes the transferable parameters. The mode
  switch injects either the three direct weights or the network kernel plus its
  row-major matrices and biases into a live three-channel Neural CA. Synaptic
  `tanh` and Automata activation `1` are equivalent. Matrix heatmaps make the
  network artifact visible, and held-out digit seeds or independent noise
  reveal each rule's recurrent dynamics. The 10-way classifier readout is
  deliberately not transferred because it is not part of either local CA rule.
- Sequence recall trains the length 4→10 DSR curriculum, then animates random
  length-10 sequences one symbol at a time. Colored inputs and outputs make
  targets, distractors, prompts, silence, and recall errors visible.
- Learn to paint is inspired by the original gh-pages demo. A 180→48→3 neural
  field receives only a Fourier encoding of `(x, y)` coordinates and
  continuously learns and displays the bundled portrait at its native 400×400.
  Play/pause controls the loop. Its UI reports sampled error, elapsed time, and
  GPU submissions so batching is observable: 4,096 distributed
  full-resolution updates cross the queue in one training submission per
  epoch, while 8,100 evenly distributed preview samples return through one
  inference submission and readback. The network preview is smoothly resolved
  into the native 400×400 output canvas every epoch.
- Growing Neural CA unrolls an exact 16-channel local rule through a
  48-generation WebGPU tape, backpropagates the image loss without f32
  atomics, applies per-tensor gradient normalization and Adam, and trains
  growth, persistence, and damage repair from a bounded sample pool. Every
  fourth update streams its four tensors into Automata's live
  `GrowingNeural` simulation. The adjacent smoke page verifies numerical loss
  descent and that Automata accepts the exported artifact.

Browser console errors or non-pass smoke results fail the hardware check.

## Adding backend behavior

For a semantic change:

1. add or update a hand-calculated Paper test;
2. add it to a shared conformance fixture;
3. implement CPU and verify parity;
4. implement Wasm and verify scalar/SIMD behavior;
5. implement WebGPU and run the hardware smoke pages;
6. add artifact compatibility tests if state or schema changed.

Do not loosen parity tolerances to hide a deterministic semantic disagreement.
