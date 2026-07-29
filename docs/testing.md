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
- seeded XOR, MNIST, and discrete sequence recall learning;
- WebGPU heap offsets and uniform records;
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
  seeded length curriculum trains a 6-4-2 LSTM; 100 separately seeded
  validation sequences require at least 95% prompt and whole-sequence
  accuracy.

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

Build and serve the repository over localhost:

```sh
npm run build
python3 -m http.server 4173
```

Then open:

- `http://localhost:4173/packages/backend-webgpu/test/shader-smoke.html`
- `http://localhost:4173/packages/backend-webgpu/test/forward-smoke.html`
- `http://localhost:4173/packages/backend-webgpu/test/training-parity.html`
- `http://localhost:4173/packages/backend-webgpu/test/learning-workloads.html`

The pages compile every WGSL entry point and compare forward, recurrent,
training, trace, and shared-parameter behavior with CPU. The learning page runs
the same XOR, MNIST, and sequence-recall helpers as the Node suite with fallback
disabled, so its reported backend must be `webgpu`. Browser console errors or
non-pass results fail the smoke check.

## Adding backend behavior

For a semantic change:

1. add or update a hand-calculated Paper test;
2. add it to a shared conformance fixture;
3. implement CPU and verify parity;
4. implement Wasm and verify scalar/SIMD behavior;
5. implement WebGPU and run the hardware smoke pages;
6. add artifact compatibility tests if state or schema changed.

Do not loosen parity tolerances to hide a deterministic semantic disagreement.
