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

The pages compile every WGSL entry point and compare forward, recurrent,
training, trace, and shared-parameter behavior with CPU. Browser console errors
or non-pass results fail the smoke check.

## Adding backend behavior

For a semantic change:

1. add or update a hand-calculated Paper test;
2. add it to a shared conformance fixture;
3. implement CPU and verify parity;
4. implement Wasm and verify scalar/SIMD behavior;
5. implement WebGPU and run the hardware smoke pages;
6. add artifact compatibility tests if state or schema changed.

Do not loosen parity tolerances to hide a deterministic semantic disagreement.
