# Backends

All backends consume one validated `ExecutionPlan` and expose one asynchronous
`Session` contract.

| Backend | Precision | Inference | Training | Intended use |
| --- | --- | --- | --- | --- |
| Paper | `f32`, `f64` | yes | yes | correctness, education, diagnosis |
| CPU | `f32` | yes | yes | universal production fallback |
| Wasm | `f32` | yes | yes | explicit portable compiled runtime |
| WebGPU | `f32` | yes | yes | repeated parallel browser workloads |

## Paper

Paper traverses the semantic topology and keeps intermediate operations close
to the published LSTM-g equations. It intentionally avoids compiler
specialization. Use it for small golden cases and debugging disagreement
between optimized backends.

Paper is the only backend that accepts `f64`. It is never auto-selected.

## CPU

CPU uses the compiled structure-of-arrays plan and persistent typed arrays. It
has a generic path for portable graphs and a dense-stage specialization.
Shared-parameter updates are reduced deterministically.

CPU is the default for small workloads because it has no adapter, compilation,
dispatch, or readback cost.

## WebAssembly

Wasm is a precompiled persistent AssemblyScript runtime. The plan and mutable
arrays are initialized once and retained in linear memory; input, target, and
output use buffered boundaries. The package includes:

- `runtime.wasm`, the portable scalar artifact;
- `runtime-simd.wasm`, an optimized artifact that explicitly requires Wasm
  SIMD;
- generated JavaScript and TypeScript bindings for the scalar runtime.

Instantiation tries SIMD first and falls back to scalar. The selected variant is
available on `WasmSession.variant`.

The current generic sparse interpreter benchmarks slower than modern JITed
TypeScript for representative plans, so auto-selection leaves Wasm disabled
until specialized kernels justify changing that policy. Explicit
`backend: "wasm"` remains supported.

Run the local comparison with:

```sh
npm run benchmark:wasm
```

## WebGPU

WebGPU uses:

- one packed `array<u32>` storage heap with typed WGSL accessors;
- dynamic uniform records for per-stage dispatch;
- separate prepare, forward, gather, training-clear, backward, and update
  compute pipelines;
- one dispatch boundary per semantic stage;
- deterministic one-invocation-per-parameter gradient reduction;
- explicit output, snapshot, and checkpoint readback;
- device-limit validation, error scopes, device-loss handling, and teardown.

It requires `navigator.gpu`, an adapter, and a secure browser context. Browser
support and device limits are runtime properties, so applications must expect a
fallback.

```ts
const session = await compileModel(definition, {
  backend: "webgpu",
  training: true,
  fallback: ["wasm", "cpu"],
});
```

Set `fallback: []` to require WebGPU and receive `NO_SUPPORTED_BACKEND` instead
of changing backend.

## Auto-selection

The public facade chooses:

1. WebGPU when the plan is supported, has at least 64 units, and
   `connectionCount * expectedSteps` is at least 32,768;
2. Wasm only if its exported selection policy is explicitly enabled;
3. CPU otherwise.

This is a transparent heuristic, not a benchmark oracle. Call `selectBackend`
to inspect the choice and reason, or select a backend explicitly for stable
deployment behavior.

## Support failures

Every backend reports structured support issues. Production backends reject
`f64`. Training backends reject trainable recurrent self-connections. WebGPU
also validates storage size, buffer size, binding counts, and dispatch limits
before execution.

Fallback does not suppress all failures: invalid models and corrupted snapshots
are rejected by core validation before backend selection.
