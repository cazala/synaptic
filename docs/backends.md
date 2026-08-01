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
- a parameter-to-connection adjacency index for deterministic linear gradient
  reduction, including tied parameters;
- packed ordered-sequence uploads with one command submission and one output
  readback, rather than one queue round trip per scalar step;
- optional loss suppression for asynchronous training sequences;
- one contiguous recurrent-state reset write;
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

Sequence submission removes CPU/GPU synchronization overhead but does not turn a
small recurrent graph into a wide parallel workload. Every semantic stage still
has a dispatch boundary, so tiny LSTM stages can remain slower than CPU even
after batching queue operations. Large independent-sample batching and fused
graph kernels are separate future specializations.

### Growing Neural CA specialization

`GrowingNeuralCaTrainer` is an explicit WebGPU specialization for
differentiable cellular automata. It does not pretend a large unrolled spatial
simulation is an ordinary `ExecutionPlan`: doing so would materialize millions
of graph connections and still miss image-state BPTT semantics.

The trainer instead owns:

- an f32 state tape for every batch, cell, channel, and generation;
- exact identity/Sobel-x/Sobel-y perception with zero padding;
- a shared ReLU pointwise MLP and stochastic residual update;
- separate candidate and pre/post alpha-life phases;
- reverse-time MLP and gathered convolution gradients, avoiding unavailable
  portable f32 atomics;
- per-tensor gradient normalization, Adam slots, and a bounded sample pool
  that replaces runaway rollouts with fresh center seeds;
- a versioned `artifact()` compatible with Automata `GrowingNeural`.

This API requires WebGPU and deliberately has no CPU fallback. The generic
backend/session contract remains unchanged.

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
