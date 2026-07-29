# Getting started

## Requirements

- Node.js 20 or newer for workspace development and Node execution.
- A WebGPU-capable browser in a secure context for the WebGPU backend.
- No native compiler is required. The Wasm runtime is built with
  AssemblyScript from the workspace dependency.

Install and verify the workspace:

```sh
npm install
npm run check
```

## Build a model

Layer helpers return an immutable `ModelDefinition`:

```ts
import { dense, input, lstm, sequential } from "synaptic";

const definition = sequential(
  input({ size: 3 }),
  lstm({ units: 12, peepholes: true, label: "memory" }),
  dense({ units: 2, activation: "logistic", label: "output" }),
);
```

Available activation IDs are `identity`, `logistic`, `tanh`, `relu`, and
`step`. Dense layers use a bias by default. LSTM layers use biases and
peepholes by default.

For custom precision or metadata, use `compose`:

```ts
import { compose, dense, input } from "synaptic";

const diagnostic = compose(
  [
    input({ size: 2 }),
    dense({ units: 1, activation: "identity" }),
  ],
  {
    precision: "f64",
    metadata: { experiment: "diagnostic" },
  },
);
```

Only Paper supports portable `f64` execution. Production backends use `f32`.

## Initialize and compile

`compileModel` accepts a definition, a parameter snapshot, or a full
checkpoint:

```ts
import { compileModel, createSnapshot } from "synaptic";

const snapshot = createSnapshot(definition, 1234);
const session = await compileModel(snapshot, {
  backend: "cpu",
  training: true,
});
```

Initialization is deterministic for a given unsigned seed. The execution plan
is derived from the definition and is never persisted as authoritative model
data.

## Inference and training

One `forward` or `trainStep` call advances one logical recurrent step:

```ts
const result = await session.forward([0.1, 0.2, 0.3]);

const metrics = await session.trainStep(
  {
    input: [0.1, 0.2, 0.3],
    target: [1, 0],
  },
  { learningRate: 0.025 },
);
```

Inputs and targets accept arrays or `Float32Array`. Results are a
`Float32Array` plus a one-dimensional shape. The built-in training step uses
mean squared error and online LSTM-g updates.

Use `resetState()` between independent sequences. It clears recurrent state,
traces, errors, the logical step, and the PRNG counter; it does not reset
parameters:

```ts
await session.resetState();
```

Always release the backend:

```ts
session.dispose();
```

Calls after disposal fail with a `SESSION_DISPOSED` error.

## Save and resume

Use `snapshot()` when only learned parameters matter. Use `checkpoint()` when
execution must resume exactly:

```ts
import { loadCheckpoint, saveCheckpoint } from "synaptic";

const bundle = saveCheckpoint(await session.checkpoint());
const restored = loadCheckpoint(bundle);
const next = await compileModel(restored, { backend: "wasm" });
```

See [Artifacts and migration](artifacts-and-migration.md) for storage and legacy
conversion.

## Low-level construction

`GraphBuilder` supports structures that layer helpers do not cover:

```ts
import { GraphBuilder } from "synaptic";

const graph = new GraphBuilder();
const source = graph.input(2);
graph.nextStage();
const gate = graph.units(2, { activation: "logistic", label: "gate" });
graph.nextStage();
const target = graph.units(2, { activation: "tanh", label: "target" });

const projected = graph.connect(source.units, target, "one-to-one");
graph.gate(gate, projected, { mode: "one-to-one-target" });

const recurrent = graph.connect(target, target, "one-to-one", {
  delay: 1,
  trainable: false,
  initializer: { kind: "constant", value: 1 },
});
graph.gate(gate, recurrent, {
  mode: "one-to-one-target",
  delay: 1,
});

const custom = graph.build({ inputs: source, outputs: target });
```

The builder validates IDs, stages, delays, gates, parameter storage, ports, and
acyclic zero-delay ordering when the definition is built or compiled.
