# Synaptic v2

Synaptic v2 is a TypeScript-first neural-network runtime for architecture-free
graphs. It preserves Synaptic's distinctive units, weighted connections, and
connection gates while separating portable model data from backend execution.

The same immutable model can run on:

- `paper`: a readable implementation of the LSTM-g equations and the
  correctness oracle;
- `cpu`: the portable TypeScript production backend;
- `wasm`: a persistent AssemblyScript/WebAssembly runtime with scalar and SIMD
  artifacts;
- `webgpu`: a WGSL compute backend for repeated, parallel workloads, with
  explicit fallback.

This branch is a `2.0.0-alpha.0` development line. Its artifact formats are
versioned, but the public API may still change before v2 is stable.

## Quick start

The current repository requires Node.js 20 or newer.

```sh
npm install
npm run check
```

Create and run a model:

```ts
import {
  compileModel,
  dense,
  input,
  lstm,
  sequential,
} from "synaptic";

const definition = sequential(
  input({ size: 2 }),
  lstm({ units: 8, peepholes: true }),
  dense({ units: 1, activation: "logistic" }),
);

const session = await compileModel(definition, {
  backend: "auto",
  seed: 42,
  expectedSteps: 10_000,
});

const output = await session.forward([1, 0]);
console.log(output.data);

const metrics = await session.trainStep(
  { input: [0, 1], target: [1] },
  { learningRate: 0.05 },
);
console.log(metrics);

session.dispose();
```

All session operations are asynchronous, even on CPU. This gives every backend
one honest API and avoids changing application structure when work moves to
WebAssembly or WebGPU.

## Portable state

Definitions contain topology, not devices or executable code. A snapshot adds
parameters. A checkpoint also adds recurrent state, eligibility traces,
optimizer slots, PRNG state, and the logical step:

```ts
import {
  compileModel,
  loadCheckpoint,
  saveCheckpoint,
} from "synaptic";

const first = await compileModel(definition, { backend: "cpu", seed: 42 });
await first.forward([1, 0]);

const bundle = saveCheckpoint(await first.checkpoint());
first.dispose();

// Persist bundle.manifest as JSON and each entry in bundle.buffers as a
// separate binary sidecar.
const checkpoint = loadCheckpoint(bundle);
const resumed = await compileModel(checkpoint, { backend: "wasm" });
```

Artifacts use a versioned JSON manifest, little-endian binary buffers, declared
dtypes and lengths, and checksums. Checkpoints can resume on a different
compatible backend.

## Authoring unusual graphs

Layer helpers are graph macros. The lower-level `GraphBuilder` remains available
for shared parameters, custom recurrence, and second-order gating:

```ts
import { GraphBuilder, compileModel } from "synaptic";

const graph = new GraphBuilder();
const inputs = graph.input(2);
graph.nextStage();
const outputs = graph.units(2, { activation: "tanh" });
const shared = graph.parameter({
  initializer: { kind: "constant", value: 0.25 },
  label: "shared-weight",
});

graph.connect(inputs.units, outputs, "all-to-all", { parameter: shared });
const definition = graph.build({ inputs, outputs });
const session = await compileModel(definition);
```

Stages define deterministic ordering. A zero-delay connection reads the current
step and must point to a later stage. A one-delay connection reads the previous
step. Gates follow the same explicit delay rule.

## Choosing a backend

`backend: "auto"` currently chooses CPU for small or generic workloads and
WebGPU when a sufficiently large model will be reused enough to amortize setup.
Wasm is fully supported when selected explicitly; the generic sparse Wasm
interpreter is not auto-selected until specialization benchmarks justify it.
Paper is never chosen automatically.

```ts
const cpu = await compileModel(definition, { backend: "cpu" });
const wasm = await compileModel(definition, { backend: "wasm" });
const gpu = await compileModel(definition, {
  backend: "webgpu",
  fallback: ["wasm", "cpu"],
});
```

WebGPU requires a browser with WebGPU in a secure context (`https:` or
`localhost`). Explicit WebGPU compilation uses the declared fallback order when
the API, adapter, plan feature, or device limit is unavailable.

## Legacy imports

`@synaptic/compat-v1` imports Synaptic v1 `Network.toJSON()` data:

```ts
import { compileModel, importLegacy } from "synaptic";

const imported = importLegacy(JSON.parse(json));
console.log(imported.warnings);

const session = await compileModel(imported.value.checkpoint, {
  backend: "cpu",
});
```

Warnings are explicit where the old format did not carry enough information for
lossless conversion.

## Packages

| Package | Purpose |
| --- | --- |
| `synaptic` | Public facade, backend selection, and re-exports |
| `@synaptic/core` | Schema, builder, validation, plans, artifacts, contracts |
| `@synaptic/layers` | Input, dense, LSTM, and composition macros |
| `@synaptic/backend-paper` | Equation-aligned oracle |
| `@synaptic/backend-cpu` | Portable typed-array runtime |
| `@synaptic/backend-wasm` | Persistent scalar/SIMD WebAssembly runtime |
| `@synaptic/backend-webgpu` | WGSL compute runtime and fallbacks |
| `@synaptic/compat-v1` | Synaptic v1 importer |
| `@synaptic/conformance` | Shared backend fixtures and parity tests |

## Documentation

- [Getting started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [Backend behavior](docs/backends.md)
- [Artifacts and migration](docs/artifacts-and-migration.md)
- [Testing](docs/testing.md)
- [Contributing](docs/contributing.md)
- [Original v2 research proposal](docs/v2-architecture-proposal.md)

## License

MIT. See [LICENSE](LICENSE).
