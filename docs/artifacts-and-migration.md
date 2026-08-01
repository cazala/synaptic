# Artifacts and migration

## Artifact levels

| Artifact | Contains | Resume recurrent execution |
| --- | --- | --- |
| Definition | topology and initialization rules | no |
| Snapshot | definition and parameters | no |
| Checkpoint | snapshot, runtime, optimizer, PRNG, step | yes |
| Compiled cache | backend-specific disposable machinery | not portable |

Definitions, snapshots, and checkpoints are portable. Execution plans and
compiled caches can always be regenerated and are not authoritative data.

## JSON manifest and binary sidecars

`saveSnapshot` and `saveCheckpoint` return an `ArtifactBundle`:

```ts
interface ArtifactBundle {
  manifest: ArtifactManifest;
  buffers: ReadonlyMap<string, Uint8Array>;
}
```

Persist `manifest` as JSON. Persist each map entry under its manifest buffer ID.
The manifest declares each buffer's role, optional name, dtype, element count,
byte count, and checksum. Numeric values are little-endian.

Checkpoint runtime buffers are named independently, and optimizer buffers use
their own names. This permits streaming sidecars and replacing storage
transport without changing model semantics.

Loading validates:

- artifact, model, topology, and algorithm versions;
- model structure and stage rules;
- one parameter buffer;
- unique buffer IDs and named runtime/optimizer buffers;
- dtype and declared byte/element lengths;
- sidecar presence and checksum;
- finite numeric values;
- checkpoint runtime dimensions and counters.

## Snapshot example

```ts
import {
  createSnapshot,
  loadSnapshot,
  saveSnapshot,
} from "synaptic";

const bundle = saveSnapshot(createSnapshot(definition, 42));
const snapshot = loadSnapshot(bundle);
```

## Checkpoint example

```ts
import {
  compileModel,
  loadCheckpoint,
  saveCheckpoint,
} from "synaptic";

const session = await compileModel(definition, {
  backend: "cpu",
  training: true,
});
await session.trainStep(
  { input: [1, 0], target: [1] },
  { learningRate: 0.05 },
);

const bundle = saveCheckpoint(await session.checkpoint());
session.dispose();

const checkpoint = loadCheckpoint(bundle);
const resumed = await compileModel(checkpoint, {
  backend: "wasm",
  training: true,
});
```

Passing a checkpoint to `compileModel` restores it automatically. You can also
call `session.restore(checkpoint)` on an existing compatible session.

## Import Synaptic v1

Use the original result of `Network.toJSON()`:

```ts
import { importLegacy } from "@synaptic/compat-v1";

const result = importLegacy(v1Json);
const { definition, snapshot, checkpoint } = result.value;
console.table(result.warnings);
```

The importer:

- maps v1 neuron squash names to portable activation IDs;
- converts bias fields to weighted constant-one connections;
- assigns explicit stages and delays;
- restores available neuron state;
- preserves gated connections;
- reports trace dictionaries that cannot be recovered because v1 JSON omitted
  stable runtime connection IDs.

## Migration policy

Importers are one-way boundaries. After import:

1. inspect all warnings;
2. compare representative outputs against the original runtime;
3. save the result as a v2 snapshot or checkpoint;
4. stop depending on the legacy JSON shape.

The old mutable v1 runtime and build toolchain are intentionally not included in
the v2 branch.
