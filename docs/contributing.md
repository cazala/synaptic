# Contributing

## Workspace

The repository is an npm workspace with strict TypeScript project references.
Use Node.js 20 or newer:

```sh
npm install
npm run check
```

Do not edit generated `packages/*/dist` output. The committed Wasm artifacts and
bindings are generated from `packages/backend-wasm/assembly`:

```sh
npm run build:wasm
```

Commit source and regenerated Wasm artifacts together when the AssemblyScript
runtime changes.

## Design rules

- Keep `ModelDefinition`, snapshots, and checkpoints serializable and
  backend-neutral.
- Put graph meaning in core validation and the execution plan, not in a
  backend-specific convention.
- Preserve explicit stage and delay semantics.
- Keep the public session API asynchronous.
- Add version fields before changing persisted meaning.
- Make unsupported behavior visible through `SupportReport`.
- Never silently reduce precision or switch an explicitly required backend.
- Keep Paper readable and equation-aligned.
- Avoid allocation in backend step loops.
- Preserve deterministic parameter-reduction order when weights are shared.

## Dependency boundaries

- Core must not import a backend.
- Layers are macros and depend only on core.
- Backend packages own execution and lifecycle only.
- The public facade owns default backend selection.
- Legacy parsing remains isolated in `@synaptic/compat-v1`.
- Conformance fixtures may depend on all backends but are not published.

## Pull request checks

Before submitting:

```sh
npm run clean
npm run build
npm run check
npm run test:coverage
npm audit
git diff --check
```

If WebGPU changed, also run all browser smoke pages on a hardware-backed
adapter. If Wasm changed, run `npm run benchmark:wasm` and record whether the
selection policy should change.

## Artifact compatibility

Treat artifact versions as API. A format change needs:

- a documented compatibility decision;
- validation for old and new forms;
- round-trip and corruption tests;
- a migration path or a clear unsupported-version error.

Compiled caches are disposable and need no migration.
