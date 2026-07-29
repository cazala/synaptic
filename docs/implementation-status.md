# v2 implementation status

The architecture proposal has been implemented on the `v2` branch in seven
milestones.

| Phase | Result |
| --- | --- |
| Foundation | TypeScript workspace, versioned schemas, validation, graph builder, execution plan, PRNG, async backend/session contracts |
| Semantic oracle | Paper implementation of LSTM-g and typed-array CPU runtime with shared conformance fixtures |
| WebAssembly | Persistent AssemblyScript heap, buffered boundaries, scalar and SIMD artifacts, benchmark policy |
| WebGPU inference | Packed heap, WGSL stage scheduling, recurrent context, fallback, lifecycle and browser validation |
| WebGPU training | Backward/error passes, traces, deterministic shared-parameter reduction, checkpoint readback |
| Specialization/migration | Dense CPU specialization, Synaptic v1 importer, public facade and backend selection |
| Hardening | Full checkpoint restore/artifacts, comprehensive tests/docs, package cleanup, coverage and dependency audit |

ASM.js is intentionally absent. Modern JavaScript engines, WebAssembly, and
WebGPU cover its former role without carrying a generated-source backend.

Future work should be driven by measured use cases:

- more layer macros built on the existing graph schema;
- batched tensor shapes and training APIs;
- specialized Wasm kernels before enabling Wasm auto-selection;
- larger WebGPU fusion strategies where conformance permits them;
- pluggable costs and optimizers with versioned portable state;
- worker/session hosts and cancellation;
- optional `f16` as an explicit capability, never a silent downgrade.
