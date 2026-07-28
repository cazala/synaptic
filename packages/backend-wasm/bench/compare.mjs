import { performance } from "node:perf_hooks";
import { CpuBackend } from "@synaptic/backend-cpu";
import { WasmBackend } from "@synaptic/backend-wasm";
import { compilePlan, createSnapshot } from "@synaptic/core";
import { dense, input, sequential } from "@synaptic/layers";

const definition = sequential(
  input({ size: 64 }),
  dense({ units: 64, activation: "tanh" }),
  dense({ units: 16, activation: "logistic" }),
);
const plan = compilePlan(definition);
const snapshot = createSnapshot(definition, 1234);
const values = new Float32Array(64).fill(0.25);
const iterations = 2_000;

async function measure(backend) {
  const compileStarted = performance.now();
  const session = await backend.compile(plan, snapshot);
  const compileMilliseconds = performance.now() - compileStarted;
  for (let iteration = 0; iteration < 100; iteration += 1) {
    await session.forward(values);
  }
  const runStarted = performance.now();
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    await session.forward(values);
  }
  const runMilliseconds = performance.now() - runStarted;
  session.dispose();
  return {
    backend: backend.id,
    compileMilliseconds: compileMilliseconds.toFixed(2),
    stepsPerSecond: Math.round(iterations * 1_000 / runMilliseconds),
  };
}

console.table(await Promise.all([
  measure(new CpuBackend()),
  measure(new WasmBackend()),
]));
