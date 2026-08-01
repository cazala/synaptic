import { invariant } from "./errors.js";
import { CounterPrng } from "./prng.js";
import type { ModelDefinition, ModelSnapshot } from "./types.js";
import { validateDefinition } from "./validate.js";

export function createSnapshot(definition: ModelDefinition, seed = 0x51a7_1c): ModelSnapshot {
  validateDefinition(definition);
  const random = new CounterPrng(seed);
  const parameters = definition.precision === "f64"
    ? new Float64Array(definition.topology.parameters.length)
    : new Float32Array(definition.topology.parameters.length);

  for (const parameter of definition.topology.parameters) {
    const initializer = parameter.initializer;
    let value: number;
    switch (initializer.kind) {
      case "constant":
        value = initializer.value;
        break;
      case "uniform":
        invariant(initializer.min <= initializer.max, "Uniform initializer bounds are reversed", "INVALID_INITIALIZER");
        value = initializer.min + random.nextFloat() * (initializer.max - initializer.min);
        break;
    }
    parameters[parameter.id] = value;
  }

  return { definition, parameters };
}
