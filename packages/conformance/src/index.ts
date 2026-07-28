import {
  GraphBuilder,
  compilePlan,
  createSnapshot,
  type Backend,
  type ModelCheckpoint,
  type ModelDefinition,
  type ModelSnapshot,
  type TrainingBatch,
} from "@synaptic/core";
import { dense, input, lstm, sequential } from "@synaptic/layers";

export interface ConformanceFixture {
  readonly definition: ModelDefinition;
  readonly snapshot: ModelSnapshot;
  readonly sequence: readonly TrainingBatch[];
}

export interface ConformanceResult {
  readonly backend: string;
  readonly losses: readonly number[];
  readonly outputs: readonly Float32Array[];
  readonly checkpoint: ModelCheckpoint;
}

export function feedForwardFixture(): ConformanceFixture {
  const definition = sequential(
    input({ size: 2 }),
    dense({ units: 3, activation: "tanh", label: "hidden" }),
    dense({ units: 1, activation: "logistic", label: "output" }),
  );
  return {
    definition,
    snapshot: createSnapshot(definition, 0xfeed_1234),
    sequence: [
      { input: [0, 0], target: [0] },
      { input: [0, 1], target: [1] },
      { input: [1, 0], target: [1] },
      { input: [1, 1], target: [0] },
    ],
  };
}

export function recurrentFixture(): ConformanceFixture {
  const definition = sequential(
    input({ size: 2 }),
    lstm({ units: 2, label: "memory" }),
    dense({ units: 1, activation: "logistic", label: "output" }),
  );
  return {
    definition,
    snapshot: createSnapshot(definition, 0x1a57_600d),
    sequence: [
      { input: [1, 0], target: [0] },
      { input: [0, 1], target: [1] },
      { input: [0.5, -0.25], target: [0.75] },
      { input: [-0.5, 0.25], target: [0.25] },
    ],
  };
}

export function sharedParameterFixture(): ConformanceFixture {
  const graph = new GraphBuilder();
  const inputPort = graph.input(2);
  graph.stage(1);
  const output = graph.units(2, {
    activation: "identity",
    label: "shared-output",
  });
  const shared = graph.parameter({
    initializer: { kind: "constant", value: 0.25 },
    label: "shared-kernel",
  });
  graph.connect(inputPort.units, output, "all-to-all", {
    parameter: shared,
  });
  const definition = graph.build({
    inputs: inputPort,
    outputs: output,
    metadata: { fixture: "shared-parameter" },
  });
  return {
    definition,
    snapshot: createSnapshot(definition, 1),
    sequence: [
      { input: [2, 3], target: [1, -1] },
    ],
  };
}

export async function runConformance(
  backend: Backend,
  fixture: ConformanceFixture,
  options: { readonly train?: boolean; readonly learningRate?: number } = {},
): Promise<ConformanceResult> {
  const plan = compilePlan(fixture.definition);
  const session = await backend.compile(plan, fixture.snapshot, {
    training: options.train ?? false,
  });
  const outputs: Float32Array[] = [];
  const losses: number[] = [];
  try {
    for (const batch of fixture.sequence) {
      if (options.train) {
        const metrics = await session.trainStep(batch, {
          ...(options.learningRate === undefined ? {} : { learningRate: options.learningRate }),
        });
        losses.push(metrics.loss);
        const checkpoint = await session.checkpoint();
        const output = new Float32Array(plan.outputs.length);
        plan.outputs.forEach((unit, slot) => {
          output[slot] = checkpoint.runtime.activation[unit] ?? 0;
        });
        outputs.push(output);
      } else {
        outputs.push((await session.forward(batch.input)).data);
      }
    }
    return {
      backend: session.backend,
      losses,
      outputs,
      checkpoint: await session.checkpoint(),
    };
  } finally {
    session.dispose();
  }
}
