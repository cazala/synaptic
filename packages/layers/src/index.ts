import {
  GraphBuilder,
  invariant,
  type ActivationId,
  type ModelDefinition,
  type Port,
  type Precision,
  type UnitId,
} from "@synaptic/core";

export interface InputOptions {
  readonly size: number;
  readonly shape?: readonly number[];
}

export interface DenseOptions {
  readonly units: number;
  readonly activation?: ActivationId;
  readonly bias?: boolean;
  readonly label?: string;
}

export interface LstmOptions {
  readonly units: number;
  readonly peepholes?: boolean;
  readonly bias?: boolean;
  readonly label?: string;
}

export interface SequentialOptions {
  readonly precision?: Precision;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface LayerMacro {
  readonly kind: string;
  apply(graph: GraphBuilder, input?: Port): Port;
}

function requireInput(input: Port | undefined, kind: string): Port {
  invariant(input !== undefined, `${kind} requires an input port`, "LAYER_INPUT_REQUIRED");
  return input;
}

function gateOutbound(
  graph: GraphBuilder,
  input: Port,
  connections: readonly number[],
): void {
  if (input.outboundGate !== undefined) {
    graph.gate(input.outboundGate, connections, { mode: "one-to-one-source" });
  }
}

function connectFrom(
  graph: GraphBuilder,
  input: Port,
  targets: readonly UnitId[],
): readonly number[] {
  const connections = graph.connect(input.units, targets);
  gateOutbound(graph, input, connections);
  return connections;
}

function connectBias(
  graph: GraphBuilder,
  bias: UnitId,
  targets: readonly UnitId[],
  value: number,
  label: string,
): void {
  graph.connect([bias], targets, "all-to-all", {
    initializer: { kind: "constant", value },
    label,
  });
}

export function input(options: InputOptions): LayerMacro {
  return {
    kind: "input",
    apply(graph, previous) {
      invariant(previous === undefined, "The input layer must be first", "INPUT_LAYER_ORDER");
      return graph.input(options.size, options.shape ?? [options.size]);
    },
  };
}

export function dense(options: DenseOptions): LayerMacro {
  return {
    kind: "dense",
    apply(graph, previous) {
      const source = requireInput(previous, "Dense");
      graph.nextStage();
      const units = graph.units(options.units, {
        activation: options.activation ?? "logistic",
        label: options.label ?? "dense",
      });
      connectFrom(graph, source, units);
      if (options.bias ?? true) {
        const bias = graph.constant(1);
        connectBias(graph, bias, units, 0, `${options.label ?? "dense"}.bias`);
      }
      return { units, shape: [options.units] };
    },
  };
}

export function lstm(options: LstmOptions): LayerMacro {
  return {
    kind: "lstm",
    apply(graph, previous) {
      const source = requireInput(previous, "LSTM");
      const prefix = options.label ?? "lstm";

      graph.nextStage();
      const inputGate = graph.units(options.units, {
        activation: "logistic",
        label: `${prefix}.input-gate`,
      });
      graph.nextStage();
      const forgetGate = graph.units(options.units, {
        activation: "logistic",
        label: `${prefix}.forget-gate`,
      });
      graph.nextStage();
      const memory = graph.units(options.units, {
        activation: "tanh",
        label: `${prefix}.memory`,
      });
      graph.nextStage();
      const outputGate = graph.units(options.units, {
        activation: "logistic",
        label: `${prefix}.output-gate`,
      });

      connectFrom(graph, source, inputGate);
      connectFrom(graph, source, forgetGate);
      const memoryInputs = connectFrom(graph, source, memory);
      connectFrom(graph, source, outputGate);
      graph.gate(inputGate, memoryInputs, { mode: "one-to-one-target" });

      const recurrent = graph.connect(memory, memory, "one-to-one", {
        delay: 1,
        trainable: false,
        initializer: { kind: "constant", value: 1 },
        label: `${prefix}.memory-cell`,
      });
      graph.gate(forgetGate, recurrent, { mode: "one-to-one-target" });

      if (options.peepholes ?? true) {
        graph.connect(memory, inputGate, "one-to-one", {
          delay: 1,
          label: `${prefix}.input-peephole`,
        });
        graph.connect(memory, forgetGate, "one-to-one", {
          delay: 1,
          label: `${prefix}.forget-peephole`,
        });
        graph.connect(memory, outputGate, "one-to-one", {
          delay: 0,
          label: `${prefix}.output-peephole`,
        });
      }

      if (options.bias ?? true) {
        const bias = graph.constant(1);
        connectBias(graph, bias, inputGate, 0, `${prefix}.input-bias`);
        connectBias(graph, bias, forgetGate, 1, `${prefix}.forget-bias`);
        connectBias(graph, bias, memory, 0, `${prefix}.memory-bias`);
        connectBias(graph, bias, outputGate, 0, `${prefix}.output-bias`);
      }

      return {
        units: memory,
        shape: [options.units],
        outboundGate: outputGate,
      };
    },
  };
}

export function compose(
  layers: readonly LayerMacro[],
  options: SequentialOptions = {},
): ModelDefinition {
  invariant(layers.length >= 2, "A sequential model needs an input and an output layer", "EMPTY_MODEL");
  const graph = new GraphBuilder();
  let first: Port | undefined;
  let current: Port | undefined;
  for (const layer of layers) {
    current = layer.apply(graph, current);
    first ??= current;
  }
  invariant(first !== undefined && current !== undefined, "Failed to compose the model", "EMPTY_MODEL");
  return graph.build({
    inputs: first,
    outputs: current,
    ...(options.precision === undefined ? {} : { precision: options.precision }),
    ...(options.metadata === undefined ? {} : { metadata: options.metadata }),
  });
}

export function sequential(...layers: readonly LayerMacro[]): ModelDefinition {
  return compose(layers);
}
