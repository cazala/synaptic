export const FORWARD_WGSL = /* wgsl */ `
struct Config {
  unitCount: u32,
  stageStart: u32,
  stageEnd: u32,
  stageUnits: u32,
  unitActivation: u32,
  unitConstant: u32,
  unitConstantMask: u32,
  inputSlot: u32,
  connectionFrom: u32,
  connectionParameter: u32,
  connectionDelay: u32,
  connectionGater: u32,
  gateDelay: u32,
  incomingOffsets: u32,
  incomingConnections: u32,
  parameters: u32,
  state: u32,
  activation: u32,
  previousActivation: u32,
  input: u32,
  outputs: u32,
  output: u32,
  outputCount: u32,
  padding0: u32,
  padding1: u32,
  padding2: u32,
  padding3: u32,
  padding4: u32,
}

@group(0) @binding(0) var<storage, read_write> heap: array<u32>;
@group(0) @binding(1) var<uniform> config: Config;

fn readF32(offset: u32, index: u32) -> f32 {
  return bitcast<f32>(heap[offset + index]);
}

fn writeF32(offset: u32, index: u32, value: f32) {
  heap[offset + index] = bitcast<u32>(value);
}

fn connectionGain(connection: u32) -> f32 {
  let gater = bitcast<i32>(heap[config.connectionGater + connection]);
  if (gater < 0) {
    return 1.0;
  }
  if (heap[config.gateDelay + connection] == 0u) {
    return readF32(config.activation, u32(gater));
  }
  return readF32(config.previousActivation, u32(gater));
}

fn sourceValue(connection: u32) -> f32 {
  let source = heap[config.connectionFrom + connection];
  if (heap[config.connectionDelay + connection] == 0u) {
    return readF32(config.activation, source);
  }
  return readF32(config.previousActivation, source);
}

fn squash(code: u32, value: f32) -> f32 {
  switch code {
    case 1u: {
      return 1.0 / (1.0 + exp(-value));
    }
    case 2u: {
      return tanh(value);
    }
    case 3u: {
      return max(0.0, value);
    }
    case 4u: {
      return select(0.0, 1.0, value > 0.0);
    }
    default: {
      return value;
    }
  }
}

@compute @workgroup_size(64)
fn prepare(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let unit = invocation.x;
  if (unit >= config.unitCount) {
    return;
  }
  writeF32(
    config.previousActivation,
    unit,
    readF32(config.activation, unit),
  );
}

@compute @workgroup_size(64)
fn forwardStage(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let localIndex = invocation.x;
  let stageLength = config.stageEnd - config.stageStart;
  if (localIndex >= stageLength) {
    return;
  }
  let unit = heap[config.stageUnits + config.stageStart + localIndex];
  let inputSlot = bitcast<i32>(heap[config.inputSlot + unit]);
  let constant = readF32(config.unitConstant, unit);

  if (inputSlot >= 0) {
    let value = readF32(config.input, u32(inputSlot));
    writeF32(config.state, unit, value);
    writeF32(config.activation, unit, value);
    return;
  }
  if (heap[config.unitConstantMask + unit] != 0u) {
    writeF32(config.state, unit, constant);
    writeF32(config.activation, unit, constant);
    return;
  }

  let previousState = readF32(config.state, unit);
  var nextState = 0.0;
  let incomingStart = heap[config.incomingOffsets + unit];
  let incomingEnd = heap[config.incomingOffsets + unit + 1u];
  for (var cursor = incomingStart; cursor < incomingEnd; cursor += 1u) {
    let connection = heap[config.incomingConnections + cursor];
    let source = heap[config.connectionFrom + connection];
    let parameter = heap[config.connectionParameter + connection];
    let weight = readF32(config.parameters, parameter);
    let gain = connectionGain(connection);
    if (source == unit) {
      nextState += gain * weight * previousState;
    } else {
      nextState += gain * weight * sourceValue(connection);
    }
  }
  writeF32(config.state, unit, nextState);
  writeF32(
    config.activation,
    unit,
    squash(heap[config.unitActivation + unit], nextState),
  );
}

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let slot = invocation.x;
  if (slot >= config.outputCount) {
    return;
  }
  let unit = heap[config.outputs + slot];
  writeF32(config.output, slot, readF32(config.activation, unit));
}
`;
