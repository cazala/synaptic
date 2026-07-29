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
  connectionTo: u32,
  outgoingOffsets: u32,
  outgoingConnections: u32,
  gatedTargetOffsets: u32,
  gatedTargets: u32,
  extendedTraceTarget: u32,
  extendedTraceOffsets: u32,
  parameterTrainable: u32,
  outputSlot: u32,
  derivative: u32,
  eligibilityTrace: u32,
  extendedEligibilityTrace: u32,
  projectedError: u32,
  gatedError: u32,
  error: u32,
  connectionGradient: u32,
  targets: u32,
  learningRate: u32,
  connectionCount: u32,
  parameterCount: u32,
  parameterOffsets: u32,
  parameterConnections: u32,
  padding0: u32,
  padding1: u32,
  padding2: u32,
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

fn selfFactor(unit: u32) -> f32 {
  var result = 0.0;
  let start = heap[config.incomingOffsets + unit];
  let end = heap[config.incomingOffsets + unit + 1u];
  for (var cursor = start; cursor < end; cursor += 1u) {
    let connection = heap[config.incomingConnections + cursor];
    if (heap[config.connectionFrom + connection] == unit) {
      let parameter = heap[config.connectionParameter + connection];
      result += connectionGain(connection) * readF32(config.parameters, parameter);
    }
  }
  return result;
}

fn bigParenthesis(targetUnit: u32, gater: u32) -> f32 {
  var result = 0.0;
  let start = heap[config.incomingOffsets + targetUnit];
  let end = heap[config.incomingOffsets + targetUnit + 1u];
  for (var cursor = start; cursor < end; cursor += 1u) {
    let connection = heap[config.incomingConnections + cursor];
    if (
      heap[config.connectionGater + connection] != gater
      || heap[config.gateDelay + connection] != 0u
    ) {
      continue;
    }
    let parameter = heap[config.connectionParameter + connection];
    let source = heap[config.connectionFrom + connection];
    let value = select(
      sourceValue(connection),
      readF32(config.state, targetUnit),
      source == targetUnit,
    );
    result += readF32(config.parameters, parameter) * value;
  }
  return result;
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

fn squashDerivative(code: u32, stateValue: f32, activationValue: f32) -> f32 {
  switch code {
    case 0u: {
      return 1.0;
    }
    case 1u: {
      return activationValue * (1.0 - activationValue);
    }
    case 2u: {
      return 1.0 - activationValue * activationValue;
    }
    case 3u: {
      return select(0.0, 1.0, stateValue > 0.0);
    }
    default: {
      return 0.0;
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
    writeF32(config.derivative, unit, 0.0);
    return;
  }
  if (heap[config.unitConstantMask + unit] != 0u) {
    writeF32(config.state, unit, constant);
    writeF32(config.activation, unit, constant);
    writeF32(config.derivative, unit, 0.0);
    return;
  }

  let previousState = readF32(config.state, unit);
  var nextState = 0.0;
  var recurrence = 0.0;
  let incomingStart = heap[config.incomingOffsets + unit];
  let incomingEnd = heap[config.incomingOffsets + unit + 1u];
  for (var cursor = incomingStart; cursor < incomingEnd; cursor += 1u) {
    let connection = heap[config.incomingConnections + cursor];
    let source = heap[config.connectionFrom + connection];
    let parameter = heap[config.connectionParameter + connection];
    let weight = readF32(config.parameters, parameter);
    let gain = connectionGain(connection);
    if (source == unit) {
      recurrence += gain * weight;
      nextState += gain * weight * previousState;
    } else {
      nextState += gain * weight * sourceValue(connection);
    }
  }
  writeF32(config.state, unit, nextState);
  let activationCode = heap[config.unitActivation + unit];
  let activationValue = squash(activationCode, nextState);
  let derivativeValue = squashDerivative(
    activationCode,
    nextState,
    activationValue,
  );
  writeF32(config.activation, unit, activationValue);
  writeF32(config.derivative, unit, derivativeValue);

  for (var cursor = incomingStart; cursor < incomingEnd; cursor += 1u) {
    let connection = heap[config.incomingConnections + cursor];
    if (heap[config.connectionFrom + connection] == unit) {
      continue;
    }
    let eligibility =
      recurrence * readF32(config.eligibilityTrace, connection)
      + connectionGain(connection) * sourceValue(connection);
    writeF32(config.eligibilityTrace, connection, eligibility);
    let traceStart = heap[config.extendedTraceOffsets + connection];
    let traceEnd = heap[config.extendedTraceOffsets + connection + 1u];
    for (var trace = traceStart; trace < traceEnd; trace += 1u) {
      let targetUnit = heap[config.extendedTraceTarget + trace];
      let extended =
        selfFactor(targetUnit) * readF32(config.extendedEligibilityTrace, trace)
        + derivativeValue * eligibility * bigParenthesis(targetUnit, unit);
      writeF32(config.extendedEligibilityTrace, trace, extended);
    }
  }
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

@compute @workgroup_size(64)
fn clearTraining(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index < config.unitCount) {
    writeF32(config.projectedError, index, 0.0);
    writeF32(config.gatedError, index, 0.0);
    writeF32(config.error, index, 0.0);
  }
  if (index < config.connectionCount) {
    writeF32(config.connectionGradient, index, 0.0);
  }
}

@compute @workgroup_size(64)
fn backwardStage(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let localIndex = invocation.x;
  let stageLength = config.stageEnd - config.stageStart;
  if (localIndex >= stageLength) {
    return;
  }
  let unit = heap[config.stageUnits + config.stageStart + localIndex];
  let outputSlot = bitcast<i32>(heap[config.outputSlot + unit]);
  var projected = 0.0;
  var gated = 0.0;

  if (outputSlot >= 0) {
    projected =
      readF32(config.targets, u32(outputSlot))
      - readF32(config.activation, unit);
  } else {
    let outgoingStart = heap[config.outgoingOffsets + unit];
    let outgoingEnd = heap[config.outgoingOffsets + unit + 1u];
    for (var cursor = outgoingStart; cursor < outgoingEnd; cursor += 1u) {
      let connection = heap[config.outgoingConnections + cursor];
      if (heap[config.connectionDelay + connection] == 0u) {
        let targetUnit = heap[config.connectionTo + connection];
        let parameter = heap[config.connectionParameter + connection];
        projected +=
          readF32(config.error, targetUnit)
          * connectionGain(connection)
          * readF32(config.parameters, parameter);
      }
    }
    projected *= readF32(config.derivative, unit);

    let gatedStart = heap[config.gatedTargetOffsets + unit];
    let gatedEnd = heap[config.gatedTargetOffsets + unit + 1u];
    for (var cursor = gatedStart; cursor < gatedEnd; cursor += 1u) {
      let targetUnit = heap[config.gatedTargets + cursor];
      gated +=
        readF32(config.error, targetUnit)
        * bigParenthesis(targetUnit, unit);
    }
    gated *= readF32(config.derivative, unit);
  }
  writeF32(config.projectedError, unit, projected);
  writeF32(config.gatedError, unit, gated);
  writeF32(config.error, unit, projected + gated);

  let incomingStart = heap[config.incomingOffsets + unit];
  let incomingEnd = heap[config.incomingOffsets + unit + 1u];
  for (var cursor = incomingStart; cursor < incomingEnd; cursor += 1u) {
    let connection = heap[config.incomingConnections + cursor];
    if (heap[config.connectionFrom + connection] == unit) {
      continue;
    }
    var gradient =
      projected * readF32(config.eligibilityTrace, connection);
    let traceStart = heap[config.extendedTraceOffsets + connection];
    let traceEnd = heap[config.extendedTraceOffsets + connection + 1u];
    for (var trace = traceStart; trace < traceEnd; trace += 1u) {
      let targetUnit = heap[config.extendedTraceTarget + trace];
      gradient +=
        readF32(config.error, targetUnit)
        * readF32(config.extendedEligibilityTrace, trace);
    }
    writeF32(config.connectionGradient, connection, gradient);
  }
}

@compute @workgroup_size(64)
fn updateParameters(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let parameter = invocation.x;
  if (parameter >= config.parameterCount) {
    return;
  }
  if (heap[config.parameterTrainable + parameter] == 0u) {
    return;
  }
  var gradient = 0.0;
  let start = heap[config.parameterOffsets + parameter];
  let end = heap[config.parameterOffsets + parameter + 1u];
  for (var cursor = start; cursor < end; cursor += 1u) {
    let connection = heap[config.parameterConnections + cursor];
    gradient += readF32(config.connectionGradient, connection);
  }
  let learningRate = readF32(config.learningRate, 0u);
  writeF32(
    config.parameters,
    parameter,
    readF32(config.parameters, parameter) + learningRate * gradient,
  );
}
`;
