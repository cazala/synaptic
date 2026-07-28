export const FLOAT32ARRAY_ID: u32 = idof<Float32Array>();
export const UINT32ARRAY_ID: u32 = idof<Uint32Array>();
export const INT32ARRAY_ID: u32 = idof<Int32Array>();

let unitCount: i32;
let connectionCount: i32;
let parameterCount: i32;

let stageOffsets = new Uint32Array(0);
let stageUnits = new Uint32Array(0);
let unitActivation = new Uint32Array(0);
let unitConstant = new Float32Array(0);
let connectionFrom = new Uint32Array(0);
let connectionTo = new Uint32Array(0);
let connectionParameter = new Uint32Array(0);
let connectionDelay = new Uint32Array(0);
let connectionGater = new Int32Array(0);
let gateDelay = new Uint32Array(0);
let incomingOffsets = new Uint32Array(0);
let incomingConnections = new Uint32Array(0);
let outgoingOffsets = new Uint32Array(0);
let outgoingConnections = new Uint32Array(0);
let gatedTargetOffsets = new Uint32Array(0);
let gatedTargets = new Uint32Array(0);
let extendedTraceTarget = new Uint32Array(0);
let extendedTraceOffsets = new Uint32Array(0);
let inputs = new Uint32Array(0);
let outputs = new Uint32Array(0);
let inputSlot = new Int32Array(0);
let outputSlot = new Int32Array(0);
let parameterTrainable = new Uint32Array(0);

let parameters = new Float32Array(0);
let state = new Float32Array(0);
let activation = new Float32Array(0);
let previousActivation = new Float32Array(0);
let derivative = new Float32Array(0);
let eligibilityTrace = new Float32Array(0);
let extendedEligibilityTrace = new Float32Array(0);
let projectedError = new Float32Array(0);
let gatedError = new Float32Array(0);
let error = new Float32Array(0);
let output = new Float32Array(0);
let gradients = new Float32Array(0);
let inputBuffer = new Float32Array(0);
let targetBuffer = new Float32Array(0);
let step: i32 = 0;

export function initialize(
  nextUnitCount: i32,
  nextConnectionCount: i32,
  nextParameterCount: i32,
  nextStageOffsets: Uint32Array,
  nextStageUnits: Uint32Array,
  nextUnitActivation: Uint32Array,
  nextUnitConstant: Float32Array,
  nextConnectionFrom: Uint32Array,
  nextConnectionTo: Uint32Array,
  nextConnectionParameter: Uint32Array,
  nextConnectionDelay: Uint32Array,
  nextConnectionGater: Int32Array,
  nextGateDelay: Uint32Array,
  nextIncomingOffsets: Uint32Array,
  nextIncomingConnections: Uint32Array,
  nextOutgoingOffsets: Uint32Array,
  nextOutgoingConnections: Uint32Array,
  nextGatedTargetOffsets: Uint32Array,
  nextGatedTargets: Uint32Array,
  nextExtendedTraceTarget: Uint32Array,
  nextExtendedTraceOffsets: Uint32Array,
  nextInputs: Uint32Array,
  nextOutputs: Uint32Array,
  nextInputSlot: Int32Array,
  nextOutputSlot: Int32Array,
  nextParameterTrainable: Uint32Array,
  nextParameters: Float32Array,
): void {
  unitCount = nextUnitCount;
  connectionCount = nextConnectionCount;
  parameterCount = nextParameterCount;
  stageOffsets = nextStageOffsets;
  stageUnits = nextStageUnits;
  unitActivation = nextUnitActivation;
  unitConstant = nextUnitConstant;
  connectionFrom = nextConnectionFrom;
  connectionTo = nextConnectionTo;
  connectionParameter = nextConnectionParameter;
  connectionDelay = nextConnectionDelay;
  connectionGater = nextConnectionGater;
  gateDelay = nextGateDelay;
  incomingOffsets = nextIncomingOffsets;
  incomingConnections = nextIncomingConnections;
  outgoingOffsets = nextOutgoingOffsets;
  outgoingConnections = nextOutgoingConnections;
  gatedTargetOffsets = nextGatedTargetOffsets;
  gatedTargets = nextGatedTargets;
  extendedTraceTarget = nextExtendedTraceTarget;
  extendedTraceOffsets = nextExtendedTraceOffsets;
  inputs = nextInputs;
  outputs = nextOutputs;
  inputSlot = nextInputSlot;
  outputSlot = nextOutputSlot;
  parameterTrainable = nextParameterTrainable;
  parameters = nextParameters;

  state = new Float32Array(unitCount);
  activation = new Float32Array(unitCount);
  previousActivation = new Float32Array(unitCount);
  derivative = new Float32Array(unitCount);
  eligibilityTrace = new Float32Array(connectionCount);
  extendedEligibilityTrace = new Float32Array(extendedTraceTarget.length);
  projectedError = new Float32Array(unitCount);
  gatedError = new Float32Array(unitCount);
  error = new Float32Array(unitCount);
  output = new Float32Array(outputs.length);
  gradients = new Float32Array(parameterCount);
  inputBuffer = new Float32Array(inputs.length);
  targetBuffer = new Float32Array(outputs.length);
}

@inline
function gain(connection: i32): f32 {
  const gater = connectionGater[connection];
  if (gater < 0) {
    return 1.0;
  }
  return gateDelay[connection] == 0
    ? activation[gater]
    : previousActivation[gater];
}

@inline
function sourceValue(connection: i32): f32 {
  const source = connectionFrom[connection];
  return connectionDelay[connection] == 0
    ? activation[source]
    : previousActivation[source];
}

function selfFactor(unit: i32): f32 {
  let result: f32 = 0;
  const start = incomingOffsets[unit];
  const end = incomingOffsets[unit + 1];
  for (let cursor = start; cursor < end; cursor += 1) {
    const connection = incomingConnections[cursor];
    if (connectionFrom[connection] == <u32>unit) {
      result += gain(connection) * parameters[connectionParameter[connection]];
    }
  }
  return result;
}

function bigParenthesis(target: i32, gater: i32): f32 {
  let result: f32 = 0;
  const start = incomingOffsets[target];
  const end = incomingOffsets[target + 1];
  for (let cursor = start; cursor < end; cursor += 1) {
    const connection = incomingConnections[cursor];
    if (connectionGater[connection] != gater || gateDelay[connection] != 0) {
      continue;
    }
    const value = connectionFrom[connection] == <u32>target
      ? state[target]
      : sourceValue(connection);
    result += parameters[connectionParameter[connection]] * value;
  }
  return result;
}

function activateUnit(unit: i32): void {
  const incomingStart = incomingOffsets[unit];
  const incomingEnd = incomingOffsets[unit + 1];
  const previousState = state[unit];
  let recurrence: f32 = 0;
  let nextState: f32 = 0;

  for (let cursor = incomingStart; cursor < incomingEnd; cursor += 1) {
    const connection = incomingConnections[cursor];
    const localGain = gain(connection);
    const weight = parameters[connectionParameter[connection]];
    if (connectionFrom[connection] == <u32>unit) {
      recurrence += localGain * weight;
      nextState += localGain * weight * previousState;
    } else {
      nextState += localGain * weight * sourceValue(connection);
    }
  }

  state[unit] = nextState;
  const code = unitActivation[unit];
  let nextActivation: f32;
  let nextDerivative: f32;
  if (code == 1) {
    nextActivation = 1.0 / (1.0 + <f32>Math.exp(<f64>-nextState));
    nextDerivative = nextActivation * (1.0 - nextActivation);
  } else if (code == 2) {
    nextActivation = <f32>Math.tanh(<f64>nextState);
    nextDerivative = 1.0 - nextActivation * nextActivation;
  } else if (code == 3) {
    nextActivation = max<f32>(0, nextState);
    nextDerivative = nextState > 0 ? 1.0 : 0.0;
  } else if (code == 4) {
    nextActivation = nextState > 0 ? 1.0 : 0.0;
    nextDerivative = 0;
  } else {
    nextActivation = nextState;
    nextDerivative = 1;
  }
  activation[unit] = nextActivation;
  derivative[unit] = nextDerivative;

  for (let cursor = incomingStart; cursor < incomingEnd; cursor += 1) {
    const connection = incomingConnections[cursor];
    if (connectionFrom[connection] == <u32>unit) {
      continue;
    }
    const eligibility = recurrence * eligibilityTrace[connection]
      + gain(connection) * sourceValue(connection);
    eligibilityTrace[connection] = eligibility;
    const traceStart = extendedTraceOffsets[connection];
    const traceEnd = extendedTraceOffsets[connection + 1];
    for (let trace = traceStart; trace < traceEnd; trace += 1) {
      const target = extendedTraceTarget[trace];
      extendedEligibilityTrace[trace] =
        selfFactor(target) * extendedEligibilityTrace[trace]
        + nextDerivative * eligibility * bigParenthesis(target, unit);
    }
  }
}

function forwardInternal(input: Float32Array): void {
  for (let unit = 0; unit < unitCount; unit += 1) {
    previousActivation[unit] = activation[unit];
  }
  const stageCount = stageOffsets.length - 1;
  for (let stage = 0; stage < stageCount; stage += 1) {
    const start = stageOffsets[stage];
    const end = stageOffsets[stage + 1];
    for (let cursor = start; cursor < end; cursor += 1) {
      const unit = stageUnits[cursor];
      const slot = inputSlot[unit];
      const constant = unitConstant[unit];
      if (slot >= 0) {
        const value = input[slot];
        state[unit] = value;
        activation[unit] = value;
        derivative[unit] = 0;
      } else if (!isNaN(constant)) {
        state[unit] = constant;
        activation[unit] = constant;
        derivative[unit] = 0;
      } else {
        activateUnit(unit);
      }
    }
  }
  for (let slot = 0; slot < outputs.length; slot += 1) {
    output[slot] = activation[outputs[slot]];
  }
  step += 1;
}

export function forward(input: Float32Array): Float32Array {
  forwardInternal(input);
  return output;
}

export function forwardBuffered(): void {
  forwardInternal(inputBuffer);
}

function propagate(target: Float32Array, learningRate: f32): void {
  projectedError.fill(0);
  gatedError.fill(0);
  error.fill(0);
  gradients.fill(0);
  const stageCount = stageOffsets.length - 1;

  for (let stage = stageCount - 1; stage > 0; stage -= 1) {
    const start = stageOffsets[stage];
    const end = stageOffsets[stage + 1];
    for (let cursor = <i32>end - 1; cursor >= <i32>start; cursor -= 1) {
      const unit = stageUnits[cursor];
      const slot = outputSlot[unit];
      if (slot >= 0) {
        projectedError[unit] = target[slot] - activation[unit];
        gatedError[unit] = 0;
      } else {
        let projected: f32 = 0;
        const outgoingStart = outgoingOffsets[unit];
        const outgoingEnd = outgoingOffsets[unit + 1];
        for (let outgoing = outgoingStart; outgoing < outgoingEnd; outgoing += 1) {
          const connection = outgoingConnections[outgoing];
          if (connectionDelay[connection] == 0) {
            projected += error[connectionTo[connection]]
              * gain(connection)
              * parameters[connectionParameter[connection]];
          }
        }
        projectedError[unit] = derivative[unit] * projected;

        let gated: f32 = 0;
        const gatedStart = gatedTargetOffsets[unit];
        const gatedEnd = gatedTargetOffsets[unit + 1];
        for (let gatedCursor = gatedStart; gatedCursor < gatedEnd; gatedCursor += 1) {
          const targetUnit = gatedTargets[gatedCursor];
          gated += error[targetUnit] * bigParenthesis(targetUnit, unit);
        }
        gatedError[unit] = derivative[unit] * gated;
      }
      error[unit] = projectedError[unit] + gatedError[unit];

      const incomingStart = incomingOffsets[unit];
      const incomingEnd = incomingOffsets[unit + 1];
      for (let incoming = incomingStart; incoming < incomingEnd; incoming += 1) {
        const connection = incomingConnections[incoming];
        if (connectionFrom[connection] == unit) {
          continue;
        }
        let gradient = projectedError[unit] * eligibilityTrace[connection];
        const traceStart = extendedTraceOffsets[connection];
        const traceEnd = extendedTraceOffsets[connection + 1];
        for (let trace = traceStart; trace < traceEnd; trace += 1) {
          const targetUnit = extendedTraceTarget[trace];
          gradient += error[targetUnit] * extendedEligibilityTrace[trace];
        }
        gradients[connectionParameter[connection]] += gradient;
      }
    }
  }

  for (let parameter = 0; parameter < parameterCount; parameter += 1) {
    if (parameterTrainable[parameter] != 0) {
      parameters[parameter] += learningRate * gradients[parameter];
    }
  }
}

export function trainStep(input: Float32Array, target: Float32Array, learningRate: f32): f32 {
  forwardInternal(input);
  let loss: f32 = 0;
  for (let index = 0; index < target.length; index += 1) {
    const difference = target[index] - output[index];
    loss += difference * difference;
  }
  propagate(target, learningRate);
  return loss / <f32>target.length;
}

export function trainStepBuffered(learningRate: f32): f32 {
  forwardInternal(inputBuffer);
  let loss: f32 = 0;
  for (let index = 0; index < targetBuffer.length; index += 1) {
    const difference = targetBuffer[index] - output[index];
    loss += difference * difference;
  }
  propagate(targetBuffer, learningRate);
  return loss / <f32>targetBuffer.length;
}

export function resetState(): void {
  state.fill(0);
  activation.fill(0);
  previousActivation.fill(0);
  derivative.fill(0);
  eligibilityTrace.fill(0);
  extendedEligibilityTrace.fill(0);
  projectedError.fill(0);
  gatedError.fill(0);
  error.fill(0);
  output.fill(0);
  gradients.fill(0);
  step = 0;
}

export function getParameters(): Float32Array { return parameters; }
export function getState(): Float32Array { return state; }
export function getActivation(): Float32Array { return activation; }
export function getPreviousActivation(): Float32Array { return previousActivation; }
export function getDerivative(): Float32Array { return derivative; }
export function getEligibilityTrace(): Float32Array { return eligibilityTrace; }
export function getExtendedEligibilityTrace(): Float32Array { return extendedEligibilityTrace; }
export function getProjectedError(): Float32Array { return projectedError; }
export function getGatedError(): Float32Array { return gatedError; }
export function getError(): Float32Array { return error; }
export function getStep(): i32 { return step; }
export function getInputDataStart(): usize { return inputBuffer.dataStart; }
export function getTargetDataStart(): usize { return targetBuffer.dataStart; }
export function getOutputDataStart(): usize { return output.dataStart; }
