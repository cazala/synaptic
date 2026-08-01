export async function instantiate(module, imports = {}) {
  const adaptedImports = {
    env: Object.setPrototypeOf({
      abort(message, fileName, lineNumber, columnNumber) {
        // ~lib/builtins/abort(~lib/string/String | null?, ~lib/string/String | null?, u32?, u32?) => void
        message = __liftString(message >>> 0);
        fileName = __liftString(fileName >>> 0);
        lineNumber = lineNumber >>> 0;
        columnNumber = columnNumber >>> 0;
        (() => {
          // @external.js
          throw Error(`${message} in ${fileName}:${lineNumber}:${columnNumber}`);
        })();
      },
    }, Object.assign(Object.create(globalThis), imports.env || {})),
  };
  const { exports } = await WebAssembly.instantiate(module, adaptedImports);
  const memory = exports.memory || imports.env.memory;
  const adaptedExports = Object.setPrototypeOf({
    FLOAT32ARRAY_ID: {
      // packages/backend-wasm/assembly/runtime/FLOAT32ARRAY_ID: u32
      valueOf() { return this.value; },
      get value() {
        return exports.FLOAT32ARRAY_ID.value >>> 0;
      }
    },
    UINT32ARRAY_ID: {
      // packages/backend-wasm/assembly/runtime/UINT32ARRAY_ID: u32
      valueOf() { return this.value; },
      get value() {
        return exports.UINT32ARRAY_ID.value >>> 0;
      }
    },
    INT32ARRAY_ID: {
      // packages/backend-wasm/assembly/runtime/INT32ARRAY_ID: u32
      valueOf() { return this.value; },
      get value() {
        return exports.INT32ARRAY_ID.value >>> 0;
      }
    },
    initialize(nextUnitCount, nextConnectionCount, nextParameterCount, nextStageOffsets, nextStageUnits, nextUnitActivation, nextUnitConstant, nextConnectionFrom, nextConnectionTo, nextConnectionParameter, nextConnectionDelay, nextConnectionGater, nextGateDelay, nextIncomingOffsets, nextIncomingConnections, nextOutgoingOffsets, nextOutgoingConnections, nextGatedTargetOffsets, nextGatedTargets, nextExtendedTraceTarget, nextExtendedTraceOffsets, nextInputs, nextOutputs, nextInputSlot, nextOutputSlot, nextParameterTrainable, nextParameters) {
      // packages/backend-wasm/assembly/runtime/initialize(i32, i32, i32, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Float32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Int32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Int32Array, ~lib/typedarray/Int32Array, ~lib/typedarray/Uint32Array, ~lib/typedarray/Float32Array) => void
      nextStageOffsets = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextStageOffsets) || __notnull());
      nextStageUnits = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextStageUnits) || __notnull());
      nextUnitActivation = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextUnitActivation) || __notnull());
      nextUnitConstant = __retain(__lowerTypedArray(Float32Array, 4, 2, nextUnitConstant) || __notnull());
      nextConnectionFrom = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextConnectionFrom) || __notnull());
      nextConnectionTo = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextConnectionTo) || __notnull());
      nextConnectionParameter = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextConnectionParameter) || __notnull());
      nextConnectionDelay = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextConnectionDelay) || __notnull());
      nextConnectionGater = __retain(__lowerTypedArray(Int32Array, 6, 2, nextConnectionGater) || __notnull());
      nextGateDelay = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextGateDelay) || __notnull());
      nextIncomingOffsets = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextIncomingOffsets) || __notnull());
      nextIncomingConnections = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextIncomingConnections) || __notnull());
      nextOutgoingOffsets = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextOutgoingOffsets) || __notnull());
      nextOutgoingConnections = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextOutgoingConnections) || __notnull());
      nextGatedTargetOffsets = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextGatedTargetOffsets) || __notnull());
      nextGatedTargets = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextGatedTargets) || __notnull());
      nextExtendedTraceTarget = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextExtendedTraceTarget) || __notnull());
      nextExtendedTraceOffsets = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextExtendedTraceOffsets) || __notnull());
      nextInputs = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextInputs) || __notnull());
      nextOutputs = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextOutputs) || __notnull());
      nextInputSlot = __retain(__lowerTypedArray(Int32Array, 6, 2, nextInputSlot) || __notnull());
      nextOutputSlot = __retain(__lowerTypedArray(Int32Array, 6, 2, nextOutputSlot) || __notnull());
      nextParameterTrainable = __retain(__lowerTypedArray(Uint32Array, 5, 2, nextParameterTrainable) || __notnull());
      nextParameters = __lowerTypedArray(Float32Array, 4, 2, nextParameters) || __notnull();
      try {
        exports.initialize(nextUnitCount, nextConnectionCount, nextParameterCount, nextStageOffsets, nextStageUnits, nextUnitActivation, nextUnitConstant, nextConnectionFrom, nextConnectionTo, nextConnectionParameter, nextConnectionDelay, nextConnectionGater, nextGateDelay, nextIncomingOffsets, nextIncomingConnections, nextOutgoingOffsets, nextOutgoingConnections, nextGatedTargetOffsets, nextGatedTargets, nextExtendedTraceTarget, nextExtendedTraceOffsets, nextInputs, nextOutputs, nextInputSlot, nextOutputSlot, nextParameterTrainable, nextParameters);
      } finally {
        __release(nextStageOffsets);
        __release(nextStageUnits);
        __release(nextUnitActivation);
        __release(nextUnitConstant);
        __release(nextConnectionFrom);
        __release(nextConnectionTo);
        __release(nextConnectionParameter);
        __release(nextConnectionDelay);
        __release(nextConnectionGater);
        __release(nextGateDelay);
        __release(nextIncomingOffsets);
        __release(nextIncomingConnections);
        __release(nextOutgoingOffsets);
        __release(nextOutgoingConnections);
        __release(nextGatedTargetOffsets);
        __release(nextGatedTargets);
        __release(nextExtendedTraceTarget);
        __release(nextExtendedTraceOffsets);
        __release(nextInputs);
        __release(nextOutputs);
        __release(nextInputSlot);
        __release(nextOutputSlot);
        __release(nextParameterTrainable);
      }
    },
    forward(input) {
      // packages/backend-wasm/assembly/runtime/forward(~lib/typedarray/Float32Array) => ~lib/typedarray/Float32Array
      input = __lowerTypedArray(Float32Array, 4, 2, input) || __notnull();
      return __liftTypedArray(Float32Array, exports.forward(input) >>> 0);
    },
    trainStep(input, target, learningRate) {
      // packages/backend-wasm/assembly/runtime/trainStep(~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, f32) => f32
      input = __retain(__lowerTypedArray(Float32Array, 4, 2, input) || __notnull());
      target = __lowerTypedArray(Float32Array, 4, 2, target) || __notnull();
      try {
        return exports.trainStep(input, target, learningRate);
      } finally {
        __release(input);
      }
    },
    restoreState(nextParameters, nextState, nextActivation, nextPreviousActivation, nextDerivative, nextEligibilityTrace, nextExtendedEligibilityTrace, nextProjectedError, nextGatedError, nextError, nextStep) {
      // packages/backend-wasm/assembly/runtime/restoreState(~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, i32) => void
      nextParameters = __retain(__lowerTypedArray(Float32Array, 4, 2, nextParameters) || __notnull());
      nextState = __retain(__lowerTypedArray(Float32Array, 4, 2, nextState) || __notnull());
      nextActivation = __retain(__lowerTypedArray(Float32Array, 4, 2, nextActivation) || __notnull());
      nextPreviousActivation = __retain(__lowerTypedArray(Float32Array, 4, 2, nextPreviousActivation) || __notnull());
      nextDerivative = __retain(__lowerTypedArray(Float32Array, 4, 2, nextDerivative) || __notnull());
      nextEligibilityTrace = __retain(__lowerTypedArray(Float32Array, 4, 2, nextEligibilityTrace) || __notnull());
      nextExtendedEligibilityTrace = __retain(__lowerTypedArray(Float32Array, 4, 2, nextExtendedEligibilityTrace) || __notnull());
      nextProjectedError = __retain(__lowerTypedArray(Float32Array, 4, 2, nextProjectedError) || __notnull());
      nextGatedError = __retain(__lowerTypedArray(Float32Array, 4, 2, nextGatedError) || __notnull());
      nextError = __lowerTypedArray(Float32Array, 4, 2, nextError) || __notnull();
      try {
        exports.restoreState(nextParameters, nextState, nextActivation, nextPreviousActivation, nextDerivative, nextEligibilityTrace, nextExtendedEligibilityTrace, nextProjectedError, nextGatedError, nextError, nextStep);
      } finally {
        __release(nextParameters);
        __release(nextState);
        __release(nextActivation);
        __release(nextPreviousActivation);
        __release(nextDerivative);
        __release(nextEligibilityTrace);
        __release(nextExtendedEligibilityTrace);
        __release(nextProjectedError);
        __release(nextGatedError);
      }
    },
    getParameters() {
      // packages/backend-wasm/assembly/runtime/getParameters() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getParameters() >>> 0);
    },
    getState() {
      // packages/backend-wasm/assembly/runtime/getState() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getState() >>> 0);
    },
    getActivation() {
      // packages/backend-wasm/assembly/runtime/getActivation() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getActivation() >>> 0);
    },
    getPreviousActivation() {
      // packages/backend-wasm/assembly/runtime/getPreviousActivation() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getPreviousActivation() >>> 0);
    },
    getDerivative() {
      // packages/backend-wasm/assembly/runtime/getDerivative() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getDerivative() >>> 0);
    },
    getEligibilityTrace() {
      // packages/backend-wasm/assembly/runtime/getEligibilityTrace() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getEligibilityTrace() >>> 0);
    },
    getExtendedEligibilityTrace() {
      // packages/backend-wasm/assembly/runtime/getExtendedEligibilityTrace() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getExtendedEligibilityTrace() >>> 0);
    },
    getProjectedError() {
      // packages/backend-wasm/assembly/runtime/getProjectedError() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getProjectedError() >>> 0);
    },
    getGatedError() {
      // packages/backend-wasm/assembly/runtime/getGatedError() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getGatedError() >>> 0);
    },
    getError() {
      // packages/backend-wasm/assembly/runtime/getError() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getError() >>> 0);
    },
    getInputDataStart() {
      // packages/backend-wasm/assembly/runtime/getInputDataStart() => usize
      return exports.getInputDataStart() >>> 0;
    },
    getTargetDataStart() {
      // packages/backend-wasm/assembly/runtime/getTargetDataStart() => usize
      return exports.getTargetDataStart() >>> 0;
    },
    getOutputDataStart() {
      // packages/backend-wasm/assembly/runtime/getOutputDataStart() => usize
      return exports.getOutputDataStart() >>> 0;
    },
  }, exports);
  function __liftString(pointer) {
    if (!pointer) return null;
    const
      end = pointer + new Uint32Array(memory.buffer)[pointer - 4 >>> 2] >>> 1,
      memoryU16 = new Uint16Array(memory.buffer);
    let
      start = pointer >>> 1,
      string = "";
    while (end - start > 1024) string += String.fromCharCode(...memoryU16.subarray(start, start += 1024));
    return string + String.fromCharCode(...memoryU16.subarray(start, end));
  }
  function __liftTypedArray(constructor, pointer) {
    if (!pointer) return null;
    return new constructor(
      memory.buffer,
      __getU32(pointer + 4),
      __dataview.getUint32(pointer + 8, true) / constructor.BYTES_PER_ELEMENT
    ).slice();
  }
  function __lowerTypedArray(constructor, id, align, values) {
    if (values == null) return 0;
    const
      length = values.length,
      buffer = exports.__pin(exports.__new(length << align, 1)) >>> 0,
      header = exports.__new(12, id) >>> 0;
    __setU32(header + 0, buffer);
    __dataview.setUint32(header + 4, buffer, true);
    __dataview.setUint32(header + 8, length << align, true);
    new constructor(memory.buffer, buffer, length).set(values);
    exports.__unpin(buffer);
    return header;
  }
  const refcounts = new Map();
  function __retain(pointer) {
    if (pointer) {
      const refcount = refcounts.get(pointer);
      if (refcount) refcounts.set(pointer, refcount + 1);
      else refcounts.set(exports.__pin(pointer), 1);
    }
    return pointer;
  }
  function __release(pointer) {
    if (pointer) {
      const refcount = refcounts.get(pointer);
      if (refcount === 1) exports.__unpin(pointer), refcounts.delete(pointer);
      else if (refcount) refcounts.set(pointer, refcount - 1);
      else throw Error(`invalid refcount '${refcount}' for reference '${pointer}'`);
    }
  }
  function __notnull() {
    throw TypeError("value must not be null");
  }
  let __dataview = new DataView(memory.buffer);
  function __setU32(pointer, value) {
    try {
      __dataview.setUint32(pointer, value, true);
    } catch {
      __dataview = new DataView(memory.buffer);
      __dataview.setUint32(pointer, value, true);
    }
  }
  function __getU32(pointer) {
    try {
      return __dataview.getUint32(pointer, true);
    } catch {
      __dataview = new DataView(memory.buffer);
      return __dataview.getUint32(pointer, true);
    }
  }
  return adaptedExports;
}
