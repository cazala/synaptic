import network = require('../network');
import trainer = require('../trainer');
import Layer = require('../layer');
import Synaptic = require('../synaptic');
export declare class Utils {
    static softMax<T extends Synaptic.INumericArray>(array: T, sharpen?: number): T;
    static getCosineSimilarity(arrayA: Synaptic.INumericArray, arrayB: Synaptic.INumericArray): number;
    static interpolateArray(output_inputA: Synaptic.INumericArray, inputB: Synaptic.INumericArray, g: any): Synaptic.INumericArray;
    static sharpArray(output: Synaptic.INumericArray, wn: Synaptic.INumericArray, Y: number): void;
    static scalarShifting(wg: Synaptic.INumericArray, shiftScalar: number): Float64Array;
    static vectorShifting(wg: Float64Array, shiftings: Synaptic.INumericArray): void;
}
export declare class NTM extends network.Network {
    trainer: trainer.Trainer;
    data: Float64Array[];
    blockWidth: number;
    blocks: number;
    heads: Head[];
    inputValues: Float64Array;
    inputLayer: Layer.Layer;
    hiddenLayer: Layer.Layer;
    outputLayer: Layer.Layer;
    dirty: boolean;
    constructor(inputs: number, outputs: number, memBlocks: number, blockWidth: number, heads: number, hiddenSize: number);
    clean(): void;
    activate(input: Synaptic.INumericArray): Float64Array;
    propagate(rate: number, target: Synaptic.INumericArray): void;
    addHead(subArray: Float64Array): Head;
    doTimeStep(): void;
    doAdd(w: Synaptic.INumericArray, addGate: Synaptic.INumericArray): void;
    doErase(w: Synaptic.INumericArray, eraseGate: Synaptic.INumericArray): void;
}
export declare class Head {
    static ADDITIONAL_INPUT_VALUES: number;
    memory: NTM;
    w_weightings: Float64Array;
    eraseGate: Float64Array;
    addGate: Float64Array;
    k_keys: Float64Array;
    g_interpolation: number;
    Y_focus: number;
    s_shiftingValue: number;
    s_shiftingVector: Float64Array;
    wc_focusedWeights: Float64Array;
    readVector: Float64Array;
    ß_keyStrength: number;
    prevFocus: number;
    layer: Layer.Layer;
    constructor(memory: NTM, destinationArray?: Float64Array);
    doTimeStep(): void;
}
