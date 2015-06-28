import Synaptic = require('./synaptic');
export declare class Utils {
    static transformationMatrixCache: {
        [index: number]: Float64Array[];
    };
    static softMax<T extends Synaptic.INumericArray>(outputArray: T): T;
    static softMaxDerivative<T extends Synaptic.INumericArray>(outputArray: T): T;
    static softMaxReinforcement<T extends Synaptic.INumericArray>(array: T, temperature?: number): T;
    static getCosineSimilarity(arrayA: Synaptic.INumericArray, arrayB: Synaptic.INumericArray): number;
    static interpolateArray(output_inputA: Synaptic.INumericArray, inputB: Synaptic.INumericArray, g: any): Synaptic.INumericArray;
    static sharpArray(output: Synaptic.INumericArray, wn: Synaptic.INumericArray, Y: number): void;
    static scalarShifting(wg: Synaptic.INumericArray, shiftScalar: number): Float64Array;
    static normalizeShift(shift: Float64Array): void;
    static vectorInvertedShifting(wg: Float64Array, shiftings: Synaptic.INumericArray): void;
    static initRandomSoftmaxArray(array: Float64Array): void;
    static buildCirculantMatrix(length: number, offset?: number): Float64Array[];
}
