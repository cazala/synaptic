import network = require('../network');
import trainer = require('../trainer');
import Layer = require('../layer');
import Synaptic = require('../synaptic');
export declare class MemoryTape {
    data: Float64Array[];
    blockWidth: number;
    blocks: number;
    layer: Layer.Layer;
    prevLayerActivate: any;
    memoryAttentionLocation: number;
    memoryAttentionWeight: number;
    constructor(memoryBlocks: number, layer: Layer.Layer, inputGate: Layer.Layer, forgetGate: Layer.Layer);
    static getSimilarity(arrayA: Synaptic.INumericArray, arrayB: Synaptic.INumericArray): number;
    static softMaxArray<T extends Synaptic.INumericArray>(array: T, sharpen?: number): T;
    getSimilarAdresses(weights: Synaptic.INumericArray): Float64Array;
}
export declare class MemoryBlock extends network.Network {
    trainer: trainer.Trainer;
    memoryTape: MemoryTape;
    constructor(inputSize: number, memoryBlocks: number, memoryWidth: number, outputSize: number);
}
