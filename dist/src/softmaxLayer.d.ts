import Synaptic = require('./synaptic');
import Layer = require('./layer');
export declare class SoftMaxLayer extends Layer.Layer {
    constructor(size: number, label?: string);
    activate(input?: Synaptic.INumericArray): Synaptic.INumericArray;
    propagate(rate: number, target?: Synaptic.INumericArray): void;
    static NormalizeConnectionWeights(layerConnection: Layer.Layer.LayerConnection): void;
}
