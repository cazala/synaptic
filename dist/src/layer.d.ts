import neuron = require('./neuron');
import network = require('./network');
import Synaptic = require('./synaptic');
/*******************************************************************************************
                                            LAYER
*******************************************************************************************/
export declare class Layer {
    optimizable: boolean;
    list: neuron.Neuron[];
    label: string;
    connectedto: any[];
    size: number;
    currentActivation: Float64Array;
    constructor(size: number, label?: string);
    activate(input?: Synaptic.INumericArray): Synaptic.INumericArray;
    propagate(rate: number, target?: Synaptic.INumericArray): void;
    project(layer: network.Network | Layer, type?: string, weights?: Synaptic.INumericArray): Layer.LayerConnection;
    gate(connection: any, type: any): void;
    selfconnected(): boolean;
    connected(layer: any): string;
    clear(): void;
    reset(): void;
    neurons(): neuron.Neuron[];
    add(neuron: any): void;
    set(options: any): Layer;
}
export declare module Layer {
    var layerQty: number;
    function uid(): number;
    var connectionType: {
        ALL_TO_ALL: string;
        ONE_TO_ONE: string;
        ALL_TO_ELSE: string;
    };
    var gateType: {
        INPUT: string;
        OUTPUT: string;
        ONE_TO_ONE: string;
    };
    class LayerConnection {
        ID: number;
        from: Layer;
        to: Layer;
        selfconnection: boolean;
        type: string;
        connections: Synaptic.Dictionary<neuron.Neuron.Connection>;
        list: neuron.Neuron.Connection[];
        size: number;
        gatedfrom: any[];
        constructor(fromLayer: any, toLayer: any, type: any, weights: any);
    }
}
