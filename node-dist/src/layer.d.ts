import neuron = require('./neuron');
import Synaptic = require('./synaptic');
/*******************************************************************************************
                                            LAYER
*******************************************************************************************/
export declare class Layer {
    list: neuron.Neuron[];
    label: string;
    connectedto: any[];
    size: number;
    constructor(size: number, label?: string);
    activate(input: any): any[];
    propagate(rate: any, target: any): void;
    project(layer: any, type?: any, weights?: any): Layer.LayerConnection;
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
