import Synaptic = require('./synaptic');
import _neuron = require('./neuron');
export declare class Network {
    optimized: any;
    layers: {
        input: any;
        hidden: {};
        output: any;
    };
    constructor(layers?: any);
    activate(input: Synaptic.INumericArray): any;
    propagate(rate: number, target?: Synaptic.INumericArray): void;
    project(unit: any, type: any, weights: any): any;
    gate(connection: any, type: any): void;
    clear(): void;
    reset(): void;
    optimize(): void;
    restore(): void;
    neurons(): Network.INetworkNeuron[];
    inputs(): number;
    outputs(): number;
    set(layers: any): void;
    setOptimize(bool: any): void;
    toJSON(ignoreTraces: any): {
        neurons: any[];
        connections: any[];
    };
    toDot(edgeconnection: any): {
        code: string;
        link: string;
    };
    standalone(): any;
    worker(): Worker;
    clone(ignoreTraces: any): Network;
    static fromJSON(json: any): Network;
}
export declare module Network {
    interface INetworkNeuron {
        neuron: _neuron.Neuron;
        layer: string;
    }
}
