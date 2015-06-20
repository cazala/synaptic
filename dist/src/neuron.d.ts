/// <reference path="synaptic.d.ts" />
import Synaptic = require('./synaptic');
import Squash = require('./squash');
/******************************************************************************************
                                         NEURON
*******************************************************************************************/
export declare class Neuron {
    ID: number;
    label: any;
    connections: Neuron.INeuronConnections;
    error: {
        responsibility: number;
        projected: number;
        gated: number;
    };
    trace: {
        elegibility: {};
        extended: {};
        influences: {};
    };
    state: number;
    old: number;
    activation: number;
    selfconnection: Neuron.Connection;
    squash: typeof Squash.LOGISTIC;
    neighboors: {};
    bias: number;
    derivative: number;
    activate(input?: number): number;
    propagate(rate: number, target?: number): void;
    project(neuron: any, weight?: number): Neuron.Connection;
    gate(connection: any): void;
    selfconnected(): boolean;
    connected(neuron: any): {
        type: string;
        connection: Neuron.Connection;
    };
    clear(): void;
    reset(): void;
    optimize(optimized: any, layer: any): Synaptic.ICompiledParameters;
}
export declare module Neuron {
    interface INeuronConnections {
        inputs: Synaptic.Dictionary<Neuron.Connection>;
        projected: {};
        gated: {};
    }
    class Connection {
        ID: number;
        from: any;
        to: any;
        gain: number;
        weight: number;
        gater: any;
        constructor(from: any, to: any, weight?: number);
    }
    var neuronQty: number;
    function uid(): number;
    function quantity(): {
        neurons: number;
        connections: number;
    };
}
export declare module Neuron.Connection {
    var connectionQty: number;
    function uid(): number;
}
