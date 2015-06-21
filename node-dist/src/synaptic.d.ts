import network = require('./network');
import layer = require('./layer');
import neuron = require('./neuron');
import trainer = require('./trainer');
import architect = require('./architect');
import squash = require('./squash');
declare module Synaptic {
    interface Dictionary<T> {
        [id: string]: T;
    }
    function ninja(): typeof Synaptic;
    interface ICompiledParameters {
        memory?: any;
        neurons?: number;
        inputs?: any[];
        outputs?: any[];
        targets?: any[];
        variables?: any;
        activation_sentences?: any[];
        trace_sentences?: any[];
        propagation_sentences?: any[];
        layers?: any;
    }
    var Neuron: typeof neuron.Neuron;
    var Layer: typeof layer.Layer;
    var Network: typeof network.Network;
    var Trainer: typeof trainer.Trainer;
    var Squash: typeof squash;
    var Architect: typeof architect;
}
export = Synaptic;
