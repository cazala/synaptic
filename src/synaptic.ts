/*
********************************************************************************************
                                         SYNAPTIC
********************************************************************************************

Synaptic is a javascript neural network library for node.js and the browser, its generalized
algorithm is architecture-free, so you can build and train basically any type of first order
or even second order neural network architectures.

http://en.wikipedia.org/wiki/Recurrent_neural_network#Second_Order_Recurrent_Neural_Network

The library includes a few built-in architectures like multilayer perceptrons, multilayer
long-short term memory networks (LSTM) or liquid state machines, and a trainer capable of
training any given network, and includes built-in training tasks/tests like solving an XOR,
passing a Distracted Sequence Recall test or an Embeded Reber Grammar test.

The algorithm implemented by this library has been taken from Derek D. Monner's paper:

A generalized LSTM-like training algorithm for second-order recurrent neural networks
http://www.overcomplete.net/papers/nn2012.pdf

There are references to the equations in that paper commented through the source code.


********************************************************************************************/



import network = require('./network');
import layer = require('./layer');
import neuron = require('./neuron');
import trainer = require('./trainer');
import architect = require('./architect');
import squash = require('./squash');

declare var window;

module Synaptic {
	export interface Dictionary<T> {
		[id: string] : T;
	}
	
	var oldSynaptic = typeof window != "undefined" && window && window['Synaptic'];
	
	export function ninja() {
      window['synaptic'] = oldSynaptic; 
      return Synaptic;
	}
	
	export interface ICompiledParameters {	
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
	
	export interface INumericArray {
	  [index: number] : number;
	  length : number;
	}
	
	export var Neuron = neuron.Neuron;
	export var Layer = layer.Layer;
	export var Network = network.Network;
	export var Trainer = trainer.Trainer;
	export var Squash = squash;
	export var Architect = architect;
}

export = Synaptic;

if(typeof window != "undefined") 
	window['synaptic'] = Synaptic;
