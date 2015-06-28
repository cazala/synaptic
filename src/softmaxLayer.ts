import Synaptic = require('./synaptic');
import Layer = require('./layer');
import Squash = require('./squash');
import Neuron = require('./neuron');
import _Utils = require('./utils');
var Utils = _Utils.Utils;

export class SoftMaxLayer extends Layer.Layer {
	constructor(size: number, label?: string) {
		super(size, label);

		this.optimizable = false;

		for (var n = 0; n < this.list.length; n++) {
			this.list[n].squash = Squash.EXP;
		}
	}

	activate(input?: Synaptic.INumericArray): Synaptic.INumericArray {
		if (this.currentActivation.length != this.list.length)
			this.currentActivation = new Float64Array(this.list.length);

		var activationIndex = 0;

		var sum = 0;

		if (typeof input != 'undefined') {
			if (input.length != this.size)
				throw "INPUT size and LAYER size must be the same to activate!";

			Utils.softMax(input);

			for (var id in this.list) {
				this.list[id].readIncommingConnections(input[id]);
				sum += this.list[id].activation;
			}
		} else {
			for (var id in this.list) {
				this.list[id].readIncommingConnections();
				sum += this.list[id].activation;
			}
		}
		
		if(isNaN(sum) || sum == Infinity || sum == -Infinity){
				console.log("Sum se fue al choto.", sum);
		}

		for (var n = 0; n < this.currentActivation.length; n++) {
			var x = this.list[n].activation / sum;

			if(x == Infinity){
				x = 1;
				console.log('act infinity', this.list[n].activation , sum);
			}
		
			if (isNaN(x) || x == Infinity || x == -Infinity) {
				console.log("Activacion se fue al choto.", this.list[n].derivative, x, sum);
			}

			this.list[n].activation = this.currentActivation[n] = x;
			
			this.list[n].derivative = x * (1 - x);
			
			if (isNaN(this.list[n].derivative)) {
				console.log("Derivada se fue al choto.", this.list[n].derivative, x, sum);
			}


			this.list[n].updateTraces();
		}

		return this.currentActivation;
	}
	
	propagate(rate: number, target?: Synaptic.INumericArray) {
		/*if (typeof target != 'undefined') {
			for (var n = 0; n < this.currentActivation.length; n++) {
				this.list[n].derivative =  this.list[n].activation-target[n];
			}
		}*/
		super.propagate(rate, target);
	}
	
	static NormalizeConnectionWeights(layerConnection: Layer.Layer.LayerConnection){
		var sum = 0;
		for (var c = 0; c < layerConnection.list.length; c++) {
			sum += (layerConnection.list[c].weight = Math.exp(layerConnection.list[c].weight));
		}
		for (var c = 0; c < layerConnection.list.length; c++) {
			layerConnection.list[c].weight /= sum;
		}
	}
}