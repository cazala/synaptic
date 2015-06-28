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
			this.list[n].squash = Squash.IDENTITY;
		}
	}

	activate(input?: Synaptic.INumericArray): Synaptic.INumericArray {
		if (this.currentActivation.length != this.list.length)
			this.currentActivation = new Float64Array(this.list.length);

		var activationIndex = 0;

		var sum = 0;
		var Amax = null;
		
		if (typeof input != 'undefined') {
			if (input.length != this.size)
				throw "INPUT size and LAYER size must be the same to activate!";

			Utils.softMax(input);

			for (var id in this.list) {
				this.list[id].readIncommingConnections(input[id]);
				if (Amax === null || this.list[id].activation > Amax)
					Amax = this.list[id].activation;
			}
		} else {
			for (var id in this.list) {
				this.list[id].readIncommingConnections();
				if (Amax === null || this.list[id].activation > Amax)
					Amax = this.list[id].activation;
			}
		}

		for (var n = 0; n < this.currentActivation.length; n++) {
			sum += (this.list[n].activation = Math.exp(this.list[n].activation - Amax));
		}
		
		for (var n = 0; n < this.currentActivation.length; n++) {
			// set the activations
			var x = this.list[n].activation / sum;
			this.list[n].activation = this.currentActivation[n] = x;
			
			// set the derivatives
			
			//x = this.list[n].activation / (sum - this.list[n].activation);
			
			this.list[n].derivative = x*(1-x);
			
			this.list[n].updateTraces();
		}
		
		return this.currentActivation;
	}

	static NormalizeConnectionWeights(layerConnection: Layer.Layer.LayerConnection) {
		var sum = 0;
		for (var c = 0; c < layerConnection.list.length; c++) {
			sum += (layerConnection.list[c].weight = Math.exp(layerConnection.list[c].weight));
		}
		for (var c = 0; c < layerConnection.list.length; c++) {
			layerConnection.list[c].weight /= sum;
		}
	}
}