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

			if (isNaN(x) || x == Infinity || x == -Infinity) {
				console.log("Activacion se fue al choto.", this.list[n].derivative, x, sum);
			}


			this.currentActivation[n] = x;
			var der = x;
			this.list[n].derivative = x * (1- x);//-((sum - this.list[n].activation) - this.list[n].activation)//x * (1 - x);
			
			if (isNaN(this.list[n].derivative)) {
				console.log("Derivada se fue al choto.", this.list[n].derivative, x, sum);
			}


			this.list[n].updateTraces();
		}

		return this.currentActivation;
	}
}