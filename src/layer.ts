import neuron = require('./neuron');
import network = require('./network');
import Synaptic = require('./synaptic');

/*******************************************************************************************
                                            LAYER
*******************************************************************************************/
export class Layer {
	optimizable = true;
	list: neuron.Neuron[] = [];
	label: string = null;
	connectedto = [];
	size = 0;

	currentActivation: Float64Array;

	constructor(size: number, label?: string) {
		this.size = size | 0;
		this.list = [];
		this.label = label || null;
		this.connectedto = [];

		this.currentActivation = new Float64Array(size);

		while (size--) {
			var theNeuron = new neuron.Neuron();
			this.list.push(theNeuron);
		}
	}

	
	// activates all the neurons in the layer
	activate(input?: Synaptic.INumericArray): Synaptic.INumericArray {

		if (this.currentActivation.length != this.list.length)
			this.currentActivation = new Float64Array(this.list.length);

		var activationIndex = 0;

		if (typeof input != 'undefined') {
			if (input.length != this.size)
				throw "INPUT size and LAYER size must be the same to activate!";

			for (var id in this.list) {
				this.currentActivation[activationIndex++] = this.list[id].activate(input[id]);
			}
		} else {
			for (var id in this.list) {
				this.currentActivation[activationIndex++] = this.list[id].activate();
			}
		}

		return this.currentActivation;
	}

	// propagates the error on all the neurons of the layer
	propagate(rate: number, target?: Synaptic.INumericArray) {
		if (typeof target != 'undefined') {
			if (target.length != this.size)
				throw "TARGET size and LAYER size must be the same to propagate!";

			for (var id = this.list.length - 1; id >= 0; id--) {
				var neuron = this.list[id];
				neuron.propagate(rate, target[id]);
			}
		} else {
			for (var id = this.list.length - 1; id >= 0; id--) {
				var neuron = this.list[id];
				neuron.propagate(rate);
			}
		}
	}

	// projects a connection from this layer to another one
	project(layer: network.Network | Layer, type?: string, weights?: Synaptic.INumericArray) {

		if (layer instanceof network.Network)
			layer = (<network.Network>layer).layers.input;

		if (layer instanceof Layer) {
			if (!this.connected(layer))
				return new Layer.LayerConnection(this, layer, type, weights);
		} else
			throw "Invalid argument, you can only project connections to LAYERS and NETWORKS!";


	}

	// gates a connection betwenn two layers
	gate(connection, type) {

		if (type == Layer.gateType.INPUT) {
			if (connection.to.size != this.size)
				throw "GATER layer and CONNECTION.TO layer must be the same size in order to gate!";

			for (var id in connection.to.list) {
				var neuron = connection.to.list[id];
				var gater = this.list[id];
				for (var input in neuron.connections.inputs) {
					var gated = neuron.connections.inputs[input];
					if (gated.ID in connection.connections)
						gater.gate(gated);
				}
			}
		} else if (type == Layer.gateType.OUTPUT) {
			if (connection.from.size != this.size)
				throw "GATER layer and CONNECTION.FROM layer must be the same size in order to gate!";

			for (var id in connection.from.list) {
				var neuron = connection.from.list[id];
				var gater = this.list[id];
				for (var projected in neuron.connections.projected) {
					var gated = neuron.connections.projected[projected];
					if (gated.ID in connection.connections)
						gater.gate(gated);
				}
			}
		} else if (type == Layer.gateType.ONE_TO_ONE) {
			if (connection.size != this.size)
				throw "The number of GATER UNITS must be the same as the number of CONNECTIONS to gate!";

			for (var id in connection.list) {
				var gater = this.list[id];
				var gated = connection.list[id];
				gater.gate(gated);
			}
		}
		connection.gatedfrom.push({ layer: this, type: type });
	}

	// true or false whether the whole layer is self-connected or not
	selfconnected(): boolean {

		for (var id in this.list) {
			var neuron = this.list[id];
			if (!neuron.selfconnected())
				return false;
		}
		return true;
	}

	// true of false whether the layer is connected to another layer (parameter) or not
	connected(layer) {
		// Check if ALL to ALL connection
		var connections = 0;
		for (var here in this.list) {
			for (var there in layer.list) {
				var from = this.list[here];
				var to = layer.list[there];
				var connected = from.connected(to);
				if (connected && connected.type == 'projected')
					connections++;
			}
		}
		if (connections == this.size * layer.size)
			return Layer.connectionType.ALL_TO_ALL;

		// Check if ONE to ONE connection
		connections = 0;
		for (var neuron in this.list) {
			var from = this.list[neuron];
			var to = layer.list[neuron];
			var connected = from.connected(to);
			if (connected && connected.type == 'projected')
				connections++;
		}
		if (connections == this.size)
			return Layer.connectionType.ONE_TO_ONE;
	}

	// clears all the neuorns in the layer
	clear() {
		for (var id in this.list) {
			var neuron = this.list[id];
			neuron.clear();
		}
	}

	// resets all the neurons in the layer
	reset() {
		for (var id in this.list) {
			var neuron = this.list[id];
			neuron.reset();
		}
	}

	// returns all the neurons in the layer (array)
	neurons(): neuron.Neuron[] {
		return this.list;
	}

	// adds a neuron to the layer
	add(neuron) {
		neuron = neuron || new neuron.Neuron();
		this.neurons[neuron.ID] = neuron;
		this.list.push(neuron);
		this.size++;
	}

	set(options) {
		options = options || {};

		for (var i in this.list) {
			var neuron = this.list[i];
			if (options.label)
				neuron.label = options.label + '_' + neuron.ID;
			if (options.squash)
				neuron.squash = options.squash;
			if (options.bias)
				neuron.bias = options.bias;
		}
		return this;
	}
}


export module Layer {
	export var layerQty = 0;
	export function uid() {
		return layerQty++;
	}
	
	// types of connections
	export var connectionType = {
		ALL_TO_ALL: "ALL TO ALL",
		ONE_TO_ONE: "ONE TO ONE",
		ALL_TO_ELSE: "ALL TO ELSE"
	};

	// types of gates
	export var gateType = {
		INPUT: "INPUT",
		OUTPUT: "OUTPUT",
		ONE_TO_ONE: "ONE TO ONE"
	};

	// represents a connection from one layer to another, and keeps track of its weight and gain
	export class LayerConnection {
		ID = uid();
		from: Layer;
		to: Layer;
		selfconnection: boolean = false;
		type: string;
		connections: Synaptic.Dictionary<neuron.Neuron.Connection>;
		list: neuron.Neuron.Connection[];
		size = 0;
		gatedfrom = [];

		constructor(fromLayer, toLayer, type, weights) {
			this.from = fromLayer;
			this.to = toLayer;
			this.selfconnection = toLayer == fromLayer;
			this.type = type;
			this.connections = {};
			this.list = [];
			this.size = 0;
			this.gatedfrom = [];


			if (typeof this.type == 'undefined') {
				if (fromLayer == toLayer)
					this.type = Layer.connectionType.ONE_TO_ONE;
				else
					this.type = Layer.connectionType.ALL_TO_ALL;
			}

			if (this.type == Layer.connectionType.ALL_TO_ALL ||
				this.type == Layer.connectionType.ALL_TO_ELSE) {
				for (var here in this.from.list) {
					for (var there in this.to.list) {
						var from = this.from.list[here];
						var to = this.to.list[there];
						if (this.type == Layer.connectionType.ALL_TO_ELSE && from == to)
							continue;
						var connection = from.project(to, weights);

						this.connections[connection.ID] = connection;
						this.size = this.list.push(connection);
					}
				}
			} else if (this.type == Layer.connectionType.ONE_TO_ONE) {

				for (var neuron in this.from.list) {
					var from = this.from.list[neuron];
					var to = this.to.list[neuron];
					var connection = from.project(to, weights);

					this.connections[connection.ID] = connection;
					this.size = this.list.push(connection);
				}
			}

			fromLayer.connectedto.push(this);
		}
	}
}