import layer = require('./layer');
import Squash = require('./squash');
import Synaptic = require('./synaptic');
import _neuron = require('./neuron');

/*******************************************************************************************
                                          NETWORK
*******************************************************************************************/

declare function escape(a: string): string;


export class Network {
	optimized = null;
	layers = {
		input: null,
		hidden: {},
		output: null
	};
	constructor(layers) {
		if (typeof layers != 'undefined') {
			this.layers = layers || {
				input: null,
				hidden: {},
				output: null
			};
		}
	}

	// feed-forward activation of all the layers to produce an ouput
	activate(input : Synaptic.INumericArray) {

		if (this.optimized === false) {
			this.layers.input.activate(input);
			for (var layer in this.layers.hidden)
				this.layers.hidden[layer].activate();
			return this.layers.output.activate();
		}
		else {
			if (this.optimized == null)
				this.optimize();
			return this.optimized.activate(input);
		}
	}

	// back-propagate the error thru the network
	propagate(rate: number, target?: Synaptic.INumericArray) {

		if (this.optimized === false) {
			this.layers.output.propagate(rate, target);
			var reverse = [];
			for (var layer in this.layers.hidden)
				reverse.push(this.layers.hidden[layer]);
			reverse.reverse();
			for (var layer in reverse)
				reverse[layer].propagate(rate);
		}
		else {
			if (this.optimized == null)
				this.optimize();
			this.optimized.propagate(rate, target);
		}
	}

	// project a connection to another unit (either a network or a layer)
	project(unit, type, weights) {

		if (this.optimized)
			this.optimized.reset();

		if (unit instanceof Network)
			return this.layers.output.project(unit.layers.input, type, weights);

		if (unit instanceof layer.Layer)
			return this.layers.output.project(unit, type, weights);

		throw "Invalid argument, you can only project connections to LAYERS and NETWORKS!";
	}

	// let this network gate a connection
	gate(connection, type) {
		if (this.optimized)
			this.optimized.reset();
		this.layers.output.gate(connection, type);
	}

	// clear all elegibility traces and extended elegibility traces (the network forgets its context, but not what was trained)
	clear() {

		this.restore();

		var inputLayer = this.layers.input,
			outputLayer = this.layers.output;

		inputLayer.clear();
		for (var layer in this.layers.hidden) {
			var hiddenLayer = this.layers.hidden[layer];
			hiddenLayer.clear();
		}
		outputLayer.clear();

		if (this.optimized)
			this.optimized.reset();
	}

	// reset all weights and clear all traces (ends up like a new network)
	reset() {

		this.restore();

		var inputLayer = this.layers.input,
			outputLayer = this.layers.output;

		inputLayer.reset();
		for (var layer in this.layers.hidden) {
			var hiddenLayer = this.layers.hidden[layer];
			hiddenLayer.reset();
		}
		outputLayer.reset();

		if (this.optimized)
			this.optimized.reset();
	}

	// hardcodes the behaviour of the whole network into a single optimized function
	optimize() {

		var that = this;
		var optimized: Synaptic.ICompiledParameters = {};
		var neurons = this.neurons();

		for (var i in neurons) {
			var neuron = neurons[i].neuron;
			var layer = neurons[i].layer;
			/*
			FIXME: does this worked once?
			
			while (neuron.neuron)
				neuron = neuron.neuron;
			*/

			optimized = neuron.optimize(optimized, layer);
		}

		for (var i in optimized.propagation_sentences)
			optimized.propagation_sentences[i].reverse();
		optimized.propagation_sentences.reverse();

		var hardcode = "";
		hardcode += "var F = Float64Array ? new Float64Array(" + optimized.memory +
		") : []; ";
		for (var i in optimized.variables)
			hardcode += "F[" + optimized.variables[i].id + "] = " + (optimized.variables[
				i].value || 0) + "; ";
		hardcode += "var activate = function(input){\n";
		hardcode += "influences = [];";
		for (var i in optimized.inputs)
			hardcode += "F[" + optimized.inputs[i] + "] = input[" + i + "]; ";
		for (var currentLayer in optimized.activation_sentences) {
			if (optimized.activation_sentences[currentLayer].length > 0) {
				for (var currentNeuron in optimized.activation_sentences[currentLayer]) {
					hardcode += optimized.activation_sentences[currentLayer][currentNeuron].join(" ");
					hardcode += optimized.trace_sentences[currentLayer][currentNeuron].join(" ");
				}
			}
		}
		hardcode += " var output = []; "
		for (var i in optimized.outputs)
			hardcode += "output[" + i + "] = F[" + optimized.outputs[i] + "]; ";
		hardcode += "return output; }; "
		hardcode += "var propagate = function(rate, target){\n";
		hardcode += "F[" + optimized.variables.rate.id + "] = rate; ";
		for (var i in optimized.targets)
			hardcode += "F[" + optimized.targets[i] + "] = target[" + i + "]; ";
		for (var currentLayer in optimized.propagation_sentences)
			for (var currentNeuron in optimized.propagation_sentences[currentLayer])
				hardcode += optimized.propagation_sentences[currentLayer][currentNeuron].join(" ") + " ";
		hardcode += " };\n";
		hardcode +=
		"var ownership = function(memoryBuffer){\nF = memoryBuffer;\nthis.memory = F;\n};\n";
		hardcode +=
		"return {\nmemory: F,\nactivate: activate,\npropagate: propagate,\nownership: ownership\n};";
		hardcode = hardcode.split(";").join(";\n");

		var constructor = new Function(hardcode);

		var network = constructor();

		network.data = {
			variables: optimized.variables,
			activate: optimized.activation_sentences,
			propagate: optimized.propagation_sentences,
			trace: optimized.trace_sentences,
			inputs: optimized.inputs,
			outputs: optimized.outputs,
			check_activation: this.activate,
			check_propagation: this.propagate
		}

		network.reset = function() {
			if (that.optimized) {
				that.optimized = null;
				that.activate = network.data.check_activation;
				that.propagate = network.data.check_propagation;
			}
		}

		this.optimized = network;
		this.activate = network.activate;
		this.propagate = network.propagate;
	}

	// restores all the values from the optimized network the their respective objects in order to manipulate the network
	restore() {
		if (!this.optimized)
			return;

		var optimized = this.optimized;

		var getValue = function(...args: any[]) {
			var unit = args.shift();
			var prop = args.pop();

			var id = prop + '_';
			for (var property in args)
				id += args[property] + '_';
			id += unit.ID;

			var memory = optimized.memory;
			var variables = optimized.data.variables;

			if (id in variables)
				return memory[variables[id].id];
			return 0;
		}

		var list = this.neurons();

		// link id's to positions in the array
		var ids = {};
		for (var i in list) {
			var neuron = list[i].neuron;
			/*
			FIXME: does this worked once?
			
			while (neuron.neuron)
				neuron = neuron.neuron;
			*/

			neuron.state = getValue(neuron, 'state');
			neuron.old = getValue(neuron, 'old');
			neuron.activation = getValue(neuron, 'activation');
			neuron.bias = getValue(neuron, 'bias');

			for (var input in neuron.trace.elegibility)
				neuron.trace.elegibility[input] = getValue(neuron, 'trace',
					'elegibility', input);

			for (var gated in neuron.trace.extended)
				for (var input in neuron.trace.extended[gated])
					neuron.trace.extended[gated][input] = getValue(neuron, 'trace',
						'extended', gated, input);
		}

		// get connections
		for (var i in list) {
			var neuron = list[i].neuron;
			/*
			FIXME: does this worked once?
			
			while (neuron.neuron)
				neuron = neuron.neuron;
			*/

			for (var j in neuron.connections.projected) {
				var connection = neuron.connections.projected[j];
				connection.weight = getValue(connection, 'weight');
				connection.gain = getValue(connection, 'gain');
			}
		}
	}

	// returns all the neurons in the network
	neurons(): Network.INetworkNeuron[] {
		var neurons: Network.INetworkNeuron[] = [];

		var inputLayer = this.layers.input.neurons(),
			outputLayer = this.layers.output.neurons();

		for (var neuron in inputLayer)
			neurons.push({
				neuron: inputLayer[neuron],
				layer: 'input'
			});

		for (var layer in this.layers.hidden) {
			var hiddenLayer = this.layers.hidden[layer].neurons();
			for (var neuron in hiddenLayer)
				neurons.push({
					neuron: hiddenLayer[neuron],
					layer: layer
				});
		}
		for (var neuron in outputLayer)
			neurons.push({
				neuron: outputLayer[neuron],
				layer: 'output'
			});

		return neurons;
	}

	// returns number of inputs of the network
	inputs(): number {
		return this.layers.input.size;
	}

	// returns number of outputs of hte network
	outputs(): number {
		return this.layers.output.size;
	}

	// sets the layers of the network
	set(layers) {

		this.layers = layers;
		if (this.optimized)
			this.optimized.reset();
	}

	setOptimize(bool) {
		this.restore();
		if (this.optimized)
			this.optimized.reset();
		this.optimized = bool ? null : false;
	}

	// returns a json that represents all the neurons and connections of the network
	toJSON(ignoreTraces) {

		this.restore();

		var list = this.neurons();
		var neurons = [];
		var connections = [];

		// link id's to positions in the array
		var ids = {};
		for (var i in list) {
			var neuron = list[i].neuron;
			/*
			FIXME: does this worked once?
			
			while (neuron.neuron)
				neuron = neuron.neuron;
			*/

			ids[neuron.ID] = i;

			var copy = {
				trace: {
					elegibility: {},
					extended: {}
				},
				state: neuron.state,
				old: neuron.old,
				activation: neuron.activation,
				bias: neuron.bias,
				layer: list[i].layer,
				squash: null
			};

			copy.squash = neuron.squash == Squash.LOGISTIC ? "LOGISTIC" :
				neuron.squash == Squash.TANH ? "TANH" :
					neuron.squash == Squash.IDENTITY ? "IDENTITY" :
						neuron.squash == Squash.HLIM ? "HLIM" :
							null;

			neurons.push(copy);
		}

		if (!ignoreTraces)
			for (var i in neurons) {
				var copiedNeuron = neurons[i];

				for (var input in neuron.trace.elegibility)
					copiedNeuron.trace.elegibility[input] = neuron.trace.elegibility[input];

				for (var gated in neuron.trace.extended) {
					copiedNeuron.trace.extended[gated] = {};
					for (var input in neuron.trace.extended[gated])
						copiedNeuron.trace.extended[ids[gated]][input] = neuron.trace.extended[
						gated][input];
				}
			}

		// get connections
		for (var i in list) {
			var neuron = list[i].neuron;
				
			/*
			FIXME: does this worked once?
			
			while (neuron.neuron)
				neuron = neuron.neuron;
			*/

			for (var j in neuron.connections.projected) {
				var connection = neuron.connections.projected[j];
				connections.push({
					from: ids[connection.from.ID],
					to: ids[connection.to.ID],
					weight: connection.weight,
					gater: connection.gater ? ids[connection.gater.ID] : null,
				});
			}
			if (neuron.selfconnected())
				connections.push({
					from: ids[neuron.ID],
					to: ids[neuron.ID],
					weight: neuron.selfconnection.weight,
					gater: neuron.selfconnection.gater ? ids[neuron.selfconnection.gater
						.ID] : null,
				});
		}

		return {
			neurons: neurons,
			connections: connections
		}
	}
  
	// export the topology into dot language which can be visualized as graphs using dot
	/* example: ... console.log(net.toDotLang());
				$ node example.js > example.dot
				$ dot example.dot -Tpng > out.png
	*/
	toDot(edgeconnection) {
		if (! typeof edgeconnection)
			edgeconnection = false;
		var code = "digraph nn {\n    rankdir = BT\n";
		var layers = [this.layers.input].concat(this.layers.hidden, this.layers.output);
		for (var layer in layers) {
			for (var to in layers[layer].connectedto) { // projections
				var connection = layers[layer].connectedto[to];
				var layerto = connection.to;
				var size = connection.size;
				var layerID = layers.indexOf(layers[layer]);
				var layertoID = layers.indexOf(layerto);
				/* http://stackoverflow.com/questions/26845540/connect-edges-with-graph-dot
* DOT does not support edge-to-edge connections
* This workaround produces somewhat weird graphs ...
				*/
				if (edgeconnection) {
					if (connection.gatedfrom.length) {
						var fakeNode = "fake" + layerID + "_" + layertoID;
						code += "    " + fakeNode +
						" [label = \"\", shape = point, width = 0.01, height = 0.01]\n";
						code += "    " + layerID + " -> " + fakeNode + " [label = " + size + ", arrowhead = none]\n";
						code += "    " + fakeNode + " -> " + layertoID + "\n";
					} else
						code += "    " + layerID + " -> " + layertoID + " [label = " + size + "]\n";
					for (var from in connection.gatedfrom) { // gatings
						var layerfrom = connection.gatedfrom[from].layer;
						var type = connection.gatedfrom[from].type;
						var layerfromID = layers.indexOf(layerfrom);
						code += "    " + layerfromID + " -> " + fakeNode + " [color = blue]\n";
					}
				} else {
					code += "    " + layerID + " -> " + layertoID + " [label = " + size + "]\n";
					for (var from in connection.gatedfrom) { // gatings
						var layerfrom = connection.gatedfrom[from].layer;
						var type = connection.gatedfrom[from].type;
						var layerfromID = layers.indexOf(layerfrom);
						code += "    " + layerfromID + " -> " + layertoID + " [color = blue]\n";
					}
				}
			}
		}
		code += "}\n";
		return {
			code: code,
			link: "https://chart.googleapis.com/chart?chl=" + escape(code.replace("/ /g", "+")) + "&cht=gv"
		}
	}

	// returns a function that works as the activation of the network and can be used without depending on the library
	standalone() {
		if (!this.optimized)
			this.optimize();

		var data = this.optimized.data;

		// build activation function
		var activation = "function (input) {\n";

		// build inputs
		for (var i in data.inputs)
			activation += "F[" + data.inputs[i] + "] = input[" + i + "];\n";

		// build network activation
		for (var neuron in data.activate) { // shouldn't this be layer?
			for (var sentence in data.activate[neuron])
				activation += data.activate[neuron][sentence] + "\n";
		}

		// build outputs
		activation += "var output = [];\n";
		for (var i in data.outputs)
			activation += "output[" + i + "] = F[" + data.outputs[i] + "];\n";
		activation += "return output;\n}";

		// reference all the positions in memory
		var memory = activation.match(/F\[(\d+)\]/g);
		var dimension = 0;
		var ids = {};
		for (var address in memory) {
			var tmp = memory[address].match(/\d+/)[0];
			if (!(tmp in ids)) {
				ids[tmp] = dimension++;
			}
		}
		var hardcode = "F = {\n";
		for (var i in ids)
			hardcode += ids[i] + ": " + this.optimized.memory[i] + ",\n";
		hardcode = hardcode.substring(0, hardcode.length - 2) + "\n};\n";
		hardcode = "var run = " + activation.replace(/F\[(\d+)]/g, function(
			index) {
			return 'F[' + ids[index.match(/\d+/)[0]] + ']'
		}).replace("{\n", "{\n" + hardcode + "") + ";\n";
		hardcode += "return run";

		// return standalone function
		return new Function(hardcode)();
	}

	worker() {
		if (!this.optimized)
			this.optimize();

		var hardcode = "var inputs = " + this.optimized.data.inputs.length +
			";\n";
		hardcode += "var outputs = " + this.optimized.data.outputs.length +
		";\n";
		hardcode += "var F = null;\n";
		hardcode += "var activate = " + this.optimized.activate.toString() +
		";\n";
		hardcode += "var propagate = " + this.optimized.propagate.toString() +
		";\n";
		hardcode += "onmessage = function(e){\n";
		hardcode += "F = e.data.memoryBuffer;\n";
		hardcode += "if (e.data.action == 'activate'){\n";
		hardcode += "if (e.data.input.length == inputs){\n";
		hardcode +=
		"postMessage( { action: 'activate', output: activate(e.data.input), memoryBuffer: F }, [F.buffer]);\n";
		hardcode += "}\n}\nelse if (e.data.action == 'propagate'){\n";
		hardcode += "propagate(e.data.rate, e.data.target);\n";
		hardcode +=
		"postMessage({ action: 'propagate', memoryBuffer: F }, [F.buffer]);\n";
		hardcode += "}\n}\n";

		var blob = new Blob([hardcode]);
		var blobURL = (<any>window).URL.createObjectURL(blob);

		return new Worker(blobURL);
	}

	// returns a copy of the network
	clone(ignoreTraces) {
		return Network.fromJSON(this.toJSON(ignoreTraces));
	}

	static fromJSON(json) {

		var neurons = [];

		var layers = {
			input: new layer.Layer(0),
			hidden: [],
			output: new layer.Layer(0)
		}
		

		for (var i in json.neurons) {
			var config = json.neurons[i];

			var neuron = new _neuron.Neuron();
			neuron.trace.elegibility = config.trace.elegibility;
			neuron.trace.extended = config.trace.extended;
			neuron.state = config.state;
			neuron.old = config.old;
			neuron.activation = config.activation;
			neuron.bias = config.bias;
			neuron.squash = config.squash in Squash ? Squash[config.squash] :
				Squash.LOGISTIC;
			neurons.push(neuron);

			if (config.layer == 'input')
				layers.input.add(neuron);
			else if (config.layer == 'output')
				layers.output.add(neuron);
			else {
				if (typeof layers.hidden[config.layer] == 'undefined')
					layers.hidden[config.layer] = new layer.Layer(0);
				layers.hidden[config.layer].add(neuron);
			}
		}

		for (var i in json.connections) {
			var config = json.connections[i];
			var from = neurons[config.from];
			var to = neurons[config.to];
			var weight = config.weight
			var gater = neurons[config.gater];

			var connection = from.project(to, weight);
			if (gater)
				gater.gate(connection);
		}

		return new Network(layers);
	}
}

export module Network {
	export interface INetworkNeuron {
		neuron: _neuron.Neuron;
		layer: string;
	}
}