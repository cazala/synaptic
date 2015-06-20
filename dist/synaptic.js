(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
var hopfield = require('./architect/hopfield');
var lstm = require('./architect/LSTM');
var lsm = require('./architect/Liquid');
var perceptron = require('./architect/Perceptron');
exports.LSTM = lstm.LSTM;
exports.Liquid = lsm.Liquid;
exports.Hopfield = hopfield.Hopfield;
exports.Perceptron = perceptron.Perceptron;

},{"./architect/LSTM":2,"./architect/Liquid":3,"./architect/Perceptron":4,"./architect/hopfield":5}],2:[function(require,module,exports){
var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
var network = require('../network');
var trainer = require('../trainer');
var Layer = require('../layer');
var LSTM = (function (_super) {
    __extends(LSTM, _super);
    function LSTM() {
        var args = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            args[_i - 0] = arguments[_i];
        }
        if (args.length < 3)
            throw "Error: not enough layers (minimum 3) !!";
        var last = args.pop();
        var option = {
            peepholes: Layer.Layer.connectionType.ALL_TO_ALL,
            hiddentohidden: false,
            outtohidden: false,
            outtogates: false,
            intoout: true,
        };
        if (typeof last != 'number') {
            var outputs = args.pop();
            if (last.hasOwnProperty('peepholes'))
                option.peepholes = last.peepholes;
            if (last.hasOwnProperty('hiddentohidden'))
                option.hiddentohidden = last.hiddentohidden;
            if (last.hasOwnProperty('outtohidden'))
                option.outtohidden = last.outtohidden;
            if (last.hasOwnProperty('outtogates'))
                option.outtogates = last.outtogates;
            if (last.hasOwnProperty('intoout'))
                option.intoout = last.intoout;
        }
        else
            var outputs = last;
        var inputs = args.shift();
        var layers = args;
        var inputLayer = new Layer.Layer(inputs);
        var hiddenLayers = [];
        var outputLayer = new Layer.Layer(outputs);
        var previous = null;
        // generate layers
        for (var layer in layers) {
            // generate memory blocks (memory cell and respective gates)
            var size = layers[layer];
            var inputGate = new Layer.Layer(size).set({
                bias: 1
            });
            var forgetGate = new Layer.Layer(size).set({
                bias: 1
            });
            var memoryCell = new Layer.Layer(size);
            var outputGate = new Layer.Layer(size).set({
                bias: 1
            });
            hiddenLayers.push(inputGate);
            hiddenLayers.push(forgetGate);
            hiddenLayers.push(memoryCell);
            hiddenLayers.push(outputGate);
            // connections from input layer
            var input = inputLayer.project(memoryCell);
            inputLayer.project(inputGate);
            inputLayer.project(forgetGate);
            inputLayer.project(outputGate);
            // connections from previous memory-block layer to this one
            if (previous != null) {
                var cell = previous.project(memoryCell);
                previous.project(inputGate);
                previous.project(forgetGate);
                previous.project(outputGate);
            }
            // connections from memory cell
            var output = memoryCell.project(outputLayer);
            // self-connection
            var self = memoryCell.project(memoryCell);
            // hidden to hidden recurrent connection
            if (option.hiddentohidden)
                memoryCell.project(memoryCell, Layer.Layer.connectionType.ALL_TO_ELSE);
            // out to hidden recurrent connection
            if (option.outtohidden)
                outputLayer.project(memoryCell);
            // out to gates recurrent connection
            if (option.outtogates) {
                outputLayer.project(inputGate);
                outputLayer.project(outputGate);
                outputLayer.project(forgetGate);
            }
            // peepholes
            memoryCell.project(inputGate, option.peepholes);
            memoryCell.project(forgetGate, option.peepholes);
            memoryCell.project(outputGate, option.peepholes);
            // gates
            inputGate.gate(input, Layer.Layer.gateType.INPUT);
            forgetGate.gate(self, Layer.Layer.gateType.ONE_TO_ONE);
            outputGate.gate(output, Layer.Layer.gateType.OUTPUT);
            if (previous != null)
                inputGate.gate(cell, Layer.Layer.gateType.INPUT);
            previous = memoryCell;
        }
        // input to output direct connection
        if (option.intoout)
            inputLayer.project(outputLayer);
        // set the layers of the neural network
        _super.call(this, {
            input: inputLayer,
            hidden: hiddenLayers,
            output: outputLayer
        });
        // trainer
        this.trainer = new trainer.Trainer(this);
    }
    return LSTM;
})(network.Network);
exports.LSTM = LSTM;
;

},{"../layer":6,"../network":7,"../trainer":11}],3:[function(require,module,exports){
var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
var network = require('../network');
var trainer = require('../trainer');
var layer = require('../layer');
var Liquid = (function (_super) {
    __extends(Liquid, _super);
    function Liquid(inputs, hidden, outputs, connections, gates) {
        // create layers
        var inputLayer = new layer.Layer(inputs);
        var hiddenLayer = new layer.Layer(hidden);
        var outputLayer = new layer.Layer(outputs);
        // make connections and gates randomly among the neurons
        var neurons = hiddenLayer.neurons();
        var connectionList = [];
        for (var i = 0; i < connections; i++) {
            // connect two random neurons
            var from = Math.random() * neurons.length | 0;
            var to = Math.random() * neurons.length | 0;
            var connection = neurons[from].project(neurons[to]);
            connectionList.push(connection);
        }
        for (var j = 0; j < gates; j++) {
            // pick a random gater neuron
            var gater = Math.random() * neurons.length | 0;
            // pick a random connection to gate
            var connectionNumber = Math.random() * connectionList.length | 0;
            // let the gater gate the connection
            neurons[gater].gate(connectionList[connectionNumber]);
        }
        // connect the layers
        inputLayer.project(hiddenLayer);
        hiddenLayer.project(outputLayer);
        // set the layers of the network
        _super.call(this, {
            input: inputLayer,
            hidden: [hiddenLayer],
            output: outputLayer
        });
        // trainer
        this.trainer = new trainer.Trainer(this);
    }
    return Liquid;
})(network.Network);
exports.Liquid = Liquid;

},{"../layer":6,"../network":7,"../trainer":11}],4:[function(require,module,exports){
var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
var network = require('../network');
var trainer = require('../trainer');
var layer = require('../layer');
// Multilayer Perceptron
var Perceptron = (function (_super) {
    __extends(Perceptron, _super);
    function Perceptron() {
        var args = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            args[_i - 0] = arguments[_i];
        }
        if (args.length < 3)
            throw "Error: not enough layers (minimum 3) !!";
        var inputs = args.shift(); // first argument
        var outputs = args.pop(); // last argument
        var layers = args; // all the arguments in the middle
        var input = new layer.Layer(inputs);
        var hidden = [];
        var output = new layer.Layer(outputs);
        var previous = input;
        // generate hidden layers
        for (var level in layers) {
            var size = layers[level];
            var theLayer = new layer.Layer(size);
            hidden.push(theLayer);
            previous.project(theLayer);
            previous = theLayer;
        }
        previous.project(output);
        // set layers of the neural network
        _super.call(this, {
            input: input,
            hidden: hidden,
            output: output
        });
        // trainer for the network
        this.trainer = new trainer.Trainer(this);
    }
    return Perceptron;
})(network.Network);
exports.Perceptron = Perceptron;
;

},{"../layer":6,"../network":7,"../trainer":11}],5:[function(require,module,exports){
var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
var network = require('../network');
var trainer = require('../trainer');
var layer = require('../layer');
var Hopfield = (function (_super) {
    __extends(Hopfield, _super);
    function Hopfield(size) {
        var inputLayer = new layer.Layer(size);
        var outputLayer = new layer.Layer(size);
        inputLayer.project(outputLayer, layer.Layer.connectionType.ALL_TO_ALL);
        _super.call(this, {
            input: inputLayer,
            hidden: [],
            output: outputLayer
        });
        this.trainer = new trainer.Trainer(this);
    }
    Hopfield.prototype.learn = function (patterns) {
        var set = [];
        for (var p in patterns)
            set.push({
                input: patterns[p],
                output: patterns[p]
            });
        return this.trainer.train(set, {
            iterations: 500000,
            error: .00005,
            rate: 1
        });
    };
    Hopfield.prototype.feed = function (pattern) {
        var output = this.activate(pattern);
        var patterns = [];
        for (var i in output)
            patterns[i] = output[i] > .5 ? 1 : 0;
        return patterns;
    };
    return Hopfield;
})(network.Network);
exports.Hopfield = Hopfield;

},{"../layer":6,"../network":7,"../trainer":11}],6:[function(require,module,exports){
var neuron = require('./neuron');
var network = require('./network');
/*******************************************************************************************
                                            LAYER
*******************************************************************************************/
var Layer = (function () {
    function Layer(size, label) {
        this.list = [];
        this.label = null;
        this.connectedto = [];
        this.size = 0;
        this.size = size | 0;
        this.list = [];
        this.label = label || null;
        this.connectedto = [];
        while (size--) {
            var theNeuron = new neuron.Neuron();
            this.list.push(theNeuron);
        }
    }
    // activates all the neurons in the layer
    Layer.prototype.activate = function (input) {
        var activations = [];
        if (typeof input != 'undefined') {
            if (input.length != this.size)
                throw "INPUT size and LAYER size must be the same to activate!";
            for (var id in this.list) {
                var neuron = this.list[id];
                var activation = neuron.activate(input[id]);
                activations.push(activation);
            }
        }
        else {
            for (var id in this.list) {
                var neuron = this.list[id];
                var activation = neuron.activate();
                activations.push(activation);
            }
        }
        return activations;
    };
    // propagates the error on all the neurons of the layer
    Layer.prototype.propagate = function (rate, target) {
        if (typeof target != 'undefined') {
            if (target.length != this.size)
                throw "TARGET size and LAYER size must be the same to propagate!";
            for (var id = this.list.length - 1; id >= 0; id--) {
                var neuron = this.list[id];
                neuron.propagate(rate, target[id]);
            }
        }
        else {
            for (var id = this.list.length - 1; id >= 0; id--) {
                var neuron = this.list[id];
                neuron.propagate(rate);
            }
        }
    };
    // projects a connection from this layer to another one
    Layer.prototype.project = function (layer, type, weights) {
        if (layer instanceof network.Network)
            layer = layer.layers.input;
        if (layer instanceof Layer) {
            if (!this.connected(layer))
                return new Layer.LayerConnection(this, layer, type, weights);
        }
        else
            throw "Invalid argument, you can only project connections to LAYERS and NETWORKS!";
    };
    // gates a connection betwenn two layers
    Layer.prototype.gate = function (connection, type) {
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
        }
        else if (type == Layer.gateType.OUTPUT) {
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
        }
        else if (type == Layer.gateType.ONE_TO_ONE) {
            if (connection.size != this.size)
                throw "The number of GATER UNITS must be the same as the number of CONNECTIONS to gate!";
            for (var id in connection.list) {
                var gater = this.list[id];
                var gated = connection.list[id];
                gater.gate(gated);
            }
        }
        connection.gatedfrom.push({ layer: this, type: type });
    };
    // true or false whether the whole layer is self-connected or not
    Layer.prototype.selfconnected = function () {
        for (var id in this.list) {
            var neuron = this.list[id];
            if (!neuron.selfconnected())
                return false;
        }
        return true;
    };
    // true of false whether the layer is connected to another layer (parameter) or not
    Layer.prototype.connected = function (layer) {
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
    };
    // clears all the neuorns in the layer
    Layer.prototype.clear = function () {
        for (var id in this.list) {
            var neuron = this.list[id];
            neuron.clear();
        }
    };
    // resets all the neurons in the layer
    Layer.prototype.reset = function () {
        for (var id in this.list) {
            var neuron = this.list[id];
            neuron.reset();
        }
    };
    // returns all the neurons in the layer (array)
    Layer.prototype.neurons = function () {
        return this.list;
    };
    // adds a neuron to the layer
    Layer.prototype.add = function (neuron) {
        neuron = neuron || new neuron.Neuron();
        this.neurons[neuron.ID] = neuron;
        this.list.push(neuron);
        this.size++;
    };
    Layer.prototype.set = function (options) {
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
    };
    return Layer;
})();
exports.Layer = Layer;
var Layer;
(function (Layer) {
    Layer.layerQty = 0;
    function uid() {
        return Layer.layerQty++;
    }
    Layer.uid = uid;
    // types of connections
    Layer.connectionType = {
        ALL_TO_ALL: "ALL TO ALL",
        ONE_TO_ONE: "ONE TO ONE",
        ALL_TO_ELSE: "ALL TO ELSE"
    };
    // types of gates
    Layer.gateType = {
        INPUT: "INPUT",
        OUTPUT: "OUTPUT",
        ONE_TO_ONE: "ONE TO ONE"
    };
    // represents a connection from one layer to another, and keeps track of its weight and gain
    var LayerConnection = (function () {
        function LayerConnection(fromLayer, toLayer, type, weights) {
            this.ID = uid();
            this.selfconnection = false;
            this.size = 0;
            this.gatedfrom = [];
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
            }
            else if (this.type == Layer.connectionType.ONE_TO_ONE) {
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
        return LayerConnection;
    })();
    Layer.LayerConnection = LayerConnection;
})(Layer = exports.Layer || (exports.Layer = {}));

},{"./network":7,"./neuron":8}],7:[function(require,module,exports){
var layer = require('./layer');
var Squash = require('./squash');
var _neuron = require('./neuron');
var Network = (function () {
    function Network(layers) {
        this.optimized = null;
        this.layers = {
            input: null,
            hidden: {},
            output: null
        };
        if (typeof layers != 'undefined') {
            this.layers = layers || {
                input: null,
                hidden: {},
                output: null
            };
            this.optimized = null;
        }
    }
    // feed-forward activation of all the layers to produce an ouput
    Network.prototype.activate = function (input) {
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
    };
    // back-propagate the error thru the network
    Network.prototype.propagate = function (rate, target) {
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
    };
    // project a connection to another unit (either a network or a layer)
    Network.prototype.project = function (unit, type, weights) {
        if (this.optimized)
            this.optimized.reset();
        if (unit instanceof Network)
            return this.layers.output.project(unit.layers.input, type, weights);
        if (unit instanceof layer.Layer)
            return this.layers.output.project(unit, type, weights);
        throw "Invalid argument, you can only project connections to LAYERS and NETWORKS!";
    };
    // let this network gate a connection
    Network.prototype.gate = function (connection, type) {
        if (this.optimized)
            this.optimized.reset();
        this.layers.output.gate(connection, type);
    };
    // clear all elegibility traces and extended elegibility traces (the network forgets its context, but not what was trained)
    Network.prototype.clear = function () {
        this.restore();
        var inputLayer = this.layers.input, outputLayer = this.layers.output;
        inputLayer.clear();
        for (var layer in this.layers.hidden) {
            var hiddenLayer = this.layers.hidden[layer];
            hiddenLayer.clear();
        }
        outputLayer.clear();
        if (this.optimized)
            this.optimized.reset();
    };
    // reset all weights and clear all traces (ends up like a new network)
    Network.prototype.reset = function () {
        this.restore();
        var inputLayer = this.layers.input, outputLayer = this.layers.output;
        inputLayer.reset();
        for (var layer in this.layers.hidden) {
            var hiddenLayer = this.layers.hidden[layer];
            hiddenLayer.reset();
        }
        outputLayer.reset();
        if (this.optimized)
            this.optimized.reset();
    };
    // hardcodes the behaviour of the whole network into a single optimized function
    Network.prototype.optimize = function () {
        var that = this;
        var optimized = {};
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
            hardcode += "F[" + optimized.variables[i].id + "] = " + (optimized.variables[i].value || 0) + "; ";
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
        hardcode += " var output = []; ";
        for (var i in optimized.outputs)
            hardcode += "output[" + i + "] = F[" + optimized.outputs[i] + "]; ";
        hardcode += "return output; }; ";
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
        };
        network.reset = function () {
            if (that.optimized) {
                that.optimized = null;
                that.activate = network.data.check_activation;
                that.propagate = network.data.check_propagation;
            }
        };
        this.optimized = network;
        this.activate = network.activate;
        this.propagate = network.propagate;
    };
    // restores all the values from the optimized network the their respective objects in order to manipulate the network
    Network.prototype.restore = function () {
        if (!this.optimized)
            return;
        var optimized = this.optimized;
        var getValue = function () {
            var args = [];
            for (var _i = 0; _i < arguments.length; _i++) {
                args[_i - 0] = arguments[_i];
            }
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
        };
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
                neuron.trace.elegibility[input] = getValue(neuron, 'trace', 'elegibility', input);
            for (var gated in neuron.trace.extended)
                for (var input in neuron.trace.extended[gated])
                    neuron.trace.extended[gated][input] = getValue(neuron, 'trace', 'extended', gated, input);
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
    };
    // returns all the neurons in the network
    Network.prototype.neurons = function () {
        var neurons = [];
        var inputLayer = this.layers.input.neurons(), outputLayer = this.layers.output.neurons();
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
    };
    // returns number of inputs of the network
    Network.prototype.inputs = function () {
        return this.layers.input.size;
    };
    // returns number of outputs of hte network
    Network.prototype.outputs = function () {
        return this.layers.output.size;
    };
    // sets the layers of the network
    Network.prototype.set = function (layers) {
        this.layers = layers;
        if (this.optimized)
            this.optimized.reset();
    };
    Network.prototype.setOptimize = function (bool) {
        this.restore();
        if (this.optimized)
            this.optimized.reset();
        this.optimized = bool ? null : false;
    };
    // returns a json that represents all the neurons and connections of the network
    Network.prototype.toJSON = function (ignoreTraces) {
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
                        copiedNeuron.trace.extended[ids[gated]][input] = neuron.trace.extended[gated][input];
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
        };
    };
    // export the topology into dot language which can be visualized as graphs using dot
    /* example: ... console.log(net.toDotLang());
                $ node example.js > example.dot
                $ dot example.dot -Tpng > out.png
    */
    Network.prototype.toDot = function (edgeconnection) {
        if (!typeof edgeconnection)
            edgeconnection = false;
        var code = "digraph nn {\n    rankdir = BT\n";
        var layers = [this.layers.input].concat(this.layers.hidden, this.layers.output);
        for (var layer in layers) {
            for (var to in layers[layer].connectedto) {
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
                    }
                    else
                        code += "    " + layerID + " -> " + layertoID + " [label = " + size + "]\n";
                    for (var from in connection.gatedfrom) {
                        var layerfrom = connection.gatedfrom[from].layer;
                        var type = connection.gatedfrom[from].type;
                        var layerfromID = layers.indexOf(layerfrom);
                        code += "    " + layerfromID + " -> " + fakeNode + " [color = blue]\n";
                    }
                }
                else {
                    code += "    " + layerID + " -> " + layertoID + " [label = " + size + "]\n";
                    for (var from in connection.gatedfrom) {
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
        };
    };
    // returns a function that works as the activation of the network and can be used without depending on the library
    Network.prototype.standalone = function () {
        if (!this.optimized)
            this.optimize();
        var data = this.optimized.data;
        // build activation function
        var activation = "function (input) {\n";
        // build inputs
        for (var i in data.inputs)
            activation += "F[" + data.inputs[i] + "] = input[" + i + "];\n";
        // build network activation
        for (var neuron in data.activate) {
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
        hardcode = "var run = " + activation.replace(/F\[(\d+)]/g, function (index) {
            return 'F[' + ids[index.match(/\d+/)[0]] + ']';
        }).replace("{\n", "{\n" + hardcode + "") + ";\n";
        hardcode += "return run";
        // return standalone function
        return new Function(hardcode)();
    };
    Network.prototype.worker = function () {
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
        var blobURL = window.URL.createObjectURL(blob);
        return new Worker(blobURL);
    };
    // returns a copy of the network
    Network.prototype.clone = function (ignoreTraces) {
        return Network.fromJSON(this.toJSON(ignoreTraces));
    };
    Network.fromJSON = function (json) {
        var neurons = [];
        var layers = {
            input: new layer.Layer(0),
            hidden: [],
            output: new layer.Layer(0)
        };
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
            var weight = config.weight;
            var gater = neurons[config.gater];
            var connection = from.project(to, weight);
            if (gater)
                gater.gate(connection);
        }
        return new Network(layers);
    };
    return Network;
})();
exports.Network = Network;

},{"./layer":6,"./neuron":8,"./squash":9}],8:[function(require,module,exports){
/// <reference path="synaptic.ts" />
var Squash = require('./squash');
/******************************************************************************************
                                         NEURON
*******************************************************************************************/
/* TS CHANGES:

    Now Neuron.connected(neuron) returns null instead of false

*/
var Neuron = (function () {
    function Neuron() {
        this.ID = Neuron.uid();
        this.label = null;
        this.connections = {
            inputs: {},
            projected: {},
            gated: {}
        };
        this.error = {
            responsibility: 0,
            projected: 0,
            gated: 0
        };
        this.trace = {
            elegibility: {},
            extended: {},
            influences: {}
        };
        this.state = 0;
        this.old = 0;
        this.activation = 0;
        this.selfconnection = new Neuron.Connection(this, this, 0); // weight = 0 -> not connected
        this.squash = Squash.LOGISTIC;
        this.neighboors = {};
        this.bias = Math.random() * .2 - .1;
        this.derivative = 0;
    }
    // activate the neuron
    Neuron.prototype.activate = function (input) {
        // activation from enviroment (for input neurons)
        if (typeof input != 'undefined') {
            this.activation = input;
            this.derivative = 0;
            this.bias = 0;
            return this.activation;
        }
        // old state
        this.old = this.state;
        // eq. 15
        this.state = this.selfconnection.gain * this.selfconnection.weight *
            this.state + this.bias;
        for (var i in this.connections.inputs) {
            var theInput = this.connections.inputs[i];
            this.state += theInput.from.activation * theInput.weight * theInput.gain;
        }
        // eq. 16
        this.activation = this.squash(this.state);
        // f'(s)
        this.derivative = this.squash(this.state, true);
        // update traces
        var influences = [];
        for (var id in this.trace.extended) {
            // extended elegibility trace
            var xtrace = this.trace.extended[id];
            var neuron = this.neighboors[id];
            // if gated neuron's selfconnection is gated by this unit, the influence keeps track of the neuron's old state
            var influence = neuron.selfconnection.gater == this ? neuron.old : 0;
            // index runs over all the incoming connections to the gated neuron that are gated by this unit
            for (var incoming in this.trace.influences[neuron.ID]) {
                influence += this.trace.influences[neuron.ID][incoming].weight *
                    this.trace.influences[neuron.ID][incoming].from.activation;
            }
            influences[neuron.ID] = influence;
        }
        for (var i in this.connections.inputs) {
            var theInput = this.connections.inputs[i];
            // elegibility trace - Eq. 17
            this.trace.elegibility[theInput.ID] = this.selfconnection.gain * this.selfconnection
                .weight * this.trace.elegibility[theInput.ID] + theInput.gain * theInput.from
                .activation;
            for (var id in this.trace.extended) {
                // extended elegibility trace
                var xtrace = this.trace.extended[id];
                var neuron = this.neighboors[id];
                var influence = influences[neuron.ID];
                // eq. 18
                xtrace[theInput.ID] = neuron.selfconnection.gain * neuron.selfconnection
                    .weight * xtrace[theInput.ID] + this.derivative * this.trace.elegibility[theInput.ID] * influence;
            }
        }
        //  update gated connection's gains
        for (var connection in this.connections.gated) {
            this.connections.gated[connection].gain = this.activation;
        }
        return this.activation;
    };
    // back-propagate the error
    Neuron.prototype.propagate = function (rate, target) {
        // error accumulator
        var error = 0;
        // whether or not this neuron is in the output layer
        var isOutput = typeof target != 'undefined' && target != null;
        // output neurons get their error from the enviroment
        if (isOutput)
            this.error.responsibility = this.error.projected = target - this.activation; // Eq. 10
        else {
            // error responsibilities from all the connections projected from this neuron
            for (var id in this.connections.projected) {
                var connection = this.connections.projected[id];
                var neuron = connection.to;
                // Eq. 21
                error += neuron.error.responsibility * connection.gain * connection.weight;
            }
            // projected error responsibility
            this.error.projected = this.derivative * error;
            error = 0;
            // error responsibilities from all the connections gated by this neuron
            for (var id in this.trace.extended) {
                var neuron = this.neighboors[id]; // gated neuron
                var influence = neuron.selfconnection.gater == this ? neuron.old : 0; // if gated neuron's selfconnection is gated by this neuron
                // index runs over all the connections to the gated neuron that are gated by this neuron
                for (var input in this.trace.influences[id]) {
                    influence += this.trace.influences[id][input].weight * this.trace.influences[neuron.ID][input].from.activation;
                }
                // eq. 22
                error += neuron.error.responsibility * influence;
            }
            // gated error responsibility
            this.error.gated = this.derivative * error;
            // error responsibility - Eq. 23
            this.error.responsibility = this.error.projected + this.error.gated;
        }
        // learning rate
        rate = rate || .1;
        // adjust all the neuron's incoming connections
        for (var id in this.connections.inputs) {
            var theInput = this.connections.inputs[id];
            // Eq. 24
            var gradient = this.error.projected * this.trace.elegibility[theInput.ID];
            for (var id in this.trace.extended) {
                var neuron = this.neighboors[id];
                gradient += neuron.error.responsibility * this.trace.extended[neuron.ID][theInput.ID];
            }
            theInput.weight += rate * gradient; // adjust weights - aka learn
        }
        // adjust bias
        this.bias += rate * this.error.responsibility;
    };
    Neuron.prototype.project = function (neuron, weight) {
        // self-connection
        if (neuron == this) {
            this.selfconnection.weight = 1;
            return this.selfconnection;
        }
        // check if connection already exists
        var connected = this.connected(neuron);
        if (connected && connected.type == "projected") {
            // update connection
            if (typeof weight != 'undefined')
                connected.connection.weight = weight;
            // return existing connection
            return connected.connection;
        }
        else {
            // create a new connection
            var connection = new Neuron.Connection(this, neuron, weight);
        }
        // reference all the connections and traces
        this.connections.projected[connection.ID] = connection;
        this.neighboors[neuron.ID] = neuron;
        neuron.connections.inputs[connection.ID] = connection;
        neuron.trace.elegibility[connection.ID] = 0;
        for (var id in neuron.trace.extended) {
            var trace = neuron.trace.extended[id];
            trace[connection.ID] = 0;
        }
        return connection;
    };
    Neuron.prototype.gate = function (connection) {
        // add connection to gated list
        this.connections.gated[connection.ID] = connection;
        var neuron = connection.to;
        if (!(neuron.ID in this.trace.extended)) {
            // extended trace
            this.neighboors[neuron.ID] = neuron;
            var xtrace = this.trace.extended[neuron.ID] = {};
            for (var id in this.connections.inputs) {
                var input = this.connections.inputs[id];
                xtrace[input.ID] = 0;
            }
        }
        // keep track
        if (neuron.ID in this.trace.influences)
            this.trace.influences[neuron.ID].push(connection);
        else
            this.trace.influences[neuron.ID] = [connection];
        // set gater
        connection.gater = this;
    };
    // returns true or false whether the neuron is self-connected or not
    Neuron.prototype.selfconnected = function () {
        return this.selfconnection.weight !== 0;
    };
    // returns true or false whether the neuron is connected to another neuron (parameter)
    Neuron.prototype.connected = function (neuron) {
        var result = {
            type: null,
            connection: null
        };
        if (this == neuron) {
            if (this.selfconnected()) {
                result.type = 'selfconnection';
                result.connection = this.selfconnection;
                return result;
            }
            else
                return null;
        }
        for (var type in this.connections) {
            for (var connection in this.connections[type]) {
                var connection = this.connections[type][connection];
                if (connection.to == neuron) {
                    result.type = type;
                    result.connection = connection;
                    return result;
                }
                else if (connection.from == neuron) {
                    result.type = type;
                    result.connection = connection;
                    return result;
                }
            }
        }
        return null;
    };
    // clears all the traces (the neuron forgets it's context, but the connections remain intact)
    Neuron.prototype.clear = function () {
        for (var trace in this.trace.elegibility)
            this.trace.elegibility[trace] = 0;
        for (var trace in this.trace.extended)
            for (var extended in this.trace.extended[trace])
                this.trace.extended[trace][extended] = 0;
        this.error.responsibility = this.error.projected = this.error.gated = 0;
    };
    // all the connections are randomized and the traces are cleared
    Neuron.prototype.reset = function () {
        this.clear();
        for (var type in this.connections)
            for (var connection in this.connections[type])
                this.connections[type][connection].weight = Math.random() * .2 - .1;
        this.bias = Math.random() * .2 - .1;
        this.old = this.state = this.activation = 0;
    };
    // hardcodes the behaviour of the neuron into an optimized function
    Neuron.prototype.optimize = function (optimized, layer) {
        optimized = optimized || {};
        var that = this;
        var store_activation = [];
        var store_trace = [];
        var store_propagation = [];
        var varID = optimized.memory || 0;
        var neurons = optimized.neurons || 1;
        var inputs = optimized.inputs || [];
        var targets = optimized.targets || [];
        var outputs = optimized.outputs || [];
        var variables = optimized.variables || {};
        var activation_sentences = optimized.activation_sentences || [];
        var trace_sentences = optimized.trace_sentences || [];
        var propagation_sentences = optimized.propagation_sentences || [];
        var layers = optimized.layers || { __count: 0, __neuron: 0 };
        // allocate sentences
        var allocate = function (store) {
            var allocated = layer in layers && store[layers.__count];
            if (!allocated) {
                layers.__count = store.push([]) - 1;
                layers[layer] = layers.__count;
            }
        };
        allocate(activation_sentences);
        allocate(trace_sentences);
        allocate(propagation_sentences);
        var currentLayer = layers.__count;
        // get/reserve space in memory by creating a unique ID for a variablel
        var getVar = function () {
            var args = [];
            for (var _i = 0; _i < arguments.length; _i++) {
                args[_i - 0] = arguments[_i];
            }
            var id;
            if (args.length == 1) {
                if (args[0] == 'target') {
                    id = 'target_' + targets.length;
                    targets.push(varID);
                }
                else
                    id = args[0];
                if (id in variables)
                    return variables[id];
                return variables[id] = {
                    value: 0,
                    id: varID++
                };
            }
            else {
                var extended = args.length > 2;
                if (extended)
                    var value = args.pop();
                var unit = args.shift();
                var prop = args.pop();
                if (!extended)
                    var value = unit[prop];
                id = prop + '_';
                for (var property in args)
                    id += args[property] + '_';
                id += unit.ID;
                if (id in variables)
                    return variables[id];
                return variables[id] = {
                    value: value,
                    id: varID++
                };
            }
        };
        // build sentence
        var buildSentence = function () {
            var args = [];
            for (var _i = 0; _i < arguments.length; _i++) {
                args[_i - 0] = arguments[_i];
            }
            var store = args.pop();
            var sentence = "";
            for (var i in args)
                if (typeof args[i] == 'string')
                    sentence += args[i];
                else
                    sentence += 'F[' + args[i].id + ']';
            store.push(sentence + ';');
        };
        // helper to check if an object is empty
        var isEmpty = function (obj) {
            for (var prop in obj) {
                if (obj.hasOwnProperty(prop))
                    return false;
            }
            return true;
        };
        // characteristics of the neuron
        var noProjections = isEmpty(this.connections.projected);
        var noGates = isEmpty(this.connections.gated);
        var isInput = layer == 'input' ? true : isEmpty(this.connections.inputs);
        var isOutput = layer == 'output' ? true : noProjections && noGates;
        // optimize neuron's behaviour
        var rate = getVar('rate');
        var activation = getVar(this, 'activation');
        if (isInput)
            inputs.push(activation.id);
        else {
            activation_sentences[currentLayer].push(store_activation);
            trace_sentences[currentLayer].push(store_trace);
            propagation_sentences[currentLayer].push(store_propagation);
            var old = getVar(this, 'old');
            var state = getVar(this, 'state');
            var bias = getVar(this, 'bias');
            if (this.selfconnection.gater)
                var self_gain = getVar(this.selfconnection, 'gain');
            if (this.selfconnected())
                var self_weight = getVar(this.selfconnection, 'weight');
            buildSentence(old, ' = ', state, store_activation);
            if (this.selfconnected())
                if (this.selfconnection.gater)
                    buildSentence(state, ' = ', self_gain, ' * ', self_weight, ' * ', state, ' + ', bias, store_activation);
                else
                    buildSentence(state, ' = ', self_weight, ' * ', state, ' + ', bias, store_activation);
            else
                buildSentence(state, ' = ', bias, store_activation);
            for (var i in this.connections.inputs) {
                var input = this.connections.inputs[i];
                var input_activation = getVar(input.from, 'activation');
                var input_weight = getVar(input, 'weight');
                if (input.gater)
                    var input_gain = getVar(input, 'gain');
                if (this.connections.inputs[i].gater)
                    buildSentence(state, ' += ', input_activation, ' * ', input_weight, ' * ', input_gain, store_activation);
                else
                    buildSentence(state, ' += ', input_activation, ' * ', input_weight, store_activation);
            }
            var derivative = getVar(this, 'derivative');
            switch (this.squash) {
                case Squash.LOGISTIC:
                    buildSentence(activation, ' = (1 / (1 + Math.exp(-', state, ')))', store_activation);
                    buildSentence(derivative, ' = ', activation, ' * (1 - ', activation, ')', store_activation);
                    break;
                case Squash.TANH:
                    var eP = getVar('aux');
                    var eN = getVar('aux_2');
                    buildSentence(eP, ' = Math.exp(', state, ')', store_activation);
                    buildSentence(eN, ' = 1 / ', eP, store_activation);
                    buildSentence(activation, ' = (', eP, ' - ', eN, ') / (', eP, ' + ', eN, ')', store_activation);
                    buildSentence(derivative, ' = 1 - (', activation, ' * ', activation, ')', store_activation);
                    break;
                case Squash.IDENTITY:
                    buildSentence(activation, ' = ', state, store_activation);
                    buildSentence(derivative, ' = 1', store_activation);
                    break;
                case Squash.HLIM:
                    buildSentence(activation, ' = +(', state, ' > 0)', store_activation);
                    buildSentence(derivative, ' = 1', store_activation);
                    break;
            }
            var influences = [];
            for (var id in this.trace.extended) {
                // calculate extended elegibility traces in advance
                var xtrace = this.trace.extended[id];
                var neuron = this.neighboors[id];
                var influence = getVar('aux');
                var neuron_old = getVar(neuron, 'old');
                var initialized = false;
                if (neuron.selfconnection.gater == this) {
                    buildSentence(influence, ' = ', neuron_old, store_trace);
                    initialized = true;
                }
                for (var incoming in this.trace.influences[neuron.ID]) {
                    var incoming_weight = getVar(this.trace.influences[neuron.ID][incoming], 'weight');
                    var incoming_activation = getVar(this.trace.influences[neuron.ID][incoming].from, 'activation');
                    if (initialized)
                        buildSentence(influence, ' += ', incoming_weight, ' * ', incoming_activation, store_trace);
                    else {
                        buildSentence(influence, ' = ', incoming_weight, ' * ', incoming_activation, store_trace);
                        initialized = true;
                    }
                }
                influences.push(neuron.ID);
                buildSentence("influences[" + (influences.length - 1) + "] = ", influence, store_trace);
            }
            for (var i in this.connections.inputs) {
                var input = this.connections.inputs[i];
                if (input.gater)
                    var input_gain = getVar(input, 'gain');
                var input_activation = getVar(input.from, 'activation');
                var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace
                    .elegibility[input.ID]);
                if (this.selfconnected()) {
                    if (this.selfconnection.gater) {
                        if (input.gater)
                            buildSentence(trace, ' = ', self_gain, ' * ', self_weight, ' * ', trace, ' + ', input_gain, ' * ', input_activation, store_trace);
                        else
                            buildSentence(trace, ' = ', self_gain, ' * ', self_weight, ' * ', trace, ' + ', input_activation, store_trace);
                    }
                    else {
                        if (input.gater)
                            buildSentence(trace, ' = ', self_weight, ' * ', trace, ' + ', input_gain, ' * ', input_activation, store_trace);
                        else
                            buildSentence(trace, ' = ', self_weight, ' * ', trace, ' + ', input_activation, store_trace);
                    }
                }
                else {
                    if (input.gater)
                        buildSentence(trace, ' = ', input_gain, ' * ', input_activation, store_trace);
                    else
                        buildSentence(trace, ' = ', input_activation, store_trace);
                }
                for (var id in this.trace.extended) {
                    // extended elegibility trace
                    var xtrace = this.trace.extended[id];
                    var neuron = this.neighboors[id];
                    var influence = getVar('aux');
                    var neuron_old = getVar(neuron, 'old');
                    var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace
                        .elegibility[input.ID]);
                    var xtrace = getVar(this, 'trace', 'extended', neuron.ID, input.ID, this.trace.extended[neuron.ID][input.ID]);
                    if (neuron.selfconnected())
                        var neuron_self_weight = getVar(neuron.selfconnection, 'weight');
                    if (neuron.selfconnection.gater)
                        var neuron_self_gain = getVar(neuron.selfconnection, 'gain');
                    if (neuron.selfconnected())
                        if (neuron.selfconnection.gater)
                            buildSentence(xtrace, ' = ', neuron_self_gain, ' * ', neuron_self_weight, ' * ', xtrace, ' + ', derivative, ' * ', trace, ' * ', "influences[" + influences.indexOf(neuron.ID) + "]", store_trace);
                        else
                            buildSentence(xtrace, ' = ', neuron_self_weight, ' * ', xtrace, ' + ', derivative, ' * ', trace, ' * ', "influences[" + influences.indexOf(neuron.ID) + "]", store_trace);
                    else
                        buildSentence(xtrace, ' = ', derivative, ' * ', trace, ' * ', "influences[" + influences.indexOf(neuron.ID) + "]", store_trace);
                }
            }
            for (var connection in this.connections.gated) {
                var gated_gain = getVar(this.connections.gated[connection], 'gain');
                buildSentence(gated_gain, ' = ', activation, store_activation);
            }
        }
        if (!isInput) {
            var responsibility = getVar(this, 'error', 'responsibility', this.error
                .responsibility);
            if (isOutput) {
                var target = getVar('target');
                buildSentence(responsibility, ' = ', target, ' - ', activation, store_propagation);
                for (var id in this.connections.inputs) {
                    var input = this.connections.inputs[id];
                    var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace
                        .elegibility[input.ID]);
                    var input_weight = getVar(input, 'weight');
                    buildSentence(input_weight, ' += ', rate, ' * (', responsibility, ' * ', trace, ')', store_propagation);
                }
                outputs.push(activation.id);
            }
            else {
                if (!noProjections && !noGates) {
                    var error = getVar('aux');
                    for (var id in this.connections.projected) {
                        var connection = this.connections.projected[id];
                        var neuron = connection.to;
                        var connection_weight = getVar(connection, 'weight');
                        var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                        if (connection.gater) {
                            var connection_gain = getVar(connection, 'gain');
                            buildSentence(error, ' += ', neuron_responsibility, ' * ', connection_gain, ' * ', connection_weight, store_propagation);
                        }
                        else
                            buildSentence(error, ' += ', neuron_responsibility, ' * ', connection_weight, store_propagation);
                    }
                    var projected = getVar(this, 'error', 'projected', this.error.projected);
                    buildSentence(projected, ' = ', derivative, ' * ', error, store_propagation);
                    buildSentence(error, ' = 0', store_propagation);
                    for (var id in this.trace.extended) {
                        var neuron = this.neighboors[id];
                        var influence = getVar('aux_2');
                        var neuron_old = getVar(neuron, 'old');
                        if (neuron.selfconnection.gater == this)
                            buildSentence(influence, ' = ', neuron_old, store_propagation);
                        else
                            buildSentence(influence, ' = 0', store_propagation);
                        for (var influenceInput in this.trace.influences[neuron.ID]) {
                            var connection = this.trace.influences[neuron.ID][influenceInput];
                            var connection_weight = getVar(connection, 'weight');
                            var neuron_activation = getVar(connection.from, 'activation');
                            buildSentence(influence, ' += ', connection_weight, ' * ', neuron_activation, store_propagation);
                        }
                        var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                        buildSentence(error, ' += ', neuron_responsibility, ' * ', influence, store_propagation);
                    }
                    var gated = getVar(this, 'error', 'gated', this.error.gated);
                    buildSentence(gated, ' = ', derivative, ' * ', error, store_propagation);
                    buildSentence(responsibility, ' = ', projected, ' + ', gated, store_propagation);
                    for (var id in this.connections.inputs) {
                        var input = this.connections.inputs[id];
                        var gradient = getVar('aux');
                        var trace = getVar(this, 'trace', 'elegibility', input.ID, this
                            .trace.elegibility[input.ID]);
                        buildSentence(gradient, ' = ', projected, ' * ', trace, store_propagation);
                        for (var id in this.trace.extended) {
                            var neuron = this.neighboors[id];
                            var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                            var xtrace = getVar(this, 'trace', 'extended', neuron.ID, input.ID, this.trace.extended[neuron.ID][input.ID]);
                            buildSentence(gradient, ' += ', neuron_responsibility, ' * ', xtrace, store_propagation);
                        }
                        var input_weight = getVar(input, 'weight');
                        buildSentence(input_weight, ' += ', rate, ' * ', gradient, store_propagation);
                    }
                }
                else if (noGates) {
                    buildSentence(responsibility, ' = 0', store_propagation);
                    for (var id in this.connections.projected) {
                        var connection = this.connections.projected[id];
                        var neuron = connection.to;
                        var connection_weight = getVar(connection, 'weight');
                        var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                        if (connection.gater) {
                            var connection_gain = getVar(connection, 'gain');
                            buildSentence(responsibility, ' += ', neuron_responsibility, ' * ', connection_gain, ' * ', connection_weight, store_propagation);
                        }
                        else
                            buildSentence(responsibility, ' += ', neuron_responsibility, ' * ', connection_weight, store_propagation);
                    }
                    buildSentence(responsibility, ' *= ', derivative, store_propagation);
                    for (var id in this.connections.inputs) {
                        var input = this.connections.inputs[id];
                        var trace = getVar(this, 'trace', 'elegibility', input.ID, this
                            .trace.elegibility[input.ID]);
                        var input_weight = getVar(input, 'weight');
                        buildSentence(input_weight, ' += ', rate, ' * (', responsibility, ' * ', trace, ')', store_propagation);
                    }
                }
                else if (noProjections) {
                    buildSentence(responsibility, ' = 0', store_propagation);
                    for (var id in this.trace.extended) {
                        var neuron = this.neighboors[id];
                        var influence = getVar('aux');
                        var neuron_old = getVar(neuron, 'old');
                        if (neuron.selfconnection.gater == this)
                            buildSentence(influence, ' = ', neuron_old, store_propagation);
                        else
                            buildSentence(influence, ' = 0', store_propagation);
                        for (var influenceInput in this.trace.influences[neuron.ID]) {
                            var connection = this.trace.influences[neuron.ID][influenceInput];
                            var connection_weight = getVar(connection, 'weight');
                            var neuron_activation = getVar(connection.from, 'activation');
                            buildSentence(influence, ' += ', connection_weight, ' * ', neuron_activation, store_propagation);
                        }
                        var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                        buildSentence(responsibility, ' += ', neuron_responsibility, ' * ', influence, store_propagation);
                    }
                    buildSentence(responsibility, ' *= ', derivative, store_propagation);
                    for (var id in this.connections.inputs) {
                        var input = this.connections.inputs[id];
                        var gradient = getVar('aux');
                        buildSentence(gradient, ' = 0', store_propagation);
                        for (var id in this.trace.extended) {
                            var neuron = this.neighboors[id];
                            var neuron_responsibility = getVar(neuron, 'error', 'responsibility', neuron.error.responsibility);
                            var xtrace = getVar(this, 'trace', 'extended', neuron.ID, input.ID, this.trace.extended[neuron.ID][input.ID]);
                            buildSentence(gradient, ' += ', neuron_responsibility, ' * ', xtrace, store_propagation);
                        }
                        var input_weight = getVar(input, 'weight');
                        buildSentence(input_weight, ' += ', rate, ' * ', gradient, store_propagation);
                    }
                }
            }
            buildSentence(bias, ' += ', rate, ' * ', responsibility, store_propagation);
        }
        return {
            memory: varID,
            neurons: neurons + 1,
            inputs: inputs,
            outputs: outputs,
            targets: targets,
            variables: variables,
            activation_sentences: activation_sentences,
            trace_sentences: trace_sentences,
            propagation_sentences: propagation_sentences,
            layers: layers
        };
    };
    return Neuron;
})();
exports.Neuron = Neuron;
var Neuron;
(function (Neuron) {
    var Connection = (function () {
        function Connection(from, to, weight) {
            this.ID = Connection.uid();
            this.gain = 1;
            this.weight = 0;
            this.gater = null;
            if (!from || !to)
                throw "Connection Error: Invalid neurons";
            this.from = from;
            this.to = to;
            this.weight = typeof weight == 'undefined' || isNaN(weight) ? Math.random() * .2 - .1 :
                weight;
        }
        return Connection;
    })();
    Neuron.Connection = Connection;
    Neuron.neuronQty = 0;
    function uid() {
        return Neuron.neuronQty++;
    }
    Neuron.uid = uid;
    function quantity() {
        return {
            neurons: Neuron.neuronQty,
            connections: Connection.connectionQty
        };
    }
    Neuron.quantity = quantity;
})(Neuron = exports.Neuron || (exports.Neuron = {}));
var Neuron;
(function (Neuron) {
    var Connection;
    (function (Connection) {
        Connection.connectionQty = 0;
        function uid() {
            return Connection.connectionQty++;
        }
        Connection.uid = uid;
    })(Connection = Neuron.Connection || (Neuron.Connection = {}));
})(Neuron = exports.Neuron || (exports.Neuron = {}));

},{"./squash":9}],9:[function(require,module,exports){
// squashing functions
function LOGISTIC(x, derivate) {
    if (!derivate)
        return 1 / (1 + Math.exp(-x));
    var fx = LOGISTIC(x);
    return fx * (1 - fx);
}
exports.LOGISTIC = LOGISTIC;
function TANH(x, derivate) {
    if (derivate)
        return 1 - Math.pow(TANH(x), 2);
    var eP = Math.exp(x);
    var eN = 1 / eP;
    return (eP - eN) / (eP + eN);
}
exports.TANH = TANH;
function IDENTITY(x, derivate) {
    return derivate ? 1 : x;
}
exports.IDENTITY = IDENTITY;
function HLIM(x, derivate) {
    return derivate ? 1 : +(x > 0);
}
exports.HLIM = HLIM;

},{}],10:[function(require,module,exports){
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
var network = require('./network');
var layer = require('./layer');
var neuron = require('./neuron');
var trainer = require('./trainer');
var architect = require('./architect');
var squash = require('./squash');
var Synaptic;
(function (Synaptic) {
    var oldSynaptic = window && window['Synaptic'];
    function ninja() {
        window['synaptic'] = oldSynaptic;
        return Synaptic;
    }
    Synaptic.ninja = ninja;
    Synaptic.Neuron = neuron.Neuron;
    Synaptic.Layer = layer.Layer;
    Synaptic.Network = network.Network;
    Synaptic.Trainer = trainer.Trainer;
    Synaptic.Squash = squash;
    Synaptic.Architect = architect;
})(Synaptic || (Synaptic = {}));
if (typeof window != "undefined")
    window['synaptic'] = Synaptic;
module.exports = Synaptic;

},{"./architect":1,"./layer":6,"./network":7,"./neuron":8,"./squash":9,"./trainer":11}],11:[function(require,module,exports){
/*******************************************************************************************
                                        TRAINER
*******************************************************************************************/
var Trainer = (function () {
    function Trainer(network, options) {
        this.rate = .2;
        this.iterations = 100000;
        this.error = .005;
        options = options || {};
        this.network = network;
        this.rate = options.rate || .2;
        this.iterations = options.iterations || 100000;
        this.error = options.error || .005;
        this.cost = options.cost || Trainer.cost.CROSS_ENTROPY;
    }
    // trains any given set to a network
    Trainer.prototype.train = function (set, options) {
        var error = 1;
        var iterations = 0, bucketSize = 0;
        var abort_training = false;
        var input, output, target, currentRate;
        var start = Date.now();
        if (options) {
            if (options.shuffle) {
                //+ Jonas Raoni Soares Silva
                //@ http://jsfromhell.com/array/shuffle [v1.0]
                function shuffle(o) {
                    for (var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x)
                        ;
                    return o;
                }
                ;
            }
            if (options.iterations)
                this.iterations = options.iterations;
            if (options.error)
                this.error = options.error;
            if (options.rate)
                this.rate = options.rate;
            if (options.cost)
                this.cost = options.cost;
            if (options.schedule)
                this.schedule = options.schedule;
            if (options.customLog) {
                // for backward compatibility with code that used customLog
                console.log('Deprecated: use schedule instead of customLog');
                this.schedule = options.customLog;
            }
        }
        currentRate = this.rate;
        if (Array.isArray(this.rate)) {
            bucketSize = Math.floor(this.iterations / this.rate.length);
        }
        while (!abort_training && iterations < this.iterations && error > this.error) {
            error = 0;
            if (bucketSize > 0) {
                var currentBucket = Math.floor(iterations / bucketSize);
                currentRate = this.rate[currentBucket];
            }
            for (var train in set) {
                input = set[train].input;
                target = set[train].output;
                output = this.network.activate(input);
                this.network.propagate(currentRate, target);
                error += this.cost(target, output);
            }
            // check error
            iterations++;
            error /= set.length;
            if (options) {
                if (this.schedule && this.schedule.every && iterations %
                    this.schedule.every == 0)
                    abort_training = this.schedule.do({
                        error: error,
                        iterations: iterations,
                        rate: currentRate
                    });
                else if (options.log && iterations % options.log == 0) {
                    console.log('iterations', iterations, 'error', error, 'rate', currentRate);
                }
                ;
                if (options.shuffle)
                    shuffle(set);
            }
        }
        var results = {
            error: error,
            iterations: iterations,
            time: Date.now() - start
        };
        return results;
    };
    // trains any given set to a network using a WebWorker
    Trainer.prototype.workerTrain = function (set, callback, options) {
        var that = this;
        var error = 1;
        var iterations = 0, bucketSize = 0;
        var input, output, target, currentRate;
        var length = set.length;
        var abort_training = false;
        var start = Date.now();
        if (options) {
            if (options.shuffle) {
                //+ Jonas Raoni Soares Silva
                //@ http://jsfromhell.com/array/shuffle [v1.0]
                function shuffle(o) {
                    for (var j, x, i = o.length; i; j = Math.floor(Math.random() *
                        i), x = o[--i], o[i] = o[j], o[j] = x)
                        ;
                    return o;
                }
                ;
            }
            if (options.iterations)
                this.iterations = options.iterations;
            if (options.error)
                this.error = options.error;
            if (options.rate)
                this.rate = options.rate;
            if (options.cost)
                this.cost = options.cost;
            if (options.schedule)
                this.schedule = options.schedule;
            if (options.customLog)
                // for backward compatibility with code that used customLog
                console.log('Deprecated: use schedule instead of customLog');
            this.schedule = options.customLog;
        }
        // dynamic learning rate
        currentRate = this.rate;
        if (Array.isArray(this.rate)) {
            bucketSize = Math.floor(this.iterations / this.rate.length);
        }
        // create a worker
        var worker = this.network.worker();
        // activate the network
        function activateWorker(input) {
            worker.postMessage({
                action: "activate",
                input: input,
                memoryBuffer: that.network.optimized.memory
            }, [that.network.optimized.memory.buffer]);
        }
        // backpropagate the network
        function propagateWorker(target) {
            if (bucketSize > 0) {
                var currentBucket = Math.floor(iterations / bucketSize);
                currentRate = this.rate[currentBucket];
            }
            worker.postMessage({
                action: "propagate",
                target: target,
                rate: currentRate,
                memoryBuffer: that.network.optimized.memory
            }, [that.network.optimized.memory.buffer]);
        }
        // train the worker
        worker.onmessage = function (e) {
            // give control of the memory back to the network
            that.network.optimized.ownership(e.data.memoryBuffer);
            if (e.data.action == "propagate") {
                if (index >= length) {
                    index = 0;
                    iterations++;
                    error /= set.length;
                    // log
                    if (options) {
                        if (this.schedule && this.schedule.every && iterations % this.schedule.every == 0)
                            abort_training = this.schedule.do({
                                error: error,
                                iterations: iterations
                            });
                        else if (options.log && iterations % options.log == 0) {
                            console.log('iterations', iterations, 'error', error);
                        }
                        ;
                        if (options.shuffle)
                            shuffle(set);
                    }
                    if (!abort_training && iterations < that.iterations && error > that.error) {
                        activateWorker(set[index].input);
                    }
                    else {
                        // callback
                        callback({
                            error: error,
                            iterations: iterations,
                            time: Date.now() - start
                        });
                    }
                    error = 0;
                }
                else {
                    activateWorker(set[index].input);
                }
            }
            if (e.data.action == "activate") {
                error += that.cost(set[index].output, e.data.output);
                propagateWorker(set[index].output);
                index++;
            }
        };
        // kick it
        var index = 0;
        var iterations = 0;
        activateWorker(set[index].input);
    };
    // trains an XOR to the network
    Trainer.prototype.XOR = function (options) {
        if (this.network.inputs() != 2 || this.network.outputs() != 1)
            throw "Error: Incompatible network (2 inputs, 1 output)";
        var defaults = {
            iterations: 100000,
            log: false,
            shuffle: true,
            cost: Trainer.cost.MSE
        };
        if (options)
            for (var i in options)
                defaults[i] = options[i];
        return this.train([{
                input: [0, 0],
                output: [0]
            }, {
                input: [1, 0],
                output: [1]
            }, {
                input: [0, 1],
                output: [1]
            }, {
                input: [1, 1],
                output: [0]
            }], defaults);
    };
    // trains the network to pass a Distracted Sequence Recall test
    Trainer.prototype.DSR = function (options) {
        options = options || {};
        var targets = options.targets || [2, 4, 7, 8];
        var distractors = options.distractors || [3, 5, 6, 9];
        var prompts = options.prompts || [0, 1];
        var length = options.length || 24;
        var criterion = options.success || 0.95;
        var iterations = options.iterations || 100000;
        var rate = options.rate || .1;
        var log = options.log || 0;
        var schedule = options.schedule || {};
        var correct = 0;
        var i = 0;
        var success = 0;
        var trial = i = correct = j = success = 0, error = 1, symbols = targets.length + distractors.length + prompts.length;
        var noRepeat = function (range, avoid) {
            var number = Math.random() * range | 0;
            var used = false;
            for (var i in avoid)
                if (number == avoid[i])
                    used = true;
            return used ? noRepeat(range, avoid) : number;
        };
        var equal = function (prediction, output) {
            for (var i in prediction)
                if (Math.round(prediction[i]) != output[i])
                    return false;
            return true;
        };
        var start = Date.now();
        while (trial < iterations && (success < criterion || trial % 1000 != 0)) {
            // generate sequence
            var sequence = [], sequenceLength = length - prompts.length;
            for (i = 0; i < sequenceLength; i++) {
                var any = Math.random() * distractors.length | 0;
                sequence.push(distractors[any]);
            }
            var indexes = [], positions = [];
            for (i = 0; i < prompts.length; i++) {
                indexes.push(Math.random() * targets.length | 0);
                positions.push(noRepeat(sequenceLength, positions));
            }
            positions = positions.sort();
            for (i = 0; i < prompts.length; i++) {
                sequence[positions[i]] = targets[indexes[i]];
                sequence.push(prompts[i]);
            }
            //train sequence
            var distractorsCorrect;
            var targetsCorrect = distractorsCorrect = 0;
            error = 0;
            for (i = 0; i < length; i++) {
                // generate input from sequence
                var input = [];
                for (j = 0; j < symbols; j++)
                    input[j] = 0;
                input[sequence[i]] = 1;
                // generate target output
                var output = [];
                for (j = 0; j < targets.length; j++)
                    output[j] = 0;
                if (i >= sequenceLength) {
                    var index = i - sequenceLength;
                    output[indexes[index]] = 1;
                }
                // check result
                var prediction = this.network.activate(input);
                if (equal(prediction, output))
                    if (i < sequenceLength)
                        distractorsCorrect++;
                    else
                        targetsCorrect++;
                else {
                    this.network.propagate(rate, output);
                }
                var delta = 0;
                for (var j in prediction)
                    delta += Math.pow(output[j] - prediction[j], 2);
                error += delta / this.network.outputs();
                if (distractorsCorrect + targetsCorrect == length)
                    correct++;
            }
            // calculate error
            if (trial % 1000 == 0)
                correct = 0;
            trial++;
            var divideError = trial % 1000;
            divideError = divideError == 0 ? 1000 : divideError;
            success = correct / divideError;
            error /= length;
            // log
            if (log && trial % log == 0)
                console.log("iterations:", trial, " success:", success, " correct:", correct, " time:", Date.now() - start, " error:", error);
            if (schedule.do && schedule.every && trial % schedule.every == 0)
                schedule.do({
                    iterations: trial,
                    success: success,
                    error: error,
                    time: Date.now() - start,
                    correct: correct
                });
        }
        return {
            iterations: trial,
            success: success,
            error: error,
            time: Date.now() - start
        };
    };
    // train the network to learn an Embeded Reber Grammar
    Trainer.prototype.ERG = function (options) {
        options = options || {};
        var iterations = options.iterations || 150000;
        var criterion = options.error || .05;
        var rate = options.rate || .1;
        var log = options.log || 500;
        // gramar node
        var Node = function () {
            this.paths = [];
        };
        Node.prototype = {
            connect: function (node, value) {
                this.paths.push({
                    node: node,
                    value: value
                });
                return this;
            },
            any: function () {
                if (this.paths.length == 0)
                    return false;
                var index = Math.random() * this.paths.length | 0;
                return this.paths[index];
            },
            test: function (value) {
                for (var i in this.paths)
                    if (this.paths[i].value == value)
                        return this.paths[i];
                return false;
            }
        };
        var reberGrammar = function () {
            // build a reber grammar
            var output = new Node();
            var n1 = (new Node()).connect(output, "E");
            var n2 = (new Node()).connect(n1, "S");
            var n3 = (new Node()).connect(n1, "V").connect(n2, "P");
            var n4 = (new Node()).connect(n2, "X");
            n4.connect(n4, "S");
            var n5 = (new Node()).connect(n3, "V");
            n5.connect(n5, "T");
            n2.connect(n5, "X");
            var n6 = (new Node()).connect(n4, "T").connect(n5, "P");
            var input = (new Node()).connect(n6, "B");
            return {
                input: input,
                output: output
            };
        };
        // build an embeded reber grammar
        var embededReberGrammar = function () {
            var reber1 = reberGrammar();
            var reber2 = reberGrammar();
            var output = new Node();
            var n1 = (new Node).connect(output, "E");
            reber1.output.connect(n1, "T");
            reber2.output.connect(n1, "P");
            var n2 = (new Node).connect(reber1.input, "P").connect(reber2.input, "T");
            var input = (new Node).connect(n2, "B");
            return {
                input: input,
                output: output
            };
        };
        // generate an ERG sequence
        var generate = function () {
            var node = embededReberGrammar().input;
            var next = node.any();
            var str = "";
            while (next) {
                str += next.value;
                next = next.node.any();
            }
            return str;
        };
        // test if a string matches an embeded reber grammar
        var test = function (str) {
            var node = embededReberGrammar().input;
            var i = 0;
            var ch = str.charAt(i);
            while (i < str.length) {
                var next = node.test(ch);
                if (!next)
                    return false;
                node = next.node;
                ch = str.charAt(++i);
            }
            return true;
        };
        // helper to check if the output and the target vectors match
        var different = function (array1, array2) {
            var max1 = 0;
            var i1 = -1;
            var max2 = 0;
            var i2 = -1;
            for (var i in array1) {
                if (array1[i] > max1) {
                    max1 = array1[i];
                    i1 = i;
                }
                if (array2[i] > max2) {
                    max2 = array2[i];
                    i2 = i;
                }
            }
            return i1 != i2;
        };
        var iteration = 0;
        var error = 1;
        var table = {
            "B": 0,
            "P": 1,
            "T": 2,
            "X": 3,
            "S": 4,
            "E": 5
        };
        var start = Date.now();
        while (iteration < iterations && error > criterion) {
            var i = 0;
            error = 0;
            // ERG sequence to learn
            var sequence = generate();
            // input
            var read = sequence.charAt(i);
            // target
            var predict = sequence.charAt(i + 1);
            // train
            while (i < sequence.length - 1) {
                var input = [];
                var target = [];
                for (var j = 0; j < 6; j++) {
                    input[j] = 0;
                    target[j] = 0;
                }
                input[table[read]] = 1;
                target[table[predict]] = 1;
                var output = this.network.activate(input);
                if (different(output, target))
                    this.network.propagate(rate, target);
                read = sequence.charAt(++i);
                predict = sequence.charAt(i + 1);
                var delta = 0;
                for (var k in output)
                    delta += Math.pow(target[k] - output[k], 2);
                delta /= output.length;
                error += delta;
            }
            error /= sequence.length;
            iteration++;
            if (iteration % log == 0) {
                console.log("iterations:", iteration, " time:", Date.now() - start, " error:", error);
            }
        }
        return {
            iterations: iteration,
            error: error,
            time: Date.now() - start,
            test: test,
            generate: generate
        };
    };
    return Trainer;
})();
exports.Trainer = Trainer;
var Trainer;
(function (Trainer) {
    Trainer.cost = {
        // Eq. 9
        CROSS_ENTROPY: function (target, output) {
            var crossentropy = 0;
            for (var i in output)
                crossentropy -= (target[i] * Math.log(output[i] + 1e-15)) + ((1 - target[i]) * Math.log((1 + 1e-15) - output[i])); // +1e-15 is a tiny push away to avoid Math.log(0)
            return crossentropy;
        },
        MSE: function (target, output) {
            var mse = 0;
            for (var i in output)
                mse += Math.pow(target[i] - output[i], 2);
            return mse / output.length;
        }
    };
})(Trainer = exports.Trainer || (exports.Trainer = {}));

},{}]},{},[10])
//# sourceMappingURL=data:application/json;charset:utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyaWZ5L25vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJzcmMvYXJjaGl0ZWN0LnRzIiwic3JjL2FyY2hpdGVjdC9MU1RNLnRzIiwic3JjL2FyY2hpdGVjdC9MaXF1aWQudHMiLCJzcmMvYXJjaGl0ZWN0L1BlcmNlcHRyb24udHMiLCJzcmMvYXJjaGl0ZWN0L2hvcGZpZWxkLnRzIiwic3JjL2xheWVyLnRzIiwic3JjL25ldHdvcmsudHMiLCJzcmMvbmV1cm9uLnRzIiwic3JjL3NxdWFzaC50cyIsInNyYy9zeW5hcHRpYy50cyIsInNyYy90cmFpbmVyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBO0FDQUEsSUFBTyxRQUFRLFdBQVcsc0JBQXNCLENBQUMsQ0FBQztBQUNsRCxJQUFPLElBQUksV0FBVyxrQkFBa0IsQ0FBQyxDQUFDO0FBQzFDLElBQU8sR0FBRyxXQUFXLG9CQUFvQixDQUFDLENBQUM7QUFDM0MsSUFBTyxVQUFVLFdBQVcsd0JBQXdCLENBQUMsQ0FBQztBQUUzQyxZQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQztBQUNqQixjQUFNLEdBQUcsR0FBRyxDQUFDLE1BQU0sQ0FBQztBQUNwQixnQkFBUSxHQUFHLFFBQVEsQ0FBQyxRQUFRLENBQUM7QUFDN0Isa0JBQVUsR0FBRyxVQUFVLENBQUMsVUFBVSxDQUFDOzs7Ozs7Ozs7QUNSOUMsSUFBTyxPQUFPLFdBQVksWUFBWSxDQUFDLENBQUM7QUFDeEMsSUFBTyxPQUFPLFdBQVksWUFBWSxDQUFDLENBQUM7QUFDeEMsSUFBTyxLQUFLLFdBQVksVUFBVSxDQUFDLENBQUM7QUFHcEM7SUFBMEIsd0JBQWU7SUFHdkM7UUFBWSxjQUFjO2FBQWQsV0FBYyxDQUFkLHNCQUFjLENBQWQsSUFBYztZQUFkLDZCQUFjOztRQUV4QixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQztZQUNsQixNQUFNLHlDQUF5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QUFDLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7WUFDNUMsTUFBTSxDQUFDLEdBQUcsR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFDO1FBQzdCLENBQUM7S0FDRixDQUFBO0FBQ0gsQ0FBQyxFQXRCYSxPQUFPLEdBQVAsZUFBTyxLQUFQLGVBQU8sUUFzQnBCIiwiZmlsZSI6ImdlbmVyYXRlZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzQ29udGVudCI6WyIoZnVuY3Rpb24gZSh0LG4scil7ZnVuY3Rpb24gcyhvLHUpe2lmKCFuW29dKXtpZighdFtvXSl7dmFyIGE9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtpZighdSYmYSlyZXR1cm4gYShvLCEwKTtpZihpKXJldHVybiBpKG8sITApO3ZhciBmPW5ldyBFcnJvcihcIkNhbm5vdCBmaW5kIG1vZHVsZSAnXCIrbytcIidcIik7dGhyb3cgZi5jb2RlPVwiTU9EVUxFX05PVF9GT1VORFwiLGZ9dmFyIGw9bltvXT17ZXhwb3J0czp7fX07dFtvXVswXS5jYWxsKGwuZXhwb3J0cyxmdW5jdGlvbihlKXt2YXIgbj10W29dWzFdW2VdO3JldHVybiBzKG4/bjplKX0sbCxsLmV4cG9ydHMsZSx0LG4scil9cmV0dXJuIG5bb10uZXhwb3J0c312YXIgaT10eXBlb2YgcmVxdWlyZT09XCJmdW5jdGlvblwiJiZyZXF1aXJlO2Zvcih2YXIgbz0wO288ci5sZW5ndGg7bysrKXMocltvXSk7cmV0dXJuIHN9KSIsImltcG9ydCBob3BmaWVsZCA9IHJlcXVpcmUoJy4vYXJjaGl0ZWN0L2hvcGZpZWxkJyk7XG5pbXBvcnQgbHN0bSA9IHJlcXVpcmUoJy4vYXJjaGl0ZWN0L0xTVE0nKTtcbmltcG9ydCBsc20gPSByZXF1aXJlKCcuL2FyY2hpdGVjdC9MaXF1aWQnKTtcbmltcG9ydCBwZXJjZXB0cm9uID0gcmVxdWlyZSgnLi9hcmNoaXRlY3QvUGVyY2VwdHJvbicpO1xuXG5leHBvcnQgdmFyIExTVE0gPSBsc3RtLkxTVE07XG5leHBvcnQgdmFyIExpcXVpZCA9IGxzbS5MaXF1aWQ7XG5leHBvcnQgdmFyIEhvcGZpZWxkID0gaG9wZmllbGQuSG9wZmllbGQ7XG5leHBvcnQgdmFyIFBlcmNlcHRyb24gPSBwZXJjZXB0cm9uLlBlcmNlcHRyb247XG4iLCJpbXBvcnQgbmV0d29yayAgPSByZXF1aXJlKCcuLi9uZXR3b3JrJyk7XG5pbXBvcnQgdHJhaW5lciAgPSByZXF1aXJlKCcuLi90cmFpbmVyJyk7XG5pbXBvcnQgTGF5ZXIgID0gcmVxdWlyZSgnLi4vbGF5ZXInKTtcbmltcG9ydCBuZXVyb24gPSByZXF1aXJlKCcuLi9uZXVyb24nKTtcblxuZXhwb3J0IGNsYXNzIExTVE0gZXh0ZW5kcyBuZXR3b3JrLk5ldHdvcmsge1xuICB0cmFpbmVyOiB0cmFpbmVyLlRyYWluZXI7XG5cbiAgY29uc3RydWN0b3IoLi4uYXJnczogYW55W10pIHtcblxuICAgIGlmIChhcmdzLmxlbmd0aCA8IDMpXG4gICAgICB0aHJvdyBcIkVycm9yOiBub3QgZW5vdWdoIGxheWVycyAobWluaW11bSAzKSAhIVwiO1xuXG4gICAgdmFyIGxhc3QgPSBhcmdzLnBvcCgpO1xuICAgIHZhciBvcHRpb24gPSB7XG4gICAgICBwZWVwaG9sZXM6IExheWVyLkxheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19BTEwsXG4gICAgICBoaWRkZW50b2hpZGRlbjogZmFsc2UsXG4gICAgICBvdXR0b2hpZGRlbjogZmFsc2UsXG4gICAgICBvdXR0b2dhdGVzOiBmYWxzZSxcbiAgICAgIGludG9vdXQ6IHRydWUsXG4gICAgfTtcblxuICAgIGlmICh0eXBlb2YgbGFzdCAhPSAnbnVtYmVyJykge1xuICAgICAgdmFyIG91dHB1dHMgPSBhcmdzLnBvcCgpO1xuICAgICAgaWYgKGxhc3QuaGFzT3duUHJvcGVydHkoJ3BlZXBob2xlcycpKVxuICAgICAgICBvcHRpb24ucGVlcGhvbGVzID0gbGFzdC5wZWVwaG9sZXM7XG4gICAgICBpZiAobGFzdC5oYXNPd25Qcm9wZXJ0eSgnaGlkZGVudG9oaWRkZW4nKSlcbiAgICAgICAgb3B0aW9uLmhpZGRlbnRvaGlkZGVuID0gbGFzdC5oaWRkZW50b2hpZGRlbjtcbiAgICAgIGlmIChsYXN0Lmhhc093blByb3BlcnR5KCdvdXR0b2hpZGRlbicpKVxuICAgICAgICBvcHRpb24ub3V0dG9oaWRkZW4gPSBsYXN0Lm91dHRvaGlkZGVuO1xuICAgICAgaWYgKGxhc3QuaGFzT3duUHJvcGVydHkoJ291dHRvZ2F0ZXMnKSlcbiAgICAgICAgb3B0aW9uLm91dHRvZ2F0ZXMgPSBsYXN0Lm91dHRvZ2F0ZXM7XG4gICAgICBpZiAobGFzdC5oYXNPd25Qcm9wZXJ0eSgnaW50b291dCcpKVxuICAgICAgICBvcHRpb24uaW50b291dCA9IGxhc3QuaW50b291dDtcbiAgICB9IGVsc2VcbiAgICAgIHZhciBvdXRwdXRzID0gbGFzdDtcblxuICAgIHZhciBpbnB1dHMgPSBhcmdzLnNoaWZ0KCk7XG4gICAgdmFyIGxheWVycyA9IGFyZ3M7XG5cbiAgICB2YXIgaW5wdXRMYXllciA9IG5ldyBMYXllci5MYXllcihpbnB1dHMpO1xuICAgIHZhciBoaWRkZW5MYXllcnMgPSBbXTtcbiAgICB2YXIgb3V0cHV0TGF5ZXIgPSBuZXcgTGF5ZXIuTGF5ZXIob3V0cHV0cyk7XG5cbiAgICB2YXIgcHJldmlvdXMgPSBudWxsO1xuXG4gICAgLy8gZ2VuZXJhdGUgbGF5ZXJzXG4gICAgZm9yICh2YXIgbGF5ZXIgaW4gbGF5ZXJzKSB7XG4gICAgICAvLyBnZW5lcmF0ZSBtZW1vcnkgYmxvY2tzIChtZW1vcnkgY2VsbCBhbmQgcmVzcGVjdGl2ZSBnYXRlcylcbiAgICAgIHZhciBzaXplID0gbGF5ZXJzW2xheWVyXTtcblxuICAgICAgdmFyIGlucHV0R2F0ZSA9IG5ldyBMYXllci5MYXllcihzaXplKS5zZXQoe1xuICAgICAgICBiaWFzOiAxXG4gICAgICB9KTtcbiAgICAgIHZhciBmb3JnZXRHYXRlID0gbmV3IExheWVyLkxheWVyKHNpemUpLnNldCh7XG4gICAgICAgIGJpYXM6IDFcbiAgICAgIH0pO1xuICAgICAgdmFyIG1lbW9yeUNlbGwgPSBuZXcgTGF5ZXIuTGF5ZXIoc2l6ZSk7XG4gICAgICB2YXIgb3V0cHV0R2F0ZSA9IG5ldyBMYXllci5MYXllcihzaXplKS5zZXQoe1xuICAgICAgICBiaWFzOiAxXG4gICAgICB9KTtcblxuICAgICAgaGlkZGVuTGF5ZXJzLnB1c2goaW5wdXRHYXRlKTtcbiAgICAgIGhpZGRlbkxheWVycy5wdXNoKGZvcmdldEdhdGUpO1xuICAgICAgaGlkZGVuTGF5ZXJzLnB1c2gobWVtb3J5Q2VsbCk7XG4gICAgICBoaWRkZW5MYXllcnMucHVzaChvdXRwdXRHYXRlKTtcblxuICAgICAgLy8gY29ubmVjdGlvbnMgZnJvbSBpbnB1dCBsYXllclxuICAgICAgdmFyIGlucHV0ID0gaW5wdXRMYXllci5wcm9qZWN0KG1lbW9yeUNlbGwpO1xuICAgICAgaW5wdXRMYXllci5wcm9qZWN0KGlucHV0R2F0ZSk7XG4gICAgICBpbnB1dExheWVyLnByb2plY3QoZm9yZ2V0R2F0ZSk7XG4gICAgICBpbnB1dExheWVyLnByb2plY3Qob3V0cHV0R2F0ZSk7XG5cbiAgICAgIC8vIGNvbm5lY3Rpb25zIGZyb20gcHJldmlvdXMgbWVtb3J5LWJsb2NrIGxheWVyIHRvIHRoaXMgb25lXG4gICAgICBpZiAocHJldmlvdXMgIT0gbnVsbCkge1xuICAgICAgICB2YXIgY2VsbCA9IHByZXZpb3VzLnByb2plY3QobWVtb3J5Q2VsbCk7XG4gICAgICAgIHByZXZpb3VzLnByb2plY3QoaW5wdXRHYXRlKTtcbiAgICAgICAgcHJldmlvdXMucHJvamVjdChmb3JnZXRHYXRlKTtcbiAgICAgICAgcHJldmlvdXMucHJvamVjdChvdXRwdXRHYXRlKTtcbiAgICAgIH1cblxuICAgICAgLy8gY29ubmVjdGlvbnMgZnJvbSBtZW1vcnkgY2VsbFxuICAgICAgdmFyIG91dHB1dCA9IG1lbW9yeUNlbGwucHJvamVjdChvdXRwdXRMYXllcik7XG5cbiAgICAgIC8vIHNlbGYtY29ubmVjdGlvblxuICAgICAgdmFyIHNlbGYgPSBtZW1vcnlDZWxsLnByb2plY3QobWVtb3J5Q2VsbCk7XG5cbiAgICAgIC8vIGhpZGRlbiB0byBoaWRkZW4gcmVjdXJyZW50IGNvbm5lY3Rpb25cbiAgICAgIGlmIChvcHRpb24uaGlkZGVudG9oaWRkZW4pXG4gICAgICAgIG1lbW9yeUNlbGwucHJvamVjdChtZW1vcnlDZWxsLCBMYXllci5MYXllci5jb25uZWN0aW9uVHlwZS5BTExfVE9fRUxTRSk7XG5cbiAgICAgIC8vIG91dCB0byBoaWRkZW4gcmVjdXJyZW50IGNvbm5lY3Rpb25cbiAgICAgIGlmIChvcHRpb24ub3V0dG9oaWRkZW4pXG4gICAgICAgIG91dHB1dExheWVyLnByb2plY3QobWVtb3J5Q2VsbCk7XG5cbiAgICAgIC8vIG91dCB0byBnYXRlcyByZWN1cnJlbnQgY29ubmVjdGlvblxuICAgICAgaWYgKG9wdGlvbi5vdXR0b2dhdGVzKSB7XG4gICAgICAgIG91dHB1dExheWVyLnByb2plY3QoaW5wdXRHYXRlKTtcbiAgICAgICAgb3V0cHV0TGF5ZXIucHJvamVjdChvdXRwdXRHYXRlKTtcbiAgICAgICAgb3V0cHV0TGF5ZXIucHJvamVjdChmb3JnZXRHYXRlKTtcbiAgICAgIH1cbiAgICAgIFxuICAgICAgLy8gcGVlcGhvbGVzXG4gICAgICBtZW1vcnlDZWxsLnByb2plY3QoaW5wdXRHYXRlLCBvcHRpb24ucGVlcGhvbGVzKTtcbiAgICAgIG1lbW9yeUNlbGwucHJvamVjdChmb3JnZXRHYXRlLCBvcHRpb24ucGVlcGhvbGVzKTtcbiAgICAgIG1lbW9yeUNlbGwucHJvamVjdChvdXRwdXRHYXRlLCBvcHRpb24ucGVlcGhvbGVzKTtcblxuICAgICAgLy8gZ2F0ZXNcbiAgICAgIGlucHV0R2F0ZS5nYXRlKGlucHV0LCBMYXllci5MYXllci5nYXRlVHlwZS5JTlBVVCk7XG4gICAgICBmb3JnZXRHYXRlLmdhdGUoc2VsZiwgTGF5ZXIuTGF5ZXIuZ2F0ZVR5cGUuT05FX1RPX09ORSk7XG4gICAgICBvdXRwdXRHYXRlLmdhdGUob3V0cHV0LCBMYXllci5MYXllci5nYXRlVHlwZS5PVVRQVVQpO1xuICAgICAgaWYgKHByZXZpb3VzICE9IG51bGwpXG4gICAgICAgIGlucHV0R2F0ZS5nYXRlKGNlbGwsIExheWVyLkxheWVyLmdhdGVUeXBlLklOUFVUKTtcblxuICAgICAgcHJldmlvdXMgPSBtZW1vcnlDZWxsO1xuICAgIH1cblxuICAgIC8vIGlucHV0IHRvIG91dHB1dCBkaXJlY3QgY29ubmVjdGlvblxuICAgIGlmIChvcHRpb24uaW50b291dClcbiAgICAgIGlucHV0TGF5ZXIucHJvamVjdChvdXRwdXRMYXllcik7XG5cbiAgICAvLyBzZXQgdGhlIGxheWVycyBvZiB0aGUgbmV1cmFsIG5ldHdvcmtcbiAgICBzdXBlcih7XG4gICAgICBpbnB1dDogaW5wdXRMYXllcixcbiAgICAgIGhpZGRlbjogaGlkZGVuTGF5ZXJzLFxuICAgICAgb3V0cHV0OiBvdXRwdXRMYXllclxuICAgIH0pO1xuXG4gICAgLy8gdHJhaW5lclxuICAgIHRoaXMudHJhaW5lciA9IG5ldyB0cmFpbmVyLlRyYWluZXIodGhpcyk7XG4gIH1cbn07XG4iLCJpbXBvcnQgbmV0d29yayAgPSByZXF1aXJlKCcuLi9uZXR3b3JrJyk7XG5pbXBvcnQgdHJhaW5lciAgPSByZXF1aXJlKCcuLi90cmFpbmVyJyk7XG5pbXBvcnQgbGF5ZXIgID0gcmVxdWlyZSgnLi4vbGF5ZXInKTtcbmltcG9ydCBuZXVyb24gPSByZXF1aXJlKCcuLi9uZXVyb24nKTtcblxuZXhwb3J0IGNsYXNzIExpcXVpZCBleHRlbmRzIG5ldHdvcmsuTmV0d29yayB7XG4gIHRyYWluZXI6IHRyYWluZXIuVHJhaW5lcjtcblxuICBjb25zdHJ1Y3RvcihpbnB1dHMsIGhpZGRlbiwgb3V0cHV0cywgY29ubmVjdGlvbnMsIGdhdGVzKSB7XG5cbiAgICAvLyBjcmVhdGUgbGF5ZXJzXG4gICAgdmFyIGlucHV0TGF5ZXIgPSBuZXcgbGF5ZXIuTGF5ZXIoaW5wdXRzKTtcbiAgICB2YXIgaGlkZGVuTGF5ZXIgPSBuZXcgbGF5ZXIuTGF5ZXIoaGlkZGVuKTtcbiAgICB2YXIgb3V0cHV0TGF5ZXIgPSBuZXcgbGF5ZXIuTGF5ZXIob3V0cHV0cyk7XG5cbiAgICAvLyBtYWtlIGNvbm5lY3Rpb25zIGFuZCBnYXRlcyByYW5kb21seSBhbW9uZyB0aGUgbmV1cm9uc1xuICAgIHZhciBuZXVyb25zID0gaGlkZGVuTGF5ZXIubmV1cm9ucygpO1xuICAgIHZhciBjb25uZWN0aW9uTGlzdDogbmV1cm9uLk5ldXJvbi5Db25uZWN0aW9uW10gPSBbXTtcblxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgY29ubmVjdGlvbnM7IGkrKykge1xuICAgICAgLy8gY29ubmVjdCB0d28gcmFuZG9tIG5ldXJvbnNcbiAgICAgIHZhciBmcm9tID0gTWF0aC5yYW5kb20oKSAqIG5ldXJvbnMubGVuZ3RoIHwgMDtcbiAgICAgIHZhciB0byA9IE1hdGgucmFuZG9tKCkgKiBuZXVyb25zLmxlbmd0aCB8IDA7XG4gICAgICB2YXIgY29ubmVjdGlvbiA9IG5ldXJvbnNbZnJvbV0ucHJvamVjdChuZXVyb25zW3RvXSk7XG4gICAgICBjb25uZWN0aW9uTGlzdC5wdXNoKGNvbm5lY3Rpb24pO1xuICAgIH1cblxuICAgIGZvciAodmFyIGogPSAwOyBqIDwgZ2F0ZXM7IGorKykge1xuICAgICAgLy8gcGljayBhIHJhbmRvbSBnYXRlciBuZXVyb25cbiAgICAgIHZhciBnYXRlciA9IE1hdGgucmFuZG9tKCkgKiBuZXVyb25zLmxlbmd0aCB8IDA7XG4gICAgICAvLyBwaWNrIGEgcmFuZG9tIGNvbm5lY3Rpb24gdG8gZ2F0ZVxuICAgICAgdmFyIGNvbm5lY3Rpb25OdW1iZXIgPSBNYXRoLnJhbmRvbSgpICogY29ubmVjdGlvbkxpc3QubGVuZ3RoIHwgMDtcbiAgICAgIC8vIGxldCB0aGUgZ2F0ZXIgZ2F0ZSB0aGUgY29ubmVjdGlvblxuICAgICAgbmV1cm9uc1tnYXRlcl0uZ2F0ZShjb25uZWN0aW9uTGlzdFtjb25uZWN0aW9uTnVtYmVyXSk7XG4gICAgfVxuXG4gICAgLy8gY29ubmVjdCB0aGUgbGF5ZXJzXG4gICAgaW5wdXRMYXllci5wcm9qZWN0KGhpZGRlbkxheWVyKTtcbiAgICBoaWRkZW5MYXllci5wcm9qZWN0KG91dHB1dExheWVyKTtcblxuICAgIC8vIHNldCB0aGUgbGF5ZXJzIG9mIHRoZSBuZXR3b3JrXG4gICAgc3VwZXIoe1xuICAgICAgaW5wdXQ6IGlucHV0TGF5ZXIsXG4gICAgICBoaWRkZW46IFtoaWRkZW5MYXllcl0sXG4gICAgICBvdXRwdXQ6IG91dHB1dExheWVyXG4gICAgfSk7XG5cbiAgICAvLyB0cmFpbmVyXG4gICAgdGhpcy50cmFpbmVyID0gbmV3IHRyYWluZXIuVHJhaW5lcih0aGlzKTtcbiAgfVxufVxuIiwiaW1wb3J0IG5ldHdvcmsgID0gcmVxdWlyZSgnLi4vbmV0d29yaycpO1xuaW1wb3J0IHRyYWluZXIgID0gcmVxdWlyZSgnLi4vdHJhaW5lcicpO1xuaW1wb3J0IGxheWVyICA9IHJlcXVpcmUoJy4uL2xheWVyJyk7XG5pbXBvcnQgbmV1cm9uID0gcmVxdWlyZSgnLi4vbmV1cm9uJyk7XG4vLyBNdWx0aWxheWVyIFBlcmNlcHRyb25cbmV4cG9ydCBjbGFzcyBQZXJjZXB0cm9uIGV4dGVuZHMgbmV0d29yay5OZXR3b3JrIHtcbiAgdHJhaW5lcjogdHJhaW5lci5UcmFpbmVyO1xuXG4gIGNvbnN0cnVjdG9yKC4uLmFyZ3M6IG51bWJlcltdKSB7XG5cbiAgICBpZiAoYXJncy5sZW5ndGggPCAzKVxuICAgICAgdGhyb3cgXCJFcnJvcjogbm90IGVub3VnaCBsYXllcnMgKG1pbmltdW0gMykgISFcIjtcblxuICAgIHZhciBpbnB1dHMgPSBhcmdzLnNoaWZ0KCk7IC8vIGZpcnN0IGFyZ3VtZW50XG4gICAgdmFyIG91dHB1dHMgPSBhcmdzLnBvcCgpOyAvLyBsYXN0IGFyZ3VtZW50XG4gICAgdmFyIGxheWVycyA9IGFyZ3M7IC8vIGFsbCB0aGUgYXJndW1lbnRzIGluIHRoZSBtaWRkbGVcbiAgXG4gICAgdmFyIGlucHV0ID0gbmV3IGxheWVyLkxheWVyKGlucHV0cyk7XG4gICAgdmFyIGhpZGRlbiA9IFtdO1xuICAgIHZhciBvdXRwdXQgPSBuZXcgbGF5ZXIuTGF5ZXIob3V0cHV0cyk7XG5cbiAgICB2YXIgcHJldmlvdXMgPSBpbnB1dDtcbiAgXG4gICAgLy8gZ2VuZXJhdGUgaGlkZGVuIGxheWVyc1xuICAgIGZvciAodmFyIGxldmVsIGluIGxheWVycykge1xuICAgICAgdmFyIHNpemUgPSBsYXllcnNbbGV2ZWxdO1xuICAgICAgdmFyIHRoZUxheWVyID0gbmV3IGxheWVyLkxheWVyKHNpemUpO1xuICAgICAgaGlkZGVuLnB1c2godGhlTGF5ZXIpO1xuICAgICAgcHJldmlvdXMucHJvamVjdCh0aGVMYXllcik7XG4gICAgICBwcmV2aW91cyA9IHRoZUxheWVyO1xuICAgIH1cbiAgICBwcmV2aW91cy5wcm9qZWN0KG91dHB1dCk7XG4gIFxuICAgIC8vIHNldCBsYXllcnMgb2YgdGhlIG5ldXJhbCBuZXR3b3JrXG4gICAgICBcbiAgICBzdXBlcih7XG4gICAgICBpbnB1dDogaW5wdXQsXG4gICAgICBoaWRkZW46IGhpZGRlbixcbiAgICAgIG91dHB1dDogb3V0cHV0XG4gICAgfSk7XG4gIFxuICAgIC8vIHRyYWluZXIgZm9yIHRoZSBuZXR3b3JrXG4gICAgdGhpcy50cmFpbmVyID0gbmV3IHRyYWluZXIuVHJhaW5lcih0aGlzKTtcbiAgfVxufTsgIiwiaW1wb3J0IG5ldHdvcmsgID0gcmVxdWlyZSgnLi4vbmV0d29yaycpO1xuaW1wb3J0IHRyYWluZXIgID0gcmVxdWlyZSgnLi4vdHJhaW5lcicpO1xuaW1wb3J0IGxheWVyICA9IHJlcXVpcmUoJy4uL2xheWVyJyk7XG5pbXBvcnQgbmV1cm9uID0gcmVxdWlyZSgnLi4vbmV1cm9uJyk7XG5cbmV4cG9ydCBjbGFzcyBIb3BmaWVsZCBleHRlbmRzIG5ldHdvcmsuTmV0d29yayB7XG4gIHRyYWluZXI6IHRyYWluZXIuVHJhaW5lcjtcblxuICBjb25zdHJ1Y3RvcihzaXplOiBudW1iZXIpIHtcbiAgICB2YXIgaW5wdXRMYXllciA9IG5ldyBsYXllci5MYXllcihzaXplKTtcbiAgICB2YXIgb3V0cHV0TGF5ZXIgPSBuZXcgbGF5ZXIuTGF5ZXIoc2l6ZSk7XG5cbiAgICBpbnB1dExheWVyLnByb2plY3Qob3V0cHV0TGF5ZXIsIGxheWVyLkxheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19BTEwpO1xuXG4gICAgc3VwZXIoe1xuICAgICAgaW5wdXQ6IGlucHV0TGF5ZXIsXG4gICAgICBoaWRkZW46IFtdLFxuICAgICAgb3V0cHV0OiBvdXRwdXRMYXllclxuICAgIH0pO1xuXG4gICAgdGhpcy50cmFpbmVyID0gbmV3IHRyYWluZXIuVHJhaW5lcih0aGlzKTtcbiAgfVxuXG4gIGxlYXJuKHBhdHRlcm5zKSB7XG4gICAgdmFyIHNldCA9IFtdO1xuICAgIGZvciAodmFyIHAgaW4gcGF0dGVybnMpXG4gICAgICBzZXQucHVzaCh7XG4gICAgICAgIGlucHV0OiBwYXR0ZXJuc1twXSxcbiAgICAgICAgb3V0cHV0OiBwYXR0ZXJuc1twXVxuICAgICAgfSk7XG5cbiAgICByZXR1cm4gdGhpcy50cmFpbmVyLnRyYWluKHNldCwge1xuICAgICAgaXRlcmF0aW9uczogNTAwMDAwLFxuICAgICAgZXJyb3I6IC4wMDAwNSxcbiAgICAgIHJhdGU6IDFcbiAgICB9KTtcbiAgfVxuXG4gIGZlZWQocGF0dGVybikge1xuICAgIHZhciBvdXRwdXQgPSB0aGlzLmFjdGl2YXRlKHBhdHRlcm4pO1xuXG4gICAgdmFyIHBhdHRlcm5zID0gW107XG4gICAgZm9yICh2YXIgaSBpbiBvdXRwdXQpXG4gICAgICBwYXR0ZXJuc1tpXSA9IG91dHB1dFtpXSA+IC41ID8gMSA6IDA7XG5cbiAgICByZXR1cm4gcGF0dGVybnM7XG4gIH1cbn0iLCJpbXBvcnQgbmV1cm9uID0gcmVxdWlyZSgnLi9uZXVyb24nKTtcbmltcG9ydCBuZXR3b3JrID0gcmVxdWlyZSgnLi9uZXR3b3JrJyk7XG5pbXBvcnQgU3luYXB0aWMgPSByZXF1aXJlKCcuL3N5bmFwdGljJyk7XG5cbi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIExBWUVSXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuZXhwb3J0IGNsYXNzIExheWVyIHtcblx0XHRsaXN0OiBuZXVyb24uTmV1cm9uW10gPSBbXTtcblx0XHRsYWJlbDogc3RyaW5nID0gbnVsbDtcblx0XHRjb25uZWN0ZWR0byA9IFtdO1xuXHRcdHNpemUgPSAwO1xuXG5cdFx0Y29uc3RydWN0b3Ioc2l6ZTogbnVtYmVyLCBsYWJlbD86IHN0cmluZykge1xuXHRcdFx0dGhpcy5zaXplID0gc2l6ZSB8IDA7XG5cdFx0XHR0aGlzLmxpc3QgPSBbXTtcblx0XHRcdHRoaXMubGFiZWwgPSBsYWJlbCB8fCBudWxsO1xuXHRcdFx0dGhpcy5jb25uZWN0ZWR0byA9IFtdO1xuXG5cdFx0XHR3aGlsZSAoc2l6ZS0tKSB7XG5cdFx0XHRcdHZhciB0aGVOZXVyb24gPSBuZXcgbmV1cm9uLk5ldXJvbigpO1xuXHRcdFx0XHR0aGlzLmxpc3QucHVzaCh0aGVOZXVyb24pO1xuXHRcdFx0fVxuXHRcdH1cblxuXHRcblx0XHQvLyBhY3RpdmF0ZXMgYWxsIHRoZSBuZXVyb25zIGluIHRoZSBsYXllclxuXHRcdGFjdGl2YXRlKGlucHV0KSB7XG5cblx0XHRcdHZhciBhY3RpdmF0aW9ucyA9IFtdO1xuXG5cdFx0XHRpZiAodHlwZW9mIGlucHV0ICE9ICd1bmRlZmluZWQnKSB7XG5cdFx0XHRcdGlmIChpbnB1dC5sZW5ndGggIT0gdGhpcy5zaXplKVxuXHRcdFx0XHRcdHRocm93IFwiSU5QVVQgc2l6ZSBhbmQgTEFZRVIgc2l6ZSBtdXN0IGJlIHRoZSBzYW1lIHRvIGFjdGl2YXRlIVwiO1xuXG5cdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMubGlzdCkge1xuXHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaWRdO1xuXHRcdFx0XHRcdHZhciBhY3RpdmF0aW9uID0gbmV1cm9uLmFjdGl2YXRlKGlucHV0W2lkXSk7XG5cdFx0XHRcdFx0YWN0aXZhdGlvbnMucHVzaChhY3RpdmF0aW9uKTtcblx0XHRcdFx0fVxuXHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5saXN0KSB7XG5cdFx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubGlzdFtpZF07XG5cdFx0XHRcdFx0dmFyIGFjdGl2YXRpb24gPSBuZXVyb24uYWN0aXZhdGUoKTtcblx0XHRcdFx0XHRhY3RpdmF0aW9ucy5wdXNoKGFjdGl2YXRpb24pO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gYWN0aXZhdGlvbnM7XG5cdFx0fVxuXG5cdFx0Ly8gcHJvcGFnYXRlcyB0aGUgZXJyb3Igb24gYWxsIHRoZSBuZXVyb25zIG9mIHRoZSBsYXllclxuXHRcdHByb3BhZ2F0ZShyYXRlLCB0YXJnZXQpIHtcblxuXHRcdFx0aWYgKHR5cGVvZiB0YXJnZXQgIT0gJ3VuZGVmaW5lZCcpIHtcblx0XHRcdFx0aWYgKHRhcmdldC5sZW5ndGggIT0gdGhpcy5zaXplKVxuXHRcdFx0XHRcdHRocm93IFwiVEFSR0VUIHNpemUgYW5kIExBWUVSIHNpemUgbXVzdCBiZSB0aGUgc2FtZSB0byBwcm9wYWdhdGUhXCI7XG5cblx0XHRcdFx0Zm9yICh2YXIgaWQgPSB0aGlzLmxpc3QubGVuZ3RoIC0gMTsgaWQgPj0gMDsgaWQtLSkge1xuXHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaWRdO1xuXHRcdFx0XHRcdG5ldXJvbi5wcm9wYWdhdGUocmF0ZSwgdGFyZ2V0W2lkXSk7XG5cdFx0XHRcdH1cblx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdGZvciAodmFyIGlkID0gdGhpcy5saXN0Lmxlbmd0aCAtIDE7IGlkID49IDA7IGlkLS0pIHtcblx0XHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5saXN0W2lkXTtcblx0XHRcdFx0XHRuZXVyb24ucHJvcGFnYXRlKHJhdGUpO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFx0Ly8gcHJvamVjdHMgYSBjb25uZWN0aW9uIGZyb20gdGhpcyBsYXllciB0byBhbm90aGVyIG9uZVxuXHRcdHByb2plY3QobGF5ZXIsIHR5cGU/LCB3ZWlnaHRzPykge1xuXG5cdFx0XHRpZiAobGF5ZXIgaW5zdGFuY2VvZiBuZXR3b3JrLk5ldHdvcmspXG5cdFx0XHRcdGxheWVyID0gbGF5ZXIubGF5ZXJzLmlucHV0O1xuXG5cdFx0XHRpZiAobGF5ZXIgaW5zdGFuY2VvZiBMYXllcikge1xuXHRcdFx0XHRpZiAoIXRoaXMuY29ubmVjdGVkKGxheWVyKSlcblx0XHRcdFx0XHRyZXR1cm4gbmV3IExheWVyLkxheWVyQ29ubmVjdGlvbih0aGlzLCBsYXllciwgdHlwZSwgd2VpZ2h0cyk7XG5cdFx0XHR9IGVsc2Vcblx0XHRcdFx0dGhyb3cgXCJJbnZhbGlkIGFyZ3VtZW50LCB5b3UgY2FuIG9ubHkgcHJvamVjdCBjb25uZWN0aW9ucyB0byBMQVlFUlMgYW5kIE5FVFdPUktTIVwiO1xuXG5cblx0XHR9XG5cblx0XHQvLyBnYXRlcyBhIGNvbm5lY3Rpb24gYmV0d2VubiB0d28gbGF5ZXJzXG5cdFx0Z2F0ZShjb25uZWN0aW9uLCB0eXBlKSB7XG5cblx0XHRcdGlmICh0eXBlID09IExheWVyLmdhdGVUeXBlLklOUFVUKSB7XG5cdFx0XHRcdGlmIChjb25uZWN0aW9uLnRvLnNpemUgIT0gdGhpcy5zaXplKVxuXHRcdFx0XHRcdHRocm93IFwiR0FURVIgbGF5ZXIgYW5kIENPTk5FQ1RJT04uVE8gbGF5ZXIgbXVzdCBiZSB0aGUgc2FtZSBzaXplIGluIG9yZGVyIHRvIGdhdGUhXCI7XG5cblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gY29ubmVjdGlvbi50by5saXN0KSB7XG5cdFx0XHRcdFx0dmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24udG8ubGlzdFtpZF07XG5cdFx0XHRcdFx0dmFyIGdhdGVyID0gdGhpcy5saXN0W2lkXTtcblx0XHRcdFx0XHRmb3IgKHZhciBpbnB1dCBpbiBuZXVyb24uY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHRcdFx0XHR2YXIgZ2F0ZWQgPSBuZXVyb24uY29ubmVjdGlvbnMuaW5wdXRzW2lucHV0XTtcblx0XHRcdFx0XHRcdGlmIChnYXRlZC5JRCBpbiBjb25uZWN0aW9uLmNvbm5lY3Rpb25zKVxuXHRcdFx0XHRcdFx0XHRnYXRlci5nYXRlKGdhdGVkKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdH0gZWxzZSBpZiAodHlwZSA9PSBMYXllci5nYXRlVHlwZS5PVVRQVVQpIHtcblx0XHRcdFx0aWYgKGNvbm5lY3Rpb24uZnJvbS5zaXplICE9IHRoaXMuc2l6ZSlcblx0XHRcdFx0XHR0aHJvdyBcIkdBVEVSIGxheWVyIGFuZCBDT05ORUNUSU9OLkZST00gbGF5ZXIgbXVzdCBiZSB0aGUgc2FtZSBzaXplIGluIG9yZGVyIHRvIGdhdGUhXCI7XG5cblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gY29ubmVjdGlvbi5mcm9tLmxpc3QpIHtcblx0XHRcdFx0XHR2YXIgbmV1cm9uID0gY29ubmVjdGlvbi5mcm9tLmxpc3RbaWRdO1xuXHRcdFx0XHRcdHZhciBnYXRlciA9IHRoaXMubGlzdFtpZF07XG5cdFx0XHRcdFx0Zm9yICh2YXIgcHJvamVjdGVkIGluIG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpIHtcblx0XHRcdFx0XHRcdHZhciBnYXRlZCA9IG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbcHJvamVjdGVkXTtcblx0XHRcdFx0XHRcdGlmIChnYXRlZC5JRCBpbiBjb25uZWN0aW9uLmNvbm5lY3Rpb25zKVxuXHRcdFx0XHRcdFx0XHRnYXRlci5nYXRlKGdhdGVkKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdH0gZWxzZSBpZiAodHlwZSA9PSBMYXllci5nYXRlVHlwZS5PTkVfVE9fT05FKSB7XG5cdFx0XHRcdGlmIChjb25uZWN0aW9uLnNpemUgIT0gdGhpcy5zaXplKVxuXHRcdFx0XHRcdHRocm93IFwiVGhlIG51bWJlciBvZiBHQVRFUiBVTklUUyBtdXN0IGJlIHRoZSBzYW1lIGFzIHRoZSBudW1iZXIgb2YgQ09OTkVDVElPTlMgdG8gZ2F0ZSFcIjtcblxuXHRcdFx0XHRmb3IgKHZhciBpZCBpbiBjb25uZWN0aW9uLmxpc3QpIHtcblx0XHRcdFx0XHR2YXIgZ2F0ZXIgPSB0aGlzLmxpc3RbaWRdO1xuXHRcdFx0XHRcdHZhciBnYXRlZCA9IGNvbm5lY3Rpb24ubGlzdFtpZF07XG5cdFx0XHRcdFx0Z2F0ZXIuZ2F0ZShnYXRlZCk7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHRcdGNvbm5lY3Rpb24uZ2F0ZWRmcm9tLnB1c2goeyBsYXllcjogdGhpcywgdHlwZTogdHlwZSB9KTtcblx0XHR9XG5cblx0XHQvLyB0cnVlIG9yIGZhbHNlIHdoZXRoZXIgdGhlIHdob2xlIGxheWVyIGlzIHNlbGYtY29ubmVjdGVkIG9yIG5vdFxuXHRcdHNlbGZjb25uZWN0ZWQoKTogYm9vbGVhbiB7XG5cblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMubGlzdCkge1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5saXN0W2lkXTtcblx0XHRcdFx0aWYgKCFuZXVyb24uc2VsZmNvbm5lY3RlZCgpKVxuXHRcdFx0XHRcdHJldHVybiBmYWxzZTtcblx0XHRcdH1cblx0XHRcdHJldHVybiB0cnVlO1xuXHRcdH1cblxuXHRcdC8vIHRydWUgb2YgZmFsc2Ugd2hldGhlciB0aGUgbGF5ZXIgaXMgY29ubmVjdGVkIHRvIGFub3RoZXIgbGF5ZXIgKHBhcmFtZXRlcikgb3Igbm90XG5cdFx0Y29ubmVjdGVkKGxheWVyKSB7XG5cdFx0XHQvLyBDaGVjayBpZiBBTEwgdG8gQUxMIGNvbm5lY3Rpb25cblx0XHRcdHZhciBjb25uZWN0aW9ucyA9IDA7XG5cdFx0XHRmb3IgKHZhciBoZXJlIGluIHRoaXMubGlzdCkge1xuXHRcdFx0XHRmb3IgKHZhciB0aGVyZSBpbiBsYXllci5saXN0KSB7XG5cdFx0XHRcdFx0dmFyIGZyb20gPSB0aGlzLmxpc3RbaGVyZV07XG5cdFx0XHRcdFx0dmFyIHRvID0gbGF5ZXIubGlzdFt0aGVyZV07XG5cdFx0XHRcdFx0dmFyIGNvbm5lY3RlZCA9IGZyb20uY29ubmVjdGVkKHRvKTtcblx0XHRcdFx0XHRpZiAoY29ubmVjdGVkICYmIGNvbm5lY3RlZC50eXBlID09ICdwcm9qZWN0ZWQnKVxuXHRcdFx0XHRcdFx0Y29ubmVjdGlvbnMrKztcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdFx0aWYgKGNvbm5lY3Rpb25zID09IHRoaXMuc2l6ZSAqIGxheWVyLnNpemUpXG5cdFx0XHRcdHJldHVybiBMYXllci5jb25uZWN0aW9uVHlwZS5BTExfVE9fQUxMO1xuXG5cdFx0XHQvLyBDaGVjayBpZiBPTkUgdG8gT05FIGNvbm5lY3Rpb25cblx0XHRcdGNvbm5lY3Rpb25zID0gMDtcblx0XHRcdGZvciAodmFyIG5ldXJvbiBpbiB0aGlzLmxpc3QpIHtcblx0XHRcdFx0dmFyIGZyb20gPSB0aGlzLmxpc3RbbmV1cm9uXTtcblx0XHRcdFx0dmFyIHRvID0gbGF5ZXIubGlzdFtuZXVyb25dO1xuXHRcdFx0XHR2YXIgY29ubmVjdGVkID0gZnJvbS5jb25uZWN0ZWQodG8pO1xuXHRcdFx0XHRpZiAoY29ubmVjdGVkICYmIGNvbm5lY3RlZC50eXBlID09ICdwcm9qZWN0ZWQnKVxuXHRcdFx0XHRcdGNvbm5lY3Rpb25zKys7XG5cdFx0XHR9XG5cdFx0XHRpZiAoY29ubmVjdGlvbnMgPT0gdGhpcy5zaXplKVxuXHRcdFx0XHRyZXR1cm4gTGF5ZXIuY29ubmVjdGlvblR5cGUuT05FX1RPX09ORTtcblx0XHR9XG5cblx0XHQvLyBjbGVhcnMgYWxsIHRoZSBuZXVvcm5zIGluIHRoZSBsYXllclxuXHRcdGNsZWFyKCkge1xuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5saXN0KSB7XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaWRdO1xuXHRcdFx0XHRuZXVyb24uY2xlYXIoKTtcblx0XHRcdH1cblx0XHR9XG5cblx0XHQvLyByZXNldHMgYWxsIHRoZSBuZXVyb25zIGluIHRoZSBsYXllclxuXHRcdHJlc2V0KCkge1xuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5saXN0KSB7XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaWRdO1xuXHRcdFx0XHRuZXVyb24ucmVzZXQoKTtcblx0XHRcdH1cblx0XHR9XG5cblx0XHQvLyByZXR1cm5zIGFsbCB0aGUgbmV1cm9ucyBpbiB0aGUgbGF5ZXIgKGFycmF5KVxuXHRcdG5ldXJvbnMoKSA6IG5ldXJvbi5OZXVyb25bXSB7XG5cdFx0XHRyZXR1cm4gdGhpcy5saXN0O1xuXHRcdH1cblxuXHRcdC8vIGFkZHMgYSBuZXVyb24gdG8gdGhlIGxheWVyXG5cdFx0YWRkKG5ldXJvbikge1xuXHRcdFx0bmV1cm9uID0gbmV1cm9uIHx8IG5ldyBuZXVyb24uTmV1cm9uKCk7XG5cdFx0XHR0aGlzLm5ldXJvbnNbbmV1cm9uLklEXSA9IG5ldXJvbjtcblx0XHRcdHRoaXMubGlzdC5wdXNoKG5ldXJvbik7XG5cdFx0XHR0aGlzLnNpemUrKztcblx0XHR9XG5cblx0XHRzZXQob3B0aW9ucykge1xuXHRcdFx0b3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG5cblx0XHRcdGZvciAodmFyIGkgaW4gdGhpcy5saXN0KSB7XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaV07XG5cdFx0XHRcdGlmIChvcHRpb25zLmxhYmVsKVxuXHRcdFx0XHRcdG5ldXJvbi5sYWJlbCA9IG9wdGlvbnMubGFiZWwgKyAnXycgKyBuZXVyb24uSUQ7XG5cdFx0XHRcdGlmIChvcHRpb25zLnNxdWFzaClcblx0XHRcdFx0XHRuZXVyb24uc3F1YXNoID0gb3B0aW9ucy5zcXVhc2g7XG5cdFx0XHRcdGlmIChvcHRpb25zLmJpYXMpXG5cdFx0XHRcdFx0bmV1cm9uLmJpYXMgPSBvcHRpb25zLmJpYXM7XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gdGhpcztcblx0XHR9XG5cdH1cblxuXG5leHBvcnQgbW9kdWxlIExheWVyIHtcblx0ZXhwb3J0IHZhciBsYXllclF0eSA9IDA7XG5cdGV4cG9ydCBmdW5jdGlvbiB1aWQoKSB7XG5cdFx0cmV0dXJuIGxheWVyUXR5Kys7XG5cdH1cblx0XG5cdC8vIHR5cGVzIG9mIGNvbm5lY3Rpb25zXG5cdGV4cG9ydCB2YXIgY29ubmVjdGlvblR5cGUgPSB7XG5cdFx0QUxMX1RPX0FMTDogXCJBTEwgVE8gQUxMXCIsXG5cdFx0T05FX1RPX09ORTogXCJPTkUgVE8gT05FXCIsXG5cdFx0QUxMX1RPX0VMU0U6IFwiQUxMIFRPIEVMU0VcIlxuXHR9O1xuXG5cdC8vIHR5cGVzIG9mIGdhdGVzXG5cdGV4cG9ydCB2YXIgZ2F0ZVR5cGUgPSB7XG5cdFx0SU5QVVQ6IFwiSU5QVVRcIixcblx0XHRPVVRQVVQ6IFwiT1VUUFVUXCIsXG5cdFx0T05FX1RPX09ORTogXCJPTkUgVE8gT05FXCJcblx0fTtcblxuXHQvLyByZXByZXNlbnRzIGEgY29ubmVjdGlvbiBmcm9tIG9uZSBsYXllciB0byBhbm90aGVyLCBhbmQga2VlcHMgdHJhY2sgb2YgaXRzIHdlaWdodCBhbmQgZ2FpblxuXHRleHBvcnQgY2xhc3MgTGF5ZXJDb25uZWN0aW9uIHtcblx0XHRJRCA9IHVpZCgpO1xuXHRcdGZyb206IExheWVyO1xuXHRcdHRvOiBMYXllcjtcblx0XHRzZWxmY29ubmVjdGlvbiA6IGJvb2xlYW4gPSBmYWxzZTtcblx0XHR0eXBlOiBzdHJpbmc7XG5cdFx0Y29ubmVjdGlvbnM6IFN5bmFwdGljLkRpY3Rpb25hcnk8bmV1cm9uLk5ldXJvbi5Db25uZWN0aW9uPjtcblx0XHRsaXN0OiBuZXVyb24uTmV1cm9uLkNvbm5lY3Rpb25bXTtcblx0XHRzaXplID0gMDtcblx0XHRnYXRlZGZyb20gPSBbXTtcblxuXHRcdGNvbnN0cnVjdG9yKGZyb21MYXllciwgdG9MYXllciwgdHlwZSwgd2VpZ2h0cykge1xuXHRcdFx0dGhpcy5mcm9tID0gZnJvbUxheWVyO1xuXHRcdFx0dGhpcy50byA9IHRvTGF5ZXI7XG5cdFx0XHR0aGlzLnNlbGZjb25uZWN0aW9uID0gdG9MYXllciA9PSBmcm9tTGF5ZXI7XG5cdFx0XHR0aGlzLnR5cGUgPSB0eXBlO1xuXHRcdFx0dGhpcy5jb25uZWN0aW9ucyA9IHt9O1xuXHRcdFx0dGhpcy5saXN0ID0gW107XG5cdFx0XHR0aGlzLnNpemUgPSAwO1xuXHRcdFx0dGhpcy5nYXRlZGZyb20gPSBbXTtcblxuXG5cdFx0XHRpZiAodHlwZW9mIHRoaXMudHlwZSA9PSAndW5kZWZpbmVkJykge1xuXHRcdFx0XHRpZiAoZnJvbUxheWVyID09IHRvTGF5ZXIpXG5cdFx0XHRcdFx0dGhpcy50eXBlID0gTGF5ZXIuY29ubmVjdGlvblR5cGUuT05FX1RPX09ORTtcblx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdHRoaXMudHlwZSA9IExheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19BTEw7XG5cdFx0XHR9XG5cblx0XHRcdGlmICh0aGlzLnR5cGUgPT0gTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTCB8fFxuXHRcdFx0XHR0aGlzLnR5cGUgPT0gTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0VMU0UpIHtcblx0XHRcdFx0Zm9yICh2YXIgaGVyZSBpbiB0aGlzLmZyb20ubGlzdCkge1xuXHRcdFx0XHRcdGZvciAodmFyIHRoZXJlIGluIHRoaXMudG8ubGlzdCkge1xuXHRcdFx0XHRcdFx0dmFyIGZyb20gPSB0aGlzLmZyb20ubGlzdFtoZXJlXTtcblx0XHRcdFx0XHRcdHZhciB0byA9IHRoaXMudG8ubGlzdFt0aGVyZV07XG5cdFx0XHRcdFx0XHRpZiAodGhpcy50eXBlID09IExheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19FTFNFICYmIGZyb20gPT0gdG8pXG5cdFx0XHRcdFx0XHRcdGNvbnRpbnVlO1xuXHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBmcm9tLnByb2plY3QodG8sIHdlaWdodHMpO1xuXG5cdFx0XHRcdFx0XHR0aGlzLmNvbm5lY3Rpb25zW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcblx0XHRcdFx0XHRcdHRoaXMuc2l6ZSA9IHRoaXMubGlzdC5wdXNoKGNvbm5lY3Rpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fVxuXHRcdFx0fSBlbHNlIGlmICh0aGlzLnR5cGUgPT0gTGF5ZXIuY29ubmVjdGlvblR5cGUuT05FX1RPX09ORSkge1xuXG5cdFx0XHRcdGZvciAodmFyIG5ldXJvbiBpbiB0aGlzLmZyb20ubGlzdCkge1xuXHRcdFx0XHRcdHZhciBmcm9tID0gdGhpcy5mcm9tLmxpc3RbbmV1cm9uXTtcblx0XHRcdFx0XHR2YXIgdG8gPSB0aGlzLnRvLmxpc3RbbmV1cm9uXTtcblx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IGZyb20ucHJvamVjdCh0bywgd2VpZ2h0cyk7XG5cblx0XHRcdFx0XHR0aGlzLmNvbm5lY3Rpb25zW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcblx0XHRcdFx0XHR0aGlzLnNpemUgPSB0aGlzLmxpc3QucHVzaChjb25uZWN0aW9uKTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXG5cdFx0XHRmcm9tTGF5ZXIuY29ubmVjdGVkdG8ucHVzaCh0aGlzKTtcblx0XHR9XG5cdH1cbn0iLCJpbXBvcnQgbGF5ZXIgPSByZXF1aXJlKCcuL2xheWVyJyk7XG5pbXBvcnQgU3F1YXNoID0gcmVxdWlyZSgnLi9zcXVhc2gnKTtcbmltcG9ydCBTeW5hcHRpYyA9IHJlcXVpcmUoJy4vc3luYXB0aWMnKTtcbmltcG9ydCBfbmV1cm9uID0gcmVxdWlyZSgnLi9uZXVyb24nKTtcblxuLyoqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIE5FVFdPUktcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cbmRlY2xhcmUgZnVuY3Rpb24gZXNjYXBlKGE6IHN0cmluZyk6IHN0cmluZztcblxuXG5leHBvcnQgY2xhc3MgTmV0d29yayB7XG5cdG9wdGltaXplZCA9IG51bGw7XG5cdGxheWVycyA9IHtcblx0XHRpbnB1dDogbnVsbCxcblx0XHRoaWRkZW46IHt9LFxuXHRcdG91dHB1dDogbnVsbFxuXHR9O1xuXHRjb25zdHJ1Y3RvcihsYXllcnMpIHtcblx0XHRpZiAodHlwZW9mIGxheWVycyAhPSAndW5kZWZpbmVkJykge1xuXHRcdFx0dGhpcy5sYXllcnMgPSBsYXllcnMgfHwge1xuXHRcdFx0XHRpbnB1dDogbnVsbCxcblx0XHRcdFx0aGlkZGVuOiB7fSxcblx0XHRcdFx0b3V0cHV0OiBudWxsXG5cdFx0XHR9O1xuXHRcdFx0dGhpcy5vcHRpbWl6ZWQgPSBudWxsO1xuXHRcdH1cblx0fVxuXG5cdC8vIGZlZWQtZm9yd2FyZCBhY3RpdmF0aW9uIG9mIGFsbCB0aGUgbGF5ZXJzIHRvIHByb2R1Y2UgYW4gb3VwdXRcblx0YWN0aXZhdGUoaW5wdXQpIHtcblxuXHRcdGlmICh0aGlzLm9wdGltaXplZCA9PT0gZmFsc2UpIHtcblx0XHRcdHRoaXMubGF5ZXJzLmlucHV0LmFjdGl2YXRlKGlucHV0KTtcblx0XHRcdGZvciAodmFyIGxheWVyIGluIHRoaXMubGF5ZXJzLmhpZGRlbilcblx0XHRcdFx0dGhpcy5sYXllcnMuaGlkZGVuW2xheWVyXS5hY3RpdmF0ZSgpO1xuXHRcdFx0cmV0dXJuIHRoaXMubGF5ZXJzLm91dHB1dC5hY3RpdmF0ZSgpO1xuXHRcdH1cblx0XHRlbHNlIHtcblx0XHRcdGlmICh0aGlzLm9wdGltaXplZCA9PSBudWxsKVxuXHRcdFx0XHR0aGlzLm9wdGltaXplKCk7XG5cdFx0XHRyZXR1cm4gdGhpcy5vcHRpbWl6ZWQuYWN0aXZhdGUoaW5wdXQpO1xuXHRcdH1cblx0fVxuXG5cdC8vIGJhY2stcHJvcGFnYXRlIHRoZSBlcnJvciB0aHJ1IHRoZSBuZXR3b3JrXG5cdHByb3BhZ2F0ZShyYXRlOiBudW1iZXIsIHRhcmdldD8pIHtcblxuXHRcdGlmICh0aGlzLm9wdGltaXplZCA9PT0gZmFsc2UpIHtcblx0XHRcdHRoaXMubGF5ZXJzLm91dHB1dC5wcm9wYWdhdGUocmF0ZSwgdGFyZ2V0KTtcblx0XHRcdHZhciByZXZlcnNlID0gW107XG5cdFx0XHRmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pXG5cdFx0XHRcdHJldmVyc2UucHVzaCh0aGlzLmxheWVycy5oaWRkZW5bbGF5ZXJdKTtcblx0XHRcdHJldmVyc2UucmV2ZXJzZSgpO1xuXHRcdFx0Zm9yICh2YXIgbGF5ZXIgaW4gcmV2ZXJzZSlcblx0XHRcdFx0cmV2ZXJzZVtsYXllcl0ucHJvcGFnYXRlKHJhdGUpO1xuXHRcdH1cblx0XHRlbHNlIHtcblx0XHRcdGlmICh0aGlzLm9wdGltaXplZCA9PSBudWxsKVxuXHRcdFx0XHR0aGlzLm9wdGltaXplKCk7XG5cdFx0XHR0aGlzLm9wdGltaXplZC5wcm9wYWdhdGUocmF0ZSwgdGFyZ2V0KTtcblx0XHR9XG5cdH1cblxuXHQvLyBwcm9qZWN0IGEgY29ubmVjdGlvbiB0byBhbm90aGVyIHVuaXQgKGVpdGhlciBhIG5ldHdvcmsgb3IgYSBsYXllcilcblx0cHJvamVjdCh1bml0LCB0eXBlLCB3ZWlnaHRzKSB7XG5cblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplZC5yZXNldCgpO1xuXG5cdFx0aWYgKHVuaXQgaW5zdGFuY2VvZiBOZXR3b3JrKVxuXHRcdFx0cmV0dXJuIHRoaXMubGF5ZXJzLm91dHB1dC5wcm9qZWN0KHVuaXQubGF5ZXJzLmlucHV0LCB0eXBlLCB3ZWlnaHRzKTtcblxuXHRcdGlmICh1bml0IGluc3RhbmNlb2YgbGF5ZXIuTGF5ZXIpXG5cdFx0XHRyZXR1cm4gdGhpcy5sYXllcnMub3V0cHV0LnByb2plY3QodW5pdCwgdHlwZSwgd2VpZ2h0cyk7XG5cblx0XHR0aHJvdyBcIkludmFsaWQgYXJndW1lbnQsIHlvdSBjYW4gb25seSBwcm9qZWN0IGNvbm5lY3Rpb25zIHRvIExBWUVSUyBhbmQgTkVUV09SS1MhXCI7XG5cdH1cblxuXHQvLyBsZXQgdGhpcyBuZXR3b3JrIGdhdGUgYSBjb25uZWN0aW9uXG5cdGdhdGUoY29ubmVjdGlvbiwgdHlwZSkge1xuXHRcdGlmICh0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cdFx0dGhpcy5sYXllcnMub3V0cHV0LmdhdGUoY29ubmVjdGlvbiwgdHlwZSk7XG5cdH1cblxuXHQvLyBjbGVhciBhbGwgZWxlZ2liaWxpdHkgdHJhY2VzIGFuZCBleHRlbmRlZCBlbGVnaWJpbGl0eSB0cmFjZXMgKHRoZSBuZXR3b3JrIGZvcmdldHMgaXRzIGNvbnRleHQsIGJ1dCBub3Qgd2hhdCB3YXMgdHJhaW5lZClcblx0Y2xlYXIoKSB7XG5cblx0XHR0aGlzLnJlc3RvcmUoKTtcblxuXHRcdHZhciBpbnB1dExheWVyID0gdGhpcy5sYXllcnMuaW5wdXQsXG5cdFx0XHRvdXRwdXRMYXllciA9IHRoaXMubGF5ZXJzLm91dHB1dDtcblxuXHRcdGlucHV0TGF5ZXIuY2xlYXIoKTtcblx0XHRmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pIHtcblx0XHRcdHZhciBoaWRkZW5MYXllciA9IHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl07XG5cdFx0XHRoaWRkZW5MYXllci5jbGVhcigpO1xuXHRcdH1cblx0XHRvdXRwdXRMYXllci5jbGVhcigpO1xuXG5cdFx0aWYgKHRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcblx0fVxuXG5cdC8vIHJlc2V0IGFsbCB3ZWlnaHRzIGFuZCBjbGVhciBhbGwgdHJhY2VzIChlbmRzIHVwIGxpa2UgYSBuZXcgbmV0d29yaylcblx0cmVzZXQoKSB7XG5cblx0XHR0aGlzLnJlc3RvcmUoKTtcblxuXHRcdHZhciBpbnB1dExheWVyID0gdGhpcy5sYXllcnMuaW5wdXQsXG5cdFx0XHRvdXRwdXRMYXllciA9IHRoaXMubGF5ZXJzLm91dHB1dDtcblxuXHRcdGlucHV0TGF5ZXIucmVzZXQoKTtcblx0XHRmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pIHtcblx0XHRcdHZhciBoaWRkZW5MYXllciA9IHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl07XG5cdFx0XHRoaWRkZW5MYXllci5yZXNldCgpO1xuXHRcdH1cblx0XHRvdXRwdXRMYXllci5yZXNldCgpO1xuXG5cdFx0aWYgKHRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcblx0fVxuXG5cdC8vIGhhcmRjb2RlcyB0aGUgYmVoYXZpb3VyIG9mIHRoZSB3aG9sZSBuZXR3b3JrIGludG8gYSBzaW5nbGUgb3B0aW1pemVkIGZ1bmN0aW9uXG5cdG9wdGltaXplKCkge1xuXG5cdFx0dmFyIHRoYXQgPSB0aGlzO1xuXHRcdHZhciBvcHRpbWl6ZWQ6IFN5bmFwdGljLklDb21waWxlZFBhcmFtZXRlcnMgPSB7fTtcblx0XHR2YXIgbmV1cm9ucyA9IHRoaXMubmV1cm9ucygpO1xuXG5cdFx0Zm9yICh2YXIgaSBpbiBuZXVyb25zKSB7XG5cdFx0XHR2YXIgbmV1cm9uID0gbmV1cm9uc1tpXS5uZXVyb247XG5cdFx0XHR2YXIgbGF5ZXIgPSBuZXVyb25zW2ldLmxheWVyO1xuXHRcdFx0Lypcblx0XHRcdEZJWE1FOiBkb2VzIHRoaXMgd29ya2VkIG9uY2U/XG5cdFx0XHRcblx0XHRcdHdoaWxlIChuZXVyb24ubmV1cm9uKVxuXHRcdFx0XHRuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuXHRcdFx0Ki9cblxuXHRcdFx0b3B0aW1pemVkID0gbmV1cm9uLm9wdGltaXplKG9wdGltaXplZCwgbGF5ZXIpO1xuXHRcdH1cblxuXHRcdGZvciAodmFyIGkgaW4gb3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlcylcblx0XHRcdG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXNbaV0ucmV2ZXJzZSgpO1xuXHRcdG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMucmV2ZXJzZSgpO1xuXG5cdFx0dmFyIGhhcmRjb2RlID0gXCJcIjtcblx0XHRoYXJkY29kZSArPSBcInZhciBGID0gRmxvYXQ2NEFycmF5ID8gbmV3IEZsb2F0NjRBcnJheShcIiArIG9wdGltaXplZC5tZW1vcnkgK1xuXHRcdFwiKSA6IFtdOyBcIjtcblx0XHRmb3IgKHZhciBpIGluIG9wdGltaXplZC52YXJpYWJsZXMpXG5cdFx0XHRoYXJkY29kZSArPSBcIkZbXCIgKyBvcHRpbWl6ZWQudmFyaWFibGVzW2ldLmlkICsgXCJdID0gXCIgKyAob3B0aW1pemVkLnZhcmlhYmxlc1tcblx0XHRcdFx0aV0udmFsdWUgfHwgMCkgKyBcIjsgXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgYWN0aXZhdGUgPSBmdW5jdGlvbihpbnB1dCl7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJpbmZsdWVuY2VzID0gW107XCI7XG5cdFx0Zm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQuaW5wdXRzKVxuXHRcdFx0aGFyZGNvZGUgKz0gXCJGW1wiICsgb3B0aW1pemVkLmlucHV0c1tpXSArIFwiXSA9IGlucHV0W1wiICsgaSArIFwiXTsgXCI7XG5cdFx0Zm9yICh2YXIgY3VycmVudExheWVyIGluIG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlcykge1xuXHRcdFx0aWYgKG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdLmxlbmd0aCA+IDApIHtcblx0XHRcdFx0Zm9yICh2YXIgY3VycmVudE5ldXJvbiBpbiBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXSkge1xuXHRcdFx0XHRcdGhhcmRjb2RlICs9IG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdW2N1cnJlbnROZXVyb25dLmpvaW4oXCIgXCIpO1xuXHRcdFx0XHRcdGhhcmRjb2RlICs9IG9wdGltaXplZC50cmFjZV9zZW50ZW5jZXNbY3VycmVudExheWVyXVtjdXJyZW50TmV1cm9uXS5qb2luKFwiIFwiKTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdH1cblx0XHRoYXJkY29kZSArPSBcIiB2YXIgb3V0cHV0ID0gW107IFwiXG5cdFx0Zm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQub3V0cHV0cylcblx0XHRcdGhhcmRjb2RlICs9IFwib3V0cHV0W1wiICsgaSArIFwiXSA9IEZbXCIgKyBvcHRpbWl6ZWQub3V0cHV0c1tpXSArIFwiXTsgXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJyZXR1cm4gb3V0cHV0OyB9OyBcIlxuXHRcdGhhcmRjb2RlICs9IFwidmFyIHByb3BhZ2F0ZSA9IGZ1bmN0aW9uKHJhdGUsIHRhcmdldCl7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJGW1wiICsgb3B0aW1pemVkLnZhcmlhYmxlcy5yYXRlLmlkICsgXCJdID0gcmF0ZTsgXCI7XG5cdFx0Zm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQudGFyZ2V0cylcblx0XHRcdGhhcmRjb2RlICs9IFwiRltcIiArIG9wdGltaXplZC50YXJnZXRzW2ldICsgXCJdID0gdGFyZ2V0W1wiICsgaSArIFwiXTsgXCI7XG5cdFx0Zm9yICh2YXIgY3VycmVudExheWVyIGluIG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMpXG5cdFx0XHRmb3IgKHZhciBjdXJyZW50TmV1cm9uIGluIG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXSlcblx0XHRcdFx0aGFyZGNvZGUgKz0gb3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdW2N1cnJlbnROZXVyb25dLmpvaW4oXCIgXCIpICsgXCIgXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCIgfTtcXG5cIjtcblx0XHRoYXJkY29kZSArPVxuXHRcdFwidmFyIG93bmVyc2hpcCA9IGZ1bmN0aW9uKG1lbW9yeUJ1ZmZlcil7XFxuRiA9IG1lbW9yeUJ1ZmZlcjtcXG50aGlzLm1lbW9yeSA9IEY7XFxufTtcXG5cIjtcblx0XHRoYXJkY29kZSArPVxuXHRcdFwicmV0dXJuIHtcXG5tZW1vcnk6IEYsXFxuYWN0aXZhdGU6IGFjdGl2YXRlLFxcbnByb3BhZ2F0ZTogcHJvcGFnYXRlLFxcbm93bmVyc2hpcDogb3duZXJzaGlwXFxufTtcIjtcblx0XHRoYXJkY29kZSA9IGhhcmRjb2RlLnNwbGl0KFwiO1wiKS5qb2luKFwiO1xcblwiKTtcblxuXHRcdHZhciBjb25zdHJ1Y3RvciA9IG5ldyBGdW5jdGlvbihoYXJkY29kZSk7XG5cblx0XHR2YXIgbmV0d29yayA9IGNvbnN0cnVjdG9yKCk7XG5cblx0XHRuZXR3b3JrLmRhdGEgPSB7XG5cdFx0XHR2YXJpYWJsZXM6IG9wdGltaXplZC52YXJpYWJsZXMsXG5cdFx0XHRhY3RpdmF0ZTogb3B0aW1pemVkLmFjdGl2YXRpb25fc2VudGVuY2VzLFxuXHRcdFx0cHJvcGFnYXRlOiBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzLFxuXHRcdFx0dHJhY2U6IG9wdGltaXplZC50cmFjZV9zZW50ZW5jZXMsXG5cdFx0XHRpbnB1dHM6IG9wdGltaXplZC5pbnB1dHMsXG5cdFx0XHRvdXRwdXRzOiBvcHRpbWl6ZWQub3V0cHV0cyxcblx0XHRcdGNoZWNrX2FjdGl2YXRpb246IHRoaXMuYWN0aXZhdGUsXG5cdFx0XHRjaGVja19wcm9wYWdhdGlvbjogdGhpcy5wcm9wYWdhdGVcblx0XHR9XG5cblx0XHRuZXR3b3JrLnJlc2V0ID0gZnVuY3Rpb24oKSB7XG5cdFx0XHRpZiAodGhhdC5vcHRpbWl6ZWQpIHtcblx0XHRcdFx0dGhhdC5vcHRpbWl6ZWQgPSBudWxsO1xuXHRcdFx0XHR0aGF0LmFjdGl2YXRlID0gbmV0d29yay5kYXRhLmNoZWNrX2FjdGl2YXRpb247XG5cdFx0XHRcdHRoYXQucHJvcGFnYXRlID0gbmV0d29yay5kYXRhLmNoZWNrX3Byb3BhZ2F0aW9uO1xuXHRcdFx0fVxuXHRcdH1cblxuXHRcdHRoaXMub3B0aW1pemVkID0gbmV0d29yaztcblx0XHR0aGlzLmFjdGl2YXRlID0gbmV0d29yay5hY3RpdmF0ZTtcblx0XHR0aGlzLnByb3BhZ2F0ZSA9IG5ldHdvcmsucHJvcGFnYXRlO1xuXHR9XG5cblx0Ly8gcmVzdG9yZXMgYWxsIHRoZSB2YWx1ZXMgZnJvbSB0aGUgb3B0aW1pemVkIG5ldHdvcmsgdGhlIHRoZWlyIHJlc3BlY3RpdmUgb2JqZWN0cyBpbiBvcmRlciB0byBtYW5pcHVsYXRlIHRoZSBuZXR3b3JrXG5cdHJlc3RvcmUoKSB7XG5cdFx0aWYgKCF0aGlzLm9wdGltaXplZClcblx0XHRcdHJldHVybjtcblxuXHRcdHZhciBvcHRpbWl6ZWQgPSB0aGlzLm9wdGltaXplZDtcblxuXHRcdHZhciBnZXRWYWx1ZSA9IGZ1bmN0aW9uKC4uLmFyZ3M6IGFueVtdKSB7XG5cdFx0XHR2YXIgdW5pdCA9IGFyZ3Muc2hpZnQoKTtcblx0XHRcdHZhciBwcm9wID0gYXJncy5wb3AoKTtcblxuXHRcdFx0dmFyIGlkID0gcHJvcCArICdfJztcblx0XHRcdGZvciAodmFyIHByb3BlcnR5IGluIGFyZ3MpXG5cdFx0XHRcdGlkICs9IGFyZ3NbcHJvcGVydHldICsgJ18nO1xuXHRcdFx0aWQgKz0gdW5pdC5JRDtcblxuXHRcdFx0dmFyIG1lbW9yeSA9IG9wdGltaXplZC5tZW1vcnk7XG5cdFx0XHR2YXIgdmFyaWFibGVzID0gb3B0aW1pemVkLmRhdGEudmFyaWFibGVzO1xuXG5cdFx0XHRpZiAoaWQgaW4gdmFyaWFibGVzKVxuXHRcdFx0XHRyZXR1cm4gbWVtb3J5W3ZhcmlhYmxlc1tpZF0uaWRdO1xuXHRcdFx0cmV0dXJuIDA7XG5cdFx0fVxuXG5cdFx0dmFyIGxpc3QgPSB0aGlzLm5ldXJvbnMoKTtcblxuXHRcdC8vIGxpbmsgaWQncyB0byBwb3NpdGlvbnMgaW4gdGhlIGFycmF5XG5cdFx0dmFyIGlkcyA9IHt9O1xuXHRcdGZvciAodmFyIGkgaW4gbGlzdCkge1xuXHRcdFx0dmFyIG5ldXJvbiA9IGxpc3RbaV0ubmV1cm9uO1xuXHRcdFx0Lypcblx0XHRcdEZJWE1FOiBkb2VzIHRoaXMgd29ya2VkIG9uY2U/XG5cdFx0XHRcblx0XHRcdHdoaWxlIChuZXVyb24ubmV1cm9uKVxuXHRcdFx0XHRuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuXHRcdFx0Ki9cblxuXHRcdFx0bmV1cm9uLnN0YXRlID0gZ2V0VmFsdWUobmV1cm9uLCAnc3RhdGUnKTtcblx0XHRcdG5ldXJvbi5vbGQgPSBnZXRWYWx1ZShuZXVyb24sICdvbGQnKTtcblx0XHRcdG5ldXJvbi5hY3RpdmF0aW9uID0gZ2V0VmFsdWUobmV1cm9uLCAnYWN0aXZhdGlvbicpO1xuXHRcdFx0bmV1cm9uLmJpYXMgPSBnZXRWYWx1ZShuZXVyb24sICdiaWFzJyk7XG5cblx0XHRcdGZvciAodmFyIGlucHV0IGluIG5ldXJvbi50cmFjZS5lbGVnaWJpbGl0eSlcblx0XHRcdFx0bmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0XSA9IGdldFZhbHVlKG5ldXJvbiwgJ3RyYWNlJyxcblx0XHRcdFx0XHQnZWxlZ2liaWxpdHknLCBpbnB1dCk7XG5cblx0XHRcdGZvciAodmFyIGdhdGVkIGluIG5ldXJvbi50cmFjZS5leHRlbmRlZClcblx0XHRcdFx0Zm9yICh2YXIgaW5wdXQgaW4gbmV1cm9uLnRyYWNlLmV4dGVuZGVkW2dhdGVkXSlcblx0XHRcdFx0XHRuZXVyb24udHJhY2UuZXh0ZW5kZWRbZ2F0ZWRdW2lucHV0XSA9IGdldFZhbHVlKG5ldXJvbiwgJ3RyYWNlJyxcblx0XHRcdFx0XHRcdCdleHRlbmRlZCcsIGdhdGVkLCBpbnB1dCk7XG5cdFx0fVxuXG5cdFx0Ly8gZ2V0IGNvbm5lY3Rpb25zXG5cdFx0Zm9yICh2YXIgaSBpbiBsaXN0KSB7XG5cdFx0XHR2YXIgbmV1cm9uID0gbGlzdFtpXS5uZXVyb247XG5cdFx0XHQvKlxuXHRcdFx0RklYTUU6IGRvZXMgdGhpcyB3b3JrZWQgb25jZT9cblx0XHRcdFxuXHRcdFx0d2hpbGUgKG5ldXJvbi5uZXVyb24pXG5cdFx0XHRcdG5ldXJvbiA9IG5ldXJvbi5uZXVyb247XG5cdFx0XHQqL1xuXG5cdFx0XHRmb3IgKHZhciBqIGluIG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpIHtcblx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBuZXVyb24uY29ubmVjdGlvbnMucHJvamVjdGVkW2pdO1xuXHRcdFx0XHRjb25uZWN0aW9uLndlaWdodCA9IGdldFZhbHVlKGNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdFx0Y29ubmVjdGlvbi5nYWluID0gZ2V0VmFsdWUoY29ubmVjdGlvbiwgJ2dhaW4nKTtcblx0XHRcdH1cblx0XHR9XG5cdH1cblxuXHQvLyByZXR1cm5zIGFsbCB0aGUgbmV1cm9ucyBpbiB0aGUgbmV0d29ya1xuXHRuZXVyb25zKCk6IE5ldHdvcmsuSU5ldHdvcmtOZXVyb25bXSB7XG5cdFx0dmFyIG5ldXJvbnM6IE5ldHdvcmsuSU5ldHdvcmtOZXVyb25bXSA9IFtdO1xuXG5cdFx0dmFyIGlucHV0TGF5ZXIgPSB0aGlzLmxheWVycy5pbnB1dC5uZXVyb25zKCksXG5cdFx0XHRvdXRwdXRMYXllciA9IHRoaXMubGF5ZXJzLm91dHB1dC5uZXVyb25zKCk7XG5cblx0XHRmb3IgKHZhciBuZXVyb24gaW4gaW5wdXRMYXllcilcblx0XHRcdG5ldXJvbnMucHVzaCh7XG5cdFx0XHRcdG5ldXJvbjogaW5wdXRMYXllcltuZXVyb25dLFxuXHRcdFx0XHRsYXllcjogJ2lucHV0J1xuXHRcdFx0fSk7XG5cblx0XHRmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pIHtcblx0XHRcdHZhciBoaWRkZW5MYXllciA9IHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl0ubmV1cm9ucygpO1xuXHRcdFx0Zm9yICh2YXIgbmV1cm9uIGluIGhpZGRlbkxheWVyKVxuXHRcdFx0XHRuZXVyb25zLnB1c2goe1xuXHRcdFx0XHRcdG5ldXJvbjogaGlkZGVuTGF5ZXJbbmV1cm9uXSxcblx0XHRcdFx0XHRsYXllcjogbGF5ZXJcblx0XHRcdFx0fSk7XG5cdFx0fVxuXHRcdGZvciAodmFyIG5ldXJvbiBpbiBvdXRwdXRMYXllcilcblx0XHRcdG5ldXJvbnMucHVzaCh7XG5cdFx0XHRcdG5ldXJvbjogb3V0cHV0TGF5ZXJbbmV1cm9uXSxcblx0XHRcdFx0bGF5ZXI6ICdvdXRwdXQnXG5cdFx0XHR9KTtcblxuXHRcdHJldHVybiBuZXVyb25zO1xuXHR9XG5cblx0Ly8gcmV0dXJucyBudW1iZXIgb2YgaW5wdXRzIG9mIHRoZSBuZXR3b3JrXG5cdGlucHV0cygpOiBudW1iZXIge1xuXHRcdHJldHVybiB0aGlzLmxheWVycy5pbnB1dC5zaXplO1xuXHR9XG5cblx0Ly8gcmV0dXJucyBudW1iZXIgb2Ygb3V0cHV0cyBvZiBodGUgbmV0d29ya1xuXHRvdXRwdXRzKCk6IG51bWJlciB7XG5cdFx0cmV0dXJuIHRoaXMubGF5ZXJzLm91dHB1dC5zaXplO1xuXHR9XG5cblx0Ly8gc2V0cyB0aGUgbGF5ZXJzIG9mIHRoZSBuZXR3b3JrXG5cdHNldChsYXllcnMpIHtcblxuXHRcdHRoaXMubGF5ZXJzID0gbGF5ZXJzO1xuXHRcdGlmICh0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cdH1cblxuXHRzZXRPcHRpbWl6ZShib29sKSB7XG5cdFx0dGhpcy5yZXN0b3JlKCk7XG5cdFx0aWYgKHRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcblx0XHR0aGlzLm9wdGltaXplZCA9IGJvb2wgPyBudWxsIDogZmFsc2U7XG5cdH1cblxuXHQvLyByZXR1cm5zIGEganNvbiB0aGF0IHJlcHJlc2VudHMgYWxsIHRoZSBuZXVyb25zIGFuZCBjb25uZWN0aW9ucyBvZiB0aGUgbmV0d29ya1xuXHR0b0pTT04oaWdub3JlVHJhY2VzKSB7XG5cblx0XHR0aGlzLnJlc3RvcmUoKTtcblxuXHRcdHZhciBsaXN0ID0gdGhpcy5uZXVyb25zKCk7XG5cdFx0dmFyIG5ldXJvbnMgPSBbXTtcblx0XHR2YXIgY29ubmVjdGlvbnMgPSBbXTtcblxuXHRcdC8vIGxpbmsgaWQncyB0byBwb3NpdGlvbnMgaW4gdGhlIGFycmF5XG5cdFx0dmFyIGlkcyA9IHt9O1xuXHRcdGZvciAodmFyIGkgaW4gbGlzdCkge1xuXHRcdFx0dmFyIG5ldXJvbiA9IGxpc3RbaV0ubmV1cm9uO1xuXHRcdFx0Lypcblx0XHRcdEZJWE1FOiBkb2VzIHRoaXMgd29ya2VkIG9uY2U/XG5cdFx0XHRcblx0XHRcdHdoaWxlIChuZXVyb24ubmV1cm9uKVxuXHRcdFx0XHRuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuXHRcdFx0Ki9cblxuXHRcdFx0aWRzW25ldXJvbi5JRF0gPSBpO1xuXG5cdFx0XHR2YXIgY29weSA9IHtcblx0XHRcdFx0dHJhY2U6IHtcblx0XHRcdFx0XHRlbGVnaWJpbGl0eToge30sXG5cdFx0XHRcdFx0ZXh0ZW5kZWQ6IHt9XG5cdFx0XHRcdH0sXG5cdFx0XHRcdHN0YXRlOiBuZXVyb24uc3RhdGUsXG5cdFx0XHRcdG9sZDogbmV1cm9uLm9sZCxcblx0XHRcdFx0YWN0aXZhdGlvbjogbmV1cm9uLmFjdGl2YXRpb24sXG5cdFx0XHRcdGJpYXM6IG5ldXJvbi5iaWFzLFxuXHRcdFx0XHRsYXllcjogbGlzdFtpXS5sYXllcixcblx0XHRcdFx0c3F1YXNoOiBudWxsXG5cdFx0XHR9O1xuXG5cdFx0XHRjb3B5LnNxdWFzaCA9IG5ldXJvbi5zcXVhc2ggPT0gU3F1YXNoLkxPR0lTVElDID8gXCJMT0dJU1RJQ1wiIDpcblx0XHRcdFx0bmV1cm9uLnNxdWFzaCA9PSBTcXVhc2guVEFOSCA/IFwiVEFOSFwiIDpcblx0XHRcdFx0XHRuZXVyb24uc3F1YXNoID09IFNxdWFzaC5JREVOVElUWSA/IFwiSURFTlRJVFlcIiA6XG5cdFx0XHRcdFx0XHRuZXVyb24uc3F1YXNoID09IFNxdWFzaC5ITElNID8gXCJITElNXCIgOlxuXHRcdFx0XHRcdFx0XHRudWxsO1xuXG5cdFx0XHRuZXVyb25zLnB1c2goY29weSk7XG5cdFx0fVxuXG5cdFx0aWYgKCFpZ25vcmVUcmFjZXMpXG5cdFx0XHRmb3IgKHZhciBpIGluIG5ldXJvbnMpIHtcblx0XHRcdFx0dmFyIGNvcGllZE5ldXJvbiA9IG5ldXJvbnNbaV07XG5cblx0XHRcdFx0Zm9yICh2YXIgaW5wdXQgaW4gbmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5KVxuXHRcdFx0XHRcdGNvcGllZE5ldXJvbi50cmFjZS5lbGVnaWJpbGl0eVtpbnB1dF0gPSBuZXVyb24udHJhY2UuZWxlZ2liaWxpdHlbaW5wdXRdO1xuXG5cdFx0XHRcdGZvciAodmFyIGdhdGVkIGluIG5ldXJvbi50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHRcdGNvcGllZE5ldXJvbi50cmFjZS5leHRlbmRlZFtnYXRlZF0gPSB7fTtcblx0XHRcdFx0XHRmb3IgKHZhciBpbnB1dCBpbiBuZXVyb24udHJhY2UuZXh0ZW5kZWRbZ2F0ZWRdKVxuXHRcdFx0XHRcdFx0Y29waWVkTmV1cm9uLnRyYWNlLmV4dGVuZGVkW2lkc1tnYXRlZF1dW2lucHV0XSA9IG5ldXJvbi50cmFjZS5leHRlbmRlZFtcblx0XHRcdFx0XHRcdGdhdGVkXVtpbnB1dF07XG5cdFx0XHRcdH1cblx0XHRcdH1cblxuXHRcdC8vIGdldCBjb25uZWN0aW9uc1xuXHRcdGZvciAodmFyIGkgaW4gbGlzdCkge1xuXHRcdFx0dmFyIG5ldXJvbiA9IGxpc3RbaV0ubmV1cm9uO1xuXHRcdFx0XHRcblx0XHRcdC8qXG5cdFx0XHRGSVhNRTogZG9lcyB0aGlzIHdvcmtlZCBvbmNlP1xuXHRcdFx0XG5cdFx0XHR3aGlsZSAobmV1cm9uLm5ldXJvbilcblx0XHRcdFx0bmV1cm9uID0gbmV1cm9uLm5ldXJvbjtcblx0XHRcdCovXG5cblx0XHRcdGZvciAodmFyIGogaW4gbmV1cm9uLmNvbm5lY3Rpb25zLnByb2plY3RlZCkge1xuXHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbal07XG5cdFx0XHRcdGNvbm5lY3Rpb25zLnB1c2goe1xuXHRcdFx0XHRcdGZyb206IGlkc1tjb25uZWN0aW9uLmZyb20uSURdLFxuXHRcdFx0XHRcdHRvOiBpZHNbY29ubmVjdGlvbi50by5JRF0sXG5cdFx0XHRcdFx0d2VpZ2h0OiBjb25uZWN0aW9uLndlaWdodCxcblx0XHRcdFx0XHRnYXRlcjogY29ubmVjdGlvbi5nYXRlciA/IGlkc1tjb25uZWN0aW9uLmdhdGVyLklEXSA6IG51bGwsXG5cdFx0XHRcdH0pO1xuXHRcdFx0fVxuXHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGVkKCkpXG5cdFx0XHRcdGNvbm5lY3Rpb25zLnB1c2goe1xuXHRcdFx0XHRcdGZyb206IGlkc1tuZXVyb24uSURdLFxuXHRcdFx0XHRcdHRvOiBpZHNbbmV1cm9uLklEXSxcblx0XHRcdFx0XHR3ZWlnaHQ6IG5ldXJvbi5zZWxmY29ubmVjdGlvbi53ZWlnaHQsXG5cdFx0XHRcdFx0Z2F0ZXI6IG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA/IGlkc1tuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXJcblx0XHRcdFx0XHRcdC5JRF0gOiBudWxsLFxuXHRcdFx0XHR9KTtcblx0XHR9XG5cblx0XHRyZXR1cm4ge1xuXHRcdFx0bmV1cm9uczogbmV1cm9ucyxcblx0XHRcdGNvbm5lY3Rpb25zOiBjb25uZWN0aW9uc1xuXHRcdH1cblx0fVxuICBcblx0Ly8gZXhwb3J0IHRoZSB0b3BvbG9neSBpbnRvIGRvdCBsYW5ndWFnZSB3aGljaCBjYW4gYmUgdmlzdWFsaXplZCBhcyBncmFwaHMgdXNpbmcgZG90XG5cdC8qIGV4YW1wbGU6IC4uLiBjb25zb2xlLmxvZyhuZXQudG9Eb3RMYW5nKCkpO1xuXHRcdFx0XHQkIG5vZGUgZXhhbXBsZS5qcyA+IGV4YW1wbGUuZG90XG5cdFx0XHRcdCQgZG90IGV4YW1wbGUuZG90IC1UcG5nID4gb3V0LnBuZ1xuXHQqL1xuXHR0b0RvdChlZGdlY29ubmVjdGlvbikge1xuXHRcdGlmICghIHR5cGVvZiBlZGdlY29ubmVjdGlvbilcblx0XHRcdGVkZ2Vjb25uZWN0aW9uID0gZmFsc2U7XG5cdFx0dmFyIGNvZGUgPSBcImRpZ3JhcGggbm4ge1xcbiAgICByYW5rZGlyID0gQlRcXG5cIjtcblx0XHR2YXIgbGF5ZXJzID0gW3RoaXMubGF5ZXJzLmlucHV0XS5jb25jYXQodGhpcy5sYXllcnMuaGlkZGVuLCB0aGlzLmxheWVycy5vdXRwdXQpO1xuXHRcdGZvciAodmFyIGxheWVyIGluIGxheWVycykge1xuXHRcdFx0Zm9yICh2YXIgdG8gaW4gbGF5ZXJzW2xheWVyXS5jb25uZWN0ZWR0bykgeyAvLyBwcm9qZWN0aW9uc1xuXHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IGxheWVyc1tsYXllcl0uY29ubmVjdGVkdG9bdG9dO1xuXHRcdFx0XHR2YXIgbGF5ZXJ0byA9IGNvbm5lY3Rpb24udG87XG5cdFx0XHRcdHZhciBzaXplID0gY29ubmVjdGlvbi5zaXplO1xuXHRcdFx0XHR2YXIgbGF5ZXJJRCA9IGxheWVycy5pbmRleE9mKGxheWVyc1tsYXllcl0pO1xuXHRcdFx0XHR2YXIgbGF5ZXJ0b0lEID0gbGF5ZXJzLmluZGV4T2YobGF5ZXJ0byk7XG5cdFx0XHRcdC8qIGh0dHA6Ly9zdGFja292ZXJmbG93LmNvbS9xdWVzdGlvbnMvMjY4NDU1NDAvY29ubmVjdC1lZGdlcy13aXRoLWdyYXBoLWRvdFxuKiBET1QgZG9lcyBub3Qgc3VwcG9ydCBlZGdlLXRvLWVkZ2UgY29ubmVjdGlvbnNcbiogVGhpcyB3b3JrYXJvdW5kIHByb2R1Y2VzIHNvbWV3aGF0IHdlaXJkIGdyYXBocyAuLi5cblx0XHRcdFx0Ki9cblx0XHRcdFx0aWYgKGVkZ2Vjb25uZWN0aW9uKSB7XG5cdFx0XHRcdFx0aWYgKGNvbm5lY3Rpb24uZ2F0ZWRmcm9tLmxlbmd0aCkge1xuXHRcdFx0XHRcdFx0dmFyIGZha2VOb2RlID0gXCJmYWtlXCIgKyBsYXllcklEICsgXCJfXCIgKyBsYXllcnRvSUQ7XG5cdFx0XHRcdFx0XHRjb2RlICs9IFwiICAgIFwiICsgZmFrZU5vZGUgK1xuXHRcdFx0XHRcdFx0XCIgW2xhYmVsID0gXFxcIlxcXCIsIHNoYXBlID0gcG9pbnQsIHdpZHRoID0gMC4wMSwgaGVpZ2h0ID0gMC4wMV1cXG5cIjtcblx0XHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBsYXllcklEICsgXCIgLT4gXCIgKyBmYWtlTm9kZSArIFwiIFtsYWJlbCA9IFwiICsgc2l6ZSArIFwiLCBhcnJvd2hlYWQgPSBub25lXVxcblwiO1xuXHRcdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGZha2VOb2RlICsgXCIgLT4gXCIgKyBsYXllcnRvSUQgKyBcIlxcblwiO1xuXHRcdFx0XHRcdH0gZWxzZVxuXHRcdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGxheWVySUQgKyBcIiAtPiBcIiArIGxheWVydG9JRCArIFwiIFtsYWJlbCA9IFwiICsgc2l6ZSArIFwiXVxcblwiO1xuXHRcdFx0XHRcdGZvciAodmFyIGZyb20gaW4gY29ubmVjdGlvbi5nYXRlZGZyb20pIHsgLy8gZ2F0aW5nc1xuXHRcdFx0XHRcdFx0dmFyIGxheWVyZnJvbSA9IGNvbm5lY3Rpb24uZ2F0ZWRmcm9tW2Zyb21dLmxheWVyO1xuXHRcdFx0XHRcdFx0dmFyIHR5cGUgPSBjb25uZWN0aW9uLmdhdGVkZnJvbVtmcm9tXS50eXBlO1xuXHRcdFx0XHRcdFx0dmFyIGxheWVyZnJvbUlEID0gbGF5ZXJzLmluZGV4T2YobGF5ZXJmcm9tKTtcblx0XHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBsYXllcmZyb21JRCArIFwiIC0+IFwiICsgZmFrZU5vZGUgKyBcIiBbY29sb3IgPSBibHVlXVxcblwiO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0XHRjb2RlICs9IFwiICAgIFwiICsgbGF5ZXJJRCArIFwiIC0+IFwiICsgbGF5ZXJ0b0lEICsgXCIgW2xhYmVsID0gXCIgKyBzaXplICsgXCJdXFxuXCI7XG5cdFx0XHRcdFx0Zm9yICh2YXIgZnJvbSBpbiBjb25uZWN0aW9uLmdhdGVkZnJvbSkgeyAvLyBnYXRpbmdzXG5cdFx0XHRcdFx0XHR2YXIgbGF5ZXJmcm9tID0gY29ubmVjdGlvbi5nYXRlZGZyb21bZnJvbV0ubGF5ZXI7XG5cdFx0XHRcdFx0XHR2YXIgdHlwZSA9IGNvbm5lY3Rpb24uZ2F0ZWRmcm9tW2Zyb21dLnR5cGU7XG5cdFx0XHRcdFx0XHR2YXIgbGF5ZXJmcm9tSUQgPSBsYXllcnMuaW5kZXhPZihsYXllcmZyb20pO1xuXHRcdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGxheWVyZnJvbUlEICsgXCIgLT4gXCIgKyBsYXllcnRvSUQgKyBcIiBbY29sb3IgPSBibHVlXVxcblwiO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdH1cblx0XHRjb2RlICs9IFwifVxcblwiO1xuXHRcdHJldHVybiB7XG5cdFx0XHRjb2RlOiBjb2RlLFxuXHRcdFx0bGluazogXCJodHRwczovL2NoYXJ0Lmdvb2dsZWFwaXMuY29tL2NoYXJ0P2NobD1cIiArIGVzY2FwZShjb2RlLnJlcGxhY2UoXCIvIC9nXCIsIFwiK1wiKSkgKyBcIiZjaHQ9Z3ZcIlxuXHRcdH1cblx0fVxuXG5cdC8vIHJldHVybnMgYSBmdW5jdGlvbiB0aGF0IHdvcmtzIGFzIHRoZSBhY3RpdmF0aW9uIG9mIHRoZSBuZXR3b3JrIGFuZCBjYW4gYmUgdXNlZCB3aXRob3V0IGRlcGVuZGluZyBvbiB0aGUgbGlicmFyeVxuXHRzdGFuZGFsb25lKCkge1xuXHRcdGlmICghdGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplKCk7XG5cblx0XHR2YXIgZGF0YSA9IHRoaXMub3B0aW1pemVkLmRhdGE7XG5cblx0XHQvLyBidWlsZCBhY3RpdmF0aW9uIGZ1bmN0aW9uXG5cdFx0dmFyIGFjdGl2YXRpb24gPSBcImZ1bmN0aW9uIChpbnB1dCkge1xcblwiO1xuXG5cdFx0Ly8gYnVpbGQgaW5wdXRzXG5cdFx0Zm9yICh2YXIgaSBpbiBkYXRhLmlucHV0cylcblx0XHRcdGFjdGl2YXRpb24gKz0gXCJGW1wiICsgZGF0YS5pbnB1dHNbaV0gKyBcIl0gPSBpbnB1dFtcIiArIGkgKyBcIl07XFxuXCI7XG5cblx0XHQvLyBidWlsZCBuZXR3b3JrIGFjdGl2YXRpb25cblx0XHRmb3IgKHZhciBuZXVyb24gaW4gZGF0YS5hY3RpdmF0ZSkgeyAvLyBzaG91bGRuJ3QgdGhpcyBiZSBsYXllcj9cblx0XHRcdGZvciAodmFyIHNlbnRlbmNlIGluIGRhdGEuYWN0aXZhdGVbbmV1cm9uXSlcblx0XHRcdFx0YWN0aXZhdGlvbiArPSBkYXRhLmFjdGl2YXRlW25ldXJvbl1bc2VudGVuY2VdICsgXCJcXG5cIjtcblx0XHR9XG5cblx0XHQvLyBidWlsZCBvdXRwdXRzXG5cdFx0YWN0aXZhdGlvbiArPSBcInZhciBvdXRwdXQgPSBbXTtcXG5cIjtcblx0XHRmb3IgKHZhciBpIGluIGRhdGEub3V0cHV0cylcblx0XHRcdGFjdGl2YXRpb24gKz0gXCJvdXRwdXRbXCIgKyBpICsgXCJdID0gRltcIiArIGRhdGEub3V0cHV0c1tpXSArIFwiXTtcXG5cIjtcblx0XHRhY3RpdmF0aW9uICs9IFwicmV0dXJuIG91dHB1dDtcXG59XCI7XG5cblx0XHQvLyByZWZlcmVuY2UgYWxsIHRoZSBwb3NpdGlvbnMgaW4gbWVtb3J5XG5cdFx0dmFyIG1lbW9yeSA9IGFjdGl2YXRpb24ubWF0Y2goL0ZcXFsoXFxkKylcXF0vZyk7XG5cdFx0dmFyIGRpbWVuc2lvbiA9IDA7XG5cdFx0dmFyIGlkcyA9IHt9O1xuXHRcdGZvciAodmFyIGFkZHJlc3MgaW4gbWVtb3J5KSB7XG5cdFx0XHR2YXIgdG1wID0gbWVtb3J5W2FkZHJlc3NdLm1hdGNoKC9cXGQrLylbMF07XG5cdFx0XHRpZiAoISh0bXAgaW4gaWRzKSkge1xuXHRcdFx0XHRpZHNbdG1wXSA9IGRpbWVuc2lvbisrO1xuXHRcdFx0fVxuXHRcdH1cblx0XHR2YXIgaGFyZGNvZGUgPSBcIkYgPSB7XFxuXCI7XG5cdFx0Zm9yICh2YXIgaSBpbiBpZHMpXG5cdFx0XHRoYXJkY29kZSArPSBpZHNbaV0gKyBcIjogXCIgKyB0aGlzLm9wdGltaXplZC5tZW1vcnlbaV0gKyBcIixcXG5cIjtcblx0XHRoYXJkY29kZSA9IGhhcmRjb2RlLnN1YnN0cmluZygwLCBoYXJkY29kZS5sZW5ndGggLSAyKSArIFwiXFxufTtcXG5cIjtcblx0XHRoYXJkY29kZSA9IFwidmFyIHJ1biA9IFwiICsgYWN0aXZhdGlvbi5yZXBsYWNlKC9GXFxbKFxcZCspXS9nLCBmdW5jdGlvbihcblx0XHRcdGluZGV4KSB7XG5cdFx0XHRyZXR1cm4gJ0ZbJyArIGlkc1tpbmRleC5tYXRjaCgvXFxkKy8pWzBdXSArICddJ1xuXHRcdH0pLnJlcGxhY2UoXCJ7XFxuXCIsIFwie1xcblwiICsgaGFyZGNvZGUgKyBcIlwiKSArIFwiO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwicmV0dXJuIHJ1blwiO1xuXG5cdFx0Ly8gcmV0dXJuIHN0YW5kYWxvbmUgZnVuY3Rpb25cblx0XHRyZXR1cm4gbmV3IEZ1bmN0aW9uKGhhcmRjb2RlKSgpO1xuXHR9XG5cblx0d29ya2VyKCkge1xuXHRcdGlmICghdGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplKCk7XG5cblx0XHR2YXIgaGFyZGNvZGUgPSBcInZhciBpbnB1dHMgPSBcIiArIHRoaXMub3B0aW1pemVkLmRhdGEuaW5wdXRzLmxlbmd0aCArXG5cdFx0XHRcIjtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcInZhciBvdXRwdXRzID0gXCIgKyB0aGlzLm9wdGltaXplZC5kYXRhLm91dHB1dHMubGVuZ3RoICtcblx0XHRcIjtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcInZhciBGID0gbnVsbDtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcInZhciBhY3RpdmF0ZSA9IFwiICsgdGhpcy5vcHRpbWl6ZWQuYWN0aXZhdGUudG9TdHJpbmcoKSArXG5cdFx0XCI7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgcHJvcGFnYXRlID0gXCIgKyB0aGlzLm9wdGltaXplZC5wcm9wYWdhdGUudG9TdHJpbmcoKSArXG5cdFx0XCI7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJvbm1lc3NhZ2UgPSBmdW5jdGlvbihlKXtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcIkYgPSBlLmRhdGEubWVtb3J5QnVmZmVyO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwiaWYgKGUuZGF0YS5hY3Rpb24gPT0gJ2FjdGl2YXRlJyl7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJpZiAoZS5kYXRhLmlucHV0Lmxlbmd0aCA9PSBpbnB1dHMpe1xcblwiO1xuXHRcdGhhcmRjb2RlICs9XG5cdFx0XCJwb3N0TWVzc2FnZSggeyBhY3Rpb246ICdhY3RpdmF0ZScsIG91dHB1dDogYWN0aXZhdGUoZS5kYXRhLmlucHV0KSwgbWVtb3J5QnVmZmVyOiBGIH0sIFtGLmJ1ZmZlcl0pO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwifVxcbn1cXG5lbHNlIGlmIChlLmRhdGEuYWN0aW9uID09ICdwcm9wYWdhdGUnKXtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcInByb3BhZ2F0ZShlLmRhdGEucmF0ZSwgZS5kYXRhLnRhcmdldCk7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz1cblx0XHRcInBvc3RNZXNzYWdlKHsgYWN0aW9uOiAncHJvcGFnYXRlJywgbWVtb3J5QnVmZmVyOiBGIH0sIFtGLmJ1ZmZlcl0pO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwifVxcbn1cXG5cIjtcblxuXHRcdHZhciBibG9iID0gbmV3IEJsb2IoW2hhcmRjb2RlXSk7XG5cdFx0dmFyIGJsb2JVUkwgPSAoPGFueT53aW5kb3cpLlVSTC5jcmVhdGVPYmplY3RVUkwoYmxvYik7XG5cblx0XHRyZXR1cm4gbmV3IFdvcmtlcihibG9iVVJMKTtcblx0fVxuXG5cdC8vIHJldHVybnMgYSBjb3B5IG9mIHRoZSBuZXR3b3JrXG5cdGNsb25lKGlnbm9yZVRyYWNlcykge1xuXHRcdHJldHVybiBOZXR3b3JrLmZyb21KU09OKHRoaXMudG9KU09OKGlnbm9yZVRyYWNlcykpO1xuXHR9XG5cblx0c3RhdGljIGZyb21KU09OKGpzb24pIHtcblxuXHRcdHZhciBuZXVyb25zID0gW107XG5cblx0XHR2YXIgbGF5ZXJzID0ge1xuXHRcdFx0aW5wdXQ6IG5ldyBsYXllci5MYXllcigwKSxcblx0XHRcdGhpZGRlbjogW10sXG5cdFx0XHRvdXRwdXQ6IG5ldyBsYXllci5MYXllcigwKVxuXHRcdH1cblx0XHRcblxuXHRcdGZvciAodmFyIGkgaW4ganNvbi5uZXVyb25zKSB7XG5cdFx0XHR2YXIgY29uZmlnID0ganNvbi5uZXVyb25zW2ldO1xuXG5cdFx0XHR2YXIgbmV1cm9uID0gbmV3IF9uZXVyb24uTmV1cm9uKCk7XG5cdFx0XHRuZXVyb24udHJhY2UuZWxlZ2liaWxpdHkgPSBjb25maWcudHJhY2UuZWxlZ2liaWxpdHk7XG5cdFx0XHRuZXVyb24udHJhY2UuZXh0ZW5kZWQgPSBjb25maWcudHJhY2UuZXh0ZW5kZWQ7XG5cdFx0XHRuZXVyb24uc3RhdGUgPSBjb25maWcuc3RhdGU7XG5cdFx0XHRuZXVyb24ub2xkID0gY29uZmlnLm9sZDtcblx0XHRcdG5ldXJvbi5hY3RpdmF0aW9uID0gY29uZmlnLmFjdGl2YXRpb247XG5cdFx0XHRuZXVyb24uYmlhcyA9IGNvbmZpZy5iaWFzO1xuXHRcdFx0bmV1cm9uLnNxdWFzaCA9IGNvbmZpZy5zcXVhc2ggaW4gU3F1YXNoID8gU3F1YXNoW2NvbmZpZy5zcXVhc2hdIDpcblx0XHRcdFx0U3F1YXNoLkxPR0lTVElDO1xuXHRcdFx0bmV1cm9ucy5wdXNoKG5ldXJvbik7XG5cblx0XHRcdGlmIChjb25maWcubGF5ZXIgPT0gJ2lucHV0Jylcblx0XHRcdFx0bGF5ZXJzLmlucHV0LmFkZChuZXVyb24pO1xuXHRcdFx0ZWxzZSBpZiAoY29uZmlnLmxheWVyID09ICdvdXRwdXQnKVxuXHRcdFx0XHRsYXllcnMub3V0cHV0LmFkZChuZXVyb24pO1xuXHRcdFx0ZWxzZSB7XG5cdFx0XHRcdGlmICh0eXBlb2YgbGF5ZXJzLmhpZGRlbltjb25maWcubGF5ZXJdID09ICd1bmRlZmluZWQnKVxuXHRcdFx0XHRcdGxheWVycy5oaWRkZW5bY29uZmlnLmxheWVyXSA9IG5ldyBsYXllci5MYXllcigwKTtcblx0XHRcdFx0bGF5ZXJzLmhpZGRlbltjb25maWcubGF5ZXJdLmFkZChuZXVyb24pO1xuXHRcdFx0fVxuXHRcdH1cblxuXHRcdGZvciAodmFyIGkgaW4ganNvbi5jb25uZWN0aW9ucykge1xuXHRcdFx0dmFyIGNvbmZpZyA9IGpzb24uY29ubmVjdGlvbnNbaV07XG5cdFx0XHR2YXIgZnJvbSA9IG5ldXJvbnNbY29uZmlnLmZyb21dO1xuXHRcdFx0dmFyIHRvID0gbmV1cm9uc1tjb25maWcudG9dO1xuXHRcdFx0dmFyIHdlaWdodCA9IGNvbmZpZy53ZWlnaHRcblx0XHRcdHZhciBnYXRlciA9IG5ldXJvbnNbY29uZmlnLmdhdGVyXTtcblxuXHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBmcm9tLnByb2plY3QodG8sIHdlaWdodCk7XG5cdFx0XHRpZiAoZ2F0ZXIpXG5cdFx0XHRcdGdhdGVyLmdhdGUoY29ubmVjdGlvbik7XG5cdFx0fVxuXG5cdFx0cmV0dXJuIG5ldyBOZXR3b3JrKGxheWVycyk7XG5cdH1cbn1cblxuZXhwb3J0IG1vZHVsZSBOZXR3b3JrIHtcblx0ZXhwb3J0IGludGVyZmFjZSBJTmV0d29ya05ldXJvbiB7XG5cdFx0bmV1cm9uOiBfbmV1cm9uLk5ldXJvbjtcblx0XHRsYXllcjogc3RyaW5nO1xuXHR9XG59IiwiLy8vIDxyZWZlcmVuY2UgcGF0aD1cInN5bmFwdGljLnRzXCIgLz5cblxuaW1wb3J0IFN5bmFwdGljID0gcmVxdWlyZSgnLi9zeW5hcHRpYycpO1xuaW1wb3J0IFNxdWFzaCA9IHJlcXVpcmUoJy4vc3F1YXNoJyk7XG5cbi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgTkVVUk9OXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuXG4vKiBUUyBDSEFOR0VTOlxuXG5cdE5vdyBOZXVyb24uY29ubmVjdGVkKG5ldXJvbikgcmV0dXJucyBudWxsIGluc3RlYWQgb2YgZmFsc2VcblxuKi9cblxuZXhwb3J0IGNsYXNzIE5ldXJvbiB7XG5cdElEID0gTmV1cm9uLnVpZCgpO1xuXHRsYWJlbCA9IG51bGw7XG5cdGNvbm5lY3Rpb25zOiBOZXVyb24uSU5ldXJvbkNvbm5lY3Rpb25zID0ge1xuXHRcdGlucHV0czoge30sXG5cdFx0cHJvamVjdGVkOiB7fSxcblx0XHRnYXRlZDoge31cblx0fTtcblx0ZXJyb3IgPSB7XG5cdFx0cmVzcG9uc2liaWxpdHk6IDAsXG5cdFx0cHJvamVjdGVkOiAwLFxuXHRcdGdhdGVkOiAwXG5cdH07XG5cdHRyYWNlID0ge1xuXHRcdGVsZWdpYmlsaXR5OiB7fSxcblx0XHRleHRlbmRlZDoge30sXG5cdFx0aW5mbHVlbmNlczoge31cblx0fTtcblx0c3RhdGUgPSAwO1xuXHRvbGQgPSAwO1xuXHRhY3RpdmF0aW9uID0gMDtcblx0c2VsZmNvbm5lY3Rpb24gPSBuZXcgTmV1cm9uLkNvbm5lY3Rpb24odGhpcywgdGhpcywgMCk7IC8vIHdlaWdodCA9IDAgLT4gbm90IGNvbm5lY3RlZFxuXHRzcXVhc2ggPSBTcXVhc2guTE9HSVNUSUM7XG5cdG5laWdoYm9vcnMgPSB7fTtcblx0YmlhcyA9IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xO1xuXHRkZXJpdmF0aXZlID0gMDtcbiAgXG5cdC8vIGFjdGl2YXRlIHRoZSBuZXVyb25cblx0YWN0aXZhdGUoaW5wdXQ/OiBudW1iZXIpIHtcblx0XHQvLyBhY3RpdmF0aW9uIGZyb20gZW52aXJvbWVudCAoZm9yIGlucHV0IG5ldXJvbnMpXG5cdFx0aWYgKHR5cGVvZiBpbnB1dCAhPSAndW5kZWZpbmVkJykge1xuXHRcdFx0dGhpcy5hY3RpdmF0aW9uID0gaW5wdXQ7XG5cdFx0XHR0aGlzLmRlcml2YXRpdmUgPSAwO1xuXHRcdFx0dGhpcy5iaWFzID0gMDtcblx0XHRcdHJldHVybiB0aGlzLmFjdGl2YXRpb247XG5cdFx0fVxuXG5cdFx0Ly8gb2xkIHN0YXRlXG5cdFx0dGhpcy5vbGQgPSB0aGlzLnN0YXRlO1xuXG5cdFx0Ly8gZXEuIDE1XG5cdFx0dGhpcy5zdGF0ZSA9IHRoaXMuc2VsZmNvbm5lY3Rpb24uZ2FpbiAqIHRoaXMuc2VsZmNvbm5lY3Rpb24ud2VpZ2h0ICpcblx0XHR0aGlzLnN0YXRlICsgdGhpcy5iaWFzO1xuXG5cdFx0Zm9yICh2YXIgaSBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0dmFyIHRoZUlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cdFx0XHR0aGlzLnN0YXRlICs9IHRoZUlucHV0LmZyb20uYWN0aXZhdGlvbiAqIHRoZUlucHV0LndlaWdodCAqIHRoZUlucHV0LmdhaW47XG5cdFx0fVxuXG5cdFx0Ly8gZXEuIDE2XG5cdFx0dGhpcy5hY3RpdmF0aW9uID0gdGhpcy5zcXVhc2godGhpcy5zdGF0ZSk7XG5cblx0XHQvLyBmJyhzKVxuXHRcdHRoaXMuZGVyaXZhdGl2ZSA9IHRoaXMuc3F1YXNoKHRoaXMuc3RhdGUsIHRydWUpO1xuXG5cdFx0Ly8gdXBkYXRlIHRyYWNlc1xuXHRcdHZhciBpbmZsdWVuY2VzID0gW107XG5cdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0Ly8gZXh0ZW5kZWQgZWxlZ2liaWxpdHkgdHJhY2Vcblx0XHRcdHZhciB4dHJhY2UgPSB0aGlzLnRyYWNlLmV4dGVuZGVkW2lkXTtcblx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXG5cdFx0XHQvLyBpZiBnYXRlZCBuZXVyb24ncyBzZWxmY29ubmVjdGlvbiBpcyBnYXRlZCBieSB0aGlzIHVuaXQsIHRoZSBpbmZsdWVuY2Uga2VlcHMgdHJhY2sgb2YgdGhlIG5ldXJvbidzIG9sZCBzdGF0ZVxuXHRcdFx0dmFyIGluZmx1ZW5jZSA9IG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzID8gbmV1cm9uLm9sZCA6IDA7XG5cblx0XHRcdC8vIGluZGV4IHJ1bnMgb3ZlciBhbGwgdGhlIGluY29taW5nIGNvbm5lY3Rpb25zIHRvIHRoZSBnYXRlZCBuZXVyb24gdGhhdCBhcmUgZ2F0ZWQgYnkgdGhpcyB1bml0XG5cdFx0XHRmb3IgKHZhciBpbmNvbWluZyBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXSkgeyAvLyBjYXB0dXJlcyB0aGUgZWZmZWN0IHRoYXQgaGFzIGFuIGlucHV0IGNvbm5lY3Rpb24gdG8gdGhpcyB1bml0LCBvbiBhIG5ldXJvbiB0aGF0IGlzIGdhdGVkIGJ5IHRoaXMgdW5pdFxuXHRcdFx0XHRpbmZsdWVuY2UgKz0gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF1baW5jb21pbmddLndlaWdodCAqXG5cdFx0XHRcdHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdW2luY29taW5nXS5mcm9tLmFjdGl2YXRpb247XG5cdFx0XHR9XG5cdFx0XHRpbmZsdWVuY2VzW25ldXJvbi5JRF0gPSBpbmZsdWVuY2U7XG5cdFx0fVxuXG5cdFx0Zm9yICh2YXIgaSBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0dmFyIHRoZUlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cblx0XHRcdC8vIGVsZWdpYmlsaXR5IHRyYWNlIC0gRXEuIDE3XG5cdFx0XHR0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RoZUlucHV0LklEXSA9IHRoaXMuc2VsZmNvbm5lY3Rpb24uZ2FpbiAqIHRoaXMuc2VsZmNvbm5lY3Rpb25cblx0XHRcdC53ZWlnaHQgKiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RoZUlucHV0LklEXSArIHRoZUlucHV0LmdhaW4gKiB0aGVJbnB1dC5mcm9tXG5cdFx0XHQuYWN0aXZhdGlvbjtcblxuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHQvLyBleHRlbmRlZCBlbGVnaWJpbGl0eSB0cmFjZVxuXHRcdFx0XHR2YXIgeHRyYWNlID0gdGhpcy50cmFjZS5leHRlbmRlZFtpZF07XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHR2YXIgaW5mbHVlbmNlID0gaW5mbHVlbmNlc1tuZXVyb24uSURdO1xuXG5cdFx0XHRcdC8vIGVxLiAxOFxuXHRcdFx0XHR4dHJhY2VbdGhlSW5wdXQuSURdID0gbmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhaW4gKiBuZXVyb24uc2VsZmNvbm5lY3Rpb25cblx0XHRcdFx0LndlaWdodCAqIHh0cmFjZVt0aGVJbnB1dC5JRF0gKyB0aGlzLmRlcml2YXRpdmUgKiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W1xuXHRcdFx0XHR0aGVJbnB1dC5JRF0gKiBpbmZsdWVuY2U7XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFx0Ly8gIHVwZGF0ZSBnYXRlZCBjb25uZWN0aW9uJ3MgZ2FpbnNcblx0XHRmb3IgKHZhciBjb25uZWN0aW9uIGluIHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWQpIHtcblx0XHRcdHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWRbY29ubmVjdGlvbl0uZ2FpbiA9IHRoaXMuYWN0aXZhdGlvbjtcblx0XHR9XG5cblx0XHRyZXR1cm4gdGhpcy5hY3RpdmF0aW9uO1xuXHR9XG5cblx0Ly8gYmFjay1wcm9wYWdhdGUgdGhlIGVycm9yXG5cdHByb3BhZ2F0ZShyYXRlOiBudW1iZXIsIHRhcmdldD86IG51bWJlcikge1xuXHRcdC8vIGVycm9yIGFjY3VtdWxhdG9yXG5cdFx0dmFyIGVycm9yID0gMDtcblxuXHRcdC8vIHdoZXRoZXIgb3Igbm90IHRoaXMgbmV1cm9uIGlzIGluIHRoZSBvdXRwdXQgbGF5ZXJcblx0XHR2YXIgaXNPdXRwdXQgPSB0eXBlb2YgdGFyZ2V0ICE9ICd1bmRlZmluZWQnICYmIHRhcmdldCAhPSBudWxsO1xuXG5cdFx0Ly8gb3V0cHV0IG5ldXJvbnMgZ2V0IHRoZWlyIGVycm9yIGZyb20gdGhlIGVudmlyb21lbnRcblx0XHRpZiAoaXNPdXRwdXQpXG5cdFx0XHR0aGlzLmVycm9yLnJlc3BvbnNpYmlsaXR5ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgPSB0YXJnZXQgLSB0aGlzLmFjdGl2YXRpb247IC8vIEVxLiAxMFxuICAgIFxuXHRcdGVsc2UgLy8gdGhlIHJlc3Qgb2YgdGhlIG5ldXJvbiBjb21wdXRlIHRoZWlyIGVycm9yIHJlc3BvbnNpYmlsaXRpZXMgYnkgYmFja3Byb3BhZ2F0aW9uXG5cdFx0e1xuXHRcdFx0Ly8gZXJyb3IgcmVzcG9uc2liaWxpdGllcyBmcm9tIGFsbCB0aGUgY29ubmVjdGlvbnMgcHJvamVjdGVkIGZyb20gdGhpcyBuZXVyb25cblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG5cdFx0XHRcdHZhciBjb25uZWN0aW9uID0gdGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbaWRdO1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gY29ubmVjdGlvbi50bztcblx0XHRcdFx0Ly8gRXEuIDIxXG5cdFx0XHRcdGVycm9yICs9IG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSAqIGNvbm5lY3Rpb24uZ2FpbiAqIGNvbm5lY3Rpb24ud2VpZ2h0O1xuXHRcdFx0fVxuXG5cdFx0XHQvLyBwcm9qZWN0ZWQgZXJyb3IgcmVzcG9uc2liaWxpdHlcblx0XHRcdHRoaXMuZXJyb3IucHJvamVjdGVkID0gdGhpcy5kZXJpdmF0aXZlICogZXJyb3I7XG5cblx0XHRcdGVycm9yID0gMDtcblx0XHRcdC8vIGVycm9yIHJlc3BvbnNpYmlsaXRpZXMgZnJvbSBhbGwgdGhlIGNvbm5lY3Rpb25zIGdhdGVkIGJ5IHRoaXMgbmV1cm9uXG5cdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdOyAvLyBnYXRlZCBuZXVyb25cblx0XHRcdFx0dmFyIGluZmx1ZW5jZSA9IG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzID8gbmV1cm9uLm9sZCA6IDA7IC8vIGlmIGdhdGVkIG5ldXJvbidzIHNlbGZjb25uZWN0aW9uIGlzIGdhdGVkIGJ5IHRoaXMgbmV1cm9uXG5cblx0XHRcdFx0Ly8gaW5kZXggcnVucyBvdmVyIGFsbCB0aGUgY29ubmVjdGlvbnMgdG8gdGhlIGdhdGVkIG5ldXJvbiB0aGF0IGFyZSBnYXRlZCBieSB0aGlzIG5ldXJvblxuXHRcdFx0XHRmb3IgKHZhciBpbnB1dCBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbaWRdKSB7IC8vIGNhcHR1cmVzIHRoZSBlZmZlY3QgdGhhdCB0aGUgaW5wdXQgY29ubmVjdGlvbiBvZiB0aGlzIG5ldXJvbiBoYXZlLCBvbiBhIG5ldXJvbiB3aGljaCBpdHMgaW5wdXQvcyBpcy9hcmUgZ2F0ZWQgYnkgdGhpcyBuZXVyb25cblx0XHRcdFx0XHRpbmZsdWVuY2UgKz0gdGhpcy50cmFjZS5pbmZsdWVuY2VzW2lkXVtpbnB1dF0ud2VpZ2h0ICogdGhpcy50cmFjZS5pbmZsdWVuY2VzW1xuXHRcdFx0XHRcdG5ldXJvbi5JRF1baW5wdXRdLmZyb20uYWN0aXZhdGlvbjtcblx0XHRcdFx0fVxuXHRcdFx0XHQvLyBlcS4gMjJcblx0XHRcdFx0ZXJyb3IgKz0gbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5ICogaW5mbHVlbmNlO1xuXHRcdFx0fVxuXG5cdFx0XHQvLyBnYXRlZCBlcnJvciByZXNwb25zaWJpbGl0eVxuXHRcdFx0dGhpcy5lcnJvci5nYXRlZCA9IHRoaXMuZGVyaXZhdGl2ZSAqIGVycm9yO1xuXG5cdFx0XHQvLyBlcnJvciByZXNwb25zaWJpbGl0eSAtIEVxLiAyM1xuXHRcdFx0dGhpcy5lcnJvci5yZXNwb25zaWJpbGl0eSA9IHRoaXMuZXJyb3IucHJvamVjdGVkICsgdGhpcy5lcnJvci5nYXRlZDtcblx0XHR9XG5cblx0XHQvLyBsZWFybmluZyByYXRlXG5cdFx0cmF0ZSA9IHJhdGUgfHwgLjE7XG5cblx0XHQvLyBhZGp1c3QgYWxsIHRoZSBuZXVyb24ncyBpbmNvbWluZyBjb25uZWN0aW9uc1xuXHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHR2YXIgdGhlSW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cblx0XHRcdC8vIEVxLiAyNFxuXHRcdFx0dmFyIGdyYWRpZW50ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgKiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RoZUlucHV0LklEXTtcblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdGdyYWRpZW50ICs9IG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSAqIHRoaXMudHJhY2UuZXh0ZW5kZWRbXG5cdFx0XHRcdG5ldXJvbi5JRF1bdGhlSW5wdXQuSURdO1xuXHRcdFx0fVxuXHRcdFx0dGhlSW5wdXQud2VpZ2h0ICs9IHJhdGUgKiBncmFkaWVudDsgLy8gYWRqdXN0IHdlaWdodHMgLSBha2EgbGVhcm5cblx0XHR9XG5cblx0XHQvLyBhZGp1c3QgYmlhc1xuXHRcdHRoaXMuYmlhcyArPSByYXRlICogdGhpcy5lcnJvci5yZXNwb25zaWJpbGl0eTtcblx0fVxuXG5cdHByb2plY3QobmV1cm9uLCB3ZWlnaHQ/OiBudW1iZXIpOiBOZXVyb24uQ29ubmVjdGlvbiB7XG5cdFx0Ly8gc2VsZi1jb25uZWN0aW9uXG5cdFx0aWYgKG5ldXJvbiA9PSB0aGlzKSB7XG5cdFx0XHR0aGlzLnNlbGZjb25uZWN0aW9uLndlaWdodCA9IDE7XG5cdFx0XHRyZXR1cm4gdGhpcy5zZWxmY29ubmVjdGlvbjtcblx0XHR9XG5cblx0XHQvLyBjaGVjayBpZiBjb25uZWN0aW9uIGFscmVhZHkgZXhpc3RzXG5cdFx0dmFyIGNvbm5lY3RlZCA9IHRoaXMuY29ubmVjdGVkKG5ldXJvbik7XG5cdFx0aWYgKGNvbm5lY3RlZCAmJiBjb25uZWN0ZWQudHlwZSA9PSBcInByb2plY3RlZFwiKSB7XG5cdFx0XHQvLyB1cGRhdGUgY29ubmVjdGlvblxuXHRcdFx0aWYgKHR5cGVvZiB3ZWlnaHQgIT0gJ3VuZGVmaW5lZCcpXG5cdFx0XHRcdGNvbm5lY3RlZC5jb25uZWN0aW9uLndlaWdodCA9IHdlaWdodDtcblx0XHRcdC8vIHJldHVybiBleGlzdGluZyBjb25uZWN0aW9uXG5cdFx0XHRyZXR1cm4gY29ubmVjdGVkLmNvbm5lY3Rpb247XG5cdFx0fSBlbHNlIHtcblx0XHRcdC8vIGNyZWF0ZSBhIG5ldyBjb25uZWN0aW9uXG5cdFx0XHR2YXIgY29ubmVjdGlvbiA9IG5ldyBOZXVyb24uQ29ubmVjdGlvbih0aGlzLCBuZXVyb24sIHdlaWdodCk7XG5cdFx0fVxuXG5cdFx0Ly8gcmVmZXJlbmNlIGFsbCB0aGUgY29ubmVjdGlvbnMgYW5kIHRyYWNlc1xuXHRcdHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcblx0XHR0aGlzLm5laWdoYm9vcnNbbmV1cm9uLklEXSA9IG5ldXJvbjtcblx0XHRuZXVyb24uY29ubmVjdGlvbnMuaW5wdXRzW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcblx0XHRuZXVyb24udHJhY2UuZWxlZ2liaWxpdHlbY29ubmVjdGlvbi5JRF0gPSAwO1xuXG5cdFx0Zm9yICh2YXIgaWQgaW4gbmV1cm9uLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHR2YXIgdHJhY2UgPSBuZXVyb24udHJhY2UuZXh0ZW5kZWRbaWRdO1xuXHRcdFx0dHJhY2VbY29ubmVjdGlvbi5JRF0gPSAwO1xuXHRcdH1cblxuXHRcdHJldHVybiBjb25uZWN0aW9uO1xuXHR9XG5cblx0Z2F0ZShjb25uZWN0aW9uKSB7XG5cdFx0Ly8gYWRkIGNvbm5lY3Rpb24gdG8gZ2F0ZWQgbGlzdFxuXHRcdHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWRbY29ubmVjdGlvbi5JRF0gPSBjb25uZWN0aW9uO1xuXG5cdFx0dmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24udG87XG5cdFx0aWYgKCEobmV1cm9uLklEIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpKSB7XG5cdFx0XHQvLyBleHRlbmRlZCB0cmFjZVxuXHRcdFx0dGhpcy5uZWlnaGJvb3JzW25ldXJvbi5JRF0gPSBuZXVyb247XG5cdFx0XHR2YXIgeHRyYWNlID0gdGhpcy50cmFjZS5leHRlbmRlZFtuZXVyb24uSURdID0ge307XG5cdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHR2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cdFx0XHRcdHh0cmFjZVtpbnB1dC5JRF0gPSAwO1xuXHRcdFx0fVxuXHRcdH1cblxuXHRcdC8vIGtlZXAgdHJhY2tcblx0XHRpZiAobmV1cm9uLklEIGluIHRoaXMudHJhY2UuaW5mbHVlbmNlcylcblx0XHRcdHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdLnB1c2goY29ubmVjdGlvbik7XG5cdFx0ZWxzZVxuXHRcdFx0dGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0gPSBbY29ubmVjdGlvbl07XG5cblx0XHQvLyBzZXQgZ2F0ZXJcblx0XHRjb25uZWN0aW9uLmdhdGVyID0gdGhpcztcblx0fVxuICBcblx0Ly8gcmV0dXJucyB0cnVlIG9yIGZhbHNlIHdoZXRoZXIgdGhlIG5ldXJvbiBpcyBzZWxmLWNvbm5lY3RlZCBvciBub3Rcblx0c2VsZmNvbm5lY3RlZCgpIHtcblx0XHRyZXR1cm4gdGhpcy5zZWxmY29ubmVjdGlvbi53ZWlnaHQgIT09IDA7XG5cdH1cblxuXHQvLyByZXR1cm5zIHRydWUgb3IgZmFsc2Ugd2hldGhlciB0aGUgbmV1cm9uIGlzIGNvbm5lY3RlZCB0byBhbm90aGVyIG5ldXJvbiAocGFyYW1ldGVyKVxuXHRjb25uZWN0ZWQobmV1cm9uKSB7XG5cdFx0dmFyIHJlc3VsdDoge1xuXHRcdFx0dHlwZTogc3RyaW5nO1xuXHRcdFx0Y29ubmVjdGlvbjogTmV1cm9uLkNvbm5lY3Rpb247XG5cdFx0fSA9IHtcblx0XHRcdFx0dHlwZTogbnVsbCxcblx0XHRcdFx0Y29ubmVjdGlvbjogbnVsbFxuXHRcdFx0fTtcblxuXHRcdGlmICh0aGlzID09IG5ldXJvbikge1xuXHRcdFx0aWYgKHRoaXMuc2VsZmNvbm5lY3RlZCgpKSB7XG5cdFx0XHRcdHJlc3VsdC50eXBlID0gJ3NlbGZjb25uZWN0aW9uJztcblx0XHRcdFx0cmVzdWx0LmNvbm5lY3Rpb24gPSB0aGlzLnNlbGZjb25uZWN0aW9uO1xuXHRcdFx0XHRyZXR1cm4gcmVzdWx0O1xuXHRcdFx0fSBlbHNlXG5cdFx0XHRcdHJldHVybiBudWxsO1xuXHRcdH1cblxuXHRcdGZvciAodmFyIHR5cGUgaW4gdGhpcy5jb25uZWN0aW9ucykge1xuXHRcdFx0Zm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zW3R5cGVdKSB7XG5cdFx0XHRcdHZhciBjb25uZWN0aW9uID0gdGhpcy5jb25uZWN0aW9uc1t0eXBlXVtjb25uZWN0aW9uXTtcblx0XHRcdFx0aWYgKGNvbm5lY3Rpb24udG8gPT0gbmV1cm9uKSB7XG5cdFx0XHRcdFx0cmVzdWx0LnR5cGUgPSB0eXBlO1xuXHRcdFx0XHRcdHJlc3VsdC5jb25uZWN0aW9uID0gY29ubmVjdGlvbjtcblx0XHRcdFx0XHRyZXR1cm4gcmVzdWx0O1xuXHRcdFx0XHR9IGVsc2UgaWYgKGNvbm5lY3Rpb24uZnJvbSA9PSBuZXVyb24pIHtcblx0XHRcdFx0XHRyZXN1bHQudHlwZSA9IHR5cGU7XG5cdFx0XHRcdFx0cmVzdWx0LmNvbm5lY3Rpb24gPSBjb25uZWN0aW9uO1xuXHRcdFx0XHRcdHJldHVybiByZXN1bHQ7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHR9XG5cblx0XHRyZXR1cm4gbnVsbDtcblx0fVxuXG5cdC8vIGNsZWFycyBhbGwgdGhlIHRyYWNlcyAodGhlIG5ldXJvbiBmb3JnZXRzIGl0J3MgY29udGV4dCwgYnV0IHRoZSBjb25uZWN0aW9ucyByZW1haW4gaW50YWN0KVxuXHRjbGVhcigpIHtcblxuXHRcdGZvciAodmFyIHRyYWNlIGluIHRoaXMudHJhY2UuZWxlZ2liaWxpdHkpXG5cdFx0XHR0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RyYWNlXSA9IDA7XG5cblx0XHRmb3IgKHZhciB0cmFjZSBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKVxuXHRcdFx0Zm9yICh2YXIgZXh0ZW5kZWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZFt0cmFjZV0pXG5cdFx0XHRcdHRoaXMudHJhY2UuZXh0ZW5kZWRbdHJhY2VdW2V4dGVuZGVkXSA9IDA7XG5cblx0XHR0aGlzLmVycm9yLnJlc3BvbnNpYmlsaXR5ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgPSB0aGlzLmVycm9yLmdhdGVkID0gMDtcblx0fVxuXG5cdC8vIGFsbCB0aGUgY29ubmVjdGlvbnMgYXJlIHJhbmRvbWl6ZWQgYW5kIHRoZSB0cmFjZXMgYXJlIGNsZWFyZWRcblx0cmVzZXQoKSB7XG5cdFx0dGhpcy5jbGVhcigpO1xuXG5cdFx0Zm9yICh2YXIgdHlwZSBpbiB0aGlzLmNvbm5lY3Rpb25zKVxuXHRcdFx0Zm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zW3R5cGVdKVxuXHRcdFx0XHR0aGlzLmNvbm5lY3Rpb25zW3R5cGVdW2Nvbm5lY3Rpb25dLndlaWdodCA9IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xO1xuXHRcdHRoaXMuYmlhcyA9IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xO1xuXG5cdFx0dGhpcy5vbGQgPSB0aGlzLnN0YXRlID0gdGhpcy5hY3RpdmF0aW9uID0gMDtcblx0fVxuXHRcblxuICBcblxuXHQvLyBoYXJkY29kZXMgdGhlIGJlaGF2aW91ciBvZiB0aGUgbmV1cm9uIGludG8gYW4gb3B0aW1pemVkIGZ1bmN0aW9uXG5cdG9wdGltaXplKG9wdGltaXplZCwgbGF5ZXIpOiBTeW5hcHRpYy5JQ29tcGlsZWRQYXJhbWV0ZXJzIHtcblxuXHRcdG9wdGltaXplZCA9IG9wdGltaXplZCB8fCB7fTtcblx0XHR2YXIgdGhhdCA9IHRoaXM7XG5cdFx0dmFyIHN0b3JlX2FjdGl2YXRpb24gPSBbXTtcblx0XHR2YXIgc3RvcmVfdHJhY2UgPSBbXTtcblx0XHR2YXIgc3RvcmVfcHJvcGFnYXRpb24gPSBbXTtcblx0XHR2YXIgdmFySUQgPSBvcHRpbWl6ZWQubWVtb3J5IHx8IDA7XG5cdFx0dmFyIG5ldXJvbnMgPSBvcHRpbWl6ZWQubmV1cm9ucyB8fCAxO1xuXHRcdHZhciBpbnB1dHMgPSBvcHRpbWl6ZWQuaW5wdXRzIHx8IFtdO1xuXHRcdHZhciB0YXJnZXRzID0gb3B0aW1pemVkLnRhcmdldHMgfHwgW107XG5cdFx0dmFyIG91dHB1dHMgPSBvcHRpbWl6ZWQub3V0cHV0cyB8fCBbXTtcblx0XHR2YXIgdmFyaWFibGVzID0gb3B0aW1pemVkLnZhcmlhYmxlcyB8fCB7fTtcblx0XHR2YXIgYWN0aXZhdGlvbl9zZW50ZW5jZXMgPSBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXMgfHwgW107XG5cdFx0dmFyIHRyYWNlX3NlbnRlbmNlcyA9IG9wdGltaXplZC50cmFjZV9zZW50ZW5jZXMgfHwgW107XG5cdFx0dmFyIHByb3BhZ2F0aW9uX3NlbnRlbmNlcyA9IG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMgfHwgW107XG5cdFx0dmFyIGxheWVycyA9IG9wdGltaXplZC5sYXllcnMgfHwgeyBfX2NvdW50OiAwLCBfX25ldXJvbjogMCB9O1xuXG5cdFx0Ly8gYWxsb2NhdGUgc2VudGVuY2VzXG5cdFx0dmFyIGFsbG9jYXRlID0gZnVuY3Rpb24oc3RvcmUpIHtcblx0XHRcdHZhciBhbGxvY2F0ZWQgPSBsYXllciBpbiBsYXllcnMgJiYgc3RvcmVbbGF5ZXJzLl9fY291bnRdO1xuXHRcdFx0aWYgKCFhbGxvY2F0ZWQpIHtcblx0XHRcdFx0bGF5ZXJzLl9fY291bnQgPSBzdG9yZS5wdXNoKFtdKSAtIDE7XG5cdFx0XHRcdGxheWVyc1tsYXllcl0gPSBsYXllcnMuX19jb3VudDtcblx0XHRcdH1cblx0XHR9XG5cdFx0YWxsb2NhdGUoYWN0aXZhdGlvbl9zZW50ZW5jZXMpO1xuXHRcdGFsbG9jYXRlKHRyYWNlX3NlbnRlbmNlcyk7XG5cdFx0YWxsb2NhdGUocHJvcGFnYXRpb25fc2VudGVuY2VzKTtcblx0XHR2YXIgY3VycmVudExheWVyID0gbGF5ZXJzLl9fY291bnQ7XG5cblx0XHQvLyBnZXQvcmVzZXJ2ZSBzcGFjZSBpbiBtZW1vcnkgYnkgY3JlYXRpbmcgYSB1bmlxdWUgSUQgZm9yIGEgdmFyaWFibGVsXG5cdFx0dmFyIGdldFZhciA9IGZ1bmN0aW9uKC4uLmFyZ3M6IGFueVtdKSB7XG5cdFx0XHR2YXIgaWQ7XG5cdFx0XHRpZiAoYXJncy5sZW5ndGggPT0gMSkge1xuXG5cdFx0XHRcdGlmIChhcmdzWzBdID09ICd0YXJnZXQnKSB7XG5cdFx0XHRcdFx0aWQgPSAndGFyZ2V0XycgKyB0YXJnZXRzLmxlbmd0aDtcblx0XHRcdFx0XHR0YXJnZXRzLnB1c2godmFySUQpO1xuXHRcdFx0XHR9IGVsc2Vcblx0XHRcdFx0XHRpZCA9IGFyZ3NbMF07XG5cdFx0XHRcdGlmIChpZCBpbiB2YXJpYWJsZXMpXG5cdFx0XHRcdFx0cmV0dXJuIHZhcmlhYmxlc1tpZF07XG5cdFx0XHRcdHJldHVybiB2YXJpYWJsZXNbaWRdID0ge1xuXHRcdFx0XHRcdHZhbHVlOiAwLFxuXHRcdFx0XHRcdGlkOiB2YXJJRCsrXG5cdFx0XHRcdH07XG5cdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHR2YXIgZXh0ZW5kZWQgPSBhcmdzLmxlbmd0aCA+IDI7XG5cdFx0XHRcdGlmIChleHRlbmRlZClcblx0XHRcdFx0XHR2YXIgdmFsdWUgPSBhcmdzLnBvcCgpO1xuXG5cdFx0XHRcdHZhciB1bml0ID0gYXJncy5zaGlmdCgpO1xuXHRcdFx0XHR2YXIgcHJvcCA9IGFyZ3MucG9wKCk7XG5cblx0XHRcdFx0aWYgKCFleHRlbmRlZClcblx0XHRcdFx0XHR2YXIgdmFsdWUgPSB1bml0W3Byb3BdO1xuXG5cdFx0XHRcdGlkID0gcHJvcCArICdfJztcblx0XHRcdFx0Zm9yICh2YXIgcHJvcGVydHkgaW4gYXJncylcblx0XHRcdFx0XHRpZCArPSBhcmdzW3Byb3BlcnR5XSArICdfJztcblx0XHRcdFx0aWQgKz0gdW5pdC5JRDtcblx0XHRcdFx0aWYgKGlkIGluIHZhcmlhYmxlcylcblx0XHRcdFx0XHRyZXR1cm4gdmFyaWFibGVzW2lkXTtcblxuXHRcdFx0XHRyZXR1cm4gdmFyaWFibGVzW2lkXSA9IHtcblx0XHRcdFx0XHR2YWx1ZTogdmFsdWUsXG5cdFx0XHRcdFx0aWQ6IHZhcklEKytcblx0XHRcdFx0fTtcblx0XHRcdH1cblx0XHR9O1xuXG5cdFx0Ly8gYnVpbGQgc2VudGVuY2Vcblx0XHR2YXIgYnVpbGRTZW50ZW5jZSA9IGZ1bmN0aW9uKC4uLmFyZ3M6IGFueVtdKSB7XG5cdFx0XHR2YXIgc3RvcmUgPSBhcmdzLnBvcCgpO1xuXHRcdFx0dmFyIHNlbnRlbmNlID0gXCJcIjtcblx0XHRcdGZvciAodmFyIGkgaW4gYXJncylcblx0XHRcdFx0aWYgKHR5cGVvZiBhcmdzW2ldID09ICdzdHJpbmcnKVxuXHRcdFx0XHRcdHNlbnRlbmNlICs9IGFyZ3NbaV07XG5cdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRzZW50ZW5jZSArPSAnRlsnICsgYXJnc1tpXS5pZCArICddJztcblxuXHRcdFx0c3RvcmUucHVzaChzZW50ZW5jZSArICc7Jyk7XG5cdFx0fVxuXG5cdFx0Ly8gaGVscGVyIHRvIGNoZWNrIGlmIGFuIG9iamVjdCBpcyBlbXB0eVxuXHRcdHZhciBpc0VtcHR5ID0gZnVuY3Rpb24ob2JqKSB7XG5cdFx0XHRmb3IgKHZhciBwcm9wIGluIG9iaikge1xuXHRcdFx0XHRpZiAob2JqLmhhc093blByb3BlcnR5KHByb3ApKVxuXHRcdFx0XHRcdHJldHVybiBmYWxzZTtcblx0XHRcdH1cblx0XHRcdHJldHVybiB0cnVlO1xuXHRcdH07XG5cblx0XHQvLyBjaGFyYWN0ZXJpc3RpY3Mgb2YgdGhlIG5ldXJvblxuXHRcdHZhciBub1Byb2plY3Rpb25zID0gaXNFbXB0eSh0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZCk7XG5cdFx0dmFyIG5vR2F0ZXMgPSBpc0VtcHR5KHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWQpO1xuXHRcdHZhciBpc0lucHV0ID0gbGF5ZXIgPT0gJ2lucHV0JyA/IHRydWUgOiBpc0VtcHR5KHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKTtcblx0XHR2YXIgaXNPdXRwdXQgPSBsYXllciA9PSAnb3V0cHV0JyA/IHRydWUgOiBub1Byb2plY3Rpb25zICYmIG5vR2F0ZXM7XG5cblx0XHQvLyBvcHRpbWl6ZSBuZXVyb24ncyBiZWhhdmlvdXJcblx0XHR2YXIgcmF0ZSA9IGdldFZhcigncmF0ZScpO1xuXHRcdHZhciBhY3RpdmF0aW9uID0gZ2V0VmFyKHRoaXMsICdhY3RpdmF0aW9uJyk7XG5cdFx0aWYgKGlzSW5wdXQpXG5cdFx0XHRpbnB1dHMucHVzaChhY3RpdmF0aW9uLmlkKTtcblx0XHRlbHNlIHtcblx0XHRcdGFjdGl2YXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0ucHVzaChzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdHRyYWNlX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdLnB1c2goc3RvcmVfdHJhY2UpO1xuXHRcdFx0cHJvcGFnYXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0ucHVzaChzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHR2YXIgb2xkID0gZ2V0VmFyKHRoaXMsICdvbGQnKTtcblx0XHRcdHZhciBzdGF0ZSA9IGdldFZhcih0aGlzLCAnc3RhdGUnKTtcblx0XHRcdHZhciBiaWFzID0gZ2V0VmFyKHRoaXMsICdiaWFzJyk7XG5cdFx0XHRpZiAodGhpcy5zZWxmY29ubmVjdGlvbi5nYXRlcilcblx0XHRcdFx0dmFyIHNlbGZfZ2FpbiA9IGdldFZhcih0aGlzLnNlbGZjb25uZWN0aW9uLCAnZ2FpbicpO1xuXHRcdFx0aWYgKHRoaXMuc2VsZmNvbm5lY3RlZCgpKVxuXHRcdFx0XHR2YXIgc2VsZl93ZWlnaHQgPSBnZXRWYXIodGhpcy5zZWxmY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuXHRcdFx0YnVpbGRTZW50ZW5jZShvbGQsICcgPSAnLCBzdGF0ZSwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRpZiAodGhpcy5zZWxmY29ubmVjdGVkKCkpXG5cdFx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0aW9uLmdhdGVyKVxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2Uoc3RhdGUsICcgPSAnLCBzZWxmX2dhaW4sICcgKiAnLCBzZWxmX3dlaWdodCwgJyAqICcsXG5cdFx0XHRcdFx0XHRzdGF0ZSwgJyArICcsIGJpYXMsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShzdGF0ZSwgJyA9ICcsIHNlbGZfd2VpZ2h0LCAnICogJywgc3RhdGUsICcgKyAnLFxuXHRcdFx0XHRcdFx0Ymlhcywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRlbHNlXG5cdFx0XHRcdGJ1aWxkU2VudGVuY2Uoc3RhdGUsICcgPSAnLCBiaWFzLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdGZvciAodmFyIGkgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cdFx0XHRcdHZhciBpbnB1dF9hY3RpdmF0aW9uID0gZ2V0VmFyKGlucHV0LmZyb20sICdhY3RpdmF0aW9uJyk7XG5cdFx0XHRcdHZhciBpbnB1dF93ZWlnaHQgPSBnZXRWYXIoaW5wdXQsICd3ZWlnaHQnKTtcblx0XHRcdFx0aWYgKGlucHV0LmdhdGVyKVxuXHRcdFx0XHRcdHZhciBpbnB1dF9nYWluID0gZ2V0VmFyKGlucHV0LCAnZ2FpbicpO1xuXHRcdFx0XHRpZiAodGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV0uZ2F0ZXIpXG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShzdGF0ZSwgJyArPSAnLCBpbnB1dF9hY3RpdmF0aW9uLCAnICogJyxcblx0XHRcdFx0XHRcdGlucHV0X3dlaWdodCwgJyAqICcsIGlucHV0X2dhaW4sIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShzdGF0ZSwgJyArPSAnLCBpbnB1dF9hY3RpdmF0aW9uLCAnICogJyxcblx0XHRcdFx0XHRcdGlucHV0X3dlaWdodCwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHR9XG5cdFx0XHR2YXIgZGVyaXZhdGl2ZSA9IGdldFZhcih0aGlzLCAnZGVyaXZhdGl2ZScpO1xuXHRcdFx0c3dpdGNoICh0aGlzLnNxdWFzaCkge1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5MT0dJU1RJQzpcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSAoMSAvICgxICsgTWF0aC5leHAoLScsIHN0YXRlLCAnKSkpJyxcblx0XHRcdFx0XHRcdHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZGVyaXZhdGl2ZSwgJyA9ICcsIGFjdGl2YXRpb24sICcgKiAoMSAtICcsXG5cdFx0XHRcdFx0XHRhY3RpdmF0aW9uLCAnKScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJyZWFrO1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5UQU5IOlxuXHRcdFx0XHRcdHZhciBlUCA9IGdldFZhcignYXV4Jyk7XG5cdFx0XHRcdFx0dmFyIGVOID0gZ2V0VmFyKCdhdXhfMicpO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZVAsICcgPSBNYXRoLmV4cCgnLCBzdGF0ZSwgJyknLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVOLCAnID0gMSAvICcsIGVQLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSAoJywgZVAsICcgLSAnLCBlTiwgJykgLyAoJywgZVAsICcgKyAnLCBlTiwgJyknLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGRlcml2YXRpdmUsICcgPSAxIC0gKCcsIGFjdGl2YXRpb24sICcgKiAnLCBhY3RpdmF0aW9uLCAnKScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJyZWFrO1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5JREVOVElUWTpcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSAnLCBzdGF0ZSwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShkZXJpdmF0aXZlLCAnID0gMScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJyZWFrO1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5ITElNOlxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoYWN0aXZhdGlvbiwgJyA9ICsoJywgc3RhdGUsICcgPiAwKScsXG5cdFx0XHRcdFx0XHRzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGRlcml2YXRpdmUsICcgPSAxJywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnJlYWs7XG5cdFx0XHR9XG5cblxuXHRcdFx0dmFyIGluZmx1ZW5jZXMgPSBbXTtcblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0Ly8gY2FsY3VsYXRlIGV4dGVuZGVkIGVsZWdpYmlsaXR5IHRyYWNlcyBpbiBhZHZhbmNlXG4gICAgICAgIFxuXHRcdFx0XHR2YXIgeHRyYWNlID0gdGhpcy50cmFjZS5leHRlbmRlZFtpZF07XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHR2YXIgaW5mbHVlbmNlID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0dmFyIG5ldXJvbl9vbGQgPSBnZXRWYXIobmV1cm9uLCAnb2xkJyk7XG5cdFx0XHRcdHZhciBpbml0aWFsaXplZCA9IGZhbHNlO1xuXHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyID09IHRoaXMpIHtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9ICcsIG5ldXJvbl9vbGQsIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRpbml0aWFsaXplZCA9IHRydWU7XG5cdFx0XHRcdH1cblx0XHRcdFx0Zm9yICh2YXIgaW5jb21pbmcgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0pIHtcblx0XHRcdFx0XHR2YXIgaW5jb21pbmdfd2VpZ2h0ID0gZ2V0VmFyKHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdXG5cdFx0XHRcdFx0W2luY29taW5nXSwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdHZhciBpbmNvbWluZ19hY3RpdmF0aW9uID0gZ2V0VmFyKHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdXG5cdFx0XHRcdFx0W2luY29taW5nXS5mcm9tLCAnYWN0aXZhdGlvbicpO1xuXG5cdFx0XHRcdFx0aWYgKGluaXRpYWxpemVkKVxuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgKz0gJywgaW5jb21pbmdfd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdFx0aW5jb21pbmdfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdGVsc2Uge1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAnLCBpbmNvbWluZ193ZWlnaHQsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRpbmNvbWluZ19hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRpbml0aWFsaXplZCA9IHRydWU7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9XG5cblx0XHRcdFx0aW5mbHVlbmNlcy5wdXNoKG5ldXJvbi5JRCk7XG5cdFx0XHRcdGJ1aWxkU2VudGVuY2UoXCJpbmZsdWVuY2VzW1wiICsgKGluZmx1ZW5jZXMubGVuZ3RoIC0gMSkgKyBcIl0gPSBcIiwgaW5mbHVlbmNlLCBzdG9yZV90cmFjZSk7XG5cdFx0XHR9XG5cblx0XHRcdGZvciAodmFyIGkgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cdFx0XHRcdGlmIChpbnB1dC5nYXRlcilcblx0XHRcdFx0XHR2YXIgaW5wdXRfZ2FpbiA9IGdldFZhcihpbnB1dCwgJ2dhaW4nKTtcblx0XHRcdFx0dmFyIGlucHV0X2FjdGl2YXRpb24gPSBnZXRWYXIoaW5wdXQuZnJvbSwgJ2FjdGl2YXRpb24nKTtcblx0XHRcdFx0dmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzLnRyYWNlXG5cdFx0XHRcdFx0LmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG5cdFx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0ZWQoKSkge1xuXHRcdFx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0aW9uLmdhdGVyKSB7XG5cdFx0XHRcdFx0XHRpZiAoaW5wdXQuZ2F0ZXIpXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBzZWxmX2dhaW4sICcgKiAnLCBzZWxmX3dlaWdodCxcblx0XHRcdFx0XHRcdFx0XHQnICogJywgdHJhY2UsICcgKyAnLCBpbnB1dF9nYWluLCAnICogJywgaW5wdXRfYWN0aXZhdGlvbixcblx0XHRcdFx0XHRcdFx0XHRzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBzZWxmX2dhaW4sICcgKiAnLCBzZWxmX3dlaWdodCxcblx0XHRcdFx0XHRcdFx0XHQnICogJywgdHJhY2UsICcgKyAnLCBpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0XHRcdGlmIChpbnB1dC5nYXRlcilcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZSh0cmFjZSwgJyA9ICcsIHNlbGZfd2VpZ2h0LCAnICogJywgdHJhY2UsICcgKyAnLFxuXHRcdFx0XHRcdFx0XHRcdGlucHV0X2dhaW4sICcgKiAnLCBpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBzZWxmX3dlaWdodCwgJyAqICcsIHRyYWNlLCAnICsgJyxcblx0XHRcdFx0XHRcdFx0XHRpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRcdGlmIChpbnB1dC5nYXRlcilcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBpbnB1dF9nYWluLCAnICogJywgaW5wdXRfYWN0aXZhdGlvbixcblx0XHRcdFx0XHRcdFx0c3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdH1cblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHRcdC8vIGV4dGVuZGVkIGVsZWdpYmlsaXR5IHRyYWNlXG5cdFx0XHRcdFx0dmFyIHh0cmFjZSA9IHRoaXMudHJhY2UuZXh0ZW5kZWRbaWRdO1xuXHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHRcdHZhciBpbmZsdWVuY2UgPSBnZXRWYXIoJ2F1eCcpO1xuXHRcdFx0XHRcdHZhciBuZXVyb25fb2xkID0gZ2V0VmFyKG5ldXJvbiwgJ29sZCcpO1xuXG5cdFx0XHRcdFx0dmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzLnRyYWNlXG5cdFx0XHRcdFx0XHQuZWxlZ2liaWxpdHlbaW5wdXQuSURdKTtcblx0XHRcdFx0XHR2YXIgeHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdleHRlbmRlZCcsIG5ldXJvbi5JRCwgaW5wdXQuSUQsXG5cdFx0XHRcdFx0XHR0aGlzLnRyYWNlLmV4dGVuZGVkW25ldXJvbi5JRF1baW5wdXQuSURdKTtcblx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0ZWQoKSlcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fc2VsZl93ZWlnaHQgPSBnZXRWYXIobmV1cm9uLnNlbGZjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlcilcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fc2VsZl9nYWluID0gZ2V0VmFyKG5ldXJvbi5zZWxmY29ubmVjdGlvbiwgJ2dhaW4nKTtcblx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0ZWQoKSlcblx0XHRcdFx0XHRcdGlmIChuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIpXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoeHRyYWNlLCAnID0gJywgbmV1cm9uX3NlbGZfZ2FpbiwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0bmV1cm9uX3NlbGZfd2VpZ2h0LCAnICogJywgeHRyYWNlLCAnICsgJywgZGVyaXZhdGl2ZSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0dHJhY2UsICcgKiAnLCBcImluZmx1ZW5jZXNbXCIgKyBpbmZsdWVuY2VzLmluZGV4T2YobmV1cm9uLklEKSArIFwiXVwiLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoeHRyYWNlLCAnID0gJywgbmV1cm9uX3NlbGZfd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHR4dHJhY2UsICcgKyAnLCBkZXJpdmF0aXZlLCAnICogJywgdHJhY2UsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdFwiaW5mbHVlbmNlc1tcIiArIGluZmx1ZW5jZXMuaW5kZXhPZihuZXVyb24uSUQpICsgXCJdXCIsIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHh0cmFjZSwgJyA9ICcsIGRlcml2YXRpdmUsICcgKiAnLCB0cmFjZSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFwiaW5mbHVlbmNlc1tcIiArIGluZmx1ZW5jZXMuaW5kZXhPZihuZXVyb24uSUQpICsgXCJdXCIsIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdFx0Zm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zLmdhdGVkKSB7XG5cdFx0XHRcdHZhciBnYXRlZF9nYWluID0gZ2V0VmFyKHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWRbY29ubmVjdGlvbl0sICdnYWluJyk7XG5cdFx0XHRcdGJ1aWxkU2VudGVuY2UoZ2F0ZWRfZ2FpbiwgJyA9ICcsIGFjdGl2YXRpb24sIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0fVxuXHRcdH1cblx0XHRpZiAoIWlzSW5wdXQpIHtcblx0XHRcdHZhciByZXNwb25zaWJpbGl0eSA9IGdldFZhcih0aGlzLCAnZXJyb3InLCAncmVzcG9uc2liaWxpdHknLCB0aGlzLmVycm9yXG5cdFx0XHRcdC5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRpZiAoaXNPdXRwdXQpIHtcblx0XHRcdFx0dmFyIHRhcmdldCA9IGdldFZhcigndGFyZ2V0Jyk7XG5cdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgPSAnLCB0YXJnZXQsICcgLSAnLCBhY3RpdmF0aW9uLFxuXHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0XHR2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cdFx0XHRcdFx0dmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzLnRyYWNlXG5cdFx0XHRcdFx0XHQuZWxlZ2liaWxpdHlbaW5wdXQuSURdKTtcblx0XHRcdFx0XHR2YXIgaW5wdXRfd2VpZ2h0ID0gZ2V0VmFyKGlucHV0LCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbnB1dF93ZWlnaHQsICcgKz0gJywgcmF0ZSwgJyAqICgnLCByZXNwb25zaWJpbGl0eSxcblx0XHRcdFx0XHRcdCcgKiAnLCB0cmFjZSwgJyknLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdH1cblx0XHRcdFx0b3V0cHV0cy5wdXNoKGFjdGl2YXRpb24uaWQpO1xuXHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0aWYgKCFub1Byb2plY3Rpb25zICYmICFub0dhdGVzKSB7XG5cdFx0XHRcdFx0dmFyIGVycm9yID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZCkge1xuXHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZFtpZF07XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gY29ubmVjdGlvbi50bztcblx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX3dlaWdodCA9IGdldFZhcihjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcblx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdGlmIChjb25uZWN0aW9uLmdhdGVyKSB7XG5cdFx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX2dhaW4gPSBnZXRWYXIoY29ubmVjdGlvbiwgJ2dhaW4nKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShlcnJvciwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdGNvbm5lY3Rpb25fZ2FpbiwgJyAqICcsIGNvbm5lY3Rpb25fd2VpZ2h0LFxuXHRcdFx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdH0gZWxzZVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVycm9yLCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0Y29ubmVjdGlvbl93ZWlnaHQsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0dmFyIHByb2plY3RlZCA9IGdldFZhcih0aGlzLCAnZXJyb3InLCAncHJvamVjdGVkJywgdGhpcy5lcnJvci5wcm9qZWN0ZWQpO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocHJvamVjdGVkLCAnID0gJywgZGVyaXZhdGl2ZSwgJyAqICcsIGVycm9yLFxuXHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZXJyb3IsICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHRcdFx0dmFyIGluZmx1ZW5jZSA9IGdldFZhcignYXV4XzInKTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fb2xkID0gZ2V0VmFyKG5ldXJvbiwgJ29sZCcpO1xuXHRcdFx0XHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzKVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9ICcsIG5ldXJvbl9vbGQsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0Zm9yICh2YXIgaW5mbHVlbmNlSW5wdXQgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0pIHtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXVtpbmZsdWVuY2VJbnB1dF07XG5cdFx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX3dlaWdodCA9IGdldFZhcihjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHRcdHZhciBuZXVyb25fYWN0aXZhdGlvbiA9IGdldFZhcihjb25uZWN0aW9uLmZyb20sICdhY3RpdmF0aW9uJyk7XG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnICs9ICcsIGNvbm5lY3Rpb25fd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHRuZXVyb25fYWN0aXZhdGlvbiwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG5cdFx0XHRcdFx0XHRcdCdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVycm9yLCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdGluZmx1ZW5jZSwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHR2YXIgZ2F0ZWQgPSBnZXRWYXIodGhpcywgJ2Vycm9yJywgJ2dhdGVkJywgdGhpcy5lcnJvci5nYXRlZCk7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShnYXRlZCwgJyA9ICcsIGRlcml2YXRpdmUsICcgKiAnLCBlcnJvcixcblx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnID0gJywgcHJvamVjdGVkLCAnICsgJywgZ2F0ZWQsXG5cdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0XHRcdHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcblx0XHRcdFx0XHRcdHZhciBncmFkaWVudCA9IGdldFZhcignYXV4Jyk7XG5cdFx0XHRcdFx0XHR2YXIgdHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2VsZWdpYmlsaXR5JywgaW5wdXQuSUQsIHRoaXNcblx0XHRcdFx0XHRcdFx0LnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGdyYWRpZW50LCAnID0gJywgcHJvamVjdGVkLCAnICogJywgdHJhY2UsXG5cdFx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdFx0XHRcdHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuXHRcdFx0XHRcdFx0XHRcdCdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRcdFx0XHRcdHZhciB4dHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2V4dGVuZGVkJywgbmV1cm9uLklELFxuXHRcdFx0XHRcdFx0XHRcdGlucHV0LklELCB0aGlzLnRyYWNlLmV4dGVuZGVkW25ldXJvbi5JRF1baW5wdXQuSURdKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShncmFkaWVudCwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdHh0cmFjZSwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0dmFyIGlucHV0X3dlaWdodCA9IGdldFZhcihpbnB1dCwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbnB1dF93ZWlnaHQsICcgKz0gJywgcmF0ZSwgJyAqICcsIGdyYWRpZW50LFxuXHRcdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXG5cdFx0XHRcdH0gZWxzZSBpZiAobm9HYXRlcykge1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG5cdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkW2lkXTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb24gPSBjb25uZWN0aW9uLnRvO1xuXHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fd2VpZ2h0ID0gZ2V0VmFyKGNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuXHRcdFx0XHRcdFx0XHQncmVzcG9uc2liaWxpdHknLCBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkpO1xuXHRcdFx0XHRcdFx0aWYgKGNvbm5lY3Rpb24uZ2F0ZXIpIHtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fZ2FpbiA9IGdldFZhcihjb25uZWN0aW9uLCAnZ2FpbicpO1xuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSxcblx0XHRcdFx0XHRcdFx0XHQnICogJywgY29ubmVjdGlvbl9nYWluLCAnICogJywgY29ubmVjdGlvbl93ZWlnaHQsXG5cdFx0XHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fSBlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LFxuXHRcdFx0XHRcdFx0XHRcdCcgKiAnLCBjb25uZWN0aW9uX3dlaWdodCwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICo9ICcsIGRlcml2YXRpdmUsXG5cdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0XHRcdHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcblx0XHRcdFx0XHRcdHZhciB0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZWxlZ2liaWxpdHknLCBpbnB1dC5JRCwgdGhpc1xuXHRcdFx0XHRcdFx0XHQudHJhY2UuZWxlZ2liaWxpdHlbaW5wdXQuSURdKTtcblx0XHRcdFx0XHRcdHZhciBpbnB1dF93ZWlnaHQgPSBnZXRWYXIoaW5wdXQsICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5wdXRfd2VpZ2h0LCAnICs9ICcsIHJhdGUsICcgKiAoJyxcblx0XHRcdFx0XHRcdFx0cmVzcG9uc2liaWxpdHksICcgKiAnLCB0cmFjZSwgJyknLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9IGVsc2UgaWYgKG5vUHJvamVjdGlvbnMpIHtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0XHRcdHZhciBpbmZsdWVuY2UgPSBnZXRWYXIoJ2F1eCcpO1xuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9vbGQgPSBnZXRWYXIobmV1cm9uLCAnb2xkJyk7XG5cdFx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyID09IHRoaXMpXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnID0gJywgbmV1cm9uX29sZCwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9IDAnLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHRmb3IgKHZhciBpbmZsdWVuY2VJbnB1dCBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXSkge1xuXHRcdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdW2luZmx1ZW5jZUlucHV0XTtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fd2VpZ2h0ID0gZ2V0VmFyKGNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9hY3RpdmF0aW9uID0gZ2V0VmFyKGNvbm5lY3Rpb24uZnJvbSwgJ2FjdGl2YXRpb24nKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgKz0gJywgY29ubmVjdGlvbl93ZWlnaHQsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdG5ldXJvbl9hY3RpdmF0aW9uLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcblx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LFxuXHRcdFx0XHRcdFx0XHQnICogJywgaW5mbHVlbmNlLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgKj0gJywgZGVyaXZhdGl2ZSxcblx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaWRdO1xuXHRcdFx0XHRcdFx0dmFyIGdyYWRpZW50ID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZ3JhZGllbnQsICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG5cdFx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdFx0dmFyIHh0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZXh0ZW5kZWQnLCBuZXVyb24uSUQsXG5cdFx0XHRcdFx0XHRcdFx0aW5wdXQuSUQsIHRoaXMudHJhY2UuZXh0ZW5kZWRbbmV1cm9uLklEXVtpbnB1dC5JRF0pO1xuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGdyYWRpZW50LCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0eHRyYWNlLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHR2YXIgaW5wdXRfd2VpZ2h0ID0gZ2V0VmFyKGlucHV0LCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGlucHV0X3dlaWdodCwgJyArPSAnLCByYXRlLCAnICogJywgZ3JhZGllbnQsXG5cdFx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHRcdGJ1aWxkU2VudGVuY2UoYmlhcywgJyArPSAnLCByYXRlLCAnICogJywgcmVzcG9uc2liaWxpdHksXG5cdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHR9XG5cdFx0cmV0dXJuIHtcblx0XHRcdG1lbW9yeTogdmFySUQsXG5cdFx0XHRuZXVyb25zOiBuZXVyb25zICsgMSxcblx0XHRcdGlucHV0czogaW5wdXRzLFxuXHRcdFx0b3V0cHV0czogb3V0cHV0cyxcblx0XHRcdHRhcmdldHM6IHRhcmdldHMsXG5cdFx0XHR2YXJpYWJsZXM6IHZhcmlhYmxlcyxcblx0XHRcdGFjdGl2YXRpb25fc2VudGVuY2VzOiBhY3RpdmF0aW9uX3NlbnRlbmNlcyxcblx0XHRcdHRyYWNlX3NlbnRlbmNlczogdHJhY2Vfc2VudGVuY2VzLFxuXHRcdFx0cHJvcGFnYXRpb25fc2VudGVuY2VzOiBwcm9wYWdhdGlvbl9zZW50ZW5jZXMsXG5cdFx0XHRsYXllcnM6IGxheWVyc1xuXHRcdH1cblx0fVxufVxuXG5leHBvcnQgbW9kdWxlIE5ldXJvbiB7XG5cblx0ZXhwb3J0IGludGVyZmFjZSBJTmV1cm9uQ29ubmVjdGlvbnMge1xuXHRcdGlucHV0czogU3luYXB0aWMuRGljdGlvbmFyeTxOZXVyb24uQ29ubmVjdGlvbj47XG5cdFx0cHJvamVjdGVkOiB7fTtcblx0XHRnYXRlZDoge307XG5cdH1cblxuXHRleHBvcnQgY2xhc3MgQ29ubmVjdGlvbiB7XG5cdFx0SUQgPSBDb25uZWN0aW9uLnVpZCgpO1xuXHRcdGZyb207XG5cdFx0dG87XG5cdFx0Z2FpbjogbnVtYmVyID0gMTtcblx0XHR3ZWlnaHQ6IG51bWJlciA9IDA7XG5cdFx0Z2F0ZXI6IGFueSA9IG51bGw7XG5cdFx0Y29uc3RydWN0b3IoZnJvbSwgdG8sIHdlaWdodD86IG51bWJlcikge1xuXHRcdFx0aWYgKCFmcm9tIHx8ICF0bylcblx0XHRcdFx0dGhyb3cgXCJDb25uZWN0aW9uIEVycm9yOiBJbnZhbGlkIG5ldXJvbnNcIjtcblx0XHRcdHRoaXMuZnJvbSA9IGZyb207XG5cdFx0XHR0aGlzLnRvID0gdG87XG5cdFx0XHR0aGlzLndlaWdodCA9IHR5cGVvZiB3ZWlnaHQgPT0gJ3VuZGVmaW5lZCcgfHwgaXNOYU4od2VpZ2h0KSA/IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xIDpcblx0XHRcdFx0d2VpZ2h0O1xuXHRcdH1cblx0fVxuXG5cdGV4cG9ydCB2YXIgbmV1cm9uUXR5ID0gMDtcblx0ZXhwb3J0IGZ1bmN0aW9uIHVpZCgpOiBudW1iZXIge1xuXHRcdHJldHVybiBuZXVyb25RdHkrKztcblx0fVxuXG5cdGV4cG9ydCBmdW5jdGlvbiBxdWFudGl0eSgpIHtcblx0XHRyZXR1cm4ge1xuXHRcdFx0bmV1cm9uczogbmV1cm9uUXR5LFxuXHRcdFx0Y29ubmVjdGlvbnM6IENvbm5lY3Rpb24uY29ubmVjdGlvblF0eVxuXHRcdH1cblx0fVxufVxuXG5leHBvcnQgbW9kdWxlIE5ldXJvbi5Db25uZWN0aW9uIHtcblx0ZXhwb3J0IHZhciBjb25uZWN0aW9uUXR5ID0gMDtcblx0ZXhwb3J0IGZ1bmN0aW9uIHVpZCgpOiBudW1iZXIge1xuXHRcdHJldHVybiBjb25uZWN0aW9uUXR5Kys7XG5cdH1cbn0iLCJpbXBvcnQgU3luYXB0aWMgPSByZXF1aXJlKCcuL3N5bmFwdGljJyk7XG5cbi8vIHNxdWFzaGluZyBmdW5jdGlvbnNcblxuZXhwb3J0IGZ1bmN0aW9uIExPR0lTVElDKHg6IG51bWJlciwgZGVyaXZhdGU/OiBib29sZWFuKSB7XG5cdGlmICghZGVyaXZhdGUpXG5cdFx0cmV0dXJuIDEgLyAoMSArIE1hdGguZXhwKC14KSk7XG5cdHZhciBmeCA9IExPR0lTVElDKHgpO1xuXHRyZXR1cm4gZnggKiAoMSAtIGZ4KTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIFRBTkgoeDogbnVtYmVyLCBkZXJpdmF0ZT86IGJvb2xlYW4pIHtcblx0aWYgKGRlcml2YXRlKVxuXHRcdHJldHVybiAxIC0gTWF0aC5wb3coVEFOSCh4KSwgMik7XG5cdHZhciBlUCA9IE1hdGguZXhwKHgpO1xuXHR2YXIgZU4gPSAxIC8gZVA7XG5cdHJldHVybiAoZVAgLSBlTikgLyAoZVAgKyBlTik7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBJREVOVElUWSh4OiBudW1iZXIsIGRlcml2YXRlPzogYm9vbGVhbikge1xuXHRyZXR1cm4gZGVyaXZhdGUgPyAxIDogeDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIEhMSU0oeDogbnVtYmVyLCBkZXJpdmF0ZT86IGJvb2xlYW4pIHtcblx0cmV0dXJuIGRlcml2YXRlID8gMSA6ICsoeCA+IDApO1xufVxuIiwiLypcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFNZTkFQVElDXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxuXG5TeW5hcHRpYyBpcyBhIGphdmFzY3JpcHQgbmV1cmFsIG5ldHdvcmsgbGlicmFyeSBmb3Igbm9kZS5qcyBhbmQgdGhlIGJyb3dzZXIsIGl0cyBnZW5lcmFsaXplZFxuYWxnb3JpdGhtIGlzIGFyY2hpdGVjdHVyZS1mcmVlLCBzbyB5b3UgY2FuIGJ1aWxkIGFuZCB0cmFpbiBiYXNpY2FsbHkgYW55IHR5cGUgb2YgZmlyc3Qgb3JkZXJcbm9yIGV2ZW4gc2Vjb25kIG9yZGVyIG5ldXJhbCBuZXR3b3JrIGFyY2hpdGVjdHVyZXMuXG5cbmh0dHA6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvUmVjdXJyZW50X25ldXJhbF9uZXR3b3JrI1NlY29uZF9PcmRlcl9SZWN1cnJlbnRfTmV1cmFsX05ldHdvcmtcblxuVGhlIGxpYnJhcnkgaW5jbHVkZXMgYSBmZXcgYnVpbHQtaW4gYXJjaGl0ZWN0dXJlcyBsaWtlIG11bHRpbGF5ZXIgcGVyY2VwdHJvbnMsIG11bHRpbGF5ZXJcbmxvbmctc2hvcnQgdGVybSBtZW1vcnkgbmV0d29ya3MgKExTVE0pIG9yIGxpcXVpZCBzdGF0ZSBtYWNoaW5lcywgYW5kIGEgdHJhaW5lciBjYXBhYmxlIG9mXG50cmFpbmluZyBhbnkgZ2l2ZW4gbmV0d29yaywgYW5kIGluY2x1ZGVzIGJ1aWx0LWluIHRyYWluaW5nIHRhc2tzL3Rlc3RzIGxpa2Ugc29sdmluZyBhbiBYT1IsXG5wYXNzaW5nIGEgRGlzdHJhY3RlZCBTZXF1ZW5jZSBSZWNhbGwgdGVzdCBvciBhbiBFbWJlZGVkIFJlYmVyIEdyYW1tYXIgdGVzdC5cblxuVGhlIGFsZ29yaXRobSBpbXBsZW1lbnRlZCBieSB0aGlzIGxpYnJhcnkgaGFzIGJlZW4gdGFrZW4gZnJvbSBEZXJlayBELiBNb25uZXIncyBwYXBlcjpcblxuQSBnZW5lcmFsaXplZCBMU1RNLWxpa2UgdHJhaW5pbmcgYWxnb3JpdGhtIGZvciBzZWNvbmQtb3JkZXIgcmVjdXJyZW50IG5ldXJhbCBuZXR3b3Jrc1xuaHR0cDovL3d3dy5vdmVyY29tcGxldGUubmV0L3BhcGVycy9ubjIwMTIucGRmXG5cblRoZXJlIGFyZSByZWZlcmVuY2VzIHRvIHRoZSBlcXVhdGlvbnMgaW4gdGhhdCBwYXBlciBjb21tZW50ZWQgdGhyb3VnaCB0aGUgc291cmNlIGNvZGUuXG5cblxuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cblxuXG5pbXBvcnQgbmV0d29yayA9IHJlcXVpcmUoJy4vbmV0d29yaycpO1xuaW1wb3J0IGxheWVyID0gcmVxdWlyZSgnLi9sYXllcicpO1xuaW1wb3J0IG5ldXJvbiA9IHJlcXVpcmUoJy4vbmV1cm9uJyk7XG5pbXBvcnQgdHJhaW5lciA9IHJlcXVpcmUoJy4vdHJhaW5lcicpO1xuaW1wb3J0IGFyY2hpdGVjdCA9IHJlcXVpcmUoJy4vYXJjaGl0ZWN0Jyk7XG5pbXBvcnQgc3F1YXNoID0gcmVxdWlyZSgnLi9zcXVhc2gnKTtcblxuZGVjbGFyZSB2YXIgd2luZG93O1xuXG5tb2R1bGUgU3luYXB0aWMge1xuXHRleHBvcnQgaW50ZXJmYWNlIERpY3Rpb25hcnk8VD4ge1xuXHRcdFtpZDogc3RyaW5nXSA6IFQ7XG5cdH1cblx0XG5cdHZhciBvbGRTeW5hcHRpYyA9IHdpbmRvdyAmJiB3aW5kb3dbJ1N5bmFwdGljJ107XG5cdFxuXHRleHBvcnQgZnVuY3Rpb24gbmluamEoKSB7XG4gICAgICB3aW5kb3dbJ3N5bmFwdGljJ10gPSBvbGRTeW5hcHRpYzsgXG4gICAgICByZXR1cm4gU3luYXB0aWM7XG5cdH1cblx0XG5cdGV4cG9ydCBpbnRlcmZhY2UgSUNvbXBpbGVkUGFyYW1ldGVycyB7XHRcblx0XHRtZW1vcnk/OiBhbnk7XG5cdFx0bmV1cm9ucz86IG51bWJlcjtcblx0XHRpbnB1dHM/OiBhbnlbXTtcblx0XHRvdXRwdXRzPzogYW55W107XG5cdFx0dGFyZ2V0cz86IGFueVtdO1xuXHRcdHZhcmlhYmxlcz86IGFueTtcblx0XHRhY3RpdmF0aW9uX3NlbnRlbmNlcz86IGFueVtdO1xuXHRcdHRyYWNlX3NlbnRlbmNlcz86IGFueVtdO1xuXHRcdHByb3BhZ2F0aW9uX3NlbnRlbmNlcz86IGFueVtdO1xuXHRcdGxheWVycz86IGFueTtcblx0fVxuXHRcblx0ZXhwb3J0IHZhciBOZXVyb24gPSBuZXVyb24uTmV1cm9uO1xuXHRleHBvcnQgdmFyIExheWVyID0gbGF5ZXIuTGF5ZXI7XG5cdGV4cG9ydCB2YXIgTmV0d29yayA9IG5ldHdvcmsuTmV0d29yaztcblx0ZXhwb3J0IHZhciBUcmFpbmVyID0gdHJhaW5lci5UcmFpbmVyO1xuXHRleHBvcnQgdmFyIFNxdWFzaCA9IHNxdWFzaDtcblx0ZXhwb3J0IHZhciBBcmNoaXRlY3QgPSBhcmNoaXRlY3Q7XG59XG5cbmV4cG9ydCA9IFN5bmFwdGljO1xuXG5pZih0eXBlb2Ygd2luZG93ICE9IFwidW5kZWZpbmVkXCIpIFxuXHR3aW5kb3dbJ3N5bmFwdGljJ10gPSBTeW5hcHRpYztcbiIsImltcG9ydCBuZXQgPXJlcXVpcmUoJy4vbmV0d29yaycpO1xuXG4vKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFRSQUlORVJcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cbmV4cG9ydCBjbGFzcyBUcmFpbmVyIHtcbiAgbmV0d29yazogbmV0Lk5ldHdvcms7XG4gIHJhdGU6IGFueSA9IC4yO1xuICBpdGVyYXRpb25zID0gMTAwMDAwO1xuICBlcnJvciA9IC4wMDU7XG4gIGNvc3Q6IFRyYWluZXIuSVRyYWluZXJDb3N0Rm47XG4gIHNjaGVkdWxlOiBhbnk7XG5cbiAgY29uc3RydWN0b3IobmV0d29yazogbmV0Lk5ldHdvcmssIG9wdGlvbnM/OiBhbnkpIHtcbiAgICBvcHRpb25zID0gb3B0aW9ucyB8fCB7fTtcbiAgICB0aGlzLm5ldHdvcmsgPSBuZXR3b3JrO1xuICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZSB8fCAuMjtcbiAgICB0aGlzLml0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnMgfHwgMTAwMDAwO1xuICAgIHRoaXMuZXJyb3IgPSBvcHRpb25zLmVycm9yIHx8IC4wMDVcbiAgICB0aGlzLmNvc3QgPSBvcHRpb25zLmNvc3QgfHwgVHJhaW5lci5jb3N0LkNST1NTX0VOVFJPUFk7XG4gIH1cblxuICAvLyB0cmFpbnMgYW55IGdpdmVuIHNldCB0byBhIG5ldHdvcmtcbiAgdHJhaW4oc2V0LCBvcHRpb25zKSB7XG5cbiAgICB2YXIgZXJyb3IgPSAxO1xuICAgIHZhciBpdGVyYXRpb25zID0gMCwgYnVja2V0U2l6ZSA9IDA7XG4gICAgdmFyIGFib3J0X3RyYWluaW5nID0gZmFsc2U7XG4gICAgdmFyIGlucHV0LCBvdXRwdXQsIHRhcmdldCwgY3VycmVudFJhdGU7XG5cbiAgICB2YXIgc3RhcnQgPSBEYXRlLm5vdygpO1xuXG4gICAgaWYgKG9wdGlvbnMpIHtcbiAgICAgIGlmIChvcHRpb25zLnNodWZmbGUpIHtcbiAgICAgICAgLy8rIEpvbmFzIFJhb25pIFNvYXJlcyBTaWx2YVxuICAgICAgICAvL0AgaHR0cDovL2pzZnJvbWhlbGwuY29tL2FycmF5L3NodWZmbGUgW3YxLjBdXG4gICAgICAgIGZ1bmN0aW9uIHNodWZmbGUobykgeyAvL3YxLjBcbiAgICAgICAgICBmb3IgKHZhciBqLCB4LCBpID0gby5sZW5ndGg7IGk7IGogPSBNYXRoLmZsb29yKE1hdGgucmFuZG9tKCkgKiBpKSwgeCA9IG9bLS1pXSwgb1tpXSA9IG9bal0sIG9bal0gPSB4KTtcbiAgICAgICAgICByZXR1cm4gbztcbiAgICAgICAgfTtcbiAgICAgIH1cbiAgICAgIGlmIChvcHRpb25zLml0ZXJhdGlvbnMpXG4gICAgICAgIHRoaXMuaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucztcbiAgICAgIGlmIChvcHRpb25zLmVycm9yKVxuICAgICAgICB0aGlzLmVycm9yID0gb3B0aW9ucy5lcnJvcjtcbiAgICAgIGlmIChvcHRpb25zLnJhdGUpXG4gICAgICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZTtcbiAgICAgIGlmIChvcHRpb25zLmNvc3QpXG4gICAgICAgIHRoaXMuY29zdCA9IG9wdGlvbnMuY29zdDtcbiAgICAgIGlmIChvcHRpb25zLnNjaGVkdWxlKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5zY2hlZHVsZTtcbiAgICAgIGlmIChvcHRpb25zLmN1c3RvbUxvZykge1xuICAgICAgICAvLyBmb3IgYmFja3dhcmQgY29tcGF0aWJpbGl0eSB3aXRoIGNvZGUgdGhhdCB1c2VkIGN1c3RvbUxvZ1xuICAgICAgICBjb25zb2xlLmxvZygnRGVwcmVjYXRlZDogdXNlIHNjaGVkdWxlIGluc3RlYWQgb2YgY3VzdG9tTG9nJylcbiAgICAgICAgdGhpcy5zY2hlZHVsZSA9IG9wdGlvbnMuY3VzdG9tTG9nO1xuICAgICAgfVxuICAgIH1cblxuICAgIGN1cnJlbnRSYXRlID0gdGhpcy5yYXRlO1xuICAgIGlmIChBcnJheS5pc0FycmF5KHRoaXMucmF0ZSkpIHtcbiAgICAgIGJ1Y2tldFNpemUgPSBNYXRoLmZsb29yKHRoaXMuaXRlcmF0aW9ucyAvIHRoaXMucmF0ZS5sZW5ndGgpO1xuICAgIH1cblxuXG4gICAgd2hpbGUgKCFhYm9ydF90cmFpbmluZyAmJiBpdGVyYXRpb25zIDwgdGhpcy5pdGVyYXRpb25zICYmIGVycm9yID4gdGhpcy5lcnJvcikge1xuICAgICAgZXJyb3IgPSAwO1xuXG4gICAgICBpZiAoYnVja2V0U2l6ZSA+IDApIHtcbiAgICAgICAgdmFyIGN1cnJlbnRCdWNrZXQgPSBNYXRoLmZsb29yKGl0ZXJhdGlvbnMgLyBidWNrZXRTaXplKTtcbiAgICAgICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGVbY3VycmVudEJ1Y2tldF07XG4gICAgICB9XG5cbiAgICAgIGZvciAodmFyIHRyYWluIGluIHNldCkge1xuICAgICAgICBpbnB1dCA9IHNldFt0cmFpbl0uaW5wdXQ7XG4gICAgICAgIHRhcmdldCA9IHNldFt0cmFpbl0ub3V0cHV0O1xuXG4gICAgICAgIG91dHB1dCA9IHRoaXMubmV0d29yay5hY3RpdmF0ZShpbnB1dCk7XG4gICAgICAgIHRoaXMubmV0d29yay5wcm9wYWdhdGUoY3VycmVudFJhdGUsIHRhcmdldCk7XG5cbiAgICAgICAgZXJyb3IgKz0gdGhpcy5jb3N0KHRhcmdldCwgb3V0cHV0KTtcbiAgICAgIH1cblxuICAgICAgLy8gY2hlY2sgZXJyb3JcbiAgICAgIGl0ZXJhdGlvbnMrKztcbiAgICAgIGVycm9yIC89IHNldC5sZW5ndGg7XG5cbiAgICAgIGlmIChvcHRpb25zKSB7XG4gICAgICAgIGlmICh0aGlzLnNjaGVkdWxlICYmIHRoaXMuc2NoZWR1bGUuZXZlcnkgJiYgaXRlcmF0aW9ucyAlXG4gICAgICAgICAgdGhpcy5zY2hlZHVsZS5ldmVyeSA9PSAwKVxuICAgICAgICAgIGFib3J0X3RyYWluaW5nID0gdGhpcy5zY2hlZHVsZS5kbyh7XG4gICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgICAgICAgcmF0ZTogY3VycmVudFJhdGVcbiAgICAgICAgICB9KTtcbiAgICAgICAgZWxzZSBpZiAob3B0aW9ucy5sb2cgJiYgaXRlcmF0aW9ucyAlIG9wdGlvbnMubG9nID09IDApIHtcbiAgICAgICAgICBjb25zb2xlLmxvZygnaXRlcmF0aW9ucycsIGl0ZXJhdGlvbnMsICdlcnJvcicsIGVycm9yLCAncmF0ZScsIGN1cnJlbnRSYXRlKTtcbiAgICAgICAgfTtcbiAgICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSlcbiAgICAgICAgICBzaHVmZmxlKHNldCk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdmFyIHJlc3VsdHMgPSB7XG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0XG4gICAgfVxuXG4gICAgcmV0dXJuIHJlc3VsdHM7XG4gIH1cblxuICAvLyB0cmFpbnMgYW55IGdpdmVuIHNldCB0byBhIG5ldHdvcmsgdXNpbmcgYSBXZWJXb3JrZXJcbiAgd29ya2VyVHJhaW4oc2V0LCBjYWxsYmFjaywgb3B0aW9ucykge1xuXG4gICAgdmFyIHRoYXQgPSB0aGlzO1xuICAgIHZhciBlcnJvciA9IDE7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSAwLCBidWNrZXRTaXplID0gMDtcbiAgICB2YXIgaW5wdXQsIG91dHB1dCwgdGFyZ2V0LCBjdXJyZW50UmF0ZTtcbiAgICB2YXIgbGVuZ3RoID0gc2V0Lmxlbmd0aDtcbiAgICB2YXIgYWJvcnRfdHJhaW5pbmcgPSBmYWxzZTtcblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICBpZiAob3B0aW9ucykge1xuICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSkge1xuICAgICAgICAvLysgSm9uYXMgUmFvbmkgU29hcmVzIFNpbHZhXG4gICAgICAgIC8vQCBodHRwOi8vanNmcm9taGVsbC5jb20vYXJyYXkvc2h1ZmZsZSBbdjEuMF1cbiAgICAgICAgZnVuY3Rpb24gc2h1ZmZsZShvKSB7IC8vdjEuMFxuICAgICAgICAgIGZvciAodmFyIGosIHgsIGkgPSBvLmxlbmd0aDsgaTsgaiA9IE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqXG4gICAgICAgICAgICBpKSwgeCA9IG9bLS1pXSwgb1tpXSA9IG9bal0sIG9bal0gPSB4KTtcbiAgICAgICAgICByZXR1cm4gbztcbiAgICAgICAgfTtcbiAgICAgIH1cbiAgICAgIGlmIChvcHRpb25zLml0ZXJhdGlvbnMpXG4gICAgICAgIHRoaXMuaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucztcbiAgICAgIGlmIChvcHRpb25zLmVycm9yKVxuICAgICAgICB0aGlzLmVycm9yID0gb3B0aW9ucy5lcnJvcjtcbiAgICAgIGlmIChvcHRpb25zLnJhdGUpXG4gICAgICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZTtcbiAgICAgIGlmIChvcHRpb25zLmNvc3QpXG4gICAgICAgIHRoaXMuY29zdCA9IG9wdGlvbnMuY29zdDtcbiAgICAgIGlmIChvcHRpb25zLnNjaGVkdWxlKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5zY2hlZHVsZTtcbiAgICAgIGlmIChvcHRpb25zLmN1c3RvbUxvZylcbiAgICAgICAgLy8gZm9yIGJhY2t3YXJkIGNvbXBhdGliaWxpdHkgd2l0aCBjb2RlIHRoYXQgdXNlZCBjdXN0b21Mb2dcbiAgICAgICAgY29uc29sZS5sb2coJ0RlcHJlY2F0ZWQ6IHVzZSBzY2hlZHVsZSBpbnN0ZWFkIG9mIGN1c3RvbUxvZycpXG4gICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5jdXN0b21Mb2c7XG4gICAgfVxuXG4gICAgLy8gZHluYW1pYyBsZWFybmluZyByYXRlXG4gICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGU7XG4gICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5yYXRlKSkge1xuICAgICAgYnVja2V0U2l6ZSA9IE1hdGguZmxvb3IodGhpcy5pdGVyYXRpb25zIC8gdGhpcy5yYXRlLmxlbmd0aCk7XG4gICAgfVxuXG4gICAgLy8gY3JlYXRlIGEgd29ya2VyXG4gICAgdmFyIHdvcmtlciA9IHRoaXMubmV0d29yay53b3JrZXIoKTtcblxuICAgIC8vIGFjdGl2YXRlIHRoZSBuZXR3b3JrXG4gICAgZnVuY3Rpb24gYWN0aXZhdGVXb3JrZXIoaW5wdXQpIHtcbiAgICAgIHdvcmtlci5wb3N0TWVzc2FnZSh7XG4gICAgICAgIGFjdGlvbjogXCJhY3RpdmF0ZVwiLFxuICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgIG1lbW9yeUJ1ZmZlcjogdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnlcbiAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyBiYWNrcHJvcGFnYXRlIHRoZSBuZXR3b3JrXG4gICAgZnVuY3Rpb24gcHJvcGFnYXRlV29ya2VyKHRhcmdldCkge1xuICAgICAgaWYgKGJ1Y2tldFNpemUgPiAwKSB7XG4gICAgICAgIHZhciBjdXJyZW50QnVja2V0ID0gTWF0aC5mbG9vcihpdGVyYXRpb25zIC8gYnVja2V0U2l6ZSk7XG4gICAgICAgIGN1cnJlbnRSYXRlID0gdGhpcy5yYXRlW2N1cnJlbnRCdWNrZXRdO1xuICAgICAgfVxuICAgICAgd29ya2VyLnBvc3RNZXNzYWdlKHtcbiAgICAgICAgYWN0aW9uOiBcInByb3BhZ2F0ZVwiLFxuICAgICAgICB0YXJnZXQ6IHRhcmdldCxcbiAgICAgICAgcmF0ZTogY3VycmVudFJhdGUsXG4gICAgICAgIG1lbW9yeUJ1ZmZlcjogdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnlcbiAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyB0cmFpbiB0aGUgd29ya2VyXG4gICAgd29ya2VyLm9ubWVzc2FnZSA9IGZ1bmN0aW9uKGUpIHtcbiAgICAgIC8vIGdpdmUgY29udHJvbCBvZiB0aGUgbWVtb3J5IGJhY2sgdG8gdGhlIG5ldHdvcmtcbiAgICAgIHRoYXQubmV0d29yay5vcHRpbWl6ZWQub3duZXJzaGlwKGUuZGF0YS5tZW1vcnlCdWZmZXIpO1xuXG4gICAgICBpZiAoZS5kYXRhLmFjdGlvbiA9PSBcInByb3BhZ2F0ZVwiKSB7XG4gICAgICAgIGlmIChpbmRleCA+PSBsZW5ndGgpIHtcbiAgICAgICAgICBpbmRleCA9IDA7XG4gICAgICAgICAgaXRlcmF0aW9ucysrO1xuICAgICAgICAgIGVycm9yIC89IHNldC5sZW5ndGg7XG5cbiAgICAgICAgICAvLyBsb2dcbiAgICAgICAgICBpZiAob3B0aW9ucykge1xuICAgICAgICAgICAgaWYgKHRoaXMuc2NoZWR1bGUgJiYgdGhpcy5zY2hlZHVsZS5ldmVyeSAmJiBpdGVyYXRpb25zICUgdGhpcy5zY2hlZHVsZS5ldmVyeSA9PSAwKVxuICAgICAgICAgICAgICBhYm9ydF90cmFpbmluZyA9IHRoaXMuc2NoZWR1bGUuZG8oe1xuICAgICAgICAgICAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zXG4gICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgZWxzZSBpZiAob3B0aW9ucy5sb2cgJiYgaXRlcmF0aW9ucyAlIG9wdGlvbnMubG9nID09IDApIHtcbiAgICAgICAgICAgICAgY29uc29sZS5sb2coJ2l0ZXJhdGlvbnMnLCBpdGVyYXRpb25zLCAnZXJyb3InLCBlcnJvcik7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSlcbiAgICAgICAgICAgICAgc2h1ZmZsZShzZXQpO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIGlmICghYWJvcnRfdHJhaW5pbmcgJiYgaXRlcmF0aW9ucyA8IHRoYXQuaXRlcmF0aW9ucyAmJiBlcnJvciA+IHRoYXQuZXJyb3IpIHtcbiAgICAgICAgICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAvLyBjYWxsYmFja1xuICAgICAgICAgICAgY2FsbGJhY2soe1xuICAgICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICAgIGl0ZXJhdGlvbnM6IGl0ZXJhdGlvbnMsXG4gICAgICAgICAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydFxuICAgICAgICAgICAgfSlcbiAgICAgICAgICB9XG4gICAgICAgICAgZXJyb3IgPSAwO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChlLmRhdGEuYWN0aW9uID09IFwiYWN0aXZhdGVcIikge1xuICAgICAgICBlcnJvciArPSB0aGF0LmNvc3Qoc2V0W2luZGV4XS5vdXRwdXQsIGUuZGF0YS5vdXRwdXQpO1xuICAgICAgICBwcm9wYWdhdGVXb3JrZXIoc2V0W2luZGV4XS5vdXRwdXQpO1xuICAgICAgICBpbmRleCsrO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIGtpY2sgaXRcbiAgICB2YXIgaW5kZXggPSAwO1xuICAgIHZhciBpdGVyYXRpb25zID0gMDtcbiAgICBhY3RpdmF0ZVdvcmtlcihzZXRbaW5kZXhdLmlucHV0KTtcbiAgfVxuXG4gIC8vIHRyYWlucyBhbiBYT1IgdG8gdGhlIG5ldHdvcmtcbiAgWE9SKG9wdGlvbnMpIHtcblxuICAgIGlmICh0aGlzLm5ldHdvcmsuaW5wdXRzKCkgIT0gMiB8fCB0aGlzLm5ldHdvcmsub3V0cHV0cygpICE9IDEpXG4gICAgICB0aHJvdyBcIkVycm9yOiBJbmNvbXBhdGlibGUgbmV0d29yayAoMiBpbnB1dHMsIDEgb3V0cHV0KVwiO1xuXG4gICAgdmFyIGRlZmF1bHRzID0ge1xuICAgICAgaXRlcmF0aW9uczogMTAwMDAwLFxuICAgICAgbG9nOiBmYWxzZSxcbiAgICAgIHNodWZmbGU6IHRydWUsXG4gICAgICBjb3N0OiBUcmFpbmVyLmNvc3QuTVNFXG4gICAgfVxuXG4gICAgaWYgKG9wdGlvbnMpXG4gICAgICBmb3IgKHZhciBpIGluIG9wdGlvbnMpXG4gICAgICAgIGRlZmF1bHRzW2ldID0gb3B0aW9uc1tpXTtcblxuICAgIHJldHVybiB0aGlzLnRyYWluKFt7XG4gICAgICBpbnB1dDogWzAsIDBdLFxuICAgICAgb3V0cHV0OiBbMF1cbiAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMSwgMF0sXG4gICAgICAgIG91dHB1dDogWzFdXG4gICAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMCwgMV0sXG4gICAgICAgIG91dHB1dDogWzFdXG4gICAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMSwgMV0sXG4gICAgICAgIG91dHB1dDogWzBdXG4gICAgICB9XSwgZGVmYXVsdHMpO1xuICB9XG5cbiAgLy8gdHJhaW5zIHRoZSBuZXR3b3JrIHRvIHBhc3MgYSBEaXN0cmFjdGVkIFNlcXVlbmNlIFJlY2FsbCB0ZXN0XG4gIERTUihvcHRpb25zKSB7XG4gICAgb3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG5cbiAgICB2YXIgdGFyZ2V0cyA9IG9wdGlvbnMudGFyZ2V0cyB8fCBbMiwgNCwgNywgOF07XG4gICAgdmFyIGRpc3RyYWN0b3JzID0gb3B0aW9ucy5kaXN0cmFjdG9ycyB8fCBbMywgNSwgNiwgOV07XG4gICAgdmFyIHByb21wdHMgPSBvcHRpb25zLnByb21wdHMgfHwgWzAsIDFdO1xuICAgIHZhciBsZW5ndGggPSBvcHRpb25zLmxlbmd0aCB8fCAyNDtcbiAgICB2YXIgY3JpdGVyaW9uID0gb3B0aW9ucy5zdWNjZXNzIHx8IDAuOTU7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnMgfHwgMTAwMDAwO1xuICAgIHZhciByYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4xO1xuICAgIHZhciBsb2cgPSBvcHRpb25zLmxvZyB8fCAwO1xuICAgIHZhciBzY2hlZHVsZSA9IG9wdGlvbnMuc2NoZWR1bGUgfHwge307XG4gICAgdmFyIGNvcnJlY3QgPSAwO1xuICAgIHZhciBpID0gMDtcbiAgICB2YXIgc3VjY2VzcyA9IDA7XG4gICAgdmFyIHRyaWFsID0gaSA9IGNvcnJlY3QgPSBqID0gc3VjY2VzcyA9IDAsXG4gICAgICBlcnJvciA9IDEsXG4gICAgICBzeW1ib2xzID0gdGFyZ2V0cy5sZW5ndGggKyBkaXN0cmFjdG9ycy5sZW5ndGggKyBwcm9tcHRzLmxlbmd0aDtcblxuICAgIHZhciBub1JlcGVhdCA9IGZ1bmN0aW9uKHJhbmdlLCBhdm9pZCkge1xuICAgICAgdmFyIG51bWJlciA9IE1hdGgucmFuZG9tKCkgKiByYW5nZSB8IDA7XG4gICAgICB2YXIgdXNlZCA9IGZhbHNlO1xuICAgICAgZm9yICh2YXIgaSBpbiBhdm9pZClcbiAgICAgICAgaWYgKG51bWJlciA9PSBhdm9pZFtpXSlcbiAgICAgICAgICB1c2VkID0gdHJ1ZTtcbiAgICAgIHJldHVybiB1c2VkID8gbm9SZXBlYXQocmFuZ2UsIGF2b2lkKSA6IG51bWJlcjtcbiAgICB9XG5cbiAgICB2YXIgZXF1YWwgPSBmdW5jdGlvbihwcmVkaWN0aW9uLCBvdXRwdXQpIHtcbiAgICAgIGZvciAodmFyIGkgaW4gcHJlZGljdGlvbilcbiAgICAgICAgaWYgKE1hdGgucm91bmQocHJlZGljdGlvbltpXSkgIT0gb3V0cHV0W2ldKVxuICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICB3aGlsZSAodHJpYWwgPCBpdGVyYXRpb25zICYmIChzdWNjZXNzIDwgY3JpdGVyaW9uIHx8IHRyaWFsICUgMTAwMCAhPSAwKSkge1xuICAgICAgLy8gZ2VuZXJhdGUgc2VxdWVuY2VcbiAgICAgIHZhciBzZXF1ZW5jZSA9IFtdLFxuICAgICAgICBzZXF1ZW5jZUxlbmd0aCA9IGxlbmd0aCAtIHByb21wdHMubGVuZ3RoO1xuICAgICAgZm9yIChpID0gMDsgaSA8IHNlcXVlbmNlTGVuZ3RoOyBpKyspIHtcbiAgICAgICAgdmFyIGFueSA9IE1hdGgucmFuZG9tKCkgKiBkaXN0cmFjdG9ycy5sZW5ndGggfCAwO1xuICAgICAgICBzZXF1ZW5jZS5wdXNoKGRpc3RyYWN0b3JzW2FueV0pO1xuICAgICAgfVxuICAgICAgdmFyIGluZGV4ZXMgPSBbXSxcbiAgICAgICAgcG9zaXRpb25zID0gW107XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgcHJvbXB0cy5sZW5ndGg7IGkrKykge1xuICAgICAgICBpbmRleGVzLnB1c2goTWF0aC5yYW5kb20oKSAqIHRhcmdldHMubGVuZ3RoIHwgMCk7XG4gICAgICAgIHBvc2l0aW9ucy5wdXNoKG5vUmVwZWF0KHNlcXVlbmNlTGVuZ3RoLCBwb3NpdGlvbnMpKTtcbiAgICAgIH1cbiAgICAgIHBvc2l0aW9ucyA9IHBvc2l0aW9ucy5zb3J0KCk7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgcHJvbXB0cy5sZW5ndGg7IGkrKykge1xuICAgICAgICBzZXF1ZW5jZVtwb3NpdGlvbnNbaV1dID0gdGFyZ2V0c1tpbmRleGVzW2ldXTtcbiAgICAgICAgc2VxdWVuY2UucHVzaChwcm9tcHRzW2ldKTtcbiAgICAgIH1cblxuICAgICAgLy90cmFpbiBzZXF1ZW5jZVxuICAgICAgdmFyIGRpc3RyYWN0b3JzQ29ycmVjdDtcbiAgICAgIHZhciB0YXJnZXRzQ29ycmVjdCA9IGRpc3RyYWN0b3JzQ29ycmVjdCA9IDA7XG4gICAgICBlcnJvciA9IDA7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgbGVuZ3RoOyBpKyspIHtcbiAgICAgICAgLy8gZ2VuZXJhdGUgaW5wdXQgZnJvbSBzZXF1ZW5jZVxuICAgICAgICB2YXIgaW5wdXQgPSBbXTtcbiAgICAgICAgZm9yIChqID0gMDsgaiA8IHN5bWJvbHM7IGorKylcbiAgICAgICAgICBpbnB1dFtqXSA9IDA7XG4gICAgICAgIGlucHV0W3NlcXVlbmNlW2ldXSA9IDE7XG5cbiAgICAgICAgLy8gZ2VuZXJhdGUgdGFyZ2V0IG91dHB1dFxuICAgICAgICB2YXIgb3V0cHV0ID0gW107XG4gICAgICAgIGZvciAoaiA9IDA7IGogPCB0YXJnZXRzLmxlbmd0aDsgaisrKVxuICAgICAgICAgIG91dHB1dFtqXSA9IDA7XG5cbiAgICAgICAgaWYgKGkgPj0gc2VxdWVuY2VMZW5ndGgpIHtcbiAgICAgICAgICB2YXIgaW5kZXggPSBpIC0gc2VxdWVuY2VMZW5ndGg7XG4gICAgICAgICAgb3V0cHV0W2luZGV4ZXNbaW5kZXhdXSA9IDE7XG4gICAgICAgIH1cblxuICAgICAgICAvLyBjaGVjayByZXN1bHRcbiAgICAgICAgdmFyIHByZWRpY3Rpb24gPSB0aGlzLm5ldHdvcmsuYWN0aXZhdGUoaW5wdXQpO1xuXG4gICAgICAgIGlmIChlcXVhbChwcmVkaWN0aW9uLCBvdXRwdXQpKVxuICAgICAgICAgIGlmIChpIDwgc2VxdWVuY2VMZW5ndGgpXG4gICAgICAgICAgICBkaXN0cmFjdG9yc0NvcnJlY3QrKztcbiAgICAgICAgICBlbHNlXG4gICAgICAgICAgICB0YXJnZXRzQ29ycmVjdCsrO1xuICAgICAgICBlbHNlIHtcbiAgICAgICAgICB0aGlzLm5ldHdvcmsucHJvcGFnYXRlKHJhdGUsIG91dHB1dCk7XG4gICAgICAgIH1cblxuICAgICAgICB2YXIgZGVsdGEgPSAwO1xuICAgICAgICBmb3IgKHZhciBqIGluIHByZWRpY3Rpb24pXG4gICAgICAgICAgZGVsdGEgKz0gTWF0aC5wb3cob3V0cHV0W2pdIC0gcHJlZGljdGlvbltqXSwgMik7XG4gICAgICAgIGVycm9yICs9IGRlbHRhIC8gdGhpcy5uZXR3b3JrLm91dHB1dHMoKTtcblxuICAgICAgICBpZiAoZGlzdHJhY3RvcnNDb3JyZWN0ICsgdGFyZ2V0c0NvcnJlY3QgPT0gbGVuZ3RoKVxuICAgICAgICAgIGNvcnJlY3QrKztcbiAgICAgIH1cblxuICAgICAgLy8gY2FsY3VsYXRlIGVycm9yXG4gICAgICBpZiAodHJpYWwgJSAxMDAwID09IDApXG4gICAgICAgIGNvcnJlY3QgPSAwO1xuICAgICAgdHJpYWwrKztcbiAgICAgIHZhciBkaXZpZGVFcnJvciA9IHRyaWFsICUgMTAwMDtcbiAgICAgIGRpdmlkZUVycm9yID0gZGl2aWRlRXJyb3IgPT0gMCA/IDEwMDAgOiBkaXZpZGVFcnJvcjtcbiAgICAgIHN1Y2Nlc3MgPSBjb3JyZWN0IC8gZGl2aWRlRXJyb3I7XG4gICAgICBlcnJvciAvPSBsZW5ndGg7XG5cbiAgICAgIC8vIGxvZ1xuICAgICAgaWYgKGxvZyAmJiB0cmlhbCAlIGxvZyA9PSAwKVxuICAgICAgICBjb25zb2xlLmxvZyhcIml0ZXJhdGlvbnM6XCIsIHRyaWFsLCBcIiBzdWNjZXNzOlwiLCBzdWNjZXNzLCBcIiBjb3JyZWN0OlwiLFxuICAgICAgICAgIGNvcnJlY3QsIFwiIHRpbWU6XCIsIERhdGUubm93KCkgLSBzdGFydCwgXCIgZXJyb3I6XCIsIGVycm9yKTtcbiAgICAgIGlmIChzY2hlZHVsZS5kbyAmJiBzY2hlZHVsZS5ldmVyeSAmJiB0cmlhbCAlIHNjaGVkdWxlLmV2ZXJ5ID09IDApXG4gICAgICAgIHNjaGVkdWxlLmRvKHtcbiAgICAgICAgICBpdGVyYXRpb25zOiB0cmlhbCxcbiAgICAgICAgICBzdWNjZXNzOiBzdWNjZXNzLFxuICAgICAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgICAgICB0aW1lOiBEYXRlLm5vdygpIC0gc3RhcnQsXG4gICAgICAgICAgY29ycmVjdDogY29ycmVjdFxuICAgICAgICB9KTtcbiAgICB9XG5cbiAgICByZXR1cm4ge1xuICAgICAgaXRlcmF0aW9uczogdHJpYWwsXG4gICAgICBzdWNjZXNzOiBzdWNjZXNzLFxuICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0XG4gICAgfVxuICB9XG5cbiAgLy8gdHJhaW4gdGhlIG5ldHdvcmsgdG8gbGVhcm4gYW4gRW1iZWRlZCBSZWJlciBHcmFtbWFyXG4gIEVSRyhvcHRpb25zKSB7XG5cbiAgICBvcHRpb25zID0gb3B0aW9ucyB8fCB7fTtcbiAgICB2YXIgaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucyB8fCAxNTAwMDA7XG4gICAgdmFyIGNyaXRlcmlvbiA9IG9wdGlvbnMuZXJyb3IgfHwgLjA1O1xuICAgIHZhciByYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4xO1xuICAgIHZhciBsb2cgPSBvcHRpb25zLmxvZyB8fCA1MDA7XG5cbiAgICAvLyBncmFtYXIgbm9kZVxuICAgIHZhciBOb2RlID0gZnVuY3Rpb24oKSB7XG4gICAgICB0aGlzLnBhdGhzID0gW107XG4gICAgfVxuICAgIE5vZGUucHJvdG90eXBlID0ge1xuICAgICAgY29ubmVjdDogZnVuY3Rpb24obm9kZSwgdmFsdWUpIHtcbiAgICAgICAgdGhpcy5wYXRocy5wdXNoKHtcbiAgICAgICAgICBub2RlOiBub2RlLFxuICAgICAgICAgIHZhbHVlOiB2YWx1ZVxuICAgICAgICB9KTtcbiAgICAgICAgcmV0dXJuIHRoaXM7XG4gICAgICB9LFxuICAgICAgYW55OiBmdW5jdGlvbigpIHtcbiAgICAgICAgaWYgKHRoaXMucGF0aHMubGVuZ3RoID09IDApXG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICB2YXIgaW5kZXggPSBNYXRoLnJhbmRvbSgpICogdGhpcy5wYXRocy5sZW5ndGggfCAwO1xuICAgICAgICByZXR1cm4gdGhpcy5wYXRoc1tpbmRleF07XG4gICAgICB9LFxuICAgICAgdGVzdDogZnVuY3Rpb24odmFsdWUpIHtcbiAgICAgICAgZm9yICh2YXIgaSBpbiB0aGlzLnBhdGhzKVxuICAgICAgICAgIGlmICh0aGlzLnBhdGhzW2ldLnZhbHVlID09IHZhbHVlKVxuICAgICAgICAgICAgcmV0dXJuIHRoaXMucGF0aHNbaV07XG4gICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB2YXIgcmViZXJHcmFtbWFyID0gZnVuY3Rpb24oKSB7XG5cbiAgICAgIC8vIGJ1aWxkIGEgcmViZXIgZ3JhbW1hclxuICAgICAgdmFyIG91dHB1dCA9IG5ldyBOb2RlKCk7XG4gICAgICB2YXIgbjEgPSAobmV3IE5vZGUoKSkuY29ubmVjdChvdXRwdXQsIFwiRVwiKTtcbiAgICAgIHZhciBuMiA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4xLCBcIlNcIik7XG4gICAgICB2YXIgbjMgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuMSwgXCJWXCIpLmNvbm5lY3QobjIsIFwiUFwiKTtcbiAgICAgIHZhciBuNCA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4yLCBcIlhcIilcbiAgICAgIG40LmNvbm5lY3QobjQsIFwiU1wiKTtcbiAgICAgIHZhciBuNSA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4zLCBcIlZcIilcbiAgICAgIG41LmNvbm5lY3QobjUsIFwiVFwiKTtcbiAgICAgIG4yLmNvbm5lY3QobjUsIFwiWFwiKVxuICAgICAgdmFyIG42ID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjQsIFwiVFwiKS5jb25uZWN0KG41LCBcIlBcIik7XG4gICAgICB2YXIgaW5wdXQgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuNiwgXCJCXCIpXG5cbiAgICAgIHJldHVybiB7XG4gICAgICAgIGlucHV0OiBpbnB1dCxcbiAgICAgICAgb3V0cHV0OiBvdXRwdXRcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBidWlsZCBhbiBlbWJlZGVkIHJlYmVyIGdyYW1tYXJcbiAgICB2YXIgZW1iZWRlZFJlYmVyR3JhbW1hciA9IGZ1bmN0aW9uKCkge1xuICAgICAgdmFyIHJlYmVyMSA9IHJlYmVyR3JhbW1hcigpO1xuICAgICAgdmFyIHJlYmVyMiA9IHJlYmVyR3JhbW1hcigpO1xuXG4gICAgICB2YXIgb3V0cHV0ID0gbmV3IE5vZGUoKTtcbiAgICAgIHZhciBuMSA9IChuZXcgTm9kZSkuY29ubmVjdChvdXRwdXQsIFwiRVwiKTtcbiAgICAgIHJlYmVyMS5vdXRwdXQuY29ubmVjdChuMSwgXCJUXCIpO1xuICAgICAgcmViZXIyLm91dHB1dC5jb25uZWN0KG4xLCBcIlBcIik7XG4gICAgICB2YXIgbjIgPSAobmV3IE5vZGUpLmNvbm5lY3QocmViZXIxLmlucHV0LCBcIlBcIikuY29ubmVjdChyZWJlcjIuaW5wdXQsXG4gICAgICAgIFwiVFwiKTtcbiAgICAgIHZhciBpbnB1dCA9IChuZXcgTm9kZSkuY29ubmVjdChuMiwgXCJCXCIpO1xuXG4gICAgICByZXR1cm4ge1xuICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgIG91dHB1dDogb3V0cHV0XG4gICAgICB9XG5cbiAgICB9XG5cbiAgICAvLyBnZW5lcmF0ZSBhbiBFUkcgc2VxdWVuY2VcbiAgICB2YXIgZ2VuZXJhdGUgPSBmdW5jdGlvbigpIHtcbiAgICAgIHZhciBub2RlID0gZW1iZWRlZFJlYmVyR3JhbW1hcigpLmlucHV0O1xuICAgICAgdmFyIG5leHQgPSBub2RlLmFueSgpO1xuICAgICAgdmFyIHN0ciA9IFwiXCI7XG4gICAgICB3aGlsZSAobmV4dCkge1xuICAgICAgICBzdHIgKz0gbmV4dC52YWx1ZTtcbiAgICAgICAgbmV4dCA9IG5leHQubm9kZS5hbnkoKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBzdHI7XG4gICAgfVxuXG4gICAgLy8gdGVzdCBpZiBhIHN0cmluZyBtYXRjaGVzIGFuIGVtYmVkZWQgcmViZXIgZ3JhbW1hclxuICAgIHZhciB0ZXN0ID0gZnVuY3Rpb24oc3RyKSB7XG4gICAgICB2YXIgbm9kZSA9IGVtYmVkZWRSZWJlckdyYW1tYXIoKS5pbnB1dDtcbiAgICAgIHZhciBpID0gMDtcbiAgICAgIHZhciBjaCA9IHN0ci5jaGFyQXQoaSk7XG4gICAgICB3aGlsZSAoaSA8IHN0ci5sZW5ndGgpIHtcbiAgICAgICAgdmFyIG5leHQgPSBub2RlLnRlc3QoY2gpO1xuICAgICAgICBpZiAoIW5leHQpXG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICBub2RlID0gbmV4dC5ub2RlO1xuICAgICAgICBjaCA9IHN0ci5jaGFyQXQoKytpKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIC8vIGhlbHBlciB0byBjaGVjayBpZiB0aGUgb3V0cHV0IGFuZCB0aGUgdGFyZ2V0IHZlY3RvcnMgbWF0Y2hcbiAgICB2YXIgZGlmZmVyZW50ID0gZnVuY3Rpb24oYXJyYXkxLCBhcnJheTIpIHtcbiAgICAgIHZhciBtYXgxID0gMDtcbiAgICAgIHZhciBpMSA9IC0xO1xuICAgICAgdmFyIG1heDIgPSAwO1xuICAgICAgdmFyIGkyID0gLTE7XG4gICAgICBmb3IgKHZhciBpIGluIGFycmF5MSkge1xuICAgICAgICBpZiAoYXJyYXkxW2ldID4gbWF4MSkge1xuICAgICAgICAgIG1heDEgPSBhcnJheTFbaV07XG4gICAgICAgICAgaTEgPSBpO1xuICAgICAgICB9XG4gICAgICAgIGlmIChhcnJheTJbaV0gPiBtYXgyKSB7XG4gICAgICAgICAgbWF4MiA9IGFycmF5MltpXTtcbiAgICAgICAgICBpMiA9IGk7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgcmV0dXJuIGkxICE9IGkyO1xuICAgIH1cblxuICAgIHZhciBpdGVyYXRpb24gPSAwO1xuICAgIHZhciBlcnJvciA9IDE7XG4gICAgdmFyIHRhYmxlID0ge1xuICAgICAgXCJCXCI6IDAsXG4gICAgICBcIlBcIjogMSxcbiAgICAgIFwiVFwiOiAyLFxuICAgICAgXCJYXCI6IDMsXG4gICAgICBcIlNcIjogNCxcbiAgICAgIFwiRVwiOiA1XG4gICAgfVxuXG4gICAgdmFyIHN0YXJ0ID0gRGF0ZS5ub3coKTtcbiAgICB3aGlsZSAoaXRlcmF0aW9uIDwgaXRlcmF0aW9ucyAmJiBlcnJvciA+IGNyaXRlcmlvbikge1xuICAgICAgdmFyIGkgPSAwO1xuICAgICAgZXJyb3IgPSAwO1xuXG4gICAgICAvLyBFUkcgc2VxdWVuY2UgdG8gbGVhcm5cbiAgICAgIHZhciBzZXF1ZW5jZSA9IGdlbmVyYXRlKCk7XG5cbiAgICAgIC8vIGlucHV0XG4gICAgICB2YXIgcmVhZCA9IHNlcXVlbmNlLmNoYXJBdChpKTtcbiAgICAgIC8vIHRhcmdldFxuICAgICAgdmFyIHByZWRpY3QgPSBzZXF1ZW5jZS5jaGFyQXQoaSArIDEpO1xuXG4gICAgICAvLyB0cmFpblxuICAgICAgd2hpbGUgKGkgPCBzZXF1ZW5jZS5sZW5ndGggLSAxKSB7XG4gICAgICAgIHZhciBpbnB1dCA9IFtdO1xuICAgICAgICB2YXIgdGFyZ2V0ID0gW107XG4gICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgNjsgaisrKSB7XG4gICAgICAgICAgaW5wdXRbal0gPSAwO1xuICAgICAgICAgIHRhcmdldFtqXSA9IDA7XG4gICAgICAgIH1cbiAgICAgICAgaW5wdXRbdGFibGVbcmVhZF1dID0gMTtcbiAgICAgICAgdGFyZ2V0W3RhYmxlW3ByZWRpY3RdXSA9IDE7XG5cbiAgICAgICAgdmFyIG91dHB1dCA9IHRoaXMubmV0d29yay5hY3RpdmF0ZShpbnB1dCk7XG5cbiAgICAgICAgaWYgKGRpZmZlcmVudChvdXRwdXQsIHRhcmdldCkpXG4gICAgICAgICAgdGhpcy5uZXR3b3JrLnByb3BhZ2F0ZShyYXRlLCB0YXJnZXQpO1xuXG4gICAgICAgIHJlYWQgPSBzZXF1ZW5jZS5jaGFyQXQoKytpKTtcbiAgICAgICAgcHJlZGljdCA9IHNlcXVlbmNlLmNoYXJBdChpICsgMSk7XG5cbiAgICAgICAgdmFyIGRlbHRhID0gMDtcbiAgICAgICAgZm9yICh2YXIgayBpbiBvdXRwdXQpXG4gICAgICAgICAgZGVsdGEgKz0gTWF0aC5wb3codGFyZ2V0W2tdIC0gb3V0cHV0W2tdLCAyKVxuICAgICAgICBkZWx0YSAvPSBvdXRwdXQubGVuZ3RoO1xuXG4gICAgICAgIGVycm9yICs9IGRlbHRhO1xuICAgICAgfVxuICAgICAgZXJyb3IgLz0gc2VxdWVuY2UubGVuZ3RoO1xuICAgICAgaXRlcmF0aW9uKys7XG4gICAgICBpZiAoaXRlcmF0aW9uICUgbG9nID09IDApIHtcbiAgICAgICAgY29uc29sZS5sb2coXCJpdGVyYXRpb25zOlwiLCBpdGVyYXRpb24sIFwiIHRpbWU6XCIsIERhdGUubm93KCkgLSBzdGFydCxcbiAgICAgICAgICBcIiBlcnJvcjpcIiwgZXJyb3IpO1xuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb24sXG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICB0aW1lOiBEYXRlLm5vdygpIC0gc3RhcnQsXG4gICAgICB0ZXN0OiB0ZXN0LFxuICAgICAgZ2VuZXJhdGU6IGdlbmVyYXRlXG4gICAgfVxuICB9XG5cbn1cblxuZXhwb3J0IG1vZHVsZSBUcmFpbmVyIHtcbiAgLy8gQnVpbHQtaW4gY29zdCBmdW5jdGlvbnNcbiAgXG4gIGV4cG9ydCBpbnRlcmZhY2UgSVRyYWluZXJDb3N0Rm4ge1xuICAgICh0YXJnZXQsIG91dHB1dCk6IG51bWJlcjtcbiAgfVxuXG4gIGV4cG9ydCB2YXIgY29zdCA9IHtcbiAgICAvLyBFcS4gOVxuICAgIENST1NTX0VOVFJPUFk6IGZ1bmN0aW9uKHRhcmdldCwgb3V0cHV0KSB7XG4gICAgICB2YXIgY3Jvc3NlbnRyb3B5ID0gMDtcbiAgICAgIGZvciAodmFyIGkgaW4gb3V0cHV0KVxuICAgICAgICBjcm9zc2VudHJvcHkgLT0gKHRhcmdldFtpXSAqIE1hdGgubG9nKG91dHB1dFtpXSArIDFlLTE1KSkgKyAoKDEgLSB0YXJnZXRbaV0pICogTWF0aC5sb2coKDEgKyAxZS0xNSkgLSBvdXRwdXRbaV0pKTsgLy8gKzFlLTE1IGlzIGEgdGlueSBwdXNoIGF3YXkgdG8gYXZvaWQgTWF0aC5sb2coMClcbiAgICAgIHJldHVybiBjcm9zc2VudHJvcHk7XG4gICAgfSxcbiAgICBNU0U6IGZ1bmN0aW9uKHRhcmdldCwgb3V0cHV0KSB7XG4gICAgICB2YXIgbXNlID0gMDtcbiAgICAgIGZvciAodmFyIGkgaW4gb3V0cHV0KVxuICAgICAgICBtc2UgKz0gTWF0aC5wb3codGFyZ2V0W2ldIC0gb3V0cHV0W2ldLCAyKTtcbiAgICAgIHJldHVybiBtc2UgLyBvdXRwdXQubGVuZ3RoO1xuICAgIH1cbiAgfVxufSJdfQ==
var synaptic = synaptic || Synaptic;var Neuron = synaptic.Neuron, Layer = synaptic.Layer, Network = synaptic.Network, Trainer = synaptic.Trainer, Architect = synaptic.Architect;