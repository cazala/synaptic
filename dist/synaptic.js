(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
var hopfield = require('./architect/Hopfield');
var lstm = require('./architect/LSTM');
var lsm = require('./architect/Liquid');
var perceptron = require('./architect/Perceptron');
exports.LSTM = lstm.LSTM;
exports.Liquid = lsm.Liquid;
exports.Hopfield = hopfield.Hopfield;
exports.Perceptron = perceptron.Perceptron;

},{"./architect/Hopfield":2,"./architect/LSTM":3,"./architect/Liquid":4,"./architect/Perceptron":5}],2:[function(require,module,exports){
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

},{"../layer":6,"../network":7,"../trainer":11}],3:[function(require,module,exports){
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

},{"../layer":6,"../network":7,"../trainer":11}],6:[function(require,module,exports){
var neuron = require('./neuron');
var network = require('./network');
/*******************************************************************************************
                                            LAYER
*******************************************************************************************/
var Layer = (function () {
    function Layer(size, label) {
        this.optimizable = true;
        this.list = [];
        this.label = null;
        this.connectedto = [];
        this.size = 0;
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
    Layer.prototype.activate = function (input) {
        if (this.currentActivation.length != this.list.length)
            this.currentActivation = new Float64Array(this.list.length);
        var activationIndex = 0;
        if (typeof input != 'undefined') {
            if (input.length != this.size)
                throw "INPUT size and LAYER size must be the same to activate!";
            for (var id in this.list) {
                this.currentActivation[activationIndex++] = this.list[id].activate(input[id]);
            }
        }
        else {
            for (var id in this.list) {
                this.currentActivation[activationIndex++] = this.list[id].activate();
            }
        }
        return this.currentActivation;
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
                hidden: [],
                output: null
            };
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
        this.optimizable = true;
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
    Neuron.prototype.readIncommingConnections = function (input) {
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
    };
    Neuron.prototype.updateTraces = function () {
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
    };
    // activate the neuron
    Neuron.prototype.activate = function (input) {
        this.readIncommingConnections(input);
        this.updateTraces();
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
    if (derivate) {
        var fx = LOGISTIC(x);
        return fx * (1 - fx);
    }
    return 1 / (1 + Math.exp(-x));
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
function SOFTPLUS(x, derivate) {
    if (derivate)
        return 1 - 1 / (1 + Math.exp(x));
    return Math.log(1 + Math.exp(x));
}
exports.SOFTPLUS = SOFTPLUS;
function EXP(x, derivate) {
    return Math.exp(x);
}
exports.EXP = EXP;

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
var utils = require('./utils');
var Synaptic;
(function (Synaptic) {
    var oldSynaptic = typeof window != "undefined" && window && window['Synaptic'];
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
    Synaptic.Utils = utils.Utils;
})(Synaptic || (Synaptic = {}));
if (typeof window != "undefined")
    window['synaptic'] = Synaptic;
module.exports = Synaptic;

},{"./architect":1,"./layer":6,"./network":7,"./neuron":8,"./squash":9,"./trainer":11,"./utils":12}],11:[function(require,module,exports){
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
                if (this.schedule && this.schedule.every && iterations % this.schedule.every == 0) {
                    abort_training = this.schedule.do({
                        error: error,
                        iterations: iterations,
                        rate: currentRate
                    });
                }
                else if (options.log && iterations % options.log == 0) {
                    console.log('iterations', iterations, 'error', error, 'rate', currentRate, 'T:', target, 'O:', output);
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
            if (schedule.do && schedule.every && trial % schedule.every == 0) {
                schedule.do({
                    iterations: trial,
                    success: success,
                    error: error,
                    time: Date.now() - start,
                    correct: correct
                });
            }
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
        CROSS_ENTROPY_SOFTMAX: function (target, output) {
            var crossentropy = 0;
            for (var i in output)
                crossentropy -= target[i] * Math.log(output[i] + 1e-15);
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

},{}],12:[function(require,module,exports){
var Utils = (function () {
    function Utils() {
    }
    Utils.softMax = function (outputArray) {
        // for all i ∈ array
        // sum = ∑ array[n]^e
        // i = î^e / sum
        // where the result ∑ array[0..n] = 1
        if (!outputArray.length)
            return outputArray;
        var sum = 0;
        var Amax = outputArray[0];
        for (var i = 0; i < outputArray.length; i++) {
            if (outputArray[i] < Amax)
                Amax = outputArray[i];
        }
        // sum = ∑ array[n]^e
        for (var i = 0; i < outputArray.length; i++) {
            outputArray[i] = Math.exp(outputArray[i] - Amax);
            sum += outputArray[i];
        }
        for (var i = 0; i < outputArray.length; i++)
            outputArray[i] /= sum;
        return outputArray;
    };
    Utils.softMaxDerivative = function (outputArray) {
        // http://sysmagazine.com/posts/155235/
        if (!outputArray.length)
            return outputArray;
        var sum = 0;
        // sum = ∑ array[n]^e
        for (var i = 0; i < outputArray.length; i++) {
            outputArray[i] = Math.exp(outputArray[i]);
            sum += outputArray[i];
        }
        for (var i = 0; i < outputArray.length; i++) {
            var t = outputArray[i] /= sum;
            outputArray[i] = t * (1 - t);
        }
        return outputArray;
    };
    Utils.softMaxReinforcement = function (array, temperature) {
        // Reinforcement learning
        if (temperature === void 0) { temperature = 1; }
        if (!array.length)
            return array;
        temperature = temperature || 1;
        var sum = 0;
        // sum = ∑ array[n]^e
        for (var i = 0; i < array.length; i++) {
            array[i] = Math.exp(array[i] / temperature);
            sum += array[i];
        }
        if (sum != 0) {
            for (var i = 0; i < array.length; i++)
                array[i] /= sum;
        }
        else {
            var div = 1 / array.length;
            for (var i = 0; i < array.length; i++)
                array[i] = div;
        }
        return array;
    };
    Utils.getCosineSimilarity = function (arrayA, arrayB) {
        // http://en.wikipedia.org/wiki/Cosine_similarity
        var dotPr = 0;
        var acumA = 0, acumB = 0;
        for (var i = 0; i < arrayA.length; i++) {
            dotPr += arrayA[i] * arrayB[i];
            acumA += arrayA[i] * arrayA[i];
            acumB += arrayB[i] * arrayB[i];
        }
        return dotPr / (Math.sqrt(acumA) * Math.sqrt(acumB) + .00005);
    };
    Utils.interpolateArray = function (output_inputA, inputB, g) {
        // 3.3.2 focus by location (7)
        var gInverted = 1 - g;
        for (var i = 0; i < output_inputA.length; i++)
            output_inputA[i] = output_inputA[i] * g + gInverted * inputB[i];
        return output_inputA;
    };
    // w_sharpWn
    Utils.sharpArray = function (output, wn, Y) {
        // 3.3.2 (9)
        var sum = 0;
        // ∀ a ∈ wn → a = a^Y
        // sum = ∑ a^Y 
        for (var i = 0; i < wn.length; i++) {
            wn[i] = Math.pow(wn[i], Y);
            sum += wn[i];
        }
        // ∀ a ∈ wn → a = a^Y / sum
        if (sum != 0) {
            for (var i = 0; i < wn.length; i++)
                output[i] = wn[i] / sum;
        }
        else {
            var div = 1 / wn.length;
            for (var i = 0; i < wn.length; i++)
                output[i] = div;
        }
    };
    //wn_shift
    Utils.scalarShifting = function (wg, shiftScalar) {
        // w~ 3.3.2 (8)
        var shiftings = new Float64Array(wg.length);
        var wn = new Float64Array(wg.length);
        var intPart = shiftScalar | 0;
        var decimalPart = shiftScalar - intPart;
        shiftings[intPart % shiftings.length] = 1 - decimalPart;
        shiftings[(intPart + 1) % shiftings.length] = decimalPart;
        for (var i = 0; i < wn.length; i++) {
            var acum = 0;
            for (var j = 0; j < wn.length; j++) {
                if ((i - j) < 0)
                    acum += wg[j] * shiftings[shiftings.length - Math.abs(i - j)];
                else
                    acum += wg[j] * shiftings[(i - j) % shiftings.length];
            }
            wn[i] = acum;
        }
        return wn;
    };
    Utils.normalizeShift = function (shift) {
        var sum = 0;
        for (var i = 0; i < shift.length; i++) {
            sum += shift[i];
        }
        for (var j = 0; j < shift.length; j++) {
            shift[j] /= sum;
        }
    };
    Utils.vectorInvertedShifting = function (wg, shiftings) {
        // w~ 3.3.2 (8)
        var ret = new Float64Array(wg.length);
        var corrimientoIndex = -((shiftings.length - 1) / 2) | 0;
        var circulantMatrix = Utils.transformationMatrixCache[wg.length] || (Utils.transformationMatrixCache[wg.length] = Utils.buildCirculantMatrix(wg.length));
        for (var i = 0; i < wg.length; i++) {
            for (var x = 0; x < wg.length; x++) {
                var tmp = 0;
                for (var shift = 0; shift < shiftings.length; shift++) {
                    var matRow = i - x + corrimientoIndex + shift;
                    while (matRow < 0)
                        matRow += wg.length;
                    matRow %= wg.length;
                    tmp += wg[circulantMatrix[x][matRow]] * shiftings[shift];
                }
                ret[i] = tmp;
            }
        }
        wg.set(ret);
    };
    Utils.initRandomSoftmaxArray = function (array) {
        for (var i = 0; i < array.length; i++) {
            array[i] = Math.random();
        }
        Utils.softMax(array);
    };
    Utils.buildCirculantMatrix = function (length, offset) {
        if (offset === void 0) { offset = 0; }
        var ret = [];
        for (var i = 0; i < length; i++) {
            var arr = new Float64Array(length);
            ret.push(arr);
            for (var n = 0; n < length; n++) {
                arr[n] = ((i + n) % length);
            }
        }
        return ret;
    };
    Utils.transformationMatrixCache = {};
    return Utils;
})();
exports.Utils = Utils;

},{}]},{},[10]);
var synaptic = synaptic || Synaptic;var Neuron = synaptic.Neuron, Layer = synaptic.Layer, Network = synaptic.Network, Trainer = synaptic.Trainer, Architect = synaptic.Architect;