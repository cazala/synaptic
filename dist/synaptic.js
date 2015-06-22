(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
var hopfield = require('./architect/Hopfield');
var lstm = require('./architect/LSTM');
var lsm = require('./architect/Liquid');
var perceptron = require('./architect/Perceptron');
var mb = require('./architect/MemoryBlock');
exports.LSTM = lstm.LSTM;
exports.Liquid = lsm.Liquid;
exports.Hopfield = hopfield.Hopfield;
exports.Perceptron = perceptron.Perceptron;
exports.MemoryBlock = mb.MemoryBlock;

},{"./architect/Hopfield":2,"./architect/LSTM":3,"./architect/Liquid":4,"./architect/MemoryBlock":5,"./architect/Perceptron":6}],2:[function(require,module,exports){
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

},{"../layer":7,"../network":8,"../trainer":12}],3:[function(require,module,exports){
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

},{"../layer":7,"../network":8,"../trainer":12}],4:[function(require,module,exports){
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

},{"../layer":7,"../network":8,"../trainer":12}],5:[function(require,module,exports){
var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
var network = require('../network');
var trainer = require('../trainer');
var Layer = require('../layer');
var MemoryTape = (function () {
    function MemoryTape(memoryBlocks, layer, inputGate, forgetGate) {
        var _this = this;
        this.memoryAttentionLocation = 0;
        this.memoryAttentionWeight = 0;
        this.blocks = memoryBlocks;
        this.data = new Array(memoryBlocks);
        this.blockWidth = layer.list.length;
        this.layer = layer;
        this.outputLayer = new Layer.Layer(layer.list.length);
        this.outputLayer.project(layer, Layer.Layer.connectionType.ONE_TO_ONE);
        for (var location = 0; location < this.blocks; location++) {
            var array = this.data[location] = new Float64Array(this.blockWidth);
            for (var i = 0; i < array.length; i++) {
                array[i] = Math.random();
            }
        }
        // Hack the layer!
        var similarAddresses;
        // elegible memblocks for read/write operations
        var elegibleIndexes = [0, 1, 2];
        var elegibleWeights = [0.01, 1, 0.01]; // shifting, softmax
        var focus = 1;
        this.prevLayerActivate = this.layer.activate.bind(this.layer);
        this.prevLayerPropagate = this.layer.propagate.bind(this.layer);
        var key;
        this.layer.propagate = function (rate, target) {
            _this.prevLayerPropagate(rate, target);
            var addGate = inputGate.currentActivation;
            var eraseGate = forgetGate.currentActivation;
            for (var n = 0; n < elegibleIndexes.length; n++) {
                var M = _this.data[elegibleIndexes[n]];
                for (var i = 0; i < M.length; i++) {
                    // do erase operations on the memory tape. NTM: 3.2 (3)
                    M[i] *= 1 - eraseGate[i] * key[i] * elegibleWeights[n];
                    // do add operations on the memory tape. NTM: 3.2 (4)
                    M[i] += addGate[i] * key[i] * elegibleWeights[n] * rate;
                }
            }
        };
        this.layer.activate = function (input) {
            var result = _this.prevLayerActivate(input);
            key = MemoryTape.softMaxArray(new Float64Array(result));
            similarAddresses = _this.getSimilarAdresses(key);
            _this.memoryAttentionWeight = 0;
            for (var address = 0; address < similarAddresses.length; address++) {
                var ß = similarAddresses[address];
                if (ß > _this.memoryAttentionWeight) {
                    _this.memoryAttentionWeight = ß;
                    _this.memoryAttentionLocation = address;
                }
            }
            elegibleIndexes = [_this.memoryAttentionLocation - 1, _this.memoryAttentionLocation, _this.memoryAttentionLocation + 1];
            focus = _this.memoryAttentionWeight;
            elegibleWeights = [0.1, 0.8, 0.1]; // shifting, softmax
            for (var n = 0; n < elegibleIndexes.length; n++) {
                var index = elegibleIndexes[n];
                if (index < 0)
                    index += similarAddresses.length;
                else if (index >= similarAddresses.length)
                    index -= similarAddresses.length;
                elegibleIndexes[n] = index;
                elegibleWeights[n] = elegibleWeights[n] / focus * similarAddresses[index];
            }
            _this.outputLayer.list.forEach(function (neuron, i) {
                // modify the current key (readVector)
                var tmpKey = 0;
                for (var n = 0; n < elegibleIndexes.length; n++) {
                    tmpKey += _this.data[elegibleIndexes[n]][i] * elegibleWeights[n];
                }
                neuron.activate(tmpKey);
            });
            return result;
        };
    }
    MemoryTape.getSimilarity = function (arrayA, arrayB) {
        // http://en.wikipedia.org/wiki/Cosine_similarity
        // NTM: 3.3.1 (6)
        var dotPr = 0;
        var acumA = 0, acumB = 0;
        for (var i = 0; i < arrayA.length; i++) {
            dotPr += arrayA[i] * arrayB[i];
            acumA += arrayA[i] * arrayA[i];
            acumB += arrayB[i] * arrayB[i];
        }
        return dotPr / (Math.sqrt(acumA) * Math.sqrt(acumB) + .00005);
    };
    MemoryTape.softMaxArray = function (array, sharpen) {
        // for all i ∈ array
        // sum = ∑ array[n]^e
        // i = î^e / sum
        // where the result ∑ array[0..n] = 1
        if (sharpen === void 0) { sharpen = 1; }
        if (!array.length)
            return array;
        sharpen = sharpen || 1;
        var sum = 0;
        // sum = ∑ array[n]^e
        for (var i = 0; i < array.length; i++) {
            array[i] = Math.exp(sharpen * array[i]);
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
    // obtains an array of similarity indexes for each memoryBlock
    MemoryTape.prototype.getSimilarAdresses = function (weights) {
        //checkpoint: 10th cigarret
        var addresses = new Float64Array(this.data.length);
        for (var i = 0; i < this.data.length; i++)
            addresses[i] = MemoryTape.getSimilarity(this.data[i], weights);
        return addresses;
    };
    return MemoryTape;
})();
exports.MemoryTape = MemoryTape;
var MemoryBlock = (function (_super) {
    __extends(MemoryBlock, _super);
    function MemoryBlock(inputSize, memoryBlocks, memoryWidth, outputSize) {
        var option = {
            peepholes: Layer.Layer.connectionType.ALL_TO_ALL,
            hiddentohidden: false,
            outtohidden: false,
            outtogates: false,
            intoout: true,
        };
        var inputLayer = new Layer.Layer(inputSize);
        var hiddenLayers = [];
        var outputLayer = new Layer.Layer(outputSize);
        //#region generate layers
        // generate memory blocks (memory cell and respective gates)
        var inputGate = new Layer.Layer(memoryWidth).set({
            bias: 1
        });
        var forgetGate = new Layer.Layer(memoryWidth).set({
            bias: 1
        });
        var memoryCell = new Layer.Layer(memoryWidth);
        var outputGate = new Layer.Layer(memoryWidth).set({
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
        // connections from memory cell
        var output = memoryCell.project(outputLayer);
        // self-connection
        var self = memoryCell.project(memoryCell);
        // peepholes
        memoryCell.project(inputGate, option.peepholes);
        memoryCell.project(forgetGate, option.peepholes);
        memoryCell.project(outputGate, option.peepholes);
        // gates
        inputGate.gate(input, Layer.Layer.gateType.INPUT);
        forgetGate.gate(self, Layer.Layer.gateType.ONE_TO_ONE);
        outputGate.gate(output, Layer.Layer.gateType.OUTPUT);
        this.memoryTape = new MemoryTape(memoryBlocks, memoryCell, inputGate, forgetGate);
        hiddenLayers.push(this.memoryTape.outputLayer);
        //#endregion
        // input to output direct connection
        if (option.intoout)
            inputLayer.project(outputLayer);
        // set the layers of the neural network
        _super.call(this, {
            input: inputLayer,
            hidden: hiddenLayers,
            output: outputLayer
        });
        // DO NOT OPTIMIZE THIS NETWORK
        this.optimized = false;
        // trainer
        this.trainer = new trainer.Trainer(this);
    }
    return MemoryBlock;
})(network.Network);
exports.MemoryBlock = MemoryBlock;
;

},{"../layer":7,"../network":8,"../trainer":12}],6:[function(require,module,exports){
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

},{"../layer":7,"../network":8,"../trainer":12}],7:[function(require,module,exports){
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

},{"./network":8,"./neuron":9}],8:[function(require,module,exports){
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

},{"./layer":7,"./neuron":9,"./squash":10}],9:[function(require,module,exports){
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

},{"./squash":10}],10:[function(require,module,exports){
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

},{}],11:[function(require,module,exports){
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
})(Synaptic || (Synaptic = {}));
if (typeof window != "undefined")
    window['synaptic'] = Synaptic;
module.exports = Synaptic;

},{"./architect":1,"./layer":7,"./network":8,"./neuron":9,"./squash":10,"./trainer":12}],12:[function(require,module,exports){
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
        var _this = this;
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
                    requestAnimationFrame(function () {
                        _this.iterations -= iterations;
                        _this.train(set, options);
                    });
                    return;
                }
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
        var _this = this;
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
                requestAnimationFrame(function () {
                    _this.iterations -= trial;
                    _this.DSR(options);
                });
                return;
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
        MSE: function (target, output) {
            var mse = 0;
            for (var i in output)
                mse += Math.pow(target[i] - output[i], 2);
            return mse / output.length;
        }
    };
})(Trainer = exports.Trainer || (exports.Trainer = {}));

},{}]},{},[11])
//# sourceMappingURL=data:application/json;charset:utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyaWZ5L25vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJzcmMvYXJjaGl0ZWN0LnRzIiwic3JjL2FyY2hpdGVjdC9Ib3BmaWVsZC50cyIsInNyYy9hcmNoaXRlY3QvTFNUTS50cyIsInNyYy9hcmNoaXRlY3QvTGlxdWlkLnRzIiwic3JjL2FyY2hpdGVjdC9NZW1vcnlCbG9jay50cyIsInNyYy9hcmNoaXRlY3QvUGVyY2VwdHJvbi50cyIsInNyYy9sYXllci50cyIsInNyYy9uZXR3b3JrLnRzIiwic3JjL25ldXJvbi50cyIsInNyYy9zcXVhc2gudHMiLCJzcmMvc3luYXB0aWMudHMiLCJzcmMvdHJhaW5lci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTtBQ0FBLElBQU8sUUFBUSxXQUFXLHNCQUFzQixDQUFDLENBQUM7QUFDbEQsSUFBTyxJQUFJLFdBQVcsa0JBQWtCLENBQUMsQ0FBQztBQUMxQyxJQUFPLEdBQUcsV0FBVyxvQkFBb0IsQ0FBQyxDQUFDO0FBQzNDLElBQU8sVUFBVSxXQUFXLHdCQUF3QixDQUFDLENBQUM7QUFDdEQsSUFBTyxFQUFFLFdBQVcseUJBQXlCLENBQUMsQ0FBQztBQUVwQyxZQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQztBQUNqQixjQUFNLEdBQUcsR0FBRyxDQUFDLE1BQU0sQ0FBQztBQUNwQixnQkFBUSxHQUFHLFFBQVEsQ0FBQyxRQUFRLENBQUM7QUFDN0Isa0JBQVUsR0FBRyxVQUFVLENBQUMsVUFBVSxDQUFDO0FBQ25DLG1CQUFXLEdBQUcsRUFBRSxDQUFDLFdBQVcsQ0FBQzs7Ozs7Ozs7O0FDVnhDLElBQU8sT0FBTyxXQUFZLFlBQVksQ0FBQyxDQUFDO0FBQ3hDLElBQU8sT0FBTyxXQUFZLFlBQVksQ0FBQyxDQUFDO0FBQ3hDLElBQU8sS0FBSyxXQUFZLFVBQVUsQ0FBQyxDQUFDO0FBR3BDO0lBQThCLDRCQUFlO0lBRzNDLGtCQUFZLElBQVk7UUFDdEIsSUFBSSxVQUFVLEdBQUcsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3ZDLElBQUksV0FBVyxHQUFHLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUV4QyxVQUFVLENBQUMsT0FBTyxDQUFDLFdBQVcsRUFBRSxLQUFLLENBQUMsS0FBSyxDQUFDLGNBQWMsQ0FBQyxVQUFVLENBQUMsQ0FBQztRQUV2RSxrQkFBTTtZQUNKLEtBQUssRUFBRSxVQUFVO1lBQ2pCLE1BQU0sRUFBRSxFQUFFO1lBQ1YsTUFBTSxFQUFFLFdBQVc7U0FDcEIsQ0FBQyxDQUFDO1FBRUgsSUFBSSxDQUFDLE9BQU8sR0FBRyxJQUFJLE9BQU8sQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDM0MsQ0FBQztJQUVELHdCQUFLLEdBQUwsVUFBTSxRQUFRO1FBQ1osSUFBSSxHQUFHLEdBQUcsRUFBRSxDQUFDO1FBQ2IsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxRQUFRLENBQUM7WUFDckIsR0FBRyxDQUFDLElBQUksQ0FBQztnQkFDUCxLQUFLLEVBQUUsUUFBUSxDQUFDLENBQUMsQ0FBQztnQkFDbEIsTUFBTSxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUM7YUFDcEIsQ0FBQyxDQUFDO1FBRUwsTUFBTSxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsS0FBSyxDQUFDLEdBQUcsRUFBRTtZQUM3QixVQUFVLEVBQUUsTUFBTTtZQUNsQixLQUFLLEVBQUUsTUFBTTtZQUNiLElBQUksRUFBRSxDQUFDO1NBQ1IsQ0FBQyxDQUFDO0lBQ0wsQ0FBQztJQUVELHVCQUFJLEdBQUosVUFBSyxPQUFPO1FBQ1YsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLFFBQVEsQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUVwQyxJQUFJLFFBQVEsR0FBRyxFQUFFLENBQUM7UUFDbEIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxNQUFNLENBQUM7WUFDbkIsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxFQUFFLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUV2QyxNQUFNLENBQUMsUUFBUSxDQUFDO0lBQ2xCLENBQUM7SUFDSCxlQUFDO0FBQUQsQ0ExQ0EsQUEwQ0MsRUExQzZCLE9BQU8sQ0FBQyxPQUFPLEVBMEM1QztBQTFDWSxnQkFBUSxXQTBDcEIsQ0FBQTs7Ozs7Ozs7O0FDL0NELElBQU8sT0FBTyxXQUFZLFlBQVksQ0FBQyxDQUFDO0FBQ3hDLElBQU8sT0FBTyxXQUFZLFlBQVksQ0FBQyxDQUFDO0FBQ3hDLElBQU8sS0FBSyxXQUFZLFVBQVUsQ0FBQyxDQUFDO0FBR3BDO0lBQTBCLHdCQUFlO0lBR3ZDO1FBQVksY0FBYzthQUFkLFdBQWMsQ0FBZCxzQkFBYyxDQUFkLElBQWM7WUFBZCw2QkFBYzs7UUFFeEIsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUM7WUFDbEIsTUFBTSx5Q0FBeUMsQ0FBQztRQUVsRCxJQUFJLElBQUksR0FBRyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUM7UUFDdEIsSUFBSSxNQUFNLEdBQUc7WUFDWCxTQUFTLEVBQUUsS0FBSyxDQUFDLEtBQUssQ0FBQyxjQUFjLENBQUMsVUFBVTtZQUNoRCxjQUFjLEVBQUUsS0FBSztZQUNyQixXQUFXLEVBQUUsS0FBSztZQUNsQixVQUFVLEVBQUUsS0FBSztZQUNqQixPQUFPLEVBQUUsSUFBSTtTQUNkLENBQUM7UUFFRixFQUFFLENBQUMsQ0FBQyxPQUFPLElBQUksSUFBSSxRQUFRLENBQUMsQ0FBQyxDQUFDO1lBQzVCLElBQUksT0FBTyxHQUFHLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQztZQUN6QixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLFdBQVcsQ0FBQyxDQUFDO2dCQUNuQyxNQUFNLENBQUMsU0FBUyxHQUFHLElBQUksQ0FBQyxTQUFTLENBQUM7WUFDcEMsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDO2dCQUN4QyxNQUFNLENBQUMsY0FBYyxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUM7WUFDOUMsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxhQUFhLENBQUMsQ0FBQztnQkFDckMsTUFBTSxDQUFDLFdBQVcsR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDO1lBQ3hDLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsWUFBWSxDQUFDLENBQUM7Z0JBQ3BDLE1BQU0sQ0FBQyxVQUFVLEdBQUcsSUFBSSxDQUFDLFVBQVUsQ0FBQztZQUN0QyxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLFNBQVMsQ0FBQyxDQUFDO2dCQUNqQyxNQUFNLENBQUMsT0FBTyxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUM7UUFDbEMsQ0FBQztRQUFDLElBQUk7WUFDSixJQUFJLE9BQU8sR0FBRyxJQUFJLENBQUM7UUFFckIsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLEtBQUssRUFBRSxDQUFDO1FBQzFCLElBQUksTUFBTSxHQUFHLElBQUksQ0FBQztRQUVsQixJQUFJLFVBQVUsR0FBRyxJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDekMsSUFBSSxZQUFZLEdBQUcsRUFBRSxDQUFDO1FBQ3RCLElBQUksV0FBVyxHQUFHLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUUzQyxJQUFJLFFBQVEsR0FBRyxJQUFJLENBQUM7UUFFcEIsa0JBQWtCO1FBQ2xCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLElBQUksTUFBTSxDQUFDLENBQUMsQ0FBQztZQUN6Qiw0REFBNEQ7WUFDNUQsSUFBSSxJQUFJLEdBQUcsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDO1lBRXpCLElBQUksU0FBUyxHQUFHLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLENBQUM7Z0JBQ3hDLElBQUksRUFBRSxDQUFDO2FBQ1IsQ0FBQyxDQUFDO1lBQ0gsSUFBSSxVQUFVLEdBQUcsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsQ0FBQztnQkFDekMsSUFBSSxFQUFFLENBQUM7YUFDUixDQUFDLENBQUM7WUFDSCxJQUFJLFVBQVUsR0FBRyxJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUM7WUFDdkMsSUFBSSxVQUFVLEdBQUcsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDLEdBQUcsQ0FBQztnQkFDekMsSUFBSSxFQUFFLENBQUM7YUFDUixDQUFDLENBQUM7WUFFSCxZQUFZLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1lBQzdCLFlBQVksQ0FBQyxJQUFJLENBQUMsVUFBVSxDQUFDLENBQUM7WUFDOUIsWUFBWSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUMsQ0FBQztZQUM5QixZQUFZLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxDQUFDO1lBRTlCLCtCQUErQjtZQUMvQixJQUFJLEtBQUssR0FBRyxVQUFVLENBQUMsT0FBTyxDQUFDLFVBQVUsQ0FBQyxDQUFDO1lBQzNDLFVBQVUsQ0FBQyxPQUFPLENBQUMsU0FBUyxDQUFDLENBQUM7WUFDOUIsVUFBVSxDQUFDLE9BQU8sQ0FBQyxVQUFVLENBQUMsQ0FBQztZQUMvQixVQUFVLENBQUMsT0FBTyxDQUFDLFVBQVUsQ0FBQyxDQUFDO1lBRS9CLDJEQUEyRDtZQUMzRCxFQUFFLENBQUMsQ0FBQyxRQUFRLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQztnQkFDckIsSUFBSSxJQUFJLEdBQUcsUUFBUSxDQUFDLE9BQU8sQ0FBQyxVQUFVLENBQUMsQ0FBQztnQkFDeEMsUUFBUSxDQUFDLE9BQU8sQ0FBQyxTQUFTLENBQUMsQ0FBQztnQkFDNUIsUUFBUSxDQUFDLE9BQU8sQ0FBQyxVQUFVLENBQUMsQ0FBQztnQkFDN0IsUUFBUSxDQUFDLE9BQU8sQ0FBQyxVQUFVLENBQUMsQ0FBQztZQUMvQixDQUFDO1lBRUQsK0JBQStCO1lBQy9CLElBQUksTUFBTSxHQUFHLFVBQVUsQ0FBQyxPQUFPLENBQUMsV0FBVyxDQUFDLENBQUM7WUFFN0Msa0JBQWtCO1lBQ2xCLElBQUksSUFBSSxHQUFHLFVBQVUsQ0FBQyxPQUFPLENBQUMsVUFBVSxDQUFDLENBQUM7WUFFMUMsd0NBQXdDO1lBQ3hDLEVBQUUsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxjQUFjLENBQUM7Z0JBQ3hCLFVBQVUsQ0FBQyxPQUFPLENBQUMsVUFBVSxFQUFFLEtBQUssQ0FBQyxLQUFLLENBQUMsY0FBYyxDQUFDLFdBQVcsQ0FBQyxDQUFDO1lBRXpFLHFDQUFxQztZQUNyQyxFQUFFLENBQUMsQ0FBQyxNQUFNLENBQUMsV0FBVyxDQUFDO2dCQUNyQixXQUFXLENBQUMsT0FBTyxDQUFDLFVBQVUsQ0FBQyxDQUFDO1lBRWxDLG9DQUFvQztZQUNwQyxFQUFFLENBQUMsQ0FBQyxNQUFNLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQztnQkFDdEIsV0FBVyxDQUFDLE9BQU8sQ0FBQyxTQUFTLENBQUMsQ0FBQztnQkFDL0IsV0FBVyxDQUFDLE9BQU8sQ0FBQyxVQUFVLENBQUMsQ0FBQztnQkFDaEMsV0FBVyxDQUFDLE9BQU8sQ0FBQyxVQUFVLENBQUMsQ0FBQztZQUNsQyxDQUFDO1lBRUQsWUFBWTtZQUNaLFVBQVUsQ0FBQyxPQUFPLENBQUMsU0FBUyxFQUFFLE1BQU0sQ0FBQyxTQUFTLENBQUMsQ0FBQztZQUNoRCxVQUFVLENBQUMsT0FBTyxDQUFDLFVBQVUsRUFBRSxNQUFNLENBQUMsU0FBUyxDQUFDLENBQUM7WUFDakQsVUFBVSxDQUFDLE9BQU8sQ0FBQyxVQUFVLEVBQUUsTUFBTSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1lBRWpELFFBQVE7WUFDUixTQUFTLENBQUMsSUFBSSxDQUFDLEtBQUssRUFBRSxLQUFLLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxLQUFLLENBQUMsQ0FBQztZQUNsRCxVQUFVLENBQUMsSUFBSSxDQUFDLElBQUksRUFBRSxLQUFLLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxVQUFVLENBQUMsQ0FBQztZQUN2RCxVQUFVLENBQUMsSUFBSSxDQUFDLE1BQU0sRUFBRSxLQUFLLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxNQUFNLENBQUMsQ0FBQztZQUNyRCxFQUFFLENBQUMsQ0FBQyxRQUFRLElBQUksSUFBSSxDQUFDO2dCQUNuQixTQUFTLENBQUMsSUFBSSxDQUFDLElBQUksRUFBRSxLQUFLLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxLQUFLLENBQUMsQ0FBQztZQUVuRCxRQUFRLEdBQUcsVUFBVSxDQUFDO1FBQ3hCLENBQUM7UUFFRCxvQ0FBb0M7UUFDcEMsRUFBRSxDQUFDLENBQUMsTUFBTSxDQUFDLE9BQU8sQ0FBQztZQUNqQixVQUFVLENBQUMsT0FBTyxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBRWxDLHVDQUF1QztRQUN2QyxrQkFBTTtZQUNKLEtBQUssRUFBRSxVQUFVO1lBQ2pCLE1BQU0sRUFBRSxZQUFZO1lBQ3BCLE1BQU0sRUFBRSxXQUFXO1NBQ3BCLENBQUMsQ0FBQztRQUVILFVBQVU7UUFDVixJQUFJLENBQUMsT0FBTyxHQUFHLElBQUksT0FBTyxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUMzQyxDQUFDO0lBQ0gsV0FBQztBQUFELENBOUhBLEFBOEhDLEVBOUh5QixPQUFPLENBQUMsT0FBTyxFQThIeEM7QUE5SFksWUFBSSxPQThIaEIsQ0FBQTtBQUFBLENBQUM7Ozs7Ozs7OztBQ25JRixJQUFPLE9BQU8sV0FBWSxZQUFZLENBQUMsQ0FBQztBQUN4QyxJQUFPLE9BQU8sV0FBWSxZQUFZLENBQUMsQ0FBQztBQUN4QyxJQUFPLEtBQUssV0FBWSxVQUFVLENBQUMsQ0FBQztBQUdwQztJQUE0QiwwQkFBZTtJQUd6QyxnQkFBWSxNQUFNLEVBQUUsTUFBTSxFQUFFLE9BQU8sRUFBRSxXQUFXLEVBQUUsS0FBSztRQUVyRCxnQkFBZ0I7UUFDaEIsSUFBSSxVQUFVLEdBQUcsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ3pDLElBQUksV0FBVyxHQUFHLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUMxQyxJQUFJLFdBQVcsR0FBRyxJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUFDLENBQUM7UUFFM0Msd0RBQXdEO1FBQ3hELElBQUksT0FBTyxHQUFHLFdBQVcsQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUNwQyxJQUFJLGNBQWMsR0FBK0IsRUFBRSxDQUFDO1FBRXBELEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxXQUFXLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUNyQyw2QkFBNkI7WUFDN0IsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLE1BQU0sRUFBRSxHQUFHLE9BQU8sQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDO1lBQzlDLElBQUksRUFBRSxHQUFHLElBQUksQ0FBQyxNQUFNLEVBQUUsR0FBRyxPQUFPLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQztZQUM1QyxJQUFJLFVBQVUsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1lBQ3BELGNBQWMsQ0FBQyxJQUFJLENBQUMsVUFBVSxDQUFDLENBQUM7UUFDbEMsQ0FBQztRQUVELEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxLQUFLLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUMvQiw2QkFBNkI7WUFDN0IsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLE1BQU0sRUFBRSxHQUFHLE9BQU8sQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDO1lBQy9DLG1DQUFtQztZQUNuQyxJQUFJLGdCQUFnQixHQUFHLElBQUksQ0FBQyxNQUFNLEVBQUUsR0FBRyxjQUFjLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQztZQUNqRSxvQ0FBb0M7WUFDcEMsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDO1FBQ3hELENBQUM7UUFFRCxxQkFBcUI7UUFDckIsVUFBVSxDQUFDLE9BQU8sQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUNoQyxXQUFXLENBQUMsT0FBTyxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBRWpDLGdDQUFnQztRQUNoQyxrQkFBTTtZQUNKLEtBQUssRUFBRSxVQUFVO1lBQ2pCLE1BQU0sRUFBRSxDQUFDLFdBQVcsQ0FBQztZQUNyQixNQUFNLEVBQUUsV0FBVztTQUNwQixDQUFDLENBQUM7UUFFSCxVQUFVO1FBQ1YsSUFBSSxDQUFDLE9BQU8sR0FBRyxJQUFJLE9BQU8sQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDM0MsQ0FBQztJQUNILGFBQUM7QUFBRCxDQTdDQSxBQTZDQyxFQTdDMkIsT0FBTyxDQUFDLE9BQU8sRUE2QzFDO0FBN0NZLGNBQU0sU0E2Q2xCLENBQUE7Ozs7Ozs7OztBQ2xERCxJQUFPLE9BQU8sV0FBVyxZQUFZLENBQUMsQ0FBQztBQUN2QyxJQUFPLE9BQU8sV0FBVyxZQUFZLENBQUMsQ0FBQztBQUN2QyxJQUFPLEtBQUssV0FBVyxVQUFVLENBQUMsQ0FBQztBQUtuQztJQWdCRSxvQkFDRSxZQUFvQixFQUNwQixLQUFrQixFQUNsQixTQUFzQixFQUN0QixVQUF1QjtRQXBCM0IsaUJBZ0xDO1FBcktDLDRCQUF1QixHQUFXLENBQUMsQ0FBQztRQUNwQywwQkFBcUIsR0FBVyxDQUFDLENBQUM7UUFXaEMsSUFBSSxDQUFDLE1BQU0sR0FBRyxZQUFZLENBQUM7UUFFM0IsSUFBSSxDQUFDLElBQUksR0FBRyxJQUFJLEtBQUssQ0FBQyxZQUFZLENBQUMsQ0FBQztRQUVwQyxJQUFJLENBQUMsVUFBVSxHQUFHLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDO1FBRXBDLElBQUksQ0FBQyxLQUFLLEdBQUcsS0FBSyxDQUFDO1FBRW5CLElBQUksQ0FBQyxXQUFXLEdBQUcsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7UUFFdEQsSUFBSSxDQUFDLFdBQVcsQ0FBQyxPQUFPLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxLQUFLLENBQUMsY0FBYyxDQUFDLFVBQVUsQ0FBQyxDQUFDO1FBRXZFLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxRQUFRLEdBQUcsQ0FBQyxFQUFFLFFBQVEsR0FBRyxJQUFJLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxFQUFFLENBQUM7WUFDMUQsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsR0FBRyxJQUFJLFlBQVksQ0FBQyxJQUFJLENBQUMsVUFBVSxDQUFDLENBQUM7WUFDcEUsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEtBQUssQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztnQkFDdEMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxNQUFNLEVBQUUsQ0FBQztZQUMzQixDQUFDO1FBQ0gsQ0FBQztRQUNELGtCQUFrQjtRQUVsQixJQUFJLGdCQUErQixDQUFDO1FBQ3BDLCtDQUErQztRQUMvQyxJQUFJLGVBQWUsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDaEMsSUFBSSxlQUFlLEdBQUcsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsb0JBQW9CO1FBQzNELElBQUksS0FBSyxHQUFHLENBQUMsQ0FBQztRQUVkLElBQUksQ0FBQyxpQkFBaUIsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQzlELElBQUksQ0FBQyxrQkFBa0IsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFNBQVMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBRWhFLElBQUksR0FBaUIsQ0FBQztRQUV0QixJQUFJLENBQUMsS0FBSyxDQUFDLFNBQVMsR0FBRyxVQUFDLElBQVksRUFBRSxNQUErQjtZQUNuRSxLQUFJLENBQUMsa0JBQWtCLENBQUMsSUFBSSxFQUFFLE1BQU0sQ0FBQyxDQUFDO1lBRXRDLElBQUksT0FBTyxHQUFHLFNBQVMsQ0FBQyxpQkFBaUIsQ0FBQztZQUMxQyxJQUFJLFNBQVMsR0FBRyxVQUFVLENBQUMsaUJBQWlCLENBQUM7WUFFN0MsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLGVBQWUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztnQkFDaEQsSUFBSSxDQUFDLEdBQUcsS0FBSSxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFFdEMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztvQkFDbEMsdURBQXVEO29CQUN2RCxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLFNBQVMsQ0FBQyxDQUFDLENBQUMsR0FBRyxHQUFHLENBQUMsQ0FBQyxDQUFDLEdBQUcsZUFBZSxDQUFDLENBQUMsQ0FBQyxDQUFDO29CQUN2RCxxREFBcUQ7b0JBQ3JELENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxPQUFPLENBQUMsQ0FBQyxDQUFDLEdBQUcsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLGVBQWUsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUM7Z0JBQzFELENBQUM7WUFDSCxDQUFDO1FBQ0gsQ0FBQyxDQUFBO1FBRUQsSUFBSSxDQUFDLEtBQUssQ0FBQyxRQUFRLEdBQUcsVUFBQyxLQUE4QjtZQUNuRCxJQUFJLE1BQU0sR0FBRyxLQUFJLENBQUMsaUJBQWlCLENBQUMsS0FBSyxDQUFDLENBQUM7WUFDM0MsR0FBRyxHQUFHLFVBQVUsQ0FBQyxZQUFZLENBQUUsSUFBSSxZQUFZLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQztZQUV6RCxnQkFBZ0IsR0FBRyxLQUFJLENBQUMsa0JBQWtCLENBQUMsR0FBRyxDQUFDLENBQUM7WUFFaEQsS0FBSSxDQUFDLHFCQUFxQixHQUFHLENBQUMsQ0FBQztZQUUvQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsT0FBTyxHQUFHLENBQUMsRUFBRSxPQUFPLEdBQUcsZ0JBQWdCLENBQUMsTUFBTSxFQUFFLE9BQU8sRUFBRSxFQUFFLENBQUM7Z0JBQ25FLElBQUksQ0FBQyxHQUFHLGdCQUFnQixDQUFDLE9BQU8sQ0FBQyxDQUFDO2dCQUNsQyxFQUFFLENBQUEsQ0FBQyxDQUFDLEdBQUcsS0FBSSxDQUFDLHFCQUFxQixDQUFDLENBQUEsQ0FBQztvQkFDakMsS0FBSSxDQUFDLHFCQUFxQixHQUFHLENBQUMsQ0FBQztvQkFDL0IsS0FBSSxDQUFDLHVCQUF1QixHQUFHLE9BQU8sQ0FBQztnQkFDekMsQ0FBQztZQUNILENBQUM7WUFFRCxlQUFlLEdBQUcsQ0FBQyxLQUFJLENBQUMsdUJBQXVCLEdBQUcsQ0FBQyxFQUFFLEtBQUksQ0FBQyx1QkFBdUIsRUFBRSxLQUFJLENBQUMsdUJBQXVCLEdBQUcsQ0FBQyxDQUFDLENBQUM7WUFFckgsS0FBSyxHQUFHLEtBQUksQ0FBQyxxQkFBcUIsQ0FBQztZQUVuQyxlQUFlLEdBQUcsQ0FBQyxHQUFHLEVBQUUsR0FBRyxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUMsb0JBQW9CO1lBRXZELEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxlQUFlLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7Z0JBQ2hELElBQUksS0FBSyxHQUFHLGVBQWUsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFFL0IsRUFBRSxDQUFBLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQztvQkFDWCxLQUFLLElBQUksZ0JBQWdCLENBQUMsTUFBTSxDQUFDO2dCQUNuQyxJQUFJLENBQUMsRUFBRSxDQUFBLENBQUMsS0FBSyxJQUFJLGdCQUFnQixDQUFDLE1BQU0sQ0FBQztvQkFDdkMsS0FBSyxJQUFJLGdCQUFnQixDQUFDLE1BQU0sQ0FBQztnQkFFbkMsZUFBZSxDQUFDLENBQUMsQ0FBQyxHQUFHLEtBQUssQ0FBQztnQkFFM0IsZUFBZSxDQUFDLENBQUMsQ0FBQyxHQUFHLGVBQWUsQ0FBQyxDQUFDLENBQUMsR0FBRyxLQUFLLEdBQUcsZ0JBQWdCLENBQUMsS0FBSyxDQUFDLENBQUM7WUFDNUUsQ0FBQztZQUVELEtBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxVQUFDLE1BQU0sRUFBRSxDQUFDO2dCQUN0QyxzQ0FBc0M7Z0JBQ3RDLElBQUksTUFBTSxHQUFHLENBQUMsQ0FBQztnQkFFZixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsZUFBZSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO29CQUNoRCxNQUFNLElBQUksS0FBSSxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxlQUFlLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ2xFLENBQUM7Z0JBRUQsTUFBTSxDQUFDLFFBQVEsQ0FBQyxNQUFNLENBQUMsQ0FBQztZQUMxQixDQUFDLENBQUMsQ0FBQztZQUVILE1BQU0sQ0FBQyxNQUFNLENBQUM7UUFDaEIsQ0FBQyxDQUFBO0lBQ0gsQ0FBQztJQUVNLHdCQUFhLEdBQXBCLFVBQXFCLE1BQThCLEVBQUUsTUFBOEI7UUFDakYsaURBQWlEO1FBQ2pELGlCQUFpQjtRQUNqQixJQUFJLEtBQUssR0FBRyxDQUFDLENBQUM7UUFFZCxJQUFJLEtBQUssR0FBRyxDQUFDLEVBQUUsS0FBSyxHQUFHLENBQUMsQ0FBQztRQUV6QixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1lBQ3ZDLEtBQUssSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQy9CLEtBQUssSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQy9CLEtBQUssSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2pDLENBQUM7UUFFRCxNQUFNLENBQUMsS0FBSyxHQUFHLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxHQUFHLE1BQU0sQ0FBQyxDQUFDO0lBQ2hFLENBQUM7SUFFTSx1QkFBWSxHQUFuQixVQUFzRCxLQUFRLEVBQUUsT0FBVztRQUN6RSxvQkFBb0I7UUFDcEIscUJBQXFCO1FBQ3JCLGdCQUFnQjtRQUNoQixxQ0FBcUM7UUFKeUIsdUJBQVcsR0FBWCxXQUFXO1FBTXpFLEVBQUUsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQztZQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUM7UUFFaEMsT0FBTyxHQUFHLE9BQU8sSUFBSSxDQUFDLENBQUM7UUFFdkIsSUFBSSxHQUFHLEdBQUcsQ0FBQyxDQUFDO1FBRVoscUJBQXFCO1FBQ3JCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxLQUFLLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7WUFDdEMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsT0FBTyxHQUFHLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3hDLEdBQUcsSUFBSSxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbEIsQ0FBQztRQUVELEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ2IsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEtBQUssQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFO2dCQUFFLEtBQUssQ0FBQyxDQUFDLENBQUMsSUFBSSxHQUFHLENBQUM7UUFDekQsQ0FBQztRQUFDLElBQUksQ0FBQyxDQUFDO1lBQ04sSUFBSSxHQUFHLEdBQUcsQ0FBQyxHQUFHLEtBQUssQ0FBQyxNQUFNLENBQUM7WUFDM0IsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEtBQUssQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFO2dCQUFFLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxHQUFHLENBQUM7UUFDeEQsQ0FBQztRQUVELE1BQU0sQ0FBQyxLQUFLLENBQUM7SUFDZixDQUFDO0lBRUQsOERBQThEO0lBQzlELHVDQUFrQixHQUFsQixVQUFtQixPQUErQjtRQUNoRCwyQkFBMkI7UUFDM0IsSUFBSSxTQUFTLEdBQUcsSUFBSSxZQUFZLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUVuRCxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFO1lBQ3ZDLFNBQVMsQ0FBQyxDQUFDLENBQUMsR0FBRyxVQUFVLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFFakUsTUFBTSxDQUFDLFNBQVMsQ0FBQztJQUNuQixDQUFDO0lBQ0gsaUJBQUM7QUFBRCxDQWhMQSxBQWdMQyxJQUFBO0FBaExZLGtCQUFVLGFBZ0x0QixDQUFBO0FBR0Q7SUFBaUMsK0JBQWU7SUFJOUMscUJBQVksU0FBaUIsRUFBRSxZQUFvQixFQUFFLFdBQW1CLEVBQUUsVUFBa0I7UUFHMUYsSUFBSSxNQUFNLEdBQUc7WUFDWCxTQUFTLEVBQUUsS0FBSyxDQUFDLEtBQUssQ0FBQyxjQUFjLENBQUMsVUFBVTtZQUNoRCxjQUFjLEVBQUUsS0FBSztZQUNyQixXQUFXLEVBQUUsS0FBSztZQUNsQixVQUFVLEVBQUUsS0FBSztZQUNqQixPQUFPLEVBQUUsSUFBSTtTQUNkLENBQUM7UUFFRixJQUFJLFVBQVUsR0FBRyxJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDNUMsSUFBSSxZQUFZLEdBQUcsRUFBRSxDQUFDO1FBQ3RCLElBQUksV0FBVyxHQUFHLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxVQUFVLENBQUMsQ0FBQztRQUc5Qyx5QkFBeUI7UUFFekIsNERBQTREO1FBRTVELElBQUksU0FBUyxHQUFHLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxXQUFXLENBQUMsQ0FBQyxHQUFHLENBQUM7WUFDL0MsSUFBSSxFQUFFLENBQUM7U0FDUixDQUFDLENBQUM7UUFFSCxJQUFJLFVBQVUsR0FBRyxJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsV0FBVyxDQUFDLENBQUMsR0FBRyxDQUFDO1lBQ2hELElBQUksRUFBRSxDQUFDO1NBQ1IsQ0FBQyxDQUFDO1FBRUgsSUFBSSxVQUFVLEdBQUcsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBSTlDLElBQUksVUFBVSxHQUFHLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxXQUFXLENBQUMsQ0FBQyxHQUFHLENBQUM7WUFDaEQsSUFBSSxFQUFFLENBQUM7U0FDUixDQUFDLENBQUM7UUFFSCxZQUFZLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQzdCLFlBQVksQ0FBQyxJQUFJLENBQUMsVUFBVSxDQUFDLENBQUM7UUFDOUIsWUFBWSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUMsQ0FBQztRQUM5QixZQUFZLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxDQUFDO1FBRTlCLCtCQUErQjtRQUMvQixJQUFJLEtBQUssR0FBRyxVQUFVLENBQUMsT0FBTyxDQUFDLFVBQVUsQ0FBQyxDQUFDO1FBQzNDLFVBQVUsQ0FBQyxPQUFPLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDOUIsVUFBVSxDQUFDLE9BQU8sQ0FBQyxVQUFVLENBQUMsQ0FBQztRQUMvQixVQUFVLENBQUMsT0FBTyxDQUFDLFVBQVUsQ0FBQyxDQUFDO1FBRS9CLCtCQUErQjtRQUMvQixJQUFJLE1BQU0sR0FBRyxVQUFVLENBQUMsT0FBTyxDQUFDLFdBQVcsQ0FBQyxDQUFDO1FBRTdDLGtCQUFrQjtRQUNsQixJQUFJLElBQUksR0FBRyxVQUFVLENBQUMsT0FBTyxDQUFDLFVBQVUsQ0FBQyxDQUFDO1FBRTFDLFlBQVk7UUFDWixVQUFVLENBQUMsT0FBTyxDQUFDLFNBQVMsRUFBRSxNQUFNLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDaEQsVUFBVSxDQUFDLE9BQU8sQ0FBQyxVQUFVLEVBQUUsTUFBTSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQ2pELFVBQVUsQ0FBQyxPQUFPLENBQUMsVUFBVSxFQUFFLE1BQU0sQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUVqRCxRQUFRO1FBQ1IsU0FBUyxDQUFDLElBQUksQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDbEQsVUFBVSxDQUFDLElBQUksQ0FBQyxJQUFJLEVBQUUsS0FBSyxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsVUFBVSxDQUFDLENBQUM7UUFDdkQsVUFBVSxDQUFDLElBQUksQ0FBQyxNQUFNLEVBQUUsS0FBSyxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsTUFBTSxDQUFDLENBQUM7UUFHckQsSUFBSSxDQUFDLFVBQVUsR0FBRyxJQUFJLFVBQVUsQ0FBQyxZQUFZLEVBQUUsVUFBVSxFQUFFLFNBQVMsRUFBRSxVQUFVLENBQUMsQ0FBQztRQUVsRixZQUFZLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUMsV0FBVyxDQUFDLENBQUM7UUFFL0MsWUFBWTtRQUVaLG9DQUFvQztRQUNwQyxFQUFFLENBQUMsQ0FBQyxNQUFNLENBQUMsT0FBTyxDQUFDO1lBQ2pCLFVBQVUsQ0FBQyxPQUFPLENBQUMsV0FBVyxDQUFDLENBQUM7UUFFbEMsdUNBQXVDO1FBQ3ZDLGtCQUFNO1lBQ0osS0FBSyxFQUFFLFVBQVU7WUFDakIsTUFBTSxFQUFFLFlBQVk7WUFDcEIsTUFBTSxFQUFFLFdBQVc7U0FDcEIsQ0FBQyxDQUFDO1FBR0gsK0JBQStCO1FBQy9CLElBQUksQ0FBQyxTQUFTLEdBQUcsS0FBSyxDQUFDO1FBRXZCLFVBQVU7UUFDVixJQUFJLENBQUMsT0FBTyxHQUFHLElBQUksT0FBTyxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUMzQyxDQUFDO0lBQ0gsa0JBQUM7QUFBRCxDQTVGQSxBQTRGQyxFQTVGZ0MsT0FBTyxDQUFDLE9BQU8sRUE0Ri9DO0FBNUZZLG1CQUFXLGNBNEZ2QixDQUFBO0FBQUEsQ0FBQzs7Ozs7Ozs7O0FDdFJGLElBQU8sT0FBTyxXQUFZLFlBQVksQ0FBQyxDQUFDO0FBQ3hDLElBQU8sT0FBTyxXQUFZLFlBQVksQ0FBQyxDQUFDO0FBQ3hDLElBQU8sS0FBSyxXQUFZLFVBQVUsQ0FBQyxDQUFDO0FBRXBDLHdCQUF3QjtBQUN4QjtJQUFnQyw4QkFBZTtJQUc3QztRQUFZLGNBQWlCO2FBQWpCLFdBQWlCLENBQWpCLHNCQUFpQixDQUFqQixJQUFpQjtZQUFqQiw2QkFBaUI7O1FBRTNCLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDO1lBQ2xCLE1BQU0seUNBQXlDLENBQUM7UUFFbEQsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsaUJBQWlCO1FBQzVDLElBQUksT0FBTyxHQUFHLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLGdCQUFnQjtRQUMxQyxJQUFJLE1BQU0sR0FBRyxJQUFJLENBQUMsQ0FBQyxrQ0FBa0M7UUFFckQsSUFBSSxLQUFLLEdBQUcsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ3BDLElBQUksTUFBTSxHQUFHLEVBQUUsQ0FBQztRQUNoQixJQUFJLE1BQU0sR0FBRyxJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUFDLENBQUM7UUFFdEMsSUFBSSxRQUFRLEdBQUcsS0FBSyxDQUFDO1FBRXJCLHlCQUF5QjtRQUN6QixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsS0FBSyxJQUFJLE1BQU0sQ0FBQyxDQUFDLENBQUM7WUFDekIsSUFBSSxJQUFJLEdBQUcsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDO1lBQ3pCLElBQUksUUFBUSxHQUFHLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQztZQUNyQyxNQUFNLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDO1lBQ3RCLFFBQVEsQ0FBQyxPQUFPLENBQUMsUUFBUSxDQUFDLENBQUM7WUFDM0IsUUFBUSxHQUFHLFFBQVEsQ0FBQztRQUN0QixDQUFDO1FBQ0QsUUFBUSxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUV6QixtQ0FBbUM7UUFFbkMsa0JBQU07WUFDSixLQUFLLEVBQUUsS0FBSztZQUNaLE1BQU0sRUFBRSxNQUFNO1lBQ2QsTUFBTSxFQUFFLE1BQU07U0FDZixDQUFDLENBQUM7UUFFSCwwQkFBMEI7UUFDMUIsSUFBSSxDQUFDLE9BQU8sR0FBRyxJQUFJLE9BQU8sQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDM0MsQ0FBQztJQUNILGlCQUFDO0FBQUQsQ0F2Q0EsQUF1Q0MsRUF2QytCLE9BQU8sQ0FBQyxPQUFPLEVBdUM5QztBQXZDWSxrQkFBVSxhQXVDdEIsQ0FBQTtBQUFBLENBQUM7OztBQzVDRixJQUFPLE1BQU0sV0FBVyxVQUFVLENBQUMsQ0FBQztBQUNwQyxJQUFPLE9BQU8sV0FBVyxXQUFXLENBQUMsQ0FBQztBQUd0Qzs7NEZBRTRGO0FBQzVGO0lBUUUsZUFBWSxJQUFZLEVBQUUsS0FBYztRQVB4QyxTQUFJLEdBQW9CLEVBQUUsQ0FBQztRQUMzQixVQUFLLEdBQVcsSUFBSSxDQUFDO1FBQ3JCLGdCQUFXLEdBQUcsRUFBRSxDQUFDO1FBQ2pCLFNBQUksR0FBRyxDQUFDLENBQUM7UUFLUixJQUFJLENBQUMsSUFBSSxHQUFHLElBQUksR0FBRyxDQUFDLENBQUM7UUFDckIsSUFBSSxDQUFDLElBQUksR0FBRyxFQUFFLENBQUM7UUFDZixJQUFJLENBQUMsS0FBSyxHQUFHLEtBQUssSUFBSSxJQUFJLENBQUM7UUFDM0IsSUFBSSxDQUFDLFdBQVcsR0FBRyxFQUFFLENBQUM7UUFFdEIsSUFBSSxDQUFDLGlCQUFpQixHQUFHLElBQUksWUFBWSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBRWhELE9BQU8sSUFBSSxFQUFFLEVBQUUsQ0FBQztZQUNmLElBQUksU0FBUyxHQUFHLElBQUksTUFBTSxDQUFDLE1BQU0sRUFBRSxDQUFDO1lBQ3BDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQzNCLENBQUM7SUFDRixDQUFDO0lBR0QseUNBQXlDO0lBQ3pDLHdCQUFRLEdBQVIsVUFBUyxLQUE4QjtRQUV0QyxFQUFFLENBQUEsQ0FBQyxJQUFJLENBQUMsaUJBQWlCLENBQUMsTUFBTSxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDO1lBQ3BELElBQUksQ0FBQyxpQkFBaUIsR0FBRyxJQUFJLFlBQVksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBRTdELElBQUksZUFBZSxHQUFHLENBQUMsQ0FBQztRQUV4QixFQUFFLENBQUMsQ0FBQyxPQUFPLEtBQUssSUFBSSxXQUFXLENBQUMsQ0FBQyxDQUFDO1lBQ2pDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxNQUFNLElBQUksSUFBSSxDQUFDLElBQUksQ0FBQztnQkFDN0IsTUFBTSx5REFBeUQsQ0FBQztZQUVqRSxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDO2dCQUMxQixJQUFJLENBQUMsaUJBQWlCLENBQUMsZUFBZSxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLFFBQVEsQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztZQUMvRSxDQUFDO1FBQ0YsQ0FBQztRQUFDLElBQUksQ0FBQyxDQUFDO1lBQ1AsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsSUFBSSxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQztnQkFDMUIsSUFBSSxDQUFDLGlCQUFpQixDQUFDLGVBQWUsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQyxRQUFRLEVBQUUsQ0FBQztZQUN0RSxDQUFDO1FBQ0YsQ0FBQztRQUVELE1BQU0sQ0FBQyxJQUFJLENBQUMsaUJBQWlCLENBQUM7SUFDL0IsQ0FBQztJQUVELHVEQUF1RDtJQUN2RCx5QkFBUyxHQUFULFVBQVUsSUFBWSxFQUFFLE1BQWdDO1FBQ3ZELEVBQUUsQ0FBQyxDQUFDLE9BQU8sTUFBTSxJQUFJLFdBQVcsQ0FBQyxDQUFDLENBQUM7WUFDbEMsRUFBRSxDQUFDLENBQUMsTUFBTSxDQUFDLE1BQU0sSUFBSSxJQUFJLENBQUMsSUFBSSxDQUFDO2dCQUM5QixNQUFNLDJEQUEyRCxDQUFDO1lBRW5FLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsQ0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQztnQkFDbkQsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFDM0IsTUFBTSxDQUFDLFNBQVMsQ0FBQyxJQUFJLEVBQUUsTUFBTSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7WUFDcEMsQ0FBQztRQUNGLENBQUM7UUFBQyxJQUFJLENBQUMsQ0FBQztZQUNQLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxNQUFNLEdBQUcsQ0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQztnQkFDbkQsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFDM0IsTUFBTSxDQUFDLFNBQVMsQ0FBQyxJQUFJLENBQUMsQ0FBQztZQUN4QixDQUFDO1FBQ0YsQ0FBQztJQUNGLENBQUM7SUFFRCx1REFBdUQ7SUFDdkQsdUJBQU8sR0FBUCxVQUFRLEtBQStCLEVBQUUsSUFBSyxFQUFFLE9BQWlDO1FBRWhGLEVBQUUsQ0FBQyxDQUFDLEtBQUssWUFBWSxPQUFPLENBQUMsT0FBTyxDQUFDO1lBQ3BDLEtBQUssR0FBcUIsS0FBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUM7UUFFL0MsRUFBRSxDQUFDLENBQUMsS0FBSyxZQUFZLEtBQUssQ0FBQyxDQUFDLENBQUM7WUFDNUIsRUFBRSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLEtBQUssQ0FBQyxDQUFDO2dCQUMxQixNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsZUFBZSxDQUFDLElBQUksRUFBRSxLQUFLLEVBQUUsSUFBSSxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQy9ELENBQUM7UUFBQyxJQUFJO1lBQ0wsTUFBTSw0RUFBNEUsQ0FBQztJQUdyRixDQUFDO0lBRUQsd0NBQXdDO0lBQ3hDLG9CQUFJLEdBQUosVUFBSyxVQUFVLEVBQUUsSUFBSTtRQUVwQixFQUFFLENBQUMsQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLFFBQVEsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDO1lBQ2xDLEVBQUUsQ0FBQyxDQUFDLFVBQVUsQ0FBQyxFQUFFLENBQUMsSUFBSSxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUM7Z0JBQ25DLE1BQU0sNkVBQTZFLENBQUM7WUFFckYsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsSUFBSSxVQUFVLENBQUMsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7Z0JBQ25DLElBQUksTUFBTSxHQUFHLFVBQVUsQ0FBQyxFQUFFLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDO2dCQUNwQyxJQUFJLEtBQUssR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDO2dCQUMxQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsS0FBSyxJQUFJLE1BQU0sQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQztvQkFDN0MsSUFBSSxLQUFLLEdBQUcsTUFBTSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUM7b0JBQzdDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxFQUFFLElBQUksVUFBVSxDQUFDLFdBQVcsQ0FBQzt3QkFDdEMsS0FBSyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztnQkFDcEIsQ0FBQztZQUNGLENBQUM7UUFDRixDQUFDO1FBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLElBQUksSUFBSSxLQUFLLENBQUMsUUFBUSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUM7WUFDMUMsRUFBRSxDQUFDLENBQUMsVUFBVSxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksSUFBSSxDQUFDLElBQUksQ0FBQztnQkFDckMsTUFBTSwrRUFBK0UsQ0FBQztZQUV2RixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxJQUFJLFVBQVUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQztnQkFDckMsSUFBSSxNQUFNLEdBQUcsVUFBVSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBQ3RDLElBQUksS0FBSyxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBQzFCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxTQUFTLElBQUksTUFBTSxDQUFDLFdBQVcsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDO29CQUNwRCxJQUFJLEtBQUssR0FBRyxNQUFNLENBQUMsV0FBVyxDQUFDLFNBQVMsQ0FBQyxTQUFTLENBQUMsQ0FBQztvQkFDcEQsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLEVBQUUsSUFBSSxVQUFVLENBQUMsV0FBVyxDQUFDO3dCQUN0QyxLQUFLLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO2dCQUNwQixDQUFDO1lBQ0YsQ0FBQztRQUNGLENBQUM7UUFBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUMsSUFBSSxJQUFJLEtBQUssQ0FBQyxRQUFRLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQztZQUM5QyxFQUFFLENBQUMsQ0FBQyxVQUFVLENBQUMsSUFBSSxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUM7Z0JBQ2hDLE1BQU0sa0ZBQWtGLENBQUM7WUFFMUYsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsSUFBSSxVQUFVLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQztnQkFDaEMsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFDMUIsSUFBSSxLQUFLLEdBQUcsVUFBVSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFDaEMsS0FBSyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztZQUNuQixDQUFDO1FBQ0YsQ0FBQztRQUNELFVBQVUsQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDLEVBQUUsS0FBSyxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQztJQUN4RCxDQUFDO0lBRUQsaUVBQWlFO0lBQ2pFLDZCQUFhLEdBQWI7UUFFQyxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDO1lBQzFCLElBQUksTUFBTSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUM7WUFDM0IsRUFBRSxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUMsYUFBYSxFQUFFLENBQUM7Z0JBQzNCLE1BQU0sQ0FBQyxLQUFLLENBQUM7UUFDZixDQUFDO1FBQ0QsTUFBTSxDQUFDLElBQUksQ0FBQztJQUNiLENBQUM7SUFFRCxtRkFBbUY7SUFDbkYseUJBQVMsR0FBVCxVQUFVLEtBQUs7UUFDZCxpQ0FBaUM7UUFDakMsSUFBSSxXQUFXLEdBQUcsQ0FBQyxDQUFDO1FBQ3BCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxJQUFJLElBQUksSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDNUIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEtBQUssSUFBSSxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQztnQkFDOUIsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQztnQkFDM0IsSUFBSSxFQUFFLEdBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztnQkFDM0IsSUFBSSxTQUFTLEdBQUcsSUFBSSxDQUFDLFNBQVMsQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFDbkMsRUFBRSxDQUFDLENBQUMsU0FBUyxJQUFJLFNBQVMsQ0FBQyxJQUFJLElBQUksV0FBVyxDQUFDO29CQUM5QyxXQUFXLEVBQUUsQ0FBQztZQUNoQixDQUFDO1FBQ0YsQ0FBQztRQUNELEVBQUUsQ0FBQyxDQUFDLFdBQVcsSUFBSSxJQUFJLENBQUMsSUFBSSxHQUFHLEtBQUssQ0FBQyxJQUFJLENBQUM7WUFDekMsTUFBTSxDQUFDLEtBQUssQ0FBQyxjQUFjLENBQUMsVUFBVSxDQUFDO1FBRXhDLGlDQUFpQztRQUNqQyxXQUFXLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxNQUFNLElBQUksSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDOUIsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztZQUM3QixJQUFJLEVBQUUsR0FBRyxLQUFLLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1lBQzVCLElBQUksU0FBUyxHQUFHLElBQUksQ0FBQyxTQUFTLENBQUMsRUFBRSxDQUFDLENBQUM7WUFDbkMsRUFBRSxDQUFDLENBQUMsU0FBUyxJQUFJLFNBQVMsQ0FBQyxJQUFJLElBQUksV0FBVyxDQUFDO2dCQUM5QyxXQUFXLEVBQUUsQ0FBQztRQUNoQixDQUFDO1FBQ0QsRUFBRSxDQUFDLENBQUMsV0FBVyxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUM7WUFDNUIsTUFBTSxDQUFDLEtBQUssQ0FBQyxjQUFjLENBQUMsVUFBVSxDQUFDO0lBQ3pDLENBQUM7SUFFRCxzQ0FBc0M7SUFDdEMscUJBQUssR0FBTDtRQUNDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLElBQUksSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDMUIsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztZQUMzQixNQUFNLENBQUMsS0FBSyxFQUFFLENBQUM7UUFDaEIsQ0FBQztJQUNGLENBQUM7SUFFRCxzQ0FBc0M7SUFDdEMscUJBQUssR0FBTDtRQUNDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLElBQUksSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDMUIsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztZQUMzQixNQUFNLENBQUMsS0FBSyxFQUFFLENBQUM7UUFDaEIsQ0FBQztJQUNGLENBQUM7SUFFRCwrQ0FBK0M7SUFDL0MsdUJBQU8sR0FBUDtRQUNDLE1BQU0sQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDO0lBQ2xCLENBQUM7SUFFRCw2QkFBNkI7SUFDN0IsbUJBQUcsR0FBSCxVQUFJLE1BQU07UUFDVCxNQUFNLEdBQUcsTUFBTSxJQUFJLElBQUksTUFBTSxDQUFDLE1BQU0sRUFBRSxDQUFDO1FBQ3ZDLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQztRQUNqQyxJQUFJLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUN2QixJQUFJLENBQUMsSUFBSSxFQUFFLENBQUM7SUFDYixDQUFDO0lBRUQsbUJBQUcsR0FBSCxVQUFJLE9BQU87UUFDVixPQUFPLEdBQUcsT0FBTyxJQUFJLEVBQUUsQ0FBQztRQUV4QixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDO1lBQ3pCLElBQUksTUFBTSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDMUIsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQztnQkFDakIsTUFBTSxDQUFDLEtBQUssR0FBRyxPQUFPLENBQUMsS0FBSyxHQUFHLEdBQUcsR0FBRyxNQUFNLENBQUMsRUFBRSxDQUFDO1lBQ2hELEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUM7Z0JBQ2xCLE1BQU0sQ0FBQyxNQUFNLEdBQUcsT0FBTyxDQUFDLE1BQU0sQ0FBQztZQUNoQyxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDO2dCQUNoQixNQUFNLENBQUMsSUFBSSxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUM7UUFDN0IsQ0FBQztRQUNELE1BQU0sQ0FBQyxJQUFJLENBQUM7SUFDYixDQUFDO0lBQ0YsWUFBQztBQUFELENBN01ELEFBNk1FLElBQUE7QUE3TVcsYUFBSyxRQTZNaEIsQ0FBQTtBQUdGLElBQWMsS0FBSyxDQStFbEI7QUEvRUQsV0FBYyxLQUFLLEVBQUMsQ0FBQztJQUNULGNBQVEsR0FBRyxDQUFDLENBQUM7SUFDeEI7UUFDQyxNQUFNLENBQUMsY0FBUSxFQUFFLENBQUM7SUFDbkIsQ0FBQztJQUZlLFNBQUcsTUFFbEIsQ0FBQTtJQUVELHVCQUF1QjtJQUNaLG9CQUFjLEdBQUc7UUFDM0IsVUFBVSxFQUFFLFlBQVk7UUFDeEIsVUFBVSxFQUFFLFlBQVk7UUFDeEIsV0FBVyxFQUFFLGFBQWE7S0FDMUIsQ0FBQztJQUVGLGlCQUFpQjtJQUNOLGNBQVEsR0FBRztRQUNyQixLQUFLLEVBQUUsT0FBTztRQUNkLE1BQU0sRUFBRSxRQUFRO1FBQ2hCLFVBQVUsRUFBRSxZQUFZO0tBQ3hCLENBQUM7SUFFRiw0RkFBNEY7SUFDNUY7UUFXQyx5QkFBWSxTQUFTLEVBQUUsT0FBTyxFQUFFLElBQUksRUFBRSxPQUFPO1lBVjdDLE9BQUUsR0FBRyxHQUFHLEVBQUUsQ0FBQztZQUdYLG1CQUFjLEdBQWEsS0FBSyxDQUFDO1lBSWpDLFNBQUksR0FBRyxDQUFDLENBQUM7WUFDVCxjQUFTLEdBQUcsRUFBRSxDQUFDO1lBR2QsSUFBSSxDQUFDLElBQUksR0FBRyxTQUFTLENBQUM7WUFDdEIsSUFBSSxDQUFDLEVBQUUsR0FBRyxPQUFPLENBQUM7WUFDbEIsSUFBSSxDQUFDLGNBQWMsR0FBRyxPQUFPLElBQUksU0FBUyxDQUFDO1lBQzNDLElBQUksQ0FBQyxJQUFJLEdBQUcsSUFBSSxDQUFDO1lBQ2pCLElBQUksQ0FBQyxXQUFXLEdBQUcsRUFBRSxDQUFDO1lBQ3RCLElBQUksQ0FBQyxJQUFJLEdBQUcsRUFBRSxDQUFDO1lBQ2YsSUFBSSxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7WUFDZCxJQUFJLENBQUMsU0FBUyxHQUFHLEVBQUUsQ0FBQztZQUdwQixFQUFFLENBQUMsQ0FBQyxPQUFPLElBQUksQ0FBQyxJQUFJLElBQUksV0FBVyxDQUFDLENBQUMsQ0FBQztnQkFDckMsRUFBRSxDQUFDLENBQUMsU0FBUyxJQUFJLE9BQU8sQ0FBQztvQkFDeEIsSUFBSSxDQUFDLElBQUksR0FBRyxLQUFLLENBQUMsY0FBYyxDQUFDLFVBQVUsQ0FBQztnQkFDN0MsSUFBSTtvQkFDSCxJQUFJLENBQUMsSUFBSSxHQUFHLEtBQUssQ0FBQyxjQUFjLENBQUMsVUFBVSxDQUFDO1lBQzlDLENBQUM7WUFFRCxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLEtBQUssQ0FBQyxjQUFjLENBQUMsVUFBVTtnQkFDL0MsSUFBSSxDQUFDLElBQUksSUFBSSxLQUFLLENBQUMsY0FBYyxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUM7Z0JBQ2hELEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxJQUFJLElBQUksSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDO29CQUNqQyxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsS0FBSyxJQUFJLElBQUksQ0FBQyxFQUFFLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQzt3QkFDaEMsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7d0JBQ2hDLElBQUksRUFBRSxHQUFHLElBQUksQ0FBQyxFQUFFLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO3dCQUM3QixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxJQUFJLEtBQUssQ0FBQyxjQUFjLENBQUMsV0FBVyxJQUFJLElBQUksSUFBSSxFQUFFLENBQUM7NEJBQy9ELFFBQVEsQ0FBQzt3QkFDVixJQUFJLFVBQVUsR0FBRyxJQUFJLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxPQUFPLENBQUMsQ0FBQzt3QkFFM0MsSUFBSSxDQUFDLFdBQVcsQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLEdBQUcsVUFBVSxDQUFDO3dCQUM3QyxJQUFJLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxDQUFDO29CQUN4QyxDQUFDO2dCQUNGLENBQUM7WUFDRixDQUFDO1lBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksS0FBSyxDQUFDLGNBQWMsQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDO2dCQUV6RCxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsTUFBTSxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQztvQkFDbkMsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7b0JBQ2xDLElBQUksRUFBRSxHQUFHLElBQUksQ0FBQyxFQUFFLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO29CQUM5QixJQUFJLFVBQVUsR0FBRyxJQUFJLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxPQUFPLENBQUMsQ0FBQztvQkFFM0MsSUFBSSxDQUFDLFdBQVcsQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLEdBQUcsVUFBVSxDQUFDO29CQUM3QyxJQUFJLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxDQUFDO2dCQUN4QyxDQUFDO1lBQ0YsQ0FBQztZQUVELFNBQVMsQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ2xDLENBQUM7UUFDRixzQkFBQztJQUFELENBekRBLEFBeURDLElBQUE7SUF6RFkscUJBQWUsa0JBeUQzQixDQUFBO0FBQ0YsQ0FBQyxFQS9FYSxLQUFLLEdBQUwsYUFBSyxLQUFMLGFBQUssUUErRWxCOzs7QUN0U0QsSUFBTyxLQUFLLFdBQVcsU0FBUyxDQUFDLENBQUM7QUFDbEMsSUFBTyxNQUFNLFdBQVcsVUFBVSxDQUFDLENBQUM7QUFFcEMsSUFBTyxPQUFPLFdBQVcsVUFBVSxDQUFDLENBQUM7QUFTckM7SUFPQyxpQkFBWSxNQUFNO1FBTmxCLGNBQVMsR0FBRyxJQUFJLENBQUM7UUFDakIsV0FBTSxHQUFHO1lBQ1IsS0FBSyxFQUFFLElBQUk7WUFDWCxNQUFNLEVBQUUsRUFBRTtZQUNWLE1BQU0sRUFBRSxJQUFJO1NBQ1osQ0FBQztRQUVELEVBQUUsQ0FBQyxDQUFDLE9BQU8sTUFBTSxJQUFJLFdBQVcsQ0FBQyxDQUFDLENBQUM7WUFDbEMsSUFBSSxDQUFDLE1BQU0sR0FBRyxNQUFNLElBQUk7Z0JBQ3ZCLEtBQUssRUFBRSxJQUFJO2dCQUNYLE1BQU0sRUFBRSxFQUFFO2dCQUNWLE1BQU0sRUFBRSxJQUFJO2FBQ1osQ0FBQztRQUNILENBQUM7SUFDRixDQUFDO0lBRUQsZ0VBQWdFO0lBQ2hFLDBCQUFRLEdBQVIsVUFBUyxLQUE4QjtRQUV0QyxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxLQUFLLEtBQUssQ0FBQyxDQUFDLENBQUM7WUFDOUIsSUFBSSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsUUFBUSxDQUFDLEtBQUssQ0FBQyxDQUFDO1lBQ2xDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLElBQUksSUFBSSxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUM7Z0JBQ3BDLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLFFBQVEsRUFBRSxDQUFDO1lBQ3RDLE1BQU0sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxRQUFRLEVBQUUsQ0FBQztRQUN0QyxDQUFDO1FBQ0QsSUFBSSxDQUFDLENBQUM7WUFDTCxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxJQUFJLElBQUksQ0FBQztnQkFDMUIsSUFBSSxDQUFDLFFBQVEsRUFBRSxDQUFDO1lBQ2pCLE1BQU0sQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLFFBQVEsQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUN2QyxDQUFDO0lBQ0YsQ0FBQztJQUVELDRDQUE0QztJQUM1QywyQkFBUyxHQUFULFVBQVUsSUFBWSxFQUFFLE1BQStCO1FBRXRELEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLEtBQUssS0FBSyxDQUFDLENBQUMsQ0FBQztZQUM5QixJQUFJLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxTQUFTLENBQUMsSUFBSSxFQUFFLE1BQU0sQ0FBQyxDQUFDO1lBQzNDLElBQUksT0FBTyxHQUFHLEVBQUUsQ0FBQztZQUNqQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsS0FBSyxJQUFJLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDO2dCQUNwQyxPQUFPLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7WUFDekMsT0FBTyxDQUFDLE9BQU8sRUFBRSxDQUFDO1lBQ2xCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLElBQUksT0FBTyxDQUFDO2dCQUN6QixPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsU0FBUyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ2pDLENBQUM7UUFDRCxJQUFJLENBQUMsQ0FBQztZQUNMLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLElBQUksSUFBSSxDQUFDO2dCQUMxQixJQUFJLENBQUMsUUFBUSxFQUFFLENBQUM7WUFDakIsSUFBSSxDQUFDLFNBQVMsQ0FBQyxTQUFTLENBQUMsSUFBSSxFQUFFLE1BQU0sQ0FBQyxDQUFDO1FBQ3hDLENBQUM7SUFDRixDQUFDO0lBRUQscUVBQXFFO0lBQ3JFLHlCQUFPLEdBQVAsVUFBUSxJQUFJLEVBQUUsSUFBSSxFQUFFLE9BQU87UUFFMUIsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQztZQUNsQixJQUFJLENBQUMsU0FBUyxDQUFDLEtBQUssRUFBRSxDQUFDO1FBRXhCLEVBQUUsQ0FBQyxDQUFDLElBQUksWUFBWSxPQUFPLENBQUM7WUFDM0IsTUFBTSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLEtBQUssRUFBRSxJQUFJLEVBQUUsT0FBTyxDQUFDLENBQUM7UUFFckUsRUFBRSxDQUFDLENBQUMsSUFBSSxZQUFZLEtBQUssQ0FBQyxLQUFLLENBQUM7WUFDL0IsTUFBTSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxJQUFJLEVBQUUsSUFBSSxFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBRXhELE1BQU0sNEVBQTRFLENBQUM7SUFDcEYsQ0FBQztJQUVELHFDQUFxQztJQUNyQyxzQkFBSSxHQUFKLFVBQUssVUFBVSxFQUFFLElBQUk7UUFDcEIsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQztZQUNsQixJQUFJLENBQUMsU0FBUyxDQUFDLEtBQUssRUFBRSxDQUFDO1FBQ3hCLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLElBQUksQ0FBQyxVQUFVLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDM0MsQ0FBQztJQUVELDJIQUEySDtJQUMzSCx1QkFBSyxHQUFMO1FBRUMsSUFBSSxDQUFDLE9BQU8sRUFBRSxDQUFDO1FBRWYsSUFBSSxVQUFVLEdBQUcsSUFBSSxDQUFDLE1BQU0sQ0FBQyxLQUFLLEVBQ2pDLFdBQVcsR0FBRyxJQUFJLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQztRQUVsQyxVQUFVLENBQUMsS0FBSyxFQUFFLENBQUM7UUFDbkIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEtBQUssSUFBSSxJQUFJLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUM7WUFDdEMsSUFBSSxXQUFXLEdBQUcsSUFBSSxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUM7WUFDNUMsV0FBVyxDQUFDLEtBQUssRUFBRSxDQUFDO1FBQ3JCLENBQUM7UUFDRCxXQUFXLENBQUMsS0FBSyxFQUFFLENBQUM7UUFFcEIsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQztZQUNsQixJQUFJLENBQUMsU0FBUyxDQUFDLEtBQUssRUFBRSxDQUFDO0lBQ3pCLENBQUM7SUFFRCxzRUFBc0U7SUFDdEUsdUJBQUssR0FBTDtRQUVDLElBQUksQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUVmLElBQUksVUFBVSxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsS0FBSyxFQUNqQyxXQUFXLEdBQUcsSUFBSSxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUM7UUFFbEMsVUFBVSxDQUFDLEtBQUssRUFBRSxDQUFDO1FBQ25CLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLElBQUksSUFBSSxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDO1lBQ3RDLElBQUksV0FBVyxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDO1lBQzVDLFdBQVcsQ0FBQyxLQUFLLEVBQUUsQ0FBQztRQUNyQixDQUFDO1FBQ0QsV0FBVyxDQUFDLEtBQUssRUFBRSxDQUFDO1FBRXBCLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUM7WUFDbEIsSUFBSSxDQUFDLFNBQVMsQ0FBQyxLQUFLLEVBQUUsQ0FBQztJQUN6QixDQUFDO0lBRUQsZ0ZBQWdGO0lBQ2hGLDBCQUFRLEdBQVI7UUFFQyxJQUFJLElBQUksR0FBRyxJQUFJLENBQUM7UUFDaEIsSUFBSSxTQUFTLEdBQWlDLEVBQUUsQ0FBQztRQUNqRCxJQUFJLE9BQU8sR0FBRyxJQUFJLENBQUMsT0FBTyxFQUFFLENBQUM7UUFFN0IsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxPQUFPLENBQUMsQ0FBQyxDQUFDO1lBQ3ZCLElBQUksTUFBTSxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUM7WUFDL0IsSUFBSSxLQUFLLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQztZQUM3Qjs7Ozs7Y0FLRTtZQUVGLFNBQVMsR0FBRyxNQUFNLENBQUMsUUFBUSxDQUFDLFNBQVMsRUFBRSxLQUFLLENBQUMsQ0FBQztRQUMvQyxDQUFDO1FBRUQsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxTQUFTLENBQUMscUJBQXFCLENBQUM7WUFDN0MsU0FBUyxDQUFDLHFCQUFxQixDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU8sRUFBRSxDQUFDO1FBQzlDLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUUxQyxJQUFJLFFBQVEsR0FBRyxFQUFFLENBQUM7UUFDbEIsUUFBUSxJQUFJLDBDQUEwQyxHQUFHLFNBQVMsQ0FBQyxNQUFNO1lBQ3pFLFVBQVUsQ0FBQztRQUNYLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksU0FBUyxDQUFDLFNBQVMsQ0FBQztZQUNqQyxRQUFRLElBQUksSUFBSSxHQUFHLFNBQVMsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLE1BQU0sR0FBRyxDQUFDLFNBQVMsQ0FBQyxTQUFTLENBQzNFLENBQUMsQ0FBQyxDQUFDLEtBQUssSUFBSSxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUM7UUFDeEIsUUFBUSxJQUFJLG1DQUFtQyxDQUFDO1FBQ2hELFFBQVEsSUFBSSxrQkFBa0IsQ0FBQztRQUMvQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLFNBQVMsQ0FBQyxNQUFNLENBQUM7WUFDOUIsUUFBUSxJQUFJLElBQUksR0FBRyxTQUFTLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLFlBQVksR0FBRyxDQUFDLEdBQUcsS0FBSyxDQUFDO1FBQ25FLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxZQUFZLElBQUksU0FBUyxDQUFDLG9CQUFvQixDQUFDLENBQUMsQ0FBQztZQUN6RCxFQUFFLENBQUMsQ0FBQyxTQUFTLENBQUMsb0JBQW9CLENBQUMsWUFBWSxDQUFDLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQzdELEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxhQUFhLElBQUksU0FBUyxDQUFDLG9CQUFvQixDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsQ0FBQztvQkFDeEUsUUFBUSxJQUFJLFNBQVMsQ0FBQyxvQkFBb0IsQ0FBQyxZQUFZLENBQUMsQ0FBQyxhQUFhLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUM7b0JBQ2xGLFFBQVEsSUFBSSxTQUFTLENBQUMsZUFBZSxDQUFDLFlBQVksQ0FBQyxDQUFDLGFBQWEsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQztnQkFDOUUsQ0FBQztZQUNGLENBQUM7UUFDRixDQUFDO1FBQ0QsUUFBUSxJQUFJLG9CQUFvQixDQUFBO1FBQ2hDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksU0FBUyxDQUFDLE9BQU8sQ0FBQztZQUMvQixRQUFRLElBQUksU0FBUyxHQUFHLENBQUMsR0FBRyxRQUFRLEdBQUcsU0FBUyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsR0FBRyxLQUFLLENBQUM7UUFDckUsUUFBUSxJQUFJLG9CQUFvQixDQUFBO1FBQ2hDLFFBQVEsSUFBSSwyQ0FBMkMsQ0FBQztRQUN4RCxRQUFRLElBQUksSUFBSSxHQUFHLFNBQVMsQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDLEVBQUUsR0FBRyxZQUFZLENBQUM7UUFDOUQsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxTQUFTLENBQUMsT0FBTyxDQUFDO1lBQy9CLFFBQVEsSUFBSSxJQUFJLEdBQUcsU0FBUyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsR0FBRyxhQUFhLEdBQUcsQ0FBQyxHQUFHLEtBQUssQ0FBQztRQUNyRSxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsWUFBWSxJQUFJLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQztZQUN4RCxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsYUFBYSxJQUFJLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQyxZQUFZLENBQUMsQ0FBQztnQkFDdkUsUUFBUSxJQUFJLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQyxZQUFZLENBQUMsQ0FBQyxhQUFhLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEdBQUcsR0FBRyxDQUFDO1FBQzNGLFFBQVEsSUFBSSxPQUFPLENBQUM7UUFDcEIsUUFBUTtZQUNSLG9GQUFvRixDQUFDO1FBQ3JGLFFBQVE7WUFDUiw0RkFBNEYsQ0FBQztRQUM3RixRQUFRLEdBQUcsUUFBUSxDQUFDLEtBQUssQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7UUFFM0MsSUFBSSxXQUFXLEdBQUcsSUFBSSxRQUFRLENBQUMsUUFBUSxDQUFDLENBQUM7UUFFekMsSUFBSSxPQUFPLEdBQUcsV0FBVyxFQUFFLENBQUM7UUFFNUIsT0FBTyxDQUFDLElBQUksR0FBRztZQUNkLFNBQVMsRUFBRSxTQUFTLENBQUMsU0FBUztZQUM5QixRQUFRLEVBQUUsU0FBUyxDQUFDLG9CQUFvQjtZQUN4QyxTQUFTLEVBQUUsU0FBUyxDQUFDLHFCQUFxQjtZQUMxQyxLQUFLLEVBQUUsU0FBUyxDQUFDLGVBQWU7WUFDaEMsTUFBTSxFQUFFLFNBQVMsQ0FBQyxNQUFNO1lBQ3hCLE9BQU8sRUFBRSxTQUFTLENBQUMsT0FBTztZQUMxQixnQkFBZ0IsRUFBRSxJQUFJLENBQUMsUUFBUTtZQUMvQixpQkFBaUIsRUFBRSxJQUFJLENBQUMsU0FBUztTQUNqQyxDQUFBO1FBRUQsT0FBTyxDQUFDLEtBQUssR0FBRztZQUNmLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDO2dCQUNwQixJQUFJLENBQUMsU0FBUyxHQUFHLElBQUksQ0FBQztnQkFDdEIsSUFBSSxDQUFDLFFBQVEsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDO2dCQUM5QyxJQUFJLENBQUMsU0FBUyxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUMsaUJBQWlCLENBQUM7WUFDakQsQ0FBQztRQUNGLENBQUMsQ0FBQTtRQUVELElBQUksQ0FBQyxTQUFTLEdBQUcsT0FBTyxDQUFDO1FBQ3pCLElBQUksQ0FBQyxRQUFRLEdBQUcsT0FBTyxDQUFDLFFBQVEsQ0FBQztRQUNqQyxJQUFJLENBQUMsU0FBUyxHQUFHLE9BQU8sQ0FBQyxTQUFTLENBQUM7SUFDcEMsQ0FBQztJQUVELHFIQUFxSDtJQUNySCx5QkFBTyxHQUFQO1FBQ0MsRUFBRSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDO1lBQ25CLE1BQU0sQ0FBQztRQUVSLElBQUksU0FBUyxHQUFHLElBQUksQ0FBQyxTQUFTLENBQUM7UUFFL0IsSUFBSSxRQUFRLEdBQUc7WUFBUyxjQUFjO2lCQUFkLFdBQWMsQ0FBZCxzQkFBYyxDQUFkLElBQWM7Z0JBQWQsNkJBQWM7O1lBQ3JDLElBQUksSUFBSSxHQUFHLElBQUksQ0FBQyxLQUFLLEVBQUUsQ0FBQztZQUN4QixJQUFJLElBQUksR0FBRyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUM7WUFFdEIsSUFBSSxFQUFFLEdBQUcsSUFBSSxHQUFHLEdBQUcsQ0FBQztZQUNwQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsUUFBUSxJQUFJLElBQUksQ0FBQztnQkFDekIsRUFBRSxJQUFJLElBQUksQ0FBQyxRQUFRLENBQUMsR0FBRyxHQUFHLENBQUM7WUFDNUIsRUFBRSxJQUFJLElBQUksQ0FBQyxFQUFFLENBQUM7WUFFZCxJQUFJLE1BQU0sR0FBRyxTQUFTLENBQUMsTUFBTSxDQUFDO1lBQzlCLElBQUksU0FBUyxHQUFHLFNBQVMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDO1lBRXpDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxTQUFTLENBQUM7Z0JBQ25CLE1BQU0sQ0FBQyxNQUFNLENBQUMsU0FBUyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDO1lBQ2pDLE1BQU0sQ0FBQyxDQUFDLENBQUM7UUFDVixDQUFDLENBQUE7UUFFRCxJQUFJLElBQUksR0FBRyxJQUFJLENBQUMsT0FBTyxFQUFFLENBQUM7UUFFMUIsc0NBQXNDO1FBQ3RDLElBQUksR0FBRyxHQUFHLEVBQUUsQ0FBQztRQUNiLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQztZQUNwQixJQUFJLE1BQU0sR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTSxDQUFDO1lBQzVCOzs7OztjQUtFO1lBRUYsTUFBTSxDQUFDLEtBQUssR0FBRyxRQUFRLENBQUMsTUFBTSxFQUFFLE9BQU8sQ0FBQyxDQUFDO1lBQ3pDLE1BQU0sQ0FBQyxHQUFHLEdBQUcsUUFBUSxDQUFDLE1BQU0sRUFBRSxLQUFLLENBQUMsQ0FBQztZQUNyQyxNQUFNLENBQUMsVUFBVSxHQUFHLFFBQVEsQ0FBQyxNQUFNLEVBQUUsWUFBWSxDQUFDLENBQUM7WUFDbkQsTUFBTSxDQUFDLElBQUksR0FBRyxRQUFRLENBQUMsTUFBTSxFQUFFLE1BQU0sQ0FBQyxDQUFDO1lBRXZDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLElBQUksTUFBTSxDQUFDLEtBQUssQ0FBQyxXQUFXLENBQUM7Z0JBQzFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsV0FBVyxDQUFDLEtBQUssQ0FBQyxHQUFHLFFBQVEsQ0FBQyxNQUFNLEVBQUUsT0FBTyxFQUN6RCxhQUFhLEVBQUUsS0FBSyxDQUFDLENBQUM7WUFFeEIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEtBQUssSUFBSSxNQUFNLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQztnQkFDdkMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEtBQUssSUFBSSxNQUFNLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxLQUFLLENBQUMsQ0FBQztvQkFDOUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUMsS0FBSyxDQUFDLEdBQUcsUUFBUSxDQUFDLE1BQU0sRUFBRSxPQUFPLEVBQzdELFVBQVUsRUFBRSxLQUFLLEVBQUUsS0FBSyxDQUFDLENBQUM7UUFDOUIsQ0FBQztRQUVELGtCQUFrQjtRQUNsQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDcEIsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQztZQUM1Qjs7Ozs7Y0FLRTtZQUVGLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksTUFBTSxDQUFDLFdBQVcsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDO2dCQUM1QyxJQUFJLFVBQVUsR0FBRyxNQUFNLENBQUMsV0FBVyxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDakQsVUFBVSxDQUFDLE1BQU0sR0FBRyxRQUFRLENBQUMsVUFBVSxFQUFFLFFBQVEsQ0FBQyxDQUFDO2dCQUNuRCxVQUFVLENBQUMsSUFBSSxHQUFHLFFBQVEsQ0FBQyxVQUFVLEVBQUUsTUFBTSxDQUFDLENBQUM7WUFDaEQsQ0FBQztRQUNGLENBQUM7SUFDRixDQUFDO0lBRUQseUNBQXlDO0lBQ3pDLHlCQUFPLEdBQVA7UUFDQyxJQUFJLE9BQU8sR0FBNkIsRUFBRSxDQUFDO1FBRTNDLElBQUksVUFBVSxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLE9BQU8sRUFBRSxFQUMzQyxXQUFXLEdBQUcsSUFBSSxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsT0FBTyxFQUFFLENBQUM7UUFFNUMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLE1BQU0sSUFBSSxVQUFVLENBQUM7WUFDN0IsT0FBTyxDQUFDLElBQUksQ0FBQztnQkFDWixNQUFNLEVBQUUsVUFBVSxDQUFDLE1BQU0sQ0FBQztnQkFDMUIsS0FBSyxFQUFFLE9BQU87YUFDZCxDQUFDLENBQUM7UUFFSixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsS0FBSyxJQUFJLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQztZQUN0QyxJQUFJLFdBQVcsR0FBRyxJQUFJLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztZQUN0RCxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsTUFBTSxJQUFJLFdBQVcsQ0FBQztnQkFDOUIsT0FBTyxDQUFDLElBQUksQ0FBQztvQkFDWixNQUFNLEVBQUUsV0FBVyxDQUFDLE1BQU0sQ0FBQztvQkFDM0IsS0FBSyxFQUFFLEtBQUs7aUJBQ1osQ0FBQyxDQUFDO1FBQ0wsQ0FBQztRQUNELEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxNQUFNLElBQUksV0FBVyxDQUFDO1lBQzlCLE9BQU8sQ0FBQyxJQUFJLENBQUM7Z0JBQ1osTUFBTSxFQUFFLFdBQVcsQ0FBQyxNQUFNLENBQUM7Z0JBQzNCLEtBQUssRUFBRSxRQUFRO2FBQ2YsQ0FBQyxDQUFDO1FBRUosTUFBTSxDQUFDLE9BQU8sQ0FBQztJQUNoQixDQUFDO0lBRUQsMENBQTBDO0lBQzFDLHdCQUFNLEdBQU47UUFDQyxNQUFNLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDO0lBQy9CLENBQUM7SUFFRCwyQ0FBMkM7SUFDM0MseUJBQU8sR0FBUDtRQUNDLE1BQU0sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxJQUFJLENBQUM7SUFDaEMsQ0FBQztJQUVELGlDQUFpQztJQUNqQyxxQkFBRyxHQUFILFVBQUksTUFBTTtRQUVULElBQUksQ0FBQyxNQUFNLEdBQUcsTUFBTSxDQUFDO1FBQ3JCLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUM7WUFDbEIsSUFBSSxDQUFDLFNBQVMsQ0FBQyxLQUFLLEVBQUUsQ0FBQztJQUN6QixDQUFDO0lBRUQsNkJBQVcsR0FBWCxVQUFZLElBQUk7UUFDZixJQUFJLENBQUMsT0FBTyxFQUFFLENBQUM7UUFDZixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDO1lBQ2xCLElBQUksQ0FBQyxTQUFTLENBQUMsS0FBSyxFQUFFLENBQUM7UUFDeEIsSUFBSSxDQUFDLFNBQVMsR0FBRyxJQUFJLEdBQUcsSUFBSSxHQUFHLEtBQUssQ0FBQztJQUN0QyxDQUFDO0lBRUQsZ0ZBQWdGO0lBQ2hGLHdCQUFNLEdBQU4sVUFBTyxZQUFZO1FBRWxCLElBQUksQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUVmLElBQUksSUFBSSxHQUFHLElBQUksQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUMxQixJQUFJLE9BQU8sR0FBRyxFQUFFLENBQUM7UUFDakIsSUFBSSxXQUFXLEdBQUcsRUFBRSxDQUFDO1FBRXJCLHNDQUFzQztRQUN0QyxJQUFJLEdBQUcsR0FBRyxFQUFFLENBQUM7UUFDYixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDcEIsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQztZQUM1Qjs7Ozs7Y0FLRTtZQUVGLEdBQUcsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDO1lBRW5CLElBQUksSUFBSSxHQUFHO2dCQUNWLEtBQUssRUFBRTtvQkFDTixXQUFXLEVBQUUsRUFBRTtvQkFDZixRQUFRLEVBQUUsRUFBRTtpQkFDWjtnQkFDRCxLQUFLLEVBQUUsTUFBTSxDQUFDLEtBQUs7Z0JBQ25CLEdBQUcsRUFBRSxNQUFNLENBQUMsR0FBRztnQkFDZixVQUFVLEVBQUUsTUFBTSxDQUFDLFVBQVU7Z0JBQzdCLElBQUksRUFBRSxNQUFNLENBQUMsSUFBSTtnQkFDakIsS0FBSyxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLO2dCQUNwQixNQUFNLEVBQUUsSUFBSTthQUNaLENBQUM7WUFFRixJQUFJLENBQUMsTUFBTSxHQUFHLE1BQU0sQ0FBQyxNQUFNLElBQUksTUFBTSxDQUFDLFFBQVEsR0FBRyxVQUFVO2dCQUMxRCxNQUFNLENBQUMsTUFBTSxJQUFJLE1BQU0sQ0FBQyxJQUFJLEdBQUcsTUFBTTtvQkFDcEMsTUFBTSxDQUFDLE1BQU0sSUFBSSxNQUFNLENBQUMsUUFBUSxHQUFHLFVBQVU7d0JBQzVDLE1BQU0sQ0FBQyxNQUFNLElBQUksTUFBTSxDQUFDLElBQUksR0FBRyxNQUFNOzRCQUNwQyxJQUFJLENBQUM7WUFFVCxPQUFPLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3BCLENBQUM7UUFFRCxFQUFFLENBQUMsQ0FBQyxDQUFDLFlBQVksQ0FBQztZQUNqQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLE9BQU8sQ0FBQyxDQUFDLENBQUM7Z0JBQ3ZCLElBQUksWUFBWSxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFFOUIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEtBQUssSUFBSSxNQUFNLENBQUMsS0FBSyxDQUFDLFdBQVcsQ0FBQztvQkFDMUMsWUFBWSxDQUFDLEtBQUssQ0FBQyxXQUFXLENBQUMsS0FBSyxDQUFDLEdBQUcsTUFBTSxDQUFDLEtBQUssQ0FBQyxXQUFXLENBQUMsS0FBSyxDQUFDLENBQUM7Z0JBRXpFLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLElBQUksTUFBTSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO29CQUN6QyxZQUFZLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxLQUFLLENBQUMsR0FBRyxFQUFFLENBQUM7b0JBQ3hDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLElBQUksTUFBTSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUM7d0JBQzlDLFlBQVksQ0FBQyxLQUFLLENBQUMsUUFBUSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxHQUFHLE1BQU0sQ0FBQyxLQUFLLENBQUMsUUFBUSxDQUN0RSxLQUFLLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQztnQkFDaEIsQ0FBQztZQUNGLENBQUM7UUFFRixrQkFBa0I7UUFDbEIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDO1lBQ3BCLElBQUksTUFBTSxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxNQUFNLENBQUM7WUFFNUI7Ozs7O2NBS0U7WUFFRixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLE1BQU0sQ0FBQyxXQUFXLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQztnQkFDNUMsSUFBSSxVQUFVLEdBQUcsTUFBTSxDQUFDLFdBQVcsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ2pELFdBQVcsQ0FBQyxJQUFJLENBQUM7b0JBQ2hCLElBQUksRUFBRSxHQUFHLENBQUMsVUFBVSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUM7b0JBQzdCLEVBQUUsRUFBRSxHQUFHLENBQUMsVUFBVSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUM7b0JBQ3pCLE1BQU0sRUFBRSxVQUFVLENBQUMsTUFBTTtvQkFDekIsS0FBSyxFQUFFLFVBQVUsQ0FBQyxLQUFLLEdBQUcsR0FBRyxDQUFDLFVBQVUsQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSTtpQkFDekQsQ0FBQyxDQUFDO1lBQ0osQ0FBQztZQUNELEVBQUUsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxhQUFhLEVBQUUsQ0FBQztnQkFDMUIsV0FBVyxDQUFDLElBQUksQ0FBQztvQkFDaEIsSUFBSSxFQUFFLEdBQUcsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDO29CQUNwQixFQUFFLEVBQUUsR0FBRyxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUM7b0JBQ2xCLE1BQU0sRUFBRSxNQUFNLENBQUMsY0FBYyxDQUFDLE1BQU07b0JBQ3BDLEtBQUssRUFBRSxNQUFNLENBQUMsY0FBYyxDQUFDLEtBQUssR0FBRyxHQUFHLENBQUMsTUFBTSxDQUFDLGNBQWMsQ0FBQyxLQUFLO3lCQUNsRSxFQUFFLENBQUMsR0FBRyxJQUFJO2lCQUNaLENBQUMsQ0FBQztRQUNMLENBQUM7UUFFRCxNQUFNLENBQUM7WUFDTixPQUFPLEVBQUUsT0FBTztZQUNoQixXQUFXLEVBQUUsV0FBVztTQUN4QixDQUFBO0lBQ0YsQ0FBQztJQUVELG9GQUFvRjtJQUNwRjs7O01BR0U7SUFDRix1QkFBSyxHQUFMLFVBQU0sY0FBYztRQUNuQixFQUFFLENBQUMsQ0FBQyxDQUFFLE9BQU8sY0FBYyxDQUFDO1lBQzNCLGNBQWMsR0FBRyxLQUFLLENBQUM7UUFDeEIsSUFBSSxJQUFJLEdBQUcsa0NBQWtDLENBQUM7UUFDOUMsSUFBSSxNQUFNLEdBQUcsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLE1BQU0sRUFBRSxJQUFJLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ2hGLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLElBQUksTUFBTSxDQUFDLENBQUMsQ0FBQztZQUMxQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxJQUFJLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDO2dCQUMxQyxJQUFJLFVBQVUsR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsV0FBVyxDQUFDLEVBQUUsQ0FBQyxDQUFDO2dCQUMvQyxJQUFJLE9BQU8sR0FBRyxVQUFVLENBQUMsRUFBRSxDQUFDO2dCQUM1QixJQUFJLElBQUksR0FBRyxVQUFVLENBQUMsSUFBSSxDQUFDO2dCQUMzQixJQUFJLE9BQU8sR0FBRyxNQUFNLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDO2dCQUM1QyxJQUFJLFNBQVMsR0FBRyxNQUFNLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxDQUFDO2dCQUN4Qzs7O2tCQUdFO2dCQUNGLEVBQUUsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUM7b0JBQ3BCLEVBQUUsQ0FBQyxDQUFDLFVBQVUsQ0FBQyxTQUFTLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQzt3QkFDakMsSUFBSSxRQUFRLEdBQUcsTUFBTSxHQUFHLE9BQU8sR0FBRyxHQUFHLEdBQUcsU0FBUyxDQUFDO3dCQUNsRCxJQUFJLElBQUksTUFBTSxHQUFHLFFBQVE7NEJBQ3pCLCtEQUErRCxDQUFDO3dCQUNoRSxJQUFJLElBQUksTUFBTSxHQUFHLE9BQU8sR0FBRyxNQUFNLEdBQUcsUUFBUSxHQUFHLFlBQVksR0FBRyxJQUFJLEdBQUcsdUJBQXVCLENBQUM7d0JBQzdGLElBQUksSUFBSSxNQUFNLEdBQUcsUUFBUSxHQUFHLE1BQU0sR0FBRyxTQUFTLEdBQUcsSUFBSSxDQUFDO29CQUN2RCxDQUFDO29CQUFDLElBQUk7d0JBQ0wsSUFBSSxJQUFJLE1BQU0sR0FBRyxPQUFPLEdBQUcsTUFBTSxHQUFHLFNBQVMsR0FBRyxZQUFZLEdBQUcsSUFBSSxHQUFHLEtBQUssQ0FBQztvQkFDN0UsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLElBQUksSUFBSSxVQUFVLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQzt3QkFDdkMsSUFBSSxTQUFTLEdBQUcsVUFBVSxDQUFDLFNBQVMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxLQUFLLENBQUM7d0JBQ2pELElBQUksSUFBSSxHQUFHLFVBQVUsQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDO3dCQUMzQyxJQUFJLFdBQVcsR0FBRyxNQUFNLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxDQUFDO3dCQUM1QyxJQUFJLElBQUksTUFBTSxHQUFHLFdBQVcsR0FBRyxNQUFNLEdBQUcsUUFBUSxHQUFHLG1CQUFtQixDQUFDO29CQUN4RSxDQUFDO2dCQUNGLENBQUM7Z0JBQUMsSUFBSSxDQUFDLENBQUM7b0JBQ1AsSUFBSSxJQUFJLE1BQU0sR0FBRyxPQUFPLEdBQUcsTUFBTSxHQUFHLFNBQVMsR0FBRyxZQUFZLEdBQUcsSUFBSSxHQUFHLEtBQUssQ0FBQztvQkFDNUUsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLElBQUksSUFBSSxVQUFVLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQzt3QkFDdkMsSUFBSSxTQUFTLEdBQUcsVUFBVSxDQUFDLFNBQVMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxLQUFLLENBQUM7d0JBQ2pELElBQUksSUFBSSxHQUFHLFVBQVUsQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDLENBQUMsSUFBSSxDQUFDO3dCQUMzQyxJQUFJLFdBQVcsR0FBRyxNQUFNLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxDQUFDO3dCQUM1QyxJQUFJLElBQUksTUFBTSxHQUFHLFdBQVcsR0FBRyxNQUFNLEdBQUcsU0FBUyxHQUFHLG1CQUFtQixDQUFDO29CQUN6RSxDQUFDO2dCQUNGLENBQUM7WUFDRixDQUFDO1FBQ0YsQ0FBQztRQUNELElBQUksSUFBSSxLQUFLLENBQUM7UUFDZCxNQUFNLENBQUM7WUFDTixJQUFJLEVBQUUsSUFBSTtZQUNWLElBQUksRUFBRSx5Q0FBeUMsR0FBRyxNQUFNLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUMsR0FBRyxTQUFTO1NBQy9GLENBQUE7SUFDRixDQUFDO0lBRUQsa0hBQWtIO0lBQ2xILDRCQUFVLEdBQVY7UUFDQyxFQUFFLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUM7WUFDbkIsSUFBSSxDQUFDLFFBQVEsRUFBRSxDQUFDO1FBRWpCLElBQUksSUFBSSxHQUFHLElBQUksQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDO1FBRS9CLDRCQUE0QjtRQUM1QixJQUFJLFVBQVUsR0FBRyxzQkFBc0IsQ0FBQztRQUV4QyxlQUFlO1FBQ2YsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsTUFBTSxDQUFDO1lBQ3pCLFVBQVUsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxZQUFZLEdBQUcsQ0FBQyxHQUFHLE1BQU0sQ0FBQztRQUVqRSwyQkFBMkI7UUFDM0IsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLE1BQU0sSUFBSSxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztZQUNsQyxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsUUFBUSxJQUFJLElBQUksQ0FBQyxRQUFRLENBQUMsTUFBTSxDQUFDLENBQUM7Z0JBQzFDLFVBQVUsSUFBSSxJQUFJLENBQUMsUUFBUSxDQUFDLE1BQU0sQ0FBQyxDQUFDLFFBQVEsQ0FBQyxHQUFHLElBQUksQ0FBQztRQUN2RCxDQUFDO1FBRUQsZ0JBQWdCO1FBQ2hCLFVBQVUsSUFBSSxvQkFBb0IsQ0FBQztRQUNuQyxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxPQUFPLENBQUM7WUFDMUIsVUFBVSxJQUFJLFNBQVMsR0FBRyxDQUFDLEdBQUcsUUFBUSxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDO1FBQ25FLFVBQVUsSUFBSSxtQkFBbUIsQ0FBQztRQUVsQyx3Q0FBd0M7UUFDeEMsSUFBSSxNQUFNLEdBQUcsVUFBVSxDQUFDLEtBQUssQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUM3QyxJQUFJLFNBQVMsR0FBRyxDQUFDLENBQUM7UUFDbEIsSUFBSSxHQUFHLEdBQUcsRUFBRSxDQUFDO1FBQ2IsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLE9BQU8sSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDO1lBQzVCLElBQUksR0FBRyxHQUFHLE1BQU0sQ0FBQyxPQUFPLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDMUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ25CLEdBQUcsQ0FBQyxHQUFHLENBQUMsR0FBRyxTQUFTLEVBQUUsQ0FBQztZQUN4QixDQUFDO1FBQ0YsQ0FBQztRQUNELElBQUksUUFBUSxHQUFHLFNBQVMsQ0FBQztRQUN6QixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLEdBQUcsQ0FBQztZQUNqQixRQUFRLElBQUksR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBRyxJQUFJLENBQUMsU0FBUyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxLQUFLLENBQUM7UUFDOUQsUUFBUSxHQUFHLFFBQVEsQ0FBQyxTQUFTLENBQUMsQ0FBQyxFQUFFLFFBQVEsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDO1FBQ2pFLFFBQVEsR0FBRyxZQUFZLEdBQUcsVUFBVSxDQUFDLE9BQU8sQ0FBQyxZQUFZLEVBQUUsVUFDMUQsS0FBSztZQUNMLE1BQU0sQ0FBQyxJQUFJLEdBQUcsR0FBRyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxHQUFHLENBQUE7UUFDL0MsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLEtBQUssRUFBRSxLQUFLLEdBQUcsUUFBUSxHQUFHLEVBQUUsQ0FBQyxHQUFHLEtBQUssQ0FBQztRQUNqRCxRQUFRLElBQUksWUFBWSxDQUFDO1FBRXpCLDZCQUE2QjtRQUM3QixNQUFNLENBQUMsSUFBSSxRQUFRLENBQUMsUUFBUSxDQUFDLEVBQUUsQ0FBQztJQUNqQyxDQUFDO0lBRUQsd0JBQU0sR0FBTjtRQUNDLEVBQUUsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQztZQUNuQixJQUFJLENBQUMsUUFBUSxFQUFFLENBQUM7UUFFakIsSUFBSSxRQUFRLEdBQUcsZUFBZSxHQUFHLElBQUksQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxNQUFNO1lBQ2pFLEtBQUssQ0FBQztRQUNQLFFBQVEsSUFBSSxnQkFBZ0IsR0FBRyxJQUFJLENBQUMsU0FBUyxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTTtZQUNqRSxLQUFLLENBQUM7UUFDTixRQUFRLElBQUksaUJBQWlCLENBQUM7UUFDOUIsUUFBUSxJQUFJLGlCQUFpQixHQUFHLElBQUksQ0FBQyxTQUFTLENBQUMsUUFBUSxDQUFDLFFBQVEsRUFBRTtZQUNsRSxLQUFLLENBQUM7UUFDTixRQUFRLElBQUksa0JBQWtCLEdBQUcsSUFBSSxDQUFDLFNBQVMsQ0FBQyxTQUFTLENBQUMsUUFBUSxFQUFFO1lBQ3BFLEtBQUssQ0FBQztRQUNOLFFBQVEsSUFBSSw0QkFBNEIsQ0FBQztRQUN6QyxRQUFRLElBQUksNEJBQTRCLENBQUM7UUFDekMsUUFBUSxJQUFJLHFDQUFxQyxDQUFDO1FBQ2xELFFBQVEsSUFBSSx1Q0FBdUMsQ0FBQztRQUNwRCxRQUFRO1lBQ1Isc0dBQXNHLENBQUM7UUFDdkcsUUFBUSxJQUFJLGlEQUFpRCxDQUFDO1FBQzlELFFBQVEsSUFBSSwwQ0FBMEMsQ0FBQztRQUN2RCxRQUFRO1lBQ1Isc0VBQXNFLENBQUM7UUFDdkUsUUFBUSxJQUFJLFFBQVEsQ0FBQztRQUVyQixJQUFJLElBQUksR0FBRyxJQUFJLElBQUksQ0FBQyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7UUFDaEMsSUFBSSxPQUFPLEdBQVMsTUFBTyxDQUFDLEdBQUcsQ0FBQyxlQUFlLENBQUMsSUFBSSxDQUFDLENBQUM7UUFFdEQsTUFBTSxDQUFDLElBQUksTUFBTSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBQzVCLENBQUM7SUFFRCxnQ0FBZ0M7SUFDaEMsdUJBQUssR0FBTCxVQUFNLFlBQVk7UUFDakIsTUFBTSxDQUFDLE9BQU8sQ0FBQyxRQUFRLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxZQUFZLENBQUMsQ0FBQyxDQUFDO0lBQ3BELENBQUM7SUFFTSxnQkFBUSxHQUFmLFVBQWdCLElBQUk7UUFFbkIsSUFBSSxPQUFPLEdBQUcsRUFBRSxDQUFDO1FBRWpCLElBQUksTUFBTSxHQUFHO1lBQ1osS0FBSyxFQUFFLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7WUFDekIsTUFBTSxFQUFFLEVBQUU7WUFDVixNQUFNLEVBQUUsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQztTQUMxQixDQUFBO1FBR0QsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztZQUM1QixJQUFJLE1BQU0sR0FBRyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBRTdCLElBQUksTUFBTSxHQUFHLElBQUksT0FBTyxDQUFDLE1BQU0sRUFBRSxDQUFDO1lBQ2xDLE1BQU0sQ0FBQyxLQUFLLENBQUMsV0FBVyxHQUFHLE1BQU0sQ0FBQyxLQUFLLENBQUMsV0FBVyxDQUFDO1lBQ3BELE1BQU0sQ0FBQyxLQUFLLENBQUMsUUFBUSxHQUFHLE1BQU0sQ0FBQyxLQUFLLENBQUMsUUFBUSxDQUFDO1lBQzlDLE1BQU0sQ0FBQyxLQUFLLEdBQUcsTUFBTSxDQUFDLEtBQUssQ0FBQztZQUM1QixNQUFNLENBQUMsR0FBRyxHQUFHLE1BQU0sQ0FBQyxHQUFHLENBQUM7WUFDeEIsTUFBTSxDQUFDLFVBQVUsR0FBRyxNQUFNLENBQUMsVUFBVSxDQUFDO1lBQ3RDLE1BQU0sQ0FBQyxJQUFJLEdBQUcsTUFBTSxDQUFDLElBQUksQ0FBQztZQUMxQixNQUFNLENBQUMsTUFBTSxHQUFHLE1BQU0sQ0FBQyxNQUFNLElBQUksTUFBTSxHQUFHLE1BQU0sQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDO2dCQUM5RCxNQUFNLENBQUMsUUFBUSxDQUFDO1lBQ2pCLE9BQU8sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7WUFFckIsRUFBRSxDQUFDLENBQUMsTUFBTSxDQUFDLEtBQUssSUFBSSxPQUFPLENBQUM7Z0JBQzNCLE1BQU0sQ0FBQyxLQUFLLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1lBQzFCLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQyxNQUFNLENBQUMsS0FBSyxJQUFJLFFBQVEsQ0FBQztnQkFDakMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUM7WUFDM0IsSUFBSSxDQUFDLENBQUM7Z0JBQ0wsRUFBRSxDQUFDLENBQUMsT0FBTyxNQUFNLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxLQUFLLENBQUMsSUFBSSxXQUFXLENBQUM7b0JBQ3JELE1BQU0sQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxHQUFHLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDbEQsTUFBTSxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1lBQ3pDLENBQUM7UUFDRixDQUFDO1FBRUQsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQztZQUNoQyxJQUFJLE1BQU0sR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ2pDLElBQUksSUFBSSxHQUFHLE9BQU8sQ0FBQyxNQUFNLENBQUMsSUFBSSxDQUFDLENBQUM7WUFDaEMsSUFBSSxFQUFFLEdBQUcsT0FBTyxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsQ0FBQztZQUM1QixJQUFJLE1BQU0sR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFBO1lBQzFCLElBQUksS0FBSyxHQUFHLE9BQU8sQ0FBQyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUM7WUFFbEMsSUFBSSxVQUFVLEdBQUcsSUFBSSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsTUFBTSxDQUFDLENBQUM7WUFDMUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDO2dCQUNULEtBQUssQ0FBQyxJQUFJLENBQUMsVUFBVSxDQUFDLENBQUM7UUFDekIsQ0FBQztRQUVELE1BQU0sQ0FBQyxJQUFJLE9BQU8sQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUM1QixDQUFDO0lBQ0YsY0FBQztBQUFELENBbG1CQSxBQWttQkMsSUFBQTtBQWxtQlksZUFBTyxVQWttQm5CLENBQUE7QUFPQTs7QUNybkJELG9DQUFvQztBQUdwQyxJQUFPLE1BQU0sV0FBVyxVQUFVLENBQUMsQ0FBQztBQUVwQzs7NEZBRTRGO0FBRTVGOzs7O0VBSUU7QUFFRjtJQUFBO1FBQ0MsT0FBRSxHQUFHLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQztRQUNsQixVQUFLLEdBQUcsSUFBSSxDQUFDO1FBQ2IsZ0JBQVcsR0FBOEI7WUFDeEMsTUFBTSxFQUFFLEVBQUU7WUFDVixTQUFTLEVBQUUsRUFBRTtZQUNiLEtBQUssRUFBRSxFQUFFO1NBQ1QsQ0FBQztRQUNGLFVBQUssR0FBRztZQUNQLGNBQWMsRUFBRSxDQUFDO1lBQ2pCLFNBQVMsRUFBRSxDQUFDO1lBQ1osS0FBSyxFQUFFLENBQUM7U0FDUixDQUFDO1FBQ0YsVUFBSyxHQUFHO1lBQ1AsV0FBVyxFQUFFLEVBQUU7WUFDZixRQUFRLEVBQUUsRUFBRTtZQUNaLFVBQVUsRUFBRSxFQUFFO1NBQ2QsQ0FBQztRQUNGLFVBQUssR0FBRyxDQUFDLENBQUM7UUFDVixRQUFHLEdBQUcsQ0FBQyxDQUFDO1FBQ1IsZUFBVSxHQUFHLENBQUMsQ0FBQztRQUNmLG1CQUFjLEdBQUcsSUFBSSxNQUFNLENBQUMsVUFBVSxDQUFDLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyw4QkFBOEI7UUFDckYsV0FBTSxHQUFHLE1BQU0sQ0FBQyxRQUFRLENBQUM7UUFDekIsZUFBVSxHQUFHLEVBQUUsQ0FBQztRQUNoQixTQUFJLEdBQUcsSUFBSSxDQUFDLE1BQU0sRUFBRSxHQUFHLEVBQUUsR0FBRyxFQUFFLENBQUM7UUFDL0IsZUFBVSxHQUFHLENBQUMsQ0FBQztJQXNzQmhCLENBQUM7SUFwc0JBLHNCQUFzQjtJQUN0Qix5QkFBUSxHQUFSLFVBQVMsS0FBYztRQUN0QixpREFBaUQ7UUFDakQsRUFBRSxDQUFDLENBQUMsT0FBTyxLQUFLLElBQUksV0FBVyxDQUFDLENBQUMsQ0FBQztZQUNqQyxJQUFJLENBQUMsVUFBVSxHQUFHLEtBQUssQ0FBQztZQUN4QixJQUFJLENBQUMsVUFBVSxHQUFHLENBQUMsQ0FBQztZQUNwQixJQUFJLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztZQUNkLE1BQU0sQ0FBQyxJQUFJLENBQUMsVUFBVSxDQUFDO1FBQ3hCLENBQUM7UUFFRCxZQUFZO1FBQ1osSUFBSSxDQUFDLEdBQUcsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDO1FBRXRCLFNBQVM7UUFDVCxJQUFJLENBQUMsS0FBSyxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsTUFBTTtZQUNsRSxJQUFJLENBQUMsS0FBSyxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUM7UUFFdkIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUM7WUFDdkMsSUFBSSxRQUFRLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDMUMsSUFBSSxDQUFDLEtBQUssSUFBSSxRQUFRLENBQUMsSUFBSSxDQUFDLFVBQVUsR0FBRyxRQUFRLENBQUMsTUFBTSxHQUFHLFFBQVEsQ0FBQyxJQUFJLENBQUM7UUFDMUUsQ0FBQztRQUVELFNBQVM7UUFDVCxJQUFJLENBQUMsVUFBVSxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBRTFDLFFBQVE7UUFDUixJQUFJLENBQUMsVUFBVSxHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssRUFBRSxJQUFJLENBQUMsQ0FBQztRQUVoRCxnQkFBZ0I7UUFDaEIsSUFBSSxVQUFVLEdBQUcsRUFBRSxDQUFDO1FBQ3BCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLElBQUksSUFBSSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO1lBQ3BDLDZCQUE2QjtZQUM3QixJQUFJLE1BQU0sR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxFQUFFLENBQUMsQ0FBQztZQUNyQyxJQUFJLE1BQU0sR0FBRyxJQUFJLENBQUMsVUFBVSxDQUFDLEVBQUUsQ0FBQyxDQUFDO1lBRWpDLDhHQUE4RztZQUM5RyxJQUFJLFNBQVMsR0FBRyxNQUFNLENBQUMsY0FBYyxDQUFDLEtBQUssSUFBSSxJQUFJLEdBQUcsTUFBTSxDQUFDLEdBQUcsR0FBRyxDQUFDLENBQUM7WUFFckUsK0ZBQStGO1lBQy9GLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxRQUFRLElBQUksSUFBSSxDQUFDLEtBQUssQ0FBQyxVQUFVLENBQUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDdkQsU0FBUyxJQUFJLElBQUksQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsQ0FBQyxRQUFRLENBQUMsQ0FBQyxNQUFNO29CQUM5RCxJQUFJLENBQUMsS0FBSyxDQUFDLFVBQVUsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQztZQUM1RCxDQUFDO1lBQ0QsVUFBVSxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsR0FBRyxTQUFTLENBQUM7UUFDbkMsQ0FBQztRQUVELEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDO1lBQ3ZDLElBQUksUUFBUSxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBRTFDLDZCQUE2QjtZQUM3QixJQUFJLENBQUMsS0FBSyxDQUFDLFdBQVcsQ0FBQyxRQUFRLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLGNBQWMsQ0FBQyxJQUFJLEdBQUcsSUFBSSxDQUFDLGNBQWM7aUJBQ25GLE1BQU0sR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFdBQVcsQ0FBQyxRQUFRLENBQUMsRUFBRSxDQUFDLEdBQUcsUUFBUSxDQUFDLElBQUksR0FBRyxRQUFRLENBQUMsSUFBSTtpQkFDNUUsVUFBVSxDQUFDO1lBRVosR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7Z0JBQ3BDLDZCQUE2QjtnQkFDN0IsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBQ3JDLElBQUksTUFBTSxHQUFHLElBQUksQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBQ2pDLElBQUksU0FBUyxHQUFHLFVBQVUsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBRXRDLFNBQVM7Z0JBQ1QsTUFBTSxDQUFDLFFBQVEsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsY0FBYyxDQUFDLElBQUksR0FBRyxNQUFNLENBQUMsY0FBYztxQkFDdkUsTUFBTSxHQUFHLE1BQU0sQ0FBQyxRQUFRLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLFVBQVUsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFdBQVcsQ0FDeEUsUUFBUSxDQUFDLEVBQUUsQ0FBQyxHQUFHLFNBQVMsQ0FBQztZQUMxQixDQUFDO1FBQ0YsQ0FBQztRQUVELG1DQUFtQztRQUNuQyxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsVUFBVSxJQUFJLElBQUksQ0FBQyxXQUFXLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQztZQUMvQyxJQUFJLENBQUMsV0FBVyxDQUFDLEtBQUssQ0FBQyxVQUFVLENBQUMsQ0FBQyxJQUFJLEdBQUcsSUFBSSxDQUFDLFVBQVUsQ0FBQztRQUMzRCxDQUFDO1FBRUQsTUFBTSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUM7SUFDeEIsQ0FBQztJQUVELDJCQUEyQjtJQUMzQiwwQkFBUyxHQUFULFVBQVUsSUFBWSxFQUFFLE1BQWU7UUFDdEMsb0JBQW9CO1FBQ3BCLElBQUksS0FBSyxHQUFHLENBQUMsQ0FBQztRQUVkLG9EQUFvRDtRQUNwRCxJQUFJLFFBQVEsR0FBRyxPQUFPLE1BQU0sSUFBSSxXQUFXLElBQUksTUFBTSxJQUFJLElBQUksQ0FBQztRQUU5RCxxREFBcUQ7UUFDckQsRUFBRSxDQUFDLENBQUMsUUFBUSxDQUFDO1lBQ1osSUFBSSxDQUFDLEtBQUssQ0FBQyxjQUFjLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxTQUFTLEdBQUcsTUFBTSxHQUFHLElBQUksQ0FBQyxVQUFVLENBQUMsQ0FBQyxTQUFTO1FBRXZGLElBQUksQ0FDSixDQUFDO1lBQ0EsNkVBQTZFO1lBQzdFLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLElBQUksSUFBSSxDQUFDLFdBQVcsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDO2dCQUMzQyxJQUFJLFVBQVUsR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLFNBQVMsQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFDaEQsSUFBSSxNQUFNLEdBQUcsVUFBVSxDQUFDLEVBQUUsQ0FBQztnQkFDM0IsU0FBUztnQkFDVCxLQUFLLElBQUksTUFBTSxDQUFDLEtBQUssQ0FBQyxjQUFjLEdBQUcsVUFBVSxDQUFDLElBQUksR0FBRyxVQUFVLENBQUMsTUFBTSxDQUFDO1lBQzVFLENBQUM7WUFFRCxpQ0FBaUM7WUFDakMsSUFBSSxDQUFDLEtBQUssQ0FBQyxTQUFTLEdBQUcsSUFBSSxDQUFDLFVBQVUsR0FBRyxLQUFLLENBQUM7WUFFL0MsS0FBSyxHQUFHLENBQUMsQ0FBQztZQUNWLHVFQUF1RTtZQUN2RSxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxJQUFJLElBQUksQ0FBQyxLQUFLLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztnQkFDcEMsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLFVBQVUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLGVBQWU7Z0JBQ2pELElBQUksU0FBUyxHQUFHLE1BQU0sQ0FBQyxjQUFjLENBQUMsS0FBSyxJQUFJLElBQUksR0FBRyxNQUFNLENBQUMsR0FBRyxHQUFHLENBQUMsQ0FBQyxDQUFDLDJEQUEyRDtnQkFFakksd0ZBQXdGO2dCQUN4RixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsS0FBSyxJQUFJLElBQUksQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztvQkFDN0MsU0FBUyxJQUFJLElBQUksQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFVBQVUsQ0FDNUUsTUFBTSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUM7Z0JBQ25DLENBQUM7Z0JBQ0QsU0FBUztnQkFDVCxLQUFLLElBQUksTUFBTSxDQUFDLEtBQUssQ0FBQyxjQUFjLEdBQUcsU0FBUyxDQUFDO1lBQ2xELENBQUM7WUFFRCw2QkFBNkI7WUFDN0IsSUFBSSxDQUFDLEtBQUssQ0FBQyxLQUFLLEdBQUcsSUFBSSxDQUFDLFVBQVUsR0FBRyxLQUFLLENBQUM7WUFFM0MsZ0NBQWdDO1lBQ2hDLElBQUksQ0FBQyxLQUFLLENBQUMsY0FBYyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsU0FBUyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDO1FBQ3JFLENBQUM7UUFFRCxnQkFBZ0I7UUFDaEIsSUFBSSxHQUFHLElBQUksSUFBSSxFQUFFLENBQUM7UUFFbEIsK0NBQStDO1FBQy9DLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLElBQUksSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDO1lBQ3hDLElBQUksUUFBUSxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxDQUFDO1lBRTNDLFNBQVM7WUFDVCxJQUFJLFFBQVEsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFNBQVMsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFdBQVcsQ0FBQyxRQUFRLENBQUMsRUFBRSxDQUFDLENBQUM7WUFDMUUsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7Z0JBQ3BDLElBQUksTUFBTSxHQUFHLElBQUksQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBQ2pDLFFBQVEsSUFBSSxNQUFNLENBQUMsS0FBSyxDQUFDLGNBQWMsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FDN0QsTUFBTSxDQUFDLEVBQUUsQ0FBQyxDQUFDLFFBQVEsQ0FBQyxFQUFFLENBQUMsQ0FBQztZQUN6QixDQUFDO1lBQ0QsUUFBUSxDQUFDLE1BQU0sSUFBSSxJQUFJLEdBQUcsUUFBUSxDQUFDLENBQUMsNkJBQTZCO1FBQ2xFLENBQUM7UUFFRCxjQUFjO1FBQ2QsSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxjQUFjLENBQUM7SUFDL0MsQ0FBQztJQUVELHdCQUFPLEdBQVAsVUFBUSxNQUFNLEVBQUUsTUFBZTtRQUM5QixrQkFBa0I7UUFDbEIsRUFBRSxDQUFDLENBQUMsTUFBTSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDcEIsSUFBSSxDQUFDLGNBQWMsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDO1lBQy9CLE1BQU0sQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDO1FBQzVCLENBQUM7UUFFRCxxQ0FBcUM7UUFDckMsSUFBSSxTQUFTLEdBQUcsSUFBSSxDQUFDLFNBQVMsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUN2QyxFQUFFLENBQUMsQ0FBQyxTQUFTLElBQUksU0FBUyxDQUFDLElBQUksSUFBSSxXQUFXLENBQUMsQ0FBQyxDQUFDO1lBQ2hELG9CQUFvQjtZQUNwQixFQUFFLENBQUMsQ0FBQyxPQUFPLE1BQU0sSUFBSSxXQUFXLENBQUM7Z0JBQ2hDLFNBQVMsQ0FBQyxVQUFVLENBQUMsTUFBTSxHQUFHLE1BQU0sQ0FBQztZQUN0Qyw2QkFBNkI7WUFDN0IsTUFBTSxDQUFDLFNBQVMsQ0FBQyxVQUFVLENBQUM7UUFDN0IsQ0FBQztRQUFDLElBQUksQ0FBQyxDQUFDO1lBQ1AsMEJBQTBCO1lBQzFCLElBQUksVUFBVSxHQUFHLElBQUksTUFBTSxDQUFDLFVBQVUsQ0FBQyxJQUFJLEVBQUUsTUFBTSxFQUFFLE1BQU0sQ0FBQyxDQUFDO1FBQzlELENBQUM7UUFFRCwyQ0FBMkM7UUFDM0MsSUFBSSxDQUFDLFdBQVcsQ0FBQyxTQUFTLENBQUMsVUFBVSxDQUFDLEVBQUUsQ0FBQyxHQUFHLFVBQVUsQ0FBQztRQUN2RCxJQUFJLENBQUMsVUFBVSxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUM7UUFDcEMsTUFBTSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsVUFBVSxDQUFDLEVBQUUsQ0FBQyxHQUFHLFVBQVUsQ0FBQztRQUN0RCxNQUFNLENBQUMsS0FBSyxDQUFDLFdBQVcsQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBRTVDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLElBQUksTUFBTSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO1lBQ3RDLElBQUksS0FBSyxHQUFHLE1BQU0sQ0FBQyxLQUFLLENBQUMsUUFBUSxDQUFDLEVBQUUsQ0FBQyxDQUFDO1lBQ3RDLEtBQUssQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBQzFCLENBQUM7UUFFRCxNQUFNLENBQUMsVUFBVSxDQUFDO0lBQ25CLENBQUM7SUFFRCxxQkFBSSxHQUFKLFVBQUssVUFBVTtRQUNkLCtCQUErQjtRQUMvQixJQUFJLENBQUMsV0FBVyxDQUFDLEtBQUssQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLEdBQUcsVUFBVSxDQUFDO1FBRW5ELElBQUksTUFBTSxHQUFHLFVBQVUsQ0FBQyxFQUFFLENBQUM7UUFDM0IsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxFQUFFLElBQUksSUFBSSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDekMsaUJBQWlCO1lBQ2pCLElBQUksQ0FBQyxVQUFVLENBQUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQztZQUNwQyxJQUFJLE1BQU0sR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDO1lBQ2pELEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLElBQUksSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDO2dCQUN4QyxJQUFJLEtBQUssR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFDeEMsTUFBTSxDQUFDLEtBQUssQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUM7WUFDdEIsQ0FBQztRQUNGLENBQUM7UUFFRCxhQUFhO1FBQ2IsRUFBRSxDQUFDLENBQUMsTUFBTSxDQUFDLEVBQUUsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLFVBQVUsQ0FBQztZQUN0QyxJQUFJLENBQUMsS0FBSyxDQUFDLFVBQVUsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxDQUFDO1FBQ25ELElBQUk7WUFDSCxJQUFJLENBQUMsS0FBSyxDQUFDLFVBQVUsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxVQUFVLENBQUMsQ0FBQztRQUVqRCxZQUFZO1FBQ1osVUFBVSxDQUFDLEtBQUssR0FBRyxJQUFJLENBQUM7SUFDekIsQ0FBQztJQUVELG9FQUFvRTtJQUNwRSw4QkFBYSxHQUFiO1FBQ0MsTUFBTSxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsTUFBTSxLQUFLLENBQUMsQ0FBQztJQUN6QyxDQUFDO0lBRUQsc0ZBQXNGO0lBQ3RGLDBCQUFTLEdBQVQsVUFBVSxNQUFNO1FBQ2YsSUFBSSxNQUFNLEdBR047WUFDRixJQUFJLEVBQUUsSUFBSTtZQUNWLFVBQVUsRUFBRSxJQUFJO1NBQ2hCLENBQUM7UUFFSCxFQUFFLENBQUMsQ0FBQyxJQUFJLElBQUksTUFBTSxDQUFDLENBQUMsQ0FBQztZQUNwQixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsYUFBYSxFQUFFLENBQUMsQ0FBQyxDQUFDO2dCQUMxQixNQUFNLENBQUMsSUFBSSxHQUFHLGdCQUFnQixDQUFDO2dCQUMvQixNQUFNLENBQUMsVUFBVSxHQUFHLElBQUksQ0FBQyxjQUFjLENBQUM7Z0JBQ3hDLE1BQU0sQ0FBQyxNQUFNLENBQUM7WUFDZixDQUFDO1lBQUMsSUFBSTtnQkFDTCxNQUFNLENBQUMsSUFBSSxDQUFDO1FBQ2QsQ0FBQztRQUVELEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxJQUFJLElBQUksSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUM7WUFDbkMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLFVBQVUsSUFBSSxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDL0MsSUFBSSxVQUFVLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxVQUFVLENBQUMsQ0FBQztnQkFDcEQsRUFBRSxDQUFDLENBQUMsVUFBVSxDQUFDLEVBQUUsSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDO29CQUM3QixNQUFNLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQztvQkFDbkIsTUFBTSxDQUFDLFVBQVUsR0FBRyxVQUFVLENBQUM7b0JBQy9CLE1BQU0sQ0FBQyxNQUFNLENBQUM7Z0JBQ2YsQ0FBQztnQkFBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUMsVUFBVSxDQUFDLElBQUksSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDO29CQUN0QyxNQUFNLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQztvQkFDbkIsTUFBTSxDQUFDLFVBQVUsR0FBRyxVQUFVLENBQUM7b0JBQy9CLE1BQU0sQ0FBQyxNQUFNLENBQUM7Z0JBQ2YsQ0FBQztZQUNGLENBQUM7UUFDRixDQUFDO1FBRUQsTUFBTSxDQUFDLElBQUksQ0FBQztJQUNiLENBQUM7SUFFRCw2RkFBNkY7SUFDN0Ysc0JBQUssR0FBTDtRQUVDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLElBQUksSUFBSSxDQUFDLEtBQUssQ0FBQyxXQUFXLENBQUM7WUFDeEMsSUFBSSxDQUFDLEtBQUssQ0FBQyxXQUFXLENBQUMsS0FBSyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBRW5DLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLElBQUksSUFBSSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUM7WUFDckMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLFFBQVEsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxLQUFLLENBQUMsQ0FBQztnQkFDL0MsSUFBSSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUMsUUFBUSxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBRTNDLElBQUksQ0FBQyxLQUFLLENBQUMsY0FBYyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsU0FBUyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQztJQUN6RSxDQUFDO0lBRUQsZ0VBQWdFO0lBQ2hFLHNCQUFLLEdBQUw7UUFDQyxJQUFJLENBQUMsS0FBSyxFQUFFLENBQUM7UUFFYixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsSUFBSSxJQUFJLElBQUksQ0FBQyxXQUFXLENBQUM7WUFDakMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLFVBQVUsSUFBSSxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxDQUFDO2dCQUM3QyxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxDQUFDLFVBQVUsQ0FBQyxDQUFDLE1BQU0sR0FBRyxJQUFJLENBQUMsTUFBTSxFQUFFLEdBQUcsRUFBRSxHQUFHLEVBQUUsQ0FBQztRQUN0RSxJQUFJLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQyxNQUFNLEVBQUUsR0FBRyxFQUFFLEdBQUcsRUFBRSxDQUFDO1FBRXBDLElBQUksQ0FBQyxHQUFHLEdBQUcsSUFBSSxDQUFDLEtBQUssR0FBRyxJQUFJLENBQUMsVUFBVSxHQUFHLENBQUMsQ0FBQztJQUM3QyxDQUFDO0lBS0QsbUVBQW1FO0lBQ25FLHlCQUFRLEdBQVIsVUFBUyxTQUFTLEVBQUUsS0FBSztRQUV4QixTQUFTLEdBQUcsU0FBUyxJQUFJLEVBQUUsQ0FBQztRQUM1QixJQUFJLElBQUksR0FBRyxJQUFJLENBQUM7UUFDaEIsSUFBSSxnQkFBZ0IsR0FBRyxFQUFFLENBQUM7UUFDMUIsSUFBSSxXQUFXLEdBQUcsRUFBRSxDQUFDO1FBQ3JCLElBQUksaUJBQWlCLEdBQUcsRUFBRSxDQUFDO1FBQzNCLElBQUksS0FBSyxHQUFHLFNBQVMsQ0FBQyxNQUFNLElBQUksQ0FBQyxDQUFDO1FBQ2xDLElBQUksT0FBTyxHQUFHLFNBQVMsQ0FBQyxPQUFPLElBQUksQ0FBQyxDQUFDO1FBQ3JDLElBQUksTUFBTSxHQUFHLFNBQVMsQ0FBQyxNQUFNLElBQUksRUFBRSxDQUFDO1FBQ3BDLElBQUksT0FBTyxHQUFHLFNBQVMsQ0FBQyxPQUFPLElBQUksRUFBRSxDQUFDO1FBQ3RDLElBQUksT0FBTyxHQUFHLFNBQVMsQ0FBQyxPQUFPLElBQUksRUFBRSxDQUFDO1FBQ3RDLElBQUksU0FBUyxHQUFHLFNBQVMsQ0FBQyxTQUFTLElBQUksRUFBRSxDQUFDO1FBQzFDLElBQUksb0JBQW9CLEdBQUcsU0FBUyxDQUFDLG9CQUFvQixJQUFJLEVBQUUsQ0FBQztRQUNoRSxJQUFJLGVBQWUsR0FBRyxTQUFTLENBQUMsZUFBZSxJQUFJLEVBQUUsQ0FBQztRQUN0RCxJQUFJLHFCQUFxQixHQUFHLFNBQVMsQ0FBQyxxQkFBcUIsSUFBSSxFQUFFLENBQUM7UUFDbEUsSUFBSSxNQUFNLEdBQUcsU0FBUyxDQUFDLE1BQU0sSUFBSSxFQUFFLE9BQU8sRUFBRSxDQUFDLEVBQUUsUUFBUSxFQUFFLENBQUMsRUFBRSxDQUFDO1FBRTdELHFCQUFxQjtRQUNyQixJQUFJLFFBQVEsR0FBRyxVQUFTLEtBQUs7WUFDNUIsSUFBSSxTQUFTLEdBQUcsS0FBSyxJQUFJLE1BQU0sSUFBSSxLQUFLLENBQUMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxDQUFDO1lBQ3pELEVBQUUsQ0FBQyxDQUFDLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQztnQkFDaEIsTUFBTSxDQUFDLE9BQU8sR0FBRyxLQUFLLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsQ0FBQztnQkFDcEMsTUFBTSxDQUFDLEtBQUssQ0FBQyxHQUFHLE1BQU0sQ0FBQyxPQUFPLENBQUM7WUFDaEMsQ0FBQztRQUNGLENBQUMsQ0FBQTtRQUNELFFBQVEsQ0FBQyxvQkFBb0IsQ0FBQyxDQUFDO1FBQy9CLFFBQVEsQ0FBQyxlQUFlLENBQUMsQ0FBQztRQUMxQixRQUFRLENBQUMscUJBQXFCLENBQUMsQ0FBQztRQUNoQyxJQUFJLFlBQVksR0FBRyxNQUFNLENBQUMsT0FBTyxDQUFDO1FBRWxDLHNFQUFzRTtRQUN0RSxJQUFJLE1BQU0sR0FBRztZQUFTLGNBQWM7aUJBQWQsV0FBYyxDQUFkLHNCQUFjLENBQWQsSUFBYztnQkFBZCw2QkFBYzs7WUFDbkMsSUFBSSxFQUFFLENBQUM7WUFDUCxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsTUFBTSxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBRXRCLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsSUFBSSxRQUFRLENBQUMsQ0FBQyxDQUFDO29CQUN6QixFQUFFLEdBQUcsU0FBUyxHQUFHLE9BQU8sQ0FBQyxNQUFNLENBQUM7b0JBQ2hDLE9BQU8sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7Z0JBQ3JCLENBQUM7Z0JBQUMsSUFBSTtvQkFDTCxFQUFFLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNkLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxTQUFTLENBQUM7b0JBQ25CLE1BQU0sQ0FBQyxTQUFTLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBQ3RCLE1BQU0sQ0FBQyxTQUFTLENBQUMsRUFBRSxDQUFDLEdBQUc7b0JBQ3RCLEtBQUssRUFBRSxDQUFDO29CQUNSLEVBQUUsRUFBRSxLQUFLLEVBQUU7aUJBQ1gsQ0FBQztZQUNILENBQUM7WUFBQyxJQUFJLENBQUMsQ0FBQztnQkFDUCxJQUFJLFFBQVEsR0FBRyxJQUFJLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQztnQkFDL0IsRUFBRSxDQUFDLENBQUMsUUFBUSxDQUFDO29CQUNaLElBQUksS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQztnQkFFeEIsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLEtBQUssRUFBRSxDQUFDO2dCQUN4QixJQUFJLElBQUksR0FBRyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUM7Z0JBRXRCLEVBQUUsQ0FBQyxDQUFDLENBQUMsUUFBUSxDQUFDO29CQUNiLElBQUksS0FBSyxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQztnQkFFeEIsRUFBRSxHQUFHLElBQUksR0FBRyxHQUFHLENBQUM7Z0JBQ2hCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxRQUFRLElBQUksSUFBSSxDQUFDO29CQUN6QixFQUFFLElBQUksSUFBSSxDQUFDLFFBQVEsQ0FBQyxHQUFHLEdBQUcsQ0FBQztnQkFDNUIsRUFBRSxJQUFJLElBQUksQ0FBQyxFQUFFLENBQUM7Z0JBQ2QsRUFBRSxDQUFDLENBQUMsRUFBRSxJQUFJLFNBQVMsQ0FBQztvQkFDbkIsTUFBTSxDQUFDLFNBQVMsQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFFdEIsTUFBTSxDQUFDLFNBQVMsQ0FBQyxFQUFFLENBQUMsR0FBRztvQkFDdEIsS0FBSyxFQUFFLEtBQUs7b0JBQ1osRUFBRSxFQUFFLEtBQUssRUFBRTtpQkFDWCxDQUFDO1lBQ0gsQ0FBQztRQUNGLENBQUMsQ0FBQztRQUVGLGlCQUFpQjtRQUNqQixJQUFJLGFBQWEsR0FBRztZQUFTLGNBQWM7aUJBQWQsV0FBYyxDQUFkLHNCQUFjLENBQWQsSUFBYztnQkFBZCw2QkFBYzs7WUFDMUMsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDO1lBQ3ZCLElBQUksUUFBUSxHQUFHLEVBQUUsQ0FBQztZQUNsQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQztnQkFDbEIsRUFBRSxDQUFDLENBQUMsT0FBTyxJQUFJLENBQUMsQ0FBQyxDQUFDLElBQUksUUFBUSxDQUFDO29CQUM5QixRQUFRLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNyQixJQUFJO29CQUNILFFBQVEsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxHQUFHLENBQUM7WUFFdEMsS0FBSyxDQUFDLElBQUksQ0FBQyxRQUFRLEdBQUcsR0FBRyxDQUFDLENBQUM7UUFDNUIsQ0FBQyxDQUFBO1FBRUQsd0NBQXdDO1FBQ3hDLElBQUksT0FBTyxHQUFHLFVBQVMsR0FBRztZQUN6QixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsSUFBSSxJQUFJLEdBQUcsQ0FBQyxDQUFDLENBQUM7Z0JBQ3RCLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDLENBQUM7b0JBQzVCLE1BQU0sQ0FBQyxLQUFLLENBQUM7WUFDZixDQUFDO1lBQ0QsTUFBTSxDQUFDLElBQUksQ0FBQztRQUNiLENBQUMsQ0FBQztRQUVGLGdDQUFnQztRQUNoQyxJQUFJLGFBQWEsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUN4RCxJQUFJLE9BQU8sR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUM5QyxJQUFJLE9BQU8sR0FBRyxLQUFLLElBQUksT0FBTyxHQUFHLElBQUksR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUN6RSxJQUFJLFFBQVEsR0FBRyxLQUFLLElBQUksUUFBUSxHQUFHLElBQUksR0FBRyxhQUFhLElBQUksT0FBTyxDQUFDO1FBRW5FLDhCQUE4QjtRQUM5QixJQUFJLElBQUksR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDMUIsSUFBSSxVQUFVLEdBQUcsTUFBTSxDQUFDLElBQUksRUFBRSxZQUFZLENBQUMsQ0FBQztRQUM1QyxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUM7WUFDWCxNQUFNLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUM1QixJQUFJLENBQUMsQ0FBQztZQUNMLG9CQUFvQixDQUFDLFlBQVksQ0FBQyxDQUFDLElBQUksQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDO1lBQzFELGVBQWUsQ0FBQyxZQUFZLENBQUMsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUM7WUFDaEQscUJBQXFCLENBQUMsWUFBWSxDQUFDLENBQUMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUM7WUFDNUQsSUFBSSxHQUFHLEdBQUcsTUFBTSxDQUFDLElBQUksRUFBRSxLQUFLLENBQUMsQ0FBQztZQUM5QixJQUFJLEtBQUssR0FBRyxNQUFNLENBQUMsSUFBSSxFQUFFLE9BQU8sQ0FBQyxDQUFDO1lBQ2xDLElBQUksSUFBSSxHQUFHLE1BQU0sQ0FBQyxJQUFJLEVBQUUsTUFBTSxDQUFDLENBQUM7WUFDaEMsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxLQUFLLENBQUM7Z0JBQzdCLElBQUksU0FBUyxHQUFHLE1BQU0sQ0FBQyxJQUFJLENBQUMsY0FBYyxFQUFFLE1BQU0sQ0FBQyxDQUFDO1lBQ3JELEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxhQUFhLEVBQUUsQ0FBQztnQkFDeEIsSUFBSSxXQUFXLEdBQUcsTUFBTSxDQUFDLElBQUksQ0FBQyxjQUFjLEVBQUUsUUFBUSxDQUFDLENBQUM7WUFDekQsYUFBYSxDQUFDLEdBQUcsRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUFFLGdCQUFnQixDQUFDLENBQUM7WUFDbkQsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLGFBQWEsRUFBRSxDQUFDO2dCQUN4QixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLEtBQUssQ0FBQztvQkFDN0IsYUFBYSxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsU0FBUyxFQUFFLEtBQUssRUFBRSxXQUFXLEVBQUUsS0FBSyxFQUMvRCxLQUFLLEVBQUUsS0FBSyxFQUFFLElBQUksRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDO2dCQUN4QyxJQUFJO29CQUNILGFBQWEsQ0FBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLFdBQVcsRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUFFLEtBQUssRUFDM0QsSUFBSSxFQUFFLGdCQUFnQixDQUFDLENBQUM7WUFDM0IsSUFBSTtnQkFDSCxhQUFhLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxJQUFJLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQztZQUNyRCxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQztnQkFDdkMsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ3ZDLElBQUksZ0JBQWdCLEdBQUcsTUFBTSxDQUFDLEtBQUssQ0FBQyxJQUFJLEVBQUUsWUFBWSxDQUFDLENBQUM7Z0JBQ3hELElBQUksWUFBWSxHQUFHLE1BQU0sQ0FBQyxLQUFLLEVBQUUsUUFBUSxDQUFDLENBQUM7Z0JBQzNDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUM7b0JBQ2YsSUFBSSxVQUFVLEdBQUcsTUFBTSxDQUFDLEtBQUssRUFBRSxNQUFNLENBQUMsQ0FBQztnQkFDeEMsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDO29CQUNwQyxhQUFhLENBQUMsS0FBSyxFQUFFLE1BQU0sRUFBRSxnQkFBZ0IsRUFBRSxLQUFLLEVBQ25ELFlBQVksRUFBRSxLQUFLLEVBQUUsVUFBVSxFQUFFLGdCQUFnQixDQUFDLENBQUM7Z0JBQ3JELElBQUk7b0JBQ0gsYUFBYSxDQUFDLEtBQUssRUFBRSxNQUFNLEVBQUUsZ0JBQWdCLEVBQUUsS0FBSyxFQUNuRCxZQUFZLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQztZQUNuQyxDQUFDO1lBQ0QsSUFBSSxVQUFVLEdBQUcsTUFBTSxDQUFDLElBQUksRUFBRSxZQUFZLENBQUMsQ0FBQztZQUM1QyxNQUFNLENBQUMsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQztnQkFDckIsS0FBSyxNQUFNLENBQUMsUUFBUTtvQkFDbkIsYUFBYSxDQUFDLFVBQVUsRUFBRSx5QkFBeUIsRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUNoRSxnQkFBZ0IsQ0FBQyxDQUFDO29CQUNuQixhQUFhLENBQUMsVUFBVSxFQUFFLEtBQUssRUFBRSxVQUFVLEVBQUUsVUFBVSxFQUN0RCxVQUFVLEVBQUUsR0FBRyxFQUFFLGdCQUFnQixDQUFDLENBQUM7b0JBQ3BDLEtBQUssQ0FBQztnQkFDUCxLQUFLLE1BQU0sQ0FBQyxJQUFJO29CQUNmLElBQUksRUFBRSxHQUFHLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQztvQkFDdkIsSUFBSSxFQUFFLEdBQUcsTUFBTSxDQUFDLE9BQU8sQ0FBQyxDQUFDO29CQUN6QixhQUFhLENBQUMsRUFBRSxFQUFFLGNBQWMsRUFBRSxLQUFLLEVBQUUsR0FBRyxFQUFFLGdCQUFnQixDQUFDLENBQUM7b0JBQ2hFLGFBQWEsQ0FBQyxFQUFFLEVBQUUsU0FBUyxFQUFFLEVBQUUsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDO29CQUNuRCxhQUFhLENBQUMsVUFBVSxFQUFFLE1BQU0sRUFBRSxFQUFFLEVBQUUsS0FBSyxFQUFFLEVBQUUsRUFBRSxPQUFPLEVBQUUsRUFBRSxFQUFFLEtBQUssRUFBRSxFQUFFLEVBQUUsR0FBRyxFQUFFLGdCQUFnQixDQUFDLENBQUM7b0JBQ2hHLGFBQWEsQ0FBQyxVQUFVLEVBQUUsVUFBVSxFQUFFLFVBQVUsRUFBRSxLQUFLLEVBQUUsVUFBVSxFQUFFLEdBQUcsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDO29CQUM1RixLQUFLLENBQUM7Z0JBQ1AsS0FBSyxNQUFNLENBQUMsUUFBUTtvQkFDbkIsYUFBYSxDQUFDLFVBQVUsRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUFFLGdCQUFnQixDQUFDLENBQUM7b0JBQzFELGFBQWEsQ0FBQyxVQUFVLEVBQUUsTUFBTSxFQUFFLGdCQUFnQixDQUFDLENBQUM7b0JBQ3BELEtBQUssQ0FBQztnQkFDUCxLQUFLLE1BQU0sQ0FBQyxJQUFJO29CQUNmLGFBQWEsQ0FBQyxVQUFVLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBRSxPQUFPLEVBQ2hELGdCQUFnQixDQUFDLENBQUM7b0JBQ25CLGFBQWEsQ0FBQyxVQUFVLEVBQUUsTUFBTSxFQUFFLGdCQUFnQixDQUFDLENBQUM7b0JBQ3BELEtBQUssQ0FBQztZQUNSLENBQUM7WUFHRCxJQUFJLFVBQVUsR0FBRyxFQUFFLENBQUM7WUFDcEIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7Z0JBQ3BDLG1EQUFtRDtnQkFFbkQsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBQ3JDLElBQUksTUFBTSxHQUFHLElBQUksQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBQ2pDLElBQUksU0FBUyxHQUFHLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQztnQkFDOUIsSUFBSSxVQUFVLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxLQUFLLENBQUMsQ0FBQztnQkFDdkMsSUFBSSxXQUFXLEdBQUcsS0FBSyxDQUFDO2dCQUN4QixFQUFFLENBQUMsQ0FBQyxNQUFNLENBQUMsY0FBYyxDQUFDLEtBQUssSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDO29CQUN6QyxhQUFhLENBQUMsU0FBUyxFQUFFLEtBQUssRUFBRSxVQUFVLEVBQUUsV0FBVyxDQUFDLENBQUM7b0JBQ3pELFdBQVcsR0FBRyxJQUFJLENBQUM7Z0JBQ3BCLENBQUM7Z0JBQ0QsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLFFBQVEsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLFVBQVUsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO29CQUN2RCxJQUFJLGVBQWUsR0FBRyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxVQUFVLENBQUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxDQUM1RCxRQUFRLENBQUMsRUFBRSxRQUFRLENBQUMsQ0FBQztvQkFDdEIsSUFBSSxtQkFBbUIsR0FBRyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxVQUFVLENBQUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxDQUNoRSxRQUFRLENBQUMsQ0FBQyxJQUFJLEVBQUUsWUFBWSxDQUFDLENBQUM7b0JBRS9CLEVBQUUsQ0FBQyxDQUFDLFdBQVcsQ0FBQzt3QkFDZixhQUFhLENBQUMsU0FBUyxFQUFFLE1BQU0sRUFBRSxlQUFlLEVBQUUsS0FBSyxFQUN0RCxtQkFBbUIsRUFBRSxXQUFXLENBQUMsQ0FBQztvQkFDcEMsSUFBSSxDQUFDLENBQUM7d0JBQ0wsYUFBYSxDQUFDLFNBQVMsRUFBRSxLQUFLLEVBQUUsZUFBZSxFQUFFLEtBQUssRUFDckQsbUJBQW1CLEVBQUUsV0FBVyxDQUFDLENBQUM7d0JBQ25DLFdBQVcsR0FBRyxJQUFJLENBQUM7b0JBQ3BCLENBQUM7Z0JBQ0YsQ0FBQztnQkFFRCxVQUFVLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFDM0IsYUFBYSxDQUFDLGFBQWEsR0FBRyxDQUFDLFVBQVUsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLEdBQUcsTUFBTSxFQUFFLFNBQVMsRUFBRSxXQUFXLENBQUMsQ0FBQztZQUN6RixDQUFDO1lBRUQsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUM7Z0JBQ3ZDLElBQUksS0FBSyxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUN2QyxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDO29CQUNmLElBQUksVUFBVSxHQUFHLE1BQU0sQ0FBQyxLQUFLLEVBQUUsTUFBTSxDQUFDLENBQUM7Z0JBQ3hDLElBQUksZ0JBQWdCLEdBQUcsTUFBTSxDQUFDLEtBQUssQ0FBQyxJQUFJLEVBQUUsWUFBWSxDQUFDLENBQUM7Z0JBQ3hELElBQUksS0FBSyxHQUFHLE1BQU0sQ0FBQyxJQUFJLEVBQUUsT0FBTyxFQUFFLGFBQWEsRUFBRSxLQUFLLENBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxLQUFLO3FCQUNuRSxXQUFXLENBQUMsS0FBSyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7Z0JBQ3pCLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxhQUFhLEVBQUUsQ0FBQyxDQUFDLENBQUM7b0JBQzFCLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQzt3QkFDL0IsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLEtBQUssQ0FBQzs0QkFDZixhQUFhLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxTQUFTLEVBQUUsS0FBSyxFQUFFLFdBQVcsRUFDeEQsS0FBSyxFQUFFLEtBQUssRUFBRSxLQUFLLEVBQUUsVUFBVSxFQUFFLEtBQUssRUFBRSxnQkFBZ0IsRUFDeEQsV0FBVyxDQUFDLENBQUM7d0JBQ2YsSUFBSTs0QkFDSCxhQUFhLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxTQUFTLEVBQUUsS0FBSyxFQUFFLFdBQVcsRUFDeEQsS0FBSyxFQUFFLEtBQUssRUFBRSxLQUFLLEVBQUUsZ0JBQWdCLEVBQUUsV0FBVyxDQUFDLENBQUM7b0JBQ3ZELENBQUM7b0JBQUMsSUFBSSxDQUFDLENBQUM7d0JBQ1AsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLEtBQUssQ0FBQzs0QkFDZixhQUFhLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxXQUFXLEVBQUUsS0FBSyxFQUFFLEtBQUssRUFBRSxLQUFLLEVBQzNELFVBQVUsRUFBRSxLQUFLLEVBQUUsZ0JBQWdCLEVBQUUsV0FBVyxDQUFDLENBQUM7d0JBQ3BELElBQUk7NEJBQ0gsYUFBYSxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsV0FBVyxFQUFFLEtBQUssRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUMzRCxnQkFBZ0IsRUFBRSxXQUFXLENBQUMsQ0FBQztvQkFDbEMsQ0FBQztnQkFDRixDQUFDO2dCQUFDLElBQUksQ0FBQyxDQUFDO29CQUNQLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUM7d0JBQ2YsYUFBYSxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsVUFBVSxFQUFFLEtBQUssRUFBRSxnQkFBZ0IsRUFDOUQsV0FBVyxDQUFDLENBQUM7b0JBQ2YsSUFBSTt3QkFDSCxhQUFhLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxnQkFBZ0IsRUFBRSxXQUFXLENBQUMsQ0FBQztnQkFDN0QsQ0FBQztnQkFDRCxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxJQUFJLElBQUksQ0FBQyxLQUFLLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztvQkFDcEMsNkJBQTZCO29CQUM3QixJQUFJLE1BQU0sR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxFQUFFLENBQUMsQ0FBQztvQkFDckMsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLFVBQVUsQ0FBQyxFQUFFLENBQUMsQ0FBQztvQkFDakMsSUFBSSxTQUFTLEdBQUcsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDO29CQUM5QixJQUFJLFVBQVUsR0FBRyxNQUFNLENBQUMsTUFBTSxFQUFFLEtBQUssQ0FBQyxDQUFDO29CQUV2QyxJQUFJLEtBQUssR0FBRyxNQUFNLENBQUMsSUFBSSxFQUFFLE9BQU8sRUFBRSxhQUFhLEVBQUUsS0FBSyxDQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsS0FBSzt5QkFDbkUsV0FBVyxDQUFDLEtBQUssQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO29CQUN6QixJQUFJLE1BQU0sR0FBRyxNQUFNLENBQUMsSUFBSSxFQUFFLE9BQU8sRUFBRSxVQUFVLEVBQUUsTUFBTSxDQUFDLEVBQUUsRUFBRSxLQUFLLENBQUMsRUFBRSxFQUNqRSxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7b0JBQzNDLEVBQUUsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxhQUFhLEVBQUUsQ0FBQzt3QkFDMUIsSUFBSSxrQkFBa0IsR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFDLGNBQWMsRUFBRSxRQUFRLENBQUMsQ0FBQztvQkFDbEUsRUFBRSxDQUFDLENBQUMsTUFBTSxDQUFDLGNBQWMsQ0FBQyxLQUFLLENBQUM7d0JBQy9CLElBQUksZ0JBQWdCLEdBQUcsTUFBTSxDQUFDLE1BQU0sQ0FBQyxjQUFjLEVBQUUsTUFBTSxDQUFDLENBQUM7b0JBQzlELEVBQUUsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxhQUFhLEVBQUUsQ0FBQzt3QkFDMUIsRUFBRSxDQUFDLENBQUMsTUFBTSxDQUFDLGNBQWMsQ0FBQyxLQUFLLENBQUM7NEJBQy9CLGFBQWEsQ0FBQyxNQUFNLEVBQUUsS0FBSyxFQUFFLGdCQUFnQixFQUFFLEtBQUssRUFDbkQsa0JBQWtCLEVBQUUsS0FBSyxFQUFFLE1BQU0sRUFBRSxLQUFLLEVBQUUsVUFBVSxFQUFFLEtBQUssRUFDM0QsS0FBSyxFQUFFLEtBQUssRUFBRSxhQUFhLEdBQUcsVUFBVSxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLEdBQUcsR0FBRyxFQUFFLFdBQVcsQ0FBQyxDQUFDO3dCQUNsRixJQUFJOzRCQUNILGFBQWEsQ0FBQyxNQUFNLEVBQUUsS0FBSyxFQUFFLGtCQUFrQixFQUFFLEtBQUssRUFDckQsTUFBTSxFQUFFLEtBQUssRUFBRSxVQUFVLEVBQUUsS0FBSyxFQUFFLEtBQUssRUFBRSxLQUFLLEVBQzlDLGFBQWEsR0FBRyxVQUFVLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsR0FBRyxHQUFHLEVBQUUsV0FBVyxDQUFDLENBQUM7b0JBQ3JFLElBQUk7d0JBQ0gsYUFBYSxDQUFDLE1BQU0sRUFBRSxLQUFLLEVBQUUsVUFBVSxFQUFFLEtBQUssRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUMzRCxhQUFhLEdBQUcsVUFBVSxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLEdBQUcsR0FBRyxFQUFFLFdBQVcsQ0FBQyxDQUFDO2dCQUNyRSxDQUFDO1lBQ0YsQ0FBQztZQUNELEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxVQUFVLElBQUksSUFBSSxDQUFDLFdBQVcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDO2dCQUMvQyxJQUFJLFVBQVUsR0FBRyxNQUFNLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUM7Z0JBQ3BFLGFBQWEsQ0FBQyxVQUFVLEVBQUUsS0FBSyxFQUFFLFVBQVUsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDO1lBQ2hFLENBQUM7UUFDRixDQUFDO1FBQ0QsRUFBRSxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDO1lBQ2QsSUFBSSxjQUFjLEdBQUcsTUFBTSxDQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsZ0JBQWdCLEVBQUUsSUFBSSxDQUFDLEtBQUs7aUJBQ3JFLGNBQWMsQ0FBQyxDQUFDO1lBQ2xCLEVBQUUsQ0FBQyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7Z0JBQ2QsSUFBSSxNQUFNLEdBQUcsTUFBTSxDQUFDLFFBQVEsQ0FBQyxDQUFDO2dCQUM5QixhQUFhLENBQUMsY0FBYyxFQUFFLEtBQUssRUFBRSxNQUFNLEVBQUUsS0FBSyxFQUFFLFVBQVUsRUFDN0QsaUJBQWlCLENBQUMsQ0FBQztnQkFDcEIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsSUFBSSxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUM7b0JBQ3hDLElBQUksS0FBSyxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxDQUFDO29CQUN4QyxJQUFJLEtBQUssR0FBRyxNQUFNLENBQUMsSUFBSSxFQUFFLE9BQU8sRUFBRSxhQUFhLEVBQUUsS0FBSyxDQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsS0FBSzt5QkFDbkUsV0FBVyxDQUFDLEtBQUssQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO29CQUN6QixJQUFJLFlBQVksR0FBRyxNQUFNLENBQUMsS0FBSyxFQUFFLFFBQVEsQ0FBQyxDQUFDO29CQUMzQyxhQUFhLENBQUMsWUFBWSxFQUFFLE1BQU0sRUFBRSxJQUFJLEVBQUUsTUFBTSxFQUFFLGNBQWMsRUFDL0QsS0FBSyxFQUFFLEtBQUssRUFBRSxHQUFHLEVBQUUsaUJBQWlCLENBQUMsQ0FBQztnQkFDeEMsQ0FBQztnQkFDRCxPQUFPLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxFQUFFLENBQUMsQ0FBQztZQUM3QixDQUFDO1lBQUMsSUFBSSxDQUFDLENBQUM7Z0JBQ1AsRUFBRSxDQUFDLENBQUMsQ0FBQyxhQUFhLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDO29CQUNoQyxJQUFJLEtBQUssR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUM7b0JBQzFCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLElBQUksSUFBSSxDQUFDLFdBQVcsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDO3dCQUMzQyxJQUFJLFVBQVUsR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLFNBQVMsQ0FBQyxFQUFFLENBQUMsQ0FBQzt3QkFDaEQsSUFBSSxNQUFNLEdBQUcsVUFBVSxDQUFDLEVBQUUsQ0FBQzt3QkFDM0IsSUFBSSxpQkFBaUIsR0FBRyxNQUFNLENBQUMsVUFBVSxFQUFFLFFBQVEsQ0FBQyxDQUFDO3dCQUNyRCxJQUFJLHFCQUFxQixHQUFHLE1BQU0sQ0FBQyxNQUFNLEVBQUUsT0FBTyxFQUNqRCxnQkFBZ0IsRUFBRSxNQUFNLENBQUMsS0FBSyxDQUFDLGNBQWMsQ0FBQyxDQUFDO3dCQUNoRCxFQUFFLENBQUMsQ0FBQyxVQUFVLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQzs0QkFDdEIsSUFBSSxlQUFlLEdBQUcsTUFBTSxDQUFDLFVBQVUsRUFBRSxNQUFNLENBQUMsQ0FBQzs0QkFDakQsYUFBYSxDQUFDLEtBQUssRUFBRSxNQUFNLEVBQUUscUJBQXFCLEVBQUUsS0FBSyxFQUN4RCxlQUFlLEVBQUUsS0FBSyxFQUFFLGlCQUFpQixFQUN6QyxpQkFBaUIsQ0FBQyxDQUFDO3dCQUNyQixDQUFDO3dCQUFDLElBQUk7NEJBQ0wsYUFBYSxDQUFDLEtBQUssRUFBRSxNQUFNLEVBQUUscUJBQXFCLEVBQUUsS0FBSyxFQUN4RCxpQkFBaUIsRUFBRSxpQkFBaUIsQ0FBQyxDQUFDO29CQUN6QyxDQUFDO29CQUNELElBQUksU0FBUyxHQUFHLE1BQU0sQ0FBQyxJQUFJLEVBQUUsT0FBTyxFQUFFLFdBQVcsRUFBRSxJQUFJLENBQUMsS0FBSyxDQUFDLFNBQVMsQ0FBQyxDQUFDO29CQUN6RSxhQUFhLENBQUMsU0FBUyxFQUFFLEtBQUssRUFBRSxVQUFVLEVBQUUsS0FBSyxFQUFFLEtBQUssRUFDdkQsaUJBQWlCLENBQUMsQ0FBQztvQkFDcEIsYUFBYSxDQUFDLEtBQUssRUFBRSxNQUFNLEVBQUUsaUJBQWlCLENBQUMsQ0FBQztvQkFDaEQsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7d0JBQ3BDLElBQUksTUFBTSxHQUFHLElBQUksQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLENBQUM7d0JBQ2pDLElBQUksU0FBUyxHQUFHLE1BQU0sQ0FBQyxPQUFPLENBQUMsQ0FBQzt3QkFDaEMsSUFBSSxVQUFVLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxLQUFLLENBQUMsQ0FBQzt3QkFDdkMsRUFBRSxDQUFDLENBQUMsTUFBTSxDQUFDLGNBQWMsQ0FBQyxLQUFLLElBQUksSUFBSSxDQUFDOzRCQUN2QyxhQUFhLENBQUMsU0FBUyxFQUFFLEtBQUssRUFBRSxVQUFVLEVBQUUsaUJBQWlCLENBQUMsQ0FBQzt3QkFDaEUsSUFBSTs0QkFDSCxhQUFhLENBQUMsU0FBUyxFQUFFLE1BQU0sRUFBRSxpQkFBaUIsQ0FBQyxDQUFDO3dCQUNyRCxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsY0FBYyxJQUFJLElBQUksQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7NEJBQzdELElBQUksVUFBVSxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsQ0FBQyxjQUFjLENBQUMsQ0FBQzs0QkFDbEUsSUFBSSxpQkFBaUIsR0FBRyxNQUFNLENBQUMsVUFBVSxFQUFFLFFBQVEsQ0FBQyxDQUFDOzRCQUNyRCxJQUFJLGlCQUFpQixHQUFHLE1BQU0sQ0FBQyxVQUFVLENBQUMsSUFBSSxFQUFFLFlBQVksQ0FBQyxDQUFDOzRCQUM5RCxhQUFhLENBQUMsU0FBUyxFQUFFLE1BQU0sRUFBRSxpQkFBaUIsRUFBRSxLQUFLLEVBQ3hELGlCQUFpQixFQUFFLGlCQUFpQixDQUFDLENBQUM7d0JBQ3hDLENBQUM7d0JBQ0QsSUFBSSxxQkFBcUIsR0FBRyxNQUFNLENBQUMsTUFBTSxFQUFFLE9BQU8sRUFDakQsZ0JBQWdCLEVBQUUsTUFBTSxDQUFDLEtBQUssQ0FBQyxjQUFjLENBQUMsQ0FBQzt3QkFDaEQsYUFBYSxDQUFDLEtBQUssRUFBRSxNQUFNLEVBQUUscUJBQXFCLEVBQUUsS0FBSyxFQUN4RCxTQUFTLEVBQUUsaUJBQWlCLENBQUMsQ0FBQztvQkFDaEMsQ0FBQztvQkFDRCxJQUFJLEtBQUssR0FBRyxNQUFNLENBQUMsSUFBSSxFQUFFLE9BQU8sRUFBRSxPQUFPLEVBQUUsSUFBSSxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQztvQkFDN0QsYUFBYSxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsVUFBVSxFQUFFLEtBQUssRUFBRSxLQUFLLEVBQ25ELGlCQUFpQixDQUFDLENBQUM7b0JBQ3BCLGFBQWEsQ0FBQyxjQUFjLEVBQUUsS0FBSyxFQUFFLFNBQVMsRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUMzRCxpQkFBaUIsQ0FBQyxDQUFDO29CQUNwQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxJQUFJLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQzt3QkFDeEMsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUM7d0JBQ3hDLElBQUksUUFBUSxHQUFHLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQzt3QkFDN0IsSUFBSSxLQUFLLEdBQUcsTUFBTSxDQUFDLElBQUksRUFBRSxPQUFPLEVBQUUsYUFBYSxFQUFFLEtBQUssQ0FBQyxFQUFFLEVBQUUsSUFBSTs2QkFDN0QsS0FBSyxDQUFDLFdBQVcsQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQzt3QkFDL0IsYUFBYSxDQUFDLFFBQVEsRUFBRSxLQUFLLEVBQUUsU0FBUyxFQUFFLEtBQUssRUFBRSxLQUFLLEVBQ3JELGlCQUFpQixDQUFDLENBQUM7d0JBQ3BCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLElBQUksSUFBSSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDOzRCQUNwQyxJQUFJLE1BQU0sR0FBRyxJQUFJLENBQUMsVUFBVSxDQUFDLEVBQUUsQ0FBQyxDQUFDOzRCQUNqQyxJQUFJLHFCQUFxQixHQUFHLE1BQU0sQ0FBQyxNQUFNLEVBQUUsT0FBTyxFQUNqRCxnQkFBZ0IsRUFBRSxNQUFNLENBQUMsS0FBSyxDQUFDLGNBQWMsQ0FBQyxDQUFDOzRCQUNoRCxJQUFJLE1BQU0sR0FBRyxNQUFNLENBQUMsSUFBSSxFQUFFLE9BQU8sRUFBRSxVQUFVLEVBQUUsTUFBTSxDQUFDLEVBQUUsRUFDdkQsS0FBSyxDQUFDLEVBQUUsRUFBRSxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7NEJBQ3JELGFBQWEsQ0FBQyxRQUFRLEVBQUUsTUFBTSxFQUFFLHFCQUFxQixFQUFFLEtBQUssRUFDM0QsTUFBTSxFQUFFLGlCQUFpQixDQUFDLENBQUM7d0JBQzdCLENBQUM7d0JBQ0QsSUFBSSxZQUFZLEdBQUcsTUFBTSxDQUFDLEtBQUssRUFBRSxRQUFRLENBQUMsQ0FBQzt3QkFDM0MsYUFBYSxDQUFDLFlBQVksRUFBRSxNQUFNLEVBQUUsSUFBSSxFQUFFLEtBQUssRUFBRSxRQUFRLEVBQ3hELGlCQUFpQixDQUFDLENBQUM7b0JBQ3JCLENBQUM7Z0JBRUYsQ0FBQztnQkFBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztvQkFDcEIsYUFBYSxDQUFDLGNBQWMsRUFBRSxNQUFNLEVBQUUsaUJBQWlCLENBQUMsQ0FBQztvQkFDekQsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsSUFBSSxJQUFJLENBQUMsV0FBVyxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUM7d0JBQzNDLElBQUksVUFBVSxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsU0FBUyxDQUFDLEVBQUUsQ0FBQyxDQUFDO3dCQUNoRCxJQUFJLE1BQU0sR0FBRyxVQUFVLENBQUMsRUFBRSxDQUFDO3dCQUMzQixJQUFJLGlCQUFpQixHQUFHLE1BQU0sQ0FBQyxVQUFVLEVBQUUsUUFBUSxDQUFDLENBQUM7d0JBQ3JELElBQUkscUJBQXFCLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxPQUFPLEVBQ2pELGdCQUFnQixFQUFFLE1BQU0sQ0FBQyxLQUFLLENBQUMsY0FBYyxDQUFDLENBQUM7d0JBQ2hELEVBQUUsQ0FBQyxDQUFDLFVBQVUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDOzRCQUN0QixJQUFJLGVBQWUsR0FBRyxNQUFNLENBQUMsVUFBVSxFQUFFLE1BQU0sQ0FBQyxDQUFDOzRCQUNqRCxhQUFhLENBQUMsY0FBYyxFQUFFLE1BQU0sRUFBRSxxQkFBcUIsRUFDMUQsS0FBSyxFQUFFLGVBQWUsRUFBRSxLQUFLLEVBQUUsaUJBQWlCLEVBQ2hELGlCQUFpQixDQUFDLENBQUM7d0JBQ3JCLENBQUM7d0JBQUMsSUFBSTs0QkFDTCxhQUFhLENBQUMsY0FBYyxFQUFFLE1BQU0sRUFBRSxxQkFBcUIsRUFDMUQsS0FBSyxFQUFFLGlCQUFpQixFQUFFLGlCQUFpQixDQUFDLENBQUM7b0JBQ2hELENBQUM7b0JBQ0QsYUFBYSxDQUFDLGNBQWMsRUFBRSxNQUFNLEVBQUUsVUFBVSxFQUMvQyxpQkFBaUIsQ0FBQyxDQUFDO29CQUNwQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxJQUFJLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQzt3QkFDeEMsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUM7d0JBQ3hDLElBQUksS0FBSyxHQUFHLE1BQU0sQ0FBQyxJQUFJLEVBQUUsT0FBTyxFQUFFLGFBQWEsRUFBRSxLQUFLLENBQUMsRUFBRSxFQUFFLElBQUk7NkJBQzdELEtBQUssQ0FBQyxXQUFXLENBQUMsS0FBSyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7d0JBQy9CLElBQUksWUFBWSxHQUFHLE1BQU0sQ0FBQyxLQUFLLEVBQUUsUUFBUSxDQUFDLENBQUM7d0JBQzNDLGFBQWEsQ0FBQyxZQUFZLEVBQUUsTUFBTSxFQUFFLElBQUksRUFBRSxNQUFNLEVBQy9DLGNBQWMsRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUFFLEdBQUcsRUFBRSxpQkFBaUIsQ0FBQyxDQUFDO29CQUN4RCxDQUFDO2dCQUNGLENBQUM7Z0JBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUM7b0JBQzFCLGFBQWEsQ0FBQyxjQUFjLEVBQUUsTUFBTSxFQUFFLGlCQUFpQixDQUFDLENBQUM7b0JBQ3pELEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLElBQUksSUFBSSxDQUFDLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO3dCQUNwQyxJQUFJLE1BQU0sR0FBRyxJQUFJLENBQUMsVUFBVSxDQUFDLEVBQUUsQ0FBQyxDQUFDO3dCQUNqQyxJQUFJLFNBQVMsR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUM7d0JBQzlCLElBQUksVUFBVSxHQUFHLE1BQU0sQ0FBQyxNQUFNLEVBQUUsS0FBSyxDQUFDLENBQUM7d0JBQ3ZDLEVBQUUsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxjQUFjLENBQUMsS0FBSyxJQUFJLElBQUksQ0FBQzs0QkFDdkMsYUFBYSxDQUFDLFNBQVMsRUFBRSxLQUFLLEVBQUUsVUFBVSxFQUFFLGlCQUFpQixDQUFDLENBQUM7d0JBQ2hFLElBQUk7NEJBQ0gsYUFBYSxDQUFDLFNBQVMsRUFBRSxNQUFNLEVBQUUsaUJBQWlCLENBQUMsQ0FBQzt3QkFDckQsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLGNBQWMsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLFVBQVUsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDOzRCQUM3RCxJQUFJLFVBQVUsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFVBQVUsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUMsY0FBYyxDQUFDLENBQUM7NEJBQ2xFLElBQUksaUJBQWlCLEdBQUcsTUFBTSxDQUFDLFVBQVUsRUFBRSxRQUFRLENBQUMsQ0FBQzs0QkFDckQsSUFBSSxpQkFBaUIsR0FBRyxNQUFNLENBQUMsVUFBVSxDQUFDLElBQUksRUFBRSxZQUFZLENBQUMsQ0FBQzs0QkFDOUQsYUFBYSxDQUFDLFNBQVMsRUFBRSxNQUFNLEVBQUUsaUJBQWlCLEVBQUUsS0FBSyxFQUN4RCxpQkFBaUIsRUFBRSxpQkFBaUIsQ0FBQyxDQUFDO3dCQUN4QyxDQUFDO3dCQUNELElBQUkscUJBQXFCLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxPQUFPLEVBQ2pELGdCQUFnQixFQUFFLE1BQU0sQ0FBQyxLQUFLLENBQUMsY0FBYyxDQUFDLENBQUM7d0JBQ2hELGFBQWEsQ0FBQyxjQUFjLEVBQUUsTUFBTSxFQUFFLHFCQUFxQixFQUMxRCxLQUFLLEVBQUUsU0FBUyxFQUFFLGlCQUFpQixDQUFDLENBQUM7b0JBQ3ZDLENBQUM7b0JBQ0QsYUFBYSxDQUFDLGNBQWMsRUFBRSxNQUFNLEVBQUUsVUFBVSxFQUMvQyxpQkFBaUIsQ0FBQyxDQUFDO29CQUNwQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxJQUFJLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQzt3QkFDeEMsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUM7d0JBQ3hDLElBQUksUUFBUSxHQUFHLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQzt3QkFDN0IsYUFBYSxDQUFDLFFBQVEsRUFBRSxNQUFNLEVBQUUsaUJBQWlCLENBQUMsQ0FBQzt3QkFDbkQsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7NEJBQ3BDLElBQUksTUFBTSxHQUFHLElBQUksQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLENBQUM7NEJBQ2pDLElBQUkscUJBQXFCLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxPQUFPLEVBQ2pELGdCQUFnQixFQUFFLE1BQU0sQ0FBQyxLQUFLLENBQUMsY0FBYyxDQUFDLENBQUM7NEJBQ2hELElBQUksTUFBTSxHQUFHLE1BQU0sQ0FBQyxJQUFJLEVBQUUsT0FBTyxFQUFFLFVBQVUsRUFBRSxNQUFNLENBQUMsRUFBRSxFQUN2RCxLQUFLLENBQUMsRUFBRSxFQUFFLElBQUksQ0FBQyxLQUFLLENBQUMsUUFBUSxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQzs0QkFDckQsYUFBYSxDQUFDLFFBQVEsRUFBRSxNQUFNLEVBQUUscUJBQXFCLEVBQUUsS0FBSyxFQUMzRCxNQUFNLEVBQUUsaUJBQWlCLENBQUMsQ0FBQzt3QkFDN0IsQ0FBQzt3QkFDRCxJQUFJLFlBQVksR0FBRyxNQUFNLENBQUMsS0FBSyxFQUFFLFFBQVEsQ0FBQyxDQUFDO3dCQUMzQyxhQUFhLENBQUMsWUFBWSxFQUFFLE1BQU0sRUFBRSxJQUFJLEVBQUUsS0FBSyxFQUFFLFFBQVEsRUFDeEQsaUJBQWlCLENBQUMsQ0FBQztvQkFDckIsQ0FBQztnQkFDRixDQUFDO1lBQ0YsQ0FBQztZQUNELGFBQWEsQ0FBQyxJQUFJLEVBQUUsTUFBTSxFQUFFLElBQUksRUFBRSxLQUFLLEVBQUUsY0FBYyxFQUN0RCxpQkFBaUIsQ0FBQyxDQUFDO1FBQ3JCLENBQUM7UUFDRCxNQUFNLENBQUM7WUFDTixNQUFNLEVBQUUsS0FBSztZQUNiLE9BQU8sRUFBRSxPQUFPLEdBQUcsQ0FBQztZQUNwQixNQUFNLEVBQUUsTUFBTTtZQUNkLE9BQU8sRUFBRSxPQUFPO1lBQ2hCLE9BQU8sRUFBRSxPQUFPO1lBQ2hCLFNBQVMsRUFBRSxTQUFTO1lBQ3BCLG9CQUFvQixFQUFFLG9CQUFvQjtZQUMxQyxlQUFlLEVBQUUsZUFBZTtZQUNoQyxxQkFBcUIsRUFBRSxxQkFBcUI7WUFDNUMsTUFBTSxFQUFFLE1BQU07U0FDZCxDQUFBO0lBQ0YsQ0FBQztJQUNGLGFBQUM7QUFBRCxDQS90QkEsQUErdEJDLElBQUE7QUEvdEJZLGNBQU0sU0ErdEJsQixDQUFBO0FBRUQsSUFBYyxNQUFNLENBb0NuQjtBQXBDRCxXQUFjLE1BQU0sRUFBQyxDQUFDO0lBUXJCO1FBT0Msb0JBQVksSUFBSSxFQUFFLEVBQUUsRUFBRSxNQUFlO1lBTnJDLE9BQUUsR0FBRyxVQUFVLENBQUMsR0FBRyxFQUFFLENBQUM7WUFHdEIsU0FBSSxHQUFXLENBQUMsQ0FBQztZQUNqQixXQUFNLEdBQVcsQ0FBQyxDQUFDO1lBQ25CLFVBQUssR0FBUSxJQUFJLENBQUM7WUFFakIsRUFBRSxDQUFDLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxFQUFFLENBQUM7Z0JBQ2hCLE1BQU0sbUNBQW1DLENBQUM7WUFDM0MsSUFBSSxDQUFDLElBQUksR0FBRyxJQUFJLENBQUM7WUFDakIsSUFBSSxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUM7WUFDYixJQUFJLENBQUMsTUFBTSxHQUFHLE9BQU8sTUFBTSxJQUFJLFdBQVcsSUFBSSxLQUFLLENBQUMsTUFBTSxDQUFDLEdBQUcsSUFBSSxDQUFDLE1BQU0sRUFBRSxHQUFHLEVBQUUsR0FBRyxFQUFFO2dCQUNwRixNQUFNLENBQUM7UUFDVCxDQUFDO1FBQ0YsaUJBQUM7SUFBRCxDQWZBLEFBZUMsSUFBQTtJQWZZLGlCQUFVLGFBZXRCLENBQUE7SUFFVSxnQkFBUyxHQUFHLENBQUMsQ0FBQztJQUN6QjtRQUNDLE1BQU0sQ0FBQyxnQkFBUyxFQUFFLENBQUM7SUFDcEIsQ0FBQztJQUZlLFVBQUcsTUFFbEIsQ0FBQTtJQUVEO1FBQ0MsTUFBTSxDQUFDO1lBQ04sT0FBTyxFQUFFLGdCQUFTO1lBQ2xCLFdBQVcsRUFBRSxVQUFVLENBQUMsYUFBYTtTQUNyQyxDQUFBO0lBQ0YsQ0FBQztJQUxlLGVBQVEsV0FLdkIsQ0FBQTtBQUNGLENBQUMsRUFwQ2EsTUFBTSxHQUFOLGNBQU0sS0FBTixjQUFNLFFBb0NuQjtBQUVELElBQWMsTUFBTSxDQUtuQjtBQUxELFdBQWMsTUFBTTtJQUFDLElBQUEsVUFBVSxDQUs5QjtJQUxvQixXQUFBLFVBQVUsRUFBQyxDQUFDO1FBQ3JCLHdCQUFhLEdBQUcsQ0FBQyxDQUFDO1FBQzdCO1lBQ0MsTUFBTSxDQUFDLHdCQUFhLEVBQUUsQ0FBQztRQUN4QixDQUFDO1FBRmUsY0FBRyxNQUVsQixDQUFBO0lBQ0YsQ0FBQyxFQUxvQixVQUFVLEdBQVYsaUJBQVUsS0FBVixpQkFBVSxRQUs5QjtBQUFELENBQUMsRUFMYSxNQUFNLEdBQU4sY0FBTSxLQUFOLGNBQU0sUUFLbkI7OztBQ3p4QkQsc0JBQXNCO0FBRXRCLGtCQUF5QixDQUFTLEVBQUUsUUFBa0I7SUFDckQsRUFBRSxDQUFDLENBQUMsQ0FBQyxRQUFRLENBQUM7UUFDYixNQUFNLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQy9CLElBQUksRUFBRSxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNyQixNQUFNLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDO0FBQ3RCLENBQUM7QUFMZSxnQkFBUSxXQUt2QixDQUFBO0FBRUQsY0FBcUIsQ0FBUyxFQUFFLFFBQWtCO0lBQ2pELEVBQUUsQ0FBQyxDQUFDLFFBQVEsQ0FBQztRQUNaLE1BQU0sQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDakMsSUFBSSxFQUFFLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNyQixJQUFJLEVBQUUsR0FBRyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hCLE1BQU0sQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUMsQ0FBQztBQUM5QixDQUFDO0FBTmUsWUFBSSxPQU1uQixDQUFBO0FBRUQsa0JBQXlCLENBQVMsRUFBRSxRQUFrQjtJQUNyRCxNQUFNLENBQUMsUUFBUSxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7QUFDekIsQ0FBQztBQUZlLGdCQUFRLFdBRXZCLENBQUE7QUFFRCxjQUFxQixDQUFTLEVBQUUsUUFBa0I7SUFDakQsTUFBTSxDQUFDLFFBQVEsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUNoQyxDQUFDO0FBRmUsWUFBSSxPQUVuQixDQUFBOzs7QUN6QkQ7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs2RkF3QjZGO0FBSTdGLElBQU8sT0FBTyxXQUFXLFdBQVcsQ0FBQyxDQUFDO0FBQ3RDLElBQU8sS0FBSyxXQUFXLFNBQVMsQ0FBQyxDQUFDO0FBQ2xDLElBQU8sTUFBTSxXQUFXLFVBQVUsQ0FBQyxDQUFDO0FBQ3BDLElBQU8sT0FBTyxXQUFXLFdBQVcsQ0FBQyxDQUFDO0FBQ3RDLElBQU8sU0FBUyxXQUFXLGFBQWEsQ0FBQyxDQUFDO0FBQzFDLElBQU8sTUFBTSxXQUFXLFVBQVUsQ0FBQyxDQUFDO0FBSXBDLElBQU8sUUFBUSxDQW9DZDtBQXBDRCxXQUFPLFFBQVEsRUFBQyxDQUFDO0lBS2hCLElBQUksV0FBVyxHQUFHLE9BQU8sTUFBTSxJQUFJLFdBQVcsSUFBSSxNQUFNLElBQUksTUFBTSxDQUFDLFVBQVUsQ0FBQyxDQUFDO0lBRS9FO1FBQ0ssTUFBTSxDQUFDLFVBQVUsQ0FBQyxHQUFHLFdBQVcsQ0FBQztRQUNqQyxNQUFNLENBQUMsUUFBUSxDQUFDO0lBQ3JCLENBQUM7SUFIZSxjQUFLLFFBR3BCLENBQUE7SUFvQlUsZUFBTSxHQUFHLE1BQU0sQ0FBQyxNQUFNLENBQUM7SUFDdkIsY0FBSyxHQUFHLEtBQUssQ0FBQyxLQUFLLENBQUM7SUFDcEIsZ0JBQU8sR0FBRyxPQUFPLENBQUMsT0FBTyxDQUFDO0lBQzFCLGdCQUFPLEdBQUcsT0FBTyxDQUFDLE9BQU8sQ0FBQztJQUMxQixlQUFNLEdBQUcsTUFBTSxDQUFDO0lBQ2hCLGtCQUFTLEdBQUcsU0FBUyxDQUFDO0FBQ2xDLENBQUMsRUFwQ00sUUFBUSxLQUFSLFFBQVEsUUFvQ2Q7QUFJRCxFQUFFLENBQUEsQ0FBQyxPQUFPLE1BQU0sSUFBSSxXQUFXLENBQUM7SUFDL0IsTUFBTSxDQUFDLFVBQVUsQ0FBQyxHQUFHLFFBQVEsQ0FBQztBQUgvQixpQkFBUyxRQUFRLENBQUM7OztBQ3pFbEI7OzRGQUU0RjtBQUU1RjtJQVFFLGlCQUFZLE9BQW9CLEVBQUUsT0FBYTtRQU4vQyxTQUFJLEdBQVEsRUFBRSxDQUFDO1FBQ2YsZUFBVSxHQUFHLE1BQU0sQ0FBQztRQUNwQixVQUFLLEdBQUcsSUFBSSxDQUFDO1FBS1gsT0FBTyxHQUFHLE9BQU8sSUFBSSxFQUFFLENBQUM7UUFDeEIsSUFBSSxDQUFDLE9BQU8sR0FBRyxPQUFPLENBQUM7UUFDdkIsSUFBSSxDQUFDLElBQUksR0FBRyxPQUFPLENBQUMsSUFBSSxJQUFJLEVBQUUsQ0FBQztRQUMvQixJQUFJLENBQUMsVUFBVSxHQUFHLE9BQU8sQ0FBQyxVQUFVLElBQUksTUFBTSxDQUFDO1FBQy9DLElBQUksQ0FBQyxLQUFLLEdBQUcsT0FBTyxDQUFDLEtBQUssSUFBSSxJQUFJLENBQUE7UUFDbEMsSUFBSSxDQUFDLElBQUksR0FBRyxPQUFPLENBQUMsSUFBSSxJQUFJLE9BQU8sQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDO0lBQ3pELENBQUM7SUFFRCxvQ0FBb0M7SUFDcEMsdUJBQUssR0FBTCxVQUFNLEdBQUcsRUFBRSxPQUFPO1FBQWxCLGlCQTRGQztRQTFGQyxJQUFJLEtBQUssR0FBRyxDQUFDLENBQUM7UUFDZCxJQUFJLFVBQVUsR0FBRyxDQUFDLEVBQUUsVUFBVSxHQUFHLENBQUMsQ0FBQztRQUNuQyxJQUFJLGNBQWMsR0FBRyxLQUFLLENBQUM7UUFDM0IsSUFBSSxLQUFLLEVBQUUsTUFBTSxFQUFFLE1BQU0sRUFBRSxXQUFXLENBQUM7UUFFdkMsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDO1FBRXZCLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7WUFDWixFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztnQkFDcEIsNEJBQTRCO2dCQUM1Qiw4Q0FBOEM7Z0JBQzlDLGlCQUFpQixDQUFDO29CQUNoQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDO3dCQUFDLENBQUM7b0JBQ3RHLE1BQU0sQ0FBQyxDQUFDLENBQUM7Z0JBQ1gsQ0FBQztnQkFBQSxDQUFDO1lBQ0osQ0FBQztZQUNELEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxVQUFVLENBQUM7Z0JBQ3JCLElBQUksQ0FBQyxVQUFVLEdBQUcsT0FBTyxDQUFDLFVBQVUsQ0FBQztZQUN2QyxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsS0FBSyxDQUFDO2dCQUNoQixJQUFJLENBQUMsS0FBSyxHQUFHLE9BQU8sQ0FBQyxLQUFLLENBQUM7WUFDN0IsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQztnQkFDZixJQUFJLENBQUMsSUFBSSxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUM7WUFDM0IsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQztnQkFDZixJQUFJLENBQUMsSUFBSSxHQUFHLE9BQU8sQ0FBQyxJQUFJLENBQUM7WUFDM0IsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLFFBQVEsQ0FBQztnQkFDbkIsSUFBSSxDQUFDLFFBQVEsR0FBRyxPQUFPLENBQUMsUUFBUSxDQUFDO1lBQ25DLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDO2dCQUN0QiwyREFBMkQ7Z0JBQzNELE9BQU8sQ0FBQyxHQUFHLENBQUMsK0NBQStDLENBQUMsQ0FBQTtnQkFDNUQsSUFBSSxDQUFDLFFBQVEsR0FBRyxPQUFPLENBQUMsU0FBUyxDQUFDO1lBQ3BDLENBQUM7UUFDSCxDQUFDO1FBRUQsV0FBVyxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUM7UUFDeEIsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQzdCLFVBQVUsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxVQUFVLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUM5RCxDQUFDO1FBR0QsT0FBTyxDQUFDLGNBQWMsSUFBSSxVQUFVLEdBQUcsSUFBSSxDQUFDLFVBQVUsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLEtBQUssRUFBRSxDQUFDO1lBQzdFLEtBQUssR0FBRyxDQUFDLENBQUM7WUFFVixFQUFFLENBQUMsQ0FBQyxVQUFVLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDbkIsSUFBSSxhQUFhLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxVQUFVLEdBQUcsVUFBVSxDQUFDLENBQUM7Z0JBQ3hELFdBQVcsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDO1lBQ3pDLENBQUM7WUFFRCxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsS0FBSyxJQUFJLEdBQUcsQ0FBQyxDQUFDLENBQUM7Z0JBQ3RCLEtBQUssR0FBRyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsS0FBSyxDQUFDO2dCQUN6QixNQUFNLEdBQUcsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sQ0FBQztnQkFFM0IsTUFBTSxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUMsUUFBUSxDQUFDLEtBQUssQ0FBQyxDQUFDO2dCQUN0QyxJQUFJLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxXQUFXLEVBQUUsTUFBTSxDQUFDLENBQUM7Z0JBRTVDLEtBQUssSUFBSSxJQUFJLENBQUMsSUFBSSxDQUFDLE1BQU0sRUFBRSxNQUFNLENBQUMsQ0FBQztZQUNyQyxDQUFDO1lBRUQsY0FBYztZQUNkLFVBQVUsRUFBRSxDQUFDO1lBQ2IsS0FBSyxJQUFJLEdBQUcsQ0FBQyxNQUFNLENBQUM7WUFFcEIsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztnQkFDWixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsUUFBUSxJQUFJLElBQUksQ0FBQyxRQUFRLENBQUMsS0FBSyxJQUFJLFVBQVUsR0FBRyxJQUFJLENBQUMsUUFBUSxDQUFDLEtBQUssSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO29CQUVsRixjQUFjLEdBQUcsSUFBSSxDQUFDLFFBQVEsQ0FBQyxFQUFFLENBQUM7d0JBQ2hDLEtBQUssRUFBRSxLQUFLO3dCQUNaLFVBQVUsRUFBRSxVQUFVO3dCQUN0QixJQUFJLEVBQUUsV0FBVztxQkFDbEIsQ0FBQyxDQUFDO29CQUVILHFCQUFxQixDQUFDO3dCQUNwQixLQUFJLENBQUMsVUFBVSxJQUFJLFVBQVUsQ0FBQzt3QkFDOUIsS0FBSSxDQUFDLEtBQUssQ0FBQyxHQUFHLEVBQUUsT0FBTyxDQUFDLENBQUM7b0JBQzNCLENBQUMsQ0FBQyxDQUFDO29CQUNILE1BQU0sQ0FBQztnQkFDVCxDQUFDO2dCQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsR0FBRyxJQUFJLFVBQVUsR0FBRyxPQUFPLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQ3hELE9BQU8sQ0FBQyxHQUFHLENBQUMsWUFBWSxFQUFFLFVBQVUsRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFFLE1BQU0sRUFBRSxXQUFXLENBQUMsQ0FBQztnQkFDN0UsQ0FBQztnQkFBQSxDQUFDO2dCQUNGLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUM7b0JBQ2xCLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQztZQUNqQixDQUFDO1FBQ0gsQ0FBQztRQUVELElBQUksT0FBTyxHQUFHO1lBQ1osS0FBSyxFQUFFLEtBQUs7WUFDWixVQUFVLEVBQUUsVUFBVTtZQUN0QixJQUFJLEVBQUUsSUFBSSxDQUFDLEdBQUcsRUFBRSxHQUFHLEtBQUs7U0FDekIsQ0FBQTtRQUVELE1BQU0sQ0FBQyxPQUFPLENBQUM7SUFDakIsQ0FBQztJQUVELHNEQUFzRDtJQUN0RCw2QkFBVyxHQUFYLFVBQVksR0FBRyxFQUFFLFFBQVEsRUFBRSxPQUFPO1FBRWhDLElBQUksSUFBSSxHQUFHLElBQUksQ0FBQztRQUNoQixJQUFJLEtBQUssR0FBRyxDQUFDLENBQUM7UUFDZCxJQUFJLFVBQVUsR0FBRyxDQUFDLEVBQUUsVUFBVSxHQUFHLENBQUMsQ0FBQztRQUNuQyxJQUFJLEtBQUssRUFBRSxNQUFNLEVBQUUsTUFBTSxFQUFFLFdBQVcsQ0FBQztRQUN2QyxJQUFJLE1BQU0sR0FBRyxHQUFHLENBQUMsTUFBTSxDQUFDO1FBQ3hCLElBQUksY0FBYyxHQUFHLEtBQUssQ0FBQztRQUUzQixJQUFJLEtBQUssR0FBRyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUM7UUFFdkIsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztZQUNaLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDO2dCQUNwQiw0QkFBNEI7Z0JBQzVCLDhDQUE4QztnQkFDOUMsaUJBQWlCLENBQUM7b0JBQ2hCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsTUFBTSxFQUFFO3dCQUMxRCxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQzt3QkFBQyxDQUFDO29CQUN6QyxNQUFNLENBQUMsQ0FBQyxDQUFDO2dCQUNYLENBQUM7Z0JBQUEsQ0FBQztZQUNKLENBQUM7WUFDRCxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsVUFBVSxDQUFDO2dCQUNyQixJQUFJLENBQUMsVUFBVSxHQUFHLE9BQU8sQ0FBQyxVQUFVLENBQUM7WUFDdkMsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQztnQkFDaEIsSUFBSSxDQUFDLEtBQUssR0FBRyxPQUFPLENBQUMsS0FBSyxDQUFDO1lBQzdCLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUM7Z0JBQ2YsSUFBSSxDQUFDLElBQUksR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDO1lBQzNCLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUM7Z0JBQ2YsSUFBSSxDQUFDLElBQUksR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDO1lBQzNCLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxRQUFRLENBQUM7Z0JBQ25CLElBQUksQ0FBQyxRQUFRLEdBQUcsT0FBTyxDQUFDLFFBQVEsQ0FBQztZQUNuQyxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsU0FBUyxDQUFDO2dCQUNwQiwyREFBMkQ7Z0JBQzNELE9BQU8sQ0FBQyxHQUFHLENBQUMsK0NBQStDLENBQUMsQ0FBQTtZQUM5RCxJQUFJLENBQUMsUUFBUSxHQUFHLE9BQU8sQ0FBQyxTQUFTLENBQUM7UUFDcEMsQ0FBQztRQUVELHdCQUF3QjtRQUN4QixXQUFXLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQztRQUN4QixFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDN0IsVUFBVSxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLFVBQVUsR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQzlELENBQUM7UUFFRCxrQkFBa0I7UUFDbEIsSUFBSSxNQUFNLEdBQUcsSUFBSSxDQUFDLE9BQU8sQ0FBQyxNQUFNLEVBQUUsQ0FBQztRQUVuQyx1QkFBdUI7UUFDdkIsd0JBQXdCLEtBQUs7WUFDM0IsTUFBTSxDQUFDLFdBQVcsQ0FBQztnQkFDakIsTUFBTSxFQUFFLFVBQVU7Z0JBQ2xCLEtBQUssRUFBRSxLQUFLO2dCQUNaLFlBQVksRUFBRSxJQUFJLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxNQUFNO2FBQzVDLEVBQUUsQ0FBQyxJQUFJLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQztRQUM3QyxDQUFDO1FBRUQsNEJBQTRCO1FBQzVCLHlCQUF5QixNQUFNO1lBQzdCLEVBQUUsQ0FBQyxDQUFDLFVBQVUsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNuQixJQUFJLGFBQWEsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLFVBQVUsR0FBRyxVQUFVLENBQUMsQ0FBQztnQkFDeEQsV0FBVyxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUM7WUFDekMsQ0FBQztZQUNELE1BQU0sQ0FBQyxXQUFXLENBQUM7Z0JBQ2pCLE1BQU0sRUFBRSxXQUFXO2dCQUNuQixNQUFNLEVBQUUsTUFBTTtnQkFDZCxJQUFJLEVBQUUsV0FBVztnQkFDakIsWUFBWSxFQUFFLElBQUksQ0FBQyxPQUFPLENBQUMsU0FBUyxDQUFDLE1BQU07YUFDNUMsRUFBRSxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsU0FBUyxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDO1FBQzdDLENBQUM7UUFFRCxtQkFBbUI7UUFDbkIsTUFBTSxDQUFDLFNBQVMsR0FBRyxVQUFTLENBQUM7WUFDM0IsaURBQWlEO1lBQ2pELElBQUksQ0FBQyxPQUFPLENBQUMsU0FBUyxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLFlBQVksQ0FBQyxDQUFDO1lBRXRELEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsTUFBTSxJQUFJLFdBQVcsQ0FBQyxDQUFDLENBQUM7Z0JBQ2pDLEVBQUUsQ0FBQyxDQUFDLEtBQUssSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDO29CQUNwQixLQUFLLEdBQUcsQ0FBQyxDQUFDO29CQUNWLFVBQVUsRUFBRSxDQUFDO29CQUNiLEtBQUssSUFBSSxHQUFHLENBQUMsTUFBTSxDQUFDO29CQUVwQixNQUFNO29CQUNOLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7d0JBQ1osRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFFBQVEsSUFBSSxJQUFJLENBQUMsUUFBUSxDQUFDLEtBQUssSUFBSSxVQUFVLEdBQUcsSUFBSSxDQUFDLFFBQVEsQ0FBQyxLQUFLLElBQUksQ0FBQyxDQUFDOzRCQUNoRixjQUFjLEdBQUcsSUFBSSxDQUFDLFFBQVEsQ0FBQyxFQUFFLENBQUM7Z0NBQ2hDLEtBQUssRUFBRSxLQUFLO2dDQUNaLFVBQVUsRUFBRSxVQUFVOzZCQUN2QixDQUFDLENBQUM7d0JBQ0wsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxHQUFHLElBQUksVUFBVSxHQUFHLE9BQU8sQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQzs0QkFDdEQsT0FBTyxDQUFDLEdBQUcsQ0FBQyxZQUFZLEVBQUUsVUFBVSxFQUFFLE9BQU8sRUFBRSxLQUFLLENBQUMsQ0FBQzt3QkFDeEQsQ0FBQzt3QkFBQSxDQUFDO3dCQUNGLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUM7NEJBQ2xCLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQztvQkFDakIsQ0FBQztvQkFFRCxFQUFFLENBQUMsQ0FBQyxDQUFDLGNBQWMsSUFBSSxVQUFVLEdBQUcsSUFBSSxDQUFDLFVBQVUsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7d0JBQzFFLGNBQWMsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7b0JBQ25DLENBQUM7b0JBQUMsSUFBSSxDQUFDLENBQUM7d0JBQ04sV0FBVzt3QkFDWCxRQUFRLENBQUM7NEJBQ1AsS0FBSyxFQUFFLEtBQUs7NEJBQ1osVUFBVSxFQUFFLFVBQVU7NEJBQ3RCLElBQUksRUFBRSxJQUFJLENBQUMsR0FBRyxFQUFFLEdBQUcsS0FBSzt5QkFDekIsQ0FBQyxDQUFBO29CQUNKLENBQUM7b0JBQ0QsS0FBSyxHQUFHLENBQUMsQ0FBQztnQkFDWixDQUFDO2dCQUFDLElBQUksQ0FBQyxDQUFDO29CQUNOLGNBQWMsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7Z0JBQ25DLENBQUM7WUFDSCxDQUFDO1lBRUQsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxNQUFNLElBQUksVUFBVSxDQUFDLENBQUMsQ0FBQztnQkFDaEMsS0FBSyxJQUFJLElBQUksQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO2dCQUNyRCxlQUFlLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDO2dCQUNuQyxLQUFLLEVBQUUsQ0FBQztZQUNWLENBQUM7UUFDSCxDQUFDLENBQUE7UUFFRCxVQUFVO1FBQ1YsSUFBSSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2QsSUFBSSxVQUFVLEdBQUcsQ0FBQyxDQUFDO1FBQ25CLGNBQWMsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDbkMsQ0FBQztJQUVELCtCQUErQjtJQUMvQixxQkFBRyxHQUFILFVBQUksT0FBTztRQUVULEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTSxFQUFFLElBQUksQ0FBQyxJQUFJLElBQUksQ0FBQyxPQUFPLENBQUMsT0FBTyxFQUFFLElBQUksQ0FBQyxDQUFDO1lBQzVELE1BQU0sa0RBQWtELENBQUM7UUFFM0QsSUFBSSxRQUFRLEdBQUc7WUFDYixVQUFVLEVBQUUsTUFBTTtZQUNsQixHQUFHLEVBQUUsS0FBSztZQUNWLE9BQU8sRUFBRSxJQUFJO1lBQ2IsSUFBSSxFQUFFLE9BQU8sQ0FBQyxJQUFJLENBQUMsR0FBRztTQUN2QixDQUFBO1FBRUQsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDO1lBQ1YsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxPQUFPLENBQUM7Z0JBQ3BCLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFN0IsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztnQkFDakIsS0FBSyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFDYixNQUFNLEVBQUUsQ0FBQyxDQUFDLENBQUM7YUFDWixFQUFFO2dCQUNDLEtBQUssRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBQ2IsTUFBTSxFQUFFLENBQUMsQ0FBQyxDQUFDO2FBQ1osRUFBRTtnQkFDRCxLQUFLLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDO2dCQUNiLE1BQU0sRUFBRSxDQUFDLENBQUMsQ0FBQzthQUNaLEVBQUU7Z0JBQ0QsS0FBSyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFDYixNQUFNLEVBQUUsQ0FBQyxDQUFDLENBQUM7YUFDWixDQUFDLEVBQUUsUUFBUSxDQUFDLENBQUM7SUFDbEIsQ0FBQztJQUVELCtEQUErRDtJQUMvRCxxQkFBRyxHQUFILFVBQUksT0FBTztRQUFYLGlCQXdJQztRQXZJQyxPQUFPLEdBQUcsT0FBTyxJQUFJLEVBQUUsQ0FBQztRQUV4QixJQUFJLE9BQU8sR0FBRyxPQUFPLENBQUMsT0FBTyxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDOUMsSUFBSSxXQUFXLEdBQUcsT0FBTyxDQUFDLFdBQVcsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3RELElBQUksT0FBTyxHQUFHLE9BQU8sQ0FBQyxPQUFPLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDeEMsSUFBSSxNQUFNLEdBQUcsT0FBTyxDQUFDLE1BQU0sSUFBSSxFQUFFLENBQUM7UUFDbEMsSUFBSSxTQUFTLEdBQUcsT0FBTyxDQUFDLE9BQU8sSUFBSSxJQUFJLENBQUM7UUFDeEMsSUFBSSxVQUFVLEdBQUcsT0FBTyxDQUFDLFVBQVUsSUFBSSxNQUFNLENBQUM7UUFDOUMsSUFBSSxJQUFJLEdBQUcsT0FBTyxDQUFDLElBQUksSUFBSSxFQUFFLENBQUM7UUFDOUIsSUFBSSxHQUFHLEdBQUcsT0FBTyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUM7UUFDM0IsSUFBSSxRQUFRLEdBQUcsT0FBTyxDQUFDLFFBQVEsSUFBSSxFQUFFLENBQUM7UUFDdEMsSUFBSSxPQUFPLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQztRQUNWLElBQUksT0FBTyxHQUFHLENBQUMsQ0FBQztRQUNoQixJQUFJLEtBQUssR0FBRyxDQUFDLEdBQUcsT0FBTyxHQUFHLENBQUMsR0FBRyxPQUFPLEdBQUcsQ0FBQyxFQUN2QyxLQUFLLEdBQUcsQ0FBQyxFQUNULE9BQU8sR0FBRyxPQUFPLENBQUMsTUFBTSxHQUFHLFdBQVcsQ0FBQyxNQUFNLEdBQUcsT0FBTyxDQUFDLE1BQU0sQ0FBQztRQUVqRSxJQUFJLFFBQVEsR0FBRyxVQUFTLEtBQUssRUFBRSxLQUFLO1lBQ2xDLElBQUksTUFBTSxHQUFHLElBQUksQ0FBQyxNQUFNLEVBQUUsR0FBRyxLQUFLLEdBQUcsQ0FBQyxDQUFDO1lBQ3ZDLElBQUksSUFBSSxHQUFHLEtBQUssQ0FBQztZQUNqQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQztnQkFDbEIsRUFBRSxDQUFDLENBQUMsTUFBTSxJQUFJLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztvQkFDckIsSUFBSSxHQUFHLElBQUksQ0FBQztZQUNoQixNQUFNLENBQUMsSUFBSSxHQUFHLFFBQVEsQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLEdBQUcsTUFBTSxDQUFDO1FBQ2hELENBQUMsQ0FBQTtRQUVELElBQUksS0FBSyxHQUFHLFVBQVMsVUFBVSxFQUFFLE1BQU07WUFDckMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxVQUFVLENBQUM7Z0JBQ3ZCLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO29CQUN6QyxNQUFNLENBQUMsS0FBSyxDQUFDO1lBQ2pCLE1BQU0sQ0FBQyxJQUFJLENBQUM7UUFDZCxDQUFDLENBQUE7UUFFRCxJQUFJLEtBQUssR0FBRyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUM7UUFFdkIsT0FBTyxLQUFLLEdBQUcsVUFBVSxJQUFJLENBQUMsT0FBTyxHQUFHLFNBQVMsSUFBSSxLQUFLLEdBQUcsSUFBSSxJQUFJLENBQUMsQ0FBQyxFQUFFLENBQUM7WUFDeEUsb0JBQW9CO1lBQ3BCLElBQUksUUFBUSxHQUFHLEVBQUUsRUFDZixjQUFjLEdBQUcsTUFBTSxHQUFHLE9BQU8sQ0FBQyxNQUFNLENBQUM7WUFDM0MsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsY0FBYyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7Z0JBQ3BDLElBQUksR0FBRyxHQUFHLElBQUksQ0FBQyxNQUFNLEVBQUUsR0FBRyxXQUFXLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQztnQkFDakQsUUFBUSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztZQUNsQyxDQUFDO1lBQ0QsSUFBSSxPQUFPLEdBQUcsRUFBRSxFQUNkLFNBQVMsR0FBRyxFQUFFLENBQUM7WUFDakIsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO2dCQUNwQyxPQUFPLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxNQUFNLEVBQUUsR0FBRyxPQUFPLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQyxDQUFDO2dCQUNqRCxTQUFTLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxjQUFjLEVBQUUsU0FBUyxDQUFDLENBQUMsQ0FBQztZQUN0RCxDQUFDO1lBQ0QsU0FBUyxHQUFHLFNBQVMsQ0FBQyxJQUFJLEVBQUUsQ0FBQztZQUM3QixHQUFHLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxPQUFPLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7Z0JBQ3BDLFFBQVEsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQzdDLFFBQVEsQ0FBQyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDNUIsQ0FBQztZQUVELGdCQUFnQjtZQUNoQixJQUFJLGtCQUFrQixDQUFDO1lBQ3ZCLElBQUksY0FBYyxHQUFHLGtCQUFrQixHQUFHLENBQUMsQ0FBQztZQUM1QyxLQUFLLEdBQUcsQ0FBQyxDQUFDO1lBQ1YsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7Z0JBQzVCLCtCQUErQjtnQkFDL0IsSUFBSSxLQUFLLEdBQUcsRUFBRSxDQUFDO2dCQUNmLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxDQUFDLEVBQUU7b0JBQzFCLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUM7Z0JBQ2YsS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQztnQkFFdkIseUJBQXlCO2dCQUN6QixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUM7Z0JBQ2hCLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFO29CQUNqQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDO2dCQUVoQixFQUFFLENBQUMsQ0FBQyxDQUFDLElBQUksY0FBYyxDQUFDLENBQUMsQ0FBQztvQkFDeEIsSUFBSSxLQUFLLEdBQUcsQ0FBQyxHQUFHLGNBQWMsQ0FBQztvQkFDL0IsTUFBTSxDQUFDLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQztnQkFDN0IsQ0FBQztnQkFFRCxlQUFlO2dCQUNmLElBQUksVUFBVSxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUMsUUFBUSxDQUFDLEtBQUssQ0FBQyxDQUFDO2dCQUU5QyxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsVUFBVSxFQUFFLE1BQU0sQ0FBQyxDQUFDO29CQUM1QixFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsY0FBYyxDQUFDO3dCQUNyQixrQkFBa0IsRUFBRSxDQUFDO29CQUN2QixJQUFJO3dCQUNGLGNBQWMsRUFBRSxDQUFDO2dCQUNyQixJQUFJLENBQUMsQ0FBQztvQkFDSixJQUFJLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxJQUFJLEVBQUUsTUFBTSxDQUFDLENBQUM7Z0JBQ3ZDLENBQUM7Z0JBRUQsSUFBSSxLQUFLLEdBQUcsQ0FBQyxDQUFDO2dCQUNkLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksVUFBVSxDQUFDO29CQUN2QixLQUFLLElBQUksSUFBSSxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO2dCQUNsRCxLQUFLLElBQUksS0FBSyxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUMsT0FBTyxFQUFFLENBQUM7Z0JBRXhDLEVBQUUsQ0FBQyxDQUFDLGtCQUFrQixHQUFHLGNBQWMsSUFBSSxNQUFNLENBQUM7b0JBQ2hELE9BQU8sRUFBRSxDQUFDO1lBQ2QsQ0FBQztZQUVELGtCQUFrQjtZQUNsQixFQUFFLENBQUMsQ0FBQyxLQUFLLEdBQUcsSUFBSSxJQUFJLENBQUMsQ0FBQztnQkFDcEIsT0FBTyxHQUFHLENBQUMsQ0FBQztZQUNkLEtBQUssRUFBRSxDQUFDO1lBQ1IsSUFBSSxXQUFXLEdBQUcsS0FBSyxHQUFHLElBQUksQ0FBQztZQUMvQixXQUFXLEdBQUcsV0FBVyxJQUFJLENBQUMsR0FBRyxJQUFJLEdBQUcsV0FBVyxDQUFDO1lBQ3BELE9BQU8sR0FBRyxPQUFPLEdBQUcsV0FBVyxDQUFDO1lBQ2hDLEtBQUssSUFBSSxNQUFNLENBQUM7WUFFaEIsTUFBTTtZQUNOLEVBQUUsQ0FBQyxDQUFDLEdBQUcsSUFBSSxLQUFLLEdBQUcsR0FBRyxJQUFJLENBQUMsQ0FBQztnQkFDMUIsT0FBTyxDQUFDLEdBQUcsQ0FBQyxhQUFhLEVBQUUsS0FBSyxFQUFFLFdBQVcsRUFBRSxPQUFPLEVBQUUsV0FBVyxFQUNqRSxPQUFPLEVBQUUsUUFBUSxFQUFFLElBQUksQ0FBQyxHQUFHLEVBQUUsR0FBRyxLQUFLLEVBQUUsU0FBUyxFQUFFLEtBQUssQ0FBQyxDQUFDO1lBQzdELEVBQUUsQ0FBQyxDQUFDLFFBQVEsQ0FBQyxFQUFFLElBQUksUUFBUSxDQUFDLEtBQUssSUFBSSxLQUFLLEdBQUcsUUFBUSxDQUFDLEtBQUssSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNqRSxRQUFRLENBQUMsRUFBRSxDQUFDO29CQUNWLFVBQVUsRUFBRSxLQUFLO29CQUNqQixPQUFPLEVBQUUsT0FBTztvQkFDaEIsS0FBSyxFQUFFLEtBQUs7b0JBQ1osSUFBSSxFQUFFLElBQUksQ0FBQyxHQUFHLEVBQUUsR0FBRyxLQUFLO29CQUN4QixPQUFPLEVBQUUsT0FBTztpQkFDakIsQ0FBQyxDQUFDO2dCQUVILHFCQUFxQixDQUFDO29CQUNwQixLQUFJLENBQUMsVUFBVSxJQUFJLEtBQUssQ0FBQztvQkFDekIsS0FBSSxDQUFDLEdBQUcsQ0FBQyxPQUFPLENBQUMsQ0FBQztnQkFDcEIsQ0FBQyxDQUFDLENBQUM7Z0JBRUgsTUFBTSxDQUFDO1lBQ1QsQ0FBQztRQUNILENBQUM7UUFFRCxNQUFNLENBQUM7WUFDTCxVQUFVLEVBQUUsS0FBSztZQUNqQixPQUFPLEVBQUUsT0FBTztZQUNoQixLQUFLLEVBQUUsS0FBSztZQUNaLElBQUksRUFBRSxJQUFJLENBQUMsR0FBRyxFQUFFLEdBQUcsS0FBSztTQUN6QixDQUFBO0lBQ0gsQ0FBQztJQUVELHNEQUFzRDtJQUN0RCxxQkFBRyxHQUFILFVBQUksT0FBTztRQUVULE9BQU8sR0FBRyxPQUFPLElBQUksRUFBRSxDQUFDO1FBQ3hCLElBQUksVUFBVSxHQUFHLE9BQU8sQ0FBQyxVQUFVLElBQUksTUFBTSxDQUFDO1FBQzlDLElBQUksU0FBUyxHQUFHLE9BQU8sQ0FBQyxLQUFLLElBQUksR0FBRyxDQUFDO1FBQ3JDLElBQUksSUFBSSxHQUFHLE9BQU8sQ0FBQyxJQUFJLElBQUksRUFBRSxDQUFDO1FBQzlCLElBQUksR0FBRyxHQUFHLE9BQU8sQ0FBQyxHQUFHLElBQUksR0FBRyxDQUFDO1FBRTdCLGNBQWM7UUFDZCxJQUFJLElBQUksR0FBRztZQUNULElBQUksQ0FBQyxLQUFLLEdBQUcsRUFBRSxDQUFDO1FBQ2xCLENBQUMsQ0FBQTtRQUNELElBQUksQ0FBQyxTQUFTLEdBQUc7WUFDZixPQUFPLEVBQUUsVUFBUyxJQUFJLEVBQUUsS0FBSztnQkFDM0IsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUM7b0JBQ2QsSUFBSSxFQUFFLElBQUk7b0JBQ1YsS0FBSyxFQUFFLEtBQUs7aUJBQ2IsQ0FBQyxDQUFDO2dCQUNILE1BQU0sQ0FBQyxJQUFJLENBQUM7WUFDZCxDQUFDO1lBQ0QsR0FBRyxFQUFFO2dCQUNILEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsTUFBTSxJQUFJLENBQUMsQ0FBQztvQkFDekIsTUFBTSxDQUFDLEtBQUssQ0FBQztnQkFDZixJQUFJLEtBQUssR0FBRyxJQUFJLENBQUMsTUFBTSxFQUFFLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDO2dCQUNsRCxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQztZQUMzQixDQUFDO1lBQ0QsSUFBSSxFQUFFLFVBQVMsS0FBSztnQkFDbEIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDO29CQUN2QixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssSUFBSSxLQUFLLENBQUM7d0JBQy9CLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUN6QixNQUFNLENBQUMsS0FBSyxDQUFDO1lBQ2YsQ0FBQztTQUNGLENBQUE7UUFFRCxJQUFJLFlBQVksR0FBRztZQUVqQix3QkFBd0I7WUFDeEIsSUFBSSxNQUFNLEdBQUcsSUFBSSxJQUFJLEVBQUUsQ0FBQztZQUN4QixJQUFJLEVBQUUsR0FBRyxDQUFDLElBQUksSUFBSSxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1lBQzNDLElBQUksRUFBRSxHQUFHLENBQUMsSUFBSSxJQUFJLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDdkMsSUFBSSxFQUFFLEdBQUcsQ0FBQyxJQUFJLElBQUksRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1lBQ3hELElBQUksRUFBRSxHQUFHLENBQUMsSUFBSSxJQUFJLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUE7WUFDdEMsRUFBRSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDcEIsSUFBSSxFQUFFLEdBQUcsQ0FBQyxJQUFJLElBQUksRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQTtZQUN0QyxFQUFFLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQztZQUNwQixFQUFFLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQTtZQUNuQixJQUFJLEVBQUUsR0FBRyxDQUFDLElBQUksSUFBSSxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDeEQsSUFBSSxLQUFLLEdBQUcsQ0FBQyxJQUFJLElBQUksRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQTtZQUV6QyxNQUFNLENBQUM7Z0JBQ0wsS0FBSyxFQUFFLEtBQUs7Z0JBQ1osTUFBTSxFQUFFLE1BQU07YUFDZixDQUFBO1FBQ0gsQ0FBQyxDQUFBO1FBRUQsaUNBQWlDO1FBQ2pDLElBQUksbUJBQW1CLEdBQUc7WUFDeEIsSUFBSSxNQUFNLEdBQUcsWUFBWSxFQUFFLENBQUM7WUFDNUIsSUFBSSxNQUFNLEdBQUcsWUFBWSxFQUFFLENBQUM7WUFFNUIsSUFBSSxNQUFNLEdBQUcsSUFBSSxJQUFJLEVBQUUsQ0FBQztZQUN4QixJQUFJLEVBQUUsR0FBRyxDQUFDLElBQUksSUFBSSxDQUFDLENBQUMsT0FBTyxDQUFDLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQztZQUN6QyxNQUFNLENBQUMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDL0IsTUFBTSxDQUFDLE1BQU0sQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1lBQy9CLElBQUksRUFBRSxHQUFHLENBQUMsSUFBSSxJQUFJLENBQUMsQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLEtBQUssRUFBRSxHQUFHLENBQUMsQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLEtBQUssRUFDakUsR0FBRyxDQUFDLENBQUM7WUFDUCxJQUFJLEtBQUssR0FBRyxDQUFDLElBQUksSUFBSSxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQztZQUV4QyxNQUFNLENBQUM7Z0JBQ0wsS0FBSyxFQUFFLEtBQUs7Z0JBQ1osTUFBTSxFQUFFLE1BQU07YUFDZixDQUFBO1FBRUgsQ0FBQyxDQUFBO1FBRUQsMkJBQTJCO1FBQzNCLElBQUksUUFBUSxHQUFHO1lBQ2IsSUFBSSxJQUFJLEdBQUcsbUJBQW1CLEVBQUUsQ0FBQyxLQUFLLENBQUM7WUFDdkMsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDO1lBQ3RCLElBQUksR0FBRyxHQUFHLEVBQUUsQ0FBQztZQUNiLE9BQU8sSUFBSSxFQUFFLENBQUM7Z0JBQ1osR0FBRyxJQUFJLElBQUksQ0FBQyxLQUFLLENBQUM7Z0JBQ2xCLElBQUksR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDO1lBQ3pCLENBQUM7WUFDRCxNQUFNLENBQUMsR0FBRyxDQUFDO1FBQ2IsQ0FBQyxDQUFBO1FBRUQsb0RBQW9EO1FBQ3BELElBQUksSUFBSSxHQUFHLFVBQVMsR0FBRztZQUNyQixJQUFJLElBQUksR0FBRyxtQkFBbUIsRUFBRSxDQUFDLEtBQUssQ0FBQztZQUN2QyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUM7WUFDVixJQUFJLEVBQUUsR0FBRyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3ZCLE9BQU8sQ0FBQyxHQUFHLEdBQUcsQ0FBQyxNQUFNLEVBQUUsQ0FBQztnQkFDdEIsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQztnQkFDekIsRUFBRSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUM7b0JBQ1IsTUFBTSxDQUFDLEtBQUssQ0FBQztnQkFDZixJQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQztnQkFDakIsRUFBRSxHQUFHLEdBQUcsQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztZQUN2QixDQUFDO1lBQ0QsTUFBTSxDQUFDLElBQUksQ0FBQztRQUNkLENBQUMsQ0FBQTtRQUVELDZEQUE2RDtRQUM3RCxJQUFJLFNBQVMsR0FBRyxVQUFTLE1BQU0sRUFBRSxNQUFNO1lBQ3JDLElBQUksSUFBSSxHQUFHLENBQUMsQ0FBQztZQUNiLElBQUksRUFBRSxHQUFHLENBQUMsQ0FBQyxDQUFDO1lBQ1osSUFBSSxJQUFJLEdBQUcsQ0FBQyxDQUFDO1lBQ2IsSUFBSSxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUM7WUFDWixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLE1BQU0sQ0FBQyxDQUFDLENBQUM7Z0JBQ3JCLEVBQUUsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDO29CQUNyQixJQUFJLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO29CQUNqQixFQUFFLEdBQUcsQ0FBQyxDQUFDO2dCQUNULENBQUM7Z0JBQ0QsRUFBRSxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUM7b0JBQ3JCLElBQUksR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQ2pCLEVBQUUsR0FBRyxDQUFDLENBQUM7Z0JBQ1QsQ0FBQztZQUNILENBQUM7WUFFRCxNQUFNLENBQUMsRUFBRSxJQUFJLEVBQUUsQ0FBQztRQUNsQixDQUFDLENBQUE7UUFFRCxJQUFJLFNBQVMsR0FBRyxDQUFDLENBQUM7UUFDbEIsSUFBSSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2QsSUFBSSxLQUFLLEdBQUc7WUFDVixHQUFHLEVBQUUsQ0FBQztZQUNOLEdBQUcsRUFBRSxDQUFDO1lBQ04sR0FBRyxFQUFFLENBQUM7WUFDTixHQUFHLEVBQUUsQ0FBQztZQUNOLEdBQUcsRUFBRSxDQUFDO1lBQ04sR0FBRyxFQUFFLENBQUM7U0FDUCxDQUFBO1FBRUQsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDO1FBQ3ZCLE9BQU8sU0FBUyxHQUFHLFVBQVUsSUFBSSxLQUFLLEdBQUcsU0FBUyxFQUFFLENBQUM7WUFDbkQsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDO1lBQ1YsS0FBSyxHQUFHLENBQUMsQ0FBQztZQUVWLHdCQUF3QjtZQUN4QixJQUFJLFFBQVEsR0FBRyxRQUFRLEVBQUUsQ0FBQztZQUUxQixRQUFRO1lBQ1IsSUFBSSxJQUFJLEdBQUcsUUFBUSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUM5QixTQUFTO1lBQ1QsSUFBSSxPQUFPLEdBQUcsUUFBUSxDQUFDLE1BQU0sQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7WUFFckMsUUFBUTtZQUNSLE9BQU8sQ0FBQyxHQUFHLFFBQVEsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxFQUFFLENBQUM7Z0JBQy9CLElBQUksS0FBSyxHQUFHLEVBQUUsQ0FBQztnQkFDZixJQUFJLE1BQU0sR0FBRyxFQUFFLENBQUM7Z0JBQ2hCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztvQkFDM0IsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQztvQkFDYixNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDO2dCQUNoQixDQUFDO2dCQUNELEtBQUssQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUM7Z0JBQ3ZCLE1BQU0sQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUM7Z0JBRTNCLElBQUksTUFBTSxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUMsUUFBUSxDQUFDLEtBQUssQ0FBQyxDQUFDO2dCQUUxQyxFQUFFLENBQUMsQ0FBQyxTQUFTLENBQUMsTUFBTSxFQUFFLE1BQU0sQ0FBQyxDQUFDO29CQUM1QixJQUFJLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxJQUFJLEVBQUUsTUFBTSxDQUFDLENBQUM7Z0JBRXZDLElBQUksR0FBRyxRQUFRLENBQUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7Z0JBQzVCLE9BQU8sR0FBRyxRQUFRLENBQUMsTUFBTSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztnQkFFakMsSUFBSSxLQUFLLEdBQUcsQ0FBQyxDQUFDO2dCQUNkLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksTUFBTSxDQUFDO29CQUNuQixLQUFLLElBQUksSUFBSSxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFBO2dCQUM3QyxLQUFLLElBQUksTUFBTSxDQUFDLE1BQU0sQ0FBQztnQkFFdkIsS0FBSyxJQUFJLEtBQUssQ0FBQztZQUNqQixDQUFDO1lBQ0QsS0FBSyxJQUFJLFFBQVEsQ0FBQyxNQUFNLENBQUM7WUFDekIsU0FBUyxFQUFFLENBQUM7WUFDWixFQUFFLENBQUMsQ0FBQyxTQUFTLEdBQUcsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ3pCLE9BQU8sQ0FBQyxHQUFHLENBQUMsYUFBYSxFQUFFLFNBQVMsRUFBRSxRQUFRLEVBQUUsSUFBSSxDQUFDLEdBQUcsRUFBRSxHQUFHLEtBQUssRUFDaEUsU0FBUyxFQUFFLEtBQUssQ0FBQyxDQUFDO1lBQ3RCLENBQUM7UUFDSCxDQUFDO1FBRUQsTUFBTSxDQUFDO1lBQ0wsVUFBVSxFQUFFLFNBQVM7WUFDckIsS0FBSyxFQUFFLEtBQUs7WUFDWixJQUFJLEVBQUUsSUFBSSxDQUFDLEdBQUcsRUFBRSxHQUFHLEtBQUs7WUFDeEIsSUFBSSxFQUFFLElBQUk7WUFDVixRQUFRLEVBQUUsUUFBUTtTQUNuQixDQUFBO0lBQ0gsQ0FBQztJQUVILGNBQUM7QUFBRCxDQXJsQkEsQUFxbEJDLElBQUE7QUFybEJZLGVBQU8sVUFxbEJuQixDQUFBO0FBRUQsSUFBYyxPQUFPLENBc0JwQjtBQXRCRCxXQUFjLE9BQU8sRUFBQyxDQUFDO0lBT1YsWUFBSSxHQUFHO1FBQ2hCLFFBQVE7UUFDUixhQUFhLEVBQUUsVUFBUyxNQUFNLEVBQUUsTUFBTTtZQUNwQyxJQUFJLFlBQVksR0FBRyxDQUFDLENBQUM7WUFDckIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxNQUFNLENBQUM7Z0JBQ25CLFlBQVksSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxLQUFLLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxLQUFLLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsa0RBQWtEO1lBQ3ZLLE1BQU0sQ0FBQyxZQUFZLENBQUM7UUFDdEIsQ0FBQztRQUNELEdBQUcsRUFBRSxVQUFTLE1BQU0sRUFBRSxNQUFNO1lBQzFCLElBQUksR0FBRyxHQUFHLENBQUMsQ0FBQztZQUNaLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksTUFBTSxDQUFDO2dCQUNuQixHQUFHLElBQUksSUFBSSxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1lBQzVDLE1BQU0sQ0FBQyxHQUFHLEdBQUcsTUFBTSxDQUFDLE1BQU0sQ0FBQztRQUM3QixDQUFDO0tBQ0YsQ0FBQTtBQUNILENBQUMsRUF0QmEsT0FBTyxHQUFQLGVBQU8sS0FBUCxlQUFPLFFBc0JwQiIsImZpbGUiOiJnZW5lcmF0ZWQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlc0NvbnRlbnQiOlsiKGZ1bmN0aW9uIGUodCxuLHIpe2Z1bmN0aW9uIHMobyx1KXtpZighbltvXSl7aWYoIXRbb10pe3ZhciBhPXR5cGVvZiByZXF1aXJlPT1cImZ1bmN0aW9uXCImJnJlcXVpcmU7aWYoIXUmJmEpcmV0dXJuIGEobywhMCk7aWYoaSlyZXR1cm4gaShvLCEwKTt2YXIgZj1uZXcgRXJyb3IoXCJDYW5ub3QgZmluZCBtb2R1bGUgJ1wiK28rXCInXCIpO3Rocm93IGYuY29kZT1cIk1PRFVMRV9OT1RfRk9VTkRcIixmfXZhciBsPW5bb109e2V4cG9ydHM6e319O3Rbb11bMF0uY2FsbChsLmV4cG9ydHMsZnVuY3Rpb24oZSl7dmFyIG49dFtvXVsxXVtlXTtyZXR1cm4gcyhuP246ZSl9LGwsbC5leHBvcnRzLGUsdCxuLHIpfXJldHVybiBuW29dLmV4cG9ydHN9dmFyIGk9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtmb3IodmFyIG89MDtvPHIubGVuZ3RoO28rKylzKHJbb10pO3JldHVybiBzfSkiLCJpbXBvcnQgaG9wZmllbGQgPSByZXF1aXJlKCcuL2FyY2hpdGVjdC9Ib3BmaWVsZCcpO1xuaW1wb3J0IGxzdG0gPSByZXF1aXJlKCcuL2FyY2hpdGVjdC9MU1RNJyk7XG5pbXBvcnQgbHNtID0gcmVxdWlyZSgnLi9hcmNoaXRlY3QvTGlxdWlkJyk7XG5pbXBvcnQgcGVyY2VwdHJvbiA9IHJlcXVpcmUoJy4vYXJjaGl0ZWN0L1BlcmNlcHRyb24nKTtcbmltcG9ydCBtYiA9IHJlcXVpcmUoJy4vYXJjaGl0ZWN0L01lbW9yeUJsb2NrJyk7XG5cbmV4cG9ydCB2YXIgTFNUTSA9IGxzdG0uTFNUTTtcbmV4cG9ydCB2YXIgTGlxdWlkID0gbHNtLkxpcXVpZDtcbmV4cG9ydCB2YXIgSG9wZmllbGQgPSBob3BmaWVsZC5Ib3BmaWVsZDtcbmV4cG9ydCB2YXIgUGVyY2VwdHJvbiA9IHBlcmNlcHRyb24uUGVyY2VwdHJvbjtcbmV4cG9ydCB2YXIgTWVtb3J5QmxvY2sgPSBtYi5NZW1vcnlCbG9jazsiLCJpbXBvcnQgbmV0d29yayAgPSByZXF1aXJlKCcuLi9uZXR3b3JrJyk7XG5pbXBvcnQgdHJhaW5lciAgPSByZXF1aXJlKCcuLi90cmFpbmVyJyk7XG5pbXBvcnQgbGF5ZXIgID0gcmVxdWlyZSgnLi4vbGF5ZXInKTtcbmltcG9ydCBuZXVyb24gPSByZXF1aXJlKCcuLi9uZXVyb24nKTtcblxuZXhwb3J0IGNsYXNzIEhvcGZpZWxkIGV4dGVuZHMgbmV0d29yay5OZXR3b3JrIHtcbiAgdHJhaW5lcjogdHJhaW5lci5UcmFpbmVyO1xuXG4gIGNvbnN0cnVjdG9yKHNpemU6IG51bWJlcikge1xuICAgIHZhciBpbnB1dExheWVyID0gbmV3IGxheWVyLkxheWVyKHNpemUpO1xuICAgIHZhciBvdXRwdXRMYXllciA9IG5ldyBsYXllci5MYXllcihzaXplKTtcblxuICAgIGlucHV0TGF5ZXIucHJvamVjdChvdXRwdXRMYXllciwgbGF5ZXIuTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTCk7XG5cbiAgICBzdXBlcih7XG4gICAgICBpbnB1dDogaW5wdXRMYXllcixcbiAgICAgIGhpZGRlbjogW10sXG4gICAgICBvdXRwdXQ6IG91dHB1dExheWVyXG4gICAgfSk7XG5cbiAgICB0aGlzLnRyYWluZXIgPSBuZXcgdHJhaW5lci5UcmFpbmVyKHRoaXMpO1xuICB9XG5cbiAgbGVhcm4ocGF0dGVybnMpIHtcbiAgICB2YXIgc2V0ID0gW107XG4gICAgZm9yICh2YXIgcCBpbiBwYXR0ZXJucylcbiAgICAgIHNldC5wdXNoKHtcbiAgICAgICAgaW5wdXQ6IHBhdHRlcm5zW3BdLFxuICAgICAgICBvdXRwdXQ6IHBhdHRlcm5zW3BdXG4gICAgICB9KTtcblxuICAgIHJldHVybiB0aGlzLnRyYWluZXIudHJhaW4oc2V0LCB7XG4gICAgICBpdGVyYXRpb25zOiA1MDAwMDAsXG4gICAgICBlcnJvcjogLjAwMDA1LFxuICAgICAgcmF0ZTogMVxuICAgIH0pO1xuICB9XG5cbiAgZmVlZChwYXR0ZXJuKSB7XG4gICAgdmFyIG91dHB1dCA9IHRoaXMuYWN0aXZhdGUocGF0dGVybik7XG5cbiAgICB2YXIgcGF0dGVybnMgPSBbXTtcbiAgICBmb3IgKHZhciBpIGluIG91dHB1dClcbiAgICAgIHBhdHRlcm5zW2ldID0gb3V0cHV0W2ldID4gLjUgPyAxIDogMDtcblxuICAgIHJldHVybiBwYXR0ZXJucztcbiAgfVxufSIsImltcG9ydCBuZXR3b3JrICA9IHJlcXVpcmUoJy4uL25ldHdvcmsnKTtcbmltcG9ydCB0cmFpbmVyICA9IHJlcXVpcmUoJy4uL3RyYWluZXInKTtcbmltcG9ydCBMYXllciAgPSByZXF1aXJlKCcuLi9sYXllcicpO1xuaW1wb3J0IG5ldXJvbiA9IHJlcXVpcmUoJy4uL25ldXJvbicpO1xuXG5leHBvcnQgY2xhc3MgTFNUTSBleHRlbmRzIG5ldHdvcmsuTmV0d29yayB7XG4gIHRyYWluZXI6IHRyYWluZXIuVHJhaW5lcjtcblxuICBjb25zdHJ1Y3RvciguLi5hcmdzOiBhbnlbXSkge1xuXG4gICAgaWYgKGFyZ3MubGVuZ3RoIDwgMylcbiAgICAgIHRocm93IFwiRXJyb3I6IG5vdCBlbm91Z2ggbGF5ZXJzIChtaW5pbXVtIDMpICEhXCI7XG5cbiAgICB2YXIgbGFzdCA9IGFyZ3MucG9wKCk7XG4gICAgdmFyIG9wdGlvbiA9IHtcbiAgICAgIHBlZXBob2xlczogTGF5ZXIuTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTCxcbiAgICAgIGhpZGRlbnRvaGlkZGVuOiBmYWxzZSxcbiAgICAgIG91dHRvaGlkZGVuOiBmYWxzZSxcbiAgICAgIG91dHRvZ2F0ZXM6IGZhbHNlLFxuICAgICAgaW50b291dDogdHJ1ZSxcbiAgICB9O1xuXG4gICAgaWYgKHR5cGVvZiBsYXN0ICE9ICdudW1iZXInKSB7XG4gICAgICB2YXIgb3V0cHV0cyA9IGFyZ3MucG9wKCk7XG4gICAgICBpZiAobGFzdC5oYXNPd25Qcm9wZXJ0eSgncGVlcGhvbGVzJykpXG4gICAgICAgIG9wdGlvbi5wZWVwaG9sZXMgPSBsYXN0LnBlZXBob2xlcztcbiAgICAgIGlmIChsYXN0Lmhhc093blByb3BlcnR5KCdoaWRkZW50b2hpZGRlbicpKVxuICAgICAgICBvcHRpb24uaGlkZGVudG9oaWRkZW4gPSBsYXN0LmhpZGRlbnRvaGlkZGVuO1xuICAgICAgaWYgKGxhc3QuaGFzT3duUHJvcGVydHkoJ291dHRvaGlkZGVuJykpXG4gICAgICAgIG9wdGlvbi5vdXR0b2hpZGRlbiA9IGxhc3Qub3V0dG9oaWRkZW47XG4gICAgICBpZiAobGFzdC5oYXNPd25Qcm9wZXJ0eSgnb3V0dG9nYXRlcycpKVxuICAgICAgICBvcHRpb24ub3V0dG9nYXRlcyA9IGxhc3Qub3V0dG9nYXRlcztcbiAgICAgIGlmIChsYXN0Lmhhc093blByb3BlcnR5KCdpbnRvb3V0JykpXG4gICAgICAgIG9wdGlvbi5pbnRvb3V0ID0gbGFzdC5pbnRvb3V0O1xuICAgIH0gZWxzZVxuICAgICAgdmFyIG91dHB1dHMgPSBsYXN0O1xuXG4gICAgdmFyIGlucHV0cyA9IGFyZ3Muc2hpZnQoKTtcbiAgICB2YXIgbGF5ZXJzID0gYXJncztcblxuICAgIHZhciBpbnB1dExheWVyID0gbmV3IExheWVyLkxheWVyKGlucHV0cyk7XG4gICAgdmFyIGhpZGRlbkxheWVycyA9IFtdO1xuICAgIHZhciBvdXRwdXRMYXllciA9IG5ldyBMYXllci5MYXllcihvdXRwdXRzKTtcblxuICAgIHZhciBwcmV2aW91cyA9IG51bGw7XG5cbiAgICAvLyBnZW5lcmF0ZSBsYXllcnNcbiAgICBmb3IgKHZhciBsYXllciBpbiBsYXllcnMpIHtcbiAgICAgIC8vIGdlbmVyYXRlIG1lbW9yeSBibG9ja3MgKG1lbW9yeSBjZWxsIGFuZCByZXNwZWN0aXZlIGdhdGVzKVxuICAgICAgdmFyIHNpemUgPSBsYXllcnNbbGF5ZXJdO1xuXG4gICAgICB2YXIgaW5wdXRHYXRlID0gbmV3IExheWVyLkxheWVyKHNpemUpLnNldCh7XG4gICAgICAgIGJpYXM6IDFcbiAgICAgIH0pO1xuICAgICAgdmFyIGZvcmdldEdhdGUgPSBuZXcgTGF5ZXIuTGF5ZXIoc2l6ZSkuc2V0KHtcbiAgICAgICAgYmlhczogMVxuICAgICAgfSk7XG4gICAgICB2YXIgbWVtb3J5Q2VsbCA9IG5ldyBMYXllci5MYXllcihzaXplKTtcbiAgICAgIHZhciBvdXRwdXRHYXRlID0gbmV3IExheWVyLkxheWVyKHNpemUpLnNldCh7XG4gICAgICAgIGJpYXM6IDFcbiAgICAgIH0pO1xuXG4gICAgICBoaWRkZW5MYXllcnMucHVzaChpbnB1dEdhdGUpO1xuICAgICAgaGlkZGVuTGF5ZXJzLnB1c2goZm9yZ2V0R2F0ZSk7XG4gICAgICBoaWRkZW5MYXllcnMucHVzaChtZW1vcnlDZWxsKTtcbiAgICAgIGhpZGRlbkxheWVycy5wdXNoKG91dHB1dEdhdGUpO1xuXG4gICAgICAvLyBjb25uZWN0aW9ucyBmcm9tIGlucHV0IGxheWVyXG4gICAgICB2YXIgaW5wdXQgPSBpbnB1dExheWVyLnByb2plY3QobWVtb3J5Q2VsbCk7XG4gICAgICBpbnB1dExheWVyLnByb2plY3QoaW5wdXRHYXRlKTtcbiAgICAgIGlucHV0TGF5ZXIucHJvamVjdChmb3JnZXRHYXRlKTtcbiAgICAgIGlucHV0TGF5ZXIucHJvamVjdChvdXRwdXRHYXRlKTtcblxuICAgICAgLy8gY29ubmVjdGlvbnMgZnJvbSBwcmV2aW91cyBtZW1vcnktYmxvY2sgbGF5ZXIgdG8gdGhpcyBvbmVcbiAgICAgIGlmIChwcmV2aW91cyAhPSBudWxsKSB7XG4gICAgICAgIHZhciBjZWxsID0gcHJldmlvdXMucHJvamVjdChtZW1vcnlDZWxsKTtcbiAgICAgICAgcHJldmlvdXMucHJvamVjdChpbnB1dEdhdGUpO1xuICAgICAgICBwcmV2aW91cy5wcm9qZWN0KGZvcmdldEdhdGUpO1xuICAgICAgICBwcmV2aW91cy5wcm9qZWN0KG91dHB1dEdhdGUpO1xuICAgICAgfVxuXG4gICAgICAvLyBjb25uZWN0aW9ucyBmcm9tIG1lbW9yeSBjZWxsXG4gICAgICB2YXIgb3V0cHV0ID0gbWVtb3J5Q2VsbC5wcm9qZWN0KG91dHB1dExheWVyKTtcblxuICAgICAgLy8gc2VsZi1jb25uZWN0aW9uXG4gICAgICB2YXIgc2VsZiA9IG1lbW9yeUNlbGwucHJvamVjdChtZW1vcnlDZWxsKTtcblxuICAgICAgLy8gaGlkZGVuIHRvIGhpZGRlbiByZWN1cnJlbnQgY29ubmVjdGlvblxuICAgICAgaWYgKG9wdGlvbi5oaWRkZW50b2hpZGRlbilcbiAgICAgICAgbWVtb3J5Q2VsbC5wcm9qZWN0KG1lbW9yeUNlbGwsIExheWVyLkxheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19FTFNFKTtcblxuICAgICAgLy8gb3V0IHRvIGhpZGRlbiByZWN1cnJlbnQgY29ubmVjdGlvblxuICAgICAgaWYgKG9wdGlvbi5vdXR0b2hpZGRlbilcbiAgICAgICAgb3V0cHV0TGF5ZXIucHJvamVjdChtZW1vcnlDZWxsKTtcblxuICAgICAgLy8gb3V0IHRvIGdhdGVzIHJlY3VycmVudCBjb25uZWN0aW9uXG4gICAgICBpZiAob3B0aW9uLm91dHRvZ2F0ZXMpIHtcbiAgICAgICAgb3V0cHV0TGF5ZXIucHJvamVjdChpbnB1dEdhdGUpO1xuICAgICAgICBvdXRwdXRMYXllci5wcm9qZWN0KG91dHB1dEdhdGUpO1xuICAgICAgICBvdXRwdXRMYXllci5wcm9qZWN0KGZvcmdldEdhdGUpO1xuICAgICAgfVxuICAgICAgXG4gICAgICAvLyBwZWVwaG9sZXNcbiAgICAgIG1lbW9yeUNlbGwucHJvamVjdChpbnB1dEdhdGUsIG9wdGlvbi5wZWVwaG9sZXMpO1xuICAgICAgbWVtb3J5Q2VsbC5wcm9qZWN0KGZvcmdldEdhdGUsIG9wdGlvbi5wZWVwaG9sZXMpO1xuICAgICAgbWVtb3J5Q2VsbC5wcm9qZWN0KG91dHB1dEdhdGUsIG9wdGlvbi5wZWVwaG9sZXMpO1xuXG4gICAgICAvLyBnYXRlc1xuICAgICAgaW5wdXRHYXRlLmdhdGUoaW5wdXQsIExheWVyLkxheWVyLmdhdGVUeXBlLklOUFVUKTtcbiAgICAgIGZvcmdldEdhdGUuZ2F0ZShzZWxmLCBMYXllci5MYXllci5nYXRlVHlwZS5PTkVfVE9fT05FKTtcbiAgICAgIG91dHB1dEdhdGUuZ2F0ZShvdXRwdXQsIExheWVyLkxheWVyLmdhdGVUeXBlLk9VVFBVVCk7XG4gICAgICBpZiAocHJldmlvdXMgIT0gbnVsbClcbiAgICAgICAgaW5wdXRHYXRlLmdhdGUoY2VsbCwgTGF5ZXIuTGF5ZXIuZ2F0ZVR5cGUuSU5QVVQpO1xuXG4gICAgICBwcmV2aW91cyA9IG1lbW9yeUNlbGw7XG4gICAgfVxuXG4gICAgLy8gaW5wdXQgdG8gb3V0cHV0IGRpcmVjdCBjb25uZWN0aW9uXG4gICAgaWYgKG9wdGlvbi5pbnRvb3V0KVxuICAgICAgaW5wdXRMYXllci5wcm9qZWN0KG91dHB1dExheWVyKTtcblxuICAgIC8vIHNldCB0aGUgbGF5ZXJzIG9mIHRoZSBuZXVyYWwgbmV0d29ya1xuICAgIHN1cGVyKHtcbiAgICAgIGlucHV0OiBpbnB1dExheWVyLFxuICAgICAgaGlkZGVuOiBoaWRkZW5MYXllcnMsXG4gICAgICBvdXRwdXQ6IG91dHB1dExheWVyXG4gICAgfSk7XG5cbiAgICAvLyB0cmFpbmVyXG4gICAgdGhpcy50cmFpbmVyID0gbmV3IHRyYWluZXIuVHJhaW5lcih0aGlzKTtcbiAgfVxufTtcbiIsImltcG9ydCBuZXR3b3JrICA9IHJlcXVpcmUoJy4uL25ldHdvcmsnKTtcbmltcG9ydCB0cmFpbmVyICA9IHJlcXVpcmUoJy4uL3RyYWluZXInKTtcbmltcG9ydCBsYXllciAgPSByZXF1aXJlKCcuLi9sYXllcicpO1xuaW1wb3J0IG5ldXJvbiA9IHJlcXVpcmUoJy4uL25ldXJvbicpO1xuXG5leHBvcnQgY2xhc3MgTGlxdWlkIGV4dGVuZHMgbmV0d29yay5OZXR3b3JrIHtcbiAgdHJhaW5lcjogdHJhaW5lci5UcmFpbmVyO1xuXG4gIGNvbnN0cnVjdG9yKGlucHV0cywgaGlkZGVuLCBvdXRwdXRzLCBjb25uZWN0aW9ucywgZ2F0ZXMpIHtcblxuICAgIC8vIGNyZWF0ZSBsYXllcnNcbiAgICB2YXIgaW5wdXRMYXllciA9IG5ldyBsYXllci5MYXllcihpbnB1dHMpO1xuICAgIHZhciBoaWRkZW5MYXllciA9IG5ldyBsYXllci5MYXllcihoaWRkZW4pO1xuICAgIHZhciBvdXRwdXRMYXllciA9IG5ldyBsYXllci5MYXllcihvdXRwdXRzKTtcblxuICAgIC8vIG1ha2UgY29ubmVjdGlvbnMgYW5kIGdhdGVzIHJhbmRvbWx5IGFtb25nIHRoZSBuZXVyb25zXG4gICAgdmFyIG5ldXJvbnMgPSBoaWRkZW5MYXllci5uZXVyb25zKCk7XG4gICAgdmFyIGNvbm5lY3Rpb25MaXN0OiBuZXVyb24uTmV1cm9uLkNvbm5lY3Rpb25bXSA9IFtdO1xuXG4gICAgZm9yICh2YXIgaSA9IDA7IGkgPCBjb25uZWN0aW9uczsgaSsrKSB7XG4gICAgICAvLyBjb25uZWN0IHR3byByYW5kb20gbmV1cm9uc1xuICAgICAgdmFyIGZyb20gPSBNYXRoLnJhbmRvbSgpICogbmV1cm9ucy5sZW5ndGggfCAwO1xuICAgICAgdmFyIHRvID0gTWF0aC5yYW5kb20oKSAqIG5ldXJvbnMubGVuZ3RoIHwgMDtcbiAgICAgIHZhciBjb25uZWN0aW9uID0gbmV1cm9uc1tmcm9tXS5wcm9qZWN0KG5ldXJvbnNbdG9dKTtcbiAgICAgIGNvbm5lY3Rpb25MaXN0LnB1c2goY29ubmVjdGlvbik7XG4gICAgfVxuXG4gICAgZm9yICh2YXIgaiA9IDA7IGogPCBnYXRlczsgaisrKSB7XG4gICAgICAvLyBwaWNrIGEgcmFuZG9tIGdhdGVyIG5ldXJvblxuICAgICAgdmFyIGdhdGVyID0gTWF0aC5yYW5kb20oKSAqIG5ldXJvbnMubGVuZ3RoIHwgMDtcbiAgICAgIC8vIHBpY2sgYSByYW5kb20gY29ubmVjdGlvbiB0byBnYXRlXG4gICAgICB2YXIgY29ubmVjdGlvbk51bWJlciA9IE1hdGgucmFuZG9tKCkgKiBjb25uZWN0aW9uTGlzdC5sZW5ndGggfCAwO1xuICAgICAgLy8gbGV0IHRoZSBnYXRlciBnYXRlIHRoZSBjb25uZWN0aW9uXG4gICAgICBuZXVyb25zW2dhdGVyXS5nYXRlKGNvbm5lY3Rpb25MaXN0W2Nvbm5lY3Rpb25OdW1iZXJdKTtcbiAgICB9XG5cbiAgICAvLyBjb25uZWN0IHRoZSBsYXllcnNcbiAgICBpbnB1dExheWVyLnByb2plY3QoaGlkZGVuTGF5ZXIpO1xuICAgIGhpZGRlbkxheWVyLnByb2plY3Qob3V0cHV0TGF5ZXIpO1xuXG4gICAgLy8gc2V0IHRoZSBsYXllcnMgb2YgdGhlIG5ldHdvcmtcbiAgICBzdXBlcih7XG4gICAgICBpbnB1dDogaW5wdXRMYXllcixcbiAgICAgIGhpZGRlbjogW2hpZGRlbkxheWVyXSxcbiAgICAgIG91dHB1dDogb3V0cHV0TGF5ZXJcbiAgICB9KTtcblxuICAgIC8vIHRyYWluZXJcbiAgICB0aGlzLnRyYWluZXIgPSBuZXcgdHJhaW5lci5UcmFpbmVyKHRoaXMpO1xuICB9XG59XG4iLCJpbXBvcnQgbmV0d29yayA9IHJlcXVpcmUoJy4uL25ldHdvcmsnKTtcbmltcG9ydCB0cmFpbmVyID0gcmVxdWlyZSgnLi4vdHJhaW5lcicpO1xuaW1wb3J0IExheWVyID0gcmVxdWlyZSgnLi4vbGF5ZXInKTtcbmltcG9ydCBuZXVyb24gPSByZXF1aXJlKCcuLi9uZXVyb24nKTtcbmltcG9ydCBTeW5hcHRpYyA9IHJlcXVpcmUoJy4uL3N5bmFwdGljJyk7XG5pbXBvcnQgU3F1YXNoID0gcmVxdWlyZSgnLi4vc3F1YXNoJyk7XG5cbmV4cG9ydCBjbGFzcyBNZW1vcnlUYXBlIHtcbiAgZGF0YTogRmxvYXQ2NEFycmF5W107XG5cbiAgYmxvY2tXaWR0aDogbnVtYmVyO1xuICBibG9ja3M6IG51bWJlcjtcblxuICBsYXllcjogTGF5ZXIuTGF5ZXI7XG5cbiAgcHJldkxheWVyQWN0aXZhdGU6IGFueTtcbiAgcHJldkxheWVyUHJvcGFnYXRlOiBhbnk7XG5cbiAgbWVtb3J5QXR0ZW50aW9uTG9jYXRpb246IG51bWJlciA9IDA7XG4gIG1lbW9yeUF0dGVudGlvbldlaWdodDogbnVtYmVyID0gMDtcblxuICBvdXRwdXRMYXllcjogTGF5ZXIuTGF5ZXI7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgbWVtb3J5QmxvY2tzOiBudW1iZXIsXG4gICAgbGF5ZXI6IExheWVyLkxheWVyLFxuICAgIGlucHV0R2F0ZTogTGF5ZXIuTGF5ZXIsXG4gICAgZm9yZ2V0R2F0ZTogTGF5ZXIuTGF5ZXJcbiAgICApIHtcblxuICAgIHRoaXMuYmxvY2tzID0gbWVtb3J5QmxvY2tzO1xuXG4gICAgdGhpcy5kYXRhID0gbmV3IEFycmF5KG1lbW9yeUJsb2Nrcyk7XG5cbiAgICB0aGlzLmJsb2NrV2lkdGggPSBsYXllci5saXN0Lmxlbmd0aDtcblxuICAgIHRoaXMubGF5ZXIgPSBsYXllcjtcbiAgICBcbiAgICB0aGlzLm91dHB1dExheWVyID0gbmV3IExheWVyLkxheWVyKGxheWVyLmxpc3QubGVuZ3RoKTtcbiAgICBcbiAgICB0aGlzLm91dHB1dExheWVyLnByb2plY3QobGF5ZXIsIExheWVyLkxheWVyLmNvbm5lY3Rpb25UeXBlLk9ORV9UT19PTkUpO1xuXG4gICAgZm9yICh2YXIgbG9jYXRpb24gPSAwOyBsb2NhdGlvbiA8IHRoaXMuYmxvY2tzOyBsb2NhdGlvbisrKSB7XG4gICAgICB2YXIgYXJyYXkgPSB0aGlzLmRhdGFbbG9jYXRpb25dID0gbmV3IEZsb2F0NjRBcnJheSh0aGlzLmJsb2NrV2lkdGgpO1xuICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBhcnJheS5sZW5ndGg7IGkrKykge1xuICAgICAgICBhcnJheVtpXSA9IE1hdGgucmFuZG9tKCk7XG4gICAgICB9XG4gICAgfVxuICAgIC8vIEhhY2sgdGhlIGxheWVyIVxuICAgIFxuICAgIHZhciBzaW1pbGFyQWRkcmVzc2VzIDogRmxvYXQ2NEFycmF5O1xuICAgIC8vIGVsZWdpYmxlIG1lbWJsb2NrcyBmb3IgcmVhZC93cml0ZSBvcGVyYXRpb25zXG4gICAgdmFyIGVsZWdpYmxlSW5kZXhlcyA9IFswLCAxLCAyXTtcbiAgICB2YXIgZWxlZ2libGVXZWlnaHRzID0gWzAuMDEsIDEsIDAuMDFdOyAvLyBzaGlmdGluZywgc29mdG1heFxuICAgIHZhciBmb2N1cyA9IDE7XG4gICAgXG4gICAgdGhpcy5wcmV2TGF5ZXJBY3RpdmF0ZSA9IHRoaXMubGF5ZXIuYWN0aXZhdGUuYmluZCh0aGlzLmxheWVyKTtcbiAgICB0aGlzLnByZXZMYXllclByb3BhZ2F0ZSA9IHRoaXMubGF5ZXIucHJvcGFnYXRlLmJpbmQodGhpcy5sYXllcik7XG4gICAgXG4gICAgdmFyIGtleTogRmxvYXQ2NEFycmF5O1xuICAgIFxuICAgIHRoaXMubGF5ZXIucHJvcGFnYXRlID0gKHJhdGU6IG51bWJlciwgdGFyZ2V0PzogU3luYXB0aWMuSU51bWVyaWNBcnJheSkgPT4ge1xuICAgICAgdGhpcy5wcmV2TGF5ZXJQcm9wYWdhdGUocmF0ZSwgdGFyZ2V0KTtcbiAgICAgIFxuICAgICAgdmFyIGFkZEdhdGUgPSBpbnB1dEdhdGUuY3VycmVudEFjdGl2YXRpb247XG4gICAgICB2YXIgZXJhc2VHYXRlID0gZm9yZ2V0R2F0ZS5jdXJyZW50QWN0aXZhdGlvbjtcbiAgICAgIFxuICAgICAgZm9yICh2YXIgbiA9IDA7IG4gPCBlbGVnaWJsZUluZGV4ZXMubGVuZ3RoOyBuKyspIHtcbiAgICAgICAgdmFyIE0gPSB0aGlzLmRhdGFbZWxlZ2libGVJbmRleGVzW25dXTtcbiAgICAgICBcbiAgICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBNLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgLy8gZG8gZXJhc2Ugb3BlcmF0aW9ucyBvbiB0aGUgbWVtb3J5IHRhcGUuIE5UTTogMy4yICgzKVxuICAgICAgICAgIE1baV0gKj0gMSAtIGVyYXNlR2F0ZVtpXSAqIGtleVtpXSAqIGVsZWdpYmxlV2VpZ2h0c1tuXTtcbiAgICAgICAgICAvLyBkbyBhZGQgb3BlcmF0aW9ucyBvbiB0aGUgbWVtb3J5IHRhcGUuIE5UTTogMy4yICg0KVxuICAgICAgICAgIE1baV0gKz0gYWRkR2F0ZVtpXSAqIGtleVtpXSAqIGVsZWdpYmxlV2VpZ2h0c1tuXSAqIHJhdGU7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICB0aGlzLmxheWVyLmFjdGl2YXRlID0gKGlucHV0PzogU3luYXB0aWMuSU51bWVyaWNBcnJheSk6IEZsb2F0NjRBcnJheSA9PiB7XG4gICAgICB2YXIgcmVzdWx0ID0gdGhpcy5wcmV2TGF5ZXJBY3RpdmF0ZShpbnB1dCk7IFxuICAgICAga2V5ID0gTWVtb3J5VGFwZS5zb2Z0TWF4QXJyYXkoIG5ldyBGbG9hdDY0QXJyYXkocmVzdWx0KSk7XG4gICAgICBcbiAgICAgIHNpbWlsYXJBZGRyZXNzZXMgPSB0aGlzLmdldFNpbWlsYXJBZHJlc3NlcyhrZXkpO1xuICAgICAgXG4gICAgICB0aGlzLm1lbW9yeUF0dGVudGlvbldlaWdodCA9IDA7XG4gICAgICBcbiAgICAgIGZvciAodmFyIGFkZHJlc3MgPSAwOyBhZGRyZXNzIDwgc2ltaWxhckFkZHJlc3Nlcy5sZW5ndGg7IGFkZHJlc3MrKykge1xuICAgICAgICB2YXIgw58gPSBzaW1pbGFyQWRkcmVzc2VzW2FkZHJlc3NdO1xuICAgICAgICBpZijDnyA+IHRoaXMubWVtb3J5QXR0ZW50aW9uV2VpZ2h0KXtcbiAgICAgICAgICB0aGlzLm1lbW9yeUF0dGVudGlvbldlaWdodCA9IMOfO1xuICAgICAgICAgIHRoaXMubWVtb3J5QXR0ZW50aW9uTG9jYXRpb24gPSBhZGRyZXNzO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBcbiAgICAgIGVsZWdpYmxlSW5kZXhlcyA9IFt0aGlzLm1lbW9yeUF0dGVudGlvbkxvY2F0aW9uIC0gMSwgdGhpcy5tZW1vcnlBdHRlbnRpb25Mb2NhdGlvbiwgdGhpcy5tZW1vcnlBdHRlbnRpb25Mb2NhdGlvbiArIDFdO1xuICAgICAgXG4gICAgICBmb2N1cyA9IHRoaXMubWVtb3J5QXR0ZW50aW9uV2VpZ2h0O1xuXG4gICAgICBlbGVnaWJsZVdlaWdodHMgPSBbMC4xLCAwLjgsIDAuMV07IC8vIHNoaWZ0aW5nLCBzb2Z0bWF4XG5cbiAgICAgIGZvciAodmFyIG4gPSAwOyBuIDwgZWxlZ2libGVJbmRleGVzLmxlbmd0aDsgbisrKSB7XG4gICAgICAgIHZhciBpbmRleCA9IGVsZWdpYmxlSW5kZXhlc1tuXTtcbiAgICAgICAgXG4gICAgICAgIGlmKGluZGV4IDwgMClcbiAgICAgICAgICBpbmRleCArPSBzaW1pbGFyQWRkcmVzc2VzLmxlbmd0aDtcbiAgICAgICAgZWxzZSBpZihpbmRleCA+PSBzaW1pbGFyQWRkcmVzc2VzLmxlbmd0aClcbiAgICAgICAgICBpbmRleCAtPSBzaW1pbGFyQWRkcmVzc2VzLmxlbmd0aDtcbiAgICAgICAgICBcbiAgICAgICAgZWxlZ2libGVJbmRleGVzW25dID0gaW5kZXg7XG4gICAgICAgIFxuICAgICAgICBlbGVnaWJsZVdlaWdodHNbbl0gPSBlbGVnaWJsZVdlaWdodHNbbl0gLyBmb2N1cyAqIHNpbWlsYXJBZGRyZXNzZXNbaW5kZXhdO1xuICAgICAgfVxuXG4gICAgICB0aGlzLm91dHB1dExheWVyLmxpc3QuZm9yRWFjaCgobmV1cm9uLCBpKSA9PiB7XG4gICAgICAgIC8vIG1vZGlmeSB0aGUgY3VycmVudCBrZXkgKHJlYWRWZWN0b3IpXG4gICAgICAgIHZhciB0bXBLZXkgPSAwO1xuICAgICAgICBcbiAgICAgICAgZm9yICh2YXIgbiA9IDA7IG4gPCBlbGVnaWJsZUluZGV4ZXMubGVuZ3RoOyBuKyspIHtcbiAgICAgICAgICB0bXBLZXkgKz0gdGhpcy5kYXRhW2VsZWdpYmxlSW5kZXhlc1tuXV1baV0gKiBlbGVnaWJsZVdlaWdodHNbbl07XG4gICAgICAgIH1cblxuICAgICAgICBuZXVyb24uYWN0aXZhdGUodG1wS2V5KTtcbiAgICAgIH0pO1xuXG4gICAgICByZXR1cm4gcmVzdWx0O1xuICAgIH1cbiAgfVxuXG4gIHN0YXRpYyBnZXRTaW1pbGFyaXR5KGFycmF5QTogU3luYXB0aWMuSU51bWVyaWNBcnJheSwgYXJyYXlCOiBTeW5hcHRpYy5JTnVtZXJpY0FycmF5KTogbnVtYmVyIHtcbiAgICAvLyBodHRwOi8vZW4ud2lraXBlZGlhLm9yZy93aWtpL0Nvc2luZV9zaW1pbGFyaXR5XG4gICAgLy8gTlRNOiAzLjMuMSAoNilcbiAgICB2YXIgZG90UHIgPSAwO1xuXG4gICAgdmFyIGFjdW1BID0gMCwgYWN1bUIgPSAwO1xuXG4gICAgZm9yICh2YXIgaSA9IDA7IGkgPCBhcnJheUEubGVuZ3RoOyBpKyspIHtcbiAgICAgIGRvdFByICs9IGFycmF5QVtpXSAqIGFycmF5QltpXTtcbiAgICAgIGFjdW1BICs9IGFycmF5QVtpXSAqIGFycmF5QVtpXTtcbiAgICAgIGFjdW1CICs9IGFycmF5QltpXSAqIGFycmF5QltpXTtcbiAgICB9XG5cbiAgICByZXR1cm4gZG90UHIgLyAoTWF0aC5zcXJ0KGFjdW1BKSAqIE1hdGguc3FydChhY3VtQikgKyAuMDAwMDUpO1xuICB9XG5cbiAgc3RhdGljIHNvZnRNYXhBcnJheTxUIGV4dGVuZHMgU3luYXB0aWMuSU51bWVyaWNBcnJheT4oYXJyYXk6IFQsIHNoYXJwZW4gPSAxKTogVCB7XG4gICAgLy8gZm9yIGFsbCBpIOKIiCBhcnJheVxuICAgIC8vIHN1bSA9IOKIkSBhcnJheVtuXV5lXG4gICAgLy8gaSA9IMOuXmUgLyBzdW1cbiAgICAvLyB3aGVyZSB0aGUgcmVzdWx0IOKIkSBhcnJheVswLi5uXSA9IDFcblxuICAgIGlmICghYXJyYXkubGVuZ3RoKSByZXR1cm4gYXJyYXk7XG5cbiAgICBzaGFycGVuID0gc2hhcnBlbiB8fCAxO1xuXG4gICAgdmFyIHN1bSA9IDA7XG5cbiAgICAvLyBzdW0gPSDiiJEgYXJyYXlbbl1eZVxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgYXJyYXkubGVuZ3RoOyBpKyspIHtcbiAgICAgIGFycmF5W2ldID0gTWF0aC5leHAoc2hhcnBlbiAqIGFycmF5W2ldKTtcbiAgICAgIHN1bSArPSBhcnJheVtpXTtcbiAgICB9XG5cbiAgICBpZiAoc3VtICE9IDApIHtcbiAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgYXJyYXkubGVuZ3RoOyBpKyspIGFycmF5W2ldIC89IHN1bTtcbiAgICB9IGVsc2Uge1xuICAgICAgdmFyIGRpdiA9IDEgLyBhcnJheS5sZW5ndGg7XG4gICAgICBmb3IgKHZhciBpID0gMDsgaSA8IGFycmF5Lmxlbmd0aDsgaSsrKSBhcnJheVtpXSA9IGRpdjtcbiAgICB9XG5cbiAgICByZXR1cm4gYXJyYXk7XG4gIH1cbiAgXG4gIC8vIG9idGFpbnMgYW4gYXJyYXkgb2Ygc2ltaWxhcml0eSBpbmRleGVzIGZvciBlYWNoIG1lbW9yeUJsb2NrXG4gIGdldFNpbWlsYXJBZHJlc3Nlcyh3ZWlnaHRzOiBTeW5hcHRpYy5JTnVtZXJpY0FycmF5KTogRmxvYXQ2NEFycmF5IHtcbiAgICAvL2NoZWNrcG9pbnQ6IDEwdGggY2lnYXJyZXRcbiAgICB2YXIgYWRkcmVzc2VzID0gbmV3IEZsb2F0NjRBcnJheSh0aGlzLmRhdGEubGVuZ3RoKTtcblxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgdGhpcy5kYXRhLmxlbmd0aDsgaSsrKVxuICAgICAgYWRkcmVzc2VzW2ldID0gTWVtb3J5VGFwZS5nZXRTaW1pbGFyaXR5KHRoaXMuZGF0YVtpXSwgd2VpZ2h0cyk7XG5cbiAgICByZXR1cm4gYWRkcmVzc2VzO1xuICB9XG59XG5cblxuZXhwb3J0IGNsYXNzIE1lbW9yeUJsb2NrIGV4dGVuZHMgbmV0d29yay5OZXR3b3JrIHtcbiAgdHJhaW5lcjogdHJhaW5lci5UcmFpbmVyO1xuICBtZW1vcnlUYXBlOiBNZW1vcnlUYXBlO1xuXG4gIGNvbnN0cnVjdG9yKGlucHV0U2l6ZTogbnVtYmVyLCBtZW1vcnlCbG9ja3M6IG51bWJlciwgbWVtb3J5V2lkdGg6IG51bWJlciwgb3V0cHV0U2l6ZTogbnVtYmVyKSB7XG5cblxuICAgIHZhciBvcHRpb24gPSB7XG4gICAgICBwZWVwaG9sZXM6IExheWVyLkxheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19BTEwsXG4gICAgICBoaWRkZW50b2hpZGRlbjogZmFsc2UsXG4gICAgICBvdXR0b2hpZGRlbjogZmFsc2UsXG4gICAgICBvdXR0b2dhdGVzOiBmYWxzZSxcbiAgICAgIGludG9vdXQ6IHRydWUsXG4gICAgfTtcblxuICAgIHZhciBpbnB1dExheWVyID0gbmV3IExheWVyLkxheWVyKGlucHV0U2l6ZSk7XG4gICAgdmFyIGhpZGRlbkxheWVycyA9IFtdO1xuICAgIHZhciBvdXRwdXRMYXllciA9IG5ldyBMYXllci5MYXllcihvdXRwdXRTaXplKTtcblxuXG4gICAgLy8jcmVnaW9uIGdlbmVyYXRlIGxheWVyc1xuXG4gICAgLy8gZ2VuZXJhdGUgbWVtb3J5IGJsb2NrcyAobWVtb3J5IGNlbGwgYW5kIHJlc3BlY3RpdmUgZ2F0ZXMpXG5cbiAgICB2YXIgaW5wdXRHYXRlID0gbmV3IExheWVyLkxheWVyKG1lbW9yeVdpZHRoKS5zZXQoe1xuICAgICAgYmlhczogMVxuICAgIH0pO1xuXG4gICAgdmFyIGZvcmdldEdhdGUgPSBuZXcgTGF5ZXIuTGF5ZXIobWVtb3J5V2lkdGgpLnNldCh7XG4gICAgICBiaWFzOiAxXG4gICAgfSk7XG5cbiAgICB2YXIgbWVtb3J5Q2VsbCA9IG5ldyBMYXllci5MYXllcihtZW1vcnlXaWR0aCk7XG5cblxuXG4gICAgdmFyIG91dHB1dEdhdGUgPSBuZXcgTGF5ZXIuTGF5ZXIobWVtb3J5V2lkdGgpLnNldCh7XG4gICAgICBiaWFzOiAxXG4gICAgfSk7XG5cbiAgICBoaWRkZW5MYXllcnMucHVzaChpbnB1dEdhdGUpO1xuICAgIGhpZGRlbkxheWVycy5wdXNoKGZvcmdldEdhdGUpO1xuICAgIGhpZGRlbkxheWVycy5wdXNoKG1lbW9yeUNlbGwpO1xuICAgIGhpZGRlbkxheWVycy5wdXNoKG91dHB1dEdhdGUpO1xuXG4gICAgLy8gY29ubmVjdGlvbnMgZnJvbSBpbnB1dCBsYXllclxuICAgIHZhciBpbnB1dCA9IGlucHV0TGF5ZXIucHJvamVjdChtZW1vcnlDZWxsKTtcbiAgICBpbnB1dExheWVyLnByb2plY3QoaW5wdXRHYXRlKTtcbiAgICBpbnB1dExheWVyLnByb2plY3QoZm9yZ2V0R2F0ZSk7XG4gICAgaW5wdXRMYXllci5wcm9qZWN0KG91dHB1dEdhdGUpO1xuXG4gICAgLy8gY29ubmVjdGlvbnMgZnJvbSBtZW1vcnkgY2VsbFxuICAgIHZhciBvdXRwdXQgPSBtZW1vcnlDZWxsLnByb2plY3Qob3V0cHV0TGF5ZXIpO1xuXG4gICAgLy8gc2VsZi1jb25uZWN0aW9uXG4gICAgdmFyIHNlbGYgPSBtZW1vcnlDZWxsLnByb2plY3QobWVtb3J5Q2VsbCk7XG4gICAgICBcbiAgICAvLyBwZWVwaG9sZXNcbiAgICBtZW1vcnlDZWxsLnByb2plY3QoaW5wdXRHYXRlLCBvcHRpb24ucGVlcGhvbGVzKTtcbiAgICBtZW1vcnlDZWxsLnByb2plY3QoZm9yZ2V0R2F0ZSwgb3B0aW9uLnBlZXBob2xlcyk7XG4gICAgbWVtb3J5Q2VsbC5wcm9qZWN0KG91dHB1dEdhdGUsIG9wdGlvbi5wZWVwaG9sZXMpO1xuXG4gICAgLy8gZ2F0ZXNcbiAgICBpbnB1dEdhdGUuZ2F0ZShpbnB1dCwgTGF5ZXIuTGF5ZXIuZ2F0ZVR5cGUuSU5QVVQpO1xuICAgIGZvcmdldEdhdGUuZ2F0ZShzZWxmLCBMYXllci5MYXllci5nYXRlVHlwZS5PTkVfVE9fT05FKTtcbiAgICBvdXRwdXRHYXRlLmdhdGUob3V0cHV0LCBMYXllci5MYXllci5nYXRlVHlwZS5PVVRQVVQpO1xuXG5cbiAgICB0aGlzLm1lbW9yeVRhcGUgPSBuZXcgTWVtb3J5VGFwZShtZW1vcnlCbG9ja3MsIG1lbW9yeUNlbGwsIGlucHV0R2F0ZSwgZm9yZ2V0R2F0ZSk7XG4gIFxuICAgIGhpZGRlbkxheWVycy5wdXNoKHRoaXMubWVtb3J5VGFwZS5vdXRwdXRMYXllcik7ICAgIFxuICAgIFxuICAgIC8vI2VuZHJlZ2lvblxuXG4gICAgLy8gaW5wdXQgdG8gb3V0cHV0IGRpcmVjdCBjb25uZWN0aW9uXG4gICAgaWYgKG9wdGlvbi5pbnRvb3V0KVxuICAgICAgaW5wdXRMYXllci5wcm9qZWN0KG91dHB1dExheWVyKTtcblxuICAgIC8vIHNldCB0aGUgbGF5ZXJzIG9mIHRoZSBuZXVyYWwgbmV0d29ya1xuICAgIHN1cGVyKHtcbiAgICAgIGlucHV0OiBpbnB1dExheWVyLFxuICAgICAgaGlkZGVuOiBoaWRkZW5MYXllcnMsXG4gICAgICBvdXRwdXQ6IG91dHB1dExheWVyXG4gICAgfSk7XG4gICAgXG4gICAgXG4gICAgLy8gRE8gTk9UIE9QVElNSVpFIFRISVMgTkVUV09SS1xuICAgIHRoaXMub3B0aW1pemVkID0gZmFsc2U7XG5cbiAgICAvLyB0cmFpbmVyXG4gICAgdGhpcy50cmFpbmVyID0gbmV3IHRyYWluZXIuVHJhaW5lcih0aGlzKTtcbiAgfVxufTtcbiIsImltcG9ydCBuZXR3b3JrICA9IHJlcXVpcmUoJy4uL25ldHdvcmsnKTtcbmltcG9ydCB0cmFpbmVyICA9IHJlcXVpcmUoJy4uL3RyYWluZXInKTtcbmltcG9ydCBsYXllciAgPSByZXF1aXJlKCcuLi9sYXllcicpO1xuaW1wb3J0IG5ldXJvbiA9IHJlcXVpcmUoJy4uL25ldXJvbicpO1xuLy8gTXVsdGlsYXllciBQZXJjZXB0cm9uXG5leHBvcnQgY2xhc3MgUGVyY2VwdHJvbiBleHRlbmRzIG5ldHdvcmsuTmV0d29yayB7XG4gIHRyYWluZXI6IHRyYWluZXIuVHJhaW5lcjtcblxuICBjb25zdHJ1Y3RvciguLi5hcmdzOiBudW1iZXJbXSkge1xuXG4gICAgaWYgKGFyZ3MubGVuZ3RoIDwgMylcbiAgICAgIHRocm93IFwiRXJyb3I6IG5vdCBlbm91Z2ggbGF5ZXJzIChtaW5pbXVtIDMpICEhXCI7XG5cbiAgICB2YXIgaW5wdXRzID0gYXJncy5zaGlmdCgpOyAvLyBmaXJzdCBhcmd1bWVudFxuICAgIHZhciBvdXRwdXRzID0gYXJncy5wb3AoKTsgLy8gbGFzdCBhcmd1bWVudFxuICAgIHZhciBsYXllcnMgPSBhcmdzOyAvLyBhbGwgdGhlIGFyZ3VtZW50cyBpbiB0aGUgbWlkZGxlXG4gIFxuICAgIHZhciBpbnB1dCA9IG5ldyBsYXllci5MYXllcihpbnB1dHMpO1xuICAgIHZhciBoaWRkZW4gPSBbXTtcbiAgICB2YXIgb3V0cHV0ID0gbmV3IGxheWVyLkxheWVyKG91dHB1dHMpO1xuXG4gICAgdmFyIHByZXZpb3VzID0gaW5wdXQ7XG4gIFxuICAgIC8vIGdlbmVyYXRlIGhpZGRlbiBsYXllcnNcbiAgICBmb3IgKHZhciBsZXZlbCBpbiBsYXllcnMpIHtcbiAgICAgIHZhciBzaXplID0gbGF5ZXJzW2xldmVsXTtcbiAgICAgIHZhciB0aGVMYXllciA9IG5ldyBsYXllci5MYXllcihzaXplKTtcbiAgICAgIGhpZGRlbi5wdXNoKHRoZUxheWVyKTtcbiAgICAgIHByZXZpb3VzLnByb2plY3QodGhlTGF5ZXIpO1xuICAgICAgcHJldmlvdXMgPSB0aGVMYXllcjtcbiAgICB9XG4gICAgcHJldmlvdXMucHJvamVjdChvdXRwdXQpO1xuICBcbiAgICAvLyBzZXQgbGF5ZXJzIG9mIHRoZSBuZXVyYWwgbmV0d29ya1xuICAgICAgXG4gICAgc3VwZXIoe1xuICAgICAgaW5wdXQ6IGlucHV0LFxuICAgICAgaGlkZGVuOiBoaWRkZW4sXG4gICAgICBvdXRwdXQ6IG91dHB1dFxuICAgIH0pO1xuICBcbiAgICAvLyB0cmFpbmVyIGZvciB0aGUgbmV0d29ya1xuICAgIHRoaXMudHJhaW5lciA9IG5ldyB0cmFpbmVyLlRyYWluZXIodGhpcyk7XG4gIH1cbn07ICIsImltcG9ydCBuZXVyb24gPSByZXF1aXJlKCcuL25ldXJvbicpO1xuaW1wb3J0IG5ldHdvcmsgPSByZXF1aXJlKCcuL25ldHdvcmsnKTtcbmltcG9ydCBTeW5hcHRpYyA9IHJlcXVpcmUoJy4vc3luYXB0aWMnKTtcblxuLyoqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgTEFZRVJcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5leHBvcnQgY2xhc3MgTGF5ZXIge1xuXHRcdGxpc3Q6IG5ldXJvbi5OZXVyb25bXSA9IFtdO1xuXHRcdGxhYmVsOiBzdHJpbmcgPSBudWxsO1xuXHRcdGNvbm5lY3RlZHRvID0gW107XG5cdFx0c2l6ZSA9IDA7XG5cdFx0XG5cdFx0Y3VycmVudEFjdGl2YXRpb246IEZsb2F0NjRBcnJheTtcblxuXHRcdGNvbnN0cnVjdG9yKHNpemU6IG51bWJlciwgbGFiZWw/OiBzdHJpbmcpIHtcblx0XHRcdHRoaXMuc2l6ZSA9IHNpemUgfCAwO1xuXHRcdFx0dGhpcy5saXN0ID0gW107XG5cdFx0XHR0aGlzLmxhYmVsID0gbGFiZWwgfHwgbnVsbDtcblx0XHRcdHRoaXMuY29ubmVjdGVkdG8gPSBbXTtcblx0XHRcdFxuXHRcdFx0dGhpcy5jdXJyZW50QWN0aXZhdGlvbiA9IG5ldyBGbG9hdDY0QXJyYXkoc2l6ZSk7XG5cblx0XHRcdHdoaWxlIChzaXplLS0pIHtcblx0XHRcdFx0dmFyIHRoZU5ldXJvbiA9IG5ldyBuZXVyb24uTmV1cm9uKCk7XG5cdFx0XHRcdHRoaXMubGlzdC5wdXNoKHRoZU5ldXJvbik7XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFxuXHRcdC8vIGFjdGl2YXRlcyBhbGwgdGhlIG5ldXJvbnMgaW4gdGhlIGxheWVyXG5cdFx0YWN0aXZhdGUoaW5wdXQ/OiBTeW5hcHRpYy5JTnVtZXJpY0FycmF5KSA6IEZsb2F0NjRBcnJheSB7XG5cblx0XHRcdGlmKHRoaXMuY3VycmVudEFjdGl2YXRpb24ubGVuZ3RoICE9IHRoaXMubGlzdC5sZW5ndGgpXG5cdFx0XHRcdHRoaXMuY3VycmVudEFjdGl2YXRpb24gPSBuZXcgRmxvYXQ2NEFycmF5KHRoaXMubGlzdC5sZW5ndGgpO1xuXG5cdFx0XHR2YXIgYWN0aXZhdGlvbkluZGV4ID0gMDtcblxuXHRcdFx0aWYgKHR5cGVvZiBpbnB1dCAhPSAndW5kZWZpbmVkJykge1xuXHRcdFx0XHRpZiAoaW5wdXQubGVuZ3RoICE9IHRoaXMuc2l6ZSlcblx0XHRcdFx0XHR0aHJvdyBcIklOUFVUIHNpemUgYW5kIExBWUVSIHNpemUgbXVzdCBiZSB0aGUgc2FtZSB0byBhY3RpdmF0ZSFcIjtcblxuXHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmxpc3QpIHtcblx0XHRcdFx0XHR0aGlzLmN1cnJlbnRBY3RpdmF0aW9uW2FjdGl2YXRpb25JbmRleCsrXSA9IHRoaXMubGlzdFtpZF0uYWN0aXZhdGUoaW5wdXRbaWRdKTtcblx0XHRcdFx0fVxuXHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5saXN0KSB7XG5cdFx0XHRcdFx0dGhpcy5jdXJyZW50QWN0aXZhdGlvblthY3RpdmF0aW9uSW5kZXgrK10gPSB0aGlzLmxpc3RbaWRdLmFjdGl2YXRlKCk7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHRcdFxuXHRcdFx0cmV0dXJuIHRoaXMuY3VycmVudEFjdGl2YXRpb247XG5cdFx0fVxuXG5cdFx0Ly8gcHJvcGFnYXRlcyB0aGUgZXJyb3Igb24gYWxsIHRoZSBuZXVyb25zIG9mIHRoZSBsYXllclxuXHRcdHByb3BhZ2F0ZShyYXRlOiBudW1iZXIsIHRhcmdldD8gOiBTeW5hcHRpYy5JTnVtZXJpY0FycmF5KSB7XG5cdFx0XHRpZiAodHlwZW9mIHRhcmdldCAhPSAndW5kZWZpbmVkJykge1xuXHRcdFx0XHRpZiAodGFyZ2V0Lmxlbmd0aCAhPSB0aGlzLnNpemUpXG5cdFx0XHRcdFx0dGhyb3cgXCJUQVJHRVQgc2l6ZSBhbmQgTEFZRVIgc2l6ZSBtdXN0IGJlIHRoZSBzYW1lIHRvIHByb3BhZ2F0ZSFcIjtcblxuXHRcdFx0XHRmb3IgKHZhciBpZCA9IHRoaXMubGlzdC5sZW5ndGggLSAxOyBpZCA+PSAwOyBpZC0tKSB7XG5cdFx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubGlzdFtpZF07XG5cdFx0XHRcdFx0bmV1cm9uLnByb3BhZ2F0ZShyYXRlLCB0YXJnZXRbaWRdKTtcblx0XHRcdFx0fVxuXHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0Zm9yICh2YXIgaWQgPSB0aGlzLmxpc3QubGVuZ3RoIC0gMTsgaWQgPj0gMDsgaWQtLSkge1xuXHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaWRdO1xuXHRcdFx0XHRcdG5ldXJvbi5wcm9wYWdhdGUocmF0ZSk7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHR9XG5cblx0XHQvLyBwcm9qZWN0cyBhIGNvbm5lY3Rpb24gZnJvbSB0aGlzIGxheWVyIHRvIGFub3RoZXIgb25lXG5cdFx0cHJvamVjdChsYXllciA6IG5ldHdvcmsuTmV0d29yayB8IExheWVyLCB0eXBlPywgd2VpZ2h0cz8gOiBTeW5hcHRpYy5JTnVtZXJpY0FycmF5KSB7XG5cblx0XHRcdGlmIChsYXllciBpbnN0YW5jZW9mIG5ldHdvcmsuTmV0d29yaylcblx0XHRcdFx0bGF5ZXIgPSAoPG5ldHdvcmsuTmV0d29yaz5sYXllcikubGF5ZXJzLmlucHV0O1xuXG5cdFx0XHRpZiAobGF5ZXIgaW5zdGFuY2VvZiBMYXllcikge1xuXHRcdFx0XHRpZiAoIXRoaXMuY29ubmVjdGVkKGxheWVyKSlcblx0XHRcdFx0XHRyZXR1cm4gbmV3IExheWVyLkxheWVyQ29ubmVjdGlvbih0aGlzLCBsYXllciwgdHlwZSwgd2VpZ2h0cyk7XG5cdFx0XHR9IGVsc2Vcblx0XHRcdFx0dGhyb3cgXCJJbnZhbGlkIGFyZ3VtZW50LCB5b3UgY2FuIG9ubHkgcHJvamVjdCBjb25uZWN0aW9ucyB0byBMQVlFUlMgYW5kIE5FVFdPUktTIVwiO1xuXG5cblx0XHR9XG5cblx0XHQvLyBnYXRlcyBhIGNvbm5lY3Rpb24gYmV0d2VubiB0d28gbGF5ZXJzXG5cdFx0Z2F0ZShjb25uZWN0aW9uLCB0eXBlKSB7XG5cblx0XHRcdGlmICh0eXBlID09IExheWVyLmdhdGVUeXBlLklOUFVUKSB7XG5cdFx0XHRcdGlmIChjb25uZWN0aW9uLnRvLnNpemUgIT0gdGhpcy5zaXplKVxuXHRcdFx0XHRcdHRocm93IFwiR0FURVIgbGF5ZXIgYW5kIENPTk5FQ1RJT04uVE8gbGF5ZXIgbXVzdCBiZSB0aGUgc2FtZSBzaXplIGluIG9yZGVyIHRvIGdhdGUhXCI7XG5cblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gY29ubmVjdGlvbi50by5saXN0KSB7XG5cdFx0XHRcdFx0dmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24udG8ubGlzdFtpZF07XG5cdFx0XHRcdFx0dmFyIGdhdGVyID0gdGhpcy5saXN0W2lkXTtcblx0XHRcdFx0XHRmb3IgKHZhciBpbnB1dCBpbiBuZXVyb24uY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHRcdFx0XHR2YXIgZ2F0ZWQgPSBuZXVyb24uY29ubmVjdGlvbnMuaW5wdXRzW2lucHV0XTtcblx0XHRcdFx0XHRcdGlmIChnYXRlZC5JRCBpbiBjb25uZWN0aW9uLmNvbm5lY3Rpb25zKVxuXHRcdFx0XHRcdFx0XHRnYXRlci5nYXRlKGdhdGVkKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdH0gZWxzZSBpZiAodHlwZSA9PSBMYXllci5nYXRlVHlwZS5PVVRQVVQpIHtcblx0XHRcdFx0aWYgKGNvbm5lY3Rpb24uZnJvbS5zaXplICE9IHRoaXMuc2l6ZSlcblx0XHRcdFx0XHR0aHJvdyBcIkdBVEVSIGxheWVyIGFuZCBDT05ORUNUSU9OLkZST00gbGF5ZXIgbXVzdCBiZSB0aGUgc2FtZSBzaXplIGluIG9yZGVyIHRvIGdhdGUhXCI7XG5cblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gY29ubmVjdGlvbi5mcm9tLmxpc3QpIHtcblx0XHRcdFx0XHR2YXIgbmV1cm9uID0gY29ubmVjdGlvbi5mcm9tLmxpc3RbaWRdO1xuXHRcdFx0XHRcdHZhciBnYXRlciA9IHRoaXMubGlzdFtpZF07XG5cdFx0XHRcdFx0Zm9yICh2YXIgcHJvamVjdGVkIGluIG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpIHtcblx0XHRcdFx0XHRcdHZhciBnYXRlZCA9IG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbcHJvamVjdGVkXTtcblx0XHRcdFx0XHRcdGlmIChnYXRlZC5JRCBpbiBjb25uZWN0aW9uLmNvbm5lY3Rpb25zKVxuXHRcdFx0XHRcdFx0XHRnYXRlci5nYXRlKGdhdGVkKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdH0gZWxzZSBpZiAodHlwZSA9PSBMYXllci5nYXRlVHlwZS5PTkVfVE9fT05FKSB7XG5cdFx0XHRcdGlmIChjb25uZWN0aW9uLnNpemUgIT0gdGhpcy5zaXplKVxuXHRcdFx0XHRcdHRocm93IFwiVGhlIG51bWJlciBvZiBHQVRFUiBVTklUUyBtdXN0IGJlIHRoZSBzYW1lIGFzIHRoZSBudW1iZXIgb2YgQ09OTkVDVElPTlMgdG8gZ2F0ZSFcIjtcblxuXHRcdFx0XHRmb3IgKHZhciBpZCBpbiBjb25uZWN0aW9uLmxpc3QpIHtcblx0XHRcdFx0XHR2YXIgZ2F0ZXIgPSB0aGlzLmxpc3RbaWRdO1xuXHRcdFx0XHRcdHZhciBnYXRlZCA9IGNvbm5lY3Rpb24ubGlzdFtpZF07XG5cdFx0XHRcdFx0Z2F0ZXIuZ2F0ZShnYXRlZCk7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHRcdGNvbm5lY3Rpb24uZ2F0ZWRmcm9tLnB1c2goeyBsYXllcjogdGhpcywgdHlwZTogdHlwZSB9KTtcblx0XHR9XG5cblx0XHQvLyB0cnVlIG9yIGZhbHNlIHdoZXRoZXIgdGhlIHdob2xlIGxheWVyIGlzIHNlbGYtY29ubmVjdGVkIG9yIG5vdFxuXHRcdHNlbGZjb25uZWN0ZWQoKTogYm9vbGVhbiB7XG5cblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMubGlzdCkge1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5saXN0W2lkXTtcblx0XHRcdFx0aWYgKCFuZXVyb24uc2VsZmNvbm5lY3RlZCgpKVxuXHRcdFx0XHRcdHJldHVybiBmYWxzZTtcblx0XHRcdH1cblx0XHRcdHJldHVybiB0cnVlO1xuXHRcdH1cblxuXHRcdC8vIHRydWUgb2YgZmFsc2Ugd2hldGhlciB0aGUgbGF5ZXIgaXMgY29ubmVjdGVkIHRvIGFub3RoZXIgbGF5ZXIgKHBhcmFtZXRlcikgb3Igbm90XG5cdFx0Y29ubmVjdGVkKGxheWVyKSB7XG5cdFx0XHQvLyBDaGVjayBpZiBBTEwgdG8gQUxMIGNvbm5lY3Rpb25cblx0XHRcdHZhciBjb25uZWN0aW9ucyA9IDA7XG5cdFx0XHRmb3IgKHZhciBoZXJlIGluIHRoaXMubGlzdCkge1xuXHRcdFx0XHRmb3IgKHZhciB0aGVyZSBpbiBsYXllci5saXN0KSB7XG5cdFx0XHRcdFx0dmFyIGZyb20gPSB0aGlzLmxpc3RbaGVyZV07XG5cdFx0XHRcdFx0dmFyIHRvID0gbGF5ZXIubGlzdFt0aGVyZV07XG5cdFx0XHRcdFx0dmFyIGNvbm5lY3RlZCA9IGZyb20uY29ubmVjdGVkKHRvKTtcblx0XHRcdFx0XHRpZiAoY29ubmVjdGVkICYmIGNvbm5lY3RlZC50eXBlID09ICdwcm9qZWN0ZWQnKVxuXHRcdFx0XHRcdFx0Y29ubmVjdGlvbnMrKztcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdFx0aWYgKGNvbm5lY3Rpb25zID09IHRoaXMuc2l6ZSAqIGxheWVyLnNpemUpXG5cdFx0XHRcdHJldHVybiBMYXllci5jb25uZWN0aW9uVHlwZS5BTExfVE9fQUxMO1xuXG5cdFx0XHQvLyBDaGVjayBpZiBPTkUgdG8gT05FIGNvbm5lY3Rpb25cblx0XHRcdGNvbm5lY3Rpb25zID0gMDtcblx0XHRcdGZvciAodmFyIG5ldXJvbiBpbiB0aGlzLmxpc3QpIHtcblx0XHRcdFx0dmFyIGZyb20gPSB0aGlzLmxpc3RbbmV1cm9uXTtcblx0XHRcdFx0dmFyIHRvID0gbGF5ZXIubGlzdFtuZXVyb25dO1xuXHRcdFx0XHR2YXIgY29ubmVjdGVkID0gZnJvbS5jb25uZWN0ZWQodG8pO1xuXHRcdFx0XHRpZiAoY29ubmVjdGVkICYmIGNvbm5lY3RlZC50eXBlID09ICdwcm9qZWN0ZWQnKVxuXHRcdFx0XHRcdGNvbm5lY3Rpb25zKys7XG5cdFx0XHR9XG5cdFx0XHRpZiAoY29ubmVjdGlvbnMgPT0gdGhpcy5zaXplKVxuXHRcdFx0XHRyZXR1cm4gTGF5ZXIuY29ubmVjdGlvblR5cGUuT05FX1RPX09ORTtcblx0XHR9XG5cblx0XHQvLyBjbGVhcnMgYWxsIHRoZSBuZXVvcm5zIGluIHRoZSBsYXllclxuXHRcdGNsZWFyKCkge1xuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5saXN0KSB7XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaWRdO1xuXHRcdFx0XHRuZXVyb24uY2xlYXIoKTtcblx0XHRcdH1cblx0XHR9XG5cblx0XHQvLyByZXNldHMgYWxsIHRoZSBuZXVyb25zIGluIHRoZSBsYXllclxuXHRcdHJlc2V0KCkge1xuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5saXN0KSB7XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaWRdO1xuXHRcdFx0XHRuZXVyb24ucmVzZXQoKTtcblx0XHRcdH1cblx0XHR9XG5cblx0XHQvLyByZXR1cm5zIGFsbCB0aGUgbmV1cm9ucyBpbiB0aGUgbGF5ZXIgKGFycmF5KVxuXHRcdG5ldXJvbnMoKSA6IG5ldXJvbi5OZXVyb25bXSB7XG5cdFx0XHRyZXR1cm4gdGhpcy5saXN0O1xuXHRcdH1cblxuXHRcdC8vIGFkZHMgYSBuZXVyb24gdG8gdGhlIGxheWVyXG5cdFx0YWRkKG5ldXJvbikge1xuXHRcdFx0bmV1cm9uID0gbmV1cm9uIHx8IG5ldyBuZXVyb24uTmV1cm9uKCk7XG5cdFx0XHR0aGlzLm5ldXJvbnNbbmV1cm9uLklEXSA9IG5ldXJvbjtcblx0XHRcdHRoaXMubGlzdC5wdXNoKG5ldXJvbik7XG5cdFx0XHR0aGlzLnNpemUrKztcblx0XHR9XG5cblx0XHRzZXQob3B0aW9ucykge1xuXHRcdFx0b3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG5cblx0XHRcdGZvciAodmFyIGkgaW4gdGhpcy5saXN0KSB7XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaV07XG5cdFx0XHRcdGlmIChvcHRpb25zLmxhYmVsKVxuXHRcdFx0XHRcdG5ldXJvbi5sYWJlbCA9IG9wdGlvbnMubGFiZWwgKyAnXycgKyBuZXVyb24uSUQ7XG5cdFx0XHRcdGlmIChvcHRpb25zLnNxdWFzaClcblx0XHRcdFx0XHRuZXVyb24uc3F1YXNoID0gb3B0aW9ucy5zcXVhc2g7XG5cdFx0XHRcdGlmIChvcHRpb25zLmJpYXMpXG5cdFx0XHRcdFx0bmV1cm9uLmJpYXMgPSBvcHRpb25zLmJpYXM7XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gdGhpcztcblx0XHR9XG5cdH1cblxuXG5leHBvcnQgbW9kdWxlIExheWVyIHtcblx0ZXhwb3J0IHZhciBsYXllclF0eSA9IDA7XG5cdGV4cG9ydCBmdW5jdGlvbiB1aWQoKSB7XG5cdFx0cmV0dXJuIGxheWVyUXR5Kys7XG5cdH1cblx0XG5cdC8vIHR5cGVzIG9mIGNvbm5lY3Rpb25zXG5cdGV4cG9ydCB2YXIgY29ubmVjdGlvblR5cGUgPSB7XG5cdFx0QUxMX1RPX0FMTDogXCJBTEwgVE8gQUxMXCIsXG5cdFx0T05FX1RPX09ORTogXCJPTkUgVE8gT05FXCIsXG5cdFx0QUxMX1RPX0VMU0U6IFwiQUxMIFRPIEVMU0VcIlxuXHR9O1xuXG5cdC8vIHR5cGVzIG9mIGdhdGVzXG5cdGV4cG9ydCB2YXIgZ2F0ZVR5cGUgPSB7XG5cdFx0SU5QVVQ6IFwiSU5QVVRcIixcblx0XHRPVVRQVVQ6IFwiT1VUUFVUXCIsXG5cdFx0T05FX1RPX09ORTogXCJPTkUgVE8gT05FXCJcblx0fTtcblxuXHQvLyByZXByZXNlbnRzIGEgY29ubmVjdGlvbiBmcm9tIG9uZSBsYXllciB0byBhbm90aGVyLCBhbmQga2VlcHMgdHJhY2sgb2YgaXRzIHdlaWdodCBhbmQgZ2FpblxuXHRleHBvcnQgY2xhc3MgTGF5ZXJDb25uZWN0aW9uIHtcblx0XHRJRCA9IHVpZCgpO1xuXHRcdGZyb206IExheWVyO1xuXHRcdHRvOiBMYXllcjtcblx0XHRzZWxmY29ubmVjdGlvbiA6IGJvb2xlYW4gPSBmYWxzZTtcblx0XHR0eXBlOiBzdHJpbmc7XG5cdFx0Y29ubmVjdGlvbnM6IFN5bmFwdGljLkRpY3Rpb25hcnk8bmV1cm9uLk5ldXJvbi5Db25uZWN0aW9uPjtcblx0XHRsaXN0OiBuZXVyb24uTmV1cm9uLkNvbm5lY3Rpb25bXTtcblx0XHRzaXplID0gMDtcblx0XHRnYXRlZGZyb20gPSBbXTtcblxuXHRcdGNvbnN0cnVjdG9yKGZyb21MYXllciwgdG9MYXllciwgdHlwZSwgd2VpZ2h0cykge1xuXHRcdFx0dGhpcy5mcm9tID0gZnJvbUxheWVyO1xuXHRcdFx0dGhpcy50byA9IHRvTGF5ZXI7XG5cdFx0XHR0aGlzLnNlbGZjb25uZWN0aW9uID0gdG9MYXllciA9PSBmcm9tTGF5ZXI7XG5cdFx0XHR0aGlzLnR5cGUgPSB0eXBlO1xuXHRcdFx0dGhpcy5jb25uZWN0aW9ucyA9IHt9O1xuXHRcdFx0dGhpcy5saXN0ID0gW107XG5cdFx0XHR0aGlzLnNpemUgPSAwO1xuXHRcdFx0dGhpcy5nYXRlZGZyb20gPSBbXTtcblxuXG5cdFx0XHRpZiAodHlwZW9mIHRoaXMudHlwZSA9PSAndW5kZWZpbmVkJykge1xuXHRcdFx0XHRpZiAoZnJvbUxheWVyID09IHRvTGF5ZXIpXG5cdFx0XHRcdFx0dGhpcy50eXBlID0gTGF5ZXIuY29ubmVjdGlvblR5cGUuT05FX1RPX09ORTtcblx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdHRoaXMudHlwZSA9IExheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19BTEw7XG5cdFx0XHR9XG5cblx0XHRcdGlmICh0aGlzLnR5cGUgPT0gTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTCB8fFxuXHRcdFx0XHR0aGlzLnR5cGUgPT0gTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0VMU0UpIHtcblx0XHRcdFx0Zm9yICh2YXIgaGVyZSBpbiB0aGlzLmZyb20ubGlzdCkge1xuXHRcdFx0XHRcdGZvciAodmFyIHRoZXJlIGluIHRoaXMudG8ubGlzdCkge1xuXHRcdFx0XHRcdFx0dmFyIGZyb20gPSB0aGlzLmZyb20ubGlzdFtoZXJlXTtcblx0XHRcdFx0XHRcdHZhciB0byA9IHRoaXMudG8ubGlzdFt0aGVyZV07XG5cdFx0XHRcdFx0XHRpZiAodGhpcy50eXBlID09IExheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19FTFNFICYmIGZyb20gPT0gdG8pXG5cdFx0XHRcdFx0XHRcdGNvbnRpbnVlO1xuXHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBmcm9tLnByb2plY3QodG8sIHdlaWdodHMpO1xuXG5cdFx0XHRcdFx0XHR0aGlzLmNvbm5lY3Rpb25zW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcblx0XHRcdFx0XHRcdHRoaXMuc2l6ZSA9IHRoaXMubGlzdC5wdXNoKGNvbm5lY3Rpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fVxuXHRcdFx0fSBlbHNlIGlmICh0aGlzLnR5cGUgPT0gTGF5ZXIuY29ubmVjdGlvblR5cGUuT05FX1RPX09ORSkge1xuXG5cdFx0XHRcdGZvciAodmFyIG5ldXJvbiBpbiB0aGlzLmZyb20ubGlzdCkge1xuXHRcdFx0XHRcdHZhciBmcm9tID0gdGhpcy5mcm9tLmxpc3RbbmV1cm9uXTtcblx0XHRcdFx0XHR2YXIgdG8gPSB0aGlzLnRvLmxpc3RbbmV1cm9uXTtcblx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IGZyb20ucHJvamVjdCh0bywgd2VpZ2h0cyk7XG5cblx0XHRcdFx0XHR0aGlzLmNvbm5lY3Rpb25zW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcblx0XHRcdFx0XHR0aGlzLnNpemUgPSB0aGlzLmxpc3QucHVzaChjb25uZWN0aW9uKTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXG5cdFx0XHRmcm9tTGF5ZXIuY29ubmVjdGVkdG8ucHVzaCh0aGlzKTtcblx0XHR9XG5cdH1cbn0iLCJpbXBvcnQgbGF5ZXIgPSByZXF1aXJlKCcuL2xheWVyJyk7XG5pbXBvcnQgU3F1YXNoID0gcmVxdWlyZSgnLi9zcXVhc2gnKTtcbmltcG9ydCBTeW5hcHRpYyA9IHJlcXVpcmUoJy4vc3luYXB0aWMnKTtcbmltcG9ydCBfbmV1cm9uID0gcmVxdWlyZSgnLi9uZXVyb24nKTtcblxuLyoqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIE5FVFdPUktcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cbmRlY2xhcmUgZnVuY3Rpb24gZXNjYXBlKGE6IHN0cmluZyk6IHN0cmluZztcblxuXG5leHBvcnQgY2xhc3MgTmV0d29yayB7XG5cdG9wdGltaXplZCA9IG51bGw7XG5cdGxheWVycyA9IHtcblx0XHRpbnB1dDogbnVsbCxcblx0XHRoaWRkZW46IHt9LFxuXHRcdG91dHB1dDogbnVsbFxuXHR9O1xuXHRjb25zdHJ1Y3RvcihsYXllcnMpIHtcblx0XHRpZiAodHlwZW9mIGxheWVycyAhPSAndW5kZWZpbmVkJykge1xuXHRcdFx0dGhpcy5sYXllcnMgPSBsYXllcnMgfHwge1xuXHRcdFx0XHRpbnB1dDogbnVsbCxcblx0XHRcdFx0aGlkZGVuOiB7fSxcblx0XHRcdFx0b3V0cHV0OiBudWxsXG5cdFx0XHR9O1xuXHRcdH1cblx0fVxuXG5cdC8vIGZlZWQtZm9yd2FyZCBhY3RpdmF0aW9uIG9mIGFsbCB0aGUgbGF5ZXJzIHRvIHByb2R1Y2UgYW4gb3VwdXRcblx0YWN0aXZhdGUoaW5wdXQgOiBTeW5hcHRpYy5JTnVtZXJpY0FycmF5KSB7XG5cblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQgPT09IGZhbHNlKSB7XG5cdFx0XHR0aGlzLmxheWVycy5pbnB1dC5hY3RpdmF0ZShpbnB1dCk7XG5cdFx0XHRmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pXG5cdFx0XHRcdHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl0uYWN0aXZhdGUoKTtcblx0XHRcdHJldHVybiB0aGlzLmxheWVycy5vdXRwdXQuYWN0aXZhdGUoKTtcblx0XHR9XG5cdFx0ZWxzZSB7XG5cdFx0XHRpZiAodGhpcy5vcHRpbWl6ZWQgPT0gbnVsbClcblx0XHRcdFx0dGhpcy5vcHRpbWl6ZSgpO1xuXHRcdFx0cmV0dXJuIHRoaXMub3B0aW1pemVkLmFjdGl2YXRlKGlucHV0KTtcblx0XHR9XG5cdH1cblxuXHQvLyBiYWNrLXByb3BhZ2F0ZSB0aGUgZXJyb3IgdGhydSB0aGUgbmV0d29ya1xuXHRwcm9wYWdhdGUocmF0ZTogbnVtYmVyLCB0YXJnZXQ/OiBTeW5hcHRpYy5JTnVtZXJpY0FycmF5KSB7XG5cblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQgPT09IGZhbHNlKSB7XG5cdFx0XHR0aGlzLmxheWVycy5vdXRwdXQucHJvcGFnYXRlKHJhdGUsIHRhcmdldCk7XG5cdFx0XHR2YXIgcmV2ZXJzZSA9IFtdO1xuXHRcdFx0Zm9yICh2YXIgbGF5ZXIgaW4gdGhpcy5sYXllcnMuaGlkZGVuKVxuXHRcdFx0XHRyZXZlcnNlLnB1c2godGhpcy5sYXllcnMuaGlkZGVuW2xheWVyXSk7XG5cdFx0XHRyZXZlcnNlLnJldmVyc2UoKTtcblx0XHRcdGZvciAodmFyIGxheWVyIGluIHJldmVyc2UpXG5cdFx0XHRcdHJldmVyc2VbbGF5ZXJdLnByb3BhZ2F0ZShyYXRlKTtcblx0XHR9XG5cdFx0ZWxzZSB7XG5cdFx0XHRpZiAodGhpcy5vcHRpbWl6ZWQgPT0gbnVsbClcblx0XHRcdFx0dGhpcy5vcHRpbWl6ZSgpO1xuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucHJvcGFnYXRlKHJhdGUsIHRhcmdldCk7XG5cdFx0fVxuXHR9XG5cblx0Ly8gcHJvamVjdCBhIGNvbm5lY3Rpb24gdG8gYW5vdGhlciB1bml0IChlaXRoZXIgYSBuZXR3b3JrIG9yIGEgbGF5ZXIpXG5cdHByb2plY3QodW5pdCwgdHlwZSwgd2VpZ2h0cykge1xuXG5cdFx0aWYgKHRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcblxuXHRcdGlmICh1bml0IGluc3RhbmNlb2YgTmV0d29yaylcblx0XHRcdHJldHVybiB0aGlzLmxheWVycy5vdXRwdXQucHJvamVjdCh1bml0LmxheWVycy5pbnB1dCwgdHlwZSwgd2VpZ2h0cyk7XG5cblx0XHRpZiAodW5pdCBpbnN0YW5jZW9mIGxheWVyLkxheWVyKVxuXHRcdFx0cmV0dXJuIHRoaXMubGF5ZXJzLm91dHB1dC5wcm9qZWN0KHVuaXQsIHR5cGUsIHdlaWdodHMpO1xuXG5cdFx0dGhyb3cgXCJJbnZhbGlkIGFyZ3VtZW50LCB5b3UgY2FuIG9ubHkgcHJvamVjdCBjb25uZWN0aW9ucyB0byBMQVlFUlMgYW5kIE5FVFdPUktTIVwiO1xuXHR9XG5cblx0Ly8gbGV0IHRoaXMgbmV0d29yayBnYXRlIGEgY29ubmVjdGlvblxuXHRnYXRlKGNvbm5lY3Rpb24sIHR5cGUpIHtcblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplZC5yZXNldCgpO1xuXHRcdHRoaXMubGF5ZXJzLm91dHB1dC5nYXRlKGNvbm5lY3Rpb24sIHR5cGUpO1xuXHR9XG5cblx0Ly8gY2xlYXIgYWxsIGVsZWdpYmlsaXR5IHRyYWNlcyBhbmQgZXh0ZW5kZWQgZWxlZ2liaWxpdHkgdHJhY2VzICh0aGUgbmV0d29yayBmb3JnZXRzIGl0cyBjb250ZXh0LCBidXQgbm90IHdoYXQgd2FzIHRyYWluZWQpXG5cdGNsZWFyKCkge1xuXG5cdFx0dGhpcy5yZXN0b3JlKCk7XG5cblx0XHR2YXIgaW5wdXRMYXllciA9IHRoaXMubGF5ZXJzLmlucHV0LFxuXHRcdFx0b3V0cHV0TGF5ZXIgPSB0aGlzLmxheWVycy5vdXRwdXQ7XG5cblx0XHRpbnB1dExheWVyLmNsZWFyKCk7XG5cdFx0Zm9yICh2YXIgbGF5ZXIgaW4gdGhpcy5sYXllcnMuaGlkZGVuKSB7XG5cdFx0XHR2YXIgaGlkZGVuTGF5ZXIgPSB0aGlzLmxheWVycy5oaWRkZW5bbGF5ZXJdO1xuXHRcdFx0aGlkZGVuTGF5ZXIuY2xlYXIoKTtcblx0XHR9XG5cdFx0b3V0cHV0TGF5ZXIuY2xlYXIoKTtcblxuXHRcdGlmICh0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cdH1cblxuXHQvLyByZXNldCBhbGwgd2VpZ2h0cyBhbmQgY2xlYXIgYWxsIHRyYWNlcyAoZW5kcyB1cCBsaWtlIGEgbmV3IG5ldHdvcmspXG5cdHJlc2V0KCkge1xuXG5cdFx0dGhpcy5yZXN0b3JlKCk7XG5cblx0XHR2YXIgaW5wdXRMYXllciA9IHRoaXMubGF5ZXJzLmlucHV0LFxuXHRcdFx0b3V0cHV0TGF5ZXIgPSB0aGlzLmxheWVycy5vdXRwdXQ7XG5cblx0XHRpbnB1dExheWVyLnJlc2V0KCk7XG5cdFx0Zm9yICh2YXIgbGF5ZXIgaW4gdGhpcy5sYXllcnMuaGlkZGVuKSB7XG5cdFx0XHR2YXIgaGlkZGVuTGF5ZXIgPSB0aGlzLmxheWVycy5oaWRkZW5bbGF5ZXJdO1xuXHRcdFx0aGlkZGVuTGF5ZXIucmVzZXQoKTtcblx0XHR9XG5cdFx0b3V0cHV0TGF5ZXIucmVzZXQoKTtcblxuXHRcdGlmICh0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cdH1cblxuXHQvLyBoYXJkY29kZXMgdGhlIGJlaGF2aW91ciBvZiB0aGUgd2hvbGUgbmV0d29yayBpbnRvIGEgc2luZ2xlIG9wdGltaXplZCBmdW5jdGlvblxuXHRvcHRpbWl6ZSgpIHtcblxuXHRcdHZhciB0aGF0ID0gdGhpcztcblx0XHR2YXIgb3B0aW1pemVkOiBTeW5hcHRpYy5JQ29tcGlsZWRQYXJhbWV0ZXJzID0ge307XG5cdFx0dmFyIG5ldXJvbnMgPSB0aGlzLm5ldXJvbnMoKTtcblxuXHRcdGZvciAodmFyIGkgaW4gbmV1cm9ucykge1xuXHRcdFx0dmFyIG5ldXJvbiA9IG5ldXJvbnNbaV0ubmV1cm9uO1xuXHRcdFx0dmFyIGxheWVyID0gbmV1cm9uc1tpXS5sYXllcjtcblx0XHRcdC8qXG5cdFx0XHRGSVhNRTogZG9lcyB0aGlzIHdvcmtlZCBvbmNlP1xuXHRcdFx0XG5cdFx0XHR3aGlsZSAobmV1cm9uLm5ldXJvbilcblx0XHRcdFx0bmV1cm9uID0gbmV1cm9uLm5ldXJvbjtcblx0XHRcdCovXG5cblx0XHRcdG9wdGltaXplZCA9IG5ldXJvbi5vcHRpbWl6ZShvcHRpbWl6ZWQsIGxheWVyKTtcblx0XHR9XG5cblx0XHRmb3IgKHZhciBpIGluIG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMpXG5cdFx0XHRvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzW2ldLnJldmVyc2UoKTtcblx0XHRvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzLnJldmVyc2UoKTtcblxuXHRcdHZhciBoYXJkY29kZSA9IFwiXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgRiA9IEZsb2F0NjRBcnJheSA/IG5ldyBGbG9hdDY0QXJyYXkoXCIgKyBvcHRpbWl6ZWQubWVtb3J5ICtcblx0XHRcIikgOiBbXTsgXCI7XG5cdFx0Zm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQudmFyaWFibGVzKVxuXHRcdFx0aGFyZGNvZGUgKz0gXCJGW1wiICsgb3B0aW1pemVkLnZhcmlhYmxlc1tpXS5pZCArIFwiXSA9IFwiICsgKG9wdGltaXplZC52YXJpYWJsZXNbXG5cdFx0XHRcdGldLnZhbHVlIHx8IDApICsgXCI7IFwiO1xuXHRcdGhhcmRjb2RlICs9IFwidmFyIGFjdGl2YXRlID0gZnVuY3Rpb24oaW5wdXQpe1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwiaW5mbHVlbmNlcyA9IFtdO1wiO1xuXHRcdGZvciAodmFyIGkgaW4gb3B0aW1pemVkLmlucHV0cylcblx0XHRcdGhhcmRjb2RlICs9IFwiRltcIiArIG9wdGltaXplZC5pbnB1dHNbaV0gKyBcIl0gPSBpbnB1dFtcIiArIGkgKyBcIl07IFwiO1xuXHRcdGZvciAodmFyIGN1cnJlbnRMYXllciBpbiBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXMpIHtcblx0XHRcdGlmIChvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXS5sZW5ndGggPiAwKSB7XG5cdFx0XHRcdGZvciAodmFyIGN1cnJlbnROZXVyb24gaW4gb3B0aW1pemVkLmFjdGl2YXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0pIHtcblx0XHRcdFx0XHRoYXJkY29kZSArPSBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXVtjdXJyZW50TmV1cm9uXS5qb2luKFwiIFwiKTtcblx0XHRcdFx0XHRoYXJkY29kZSArPSBvcHRpbWl6ZWQudHJhY2Vfc2VudGVuY2VzW2N1cnJlbnRMYXllcl1bY3VycmVudE5ldXJvbl0uam9pbihcIiBcIik7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHR9XG5cdFx0aGFyZGNvZGUgKz0gXCIgdmFyIG91dHB1dCA9IFtdOyBcIlxuXHRcdGZvciAodmFyIGkgaW4gb3B0aW1pemVkLm91dHB1dHMpXG5cdFx0XHRoYXJkY29kZSArPSBcIm91dHB1dFtcIiArIGkgKyBcIl0gPSBGW1wiICsgb3B0aW1pemVkLm91dHB1dHNbaV0gKyBcIl07IFwiO1xuXHRcdGhhcmRjb2RlICs9IFwicmV0dXJuIG91dHB1dDsgfTsgXCJcblx0XHRoYXJkY29kZSArPSBcInZhciBwcm9wYWdhdGUgPSBmdW5jdGlvbihyYXRlLCB0YXJnZXQpe1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwiRltcIiArIG9wdGltaXplZC52YXJpYWJsZXMucmF0ZS5pZCArIFwiXSA9IHJhdGU7IFwiO1xuXHRcdGZvciAodmFyIGkgaW4gb3B0aW1pemVkLnRhcmdldHMpXG5cdFx0XHRoYXJkY29kZSArPSBcIkZbXCIgKyBvcHRpbWl6ZWQudGFyZ2V0c1tpXSArIFwiXSA9IHRhcmdldFtcIiArIGkgKyBcIl07IFwiO1xuXHRcdGZvciAodmFyIGN1cnJlbnRMYXllciBpbiBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzKVxuXHRcdFx0Zm9yICh2YXIgY3VycmVudE5ldXJvbiBpbiBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0pXG5cdFx0XHRcdGhhcmRjb2RlICs9IG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXVtjdXJyZW50TmV1cm9uXS5qb2luKFwiIFwiKSArIFwiIFwiO1xuXHRcdGhhcmRjb2RlICs9IFwiIH07XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz1cblx0XHRcInZhciBvd25lcnNoaXAgPSBmdW5jdGlvbihtZW1vcnlCdWZmZXIpe1xcbkYgPSBtZW1vcnlCdWZmZXI7XFxudGhpcy5tZW1vcnkgPSBGO1xcbn07XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz1cblx0XHRcInJldHVybiB7XFxubWVtb3J5OiBGLFxcbmFjdGl2YXRlOiBhY3RpdmF0ZSxcXG5wcm9wYWdhdGU6IHByb3BhZ2F0ZSxcXG5vd25lcnNoaXA6IG93bmVyc2hpcFxcbn07XCI7XG5cdFx0aGFyZGNvZGUgPSBoYXJkY29kZS5zcGxpdChcIjtcIikuam9pbihcIjtcXG5cIik7XG5cblx0XHR2YXIgY29uc3RydWN0b3IgPSBuZXcgRnVuY3Rpb24oaGFyZGNvZGUpO1xuXG5cdFx0dmFyIG5ldHdvcmsgPSBjb25zdHJ1Y3RvcigpO1xuXG5cdFx0bmV0d29yay5kYXRhID0ge1xuXHRcdFx0dmFyaWFibGVzOiBvcHRpbWl6ZWQudmFyaWFibGVzLFxuXHRcdFx0YWN0aXZhdGU6IG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlcyxcblx0XHRcdHByb3BhZ2F0ZTogb3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlcyxcblx0XHRcdHRyYWNlOiBvcHRpbWl6ZWQudHJhY2Vfc2VudGVuY2VzLFxuXHRcdFx0aW5wdXRzOiBvcHRpbWl6ZWQuaW5wdXRzLFxuXHRcdFx0b3V0cHV0czogb3B0aW1pemVkLm91dHB1dHMsXG5cdFx0XHRjaGVja19hY3RpdmF0aW9uOiB0aGlzLmFjdGl2YXRlLFxuXHRcdFx0Y2hlY2tfcHJvcGFnYXRpb246IHRoaXMucHJvcGFnYXRlXG5cdFx0fVxuXG5cdFx0bmV0d29yay5yZXNldCA9IGZ1bmN0aW9uKCkge1xuXHRcdFx0aWYgKHRoYXQub3B0aW1pemVkKSB7XG5cdFx0XHRcdHRoYXQub3B0aW1pemVkID0gbnVsbDtcblx0XHRcdFx0dGhhdC5hY3RpdmF0ZSA9IG5ldHdvcmsuZGF0YS5jaGVja19hY3RpdmF0aW9uO1xuXHRcdFx0XHR0aGF0LnByb3BhZ2F0ZSA9IG5ldHdvcmsuZGF0YS5jaGVja19wcm9wYWdhdGlvbjtcblx0XHRcdH1cblx0XHR9XG5cblx0XHR0aGlzLm9wdGltaXplZCA9IG5ldHdvcms7XG5cdFx0dGhpcy5hY3RpdmF0ZSA9IG5ldHdvcmsuYWN0aXZhdGU7XG5cdFx0dGhpcy5wcm9wYWdhdGUgPSBuZXR3b3JrLnByb3BhZ2F0ZTtcblx0fVxuXG5cdC8vIHJlc3RvcmVzIGFsbCB0aGUgdmFsdWVzIGZyb20gdGhlIG9wdGltaXplZCBuZXR3b3JrIHRoZSB0aGVpciByZXNwZWN0aXZlIG9iamVjdHMgaW4gb3JkZXIgdG8gbWFuaXB1bGF0ZSB0aGUgbmV0d29ya1xuXHRyZXN0b3JlKCkge1xuXHRcdGlmICghdGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHRyZXR1cm47XG5cblx0XHR2YXIgb3B0aW1pemVkID0gdGhpcy5vcHRpbWl6ZWQ7XG5cblx0XHR2YXIgZ2V0VmFsdWUgPSBmdW5jdGlvbiguLi5hcmdzOiBhbnlbXSkge1xuXHRcdFx0dmFyIHVuaXQgPSBhcmdzLnNoaWZ0KCk7XG5cdFx0XHR2YXIgcHJvcCA9IGFyZ3MucG9wKCk7XG5cblx0XHRcdHZhciBpZCA9IHByb3AgKyAnXyc7XG5cdFx0XHRmb3IgKHZhciBwcm9wZXJ0eSBpbiBhcmdzKVxuXHRcdFx0XHRpZCArPSBhcmdzW3Byb3BlcnR5XSArICdfJztcblx0XHRcdGlkICs9IHVuaXQuSUQ7XG5cblx0XHRcdHZhciBtZW1vcnkgPSBvcHRpbWl6ZWQubWVtb3J5O1xuXHRcdFx0dmFyIHZhcmlhYmxlcyA9IG9wdGltaXplZC5kYXRhLnZhcmlhYmxlcztcblxuXHRcdFx0aWYgKGlkIGluIHZhcmlhYmxlcylcblx0XHRcdFx0cmV0dXJuIG1lbW9yeVt2YXJpYWJsZXNbaWRdLmlkXTtcblx0XHRcdHJldHVybiAwO1xuXHRcdH1cblxuXHRcdHZhciBsaXN0ID0gdGhpcy5uZXVyb25zKCk7XG5cblx0XHQvLyBsaW5rIGlkJ3MgdG8gcG9zaXRpb25zIGluIHRoZSBhcnJheVxuXHRcdHZhciBpZHMgPSB7fTtcblx0XHRmb3IgKHZhciBpIGluIGxpc3QpIHtcblx0XHRcdHZhciBuZXVyb24gPSBsaXN0W2ldLm5ldXJvbjtcblx0XHRcdC8qXG5cdFx0XHRGSVhNRTogZG9lcyB0aGlzIHdvcmtlZCBvbmNlP1xuXHRcdFx0XG5cdFx0XHR3aGlsZSAobmV1cm9uLm5ldXJvbilcblx0XHRcdFx0bmV1cm9uID0gbmV1cm9uLm5ldXJvbjtcblx0XHRcdCovXG5cblx0XHRcdG5ldXJvbi5zdGF0ZSA9IGdldFZhbHVlKG5ldXJvbiwgJ3N0YXRlJyk7XG5cdFx0XHRuZXVyb24ub2xkID0gZ2V0VmFsdWUobmV1cm9uLCAnb2xkJyk7XG5cdFx0XHRuZXVyb24uYWN0aXZhdGlvbiA9IGdldFZhbHVlKG5ldXJvbiwgJ2FjdGl2YXRpb24nKTtcblx0XHRcdG5ldXJvbi5iaWFzID0gZ2V0VmFsdWUobmV1cm9uLCAnYmlhcycpO1xuXG5cdFx0XHRmb3IgKHZhciBpbnB1dCBpbiBuZXVyb24udHJhY2UuZWxlZ2liaWxpdHkpXG5cdFx0XHRcdG5ldXJvbi50cmFjZS5lbGVnaWJpbGl0eVtpbnB1dF0gPSBnZXRWYWx1ZShuZXVyb24sICd0cmFjZScsXG5cdFx0XHRcdFx0J2VsZWdpYmlsaXR5JywgaW5wdXQpO1xuXG5cdFx0XHRmb3IgKHZhciBnYXRlZCBpbiBuZXVyb24udHJhY2UuZXh0ZW5kZWQpXG5cdFx0XHRcdGZvciAodmFyIGlucHV0IGluIG5ldXJvbi50cmFjZS5leHRlbmRlZFtnYXRlZF0pXG5cdFx0XHRcdFx0bmV1cm9uLnRyYWNlLmV4dGVuZGVkW2dhdGVkXVtpbnB1dF0gPSBnZXRWYWx1ZShuZXVyb24sICd0cmFjZScsXG5cdFx0XHRcdFx0XHQnZXh0ZW5kZWQnLCBnYXRlZCwgaW5wdXQpO1xuXHRcdH1cblxuXHRcdC8vIGdldCBjb25uZWN0aW9uc1xuXHRcdGZvciAodmFyIGkgaW4gbGlzdCkge1xuXHRcdFx0dmFyIG5ldXJvbiA9IGxpc3RbaV0ubmV1cm9uO1xuXHRcdFx0Lypcblx0XHRcdEZJWE1FOiBkb2VzIHRoaXMgd29ya2VkIG9uY2U/XG5cdFx0XHRcblx0XHRcdHdoaWxlIChuZXVyb24ubmV1cm9uKVxuXHRcdFx0XHRuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuXHRcdFx0Ki9cblxuXHRcdFx0Zm9yICh2YXIgaiBpbiBuZXVyb24uY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG5cdFx0XHRcdHZhciBjb25uZWN0aW9uID0gbmV1cm9uLmNvbm5lY3Rpb25zLnByb2plY3RlZFtqXTtcblx0XHRcdFx0Y29ubmVjdGlvbi53ZWlnaHQgPSBnZXRWYWx1ZShjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdGNvbm5lY3Rpb24uZ2FpbiA9IGdldFZhbHVlKGNvbm5lY3Rpb24sICdnYWluJyk7XG5cdFx0XHR9XG5cdFx0fVxuXHR9XG5cblx0Ly8gcmV0dXJucyBhbGwgdGhlIG5ldXJvbnMgaW4gdGhlIG5ldHdvcmtcblx0bmV1cm9ucygpOiBOZXR3b3JrLklOZXR3b3JrTmV1cm9uW10ge1xuXHRcdHZhciBuZXVyb25zOiBOZXR3b3JrLklOZXR3b3JrTmV1cm9uW10gPSBbXTtcblxuXHRcdHZhciBpbnB1dExheWVyID0gdGhpcy5sYXllcnMuaW5wdXQubmV1cm9ucygpLFxuXHRcdFx0b3V0cHV0TGF5ZXIgPSB0aGlzLmxheWVycy5vdXRwdXQubmV1cm9ucygpO1xuXG5cdFx0Zm9yICh2YXIgbmV1cm9uIGluIGlucHV0TGF5ZXIpXG5cdFx0XHRuZXVyb25zLnB1c2goe1xuXHRcdFx0XHRuZXVyb246IGlucHV0TGF5ZXJbbmV1cm9uXSxcblx0XHRcdFx0bGF5ZXI6ICdpbnB1dCdcblx0XHRcdH0pO1xuXG5cdFx0Zm9yICh2YXIgbGF5ZXIgaW4gdGhpcy5sYXllcnMuaGlkZGVuKSB7XG5cdFx0XHR2YXIgaGlkZGVuTGF5ZXIgPSB0aGlzLmxheWVycy5oaWRkZW5bbGF5ZXJdLm5ldXJvbnMoKTtcblx0XHRcdGZvciAodmFyIG5ldXJvbiBpbiBoaWRkZW5MYXllcilcblx0XHRcdFx0bmV1cm9ucy5wdXNoKHtcblx0XHRcdFx0XHRuZXVyb246IGhpZGRlbkxheWVyW25ldXJvbl0sXG5cdFx0XHRcdFx0bGF5ZXI6IGxheWVyXG5cdFx0XHRcdH0pO1xuXHRcdH1cblx0XHRmb3IgKHZhciBuZXVyb24gaW4gb3V0cHV0TGF5ZXIpXG5cdFx0XHRuZXVyb25zLnB1c2goe1xuXHRcdFx0XHRuZXVyb246IG91dHB1dExheWVyW25ldXJvbl0sXG5cdFx0XHRcdGxheWVyOiAnb3V0cHV0J1xuXHRcdFx0fSk7XG5cblx0XHRyZXR1cm4gbmV1cm9ucztcblx0fVxuXG5cdC8vIHJldHVybnMgbnVtYmVyIG9mIGlucHV0cyBvZiB0aGUgbmV0d29ya1xuXHRpbnB1dHMoKTogbnVtYmVyIHtcblx0XHRyZXR1cm4gdGhpcy5sYXllcnMuaW5wdXQuc2l6ZTtcblx0fVxuXG5cdC8vIHJldHVybnMgbnVtYmVyIG9mIG91dHB1dHMgb2YgaHRlIG5ldHdvcmtcblx0b3V0cHV0cygpOiBudW1iZXIge1xuXHRcdHJldHVybiB0aGlzLmxheWVycy5vdXRwdXQuc2l6ZTtcblx0fVxuXG5cdC8vIHNldHMgdGhlIGxheWVycyBvZiB0aGUgbmV0d29ya1xuXHRzZXQobGF5ZXJzKSB7XG5cblx0XHR0aGlzLmxheWVycyA9IGxheWVycztcblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplZC5yZXNldCgpO1xuXHR9XG5cblx0c2V0T3B0aW1pemUoYm9vbCkge1xuXHRcdHRoaXMucmVzdG9yZSgpO1xuXHRcdGlmICh0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cdFx0dGhpcy5vcHRpbWl6ZWQgPSBib29sID8gbnVsbCA6IGZhbHNlO1xuXHR9XG5cblx0Ly8gcmV0dXJucyBhIGpzb24gdGhhdCByZXByZXNlbnRzIGFsbCB0aGUgbmV1cm9ucyBhbmQgY29ubmVjdGlvbnMgb2YgdGhlIG5ldHdvcmtcblx0dG9KU09OKGlnbm9yZVRyYWNlcykge1xuXG5cdFx0dGhpcy5yZXN0b3JlKCk7XG5cblx0XHR2YXIgbGlzdCA9IHRoaXMubmV1cm9ucygpO1xuXHRcdHZhciBuZXVyb25zID0gW107XG5cdFx0dmFyIGNvbm5lY3Rpb25zID0gW107XG5cblx0XHQvLyBsaW5rIGlkJ3MgdG8gcG9zaXRpb25zIGluIHRoZSBhcnJheVxuXHRcdHZhciBpZHMgPSB7fTtcblx0XHRmb3IgKHZhciBpIGluIGxpc3QpIHtcblx0XHRcdHZhciBuZXVyb24gPSBsaXN0W2ldLm5ldXJvbjtcblx0XHRcdC8qXG5cdFx0XHRGSVhNRTogZG9lcyB0aGlzIHdvcmtlZCBvbmNlP1xuXHRcdFx0XG5cdFx0XHR3aGlsZSAobmV1cm9uLm5ldXJvbilcblx0XHRcdFx0bmV1cm9uID0gbmV1cm9uLm5ldXJvbjtcblx0XHRcdCovXG5cblx0XHRcdGlkc1tuZXVyb24uSURdID0gaTtcblxuXHRcdFx0dmFyIGNvcHkgPSB7XG5cdFx0XHRcdHRyYWNlOiB7XG5cdFx0XHRcdFx0ZWxlZ2liaWxpdHk6IHt9LFxuXHRcdFx0XHRcdGV4dGVuZGVkOiB7fVxuXHRcdFx0XHR9LFxuXHRcdFx0XHRzdGF0ZTogbmV1cm9uLnN0YXRlLFxuXHRcdFx0XHRvbGQ6IG5ldXJvbi5vbGQsXG5cdFx0XHRcdGFjdGl2YXRpb246IG5ldXJvbi5hY3RpdmF0aW9uLFxuXHRcdFx0XHRiaWFzOiBuZXVyb24uYmlhcyxcblx0XHRcdFx0bGF5ZXI6IGxpc3RbaV0ubGF5ZXIsXG5cdFx0XHRcdHNxdWFzaDogbnVsbFxuXHRcdFx0fTtcblxuXHRcdFx0Y29weS5zcXVhc2ggPSBuZXVyb24uc3F1YXNoID09IFNxdWFzaC5MT0dJU1RJQyA/IFwiTE9HSVNUSUNcIiA6XG5cdFx0XHRcdG5ldXJvbi5zcXVhc2ggPT0gU3F1YXNoLlRBTkggPyBcIlRBTkhcIiA6XG5cdFx0XHRcdFx0bmV1cm9uLnNxdWFzaCA9PSBTcXVhc2guSURFTlRJVFkgPyBcIklERU5USVRZXCIgOlxuXHRcdFx0XHRcdFx0bmV1cm9uLnNxdWFzaCA9PSBTcXVhc2guSExJTSA/IFwiSExJTVwiIDpcblx0XHRcdFx0XHRcdFx0bnVsbDtcblxuXHRcdFx0bmV1cm9ucy5wdXNoKGNvcHkpO1xuXHRcdH1cblxuXHRcdGlmICghaWdub3JlVHJhY2VzKVxuXHRcdFx0Zm9yICh2YXIgaSBpbiBuZXVyb25zKSB7XG5cdFx0XHRcdHZhciBjb3BpZWROZXVyb24gPSBuZXVyb25zW2ldO1xuXG5cdFx0XHRcdGZvciAodmFyIGlucHV0IGluIG5ldXJvbi50cmFjZS5lbGVnaWJpbGl0eSlcblx0XHRcdFx0XHRjb3BpZWROZXVyb24udHJhY2UuZWxlZ2liaWxpdHlbaW5wdXRdID0gbmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0XTtcblxuXHRcdFx0XHRmb3IgKHZhciBnYXRlZCBpbiBuZXVyb24udHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0XHRjb3BpZWROZXVyb24udHJhY2UuZXh0ZW5kZWRbZ2F0ZWRdID0ge307XG5cdFx0XHRcdFx0Zm9yICh2YXIgaW5wdXQgaW4gbmV1cm9uLnRyYWNlLmV4dGVuZGVkW2dhdGVkXSlcblx0XHRcdFx0XHRcdGNvcGllZE5ldXJvbi50cmFjZS5leHRlbmRlZFtpZHNbZ2F0ZWRdXVtpbnB1dF0gPSBuZXVyb24udHJhY2UuZXh0ZW5kZWRbXG5cdFx0XHRcdFx0XHRnYXRlZF1baW5wdXRdO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cblx0XHQvLyBnZXQgY29ubmVjdGlvbnNcblx0XHRmb3IgKHZhciBpIGluIGxpc3QpIHtcblx0XHRcdHZhciBuZXVyb24gPSBsaXN0W2ldLm5ldXJvbjtcblx0XHRcdFx0XG5cdFx0XHQvKlxuXHRcdFx0RklYTUU6IGRvZXMgdGhpcyB3b3JrZWQgb25jZT9cblx0XHRcdFxuXHRcdFx0d2hpbGUgKG5ldXJvbi5uZXVyb24pXG5cdFx0XHRcdG5ldXJvbiA9IG5ldXJvbi5uZXVyb247XG5cdFx0XHQqL1xuXG5cdFx0XHRmb3IgKHZhciBqIGluIG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpIHtcblx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBuZXVyb24uY29ubmVjdGlvbnMucHJvamVjdGVkW2pdO1xuXHRcdFx0XHRjb25uZWN0aW9ucy5wdXNoKHtcblx0XHRcdFx0XHRmcm9tOiBpZHNbY29ubmVjdGlvbi5mcm9tLklEXSxcblx0XHRcdFx0XHR0bzogaWRzW2Nvbm5lY3Rpb24udG8uSURdLFxuXHRcdFx0XHRcdHdlaWdodDogY29ubmVjdGlvbi53ZWlnaHQsXG5cdFx0XHRcdFx0Z2F0ZXI6IGNvbm5lY3Rpb24uZ2F0ZXIgPyBpZHNbY29ubmVjdGlvbi5nYXRlci5JRF0gOiBudWxsLFxuXHRcdFx0XHR9KTtcblx0XHRcdH1cblx0XHRcdGlmIChuZXVyb24uc2VsZmNvbm5lY3RlZCgpKVxuXHRcdFx0XHRjb25uZWN0aW9ucy5wdXNoKHtcblx0XHRcdFx0XHRmcm9tOiBpZHNbbmV1cm9uLklEXSxcblx0XHRcdFx0XHR0bzogaWRzW25ldXJvbi5JRF0sXG5cdFx0XHRcdFx0d2VpZ2h0OiBuZXVyb24uc2VsZmNvbm5lY3Rpb24ud2VpZ2h0LFxuXHRcdFx0XHRcdGdhdGVyOiBuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIgPyBpZHNbbmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyXG5cdFx0XHRcdFx0XHQuSURdIDogbnVsbCxcblx0XHRcdFx0fSk7XG5cdFx0fVxuXG5cdFx0cmV0dXJuIHtcblx0XHRcdG5ldXJvbnM6IG5ldXJvbnMsXG5cdFx0XHRjb25uZWN0aW9uczogY29ubmVjdGlvbnNcblx0XHR9XG5cdH1cbiAgXG5cdC8vIGV4cG9ydCB0aGUgdG9wb2xvZ3kgaW50byBkb3QgbGFuZ3VhZ2Ugd2hpY2ggY2FuIGJlIHZpc3VhbGl6ZWQgYXMgZ3JhcGhzIHVzaW5nIGRvdFxuXHQvKiBleGFtcGxlOiAuLi4gY29uc29sZS5sb2cobmV0LnRvRG90TGFuZygpKTtcblx0XHRcdFx0JCBub2RlIGV4YW1wbGUuanMgPiBleGFtcGxlLmRvdFxuXHRcdFx0XHQkIGRvdCBleGFtcGxlLmRvdCAtVHBuZyA+IG91dC5wbmdcblx0Ki9cblx0dG9Eb3QoZWRnZWNvbm5lY3Rpb24pIHtcblx0XHRpZiAoISB0eXBlb2YgZWRnZWNvbm5lY3Rpb24pXG5cdFx0XHRlZGdlY29ubmVjdGlvbiA9IGZhbHNlO1xuXHRcdHZhciBjb2RlID0gXCJkaWdyYXBoIG5uIHtcXG4gICAgcmFua2RpciA9IEJUXFxuXCI7XG5cdFx0dmFyIGxheWVycyA9IFt0aGlzLmxheWVycy5pbnB1dF0uY29uY2F0KHRoaXMubGF5ZXJzLmhpZGRlbiwgdGhpcy5sYXllcnMub3V0cHV0KTtcblx0XHRmb3IgKHZhciBsYXllciBpbiBsYXllcnMpIHtcblx0XHRcdGZvciAodmFyIHRvIGluIGxheWVyc1tsYXllcl0uY29ubmVjdGVkdG8pIHsgLy8gcHJvamVjdGlvbnNcblx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBsYXllcnNbbGF5ZXJdLmNvbm5lY3RlZHRvW3RvXTtcblx0XHRcdFx0dmFyIGxheWVydG8gPSBjb25uZWN0aW9uLnRvO1xuXHRcdFx0XHR2YXIgc2l6ZSA9IGNvbm5lY3Rpb24uc2l6ZTtcblx0XHRcdFx0dmFyIGxheWVySUQgPSBsYXllcnMuaW5kZXhPZihsYXllcnNbbGF5ZXJdKTtcblx0XHRcdFx0dmFyIGxheWVydG9JRCA9IGxheWVycy5pbmRleE9mKGxheWVydG8pO1xuXHRcdFx0XHQvKiBodHRwOi8vc3RhY2tvdmVyZmxvdy5jb20vcXVlc3Rpb25zLzI2ODQ1NTQwL2Nvbm5lY3QtZWRnZXMtd2l0aC1ncmFwaC1kb3RcbiogRE9UIGRvZXMgbm90IHN1cHBvcnQgZWRnZS10by1lZGdlIGNvbm5lY3Rpb25zXG4qIFRoaXMgd29ya2Fyb3VuZCBwcm9kdWNlcyBzb21ld2hhdCB3ZWlyZCBncmFwaHMgLi4uXG5cdFx0XHRcdCovXG5cdFx0XHRcdGlmIChlZGdlY29ubmVjdGlvbikge1xuXHRcdFx0XHRcdGlmIChjb25uZWN0aW9uLmdhdGVkZnJvbS5sZW5ndGgpIHtcblx0XHRcdFx0XHRcdHZhciBmYWtlTm9kZSA9IFwiZmFrZVwiICsgbGF5ZXJJRCArIFwiX1wiICsgbGF5ZXJ0b0lEO1xuXHRcdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGZha2VOb2RlICtcblx0XHRcdFx0XHRcdFwiIFtsYWJlbCA9IFxcXCJcXFwiLCBzaGFwZSA9IHBvaW50LCB3aWR0aCA9IDAuMDEsIGhlaWdodCA9IDAuMDFdXFxuXCI7XG5cdFx0XHRcdFx0XHRjb2RlICs9IFwiICAgIFwiICsgbGF5ZXJJRCArIFwiIC0+IFwiICsgZmFrZU5vZGUgKyBcIiBbbGFiZWwgPSBcIiArIHNpemUgKyBcIiwgYXJyb3doZWFkID0gbm9uZV1cXG5cIjtcblx0XHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBmYWtlTm9kZSArIFwiIC0+IFwiICsgbGF5ZXJ0b0lEICsgXCJcXG5cIjtcblx0XHRcdFx0XHR9IGVsc2Vcblx0XHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBsYXllcklEICsgXCIgLT4gXCIgKyBsYXllcnRvSUQgKyBcIiBbbGFiZWwgPSBcIiArIHNpemUgKyBcIl1cXG5cIjtcblx0XHRcdFx0XHRmb3IgKHZhciBmcm9tIGluIGNvbm5lY3Rpb24uZ2F0ZWRmcm9tKSB7IC8vIGdhdGluZ3Ncblx0XHRcdFx0XHRcdHZhciBsYXllcmZyb20gPSBjb25uZWN0aW9uLmdhdGVkZnJvbVtmcm9tXS5sYXllcjtcblx0XHRcdFx0XHRcdHZhciB0eXBlID0gY29ubmVjdGlvbi5nYXRlZGZyb21bZnJvbV0udHlwZTtcblx0XHRcdFx0XHRcdHZhciBsYXllcmZyb21JRCA9IGxheWVycy5pbmRleE9mKGxheWVyZnJvbSk7XG5cdFx0XHRcdFx0XHRjb2RlICs9IFwiICAgIFwiICsgbGF5ZXJmcm9tSUQgKyBcIiAtPiBcIiArIGZha2VOb2RlICsgXCIgW2NvbG9yID0gYmx1ZV1cXG5cIjtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGxheWVySUQgKyBcIiAtPiBcIiArIGxheWVydG9JRCArIFwiIFtsYWJlbCA9IFwiICsgc2l6ZSArIFwiXVxcblwiO1xuXHRcdFx0XHRcdGZvciAodmFyIGZyb20gaW4gY29ubmVjdGlvbi5nYXRlZGZyb20pIHsgLy8gZ2F0aW5nc1xuXHRcdFx0XHRcdFx0dmFyIGxheWVyZnJvbSA9IGNvbm5lY3Rpb24uZ2F0ZWRmcm9tW2Zyb21dLmxheWVyO1xuXHRcdFx0XHRcdFx0dmFyIHR5cGUgPSBjb25uZWN0aW9uLmdhdGVkZnJvbVtmcm9tXS50eXBlO1xuXHRcdFx0XHRcdFx0dmFyIGxheWVyZnJvbUlEID0gbGF5ZXJzLmluZGV4T2YobGF5ZXJmcm9tKTtcblx0XHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBsYXllcmZyb21JRCArIFwiIC0+IFwiICsgbGF5ZXJ0b0lEICsgXCIgW2NvbG9yID0gYmx1ZV1cXG5cIjtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHR9XG5cdFx0Y29kZSArPSBcIn1cXG5cIjtcblx0XHRyZXR1cm4ge1xuXHRcdFx0Y29kZTogY29kZSxcblx0XHRcdGxpbms6IFwiaHR0cHM6Ly9jaGFydC5nb29nbGVhcGlzLmNvbS9jaGFydD9jaGw9XCIgKyBlc2NhcGUoY29kZS5yZXBsYWNlKFwiLyAvZ1wiLCBcIitcIikpICsgXCImY2h0PWd2XCJcblx0XHR9XG5cdH1cblxuXHQvLyByZXR1cm5zIGEgZnVuY3Rpb24gdGhhdCB3b3JrcyBhcyB0aGUgYWN0aXZhdGlvbiBvZiB0aGUgbmV0d29yayBhbmQgY2FuIGJlIHVzZWQgd2l0aG91dCBkZXBlbmRpbmcgb24gdGhlIGxpYnJhcnlcblx0c3RhbmRhbG9uZSgpIHtcblx0XHRpZiAoIXRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZSgpO1xuXG5cdFx0dmFyIGRhdGEgPSB0aGlzLm9wdGltaXplZC5kYXRhO1xuXG5cdFx0Ly8gYnVpbGQgYWN0aXZhdGlvbiBmdW5jdGlvblxuXHRcdHZhciBhY3RpdmF0aW9uID0gXCJmdW5jdGlvbiAoaW5wdXQpIHtcXG5cIjtcblxuXHRcdC8vIGJ1aWxkIGlucHV0c1xuXHRcdGZvciAodmFyIGkgaW4gZGF0YS5pbnB1dHMpXG5cdFx0XHRhY3RpdmF0aW9uICs9IFwiRltcIiArIGRhdGEuaW5wdXRzW2ldICsgXCJdID0gaW5wdXRbXCIgKyBpICsgXCJdO1xcblwiO1xuXG5cdFx0Ly8gYnVpbGQgbmV0d29yayBhY3RpdmF0aW9uXG5cdFx0Zm9yICh2YXIgbmV1cm9uIGluIGRhdGEuYWN0aXZhdGUpIHsgLy8gc2hvdWxkbid0IHRoaXMgYmUgbGF5ZXI/XG5cdFx0XHRmb3IgKHZhciBzZW50ZW5jZSBpbiBkYXRhLmFjdGl2YXRlW25ldXJvbl0pXG5cdFx0XHRcdGFjdGl2YXRpb24gKz0gZGF0YS5hY3RpdmF0ZVtuZXVyb25dW3NlbnRlbmNlXSArIFwiXFxuXCI7XG5cdFx0fVxuXG5cdFx0Ly8gYnVpbGQgb3V0cHV0c1xuXHRcdGFjdGl2YXRpb24gKz0gXCJ2YXIgb3V0cHV0ID0gW107XFxuXCI7XG5cdFx0Zm9yICh2YXIgaSBpbiBkYXRhLm91dHB1dHMpXG5cdFx0XHRhY3RpdmF0aW9uICs9IFwib3V0cHV0W1wiICsgaSArIFwiXSA9IEZbXCIgKyBkYXRhLm91dHB1dHNbaV0gKyBcIl07XFxuXCI7XG5cdFx0YWN0aXZhdGlvbiArPSBcInJldHVybiBvdXRwdXQ7XFxufVwiO1xuXG5cdFx0Ly8gcmVmZXJlbmNlIGFsbCB0aGUgcG9zaXRpb25zIGluIG1lbW9yeVxuXHRcdHZhciBtZW1vcnkgPSBhY3RpdmF0aW9uLm1hdGNoKC9GXFxbKFxcZCspXFxdL2cpO1xuXHRcdHZhciBkaW1lbnNpb24gPSAwO1xuXHRcdHZhciBpZHMgPSB7fTtcblx0XHRmb3IgKHZhciBhZGRyZXNzIGluIG1lbW9yeSkge1xuXHRcdFx0dmFyIHRtcCA9IG1lbW9yeVthZGRyZXNzXS5tYXRjaCgvXFxkKy8pWzBdO1xuXHRcdFx0aWYgKCEodG1wIGluIGlkcykpIHtcblx0XHRcdFx0aWRzW3RtcF0gPSBkaW1lbnNpb24rKztcblx0XHRcdH1cblx0XHR9XG5cdFx0dmFyIGhhcmRjb2RlID0gXCJGID0ge1xcblwiO1xuXHRcdGZvciAodmFyIGkgaW4gaWRzKVxuXHRcdFx0aGFyZGNvZGUgKz0gaWRzW2ldICsgXCI6IFwiICsgdGhpcy5vcHRpbWl6ZWQubWVtb3J5W2ldICsgXCIsXFxuXCI7XG5cdFx0aGFyZGNvZGUgPSBoYXJkY29kZS5zdWJzdHJpbmcoMCwgaGFyZGNvZGUubGVuZ3RoIC0gMikgKyBcIlxcbn07XFxuXCI7XG5cdFx0aGFyZGNvZGUgPSBcInZhciBydW4gPSBcIiArIGFjdGl2YXRpb24ucmVwbGFjZSgvRlxcWyhcXGQrKV0vZywgZnVuY3Rpb24oXG5cdFx0XHRpbmRleCkge1xuXHRcdFx0cmV0dXJuICdGWycgKyBpZHNbaW5kZXgubWF0Y2goL1xcZCsvKVswXV0gKyAnXSdcblx0XHR9KS5yZXBsYWNlKFwie1xcblwiLCBcIntcXG5cIiArIGhhcmRjb2RlICsgXCJcIikgKyBcIjtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcInJldHVybiBydW5cIjtcblxuXHRcdC8vIHJldHVybiBzdGFuZGFsb25lIGZ1bmN0aW9uXG5cdFx0cmV0dXJuIG5ldyBGdW5jdGlvbihoYXJkY29kZSkoKTtcblx0fVxuXG5cdHdvcmtlcigpIHtcblx0XHRpZiAoIXRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZSgpO1xuXG5cdFx0dmFyIGhhcmRjb2RlID0gXCJ2YXIgaW5wdXRzID0gXCIgKyB0aGlzLm9wdGltaXplZC5kYXRhLmlucHV0cy5sZW5ndGggK1xuXHRcdFx0XCI7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgb3V0cHV0cyA9IFwiICsgdGhpcy5vcHRpbWl6ZWQuZGF0YS5vdXRwdXRzLmxlbmd0aCArXG5cdFx0XCI7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgRiA9IG51bGw7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgYWN0aXZhdGUgPSBcIiArIHRoaXMub3B0aW1pemVkLmFjdGl2YXRlLnRvU3RyaW5nKCkgK1xuXHRcdFwiO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwidmFyIHByb3BhZ2F0ZSA9IFwiICsgdGhpcy5vcHRpbWl6ZWQucHJvcGFnYXRlLnRvU3RyaW5nKCkgK1xuXHRcdFwiO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwib25tZXNzYWdlID0gZnVuY3Rpb24oZSl7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJGID0gZS5kYXRhLm1lbW9yeUJ1ZmZlcjtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcImlmIChlLmRhdGEuYWN0aW9uID09ICdhY3RpdmF0ZScpe1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwiaWYgKGUuZGF0YS5pbnB1dC5sZW5ndGggPT0gaW5wdXRzKXtcXG5cIjtcblx0XHRoYXJkY29kZSArPVxuXHRcdFwicG9zdE1lc3NhZ2UoIHsgYWN0aW9uOiAnYWN0aXZhdGUnLCBvdXRwdXQ6IGFjdGl2YXRlKGUuZGF0YS5pbnB1dCksIG1lbW9yeUJ1ZmZlcjogRiB9LCBbRi5idWZmZXJdKTtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcIn1cXG59XFxuZWxzZSBpZiAoZS5kYXRhLmFjdGlvbiA9PSAncHJvcGFnYXRlJyl7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJwcm9wYWdhdGUoZS5kYXRhLnJhdGUsIGUuZGF0YS50YXJnZXQpO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9XG5cdFx0XCJwb3N0TWVzc2FnZSh7IGFjdGlvbjogJ3Byb3BhZ2F0ZScsIG1lbW9yeUJ1ZmZlcjogRiB9LCBbRi5idWZmZXJdKTtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcIn1cXG59XFxuXCI7XG5cblx0XHR2YXIgYmxvYiA9IG5ldyBCbG9iKFtoYXJkY29kZV0pO1xuXHRcdHZhciBibG9iVVJMID0gKDxhbnk+d2luZG93KS5VUkwuY3JlYXRlT2JqZWN0VVJMKGJsb2IpO1xuXG5cdFx0cmV0dXJuIG5ldyBXb3JrZXIoYmxvYlVSTCk7XG5cdH1cblxuXHQvLyByZXR1cm5zIGEgY29weSBvZiB0aGUgbmV0d29ya1xuXHRjbG9uZShpZ25vcmVUcmFjZXMpIHtcblx0XHRyZXR1cm4gTmV0d29yay5mcm9tSlNPTih0aGlzLnRvSlNPTihpZ25vcmVUcmFjZXMpKTtcblx0fVxuXG5cdHN0YXRpYyBmcm9tSlNPTihqc29uKSB7XG5cblx0XHR2YXIgbmV1cm9ucyA9IFtdO1xuXG5cdFx0dmFyIGxheWVycyA9IHtcblx0XHRcdGlucHV0OiBuZXcgbGF5ZXIuTGF5ZXIoMCksXG5cdFx0XHRoaWRkZW46IFtdLFxuXHRcdFx0b3V0cHV0OiBuZXcgbGF5ZXIuTGF5ZXIoMClcblx0XHR9XG5cdFx0XG5cblx0XHRmb3IgKHZhciBpIGluIGpzb24ubmV1cm9ucykge1xuXHRcdFx0dmFyIGNvbmZpZyA9IGpzb24ubmV1cm9uc1tpXTtcblxuXHRcdFx0dmFyIG5ldXJvbiA9IG5ldyBfbmV1cm9uLk5ldXJvbigpO1xuXHRcdFx0bmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5ID0gY29uZmlnLnRyYWNlLmVsZWdpYmlsaXR5O1xuXHRcdFx0bmV1cm9uLnRyYWNlLmV4dGVuZGVkID0gY29uZmlnLnRyYWNlLmV4dGVuZGVkO1xuXHRcdFx0bmV1cm9uLnN0YXRlID0gY29uZmlnLnN0YXRlO1xuXHRcdFx0bmV1cm9uLm9sZCA9IGNvbmZpZy5vbGQ7XG5cdFx0XHRuZXVyb24uYWN0aXZhdGlvbiA9IGNvbmZpZy5hY3RpdmF0aW9uO1xuXHRcdFx0bmV1cm9uLmJpYXMgPSBjb25maWcuYmlhcztcblx0XHRcdG5ldXJvbi5zcXVhc2ggPSBjb25maWcuc3F1YXNoIGluIFNxdWFzaCA/IFNxdWFzaFtjb25maWcuc3F1YXNoXSA6XG5cdFx0XHRcdFNxdWFzaC5MT0dJU1RJQztcblx0XHRcdG5ldXJvbnMucHVzaChuZXVyb24pO1xuXG5cdFx0XHRpZiAoY29uZmlnLmxheWVyID09ICdpbnB1dCcpXG5cdFx0XHRcdGxheWVycy5pbnB1dC5hZGQobmV1cm9uKTtcblx0XHRcdGVsc2UgaWYgKGNvbmZpZy5sYXllciA9PSAnb3V0cHV0Jylcblx0XHRcdFx0bGF5ZXJzLm91dHB1dC5hZGQobmV1cm9uKTtcblx0XHRcdGVsc2Uge1xuXHRcdFx0XHRpZiAodHlwZW9mIGxheWVycy5oaWRkZW5bY29uZmlnLmxheWVyXSA9PSAndW5kZWZpbmVkJylcblx0XHRcdFx0XHRsYXllcnMuaGlkZGVuW2NvbmZpZy5sYXllcl0gPSBuZXcgbGF5ZXIuTGF5ZXIoMCk7XG5cdFx0XHRcdGxheWVycy5oaWRkZW5bY29uZmlnLmxheWVyXS5hZGQobmV1cm9uKTtcblx0XHRcdH1cblx0XHR9XG5cblx0XHRmb3IgKHZhciBpIGluIGpzb24uY29ubmVjdGlvbnMpIHtcblx0XHRcdHZhciBjb25maWcgPSBqc29uLmNvbm5lY3Rpb25zW2ldO1xuXHRcdFx0dmFyIGZyb20gPSBuZXVyb25zW2NvbmZpZy5mcm9tXTtcblx0XHRcdHZhciB0byA9IG5ldXJvbnNbY29uZmlnLnRvXTtcblx0XHRcdHZhciB3ZWlnaHQgPSBjb25maWcud2VpZ2h0XG5cdFx0XHR2YXIgZ2F0ZXIgPSBuZXVyb25zW2NvbmZpZy5nYXRlcl07XG5cblx0XHRcdHZhciBjb25uZWN0aW9uID0gZnJvbS5wcm9qZWN0KHRvLCB3ZWlnaHQpO1xuXHRcdFx0aWYgKGdhdGVyKVxuXHRcdFx0XHRnYXRlci5nYXRlKGNvbm5lY3Rpb24pO1xuXHRcdH1cblxuXHRcdHJldHVybiBuZXcgTmV0d29yayhsYXllcnMpO1xuXHR9XG59XG5cbmV4cG9ydCBtb2R1bGUgTmV0d29yayB7XG5cdGV4cG9ydCBpbnRlcmZhY2UgSU5ldHdvcmtOZXVyb24ge1xuXHRcdG5ldXJvbjogX25ldXJvbi5OZXVyb247XG5cdFx0bGF5ZXI6IHN0cmluZztcblx0fVxufSIsIi8vLyA8cmVmZXJlbmNlIHBhdGg9XCJzeW5hcHRpYy50c1wiIC8+XG5cbmltcG9ydCBTeW5hcHRpYyA9IHJlcXVpcmUoJy4vc3luYXB0aWMnKTtcbmltcG9ydCBTcXVhc2ggPSByZXF1aXJlKCcuL3NxdWFzaCcpO1xuXG4vKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIE5FVVJPTlxuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKi9cblxuLyogVFMgQ0hBTkdFUzpcblxuXHROb3cgTmV1cm9uLmNvbm5lY3RlZChuZXVyb24pIHJldHVybnMgbnVsbCBpbnN0ZWFkIG9mIGZhbHNlXG5cbiovXG5cbmV4cG9ydCBjbGFzcyBOZXVyb24ge1xuXHRJRCA9IE5ldXJvbi51aWQoKTtcblx0bGFiZWwgPSBudWxsO1xuXHRjb25uZWN0aW9uczogTmV1cm9uLklOZXVyb25Db25uZWN0aW9ucyA9IHtcblx0XHRpbnB1dHM6IHt9LFxuXHRcdHByb2plY3RlZDoge30sXG5cdFx0Z2F0ZWQ6IHt9XG5cdH07XG5cdGVycm9yID0ge1xuXHRcdHJlc3BvbnNpYmlsaXR5OiAwLFxuXHRcdHByb2plY3RlZDogMCxcblx0XHRnYXRlZDogMFxuXHR9O1xuXHR0cmFjZSA9IHtcblx0XHRlbGVnaWJpbGl0eToge30sXG5cdFx0ZXh0ZW5kZWQ6IHt9LFxuXHRcdGluZmx1ZW5jZXM6IHt9XG5cdH07XG5cdHN0YXRlID0gMDtcblx0b2xkID0gMDtcblx0YWN0aXZhdGlvbiA9IDA7XG5cdHNlbGZjb25uZWN0aW9uID0gbmV3IE5ldXJvbi5Db25uZWN0aW9uKHRoaXMsIHRoaXMsIDApOyAvLyB3ZWlnaHQgPSAwIC0+IG5vdCBjb25uZWN0ZWRcblx0c3F1YXNoID0gU3F1YXNoLkxPR0lTVElDO1xuXHRuZWlnaGJvb3JzID0ge307XG5cdGJpYXMgPSBNYXRoLnJhbmRvbSgpICogLjIgLSAuMTtcblx0ZGVyaXZhdGl2ZSA9IDA7XG4gIFxuXHQvLyBhY3RpdmF0ZSB0aGUgbmV1cm9uXG5cdGFjdGl2YXRlKGlucHV0PzogbnVtYmVyKSB7XG5cdFx0Ly8gYWN0aXZhdGlvbiBmcm9tIGVudmlyb21lbnQgKGZvciBpbnB1dCBuZXVyb25zKVxuXHRcdGlmICh0eXBlb2YgaW5wdXQgIT0gJ3VuZGVmaW5lZCcpIHtcblx0XHRcdHRoaXMuYWN0aXZhdGlvbiA9IGlucHV0O1xuXHRcdFx0dGhpcy5kZXJpdmF0aXZlID0gMDtcblx0XHRcdHRoaXMuYmlhcyA9IDA7XG5cdFx0XHRyZXR1cm4gdGhpcy5hY3RpdmF0aW9uO1xuXHRcdH1cblxuXHRcdC8vIG9sZCBzdGF0ZVxuXHRcdHRoaXMub2xkID0gdGhpcy5zdGF0ZTtcblxuXHRcdC8vIGVxLiAxNVxuXHRcdHRoaXMuc3RhdGUgPSB0aGlzLnNlbGZjb25uZWN0aW9uLmdhaW4gKiB0aGlzLnNlbGZjb25uZWN0aW9uLndlaWdodCAqXG5cdFx0dGhpcy5zdGF0ZSArIHRoaXMuYmlhcztcblxuXHRcdGZvciAodmFyIGkgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdHZhciB0aGVJbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2ldO1xuXHRcdFx0dGhpcy5zdGF0ZSArPSB0aGVJbnB1dC5mcm9tLmFjdGl2YXRpb24gKiB0aGVJbnB1dC53ZWlnaHQgKiB0aGVJbnB1dC5nYWluO1xuXHRcdH1cblxuXHRcdC8vIGVxLiAxNlxuXHRcdHRoaXMuYWN0aXZhdGlvbiA9IHRoaXMuc3F1YXNoKHRoaXMuc3RhdGUpO1xuXG5cdFx0Ly8gZicocylcblx0XHR0aGlzLmRlcml2YXRpdmUgPSB0aGlzLnNxdWFzaCh0aGlzLnN0YXRlLCB0cnVlKTtcblxuXHRcdC8vIHVwZGF0ZSB0cmFjZXNcblx0XHR2YXIgaW5mbHVlbmNlcyA9IFtdO1xuXHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdC8vIGV4dGVuZGVkIGVsZWdpYmlsaXR5IHRyYWNlXG5cdFx0XHR2YXIgeHRyYWNlID0gdGhpcy50cmFjZS5leHRlbmRlZFtpZF07XG5cdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblxuXHRcdFx0Ly8gaWYgZ2F0ZWQgbmV1cm9uJ3Mgc2VsZmNvbm5lY3Rpb24gaXMgZ2F0ZWQgYnkgdGhpcyB1bml0LCB0aGUgaW5mbHVlbmNlIGtlZXBzIHRyYWNrIG9mIHRoZSBuZXVyb24ncyBvbGQgc3RhdGVcblx0XHRcdHZhciBpbmZsdWVuY2UgPSBuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIgPT0gdGhpcyA/IG5ldXJvbi5vbGQgOiAwO1xuXG5cdFx0XHQvLyBpbmRleCBydW5zIG92ZXIgYWxsIHRoZSBpbmNvbWluZyBjb25uZWN0aW9ucyB0byB0aGUgZ2F0ZWQgbmV1cm9uIHRoYXQgYXJlIGdhdGVkIGJ5IHRoaXMgdW5pdFxuXHRcdFx0Zm9yICh2YXIgaW5jb21pbmcgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0pIHsgLy8gY2FwdHVyZXMgdGhlIGVmZmVjdCB0aGF0IGhhcyBhbiBpbnB1dCBjb25uZWN0aW9uIHRvIHRoaXMgdW5pdCwgb24gYSBuZXVyb24gdGhhdCBpcyBnYXRlZCBieSB0aGlzIHVuaXRcblx0XHRcdFx0aW5mbHVlbmNlICs9IHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdW2luY29taW5nXS53ZWlnaHQgKlxuXHRcdFx0XHR0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXVtpbmNvbWluZ10uZnJvbS5hY3RpdmF0aW9uO1xuXHRcdFx0fVxuXHRcdFx0aW5mbHVlbmNlc1tuZXVyb24uSURdID0gaW5mbHVlbmNlO1xuXHRcdH1cblxuXHRcdGZvciAodmFyIGkgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdHZhciB0aGVJbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2ldO1xuXG5cdFx0XHQvLyBlbGVnaWJpbGl0eSB0cmFjZSAtIEVxLiAxN1xuXHRcdFx0dGhpcy50cmFjZS5lbGVnaWJpbGl0eVt0aGVJbnB1dC5JRF0gPSB0aGlzLnNlbGZjb25uZWN0aW9uLmdhaW4gKiB0aGlzLnNlbGZjb25uZWN0aW9uXG5cdFx0XHQud2VpZ2h0ICogdGhpcy50cmFjZS5lbGVnaWJpbGl0eVt0aGVJbnB1dC5JRF0gKyB0aGVJbnB1dC5nYWluICogdGhlSW5wdXQuZnJvbVxuXHRcdFx0LmFjdGl2YXRpb247XG5cblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0Ly8gZXh0ZW5kZWQgZWxlZ2liaWxpdHkgdHJhY2Vcblx0XHRcdFx0dmFyIHh0cmFjZSA9IHRoaXMudHJhY2UuZXh0ZW5kZWRbaWRdO1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0dmFyIGluZmx1ZW5jZSA9IGluZmx1ZW5jZXNbbmV1cm9uLklEXTtcblxuXHRcdFx0XHQvLyBlcS4gMThcblx0XHRcdFx0eHRyYWNlW3RoZUlucHV0LklEXSA9IG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYWluICogbmV1cm9uLnNlbGZjb25uZWN0aW9uXG5cdFx0XHRcdC53ZWlnaHQgKiB4dHJhY2VbdGhlSW5wdXQuSURdICsgdGhpcy5kZXJpdmF0aXZlICogdGhpcy50cmFjZS5lbGVnaWJpbGl0eVtcblx0XHRcdFx0dGhlSW5wdXQuSURdICogaW5mbHVlbmNlO1xuXHRcdFx0fVxuXHRcdH1cblxuXHRcdC8vICB1cGRhdGUgZ2F0ZWQgY29ubmVjdGlvbidzIGdhaW5zXG5cdFx0Zm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zLmdhdGVkKSB7XG5cdFx0XHR0aGlzLmNvbm5lY3Rpb25zLmdhdGVkW2Nvbm5lY3Rpb25dLmdhaW4gPSB0aGlzLmFjdGl2YXRpb247XG5cdFx0fVxuXG5cdFx0cmV0dXJuIHRoaXMuYWN0aXZhdGlvbjtcblx0fVxuXG5cdC8vIGJhY2stcHJvcGFnYXRlIHRoZSBlcnJvclxuXHRwcm9wYWdhdGUocmF0ZTogbnVtYmVyLCB0YXJnZXQ/OiBudW1iZXIpIHtcblx0XHQvLyBlcnJvciBhY2N1bXVsYXRvclxuXHRcdHZhciBlcnJvciA9IDA7XG5cblx0XHQvLyB3aGV0aGVyIG9yIG5vdCB0aGlzIG5ldXJvbiBpcyBpbiB0aGUgb3V0cHV0IGxheWVyXG5cdFx0dmFyIGlzT3V0cHV0ID0gdHlwZW9mIHRhcmdldCAhPSAndW5kZWZpbmVkJyAmJiB0YXJnZXQgIT0gbnVsbDtcblxuXHRcdC8vIG91dHB1dCBuZXVyb25zIGdldCB0aGVpciBlcnJvciBmcm9tIHRoZSBlbnZpcm9tZW50XG5cdFx0aWYgKGlzT3V0cHV0KVxuXHRcdFx0dGhpcy5lcnJvci5yZXNwb25zaWJpbGl0eSA9IHRoaXMuZXJyb3IucHJvamVjdGVkID0gdGFyZ2V0IC0gdGhpcy5hY3RpdmF0aW9uOyAvLyBFcS4gMTBcbiAgICBcblx0XHRlbHNlIC8vIHRoZSByZXN0IG9mIHRoZSBuZXVyb24gY29tcHV0ZSB0aGVpciBlcnJvciByZXNwb25zaWJpbGl0aWVzIGJ5IGJhY2twcm9wYWdhdGlvblxuXHRcdHtcblx0XHRcdC8vIGVycm9yIHJlc3BvbnNpYmlsaXRpZXMgZnJvbSBhbGwgdGhlIGNvbm5lY3Rpb25zIHByb2plY3RlZCBmcm9tIHRoaXMgbmV1cm9uXG5cdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZCkge1xuXHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkW2lkXTtcblx0XHRcdFx0dmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24udG87XG5cdFx0XHRcdC8vIEVxLiAyMVxuXHRcdFx0XHRlcnJvciArPSBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkgKiBjb25uZWN0aW9uLmdhaW4gKiBjb25uZWN0aW9uLndlaWdodDtcblx0XHRcdH1cblxuXHRcdFx0Ly8gcHJvamVjdGVkIGVycm9yIHJlc3BvbnNpYmlsaXR5XG5cdFx0XHR0aGlzLmVycm9yLnByb2plY3RlZCA9IHRoaXMuZGVyaXZhdGl2ZSAqIGVycm9yO1xuXG5cdFx0XHRlcnJvciA9IDA7XG5cdFx0XHQvLyBlcnJvciByZXNwb25zaWJpbGl0aWVzIGZyb20gYWxsIHRoZSBjb25uZWN0aW9ucyBnYXRlZCBieSB0aGlzIG5ldXJvblxuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTsgLy8gZ2F0ZWQgbmV1cm9uXG5cdFx0XHRcdHZhciBpbmZsdWVuY2UgPSBuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIgPT0gdGhpcyA/IG5ldXJvbi5vbGQgOiAwOyAvLyBpZiBnYXRlZCBuZXVyb24ncyBzZWxmY29ubmVjdGlvbiBpcyBnYXRlZCBieSB0aGlzIG5ldXJvblxuXG5cdFx0XHRcdC8vIGluZGV4IHJ1bnMgb3ZlciBhbGwgdGhlIGNvbm5lY3Rpb25zIHRvIHRoZSBnYXRlZCBuZXVyb24gdGhhdCBhcmUgZ2F0ZWQgYnkgdGhpcyBuZXVyb25cblx0XHRcdFx0Zm9yICh2YXIgaW5wdXQgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzW2lkXSkgeyAvLyBjYXB0dXJlcyB0aGUgZWZmZWN0IHRoYXQgdGhlIGlucHV0IGNvbm5lY3Rpb24gb2YgdGhpcyBuZXVyb24gaGF2ZSwgb24gYSBuZXVyb24gd2hpY2ggaXRzIGlucHV0L3MgaXMvYXJlIGdhdGVkIGJ5IHRoaXMgbmV1cm9uXG5cdFx0XHRcdFx0aW5mbHVlbmNlICs9IHRoaXMudHJhY2UuaW5mbHVlbmNlc1tpZF1baW5wdXRdLndlaWdodCAqIHRoaXMudHJhY2UuaW5mbHVlbmNlc1tcblx0XHRcdFx0XHRuZXVyb24uSURdW2lucHV0XS5mcm9tLmFjdGl2YXRpb247XG5cdFx0XHRcdH1cblx0XHRcdFx0Ly8gZXEuIDIyXG5cdFx0XHRcdGVycm9yICs9IG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSAqIGluZmx1ZW5jZTtcblx0XHRcdH1cblxuXHRcdFx0Ly8gZ2F0ZWQgZXJyb3IgcmVzcG9uc2liaWxpdHlcblx0XHRcdHRoaXMuZXJyb3IuZ2F0ZWQgPSB0aGlzLmRlcml2YXRpdmUgKiBlcnJvcjtcblxuXHRcdFx0Ly8gZXJyb3IgcmVzcG9uc2liaWxpdHkgLSBFcS4gMjNcblx0XHRcdHRoaXMuZXJyb3IucmVzcG9uc2liaWxpdHkgPSB0aGlzLmVycm9yLnByb2plY3RlZCArIHRoaXMuZXJyb3IuZ2F0ZWQ7XG5cdFx0fVxuXG5cdFx0Ly8gbGVhcm5pbmcgcmF0ZVxuXHRcdHJhdGUgPSByYXRlIHx8IC4xO1xuXG5cdFx0Ly8gYWRqdXN0IGFsbCB0aGUgbmV1cm9uJ3MgaW5jb21pbmcgY29ubmVjdGlvbnNcblx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0dmFyIHRoZUlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaWRdO1xuXG5cdFx0XHQvLyBFcS4gMjRcblx0XHRcdHZhciBncmFkaWVudCA9IHRoaXMuZXJyb3IucHJvamVjdGVkICogdGhpcy50cmFjZS5lbGVnaWJpbGl0eVt0aGVJbnB1dC5JRF07XG5cdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHRncmFkaWVudCArPSBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkgKiB0aGlzLnRyYWNlLmV4dGVuZGVkW1xuXHRcdFx0XHRuZXVyb24uSURdW3RoZUlucHV0LklEXTtcblx0XHRcdH1cblx0XHRcdHRoZUlucHV0LndlaWdodCArPSByYXRlICogZ3JhZGllbnQ7IC8vIGFkanVzdCB3ZWlnaHRzIC0gYWthIGxlYXJuXG5cdFx0fVxuXG5cdFx0Ly8gYWRqdXN0IGJpYXNcblx0XHR0aGlzLmJpYXMgKz0gcmF0ZSAqIHRoaXMuZXJyb3IucmVzcG9uc2liaWxpdHk7XG5cdH1cblxuXHRwcm9qZWN0KG5ldXJvbiwgd2VpZ2h0PzogbnVtYmVyKTogTmV1cm9uLkNvbm5lY3Rpb24ge1xuXHRcdC8vIHNlbGYtY29ubmVjdGlvblxuXHRcdGlmIChuZXVyb24gPT0gdGhpcykge1xuXHRcdFx0dGhpcy5zZWxmY29ubmVjdGlvbi53ZWlnaHQgPSAxO1xuXHRcdFx0cmV0dXJuIHRoaXMuc2VsZmNvbm5lY3Rpb247XG5cdFx0fVxuXG5cdFx0Ly8gY2hlY2sgaWYgY29ubmVjdGlvbiBhbHJlYWR5IGV4aXN0c1xuXHRcdHZhciBjb25uZWN0ZWQgPSB0aGlzLmNvbm5lY3RlZChuZXVyb24pO1xuXHRcdGlmIChjb25uZWN0ZWQgJiYgY29ubmVjdGVkLnR5cGUgPT0gXCJwcm9qZWN0ZWRcIikge1xuXHRcdFx0Ly8gdXBkYXRlIGNvbm5lY3Rpb25cblx0XHRcdGlmICh0eXBlb2Ygd2VpZ2h0ICE9ICd1bmRlZmluZWQnKVxuXHRcdFx0XHRjb25uZWN0ZWQuY29ubmVjdGlvbi53ZWlnaHQgPSB3ZWlnaHQ7XG5cdFx0XHQvLyByZXR1cm4gZXhpc3RpbmcgY29ubmVjdGlvblxuXHRcdFx0cmV0dXJuIGNvbm5lY3RlZC5jb25uZWN0aW9uO1xuXHRcdH0gZWxzZSB7XG5cdFx0XHQvLyBjcmVhdGUgYSBuZXcgY29ubmVjdGlvblxuXHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBuZXcgTmV1cm9uLkNvbm5lY3Rpb24odGhpcywgbmV1cm9uLCB3ZWlnaHQpO1xuXHRcdH1cblxuXHRcdC8vIHJlZmVyZW5jZSBhbGwgdGhlIGNvbm5lY3Rpb25zIGFuZCB0cmFjZXNcblx0XHR0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZFtjb25uZWN0aW9uLklEXSA9IGNvbm5lY3Rpb247XG5cdFx0dGhpcy5uZWlnaGJvb3JzW25ldXJvbi5JRF0gPSBuZXVyb247XG5cdFx0bmV1cm9uLmNvbm5lY3Rpb25zLmlucHV0c1tjb25uZWN0aW9uLklEXSA9IGNvbm5lY3Rpb247XG5cdFx0bmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5W2Nvbm5lY3Rpb24uSURdID0gMDtcblxuXHRcdGZvciAodmFyIGlkIGluIG5ldXJvbi50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0dmFyIHRyYWNlID0gbmV1cm9uLnRyYWNlLmV4dGVuZGVkW2lkXTtcblx0XHRcdHRyYWNlW2Nvbm5lY3Rpb24uSURdID0gMDtcblx0XHR9XG5cblx0XHRyZXR1cm4gY29ubmVjdGlvbjtcblx0fVxuXG5cdGdhdGUoY29ubmVjdGlvbikge1xuXHRcdC8vIGFkZCBjb25uZWN0aW9uIHRvIGdhdGVkIGxpc3Rcblx0XHR0aGlzLmNvbm5lY3Rpb25zLmdhdGVkW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcblxuXHRcdHZhciBuZXVyb24gPSBjb25uZWN0aW9uLnRvO1xuXHRcdGlmICghKG5ldXJvbi5JRCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSkge1xuXHRcdFx0Ly8gZXh0ZW5kZWQgdHJhY2Vcblx0XHRcdHRoaXMubmVpZ2hib29yc1tuZXVyb24uSURdID0gbmV1cm9uO1xuXHRcdFx0dmFyIHh0cmFjZSA9IHRoaXMudHJhY2UuZXh0ZW5kZWRbbmV1cm9uLklEXSA9IHt9O1xuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaWRdO1xuXHRcdFx0XHR4dHJhY2VbaW5wdXQuSURdID0gMDtcblx0XHRcdH1cblx0XHR9XG5cblx0XHQvLyBrZWVwIHRyYWNrXG5cdFx0aWYgKG5ldXJvbi5JRCBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXMpXG5cdFx0XHR0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXS5wdXNoKGNvbm5lY3Rpb24pO1xuXHRcdGVsc2Vcblx0XHRcdHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdID0gW2Nvbm5lY3Rpb25dO1xuXG5cdFx0Ly8gc2V0IGdhdGVyXG5cdFx0Y29ubmVjdGlvbi5nYXRlciA9IHRoaXM7XG5cdH1cbiAgXG5cdC8vIHJldHVybnMgdHJ1ZSBvciBmYWxzZSB3aGV0aGVyIHRoZSBuZXVyb24gaXMgc2VsZi1jb25uZWN0ZWQgb3Igbm90XG5cdHNlbGZjb25uZWN0ZWQoKSB7XG5cdFx0cmV0dXJuIHRoaXMuc2VsZmNvbm5lY3Rpb24ud2VpZ2h0ICE9PSAwO1xuXHR9XG5cblx0Ly8gcmV0dXJucyB0cnVlIG9yIGZhbHNlIHdoZXRoZXIgdGhlIG5ldXJvbiBpcyBjb25uZWN0ZWQgdG8gYW5vdGhlciBuZXVyb24gKHBhcmFtZXRlcilcblx0Y29ubmVjdGVkKG5ldXJvbikge1xuXHRcdHZhciByZXN1bHQ6IHtcblx0XHRcdHR5cGU6IHN0cmluZztcblx0XHRcdGNvbm5lY3Rpb246IE5ldXJvbi5Db25uZWN0aW9uO1xuXHRcdH0gPSB7XG5cdFx0XHRcdHR5cGU6IG51bGwsXG5cdFx0XHRcdGNvbm5lY3Rpb246IG51bGxcblx0XHRcdH07XG5cblx0XHRpZiAodGhpcyA9PSBuZXVyb24pIHtcblx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0ZWQoKSkge1xuXHRcdFx0XHRyZXN1bHQudHlwZSA9ICdzZWxmY29ubmVjdGlvbic7XG5cdFx0XHRcdHJlc3VsdC5jb25uZWN0aW9uID0gdGhpcy5zZWxmY29ubmVjdGlvbjtcblx0XHRcdFx0cmV0dXJuIHJlc3VsdDtcblx0XHRcdH0gZWxzZVxuXHRcdFx0XHRyZXR1cm4gbnVsbDtcblx0XHR9XG5cblx0XHRmb3IgKHZhciB0eXBlIGluIHRoaXMuY29ubmVjdGlvbnMpIHtcblx0XHRcdGZvciAodmFyIGNvbm5lY3Rpb24gaW4gdGhpcy5jb25uZWN0aW9uc1t0eXBlXSkge1xuXHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IHRoaXMuY29ubmVjdGlvbnNbdHlwZV1bY29ubmVjdGlvbl07XG5cdFx0XHRcdGlmIChjb25uZWN0aW9uLnRvID09IG5ldXJvbikge1xuXHRcdFx0XHRcdHJlc3VsdC50eXBlID0gdHlwZTtcblx0XHRcdFx0XHRyZXN1bHQuY29ubmVjdGlvbiA9IGNvbm5lY3Rpb247XG5cdFx0XHRcdFx0cmV0dXJuIHJlc3VsdDtcblx0XHRcdFx0fSBlbHNlIGlmIChjb25uZWN0aW9uLmZyb20gPT0gbmV1cm9uKSB7XG5cdFx0XHRcdFx0cmVzdWx0LnR5cGUgPSB0eXBlO1xuXHRcdFx0XHRcdHJlc3VsdC5jb25uZWN0aW9uID0gY29ubmVjdGlvbjtcblx0XHRcdFx0XHRyZXR1cm4gcmVzdWx0O1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFx0cmV0dXJuIG51bGw7XG5cdH1cblxuXHQvLyBjbGVhcnMgYWxsIHRoZSB0cmFjZXMgKHRoZSBuZXVyb24gZm9yZ2V0cyBpdCdzIGNvbnRleHQsIGJ1dCB0aGUgY29ubmVjdGlvbnMgcmVtYWluIGludGFjdClcblx0Y2xlYXIoKSB7XG5cblx0XHRmb3IgKHZhciB0cmFjZSBpbiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5KVxuXHRcdFx0dGhpcy50cmFjZS5lbGVnaWJpbGl0eVt0cmFjZV0gPSAwO1xuXG5cdFx0Zm9yICh2YXIgdHJhY2UgaW4gdGhpcy50cmFjZS5leHRlbmRlZClcblx0XHRcdGZvciAodmFyIGV4dGVuZGVkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWRbdHJhY2VdKVxuXHRcdFx0XHR0aGlzLnRyYWNlLmV4dGVuZGVkW3RyYWNlXVtleHRlbmRlZF0gPSAwO1xuXG5cdFx0dGhpcy5lcnJvci5yZXNwb25zaWJpbGl0eSA9IHRoaXMuZXJyb3IucHJvamVjdGVkID0gdGhpcy5lcnJvci5nYXRlZCA9IDA7XG5cdH1cblxuXHQvLyBhbGwgdGhlIGNvbm5lY3Rpb25zIGFyZSByYW5kb21pemVkIGFuZCB0aGUgdHJhY2VzIGFyZSBjbGVhcmVkXG5cdHJlc2V0KCkge1xuXHRcdHRoaXMuY2xlYXIoKTtcblxuXHRcdGZvciAodmFyIHR5cGUgaW4gdGhpcy5jb25uZWN0aW9ucylcblx0XHRcdGZvciAodmFyIGNvbm5lY3Rpb24gaW4gdGhpcy5jb25uZWN0aW9uc1t0eXBlXSlcblx0XHRcdFx0dGhpcy5jb25uZWN0aW9uc1t0eXBlXVtjb25uZWN0aW9uXS53ZWlnaHQgPSBNYXRoLnJhbmRvbSgpICogLjIgLSAuMTtcblx0XHR0aGlzLmJpYXMgPSBNYXRoLnJhbmRvbSgpICogLjIgLSAuMTtcblxuXHRcdHRoaXMub2xkID0gdGhpcy5zdGF0ZSA9IHRoaXMuYWN0aXZhdGlvbiA9IDA7XG5cdH1cblx0XG5cbiAgXG5cblx0Ly8gaGFyZGNvZGVzIHRoZSBiZWhhdmlvdXIgb2YgdGhlIG5ldXJvbiBpbnRvIGFuIG9wdGltaXplZCBmdW5jdGlvblxuXHRvcHRpbWl6ZShvcHRpbWl6ZWQsIGxheWVyKTogU3luYXB0aWMuSUNvbXBpbGVkUGFyYW1ldGVycyB7XG5cblx0XHRvcHRpbWl6ZWQgPSBvcHRpbWl6ZWQgfHwge307XG5cdFx0dmFyIHRoYXQgPSB0aGlzO1xuXHRcdHZhciBzdG9yZV9hY3RpdmF0aW9uID0gW107XG5cdFx0dmFyIHN0b3JlX3RyYWNlID0gW107XG5cdFx0dmFyIHN0b3JlX3Byb3BhZ2F0aW9uID0gW107XG5cdFx0dmFyIHZhcklEID0gb3B0aW1pemVkLm1lbW9yeSB8fCAwO1xuXHRcdHZhciBuZXVyb25zID0gb3B0aW1pemVkLm5ldXJvbnMgfHwgMTtcblx0XHR2YXIgaW5wdXRzID0gb3B0aW1pemVkLmlucHV0cyB8fCBbXTtcblx0XHR2YXIgdGFyZ2V0cyA9IG9wdGltaXplZC50YXJnZXRzIHx8IFtdO1xuXHRcdHZhciBvdXRwdXRzID0gb3B0aW1pemVkLm91dHB1dHMgfHwgW107XG5cdFx0dmFyIHZhcmlhYmxlcyA9IG9wdGltaXplZC52YXJpYWJsZXMgfHwge307XG5cdFx0dmFyIGFjdGl2YXRpb25fc2VudGVuY2VzID0gb3B0aW1pemVkLmFjdGl2YXRpb25fc2VudGVuY2VzIHx8IFtdO1xuXHRcdHZhciB0cmFjZV9zZW50ZW5jZXMgPSBvcHRpbWl6ZWQudHJhY2Vfc2VudGVuY2VzIHx8IFtdO1xuXHRcdHZhciBwcm9wYWdhdGlvbl9zZW50ZW5jZXMgPSBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzIHx8IFtdO1xuXHRcdHZhciBsYXllcnMgPSBvcHRpbWl6ZWQubGF5ZXJzIHx8IHsgX19jb3VudDogMCwgX19uZXVyb246IDAgfTtcblxuXHRcdC8vIGFsbG9jYXRlIHNlbnRlbmNlc1xuXHRcdHZhciBhbGxvY2F0ZSA9IGZ1bmN0aW9uKHN0b3JlKSB7XG5cdFx0XHR2YXIgYWxsb2NhdGVkID0gbGF5ZXIgaW4gbGF5ZXJzICYmIHN0b3JlW2xheWVycy5fX2NvdW50XTtcblx0XHRcdGlmICghYWxsb2NhdGVkKSB7XG5cdFx0XHRcdGxheWVycy5fX2NvdW50ID0gc3RvcmUucHVzaChbXSkgLSAxO1xuXHRcdFx0XHRsYXllcnNbbGF5ZXJdID0gbGF5ZXJzLl9fY291bnQ7XG5cdFx0XHR9XG5cdFx0fVxuXHRcdGFsbG9jYXRlKGFjdGl2YXRpb25fc2VudGVuY2VzKTtcblx0XHRhbGxvY2F0ZSh0cmFjZV9zZW50ZW5jZXMpO1xuXHRcdGFsbG9jYXRlKHByb3BhZ2F0aW9uX3NlbnRlbmNlcyk7XG5cdFx0dmFyIGN1cnJlbnRMYXllciA9IGxheWVycy5fX2NvdW50O1xuXG5cdFx0Ly8gZ2V0L3Jlc2VydmUgc3BhY2UgaW4gbWVtb3J5IGJ5IGNyZWF0aW5nIGEgdW5pcXVlIElEIGZvciBhIHZhcmlhYmxlbFxuXHRcdHZhciBnZXRWYXIgPSBmdW5jdGlvbiguLi5hcmdzOiBhbnlbXSkge1xuXHRcdFx0dmFyIGlkO1xuXHRcdFx0aWYgKGFyZ3MubGVuZ3RoID09IDEpIHtcblxuXHRcdFx0XHRpZiAoYXJnc1swXSA9PSAndGFyZ2V0Jykge1xuXHRcdFx0XHRcdGlkID0gJ3RhcmdldF8nICsgdGFyZ2V0cy5sZW5ndGg7XG5cdFx0XHRcdFx0dGFyZ2V0cy5wdXNoKHZhcklEKTtcblx0XHRcdFx0fSBlbHNlXG5cdFx0XHRcdFx0aWQgPSBhcmdzWzBdO1xuXHRcdFx0XHRpZiAoaWQgaW4gdmFyaWFibGVzKVxuXHRcdFx0XHRcdHJldHVybiB2YXJpYWJsZXNbaWRdO1xuXHRcdFx0XHRyZXR1cm4gdmFyaWFibGVzW2lkXSA9IHtcblx0XHRcdFx0XHR2YWx1ZTogMCxcblx0XHRcdFx0XHRpZDogdmFySUQrK1xuXHRcdFx0XHR9O1xuXHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0dmFyIGV4dGVuZGVkID0gYXJncy5sZW5ndGggPiAyO1xuXHRcdFx0XHRpZiAoZXh0ZW5kZWQpXG5cdFx0XHRcdFx0dmFyIHZhbHVlID0gYXJncy5wb3AoKTtcblxuXHRcdFx0XHR2YXIgdW5pdCA9IGFyZ3Muc2hpZnQoKTtcblx0XHRcdFx0dmFyIHByb3AgPSBhcmdzLnBvcCgpO1xuXG5cdFx0XHRcdGlmICghZXh0ZW5kZWQpXG5cdFx0XHRcdFx0dmFyIHZhbHVlID0gdW5pdFtwcm9wXTtcblxuXHRcdFx0XHRpZCA9IHByb3AgKyAnXyc7XG5cdFx0XHRcdGZvciAodmFyIHByb3BlcnR5IGluIGFyZ3MpXG5cdFx0XHRcdFx0aWQgKz0gYXJnc1twcm9wZXJ0eV0gKyAnXyc7XG5cdFx0XHRcdGlkICs9IHVuaXQuSUQ7XG5cdFx0XHRcdGlmIChpZCBpbiB2YXJpYWJsZXMpXG5cdFx0XHRcdFx0cmV0dXJuIHZhcmlhYmxlc1tpZF07XG5cblx0XHRcdFx0cmV0dXJuIHZhcmlhYmxlc1tpZF0gPSB7XG5cdFx0XHRcdFx0dmFsdWU6IHZhbHVlLFxuXHRcdFx0XHRcdGlkOiB2YXJJRCsrXG5cdFx0XHRcdH07XG5cdFx0XHR9XG5cdFx0fTtcblxuXHRcdC8vIGJ1aWxkIHNlbnRlbmNlXG5cdFx0dmFyIGJ1aWxkU2VudGVuY2UgPSBmdW5jdGlvbiguLi5hcmdzOiBhbnlbXSkge1xuXHRcdFx0dmFyIHN0b3JlID0gYXJncy5wb3AoKTtcblx0XHRcdHZhciBzZW50ZW5jZSA9IFwiXCI7XG5cdFx0XHRmb3IgKHZhciBpIGluIGFyZ3MpXG5cdFx0XHRcdGlmICh0eXBlb2YgYXJnc1tpXSA9PSAnc3RyaW5nJylcblx0XHRcdFx0XHRzZW50ZW5jZSArPSBhcmdzW2ldO1xuXHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0c2VudGVuY2UgKz0gJ0ZbJyArIGFyZ3NbaV0uaWQgKyAnXSc7XG5cblx0XHRcdHN0b3JlLnB1c2goc2VudGVuY2UgKyAnOycpO1xuXHRcdH1cblxuXHRcdC8vIGhlbHBlciB0byBjaGVjayBpZiBhbiBvYmplY3QgaXMgZW1wdHlcblx0XHR2YXIgaXNFbXB0eSA9IGZ1bmN0aW9uKG9iaikge1xuXHRcdFx0Zm9yICh2YXIgcHJvcCBpbiBvYmopIHtcblx0XHRcdFx0aWYgKG9iai5oYXNPd25Qcm9wZXJ0eShwcm9wKSlcblx0XHRcdFx0XHRyZXR1cm4gZmFsc2U7XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gdHJ1ZTtcblx0XHR9O1xuXG5cdFx0Ly8gY2hhcmFjdGVyaXN0aWNzIG9mIHRoZSBuZXVyb25cblx0XHR2YXIgbm9Qcm9qZWN0aW9ucyA9IGlzRW1wdHkodGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpO1xuXHRcdHZhciBub0dhdGVzID0gaXNFbXB0eSh0aGlzLmNvbm5lY3Rpb25zLmdhdGVkKTtcblx0XHR2YXIgaXNJbnB1dCA9IGxheWVyID09ICdpbnB1dCcgPyB0cnVlIDogaXNFbXB0eSh0aGlzLmNvbm5lY3Rpb25zLmlucHV0cyk7XG5cdFx0dmFyIGlzT3V0cHV0ID0gbGF5ZXIgPT0gJ291dHB1dCcgPyB0cnVlIDogbm9Qcm9qZWN0aW9ucyAmJiBub0dhdGVzO1xuXG5cdFx0Ly8gb3B0aW1pemUgbmV1cm9uJ3MgYmVoYXZpb3VyXG5cdFx0dmFyIHJhdGUgPSBnZXRWYXIoJ3JhdGUnKTtcblx0XHR2YXIgYWN0aXZhdGlvbiA9IGdldFZhcih0aGlzLCAnYWN0aXZhdGlvbicpO1xuXHRcdGlmIChpc0lucHV0KVxuXHRcdFx0aW5wdXRzLnB1c2goYWN0aXZhdGlvbi5pZCk7XG5cdFx0ZWxzZSB7XG5cdFx0XHRhY3RpdmF0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdLnB1c2goc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHR0cmFjZV9zZW50ZW5jZXNbY3VycmVudExheWVyXS5wdXNoKHN0b3JlX3RyYWNlKTtcblx0XHRcdHByb3BhZ2F0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdLnB1c2goc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0dmFyIG9sZCA9IGdldFZhcih0aGlzLCAnb2xkJyk7XG5cdFx0XHR2YXIgc3RhdGUgPSBnZXRWYXIodGhpcywgJ3N0YXRlJyk7XG5cdFx0XHR2YXIgYmlhcyA9IGdldFZhcih0aGlzLCAnYmlhcycpO1xuXHRcdFx0aWYgKHRoaXMuc2VsZmNvbm5lY3Rpb24uZ2F0ZXIpXG5cdFx0XHRcdHZhciBzZWxmX2dhaW4gPSBnZXRWYXIodGhpcy5zZWxmY29ubmVjdGlvbiwgJ2dhaW4nKTtcblx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0ZWQoKSlcblx0XHRcdFx0dmFyIHNlbGZfd2VpZ2h0ID0gZ2V0VmFyKHRoaXMuc2VsZmNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdGJ1aWxkU2VudGVuY2Uob2xkLCAnID0gJywgc3RhdGUsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0aWYgKHRoaXMuc2VsZmNvbm5lY3RlZCgpKVxuXHRcdFx0XHRpZiAodGhpcy5zZWxmY29ubmVjdGlvbi5nYXRlcilcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHN0YXRlLCAnID0gJywgc2VsZl9nYWluLCAnICogJywgc2VsZl93ZWlnaHQsICcgKiAnLFxuXHRcdFx0XHRcdFx0c3RhdGUsICcgKyAnLCBiaWFzLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2Uoc3RhdGUsICcgPSAnLCBzZWxmX3dlaWdodCwgJyAqICcsIHN0YXRlLCAnICsgJyxcblx0XHRcdFx0XHRcdGJpYXMsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0ZWxzZVxuXHRcdFx0XHRidWlsZFNlbnRlbmNlKHN0YXRlLCAnID0gJywgYmlhcywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRmb3IgKHZhciBpIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHRcdHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2ldO1xuXHRcdFx0XHR2YXIgaW5wdXRfYWN0aXZhdGlvbiA9IGdldFZhcihpbnB1dC5mcm9tLCAnYWN0aXZhdGlvbicpO1xuXHRcdFx0XHR2YXIgaW5wdXRfd2VpZ2h0ID0gZ2V0VmFyKGlucHV0LCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdGlmIChpbnB1dC5nYXRlcilcblx0XHRcdFx0XHR2YXIgaW5wdXRfZ2FpbiA9IGdldFZhcihpbnB1dCwgJ2dhaW4nKTtcblx0XHRcdFx0aWYgKHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2ldLmdhdGVyKVxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2Uoc3RhdGUsICcgKz0gJywgaW5wdXRfYWN0aXZhdGlvbiwgJyAqICcsXG5cdFx0XHRcdFx0XHRpbnB1dF93ZWlnaHQsICcgKiAnLCBpbnB1dF9nYWluLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2Uoc3RhdGUsICcgKz0gJywgaW5wdXRfYWN0aXZhdGlvbiwgJyAqICcsXG5cdFx0XHRcdFx0XHRpbnB1dF93ZWlnaHQsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0fVxuXHRcdFx0dmFyIGRlcml2YXRpdmUgPSBnZXRWYXIodGhpcywgJ2Rlcml2YXRpdmUnKTtcblx0XHRcdHN3aXRjaCAodGhpcy5zcXVhc2gpIHtcblx0XHRcdFx0Y2FzZSBTcXVhc2guTE9HSVNUSUM6XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShhY3RpdmF0aW9uLCAnID0gKDEgLyAoMSArIE1hdGguZXhwKC0nLCBzdGF0ZSwgJykpKScsXG5cdFx0XHRcdFx0XHRzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGRlcml2YXRpdmUsICcgPSAnLCBhY3RpdmF0aW9uLCAnICogKDEgLSAnLFxuXHRcdFx0XHRcdFx0YWN0aXZhdGlvbiwgJyknLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRicmVhaztcblx0XHRcdFx0Y2FzZSBTcXVhc2guVEFOSDpcblx0XHRcdFx0XHR2YXIgZVAgPSBnZXRWYXIoJ2F1eCcpO1xuXHRcdFx0XHRcdHZhciBlTiA9IGdldFZhcignYXV4XzInKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVQLCAnID0gTWF0aC5leHAoJywgc3RhdGUsICcpJywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShlTiwgJyA9IDEgLyAnLCBlUCwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShhY3RpdmF0aW9uLCAnID0gKCcsIGVQLCAnIC0gJywgZU4sICcpIC8gKCcsIGVQLCAnICsgJywgZU4sICcpJywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShkZXJpdmF0aXZlLCAnID0gMSAtICgnLCBhY3RpdmF0aW9uLCAnICogJywgYWN0aXZhdGlvbiwgJyknLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRicmVhaztcblx0XHRcdFx0Y2FzZSBTcXVhc2guSURFTlRJVFk6XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShhY3RpdmF0aW9uLCAnID0gJywgc3RhdGUsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZGVyaXZhdGl2ZSwgJyA9IDEnLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRicmVhaztcblx0XHRcdFx0Y2FzZSBTcXVhc2guSExJTTpcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSArKCcsIHN0YXRlLCAnID4gMCknLFxuXHRcdFx0XHRcdFx0c3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShkZXJpdmF0aXZlLCAnID0gMScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJyZWFrO1xuXHRcdFx0fVxuXG5cblx0XHRcdHZhciBpbmZsdWVuY2VzID0gW107XG5cdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdC8vIGNhbGN1bGF0ZSBleHRlbmRlZCBlbGVnaWJpbGl0eSB0cmFjZXMgaW4gYWR2YW5jZVxuICAgICAgICBcblx0XHRcdFx0dmFyIHh0cmFjZSA9IHRoaXMudHJhY2UuZXh0ZW5kZWRbaWRdO1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0dmFyIGluZmx1ZW5jZSA9IGdldFZhcignYXV4Jyk7XG5cdFx0XHRcdHZhciBuZXVyb25fb2xkID0gZ2V0VmFyKG5ldXJvbiwgJ29sZCcpO1xuXHRcdFx0XHR2YXIgaW5pdGlhbGl6ZWQgPSBmYWxzZTtcblx0XHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzKSB7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAnLCBuZXVyb25fb2xkLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0aW5pdGlhbGl6ZWQgPSB0cnVlO1xuXHRcdFx0XHR9XG5cdFx0XHRcdGZvciAodmFyIGluY29taW5nIGluIHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdKSB7XG5cdFx0XHRcdFx0dmFyIGluY29taW5nX3dlaWdodCA9IGdldFZhcih0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXVxuXHRcdFx0XHRcdFtpbmNvbWluZ10sICd3ZWlnaHQnKTtcblx0XHRcdFx0XHR2YXIgaW5jb21pbmdfYWN0aXZhdGlvbiA9IGdldFZhcih0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXVxuXHRcdFx0XHRcdFtpbmNvbWluZ10uZnJvbSwgJ2FjdGl2YXRpb24nKTtcblxuXHRcdFx0XHRcdGlmIChpbml0aWFsaXplZClcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnICs9ICcsIGluY29taW5nX3dlaWdodCwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdGluY29taW5nX2FjdGl2YXRpb24sIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRlbHNlIHtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnID0gJywgaW5jb21pbmdfd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdFx0aW5jb21pbmdfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdFx0aW5pdGlhbGl6ZWQgPSB0cnVlO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fVxuXG5cdFx0XHRcdGluZmx1ZW5jZXMucHVzaChuZXVyb24uSUQpO1xuXHRcdFx0XHRidWlsZFNlbnRlbmNlKFwiaW5mbHVlbmNlc1tcIiArIChpbmZsdWVuY2VzLmxlbmd0aCAtIDEpICsgXCJdID0gXCIsIGluZmx1ZW5jZSwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0fVxuXG5cdFx0XHRmb3IgKHZhciBpIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHRcdHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2ldO1xuXHRcdFx0XHRpZiAoaW5wdXQuZ2F0ZXIpXG5cdFx0XHRcdFx0dmFyIGlucHV0X2dhaW4gPSBnZXRWYXIoaW5wdXQsICdnYWluJyk7XG5cdFx0XHRcdHZhciBpbnB1dF9hY3RpdmF0aW9uID0gZ2V0VmFyKGlucHV0LmZyb20sICdhY3RpdmF0aW9uJyk7XG5cdFx0XHRcdHZhciB0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZWxlZ2liaWxpdHknLCBpbnB1dC5JRCwgdGhpcy50cmFjZVxuXHRcdFx0XHRcdC5lbGVnaWJpbGl0eVtpbnB1dC5JRF0pO1xuXHRcdFx0XHRpZiAodGhpcy5zZWxmY29ubmVjdGVkKCkpIHtcblx0XHRcdFx0XHRpZiAodGhpcy5zZWxmY29ubmVjdGlvbi5nYXRlcikge1xuXHRcdFx0XHRcdFx0aWYgKGlucHV0LmdhdGVyKVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHRyYWNlLCAnID0gJywgc2VsZl9nYWluLCAnICogJywgc2VsZl93ZWlnaHQsXG5cdFx0XHRcdFx0XHRcdFx0JyAqICcsIHRyYWNlLCAnICsgJywgaW5wdXRfZ2FpbiwgJyAqICcsIGlucHV0X2FjdGl2YXRpb24sXG5cdFx0XHRcdFx0XHRcdFx0c3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHRyYWNlLCAnID0gJywgc2VsZl9nYWluLCAnICogJywgc2VsZl93ZWlnaHQsXG5cdFx0XHRcdFx0XHRcdFx0JyAqICcsIHRyYWNlLCAnICsgJywgaW5wdXRfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdFx0XHRpZiAoaW5wdXQuZ2F0ZXIpXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBzZWxmX3dlaWdodCwgJyAqICcsIHRyYWNlLCAnICsgJyxcblx0XHRcdFx0XHRcdFx0XHRpbnB1dF9nYWluLCAnICogJywgaW5wdXRfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHRyYWNlLCAnID0gJywgc2VsZl93ZWlnaHQsICcgKiAnLCB0cmFjZSwgJyArICcsXG5cdFx0XHRcdFx0XHRcdFx0aW5wdXRfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0XHRpZiAoaW5wdXQuZ2F0ZXIpXG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHRyYWNlLCAnID0gJywgaW5wdXRfZ2FpbiwgJyAqICcsIGlucHV0X2FjdGl2YXRpb24sXG5cdFx0XHRcdFx0XHRcdHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHRyYWNlLCAnID0gJywgaW5wdXRfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHR9XG5cdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0XHQvLyBleHRlbmRlZCBlbGVnaWJpbGl0eSB0cmFjZVxuXHRcdFx0XHRcdHZhciB4dHJhY2UgPSB0aGlzLnRyYWNlLmV4dGVuZGVkW2lkXTtcblx0XHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0XHR2YXIgaW5mbHVlbmNlID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0XHR2YXIgbmV1cm9uX29sZCA9IGdldFZhcihuZXVyb24sICdvbGQnKTtcblxuXHRcdFx0XHRcdHZhciB0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZWxlZ2liaWxpdHknLCBpbnB1dC5JRCwgdGhpcy50cmFjZVxuXHRcdFx0XHRcdFx0LmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG5cdFx0XHRcdFx0dmFyIHh0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZXh0ZW5kZWQnLCBuZXVyb24uSUQsIGlucHV0LklELFxuXHRcdFx0XHRcdFx0dGhpcy50cmFjZS5leHRlbmRlZFtuZXVyb24uSURdW2lucHV0LklEXSk7XG5cdFx0XHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGVkKCkpXG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3NlbGZfd2VpZ2h0ID0gZ2V0VmFyKG5ldXJvbi5zZWxmY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdGlmIChuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIpXG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3NlbGZfZ2FpbiA9IGdldFZhcihuZXVyb24uc2VsZmNvbm5lY3Rpb24sICdnYWluJyk7XG5cdFx0XHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGVkKCkpXG5cdFx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyKVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHh0cmFjZSwgJyA9ICcsIG5ldXJvbl9zZWxmX2dhaW4sICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdG5ldXJvbl9zZWxmX3dlaWdodCwgJyAqICcsIHh0cmFjZSwgJyArICcsIGRlcml2YXRpdmUsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdHRyYWNlLCAnICogJywgXCJpbmZsdWVuY2VzW1wiICsgaW5mbHVlbmNlcy5pbmRleE9mKG5ldXJvbi5JRCkgKyBcIl1cIiwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHh0cmFjZSwgJyA9ICcsIG5ldXJvbl9zZWxmX3dlaWdodCwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0eHRyYWNlLCAnICsgJywgZGVyaXZhdGl2ZSwgJyAqICcsIHRyYWNlLCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHRcImluZmx1ZW5jZXNbXCIgKyBpbmZsdWVuY2VzLmluZGV4T2YobmV1cm9uLklEKSArIFwiXVwiLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZSh4dHJhY2UsICcgPSAnLCBkZXJpdmF0aXZlLCAnICogJywgdHJhY2UsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcImluZmx1ZW5jZXNbXCIgKyBpbmZsdWVuY2VzLmluZGV4T2YobmV1cm9uLklEKSArIFwiXVwiLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHRcdGZvciAodmFyIGNvbm5lY3Rpb24gaW4gdGhpcy5jb25uZWN0aW9ucy5nYXRlZCkge1xuXHRcdFx0XHR2YXIgZ2F0ZWRfZ2FpbiA9IGdldFZhcih0aGlzLmNvbm5lY3Rpb25zLmdhdGVkW2Nvbm5lY3Rpb25dLCAnZ2FpbicpO1xuXHRcdFx0XHRidWlsZFNlbnRlbmNlKGdhdGVkX2dhaW4sICcgPSAnLCBhY3RpdmF0aW9uLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdH1cblx0XHR9XG5cdFx0aWYgKCFpc0lucHV0KSB7XG5cdFx0XHR2YXIgcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIodGhpcywgJ2Vycm9yJywgJ3Jlc3BvbnNpYmlsaXR5JywgdGhpcy5lcnJvclxuXHRcdFx0XHQucmVzcG9uc2liaWxpdHkpO1xuXHRcdFx0aWYgKGlzT3V0cHV0KSB7XG5cdFx0XHRcdHZhciB0YXJnZXQgPSBnZXRWYXIoJ3RhcmdldCcpO1xuXHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnID0gJywgdGFyZ2V0LCAnIC0gJywgYWN0aXZhdGlvbixcblx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaWRdO1xuXHRcdFx0XHRcdHZhciB0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZWxlZ2liaWxpdHknLCBpbnB1dC5JRCwgdGhpcy50cmFjZVxuXHRcdFx0XHRcdFx0LmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG5cdFx0XHRcdFx0dmFyIGlucHV0X3dlaWdodCA9IGdldFZhcihpbnB1dCwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5wdXRfd2VpZ2h0LCAnICs9ICcsIHJhdGUsICcgKiAoJywgcmVzcG9uc2liaWxpdHksXG5cdFx0XHRcdFx0XHQnICogJywgdHJhY2UsICcpJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHR9XG5cdFx0XHRcdG91dHB1dHMucHVzaChhY3RpdmF0aW9uLmlkKTtcblx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdGlmICghbm9Qcm9qZWN0aW9ucyAmJiAhbm9HYXRlcykge1xuXHRcdFx0XHRcdHZhciBlcnJvciA9IGdldFZhcignYXV4Jyk7XG5cdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpIHtcblx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uID0gdGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbaWRdO1xuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24udG87XG5cdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbl93ZWlnaHQgPSBnZXRWYXIoY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG5cdFx0XHRcdFx0XHRcdCdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRcdFx0XHRpZiAoY29ubmVjdGlvbi5nYXRlcikge1xuXHRcdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbl9nYWluID0gZ2V0VmFyKGNvbm5lY3Rpb24sICdnYWluJyk7XG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZXJyb3IsICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHRjb25uZWN0aW9uX2dhaW4sICcgKiAnLCBjb25uZWN0aW9uX3dlaWdodCxcblx0XHRcdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHR9IGVsc2Vcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShlcnJvciwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdGNvbm5lY3Rpb25fd2VpZ2h0LCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHRcdHZhciBwcm9qZWN0ZWQgPSBnZXRWYXIodGhpcywgJ2Vycm9yJywgJ3Byb2plY3RlZCcsIHRoaXMuZXJyb3IucHJvamVjdGVkKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHByb2plY3RlZCwgJyA9ICcsIGRlcml2YXRpdmUsICcgKiAnLCBlcnJvcixcblx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVycm9yLCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0XHRcdHZhciBpbmZsdWVuY2UgPSBnZXRWYXIoJ2F1eF8yJyk7XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX29sZCA9IGdldFZhcihuZXVyb24sICdvbGQnKTtcblx0XHRcdFx0XHRcdGlmIChuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIgPT0gdGhpcylcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAnLCBuZXVyb25fb2xkLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdGZvciAodmFyIGluZmx1ZW5jZUlucHV0IGluIHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdKSB7XG5cdFx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uID0gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF1baW5mbHVlbmNlSW5wdXRdO1xuXHRcdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbl93ZWlnaHQgPSBnZXRWYXIoY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdFx0XHR2YXIgbmV1cm9uX2FjdGl2YXRpb24gPSBnZXRWYXIoY29ubmVjdGlvbi5mcm9tLCAnYWN0aXZhdGlvbicpO1xuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyArPSAnLCBjb25uZWN0aW9uX3dlaWdodCwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0bmV1cm9uX2FjdGl2YXRpb24sIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdH1cblx0XHRcdFx0XHRcdHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuXHRcdFx0XHRcdFx0XHQncmVzcG9uc2liaWxpdHknLCBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkpO1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShlcnJvciwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRpbmZsdWVuY2UsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0dmFyIGdhdGVkID0gZ2V0VmFyKHRoaXMsICdlcnJvcicsICdnYXRlZCcsIHRoaXMuZXJyb3IuZ2F0ZWQpO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZ2F0ZWQsICcgPSAnLCBkZXJpdmF0aXZlLCAnICogJywgZXJyb3IsXG5cdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyA9ICcsIHByb2plY3RlZCwgJyArICcsIGdhdGVkLFxuXHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHRcdFx0XHR2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cdFx0XHRcdFx0XHR2YXIgZ3JhZGllbnQgPSBnZXRWYXIoJ2F1eCcpO1xuXHRcdFx0XHRcdFx0dmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzXG5cdFx0XHRcdFx0XHRcdC50cmFjZS5lbGVnaWJpbGl0eVtpbnB1dC5JRF0pO1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShncmFkaWVudCwgJyA9ICcsIHByb2plY3RlZCwgJyAqICcsIHRyYWNlLFxuXHRcdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcblx0XHRcdFx0XHRcdFx0XHQncmVzcG9uc2liaWxpdHknLCBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkpO1xuXHRcdFx0XHRcdFx0XHR2YXIgeHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdleHRlbmRlZCcsIG5ldXJvbi5JRCxcblx0XHRcdFx0XHRcdFx0XHRpbnB1dC5JRCwgdGhpcy50cmFjZS5leHRlbmRlZFtuZXVyb24uSURdW2lucHV0LklEXSk7XG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZ3JhZGllbnQsICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHR4dHJhY2UsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdH1cblx0XHRcdFx0XHRcdHZhciBpbnB1dF93ZWlnaHQgPSBnZXRWYXIoaW5wdXQsICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5wdXRfd2VpZ2h0LCAnICs9ICcsIHJhdGUsICcgKiAnLCBncmFkaWVudCxcblx0XHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdH1cblxuXHRcdFx0XHR9IGVsc2UgaWYgKG5vR2F0ZXMpIHtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZCkge1xuXHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZFtpZF07XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gY29ubmVjdGlvbi50bztcblx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX3dlaWdodCA9IGdldFZhcihjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcblx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdGlmIChjb25uZWN0aW9uLmdhdGVyKSB7XG5cdFx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX2dhaW4gPSBnZXRWYXIoY29ubmVjdGlvbiwgJ2dhaW4nKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksXG5cdFx0XHRcdFx0XHRcdFx0JyAqICcsIGNvbm5lY3Rpb25fZ2FpbiwgJyAqICcsIGNvbm5lY3Rpb25fd2VpZ2h0LFxuXHRcdFx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdH0gZWxzZVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSxcblx0XHRcdFx0XHRcdFx0XHQnICogJywgY29ubmVjdGlvbl93ZWlnaHQsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyAqPSAnLCBkZXJpdmF0aXZlLFxuXHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHRcdFx0XHR2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cdFx0XHRcdFx0XHR2YXIgdHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2VsZWdpYmlsaXR5JywgaW5wdXQuSUQsIHRoaXNcblx0XHRcdFx0XHRcdFx0LnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG5cdFx0XHRcdFx0XHR2YXIgaW5wdXRfd2VpZ2h0ID0gZ2V0VmFyKGlucHV0LCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGlucHV0X3dlaWdodCwgJyArPSAnLCByYXRlLCAnICogKCcsXG5cdFx0XHRcdFx0XHRcdHJlc3BvbnNpYmlsaXR5LCAnICogJywgdHJhY2UsICcpJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fSBlbHNlIGlmIChub1Byb2plY3Rpb25zKSB7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyA9IDAnLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdFx0XHR2YXIgaW5mbHVlbmNlID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fb2xkID0gZ2V0VmFyKG5ldXJvbiwgJ29sZCcpO1xuXHRcdFx0XHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzKVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9ICcsIG5ldXJvbl9vbGQsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0Zm9yICh2YXIgaW5mbHVlbmNlSW5wdXQgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0pIHtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXVtpbmZsdWVuY2VJbnB1dF07XG5cdFx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX3dlaWdodCA9IGdldFZhcihjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHRcdHZhciBuZXVyb25fYWN0aXZhdGlvbiA9IGdldFZhcihjb25uZWN0aW9uLmZyb20sICdhY3RpdmF0aW9uJyk7XG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnICs9ICcsIGNvbm5lY3Rpb25fd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHRuZXVyb25fYWN0aXZhdGlvbiwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG5cdFx0XHRcdFx0XHRcdCdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSxcblx0XHRcdFx0XHRcdFx0JyAqICcsIGluZmx1ZW5jZSwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICo9ICcsIGRlcml2YXRpdmUsXG5cdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0XHRcdHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcblx0XHRcdFx0XHRcdHZhciBncmFkaWVudCA9IGdldFZhcignYXV4Jyk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGdyYWRpZW50LCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdFx0XHRcdHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuXHRcdFx0XHRcdFx0XHRcdCdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRcdFx0XHRcdHZhciB4dHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2V4dGVuZGVkJywgbmV1cm9uLklELFxuXHRcdFx0XHRcdFx0XHRcdGlucHV0LklELCB0aGlzLnRyYWNlLmV4dGVuZGVkW25ldXJvbi5JRF1baW5wdXQuSURdKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShncmFkaWVudCwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdHh0cmFjZSwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0dmFyIGlucHV0X3dlaWdodCA9IGdldFZhcihpbnB1dCwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbnB1dF93ZWlnaHQsICcgKz0gJywgcmF0ZSwgJyAqICcsIGdyYWRpZW50LFxuXHRcdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0XHRidWlsZFNlbnRlbmNlKGJpYXMsICcgKz0gJywgcmF0ZSwgJyAqICcsIHJlc3BvbnNpYmlsaXR5LFxuXHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0fVxuXHRcdHJldHVybiB7XG5cdFx0XHRtZW1vcnk6IHZhcklELFxuXHRcdFx0bmV1cm9uczogbmV1cm9ucyArIDEsXG5cdFx0XHRpbnB1dHM6IGlucHV0cyxcblx0XHRcdG91dHB1dHM6IG91dHB1dHMsXG5cdFx0XHR0YXJnZXRzOiB0YXJnZXRzLFxuXHRcdFx0dmFyaWFibGVzOiB2YXJpYWJsZXMsXG5cdFx0XHRhY3RpdmF0aW9uX3NlbnRlbmNlczogYWN0aXZhdGlvbl9zZW50ZW5jZXMsXG5cdFx0XHR0cmFjZV9zZW50ZW5jZXM6IHRyYWNlX3NlbnRlbmNlcyxcblx0XHRcdHByb3BhZ2F0aW9uX3NlbnRlbmNlczogcHJvcGFnYXRpb25fc2VudGVuY2VzLFxuXHRcdFx0bGF5ZXJzOiBsYXllcnNcblx0XHR9XG5cdH1cbn1cblxuZXhwb3J0IG1vZHVsZSBOZXVyb24ge1xuXG5cdGV4cG9ydCBpbnRlcmZhY2UgSU5ldXJvbkNvbm5lY3Rpb25zIHtcblx0XHRpbnB1dHM6IFN5bmFwdGljLkRpY3Rpb25hcnk8TmV1cm9uLkNvbm5lY3Rpb24+O1xuXHRcdHByb2plY3RlZDoge307XG5cdFx0Z2F0ZWQ6IHt9O1xuXHR9XG5cblx0ZXhwb3J0IGNsYXNzIENvbm5lY3Rpb24ge1xuXHRcdElEID0gQ29ubmVjdGlvbi51aWQoKTtcblx0XHRmcm9tO1xuXHRcdHRvO1xuXHRcdGdhaW46IG51bWJlciA9IDE7XG5cdFx0d2VpZ2h0OiBudW1iZXIgPSAwO1xuXHRcdGdhdGVyOiBhbnkgPSBudWxsO1xuXHRcdGNvbnN0cnVjdG9yKGZyb20sIHRvLCB3ZWlnaHQ/OiBudW1iZXIpIHtcblx0XHRcdGlmICghZnJvbSB8fCAhdG8pXG5cdFx0XHRcdHRocm93IFwiQ29ubmVjdGlvbiBFcnJvcjogSW52YWxpZCBuZXVyb25zXCI7XG5cdFx0XHR0aGlzLmZyb20gPSBmcm9tO1xuXHRcdFx0dGhpcy50byA9IHRvO1xuXHRcdFx0dGhpcy53ZWlnaHQgPSB0eXBlb2Ygd2VpZ2h0ID09ICd1bmRlZmluZWQnIHx8IGlzTmFOKHdlaWdodCkgPyBNYXRoLnJhbmRvbSgpICogLjIgLSAuMSA6XG5cdFx0XHRcdHdlaWdodDtcblx0XHR9XG5cdH1cblxuXHRleHBvcnQgdmFyIG5ldXJvblF0eSA9IDA7XG5cdGV4cG9ydCBmdW5jdGlvbiB1aWQoKTogbnVtYmVyIHtcblx0XHRyZXR1cm4gbmV1cm9uUXR5Kys7XG5cdH1cblxuXHRleHBvcnQgZnVuY3Rpb24gcXVhbnRpdHkoKSB7XG5cdFx0cmV0dXJuIHtcblx0XHRcdG5ldXJvbnM6IG5ldXJvblF0eSxcblx0XHRcdGNvbm5lY3Rpb25zOiBDb25uZWN0aW9uLmNvbm5lY3Rpb25RdHlcblx0XHR9XG5cdH1cbn1cblxuZXhwb3J0IG1vZHVsZSBOZXVyb24uQ29ubmVjdGlvbiB7XG5cdGV4cG9ydCB2YXIgY29ubmVjdGlvblF0eSA9IDA7XG5cdGV4cG9ydCBmdW5jdGlvbiB1aWQoKTogbnVtYmVyIHtcblx0XHRyZXR1cm4gY29ubmVjdGlvblF0eSsrO1xuXHR9XG59IiwiaW1wb3J0IFN5bmFwdGljID0gcmVxdWlyZSgnLi9zeW5hcHRpYycpO1xuXG4vLyBzcXVhc2hpbmcgZnVuY3Rpb25zXG5cbmV4cG9ydCBmdW5jdGlvbiBMT0dJU1RJQyh4OiBudW1iZXIsIGRlcml2YXRlPzogYm9vbGVhbikge1xuXHRpZiAoIWRlcml2YXRlKVxuXHRcdHJldHVybiAxIC8gKDEgKyBNYXRoLmV4cCgteCkpO1xuXHR2YXIgZnggPSBMT0dJU1RJQyh4KTtcblx0cmV0dXJuIGZ4ICogKDEgLSBmeCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBUQU5IKHg6IG51bWJlciwgZGVyaXZhdGU/OiBib29sZWFuKSB7XG5cdGlmIChkZXJpdmF0ZSlcblx0XHRyZXR1cm4gMSAtIE1hdGgucG93KFRBTkgoeCksIDIpO1xuXHR2YXIgZVAgPSBNYXRoLmV4cCh4KTtcblx0dmFyIGVOID0gMSAvIGVQO1xuXHRyZXR1cm4gKGVQIC0gZU4pIC8gKGVQICsgZU4pO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gSURFTlRJVFkoeDogbnVtYmVyLCBkZXJpdmF0ZT86IGJvb2xlYW4pIHtcblx0cmV0dXJuIGRlcml2YXRlID8gMSA6IHg7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBITElNKHg6IG51bWJlciwgZGVyaXZhdGU/OiBib29sZWFuKSB7XG5cdHJldHVybiBkZXJpdmF0ZSA/IDEgOiArKHggPiAwKTtcbn1cbiIsIi8qXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBTWU5BUFRJQ1xuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcblxuU3luYXB0aWMgaXMgYSBqYXZhc2NyaXB0IG5ldXJhbCBuZXR3b3JrIGxpYnJhcnkgZm9yIG5vZGUuanMgYW5kIHRoZSBicm93c2VyLCBpdHMgZ2VuZXJhbGl6ZWRcbmFsZ29yaXRobSBpcyBhcmNoaXRlY3R1cmUtZnJlZSwgc28geW91IGNhbiBidWlsZCBhbmQgdHJhaW4gYmFzaWNhbGx5IGFueSB0eXBlIG9mIGZpcnN0IG9yZGVyXG5vciBldmVuIHNlY29uZCBvcmRlciBuZXVyYWwgbmV0d29yayBhcmNoaXRlY3R1cmVzLlxuXG5odHRwOi8vZW4ud2lraXBlZGlhLm9yZy93aWtpL1JlY3VycmVudF9uZXVyYWxfbmV0d29yayNTZWNvbmRfT3JkZXJfUmVjdXJyZW50X05ldXJhbF9OZXR3b3JrXG5cblRoZSBsaWJyYXJ5IGluY2x1ZGVzIGEgZmV3IGJ1aWx0LWluIGFyY2hpdGVjdHVyZXMgbGlrZSBtdWx0aWxheWVyIHBlcmNlcHRyb25zLCBtdWx0aWxheWVyXG5sb25nLXNob3J0IHRlcm0gbWVtb3J5IG5ldHdvcmtzIChMU1RNKSBvciBsaXF1aWQgc3RhdGUgbWFjaGluZXMsIGFuZCBhIHRyYWluZXIgY2FwYWJsZSBvZlxudHJhaW5pbmcgYW55IGdpdmVuIG5ldHdvcmssIGFuZCBpbmNsdWRlcyBidWlsdC1pbiB0cmFpbmluZyB0YXNrcy90ZXN0cyBsaWtlIHNvbHZpbmcgYW4gWE9SLFxucGFzc2luZyBhIERpc3RyYWN0ZWQgU2VxdWVuY2UgUmVjYWxsIHRlc3Qgb3IgYW4gRW1iZWRlZCBSZWJlciBHcmFtbWFyIHRlc3QuXG5cblRoZSBhbGdvcml0aG0gaW1wbGVtZW50ZWQgYnkgdGhpcyBsaWJyYXJ5IGhhcyBiZWVuIHRha2VuIGZyb20gRGVyZWsgRC4gTW9ubmVyJ3MgcGFwZXI6XG5cbkEgZ2VuZXJhbGl6ZWQgTFNUTS1saWtlIHRyYWluaW5nIGFsZ29yaXRobSBmb3Igc2Vjb25kLW9yZGVyIHJlY3VycmVudCBuZXVyYWwgbmV0d29ya3Ncbmh0dHA6Ly93d3cub3ZlcmNvbXBsZXRlLm5ldC9wYXBlcnMvbm4yMDEyLnBkZlxuXG5UaGVyZSBhcmUgcmVmZXJlbmNlcyB0byB0aGUgZXF1YXRpb25zIGluIHRoYXQgcGFwZXIgY29tbWVudGVkIHRocm91Z2ggdGhlIHNvdXJjZSBjb2RlLlxuXG5cbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuXG5cblxuaW1wb3J0IG5ldHdvcmsgPSByZXF1aXJlKCcuL25ldHdvcmsnKTtcbmltcG9ydCBsYXllciA9IHJlcXVpcmUoJy4vbGF5ZXInKTtcbmltcG9ydCBuZXVyb24gPSByZXF1aXJlKCcuL25ldXJvbicpO1xuaW1wb3J0IHRyYWluZXIgPSByZXF1aXJlKCcuL3RyYWluZXInKTtcbmltcG9ydCBhcmNoaXRlY3QgPSByZXF1aXJlKCcuL2FyY2hpdGVjdCcpO1xuaW1wb3J0IHNxdWFzaCA9IHJlcXVpcmUoJy4vc3F1YXNoJyk7XG5cbmRlY2xhcmUgdmFyIHdpbmRvdztcblxubW9kdWxlIFN5bmFwdGljIHtcblx0ZXhwb3J0IGludGVyZmFjZSBEaWN0aW9uYXJ5PFQ+IHtcblx0XHRbaWQ6IHN0cmluZ10gOiBUO1xuXHR9XG5cdFxuXHR2YXIgb2xkU3luYXB0aWMgPSB0eXBlb2Ygd2luZG93ICE9IFwidW5kZWZpbmVkXCIgJiYgd2luZG93ICYmIHdpbmRvd1snU3luYXB0aWMnXTtcblx0XG5cdGV4cG9ydCBmdW5jdGlvbiBuaW5qYSgpIHtcbiAgICAgIHdpbmRvd1snc3luYXB0aWMnXSA9IG9sZFN5bmFwdGljOyBcbiAgICAgIHJldHVybiBTeW5hcHRpYztcblx0fVxuXHRcblx0ZXhwb3J0IGludGVyZmFjZSBJQ29tcGlsZWRQYXJhbWV0ZXJzIHtcdFxuXHRcdG1lbW9yeT86IGFueTtcblx0XHRuZXVyb25zPzogbnVtYmVyO1xuXHRcdGlucHV0cz86IGFueVtdO1xuXHRcdG91dHB1dHM/OiBhbnlbXTtcblx0XHR0YXJnZXRzPzogYW55W107XG5cdFx0dmFyaWFibGVzPzogYW55O1xuXHRcdGFjdGl2YXRpb25fc2VudGVuY2VzPzogYW55W107XG5cdFx0dHJhY2Vfc2VudGVuY2VzPzogYW55W107XG5cdFx0cHJvcGFnYXRpb25fc2VudGVuY2VzPzogYW55W107XG5cdFx0bGF5ZXJzPzogYW55O1xuXHR9XG5cdFxuXHRleHBvcnQgaW50ZXJmYWNlIElOdW1lcmljQXJyYXkge1xuXHQgIFtpbmRleDogbnVtYmVyXSA6IG51bWJlcjtcblx0ICBsZW5ndGggOiBudW1iZXI7XG5cdH1cblx0XG5cdGV4cG9ydCB2YXIgTmV1cm9uID0gbmV1cm9uLk5ldXJvbjtcblx0ZXhwb3J0IHZhciBMYXllciA9IGxheWVyLkxheWVyO1xuXHRleHBvcnQgdmFyIE5ldHdvcmsgPSBuZXR3b3JrLk5ldHdvcms7XG5cdGV4cG9ydCB2YXIgVHJhaW5lciA9IHRyYWluZXIuVHJhaW5lcjtcblx0ZXhwb3J0IHZhciBTcXVhc2ggPSBzcXVhc2g7XG5cdGV4cG9ydCB2YXIgQXJjaGl0ZWN0ID0gYXJjaGl0ZWN0O1xufVxuXG5leHBvcnQgPSBTeW5hcHRpYztcblxuaWYodHlwZW9mIHdpbmRvdyAhPSBcInVuZGVmaW5lZFwiKSBcblx0d2luZG93WydzeW5hcHRpYyddID0gU3luYXB0aWM7XG4iLCJpbXBvcnQgbmV0ID0gcmVxdWlyZSgnLi9uZXR3b3JrJyk7XG5cbi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgVFJBSU5FUlxuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKi9cblxuZXhwb3J0IGNsYXNzIFRyYWluZXIge1xuICBuZXR3b3JrOiBuZXQuTmV0d29yaztcbiAgcmF0ZTogYW55ID0gLjI7XG4gIGl0ZXJhdGlvbnMgPSAxMDAwMDA7XG4gIGVycm9yID0gLjAwNTtcbiAgY29zdDogVHJhaW5lci5JVHJhaW5lckNvc3RGbjtcbiAgc2NoZWR1bGU6IGFueTtcblxuICBjb25zdHJ1Y3RvcihuZXR3b3JrOiBuZXQuTmV0d29yaywgb3B0aW9ucz86IGFueSkge1xuICAgIG9wdGlvbnMgPSBvcHRpb25zIHx8IHt9O1xuICAgIHRoaXMubmV0d29yayA9IG5ldHdvcms7XG4gICAgdGhpcy5yYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4yO1xuICAgIHRoaXMuaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucyB8fCAxMDAwMDA7XG4gICAgdGhpcy5lcnJvciA9IG9wdGlvbnMuZXJyb3IgfHwgLjAwNVxuICAgIHRoaXMuY29zdCA9IG9wdGlvbnMuY29zdCB8fCBUcmFpbmVyLmNvc3QuQ1JPU1NfRU5UUk9QWTtcbiAgfVxuXG4gIC8vIHRyYWlucyBhbnkgZ2l2ZW4gc2V0IHRvIGEgbmV0d29ya1xuICB0cmFpbihzZXQsIG9wdGlvbnMpIHtcblxuICAgIHZhciBlcnJvciA9IDE7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSAwLCBidWNrZXRTaXplID0gMDtcbiAgICB2YXIgYWJvcnRfdHJhaW5pbmcgPSBmYWxzZTtcbiAgICB2YXIgaW5wdXQsIG91dHB1dCwgdGFyZ2V0LCBjdXJyZW50UmF0ZTtcblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICBpZiAob3B0aW9ucykge1xuICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSkge1xuICAgICAgICAvLysgSm9uYXMgUmFvbmkgU29hcmVzIFNpbHZhXG4gICAgICAgIC8vQCBodHRwOi8vanNmcm9taGVsbC5jb20vYXJyYXkvc2h1ZmZsZSBbdjEuMF1cbiAgICAgICAgZnVuY3Rpb24gc2h1ZmZsZShvKSB7IC8vdjEuMFxuICAgICAgICAgIGZvciAodmFyIGosIHgsIGkgPSBvLmxlbmd0aDsgaTsgaiA9IE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqIGkpLCB4ID0gb1stLWldLCBvW2ldID0gb1tqXSwgb1tqXSA9IHgpO1xuICAgICAgICAgIHJldHVybiBvO1xuICAgICAgICB9O1xuICAgICAgfVxuICAgICAgaWYgKG9wdGlvbnMuaXRlcmF0aW9ucylcbiAgICAgICAgdGhpcy5pdGVyYXRpb25zID0gb3B0aW9ucy5pdGVyYXRpb25zO1xuICAgICAgaWYgKG9wdGlvbnMuZXJyb3IpXG4gICAgICAgIHRoaXMuZXJyb3IgPSBvcHRpb25zLmVycm9yO1xuICAgICAgaWYgKG9wdGlvbnMucmF0ZSlcbiAgICAgICAgdGhpcy5yYXRlID0gb3B0aW9ucy5yYXRlO1xuICAgICAgaWYgKG9wdGlvbnMuY29zdClcbiAgICAgICAgdGhpcy5jb3N0ID0gb3B0aW9ucy5jb3N0O1xuICAgICAgaWYgKG9wdGlvbnMuc2NoZWR1bGUpXG4gICAgICAgIHRoaXMuc2NoZWR1bGUgPSBvcHRpb25zLnNjaGVkdWxlO1xuICAgICAgaWYgKG9wdGlvbnMuY3VzdG9tTG9nKSB7XG4gICAgICAgIC8vIGZvciBiYWNrd2FyZCBjb21wYXRpYmlsaXR5IHdpdGggY29kZSB0aGF0IHVzZWQgY3VzdG9tTG9nXG4gICAgICAgIGNvbnNvbGUubG9nKCdEZXByZWNhdGVkOiB1c2Ugc2NoZWR1bGUgaW5zdGVhZCBvZiBjdXN0b21Mb2cnKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5jdXN0b21Mb2c7XG4gICAgICB9XG4gICAgfVxuXG4gICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGU7XG4gICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5yYXRlKSkge1xuICAgICAgYnVja2V0U2l6ZSA9IE1hdGguZmxvb3IodGhpcy5pdGVyYXRpb25zIC8gdGhpcy5yYXRlLmxlbmd0aCk7XG4gICAgfVxuXG5cbiAgICB3aGlsZSAoIWFib3J0X3RyYWluaW5nICYmIGl0ZXJhdGlvbnMgPCB0aGlzLml0ZXJhdGlvbnMgJiYgZXJyb3IgPiB0aGlzLmVycm9yKSB7XG4gICAgICBlcnJvciA9IDA7XG5cbiAgICAgIGlmIChidWNrZXRTaXplID4gMCkge1xuICAgICAgICB2YXIgY3VycmVudEJ1Y2tldCA9IE1hdGguZmxvb3IoaXRlcmF0aW9ucyAvIGJ1Y2tldFNpemUpO1xuICAgICAgICBjdXJyZW50UmF0ZSA9IHRoaXMucmF0ZVtjdXJyZW50QnVja2V0XTtcbiAgICAgIH1cblxuICAgICAgZm9yICh2YXIgdHJhaW4gaW4gc2V0KSB7XG4gICAgICAgIGlucHV0ID0gc2V0W3RyYWluXS5pbnB1dDtcbiAgICAgICAgdGFyZ2V0ID0gc2V0W3RyYWluXS5vdXRwdXQ7XG5cbiAgICAgICAgb3V0cHV0ID0gdGhpcy5uZXR3b3JrLmFjdGl2YXRlKGlucHV0KTtcbiAgICAgICAgdGhpcy5uZXR3b3JrLnByb3BhZ2F0ZShjdXJyZW50UmF0ZSwgdGFyZ2V0KTtcblxuICAgICAgICBlcnJvciArPSB0aGlzLmNvc3QodGFyZ2V0LCBvdXRwdXQpO1xuICAgICAgfVxuXG4gICAgICAvLyBjaGVjayBlcnJvclxuICAgICAgaXRlcmF0aW9ucysrO1xuICAgICAgZXJyb3IgLz0gc2V0Lmxlbmd0aDtcblxuICAgICAgaWYgKG9wdGlvbnMpIHtcbiAgICAgICAgaWYgKHRoaXMuc2NoZWR1bGUgJiYgdGhpcy5zY2hlZHVsZS5ldmVyeSAmJiBpdGVyYXRpb25zICUgdGhpcy5zY2hlZHVsZS5ldmVyeSA9PSAwKSB7XG5cbiAgICAgICAgICBhYm9ydF90cmFpbmluZyA9IHRoaXMuc2NoZWR1bGUuZG8oe1xuICAgICAgICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgICAgICAgaXRlcmF0aW9uczogaXRlcmF0aW9ucyxcbiAgICAgICAgICAgIHJhdGU6IGN1cnJlbnRSYXRlXG4gICAgICAgICAgfSk7XG5cbiAgICAgICAgICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoKCkgPT4ge1xuICAgICAgICAgICAgdGhpcy5pdGVyYXRpb25zIC09IGl0ZXJhdGlvbnM7XG4gICAgICAgICAgICB0aGlzLnRyYWluKHNldCwgb3B0aW9ucyk7XG4gICAgICAgICAgfSk7XG4gICAgICAgICAgcmV0dXJuO1xuICAgICAgICB9IGVsc2UgaWYgKG9wdGlvbnMubG9nICYmIGl0ZXJhdGlvbnMgJSBvcHRpb25zLmxvZyA9PSAwKSB7XG4gICAgICAgICAgY29uc29sZS5sb2coJ2l0ZXJhdGlvbnMnLCBpdGVyYXRpb25zLCAnZXJyb3InLCBlcnJvciwgJ3JhdGUnLCBjdXJyZW50UmF0ZSk7XG4gICAgICAgIH07XG4gICAgICAgIGlmIChvcHRpb25zLnNodWZmbGUpXG4gICAgICAgICAgc2h1ZmZsZShzZXQpO1xuICAgICAgfVxuICAgIH1cblxuICAgIHZhciByZXN1bHRzID0ge1xuICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgaXRlcmF0aW9uczogaXRlcmF0aW9ucyxcbiAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydFxuICAgIH1cblxuICAgIHJldHVybiByZXN1bHRzO1xuICB9XG5cbiAgLy8gdHJhaW5zIGFueSBnaXZlbiBzZXQgdG8gYSBuZXR3b3JrIHVzaW5nIGEgV2ViV29ya2VyXG4gIHdvcmtlclRyYWluKHNldCwgY2FsbGJhY2ssIG9wdGlvbnMpIHtcblxuICAgIHZhciB0aGF0ID0gdGhpcztcbiAgICB2YXIgZXJyb3IgPSAxO1xuICAgIHZhciBpdGVyYXRpb25zID0gMCwgYnVja2V0U2l6ZSA9IDA7XG4gICAgdmFyIGlucHV0LCBvdXRwdXQsIHRhcmdldCwgY3VycmVudFJhdGU7XG4gICAgdmFyIGxlbmd0aCA9IHNldC5sZW5ndGg7XG4gICAgdmFyIGFib3J0X3RyYWluaW5nID0gZmFsc2U7XG5cbiAgICB2YXIgc3RhcnQgPSBEYXRlLm5vdygpO1xuXG4gICAgaWYgKG9wdGlvbnMpIHtcbiAgICAgIGlmIChvcHRpb25zLnNodWZmbGUpIHtcbiAgICAgICAgLy8rIEpvbmFzIFJhb25pIFNvYXJlcyBTaWx2YVxuICAgICAgICAvL0AgaHR0cDovL2pzZnJvbWhlbGwuY29tL2FycmF5L3NodWZmbGUgW3YxLjBdXG4gICAgICAgIGZ1bmN0aW9uIHNodWZmbGUobykgeyAvL3YxLjBcbiAgICAgICAgICBmb3IgKHZhciBqLCB4LCBpID0gby5sZW5ndGg7IGk7IGogPSBNYXRoLmZsb29yKE1hdGgucmFuZG9tKCkgKlxuICAgICAgICAgICAgaSksIHggPSBvWy0taV0sIG9baV0gPSBvW2pdLCBvW2pdID0geCk7XG4gICAgICAgICAgcmV0dXJuIG87XG4gICAgICAgIH07XG4gICAgICB9XG4gICAgICBpZiAob3B0aW9ucy5pdGVyYXRpb25zKVxuICAgICAgICB0aGlzLml0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnM7XG4gICAgICBpZiAob3B0aW9ucy5lcnJvcilcbiAgICAgICAgdGhpcy5lcnJvciA9IG9wdGlvbnMuZXJyb3I7XG4gICAgICBpZiAob3B0aW9ucy5yYXRlKVxuICAgICAgICB0aGlzLnJhdGUgPSBvcHRpb25zLnJhdGU7XG4gICAgICBpZiAob3B0aW9ucy5jb3N0KVxuICAgICAgICB0aGlzLmNvc3QgPSBvcHRpb25zLmNvc3Q7XG4gICAgICBpZiAob3B0aW9ucy5zY2hlZHVsZSlcbiAgICAgICAgdGhpcy5zY2hlZHVsZSA9IG9wdGlvbnMuc2NoZWR1bGU7XG4gICAgICBpZiAob3B0aW9ucy5jdXN0b21Mb2cpXG4gICAgICAgIC8vIGZvciBiYWNrd2FyZCBjb21wYXRpYmlsaXR5IHdpdGggY29kZSB0aGF0IHVzZWQgY3VzdG9tTG9nXG4gICAgICAgIGNvbnNvbGUubG9nKCdEZXByZWNhdGVkOiB1c2Ugc2NoZWR1bGUgaW5zdGVhZCBvZiBjdXN0b21Mb2cnKVxuICAgICAgdGhpcy5zY2hlZHVsZSA9IG9wdGlvbnMuY3VzdG9tTG9nO1xuICAgIH1cblxuICAgIC8vIGR5bmFtaWMgbGVhcm5pbmcgcmF0ZVxuICAgIGN1cnJlbnRSYXRlID0gdGhpcy5yYXRlO1xuICAgIGlmIChBcnJheS5pc0FycmF5KHRoaXMucmF0ZSkpIHtcbiAgICAgIGJ1Y2tldFNpemUgPSBNYXRoLmZsb29yKHRoaXMuaXRlcmF0aW9ucyAvIHRoaXMucmF0ZS5sZW5ndGgpO1xuICAgIH1cblxuICAgIC8vIGNyZWF0ZSBhIHdvcmtlclxuICAgIHZhciB3b3JrZXIgPSB0aGlzLm5ldHdvcmsud29ya2VyKCk7XG5cbiAgICAvLyBhY3RpdmF0ZSB0aGUgbmV0d29ya1xuICAgIGZ1bmN0aW9uIGFjdGl2YXRlV29ya2VyKGlucHV0KSB7XG4gICAgICB3b3JrZXIucG9zdE1lc3NhZ2Uoe1xuICAgICAgICBhY3Rpb246IFwiYWN0aXZhdGVcIixcbiAgICAgICAgaW5wdXQ6IGlucHV0LFxuICAgICAgICBtZW1vcnlCdWZmZXI6IHRoYXQubmV0d29yay5vcHRpbWl6ZWQubWVtb3J5XG4gICAgICB9LCBbdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnkuYnVmZmVyXSk7XG4gICAgfVxuXG4gICAgLy8gYmFja3Byb3BhZ2F0ZSB0aGUgbmV0d29ya1xuICAgIGZ1bmN0aW9uIHByb3BhZ2F0ZVdvcmtlcih0YXJnZXQpIHtcbiAgICAgIGlmIChidWNrZXRTaXplID4gMCkge1xuICAgICAgICB2YXIgY3VycmVudEJ1Y2tldCA9IE1hdGguZmxvb3IoaXRlcmF0aW9ucyAvIGJ1Y2tldFNpemUpO1xuICAgICAgICBjdXJyZW50UmF0ZSA9IHRoaXMucmF0ZVtjdXJyZW50QnVja2V0XTtcbiAgICAgIH1cbiAgICAgIHdvcmtlci5wb3N0TWVzc2FnZSh7XG4gICAgICAgIGFjdGlvbjogXCJwcm9wYWdhdGVcIixcbiAgICAgICAgdGFyZ2V0OiB0YXJnZXQsXG4gICAgICAgIHJhdGU6IGN1cnJlbnRSYXRlLFxuICAgICAgICBtZW1vcnlCdWZmZXI6IHRoYXQubmV0d29yay5vcHRpbWl6ZWQubWVtb3J5XG4gICAgICB9LCBbdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnkuYnVmZmVyXSk7XG4gICAgfVxuXG4gICAgLy8gdHJhaW4gdGhlIHdvcmtlclxuICAgIHdvcmtlci5vbm1lc3NhZ2UgPSBmdW5jdGlvbihlKSB7XG4gICAgICAvLyBnaXZlIGNvbnRyb2wgb2YgdGhlIG1lbW9yeSBiYWNrIHRvIHRoZSBuZXR3b3JrXG4gICAgICB0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm93bmVyc2hpcChlLmRhdGEubWVtb3J5QnVmZmVyKTtcblxuICAgICAgaWYgKGUuZGF0YS5hY3Rpb24gPT0gXCJwcm9wYWdhdGVcIikge1xuICAgICAgICBpZiAoaW5kZXggPj0gbGVuZ3RoKSB7XG4gICAgICAgICAgaW5kZXggPSAwO1xuICAgICAgICAgIGl0ZXJhdGlvbnMrKztcbiAgICAgICAgICBlcnJvciAvPSBzZXQubGVuZ3RoO1xuXG4gICAgICAgICAgLy8gbG9nXG4gICAgICAgICAgaWYgKG9wdGlvbnMpIHtcbiAgICAgICAgICAgIGlmICh0aGlzLnNjaGVkdWxlICYmIHRoaXMuc2NoZWR1bGUuZXZlcnkgJiYgaXRlcmF0aW9ucyAlIHRoaXMuc2NoZWR1bGUuZXZlcnkgPT0gMClcbiAgICAgICAgICAgICAgYWJvcnRfdHJhaW5pbmcgPSB0aGlzLnNjaGVkdWxlLmRvKHtcbiAgICAgICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICAgICAgaXRlcmF0aW9uczogaXRlcmF0aW9uc1xuICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgIGVsc2UgaWYgKG9wdGlvbnMubG9nICYmIGl0ZXJhdGlvbnMgJSBvcHRpb25zLmxvZyA9PSAwKSB7XG4gICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdpdGVyYXRpb25zJywgaXRlcmF0aW9ucywgJ2Vycm9yJywgZXJyb3IpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIGlmIChvcHRpb25zLnNodWZmbGUpXG4gICAgICAgICAgICAgIHNodWZmbGUoc2V0KTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBpZiAoIWFib3J0X3RyYWluaW5nICYmIGl0ZXJhdGlvbnMgPCB0aGF0Lml0ZXJhdGlvbnMgJiYgZXJyb3IgPiB0aGF0LmVycm9yKSB7XG4gICAgICAgICAgICBhY3RpdmF0ZVdvcmtlcihzZXRbaW5kZXhdLmlucHV0KTtcbiAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgLy8gY2FsbGJhY2tcbiAgICAgICAgICAgIGNhbGxiYWNrKHtcbiAgICAgICAgICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgICAgICAgICB0aW1lOiBEYXRlLm5vdygpIC0gc3RhcnRcbiAgICAgICAgICAgIH0pXG4gICAgICAgICAgfVxuICAgICAgICAgIGVycm9yID0gMDtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBhY3RpdmF0ZVdvcmtlcihzZXRbaW5kZXhdLmlucHV0KTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICBpZiAoZS5kYXRhLmFjdGlvbiA9PSBcImFjdGl2YXRlXCIpIHtcbiAgICAgICAgZXJyb3IgKz0gdGhhdC5jb3N0KHNldFtpbmRleF0ub3V0cHV0LCBlLmRhdGEub3V0cHV0KTtcbiAgICAgICAgcHJvcGFnYXRlV29ya2VyKHNldFtpbmRleF0ub3V0cHV0KTtcbiAgICAgICAgaW5kZXgrKztcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBraWNrIGl0XG4gICAgdmFyIGluZGV4ID0gMDtcbiAgICB2YXIgaXRlcmF0aW9ucyA9IDA7XG4gICAgYWN0aXZhdGVXb3JrZXIoc2V0W2luZGV4XS5pbnB1dCk7XG4gIH1cblxuICAvLyB0cmFpbnMgYW4gWE9SIHRvIHRoZSBuZXR3b3JrXG4gIFhPUihvcHRpb25zKSB7XG5cbiAgICBpZiAodGhpcy5uZXR3b3JrLmlucHV0cygpICE9IDIgfHwgdGhpcy5uZXR3b3JrLm91dHB1dHMoKSAhPSAxKVxuICAgICAgdGhyb3cgXCJFcnJvcjogSW5jb21wYXRpYmxlIG5ldHdvcmsgKDIgaW5wdXRzLCAxIG91dHB1dClcIjtcblxuICAgIHZhciBkZWZhdWx0cyA9IHtcbiAgICAgIGl0ZXJhdGlvbnM6IDEwMDAwMCxcbiAgICAgIGxvZzogZmFsc2UsXG4gICAgICBzaHVmZmxlOiB0cnVlLFxuICAgICAgY29zdDogVHJhaW5lci5jb3N0Lk1TRVxuICAgIH1cblxuICAgIGlmIChvcHRpb25zKVxuICAgICAgZm9yICh2YXIgaSBpbiBvcHRpb25zKVxuICAgICAgICBkZWZhdWx0c1tpXSA9IG9wdGlvbnNbaV07XG5cbiAgICByZXR1cm4gdGhpcy50cmFpbihbe1xuICAgICAgaW5wdXQ6IFswLCAwXSxcbiAgICAgIG91dHB1dDogWzBdXG4gICAgfSwge1xuICAgICAgICBpbnB1dDogWzEsIDBdLFxuICAgICAgICBvdXRwdXQ6IFsxXVxuICAgICAgfSwge1xuICAgICAgICBpbnB1dDogWzAsIDFdLFxuICAgICAgICBvdXRwdXQ6IFsxXVxuICAgICAgfSwge1xuICAgICAgICBpbnB1dDogWzEsIDFdLFxuICAgICAgICBvdXRwdXQ6IFswXVxuICAgICAgfV0sIGRlZmF1bHRzKTtcbiAgfVxuXG4gIC8vIHRyYWlucyB0aGUgbmV0d29yayB0byBwYXNzIGEgRGlzdHJhY3RlZCBTZXF1ZW5jZSBSZWNhbGwgdGVzdFxuICBEU1Iob3B0aW9ucykge1xuICAgIG9wdGlvbnMgPSBvcHRpb25zIHx8IHt9O1xuXG4gICAgdmFyIHRhcmdldHMgPSBvcHRpb25zLnRhcmdldHMgfHwgWzIsIDQsIDcsIDhdO1xuICAgIHZhciBkaXN0cmFjdG9ycyA9IG9wdGlvbnMuZGlzdHJhY3RvcnMgfHwgWzMsIDUsIDYsIDldO1xuICAgIHZhciBwcm9tcHRzID0gb3B0aW9ucy5wcm9tcHRzIHx8IFswLCAxXTtcbiAgICB2YXIgbGVuZ3RoID0gb3B0aW9ucy5sZW5ndGggfHwgMjQ7XG4gICAgdmFyIGNyaXRlcmlvbiA9IG9wdGlvbnMuc3VjY2VzcyB8fCAwLjk1O1xuICAgIHZhciBpdGVyYXRpb25zID0gb3B0aW9ucy5pdGVyYXRpb25zIHx8IDEwMDAwMDtcbiAgICB2YXIgcmF0ZSA9IG9wdGlvbnMucmF0ZSB8fCAuMTtcbiAgICB2YXIgbG9nID0gb3B0aW9ucy5sb2cgfHwgMDtcbiAgICB2YXIgc2NoZWR1bGUgPSBvcHRpb25zLnNjaGVkdWxlIHx8IHt9O1xuICAgIHZhciBjb3JyZWN0ID0gMDtcbiAgICB2YXIgaSA9IDA7XG4gICAgdmFyIHN1Y2Nlc3MgPSAwO1xuICAgIHZhciB0cmlhbCA9IGkgPSBjb3JyZWN0ID0gaiA9IHN1Y2Nlc3MgPSAwLFxuICAgICAgZXJyb3IgPSAxLFxuICAgICAgc3ltYm9scyA9IHRhcmdldHMubGVuZ3RoICsgZGlzdHJhY3RvcnMubGVuZ3RoICsgcHJvbXB0cy5sZW5ndGg7XG5cbiAgICB2YXIgbm9SZXBlYXQgPSBmdW5jdGlvbihyYW5nZSwgYXZvaWQpIHtcbiAgICAgIHZhciBudW1iZXIgPSBNYXRoLnJhbmRvbSgpICogcmFuZ2UgfCAwO1xuICAgICAgdmFyIHVzZWQgPSBmYWxzZTtcbiAgICAgIGZvciAodmFyIGkgaW4gYXZvaWQpXG4gICAgICAgIGlmIChudW1iZXIgPT0gYXZvaWRbaV0pXG4gICAgICAgICAgdXNlZCA9IHRydWU7XG4gICAgICByZXR1cm4gdXNlZCA/IG5vUmVwZWF0KHJhbmdlLCBhdm9pZCkgOiBudW1iZXI7XG4gICAgfVxuXG4gICAgdmFyIGVxdWFsID0gZnVuY3Rpb24ocHJlZGljdGlvbiwgb3V0cHV0KSB7XG4gICAgICBmb3IgKHZhciBpIGluIHByZWRpY3Rpb24pXG4gICAgICAgIGlmIChNYXRoLnJvdW5kKHByZWRpY3Rpb25baV0pICE9IG91dHB1dFtpXSlcbiAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG5cbiAgICB2YXIgc3RhcnQgPSBEYXRlLm5vdygpO1xuXG4gICAgd2hpbGUgKHRyaWFsIDwgaXRlcmF0aW9ucyAmJiAoc3VjY2VzcyA8IGNyaXRlcmlvbiB8fCB0cmlhbCAlIDEwMDAgIT0gMCkpIHtcbiAgICAgIC8vIGdlbmVyYXRlIHNlcXVlbmNlXG4gICAgICB2YXIgc2VxdWVuY2UgPSBbXSxcbiAgICAgICAgc2VxdWVuY2VMZW5ndGggPSBsZW5ndGggLSBwcm9tcHRzLmxlbmd0aDtcbiAgICAgIGZvciAoaSA9IDA7IGkgPCBzZXF1ZW5jZUxlbmd0aDsgaSsrKSB7XG4gICAgICAgIHZhciBhbnkgPSBNYXRoLnJhbmRvbSgpICogZGlzdHJhY3RvcnMubGVuZ3RoIHwgMDtcbiAgICAgICAgc2VxdWVuY2UucHVzaChkaXN0cmFjdG9yc1thbnldKTtcbiAgICAgIH1cbiAgICAgIHZhciBpbmRleGVzID0gW10sXG4gICAgICAgIHBvc2l0aW9ucyA9IFtdO1xuICAgICAgZm9yIChpID0gMDsgaSA8IHByb21wdHMubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgaW5kZXhlcy5wdXNoKE1hdGgucmFuZG9tKCkgKiB0YXJnZXRzLmxlbmd0aCB8IDApO1xuICAgICAgICBwb3NpdGlvbnMucHVzaChub1JlcGVhdChzZXF1ZW5jZUxlbmd0aCwgcG9zaXRpb25zKSk7XG4gICAgICB9XG4gICAgICBwb3NpdGlvbnMgPSBwb3NpdGlvbnMuc29ydCgpO1xuICAgICAgZm9yIChpID0gMDsgaSA8IHByb21wdHMubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgc2VxdWVuY2VbcG9zaXRpb25zW2ldXSA9IHRhcmdldHNbaW5kZXhlc1tpXV07XG4gICAgICAgIHNlcXVlbmNlLnB1c2gocHJvbXB0c1tpXSk7XG4gICAgICB9XG5cbiAgICAgIC8vdHJhaW4gc2VxdWVuY2VcbiAgICAgIHZhciBkaXN0cmFjdG9yc0NvcnJlY3Q7XG4gICAgICB2YXIgdGFyZ2V0c0NvcnJlY3QgPSBkaXN0cmFjdG9yc0NvcnJlY3QgPSAwO1xuICAgICAgZXJyb3IgPSAwO1xuICAgICAgZm9yIChpID0gMDsgaSA8IGxlbmd0aDsgaSsrKSB7XG4gICAgICAgIC8vIGdlbmVyYXRlIGlucHV0IGZyb20gc2VxdWVuY2VcbiAgICAgICAgdmFyIGlucHV0ID0gW107XG4gICAgICAgIGZvciAoaiA9IDA7IGogPCBzeW1ib2xzOyBqKyspXG4gICAgICAgICAgaW5wdXRbal0gPSAwO1xuICAgICAgICBpbnB1dFtzZXF1ZW5jZVtpXV0gPSAxO1xuXG4gICAgICAgIC8vIGdlbmVyYXRlIHRhcmdldCBvdXRwdXRcbiAgICAgICAgdmFyIG91dHB1dCA9IFtdO1xuICAgICAgICBmb3IgKGogPSAwOyBqIDwgdGFyZ2V0cy5sZW5ndGg7IGorKylcbiAgICAgICAgICBvdXRwdXRbal0gPSAwO1xuXG4gICAgICAgIGlmIChpID49IHNlcXVlbmNlTGVuZ3RoKSB7XG4gICAgICAgICAgdmFyIGluZGV4ID0gaSAtIHNlcXVlbmNlTGVuZ3RoO1xuICAgICAgICAgIG91dHB1dFtpbmRleGVzW2luZGV4XV0gPSAxO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gY2hlY2sgcmVzdWx0XG4gICAgICAgIHZhciBwcmVkaWN0aW9uID0gdGhpcy5uZXR3b3JrLmFjdGl2YXRlKGlucHV0KTtcblxuICAgICAgICBpZiAoZXF1YWwocHJlZGljdGlvbiwgb3V0cHV0KSlcbiAgICAgICAgICBpZiAoaSA8IHNlcXVlbmNlTGVuZ3RoKVxuICAgICAgICAgICAgZGlzdHJhY3RvcnNDb3JyZWN0Kys7XG4gICAgICAgICAgZWxzZVxuICAgICAgICAgICAgdGFyZ2V0c0NvcnJlY3QrKztcbiAgICAgICAgZWxzZSB7XG4gICAgICAgICAgdGhpcy5uZXR3b3JrLnByb3BhZ2F0ZShyYXRlLCBvdXRwdXQpO1xuICAgICAgICB9XG5cbiAgICAgICAgdmFyIGRlbHRhID0gMDtcbiAgICAgICAgZm9yICh2YXIgaiBpbiBwcmVkaWN0aW9uKVxuICAgICAgICAgIGRlbHRhICs9IE1hdGgucG93KG91dHB1dFtqXSAtIHByZWRpY3Rpb25bal0sIDIpO1xuICAgICAgICBlcnJvciArPSBkZWx0YSAvIHRoaXMubmV0d29yay5vdXRwdXRzKCk7XG5cbiAgICAgICAgaWYgKGRpc3RyYWN0b3JzQ29ycmVjdCArIHRhcmdldHNDb3JyZWN0ID09IGxlbmd0aClcbiAgICAgICAgICBjb3JyZWN0Kys7XG4gICAgICB9XG5cbiAgICAgIC8vIGNhbGN1bGF0ZSBlcnJvclxuICAgICAgaWYgKHRyaWFsICUgMTAwMCA9PSAwKVxuICAgICAgICBjb3JyZWN0ID0gMDtcbiAgICAgIHRyaWFsKys7XG4gICAgICB2YXIgZGl2aWRlRXJyb3IgPSB0cmlhbCAlIDEwMDA7XG4gICAgICBkaXZpZGVFcnJvciA9IGRpdmlkZUVycm9yID09IDAgPyAxMDAwIDogZGl2aWRlRXJyb3I7XG4gICAgICBzdWNjZXNzID0gY29ycmVjdCAvIGRpdmlkZUVycm9yO1xuICAgICAgZXJyb3IgLz0gbGVuZ3RoO1xuXG4gICAgICAvLyBsb2dcbiAgICAgIGlmIChsb2cgJiYgdHJpYWwgJSBsb2cgPT0gMClcbiAgICAgICAgY29uc29sZS5sb2coXCJpdGVyYXRpb25zOlwiLCB0cmlhbCwgXCIgc3VjY2VzczpcIiwgc3VjY2VzcywgXCIgY29ycmVjdDpcIixcbiAgICAgICAgICBjb3JyZWN0LCBcIiB0aW1lOlwiLCBEYXRlLm5vdygpIC0gc3RhcnQsIFwiIGVycm9yOlwiLCBlcnJvcik7XG4gICAgICBpZiAoc2NoZWR1bGUuZG8gJiYgc2NoZWR1bGUuZXZlcnkgJiYgdHJpYWwgJSBzY2hlZHVsZS5ldmVyeSA9PSAwKSB7XG4gICAgICAgIHNjaGVkdWxlLmRvKHtcbiAgICAgICAgICBpdGVyYXRpb25zOiB0cmlhbCxcbiAgICAgICAgICBzdWNjZXNzOiBzdWNjZXNzLFxuICAgICAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgICAgICB0aW1lOiBEYXRlLm5vdygpIC0gc3RhcnQsXG4gICAgICAgICAgY29ycmVjdDogY29ycmVjdFxuICAgICAgICB9KTtcblxuICAgICAgICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoKCkgPT4ge1xuICAgICAgICAgIHRoaXMuaXRlcmF0aW9ucyAtPSB0cmlhbDtcbiAgICAgICAgICB0aGlzLkRTUihvcHRpb25zKTtcbiAgICAgICAgfSk7XG5cbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICBpdGVyYXRpb25zOiB0cmlhbCxcbiAgICAgIHN1Y2Nlc3M6IHN1Y2Nlc3MsXG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICB0aW1lOiBEYXRlLm5vdygpIC0gc3RhcnRcbiAgICB9XG4gIH1cblxuICAvLyB0cmFpbiB0aGUgbmV0d29yayB0byBsZWFybiBhbiBFbWJlZGVkIFJlYmVyIEdyYW1tYXJcbiAgRVJHKG9wdGlvbnMpIHtcblxuICAgIG9wdGlvbnMgPSBvcHRpb25zIHx8IHt9O1xuICAgIHZhciBpdGVyYXRpb25zID0gb3B0aW9ucy5pdGVyYXRpb25zIHx8IDE1MDAwMDtcbiAgICB2YXIgY3JpdGVyaW9uID0gb3B0aW9ucy5lcnJvciB8fCAuMDU7XG4gICAgdmFyIHJhdGUgPSBvcHRpb25zLnJhdGUgfHwgLjE7XG4gICAgdmFyIGxvZyA9IG9wdGlvbnMubG9nIHx8IDUwMDtcblxuICAgIC8vIGdyYW1hciBub2RlXG4gICAgdmFyIE5vZGUgPSBmdW5jdGlvbigpIHtcbiAgICAgIHRoaXMucGF0aHMgPSBbXTtcbiAgICB9XG4gICAgTm9kZS5wcm90b3R5cGUgPSB7XG4gICAgICBjb25uZWN0OiBmdW5jdGlvbihub2RlLCB2YWx1ZSkge1xuICAgICAgICB0aGlzLnBhdGhzLnB1c2goe1xuICAgICAgICAgIG5vZGU6IG5vZGUsXG4gICAgICAgICAgdmFsdWU6IHZhbHVlXG4gICAgICAgIH0pO1xuICAgICAgICByZXR1cm4gdGhpcztcbiAgICAgIH0sXG4gICAgICBhbnk6IGZ1bmN0aW9uKCkge1xuICAgICAgICBpZiAodGhpcy5wYXRocy5sZW5ndGggPT0gMClcbiAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgIHZhciBpbmRleCA9IE1hdGgucmFuZG9tKCkgKiB0aGlzLnBhdGhzLmxlbmd0aCB8IDA7XG4gICAgICAgIHJldHVybiB0aGlzLnBhdGhzW2luZGV4XTtcbiAgICAgIH0sXG4gICAgICB0ZXN0OiBmdW5jdGlvbih2YWx1ZSkge1xuICAgICAgICBmb3IgKHZhciBpIGluIHRoaXMucGF0aHMpXG4gICAgICAgICAgaWYgKHRoaXMucGF0aHNbaV0udmFsdWUgPT0gdmFsdWUpXG4gICAgICAgICAgICByZXR1cm4gdGhpcy5wYXRoc1tpXTtcbiAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgfVxuICAgIH1cblxuICAgIHZhciByZWJlckdyYW1tYXIgPSBmdW5jdGlvbigpIHtcblxuICAgICAgLy8gYnVpbGQgYSByZWJlciBncmFtbWFyXG4gICAgICB2YXIgb3V0cHV0ID0gbmV3IE5vZGUoKTtcbiAgICAgIHZhciBuMSA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG91dHB1dCwgXCJFXCIpO1xuICAgICAgdmFyIG4yID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjEsIFwiU1wiKTtcbiAgICAgIHZhciBuMyA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4xLCBcIlZcIikuY29ubmVjdChuMiwgXCJQXCIpO1xuICAgICAgdmFyIG40ID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjIsIFwiWFwiKVxuICAgICAgbjQuY29ubmVjdChuNCwgXCJTXCIpO1xuICAgICAgdmFyIG41ID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjMsIFwiVlwiKVxuICAgICAgbjUuY29ubmVjdChuNSwgXCJUXCIpO1xuICAgICAgbjIuY29ubmVjdChuNSwgXCJYXCIpXG4gICAgICB2YXIgbjYgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuNCwgXCJUXCIpLmNvbm5lY3QobjUsIFwiUFwiKTtcbiAgICAgIHZhciBpbnB1dCA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG42LCBcIkJcIilcblxuICAgICAgcmV0dXJuIHtcbiAgICAgICAgaW5wdXQ6IGlucHV0LFxuICAgICAgICBvdXRwdXQ6IG91dHB1dFxuICAgICAgfVxuICAgIH1cblxuICAgIC8vIGJ1aWxkIGFuIGVtYmVkZWQgcmViZXIgZ3JhbW1hclxuICAgIHZhciBlbWJlZGVkUmViZXJHcmFtbWFyID0gZnVuY3Rpb24oKSB7XG4gICAgICB2YXIgcmViZXIxID0gcmViZXJHcmFtbWFyKCk7XG4gICAgICB2YXIgcmViZXIyID0gcmViZXJHcmFtbWFyKCk7XG5cbiAgICAgIHZhciBvdXRwdXQgPSBuZXcgTm9kZSgpO1xuICAgICAgdmFyIG4xID0gKG5ldyBOb2RlKS5jb25uZWN0KG91dHB1dCwgXCJFXCIpO1xuICAgICAgcmViZXIxLm91dHB1dC5jb25uZWN0KG4xLCBcIlRcIik7XG4gICAgICByZWJlcjIub3V0cHV0LmNvbm5lY3QobjEsIFwiUFwiKTtcbiAgICAgIHZhciBuMiA9IChuZXcgTm9kZSkuY29ubmVjdChyZWJlcjEuaW5wdXQsIFwiUFwiKS5jb25uZWN0KHJlYmVyMi5pbnB1dCxcbiAgICAgICAgXCJUXCIpO1xuICAgICAgdmFyIGlucHV0ID0gKG5ldyBOb2RlKS5jb25uZWN0KG4yLCBcIkJcIik7XG5cbiAgICAgIHJldHVybiB7XG4gICAgICAgIGlucHV0OiBpbnB1dCxcbiAgICAgICAgb3V0cHV0OiBvdXRwdXRcbiAgICAgIH1cblxuICAgIH1cblxuICAgIC8vIGdlbmVyYXRlIGFuIEVSRyBzZXF1ZW5jZVxuICAgIHZhciBnZW5lcmF0ZSA9IGZ1bmN0aW9uKCkge1xuICAgICAgdmFyIG5vZGUgPSBlbWJlZGVkUmViZXJHcmFtbWFyKCkuaW5wdXQ7XG4gICAgICB2YXIgbmV4dCA9IG5vZGUuYW55KCk7XG4gICAgICB2YXIgc3RyID0gXCJcIjtcbiAgICAgIHdoaWxlIChuZXh0KSB7XG4gICAgICAgIHN0ciArPSBuZXh0LnZhbHVlO1xuICAgICAgICBuZXh0ID0gbmV4dC5ub2RlLmFueSgpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHN0cjtcbiAgICB9XG5cbiAgICAvLyB0ZXN0IGlmIGEgc3RyaW5nIG1hdGNoZXMgYW4gZW1iZWRlZCByZWJlciBncmFtbWFyXG4gICAgdmFyIHRlc3QgPSBmdW5jdGlvbihzdHIpIHtcbiAgICAgIHZhciBub2RlID0gZW1iZWRlZFJlYmVyR3JhbW1hcigpLmlucHV0O1xuICAgICAgdmFyIGkgPSAwO1xuICAgICAgdmFyIGNoID0gc3RyLmNoYXJBdChpKTtcbiAgICAgIHdoaWxlIChpIDwgc3RyLmxlbmd0aCkge1xuICAgICAgICB2YXIgbmV4dCA9IG5vZGUudGVzdChjaCk7XG4gICAgICAgIGlmICghbmV4dClcbiAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgIG5vZGUgPSBuZXh0Lm5vZGU7XG4gICAgICAgIGNoID0gc3RyLmNoYXJBdCgrK2kpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuXG4gICAgLy8gaGVscGVyIHRvIGNoZWNrIGlmIHRoZSBvdXRwdXQgYW5kIHRoZSB0YXJnZXQgdmVjdG9ycyBtYXRjaFxuICAgIHZhciBkaWZmZXJlbnQgPSBmdW5jdGlvbihhcnJheTEsIGFycmF5Mikge1xuICAgICAgdmFyIG1heDEgPSAwO1xuICAgICAgdmFyIGkxID0gLTE7XG4gICAgICB2YXIgbWF4MiA9IDA7XG4gICAgICB2YXIgaTIgPSAtMTtcbiAgICAgIGZvciAodmFyIGkgaW4gYXJyYXkxKSB7XG4gICAgICAgIGlmIChhcnJheTFbaV0gPiBtYXgxKSB7XG4gICAgICAgICAgbWF4MSA9IGFycmF5MVtpXTtcbiAgICAgICAgICBpMSA9IGk7XG4gICAgICAgIH1cbiAgICAgICAgaWYgKGFycmF5MltpXSA+IG1heDIpIHtcbiAgICAgICAgICBtYXgyID0gYXJyYXkyW2ldO1xuICAgICAgICAgIGkyID0gaTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICByZXR1cm4gaTEgIT0gaTI7XG4gICAgfVxuXG4gICAgdmFyIGl0ZXJhdGlvbiA9IDA7XG4gICAgdmFyIGVycm9yID0gMTtcbiAgICB2YXIgdGFibGUgPSB7XG4gICAgICBcIkJcIjogMCxcbiAgICAgIFwiUFwiOiAxLFxuICAgICAgXCJUXCI6IDIsXG4gICAgICBcIlhcIjogMyxcbiAgICAgIFwiU1wiOiA0LFxuICAgICAgXCJFXCI6IDVcbiAgICB9XG5cbiAgICB2YXIgc3RhcnQgPSBEYXRlLm5vdygpO1xuICAgIHdoaWxlIChpdGVyYXRpb24gPCBpdGVyYXRpb25zICYmIGVycm9yID4gY3JpdGVyaW9uKSB7XG4gICAgICB2YXIgaSA9IDA7XG4gICAgICBlcnJvciA9IDA7XG5cbiAgICAgIC8vIEVSRyBzZXF1ZW5jZSB0byBsZWFyblxuICAgICAgdmFyIHNlcXVlbmNlID0gZ2VuZXJhdGUoKTtcblxuICAgICAgLy8gaW5wdXRcbiAgICAgIHZhciByZWFkID0gc2VxdWVuY2UuY2hhckF0KGkpO1xuICAgICAgLy8gdGFyZ2V0XG4gICAgICB2YXIgcHJlZGljdCA9IHNlcXVlbmNlLmNoYXJBdChpICsgMSk7XG5cbiAgICAgIC8vIHRyYWluXG4gICAgICB3aGlsZSAoaSA8IHNlcXVlbmNlLmxlbmd0aCAtIDEpIHtcbiAgICAgICAgdmFyIGlucHV0ID0gW107XG4gICAgICAgIHZhciB0YXJnZXQgPSBbXTtcbiAgICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCA2OyBqKyspIHtcbiAgICAgICAgICBpbnB1dFtqXSA9IDA7XG4gICAgICAgICAgdGFyZ2V0W2pdID0gMDtcbiAgICAgICAgfVxuICAgICAgICBpbnB1dFt0YWJsZVtyZWFkXV0gPSAxO1xuICAgICAgICB0YXJnZXRbdGFibGVbcHJlZGljdF1dID0gMTtcblxuICAgICAgICB2YXIgb3V0cHV0ID0gdGhpcy5uZXR3b3JrLmFjdGl2YXRlKGlucHV0KTtcblxuICAgICAgICBpZiAoZGlmZmVyZW50KG91dHB1dCwgdGFyZ2V0KSlcbiAgICAgICAgICB0aGlzLm5ldHdvcmsucHJvcGFnYXRlKHJhdGUsIHRhcmdldCk7XG5cbiAgICAgICAgcmVhZCA9IHNlcXVlbmNlLmNoYXJBdCgrK2kpO1xuICAgICAgICBwcmVkaWN0ID0gc2VxdWVuY2UuY2hhckF0KGkgKyAxKTtcblxuICAgICAgICB2YXIgZGVsdGEgPSAwO1xuICAgICAgICBmb3IgKHZhciBrIGluIG91dHB1dClcbiAgICAgICAgICBkZWx0YSArPSBNYXRoLnBvdyh0YXJnZXRba10gLSBvdXRwdXRba10sIDIpXG4gICAgICAgIGRlbHRhIC89IG91dHB1dC5sZW5ndGg7XG5cbiAgICAgICAgZXJyb3IgKz0gZGVsdGE7XG4gICAgICB9XG4gICAgICBlcnJvciAvPSBzZXF1ZW5jZS5sZW5ndGg7XG4gICAgICBpdGVyYXRpb24rKztcbiAgICAgIGlmIChpdGVyYXRpb24gJSBsb2cgPT0gMCkge1xuICAgICAgICBjb25zb2xlLmxvZyhcIml0ZXJhdGlvbnM6XCIsIGl0ZXJhdGlvbiwgXCIgdGltZTpcIiwgRGF0ZS5ub3coKSAtIHN0YXJ0LFxuICAgICAgICAgIFwiIGVycm9yOlwiLCBlcnJvcik7XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgIGl0ZXJhdGlvbnM6IGl0ZXJhdGlvbixcbiAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydCxcbiAgICAgIHRlc3Q6IHRlc3QsXG4gICAgICBnZW5lcmF0ZTogZ2VuZXJhdGVcbiAgICB9XG4gIH1cblxufVxuXG5leHBvcnQgbW9kdWxlIFRyYWluZXIge1xuICAvLyBCdWlsdC1pbiBjb3N0IGZ1bmN0aW9uc1xuICBcbiAgZXhwb3J0IGludGVyZmFjZSBJVHJhaW5lckNvc3RGbiB7XG4gICAgKHRhcmdldCwgb3V0cHV0KTogbnVtYmVyO1xuICB9XG5cbiAgZXhwb3J0IHZhciBjb3N0ID0ge1xuICAgIC8vIEVxLiA5XG4gICAgQ1JPU1NfRU5UUk9QWTogZnVuY3Rpb24odGFyZ2V0LCBvdXRwdXQpIHtcbiAgICAgIHZhciBjcm9zc2VudHJvcHkgPSAwO1xuICAgICAgZm9yICh2YXIgaSBpbiBvdXRwdXQpXG4gICAgICAgIGNyb3NzZW50cm9weSAtPSAodGFyZ2V0W2ldICogTWF0aC5sb2cob3V0cHV0W2ldICsgMWUtMTUpKSArICgoMSAtIHRhcmdldFtpXSkgKiBNYXRoLmxvZygoMSArIDFlLTE1KSAtIG91dHB1dFtpXSkpOyAvLyArMWUtMTUgaXMgYSB0aW55IHB1c2ggYXdheSB0byBhdm9pZCBNYXRoLmxvZygwKVxuICAgICAgcmV0dXJuIGNyb3NzZW50cm9weTtcbiAgICB9LFxuICAgIE1TRTogZnVuY3Rpb24odGFyZ2V0LCBvdXRwdXQpIHtcbiAgICAgIHZhciBtc2UgPSAwO1xuICAgICAgZm9yICh2YXIgaSBpbiBvdXRwdXQpXG4gICAgICAgIG1zZSArPSBNYXRoLnBvdyh0YXJnZXRbaV0gLSBvdXRwdXRbaV0sIDIpO1xuICAgICAgcmV0dXJuIG1zZSAvIG91dHB1dC5sZW5ndGg7XG4gICAgfVxuICB9XG59Il19
var synaptic = synaptic || Synaptic;var Neuron = synaptic.Neuron, Layer = synaptic.Layer, Network = synaptic.Network, Trainer = synaptic.Trainer, Architect = synaptic.Architect;