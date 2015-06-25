(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
var hopfield = require('./architect/Hopfield');
var lstm = require('./architect/LSTM');
var lsm = require('./architect/Liquid');
var perceptron = require('./architect/Perceptron');
var mb = require('./architect/NTM');
exports.LSTM = lstm.LSTM;
exports.Liquid = lsm.Liquid;
exports.Hopfield = hopfield.Hopfield;
exports.Perceptron = perceptron.Perceptron;
exports.NTM = mb.NTM;

},{"./architect/Hopfield":2,"./architect/LSTM":3,"./architect/Liquid":4,"./architect/NTM":5,"./architect/Perceptron":6}],2:[function(require,module,exports){
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
var Squash = require('../squash');
var Utils = (function () {
    function Utils() {
    }
    Utils.softMax = function (array) {
        // for all i ∈ array
        // sum = ∑ array[n]^e
        // i = î^e / sum
        // where the result ∑ array[0..n] = 1
        if (!array.length)
            return array;
        var sum = 0;
        // sum = ∑ array[n]^e
        for (var i = 0; i < array.length; i++) {
            array[i] = Math.exp(array[i]);
            sum += array[i];
        }
        for (var i = 0; i < array.length; i++)
            array[i] /= sum;
        return array;
    };
    Utils.softMaxSharpen = function (array, sharpen) {
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
    Utils.getCosineSimilarity = function (arrayA, arrayB) {
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
    Utils.vectorShifting = function (wg, shiftings) {
        // w~ 3.3.2 (8)
        var ret = new Float64Array(wg.length);
        var corrimientoIndex = -((shiftings.length - 1) / 2) | 0;
        var circulantMatrix = Utils.buildCirculantMatrix(wg.length);
        for (var i = 0; i < wg.length; i++) {
            for (var x = 0; x < wg.length; x++) {
                var tmp = 0;
                for (var shift = 0; shift < shiftings.length; shift++) {
                    var matRow = i - x + corrimientoIndex + shift;
                    if (matRow < 0)
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
    return Utils;
})();
exports.Utils = Utils;
var NTM = (function (_super) {
    __extends(NTM, _super);
    function NTM(inputs, outputs, memBlocks, blockWidth, heads, hiddenSize) {
        // build the memory
        _super.call(this);
        this.heads = new Array();
        this.dirty = false;
        this.trainer = new trainer.Trainer(this);
        this.blocks = memBlocks;
        this.blockWidth = blockWidth;
        this.data = new Array(this.blocks);
        for (var index = 0; index < this.data.length; index++) {
            this.data[index] = new Float64Array(blockWidth);
        }
        this.clean();
        // build the network
        var inputLength = inputs + heads * memBlocks;
        this.inputValues = new Float64Array(inputLength);
        this.layers.input = this.inputLayer = new Layer.Layer(inputLength);
        this.hiddenLayer = new Layer.Layer(hiddenSize);
        this.layers.output = this.outputLayer = new Layer.Layer(outputs);
        this.inputLayer.project(this.hiddenLayer, Layer.Layer.connectionType.ALL_TO_ALL);
        this.hiddenLayer.project(this.outputLayer, Layer.Layer.connectionType.ALL_TO_ALL);
        var inputCounter = inputs - 1;
        for (var headIndex = 0; headIndex < heads; headIndex++) {
            this.addHead(this.inputValues.subarray(inputCounter, inputCounter + memBlocks));
            inputCounter += memBlocks;
        }
        this.optimized = false;
    }
    NTM.prototype.clean = function () {
        for (var location = 0; location < this.blocks; location++) {
            Utils.initRandomSoftmaxArray(this.data[location]);
        }
        this.dirty = false;
    };
    NTM.prototype.activate = function (input) {
        this.inputValues.set(input);
        this.inputLayer.activate(this.inputValues);
        this.hiddenLayer.activate();
        this.doTimeStep();
        return this.outputLayer.activate();
    };
    NTM.prototype.propagate = function (rate, target) {
        this.outputLayer.propagate(rate, target);
        for (var i = this.heads.length - 1; i >= 0; i--) {
            this.heads[i].layer.propagate(rate);
        }
        this.hiddenLayer.propagate(rate);
        this.dirty = true;
    };
    NTM.prototype.addHead = function (subArray) {
        var head = new Head(this, subArray);
        this.heads.push(head);
        return head;
    };
    NTM.prototype.doTimeStep = function () {
        var _this = this;
        this.heads.forEach(function (head, headIndex) {
            head.doTimeStep();
        });
        // parallelizable
        this.heads.forEach(function (head, headIndex) {
            _this.doErase(head.w_weightings, head.eraseGate);
        });
        // parallelizable
        this.heads.forEach(function (head, headIndex) {
            _this.doAdd(head.w_weightings, head.addGate);
        });
        //this.data.forEach((e) => e = Utils.softMax(e))
    };
    NTM.prototype.doAdd = function (w, addGate) {
        for (var n = 0; n < this.blocks; n++) {
            var M = this.data[n];
            for (var i = 0; i < this.blockWidth; i++) {
                M[i] += addGate[n] * w[i];
            }
        }
    };
    NTM.prototype.doErase = function (w, eraseGate) {
        for (var n = 0; n < this.blocks; n++) {
            var M = this.data[n];
            for (var i = 0; i < this.blockWidth; i++) {
                M[i] *= 1 - eraseGate[n] * w[i];
            }
        }
    };
    return NTM;
})(network.Network);
exports.NTM = NTM;
var Head = (function () {
    function Head(memory, destinationArray) {
        this.s_shiftingValue = null;
        this.prevFocus = 1;
        this.memory = memory;
        this.wc_focusedWeights = new Float64Array(this.memory.blocks);
        this.w_weightings = new Float64Array(this.memory.blocks);
        Utils.initRandomSoftmaxArray(this.w_weightings);
        this.shiftLength = 3; //this.memory.blocks;
        this.s_shiftingVector = new Float64Array(this.shiftLength);
        this.k_keys = new Float64Array(this.memory.blockWidth);
        this.ß_keyStrength = 0;
        this.eraseGate = new Float64Array(this.memory.blocks);
        this.addGate = new Float64Array(this.memory.blocks);
        this.readVector = destinationArray || new Float64Array(this.memory.blocks);
        this.layer = new Layer.Layer(this.memory.blockWidth + this.memory.blocks * 3 + Head.ADDITIONAL_INPUT_VALUES + this.shiftLength);
        this.memory.hiddenLayer.project(this.layer, Layer.Layer.connectionType.ALL_TO_ALL);
        this.layer.project(this.memory.outputLayer, Layer.Layer.connectionType.ALL_TO_ALL);
        this.circulantMatrix = Utils.buildCirculantMatrix(this.memory.blocks);
    }
    Head.prototype.readParams = function (activation) {
        this.ß_keyStrength = Squash.SOFTPLUS(activation[0]);
        this.g_interpolation = Squash.LOGISTIC(activation[1]);
        this.Y_focus = Math.log(Math.exp(activation[2] + 1)) + 1; //Squash.SOFTPLUS(activation[2]) + 1;
        var startAt = 3;
        for (var k = 0; k < this.k_keys.length; k++) {
            this.k_keys[k] = this.layer.list[k + startAt].activation;
        }
        startAt += this.k_keys.length;
        for (var k = 0; k < this.addGate.length; k++) {
            this.addGate[k] = this.layer.list[k + startAt].derivative;
        }
        startAt += this.addGate.length;
        for (var k = 0; k < this.eraseGate.length; k++) {
            this.eraseGate[k] = Squash.LOGISTIC(this.layer.list[k + startAt].activation);
        }
        startAt += this.eraseGate.length;
        for (var k = 0; k < this.shiftLength; k++) {
            this.s_shiftingVector[k] = this.layer.list[k + startAt].activation;
        }
        var M = this.memory.data;
        // focus by content, obtains an array of similarity indexes for each memoryBlock
        for (var i = 0; i < M.length; i++)
            this.wc_focusedWeights[i] = Utils.getCosineSimilarity(M[i], this.k_keys);
        // focus by location (interpolation)
        Utils.interpolateArray(this.wc_focusedWeights, this.w_weightings, this.g_interpolation);
        // convolutional shift
        this.doShiftings();
        // sharpening
        Utils.sharpArray(this.w_weightings, this.wc_focusedWeights, this.Y_focus);
        // since ∑ w = 1, we have to softmax the array
        Utils.softMax(this.w_weightings);
        /// we got wt!
    };
    Head.prototype.doShiftings = function () {
        Utils.softMax(this.s_shiftingVector);
        Utils.vectorShifting(this.wc_focusedWeights, this.s_shiftingVector);
    };
    Head.prototype.doTimeStep = function () {
        var activation = this.layer.activate();
        this.readParams(activation);
        // reading
        for (var index = 0; index < this.memory.blocks; index++) {
            this.readVector[index] = 0;
            for (var cell = 0; cell < this.memory.blockWidth; cell++) {
                this.readVector[index] += this.memory.data[index][cell] * this.w_weightings[index];
            }
        }
    };
    Head.ADDITIONAL_INPUT_VALUES = 3;
    return Head;
})();
exports.Head = Head;

},{"../layer":7,"../network":8,"../squash":10,"../trainer":12}],6:[function(require,module,exports){
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
function SOFTPLUS(x, derivate) {
    if (derivate)
        return 1 - 1 / 1 + Math.exp(x);
    return Math.log(1 + Math.exp(x));
}
exports.SOFTPLUS = SOFTPLUS;

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
        MSE: function (target, output) {
            var mse = 0;
            for (var i in output)
                mse += Math.pow(target[i] - output[i], 2);
            return mse / output.length;
        }
    };
})(Trainer = exports.Trainer || (exports.Trainer = {}));

},{}]},{},[11])
//# sourceMappingURL=data:application/json;charset:utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyaWZ5L25vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJzcmMvYXJjaGl0ZWN0LnRzIiwic3JjL2FyY2hpdGVjdC9Ib3BmaWVsZC50cyIsInNyYy9hcmNoaXRlY3QvTFNUTS50cyIsInNyYy9hcmNoaXRlY3QvTGlxdWlkLnRzIiwic3JjL2FyY2hpdGVjdC9OVE0udHMiLCJzcmMvYXJjaGl0ZWN0L1BlcmNlcHRyb24udHMiLCJzcmMvbGF5ZXIudHMiLCJzcmMvbmV0d29yay50cyIsInNyYy9uZXVyb24udHMiLCJzcmMvc3F1YXNoLnRzIiwic3JjL3N5bmFwdGljLnRzIiwic3JjL3RyYWluZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7QUNBQSxJQUFPLFFBQVEsV0FBVyxzQkFBc0IsQ0FBQyxDQUFDO0FBQ2xELElBQU8sSUFBSSxXQUFXLGtCQUFrQixDQUFDLENBQUM7QUFDMUMsSUFBTyxHQUFHLFdBQVcsb0JBQW9CLENBQUMsQ0FBQztBQUMzQyxJQUFPLFVBQVUsV0FBVyx3QkFBd0IsQ0FBQyxDQUFDO0FBQ3RELElBQU8sRUFBRSxXQUFXLGlCQUFpQixDQUFDLENBQUM7QUFFNUIsWUFBSSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUM7QUFDakIsY0FBTSxHQUFHLEdBQUcsQ0FBQyxNQUFNLENBQUM7QUFDcEIsZ0JBQVEsR0FBRyxRQUFRLENBQUMsUUFBUSxDQUFDO0FBQzdCLGtCQUFVLEdBQUcsVUFBVSxDQUFDLFVBQVUsQ0FBQztBQUNuQyxXQUFHLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQzs7Ozs7Ozs7O0FDVnhCLElBQU8sT0FBTyxXQUFZLFlBQVksQ0FBQyxDQUFDO0FBQ3hDLElBQU8sT0FBTyxXQUFZLFlBQVksQ0FBQyxDQUFDO0FBQ3hDLElBQU8sS0FBSyxXQUFZLFVBQVUsQ0FBQyxDQUFDO0FBR3BDO0lBQThCLDRCQUFlO0lBRzNDLGtCQUFZLElBQVk7UUFDdEIsSUFBSSxV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QztRQUN0RCxJQUFJLENBQUMsT0FBTyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLEdBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQSxxQ0FBcUM7UUFFNUYsSUFBSSxPQUFPLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsTUFBTSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1lBQzVDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxDQUFDLFVBQVUsQ0FBQztRQUMzRCxDQUFDO1FBRUQsT0FBTyxJQUFJLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDO1FBQzlCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsT0FBTyxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1lBQzdDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxDQUFDLFVBQVUsQ0FBQztRQUM1RCxDQUFDO1FBRUQsT0FBTyxJQUFJLElBQUksQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDO1FBQy9CLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsU0FBUyxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1lBQy9DLElBQUksQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLFFBQVEsQ0FBQyxJQUFJLENBQUMsS0FBSyxD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LEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1lBQzVDLE1BQU0sQ0FBQyxHQUFHLEdBQUcsTUFBTSxDQUFDLE1BQU0sQ0FBQztRQUM3QixDQUFDO0tBQ0YsQ0FBQTtBQUNILENBQUMsRUF0QmEsT0FBTyxHQUFQLGVBQU8sS0FBUCxlQUFPLFFBc0JwQiIsImZpbGUiOiJnZW5lcmF0ZWQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlc0NvbnRlbnQiOlsiKGZ1bmN0aW9uIGUodCxuLHIpe2Z1bmN0aW9uIHMobyx1KXtpZighbltvXSl7aWYoIXRbb10pe3ZhciBhPXR5cGVvZiByZXF1aXJlPT1cImZ1bmN0aW9uXCImJnJlcXVpcmU7aWYoIXUmJmEpcmV0dXJuIGEobywhMCk7aWYoaSlyZXR1cm4gaShvLCEwKTt2YXIgZj1uZXcgRXJyb3IoXCJDYW5ub3QgZmluZCBtb2R1bGUgJ1wiK28rXCInXCIpO3Rocm93IGYuY29kZT1cIk1PRFVMRV9OT1RfRk9VTkRcIixmfXZhciBsPW5bb109e2V4cG9ydHM6e319O3Rbb11bMF0uY2FsbChsLmV4cG9ydHMsZnVuY3Rpb24oZSl7dmFyIG49dFtvXVsxXVtlXTtyZXR1cm4gcyhuP246ZSl9LGwsbC5leHBvcnRzLGUsdCxuLHIpfXJldHVybiBuW29dLmV4cG9ydHN9dmFyIGk9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtmb3IodmFyIG89MDtvPHIubGVuZ3RoO28rKylzKHJbb10pO3JldHVybiBzfSkiLCJpbXBvcnQgaG9wZmllbGQgPSByZXF1aXJlKCcuL2FyY2hpdGVjdC9Ib3BmaWVsZCcpO1xuaW1wb3J0IGxzdG0gPSByZXF1aXJlKCcuL2FyY2hpdGVjdC9MU1RNJyk7XG5pbXBvcnQgbHNtID0gcmVxdWlyZSgnLi9hcmNoaXRlY3QvTGlxdWlkJyk7XG5pbXBvcnQgcGVyY2VwdHJvbiA9IHJlcXVpcmUoJy4vYXJjaGl0ZWN0L1BlcmNlcHRyb24nKTtcbmltcG9ydCBtYiA9IHJlcXVpcmUoJy4vYXJjaGl0ZWN0L05UTScpO1xuXG5leHBvcnQgdmFyIExTVE0gPSBsc3RtLkxTVE07XG5leHBvcnQgdmFyIExpcXVpZCA9IGxzbS5MaXF1aWQ7XG5leHBvcnQgdmFyIEhvcGZpZWxkID0gaG9wZmllbGQuSG9wZmllbGQ7XG5leHBvcnQgdmFyIFBlcmNlcHRyb24gPSBwZXJjZXB0cm9uLlBlcmNlcHRyb247XG5leHBvcnQgdmFyIE5UTSA9IG1iLk5UTTsiLCJpbXBvcnQgbmV0d29yayAgPSByZXF1aXJlKCcuLi9uZXR3b3JrJyk7XG5pbXBvcnQgdHJhaW5lciAgPSByZXF1aXJlKCcuLi90cmFpbmVyJyk7XG5pbXBvcnQgbGF5ZXIgID0gcmVxdWlyZSgnLi4vbGF5ZXInKTtcbmltcG9ydCBuZXVyb24gPSByZXF1aXJlKCcuLi9uZXVyb24nKTtcblxuZXhwb3J0IGNsYXNzIEhvcGZpZWxkIGV4dGVuZHMgbmV0d29yay5OZXR3b3JrIHtcbiAgdHJhaW5lcjogdHJhaW5lci5UcmFpbmVyO1xuXG4gIGNvbnN0cnVjdG9yKHNpemU6IG51bWJlcikge1xuICAgIHZhciBpbnB1dExheWVyID0gbmV3IGxheWVyLkxheWVyKHNpemUpO1xuICAgIHZhciBvdXRwdXRMYXllciA9IG5ldyBsYXllci5MYXllcihzaXplKTtcblxuICAgIGlucHV0TGF5ZXIucHJvamVjdChvdXRwdXRMYXllciwgbGF5ZXIuTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTCk7XG5cbiAgICBzdXBlcih7XG4gICAgICBpbnB1dDogaW5wdXRMYXllcixcbiAgICAgIGhpZGRlbjogW10sXG4gICAgICBvdXRwdXQ6IG91dHB1dExheWVyXG4gICAgfSk7XG5cbiAgICB0aGlzLnRyYWluZXIgPSBuZXcgdHJhaW5lci5UcmFpbmVyKHRoaXMpO1xuICB9XG5cbiAgbGVhcm4ocGF0dGVybnMpIHtcbiAgICB2YXIgc2V0ID0gW107XG4gICAgZm9yICh2YXIgcCBpbiBwYXR0ZXJucylcbiAgICAgIHNldC5wdXNoKHtcbiAgICAgICAgaW5wdXQ6IHBhdHRlcm5zW3BdLFxuICAgICAgICBvdXRwdXQ6IHBhdHRlcm5zW3BdXG4gICAgICB9KTtcblxuICAgIHJldHVybiB0aGlzLnRyYWluZXIudHJhaW4oc2V0LCB7XG4gICAgICBpdGVyYXRpb25zOiA1MDAwMDAsXG4gICAgICBlcnJvcjogLjAwMDA1LFxuICAgICAgcmF0ZTogMVxuICAgIH0pO1xuICB9XG5cbiAgZmVlZChwYXR0ZXJuKSB7XG4gICAgdmFyIG91dHB1dCA9IHRoaXMuYWN0aXZhdGUocGF0dGVybik7XG5cbiAgICB2YXIgcGF0dGVybnMgPSBbXTtcbiAgICBmb3IgKHZhciBpIGluIG91dHB1dClcbiAgICAgIHBhdHRlcm5zW2ldID0gb3V0cHV0W2ldID4gLjUgPyAxIDogMDtcblxuICAgIHJldHVybiBwYXR0ZXJucztcbiAgfVxufSIsImltcG9ydCBuZXR3b3JrICA9IHJlcXVpcmUoJy4uL25ldHdvcmsnKTtcbmltcG9ydCB0cmFpbmVyICA9IHJlcXVpcmUoJy4uL3RyYWluZXInKTtcbmltcG9ydCBMYXllciAgPSByZXF1aXJlKCcuLi9sYXllcicpO1xuaW1wb3J0IG5ldXJvbiA9IHJlcXVpcmUoJy4uL25ldXJvbicpO1xuXG5leHBvcnQgY2xhc3MgTFNUTSBleHRlbmRzIG5ldHdvcmsuTmV0d29yayB7XG4gIHRyYWluZXI6IHRyYWluZXIuVHJhaW5lcjtcblxuICBjb25zdHJ1Y3RvciguLi5hcmdzOiBhbnlbXSkge1xuXG4gICAgaWYgKGFyZ3MubGVuZ3RoIDwgMylcbiAgICAgIHRocm93IFwiRXJyb3I6IG5vdCBlbm91Z2ggbGF5ZXJzIChtaW5pbXVtIDMpICEhXCI7XG5cbiAgICB2YXIgbGFzdCA9IGFyZ3MucG9wKCk7XG4gICAgdmFyIG9wdGlvbiA9IHtcbiAgICAgIHBlZXBob2xlczogTGF5ZXIuTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTCxcbiAgICAgIGhpZGRlbnRvaGlkZGVuOiBmYWxzZSxcbiAgICAgIG91dHRvaGlkZGVuOiBmYWxzZSxcbiAgICAgIG91dHRvZ2F0ZXM6IGZhbHNlLFxuICAgICAgaW50b291dDogdHJ1ZSxcbiAgICB9O1xuXG4gICAgaWYgKHR5cGVvZiBsYXN0ICE9ICdudW1iZXInKSB7XG4gICAgICB2YXIgb3V0cHV0cyA9IGFyZ3MucG9wKCk7XG4gICAgICBpZiAobGFzdC5oYXNPd25Qcm9wZXJ0eSgncGVlcGhvbGVzJykpXG4gICAgICAgIG9wdGlvbi5wZWVwaG9sZXMgPSBsYXN0LnBlZXBob2xlcztcbiAgICAgIGlmIChsYXN0Lmhhc093blByb3BlcnR5KCdoaWRkZW50b2hpZGRlbicpKVxuICAgICAgICBvcHRpb24uaGlkZGVudG9oaWRkZW4gPSBsYXN0LmhpZGRlbnRvaGlkZGVuO1xuICAgICAgaWYgKGxhc3QuaGFzT3duUHJvcGVydHkoJ291dHRvaGlkZGVuJykpXG4gICAgICAgIG9wdGlvbi5vdXR0b2hpZGRlbiA9IGxhc3Qub3V0dG9oaWRkZW47XG4gICAgICBpZiAobGFzdC5oYXNPd25Qcm9wZXJ0eSgnb3V0dG9nYXRlcycpKVxuICAgICAgICBvcHRpb24ub3V0dG9nYXRlcyA9IGxhc3Qub3V0dG9nYXRlcztcbiAgICAgIGlmIChsYXN0Lmhhc093blByb3BlcnR5KCdpbnRvb3V0JykpXG4gICAgICAgIG9wdGlvbi5pbnRvb3V0ID0gbGFzdC5pbnRvb3V0O1xuICAgIH0gZWxzZVxuICAgICAgdmFyIG91dHB1dHMgPSBsYXN0O1xuXG4gICAgdmFyIGlucHV0cyA9IGFyZ3Muc2hpZnQoKTtcbiAgICB2YXIgbGF5ZXJzID0gYXJncztcblxuICAgIHZhciBpbnB1dExheWVyID0gbmV3IExheWVyLkxheWVyKGlucHV0cyk7XG4gICAgdmFyIGhpZGRlbkxheWVycyA9IFtdO1xuICAgIHZhciBvdXRwdXRMYXllciA9IG5ldyBMYXllci5MYXllcihvdXRwdXRzKTtcblxuICAgIHZhciBwcmV2aW91cyA9IG51bGw7XG5cbiAgICAvLyBnZW5lcmF0ZSBsYXllcnNcbiAgICBmb3IgKHZhciBsYXllciBpbiBsYXllcnMpIHtcbiAgICAgIC8vIGdlbmVyYXRlIG1lbW9yeSBibG9ja3MgKG1lbW9yeSBjZWxsIGFuZCByZXNwZWN0aXZlIGdhdGVzKVxuICAgICAgdmFyIHNpemUgPSBsYXllcnNbbGF5ZXJdO1xuXG4gICAgICB2YXIgaW5wdXRHYXRlID0gbmV3IExheWVyLkxheWVyKHNpemUpLnNldCh7XG4gICAgICAgIGJpYXM6IDFcbiAgICAgIH0pO1xuICAgICAgdmFyIGZvcmdldEdhdGUgPSBuZXcgTGF5ZXIuTGF5ZXIoc2l6ZSkuc2V0KHtcbiAgICAgICAgYmlhczogMVxuICAgICAgfSk7XG4gICAgICB2YXIgbWVtb3J5Q2VsbCA9IG5ldyBMYXllci5MYXllcihzaXplKTtcbiAgICAgIHZhciBvdXRwdXRHYXRlID0gbmV3IExheWVyLkxheWVyKHNpemUpLnNldCh7XG4gICAgICAgIGJpYXM6IDFcbiAgICAgIH0pO1xuXG4gICAgICBoaWRkZW5MYXllcnMucHVzaChpbnB1dEdhdGUpO1xuICAgICAgaGlkZGVuTGF5ZXJzLnB1c2goZm9yZ2V0R2F0ZSk7XG4gICAgICBoaWRkZW5MYXllcnMucHVzaChtZW1vcnlDZWxsKTtcbiAgICAgIGhpZGRlbkxheWVycy5wdXNoKG91dHB1dEdhdGUpO1xuXG4gICAgICAvLyBjb25uZWN0aW9ucyBmcm9tIGlucHV0IGxheWVyXG4gICAgICB2YXIgaW5wdXQgPSBpbnB1dExheWVyLnByb2plY3QobWVtb3J5Q2VsbCk7XG4gICAgICBpbnB1dExheWVyLnByb2plY3QoaW5wdXRHYXRlKTtcbiAgICAgIGlucHV0TGF5ZXIucHJvamVjdChmb3JnZXRHYXRlKTtcbiAgICAgIGlucHV0TGF5ZXIucHJvamVjdChvdXRwdXRHYXRlKTtcblxuICAgICAgLy8gY29ubmVjdGlvbnMgZnJvbSBwcmV2aW91cyBtZW1vcnktYmxvY2sgbGF5ZXIgdG8gdGhpcyBvbmVcbiAgICAgIGlmIChwcmV2aW91cyAhPSBudWxsKSB7XG4gICAgICAgIHZhciBjZWxsID0gcHJldmlvdXMucHJvamVjdChtZW1vcnlDZWxsKTtcbiAgICAgICAgcHJldmlvdXMucHJvamVjdChpbnB1dEdhdGUpO1xuICAgICAgICBwcmV2aW91cy5wcm9qZWN0KGZvcmdldEdhdGUpO1xuICAgICAgICBwcmV2aW91cy5wcm9qZWN0KG91dHB1dEdhdGUpO1xuICAgICAgfVxuXG4gICAgICAvLyBjb25uZWN0aW9ucyBmcm9tIG1lbW9yeSBjZWxsXG4gICAgICB2YXIgb3V0cHV0ID0gbWVtb3J5Q2VsbC5wcm9qZWN0KG91dHB1dExheWVyKTtcblxuICAgICAgLy8gc2VsZi1jb25uZWN0aW9uXG4gICAgICB2YXIgc2VsZiA9IG1lbW9yeUNlbGwucHJvamVjdChtZW1vcnlDZWxsKTtcblxuICAgICAgLy8gaGlkZGVuIHRvIGhpZGRlbiByZWN1cnJlbnQgY29ubmVjdGlvblxuICAgICAgaWYgKG9wdGlvbi5oaWRkZW50b2hpZGRlbilcbiAgICAgICAgbWVtb3J5Q2VsbC5wcm9qZWN0KG1lbW9yeUNlbGwsIExheWVyLkxheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19FTFNFKTtcblxuICAgICAgLy8gb3V0IHRvIGhpZGRlbiByZWN1cnJlbnQgY29ubmVjdGlvblxuICAgICAgaWYgKG9wdGlvbi5vdXR0b2hpZGRlbilcbiAgICAgICAgb3V0cHV0TGF5ZXIucHJvamVjdChtZW1vcnlDZWxsKTtcblxuICAgICAgLy8gb3V0IHRvIGdhdGVzIHJlY3VycmVudCBjb25uZWN0aW9uXG4gICAgICBpZiAob3B0aW9uLm91dHRvZ2F0ZXMpIHtcbiAgICAgICAgb3V0cHV0TGF5ZXIucHJvamVjdChpbnB1dEdhdGUpO1xuICAgICAgICBvdXRwdXRMYXllci5wcm9qZWN0KG91dHB1dEdhdGUpO1xuICAgICAgICBvdXRwdXRMYXllci5wcm9qZWN0KGZvcmdldEdhdGUpO1xuICAgICAgfVxuICAgICAgXG4gICAgICAvLyBwZWVwaG9sZXNcbiAgICAgIG1lbW9yeUNlbGwucHJvamVjdChpbnB1dEdhdGUsIG9wdGlvbi5wZWVwaG9sZXMpO1xuICAgICAgbWVtb3J5Q2VsbC5wcm9qZWN0KGZvcmdldEdhdGUsIG9wdGlvbi5wZWVwaG9sZXMpO1xuICAgICAgbWVtb3J5Q2VsbC5wcm9qZWN0KG91dHB1dEdhdGUsIG9wdGlvbi5wZWVwaG9sZXMpO1xuXG4gICAgICAvLyBnYXRlc1xuICAgICAgaW5wdXRHYXRlLmdhdGUoaW5wdXQsIExheWVyLkxheWVyLmdhdGVUeXBlLklOUFVUKTtcbiAgICAgIGZvcmdldEdhdGUuZ2F0ZShzZWxmLCBMYXllci5MYXllci5nYXRlVHlwZS5PTkVfVE9fT05FKTtcbiAgICAgIG91dHB1dEdhdGUuZ2F0ZShvdXRwdXQsIExheWVyLkxheWVyLmdhdGVUeXBlLk9VVFBVVCk7XG4gICAgICBpZiAocHJldmlvdXMgIT0gbnVsbClcbiAgICAgICAgaW5wdXRHYXRlLmdhdGUoY2VsbCwgTGF5ZXIuTGF5ZXIuZ2F0ZVR5cGUuSU5QVVQpO1xuXG4gICAgICBwcmV2aW91cyA9IG1lbW9yeUNlbGw7XG4gICAgfVxuXG4gICAgLy8gaW5wdXQgdG8gb3V0cHV0IGRpcmVjdCBjb25uZWN0aW9uXG4gICAgaWYgKG9wdGlvbi5pbnRvb3V0KVxuICAgICAgaW5wdXRMYXllci5wcm9qZWN0KG91dHB1dExheWVyKTtcblxuICAgIC8vIHNldCB0aGUgbGF5ZXJzIG9mIHRoZSBuZXVyYWwgbmV0d29ya1xuICAgIHN1cGVyKHtcbiAgICAgIGlucHV0OiBpbnB1dExheWVyLFxuICAgICAgaGlkZGVuOiBoaWRkZW5MYXllcnMsXG4gICAgICBvdXRwdXQ6IG91dHB1dExheWVyXG4gICAgfSk7XG5cbiAgICAvLyB0cmFpbmVyXG4gICAgdGhpcy50cmFpbmVyID0gbmV3IHRyYWluZXIuVHJhaW5lcih0aGlzKTtcbiAgfVxufTtcbiIsImltcG9ydCBuZXR3b3JrICA9IHJlcXVpcmUoJy4uL25ldHdvcmsnKTtcbmltcG9ydCB0cmFpbmVyICA9IHJlcXVpcmUoJy4uL3RyYWluZXInKTtcbmltcG9ydCBsYXllciAgPSByZXF1aXJlKCcuLi9sYXllcicpO1xuaW1wb3J0IG5ldXJvbiA9IHJlcXVpcmUoJy4uL25ldXJvbicpO1xuXG5leHBvcnQgY2xhc3MgTGlxdWlkIGV4dGVuZHMgbmV0d29yay5OZXR3b3JrIHtcbiAgdHJhaW5lcjogdHJhaW5lci5UcmFpbmVyO1xuXG4gIGNvbnN0cnVjdG9yKGlucHV0cywgaGlkZGVuLCBvdXRwdXRzLCBjb25uZWN0aW9ucywgZ2F0ZXMpIHtcblxuICAgIC8vIGNyZWF0ZSBsYXllcnNcbiAgICB2YXIgaW5wdXRMYXllciA9IG5ldyBsYXllci5MYXllcihpbnB1dHMpO1xuICAgIHZhciBoaWRkZW5MYXllciA9IG5ldyBsYXllci5MYXllcihoaWRkZW4pO1xuICAgIHZhciBvdXRwdXRMYXllciA9IG5ldyBsYXllci5MYXllcihvdXRwdXRzKTtcblxuICAgIC8vIG1ha2UgY29ubmVjdGlvbnMgYW5kIGdhdGVzIHJhbmRvbWx5IGFtb25nIHRoZSBuZXVyb25zXG4gICAgdmFyIG5ldXJvbnMgPSBoaWRkZW5MYXllci5uZXVyb25zKCk7XG4gICAgdmFyIGNvbm5lY3Rpb25MaXN0OiBuZXVyb24uTmV1cm9uLkNvbm5lY3Rpb25bXSA9IFtdO1xuXG4gICAgZm9yICh2YXIgaSA9IDA7IGkgPCBjb25uZWN0aW9uczsgaSsrKSB7XG4gICAgICAvLyBjb25uZWN0IHR3byByYW5kb20gbmV1cm9uc1xuICAgICAgdmFyIGZyb20gPSBNYXRoLnJhbmRvbSgpICogbmV1cm9ucy5sZW5ndGggfCAwO1xuICAgICAgdmFyIHRvID0gTWF0aC5yYW5kb20oKSAqIG5ldXJvbnMubGVuZ3RoIHwgMDtcbiAgICAgIHZhciBjb25uZWN0aW9uID0gbmV1cm9uc1tmcm9tXS5wcm9qZWN0KG5ldXJvbnNbdG9dKTtcbiAgICAgIGNvbm5lY3Rpb25MaXN0LnB1c2goY29ubmVjdGlvbik7XG4gICAgfVxuXG4gICAgZm9yICh2YXIgaiA9IDA7IGogPCBnYXRlczsgaisrKSB7XG4gICAgICAvLyBwaWNrIGEgcmFuZG9tIGdhdGVyIG5ldXJvblxuICAgICAgdmFyIGdhdGVyID0gTWF0aC5yYW5kb20oKSAqIG5ldXJvbnMubGVuZ3RoIHwgMDtcbiAgICAgIC8vIHBpY2sgYSByYW5kb20gY29ubmVjdGlvbiB0byBnYXRlXG4gICAgICB2YXIgY29ubmVjdGlvbk51bWJlciA9IE1hdGgucmFuZG9tKCkgKiBjb25uZWN0aW9uTGlzdC5sZW5ndGggfCAwO1xuICAgICAgLy8gbGV0IHRoZSBnYXRlciBnYXRlIHRoZSBjb25uZWN0aW9uXG4gICAgICBuZXVyb25zW2dhdGVyXS5nYXRlKGNvbm5lY3Rpb25MaXN0W2Nvbm5lY3Rpb25OdW1iZXJdKTtcbiAgICB9XG5cbiAgICAvLyBjb25uZWN0IHRoZSBsYXllcnNcbiAgICBpbnB1dExheWVyLnByb2plY3QoaGlkZGVuTGF5ZXIpO1xuICAgIGhpZGRlbkxheWVyLnByb2plY3Qob3V0cHV0TGF5ZXIpO1xuXG4gICAgLy8gc2V0IHRoZSBsYXllcnMgb2YgdGhlIG5ldHdvcmtcbiAgICBzdXBlcih7XG4gICAgICBpbnB1dDogaW5wdXRMYXllcixcbiAgICAgIGhpZGRlbjogW2hpZGRlbkxheWVyXSxcbiAgICAgIG91dHB1dDogb3V0cHV0TGF5ZXJcbiAgICB9KTtcblxuICAgIC8vIHRyYWluZXJcbiAgICB0aGlzLnRyYWluZXIgPSBuZXcgdHJhaW5lci5UcmFpbmVyKHRoaXMpO1xuICB9XG59XG4iLCJpbXBvcnQgbmV0d29yayA9IHJlcXVpcmUoJy4uL25ldHdvcmsnKTtcbmltcG9ydCB0cmFpbmVyID0gcmVxdWlyZSgnLi4vdHJhaW5lcicpO1xuaW1wb3J0IExheWVyID0gcmVxdWlyZSgnLi4vbGF5ZXInKTtcbmltcG9ydCBuZXVyb24gPSByZXF1aXJlKCcuLi9uZXVyb24nKTtcbmltcG9ydCBTeW5hcHRpYyA9IHJlcXVpcmUoJy4uL3N5bmFwdGljJyk7XG5pbXBvcnQgU3F1YXNoID0gcmVxdWlyZSgnLi4vc3F1YXNoJyk7XG5cbmV4cG9ydCBjbGFzcyBVdGlscyB7XG4gIHN0YXRpYyBzb2Z0TWF4PFQgZXh0ZW5kcyBTeW5hcHRpYy5JTnVtZXJpY0FycmF5PihhcnJheTogVCk6IFQge1xuICAgIC8vIGZvciBhbGwgaSDiiIggYXJyYXlcbiAgICAvLyBzdW0gPSDiiJEgYXJyYXlbbl1eZVxuICAgIC8vIGkgPSDDrl5lIC8gc3VtXG4gICAgLy8gd2hlcmUgdGhlIHJlc3VsdCDiiJEgYXJyYXlbMC4ubl0gPSAxXG5cbiAgICBpZiAoIWFycmF5Lmxlbmd0aCkgcmV0dXJuIGFycmF5O1xuXG4gICAgdmFyIHN1bSA9IDA7XG5cbiAgICAvLyBzdW0gPSDiiJEgYXJyYXlbbl1eZVxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgYXJyYXkubGVuZ3RoOyBpKyspIHtcbiAgICAgIGFycmF5W2ldID0gTWF0aC5leHAoYXJyYXlbaV0pO1xuICAgICAgc3VtICs9IGFycmF5W2ldO1xuICAgIH1cblxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgYXJyYXkubGVuZ3RoOyBpKyspIGFycmF5W2ldIC89IHN1bTtcbiAgICBcbiAgICByZXR1cm4gYXJyYXk7XG4gIH1cbiAgc3RhdGljIHNvZnRNYXhTaGFycGVuPFQgZXh0ZW5kcyBTeW5hcHRpYy5JTnVtZXJpY0FycmF5PihhcnJheTogVCwgc2hhcnBlbiA9IDEpOiBUIHtcbiAgICAvLyBmb3IgYWxsIGkg4oiIIGFycmF5XG4gICAgLy8gc3VtID0g4oiRIGFycmF5W25dXmVcbiAgICAvLyBpID0gw65eZSAvIHN1bVxuICAgIC8vIHdoZXJlIHRoZSByZXN1bHQg4oiRIGFycmF5WzAuLm5dID0gMVxuXG4gICAgaWYgKCFhcnJheS5sZW5ndGgpIHJldHVybiBhcnJheTtcblxuICAgIHNoYXJwZW4gPSBzaGFycGVuIHx8IDE7XG5cbiAgICB2YXIgc3VtID0gMDtcblxuICAgIC8vIHN1bSA9IOKIkSBhcnJheVtuXV5lXG4gICAgZm9yICh2YXIgaSA9IDA7IGkgPCBhcnJheS5sZW5ndGg7IGkrKykge1xuICAgICAgYXJyYXlbaV0gPSBNYXRoLmV4cChzaGFycGVuICogYXJyYXlbaV0pO1xuICAgICAgc3VtICs9IGFycmF5W2ldO1xuICAgIH1cblxuICAgIGlmIChzdW0gIT0gMCkge1xuICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCBhcnJheS5sZW5ndGg7IGkrKykgYXJyYXlbaV0gLz0gc3VtO1xuICAgIH0gZWxzZSB7XG4gICAgICB2YXIgZGl2ID0gMSAvIGFycmF5Lmxlbmd0aDtcbiAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgYXJyYXkubGVuZ3RoOyBpKyspIGFycmF5W2ldID0gZGl2O1xuICAgIH1cblxuICAgIHJldHVybiBhcnJheTtcbiAgfVxuICBcblxuICBzdGF0aWMgZ2V0Q29zaW5lU2ltaWxhcml0eShhcnJheUE6IFN5bmFwdGljLklOdW1lcmljQXJyYXksIGFycmF5QjogU3luYXB0aWMuSU51bWVyaWNBcnJheSk6IG51bWJlciB7XG4gICAgLy8gaHR0cDovL2VuLndpa2lwZWRpYS5vcmcvd2lraS9Db3NpbmVfc2ltaWxhcml0eVxuICAgIC8vIE5UTTogMy4zLjEgKDYpXG4gICAgdmFyIGRvdFByID0gMDtcblxuICAgIHZhciBhY3VtQSA9IDAsIGFjdW1CID0gMDtcblxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgYXJyYXlBLmxlbmd0aDsgaSsrKSB7XG4gICAgICBkb3RQciArPSBhcnJheUFbaV0gKiBhcnJheUJbaV07XG4gICAgICBhY3VtQSArPSBhcnJheUFbaV0gKiBhcnJheUFbaV07XG4gICAgICBhY3VtQiArPSBhcnJheUJbaV0gKiBhcnJheUJbaV07XG4gICAgfVxuXG4gICAgcmV0dXJuIGRvdFByIC8gKE1hdGguc3FydChhY3VtQSkgKiBNYXRoLnNxcnQoYWN1bUIpICsgLjAwMDA1KTtcbiAgfVxuXG4gIHN0YXRpYyBpbnRlcnBvbGF0ZUFycmF5KG91dHB1dF9pbnB1dEE6IFN5bmFwdGljLklOdW1lcmljQXJyYXksIGlucHV0QjogU3luYXB0aWMuSU51bWVyaWNBcnJheSwgZykge1xuICAgIC8vIDMuMy4yIGZvY3VzIGJ5IGxvY2F0aW9uICg3KVxuICAgIHZhciBnSW52ZXJ0ZWQgPSAxIC0gZztcbiAgICBmb3IgKHZhciBpID0gMDsgaSA8IG91dHB1dF9pbnB1dEEubGVuZ3RoOyBpKyspXG4gICAgICBvdXRwdXRfaW5wdXRBW2ldID0gb3V0cHV0X2lucHV0QVtpXSAqIGcgKyBnSW52ZXJ0ZWQgKiBpbnB1dEJbaV07XG4gICAgcmV0dXJuIG91dHB1dF9pbnB1dEE7XG4gIH1cbiAgXG4gIC8vIHdfc2hhcnBXblxuICBzdGF0aWMgc2hhcnBBcnJheShvdXRwdXQ6IFN5bmFwdGljLklOdW1lcmljQXJyYXksIHduOiBTeW5hcHRpYy5JTnVtZXJpY0FycmF5LCBZOiBudW1iZXIpIHtcbiAgICAvLyAzLjMuMiAoOSlcbiAgICB2YXIgc3VtID0gMDtcblxuICAgIC8vIOKIgCBhIOKIiCB3biDihpIgYSA9IGFeWVxuICAgIC8vIHN1bSA9IOKIkSBhXlkgXG5cbiAgICBmb3IgKHZhciBpID0gMDsgaSA8IHduLmxlbmd0aDsgaSsrKSB7XG4gICAgICB3bltpXSA9IE1hdGgucG93KHduW2ldLCBZKTtcbiAgICAgIHN1bSArPSB3bltpXTtcbiAgICB9XG5cbiAgICAvLyDiiIAgYSDiiIggd24g4oaSIGEgPSBhXlkgLyBzdW1cbiAgICBpZiAoc3VtICE9IDApIHtcbiAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgd24ubGVuZ3RoOyBpKyspIG91dHB1dFtpXSA9IHduW2ldIC8gc3VtO1xuICAgIH0gZWxzZSB7XG4gICAgICB2YXIgZGl2ID0gMSAvIHduLmxlbmd0aDtcbiAgICAgIGZvciAodmFyIGkgPSAwOyBpIDwgd24ubGVuZ3RoOyBpKyspIG91dHB1dFtpXSA9IGRpdjtcbiAgICB9XG4gIH1cbiAgXG4gIC8vd25fc2hpZnRcbiAgc3RhdGljIHNjYWxhclNoaWZ0aW5nKHdnOiBTeW5hcHRpYy5JTnVtZXJpY0FycmF5LCBzaGlmdFNjYWxhcjogbnVtYmVyKSB7XG4gICAgLy8gd34gMy4zLjIgKDgpXG4gICAgdmFyIHNoaWZ0aW5ncyA9IG5ldyBGbG9hdDY0QXJyYXkod2cubGVuZ3RoKTtcbiAgICB2YXIgd24gPSBuZXcgRmxvYXQ2NEFycmF5KHdnLmxlbmd0aCk7XG5cbiAgICB2YXIgaW50UGFydCA9IHNoaWZ0U2NhbGFyIHwgMDtcbiAgICB2YXIgZGVjaW1hbFBhcnQgPSBzaGlmdFNjYWxhciAtIGludFBhcnQ7XG5cbiAgICBzaGlmdGluZ3NbaW50UGFydCAlIHNoaWZ0aW5ncy5sZW5ndGhdID0gMSAtIGRlY2ltYWxQYXJ0O1xuICAgIHNoaWZ0aW5nc1soaW50UGFydCArIDEpICUgc2hpZnRpbmdzLmxlbmd0aF0gPSBkZWNpbWFsUGFydDtcblxuXG4gICAgZm9yICh2YXIgaSA9IDA7IGkgPCB3bi5sZW5ndGg7IGkrKykge1xuICAgICAgdmFyIGFjdW0gPSAwO1xuICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCB3bi5sZW5ndGg7IGorKykge1xuICAgICAgICBpZiAoKGkgLSBqKSA8IDApXG4gICAgICAgICAgYWN1bSArPSB3Z1tqXSAqIHNoaWZ0aW5nc1tzaGlmdGluZ3MubGVuZ3RoIC0gTWF0aC5hYnMoaSAtIGopXTtcbiAgICAgICAgZWxzZVxuICAgICAgICAgIGFjdW0gKz0gd2dbal0gKiBzaGlmdGluZ3NbKGkgLSBqKSAlIHNoaWZ0aW5ncy5sZW5ndGhdO1xuICAgICAgfVxuICAgICAgd25baV0gPSBhY3VtO1xuICAgIH1cblxuICAgIHJldHVybiB3bjtcbiAgfVxuICBcbiAgc3RhdGljIG5vcm1hbGl6ZVNoaWZ0KHNoaWZ0OiBGbG9hdDY0QXJyYXkpIHtcbiAgICB2YXIgc3VtID0gMDtcbiAgICBmb3IgKHZhciBpID0gMDsgaSA8IHNoaWZ0Lmxlbmd0aDsgaSsrKSB7XG4gICAgICBzdW0gKz0gc2hpZnRbaV07XG4gICAgfVxuICAgIGZvciAodmFyIGogPSAwOyBqIDwgc2hpZnQubGVuZ3RoOyBqKyspIHtcbiAgICAgIHNoaWZ0W2pdIC89IHN1bTtcbiAgICB9XG4gIH1cblxuICBzdGF0aWMgdmVjdG9yU2hpZnRpbmcod2c6IEZsb2F0NjRBcnJheSwgc2hpZnRpbmdzOiBTeW5hcHRpYy5JTnVtZXJpY0FycmF5KSB7XG4gICAgLy8gd34gMy4zLjIgKDgpXG5cbiAgICBcbiAgICB2YXIgcmV0ID0gbmV3IEZsb2F0NjRBcnJheSh3Zy5sZW5ndGgpO1xuXG4gICAgdmFyIGNvcnJpbWllbnRvSW5kZXggPSAtKChzaGlmdGluZ3MubGVuZ3RoIC0gMSkgLyAyKSB8IDA7XG5cbiAgICB2YXIgY2lyY3VsYW50TWF0cml4ID0gVXRpbHMuYnVpbGRDaXJjdWxhbnRNYXRyaXgod2cubGVuZ3RoKTtcblxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgd2cubGVuZ3RoOyBpKyspIHtcbiAgICAgIGZvciAodmFyIHggPSAwOyB4IDwgd2cubGVuZ3RoOyB4KyspIHtcbiAgICAgICAgdmFyIHRtcCA9IDA7XG4gIFxuICAgICAgICBmb3IgKHZhciBzaGlmdCA9IDA7IHNoaWZ0IDwgc2hpZnRpbmdzLmxlbmd0aDsgc2hpZnQrKykge1xuICBcbiAgICAgICAgICB2YXIgbWF0Um93ID0gaSAtIHggKyBjb3JyaW1pZW50b0luZGV4ICsgc2hpZnQ7XG4gIFxuICAgICAgICAgIGlmIChtYXRSb3cgPCAwKVxuICAgICAgICAgICAgbWF0Um93ICs9IHdnLmxlbmd0aDtcbiAgXG4gICAgICAgICAgbWF0Um93ICU9IHdnLmxlbmd0aDtcbiAgXG4gICAgICAgICAgdG1wICs9IHdnW2NpcmN1bGFudE1hdHJpeFt4XVttYXRSb3ddXSAqIHNoaWZ0aW5nc1tzaGlmdF07XG4gICAgICAgIH1cbiAgXG4gICAgICAgIHJldFtpXSA9IHRtcDtcbiAgICAgIH1cbiAgICAgIFxuICAgIH1cblxuICAgIHdnLnNldChyZXQpO1xuICB9XG5cbiAgc3RhdGljIGluaXRSYW5kb21Tb2Z0bWF4QXJyYXkoYXJyYXk6IEZsb2F0NjRBcnJheSk6IHZvaWQge1xuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgYXJyYXkubGVuZ3RoOyBpKyspIHtcbiAgICAgIGFycmF5W2ldID0gTWF0aC5yYW5kb20oKTtcbiAgICB9XG5cbiAgICBVdGlscy5zb2Z0TWF4KGFycmF5KTtcbiAgfVxuXG4gIHN0YXRpYyBidWlsZENpcmN1bGFudE1hdHJpeChsZW5ndGg6IG51bWJlciwgb2Zmc2V0OiBudW1iZXIgPSAwKTogRmxvYXQ2NEFycmF5W10ge1xuICAgIHZhciByZXQgPSBbXTtcblxuICAgIGZvciAodmFyIGkgPSAwOyBpIDwgbGVuZ3RoOyBpKyspIHtcbiAgICAgIHZhciBhcnIgPSBuZXcgRmxvYXQ2NEFycmF5KGxlbmd0aCk7XG4gICAgICByZXQucHVzaChhcnIpO1xuICAgICAgZm9yICh2YXIgbiA9IDA7IG4gPCBsZW5ndGg7IG4rKykge1xuICAgICAgICBhcnJbbl0gPSAoKGkgKyBuKSAlIGxlbmd0aCk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHJldDtcbiAgfVxufVxuXG5cblxuZXhwb3J0IGNsYXNzIE5UTSBleHRlbmRzIG5ldHdvcmsuTmV0d29yayB7XG4gIHRyYWluZXI6IHRyYWluZXIuVHJhaW5lcjtcblxuICBkYXRhOiBGbG9hdDY0QXJyYXlbXTtcblxuICBibG9ja1dpZHRoOiBudW1iZXI7XG4gIGJsb2NrczogbnVtYmVyO1xuXG4gIGhlYWRzOiBIZWFkW10gPSBuZXcgQXJyYXkoKTtcblxuICBpbnB1dFZhbHVlczogRmxvYXQ2NEFycmF5O1xuXG4gIGlucHV0TGF5ZXI6IExheWVyLkxheWVyO1xuICBoaWRkZW5MYXllcjogTGF5ZXIuTGF5ZXI7XG4gIG91dHB1dExheWVyOiBMYXllci5MYXllcjtcblxuICBkaXJ0eSA9IGZhbHNlO1xuXG4gIGNvbnN0cnVjdG9yKGlucHV0czogbnVtYmVyLCBvdXRwdXRzOiBudW1iZXIsIG1lbUJsb2NrczogbnVtYmVyLCBibG9ja1dpZHRoOiBudW1iZXIsIGhlYWRzOiBudW1iZXIsIGhpZGRlblNpemU6IG51bWJlcikge1xuICAgIC8vIGJ1aWxkIHRoZSBtZW1vcnlcbiAgICBcbiAgICBzdXBlcigpO1xuXG4gICAgdGhpcy50cmFpbmVyID0gbmV3IHRyYWluZXIuVHJhaW5lcih0aGlzKTtcblxuICAgIHRoaXMuYmxvY2tzID0gbWVtQmxvY2tzO1xuICAgIHRoaXMuYmxvY2tXaWR0aCA9IGJsb2NrV2lkdGg7XG5cbiAgICB0aGlzLmRhdGEgPSBuZXcgQXJyYXkodGhpcy5ibG9ja3MpO1xuICAgIGZvciAodmFyIGluZGV4ID0gMDsgaW5kZXggPCB0aGlzLmRhdGEubGVuZ3RoOyBpbmRleCsrKSB7XG4gICAgICB0aGlzLmRhdGFbaW5kZXhdID0gbmV3IEZsb2F0NjRBcnJheShibG9ja1dpZHRoKTtcblxuICAgIH1cblxuICAgIHRoaXMuY2xlYW4oKTtcbiAgICBcbiAgICAvLyBidWlsZCB0aGUgbmV0d29ya1xuICAgIFxuICAgIHZhciBpbnB1dExlbmd0aCA9IGlucHV0cyArIGhlYWRzICogbWVtQmxvY2tzO1xuXG4gICAgdGhpcy5pbnB1dFZhbHVlcyA9IG5ldyBGbG9hdDY0QXJyYXkoaW5wdXRMZW5ndGgpO1xuXG4gICAgdGhpcy5sYXllcnMuaW5wdXQgPSB0aGlzLmlucHV0TGF5ZXIgPSBuZXcgTGF5ZXIuTGF5ZXIoaW5wdXRMZW5ndGgpO1xuICAgIHRoaXMuaGlkZGVuTGF5ZXIgPSBuZXcgTGF5ZXIuTGF5ZXIoaGlkZGVuU2l6ZSk7XG4gICAgdGhpcy5sYXllcnMub3V0cHV0ID0gdGhpcy5vdXRwdXRMYXllciA9IG5ldyBMYXllci5MYXllcihvdXRwdXRzKTtcblxuXG5cbiAgICB0aGlzLmlucHV0TGF5ZXIucHJvamVjdCh0aGlzLmhpZGRlbkxheWVyLCBMYXllci5MYXllci5jb25uZWN0aW9uVHlwZS5BTExfVE9fQUxMKTtcbiAgICB0aGlzLmhpZGRlbkxheWVyLnByb2plY3QodGhpcy5vdXRwdXRMYXllciwgTGF5ZXIuTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTCk7XG5cbiAgICB2YXIgaW5wdXRDb3VudGVyID0gaW5wdXRzIC0gMTtcblxuICAgIGZvciAodmFyIGhlYWRJbmRleCA9IDA7IGhlYWRJbmRleCA8IGhlYWRzOyBoZWFkSW5kZXgrKykge1xuICAgICAgdGhpcy5hZGRIZWFkKHRoaXMuaW5wdXRWYWx1ZXMuc3ViYXJyYXkoaW5wdXRDb3VudGVyLCBpbnB1dENvdW50ZXIgKyBtZW1CbG9ja3MpKTtcbiAgICAgIGlucHV0Q291bnRlciArPSBtZW1CbG9ja3M7XG4gICAgfVxuXG4gICAgdGhpcy5vcHRpbWl6ZWQgPSBmYWxzZTtcbiAgfVxuXG4gIGNsZWFuKCkge1xuICAgIGZvciAodmFyIGxvY2F0aW9uID0gMDsgbG9jYXRpb24gPCB0aGlzLmJsb2NrczsgbG9jYXRpb24rKykge1xuICAgICAgVXRpbHMuaW5pdFJhbmRvbVNvZnRtYXhBcnJheSh0aGlzLmRhdGFbbG9jYXRpb25dKTtcbiAgICB9XG4gICAgdGhpcy5kaXJ0eSA9IGZhbHNlO1xuICB9XG5cbiAgYWN0aXZhdGUoaW5wdXQ6IFN5bmFwdGljLklOdW1lcmljQXJyYXkpIHtcbiAgICB0aGlzLmlucHV0VmFsdWVzLnNldCg8YW55PmlucHV0KTtcblxuICAgIHRoaXMuaW5wdXRMYXllci5hY3RpdmF0ZSh0aGlzLmlucHV0VmFsdWVzKTtcbiAgICB0aGlzLmhpZGRlbkxheWVyLmFjdGl2YXRlKCk7XG5cbiAgICB0aGlzLmRvVGltZVN0ZXAoKTtcblxuICAgIHJldHVybiB0aGlzLm91dHB1dExheWVyLmFjdGl2YXRlKCk7XG4gIH1cblxuICBwcm9wYWdhdGUocmF0ZTogbnVtYmVyLCB0YXJnZXQ6IFN5bmFwdGljLklOdW1lcmljQXJyYXkpIHtcbiAgICB0aGlzLm91dHB1dExheWVyLnByb3BhZ2F0ZShyYXRlLCB0YXJnZXQpO1xuICAgIGZvciAodmFyIGkgPSB0aGlzLmhlYWRzLmxlbmd0aCAtIDE7IGkgPj0gMDsgaS0tKSB7XG4gICAgICB0aGlzLmhlYWRzW2ldLmxheWVyLnByb3BhZ2F0ZShyYXRlKTtcbiAgICB9XG4gICAgdGhpcy5oaWRkZW5MYXllci5wcm9wYWdhdGUocmF0ZSk7XG4gICAgdGhpcy5kaXJ0eSA9IHRydWU7XG4gIH1cblxuICBhZGRIZWFkKHN1YkFycmF5OiBGbG9hdDY0QXJyYXkpOiBIZWFkIHtcbiAgICB2YXIgaGVhZCA9IG5ldyBIZWFkKHRoaXMsIHN1YkFycmF5KTtcbiAgICB0aGlzLmhlYWRzLnB1c2goaGVhZCk7XG4gICAgcmV0dXJuIGhlYWQ7XG4gIH1cblxuICBkb1RpbWVTdGVwKCkge1xuICAgIHRoaXMuaGVhZHMuZm9yRWFjaCgoaGVhZCwgaGVhZEluZGV4KSA9PiB7XG4gICAgICBoZWFkLmRvVGltZVN0ZXAoKTtcbiAgICB9KTtcbiAgICBcbiAgICAvLyBwYXJhbGxlbGl6YWJsZVxuICAgIHRoaXMuaGVhZHMuZm9yRWFjaCgoaGVhZCwgaGVhZEluZGV4KSA9PiB7XG4gICAgICB0aGlzLmRvRXJhc2UoaGVhZC53X3dlaWdodGluZ3MsIGhlYWQuZXJhc2VHYXRlKTtcbiAgICB9KTtcbiAgICBcbiAgICAvLyBwYXJhbGxlbGl6YWJsZVxuICAgIHRoaXMuaGVhZHMuZm9yRWFjaCgoaGVhZCwgaGVhZEluZGV4KSA9PiB7XG4gICAgICB0aGlzLmRvQWRkKGhlYWQud193ZWlnaHRpbmdzLCBoZWFkLmFkZEdhdGUpO1xuICAgIH0pO1xuICAgIFxuICAgIC8vdGhpcy5kYXRhLmZvckVhY2goKGUpID0+IGUgPSBVdGlscy5zb2Z0TWF4KGUpKVxuICB9XG5cbiAgZG9BZGQodzogU3luYXB0aWMuSU51bWVyaWNBcnJheSwgYWRkR2F0ZTogU3luYXB0aWMuSU51bWVyaWNBcnJheSkge1xuICAgIGZvciAodmFyIG4gPSAwOyBuIDwgdGhpcy5ibG9ja3M7IG4rKykge1xuICAgICAgdmFyIE0gPSB0aGlzLmRhdGFbbl07XG4gICAgICBmb3IgKHZhciBpID0gMDsgaSA8IHRoaXMuYmxvY2tXaWR0aDsgaSsrKSB7XG4gICAgICAgIE1baV0gKz0gYWRkR2F0ZVtuXSAqIHdbaV07XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgZG9FcmFzZSh3OiBTeW5hcHRpYy5JTnVtZXJpY0FycmF5LCBlcmFzZUdhdGU6IFN5bmFwdGljLklOdW1lcmljQXJyYXkpIHtcbiAgICBmb3IgKHZhciBuID0gMDsgbiA8IHRoaXMuYmxvY2tzOyBuKyspIHtcbiAgICAgIHZhciBNID0gdGhpcy5kYXRhW25dO1xuICAgICAgZm9yICh2YXIgaSA9IDA7IGkgPCB0aGlzLmJsb2NrV2lkdGg7IGkrKykge1xuICAgICAgICBNW2ldICo9IDEgLSBlcmFzZUdhdGVbbl0gKiB3W2ldO1xuICAgICAgfVxuICAgIH1cbiAgfVxufVxuXG5cblxuZXhwb3J0IGNsYXNzIEhlYWQge1xuICBzdGF0aWMgQURESVRJT05BTF9JTlBVVF9WQUxVRVMgPSAzO1xuXG4gIG1lbW9yeTogTlRNO1xuXG4gIHdfd2VpZ2h0aW5nczogRmxvYXQ2NEFycmF5O1xuICBlcmFzZUdhdGU6IEZsb2F0NjRBcnJheTtcbiAgYWRkR2F0ZTogRmxvYXQ2NEFycmF5O1xuICBrX2tleXM6IEZsb2F0NjRBcnJheTtcbiAgZ19pbnRlcnBvbGF0aW9uOiBudW1iZXI7XG4gIFlfZm9jdXM6IG51bWJlcjtcbiAgc19zaGlmdGluZ1ZhbHVlOiBudW1iZXIgPSBudWxsO1xuICBzX3NoaWZ0aW5nVmVjdG9yOiBGbG9hdDY0QXJyYXk7XG4gIHdjX2ZvY3VzZWRXZWlnaHRzOiBGbG9hdDY0QXJyYXk7XG4gIHJlYWRWZWN0b3I6IEZsb2F0NjRBcnJheTtcbiAgw59fa2V5U3RyZW5ndGg6IG51bWJlcjtcbiAgcHJldkZvY3VzOiBudW1iZXIgPSAxO1xuXG4gIHNoaWZ0TGVuZ3RoOiBudW1iZXI7XG5cbiAgbGF5ZXI6IExheWVyLkxheWVyO1xuXG4gIGNpcmN1bGFudE1hdHJpeDogRmxvYXQ2NEFycmF5W107XG5cbiAgY29uc3RydWN0b3IobWVtb3J5OiBOVE0sIGRlc3RpbmF0aW9uQXJyYXk/OiBGbG9hdDY0QXJyYXkpIHtcbiAgICB0aGlzLm1lbW9yeSA9IG1lbW9yeTtcbiAgICB0aGlzLndjX2ZvY3VzZWRXZWlnaHRzID0gbmV3IEZsb2F0NjRBcnJheSh0aGlzLm1lbW9yeS5ibG9ja3MpO1xuICAgIHRoaXMud193ZWlnaHRpbmdzID0gbmV3IEZsb2F0NjRBcnJheSh0aGlzLm1lbW9yeS5ibG9ja3MpO1xuXG4gICAgVXRpbHMuaW5pdFJhbmRvbVNvZnRtYXhBcnJheSh0aGlzLndfd2VpZ2h0aW5ncyk7XG5cbiAgICB0aGlzLnNoaWZ0TGVuZ3RoID0gMzsgLy90aGlzLm1lbW9yeS5ibG9ja3M7XG5cbiAgICB0aGlzLnNfc2hpZnRpbmdWZWN0b3IgPSBuZXcgRmxvYXQ2NEFycmF5KHRoaXMuc2hpZnRMZW5ndGgpO1xuICAgIHRoaXMua19rZXlzID0gbmV3IEZsb2F0NjRBcnJheSh0aGlzLm1lbW9yeS5ibG9ja1dpZHRoKTtcbiAgICB0aGlzLsOfX2tleVN0cmVuZ3RoID0gMDtcbiAgICB0aGlzLmVyYXNlR2F0ZSA9IG5ldyBGbG9hdDY0QXJyYXkodGhpcy5tZW1vcnkuYmxvY2tzKTtcbiAgICB0aGlzLmFkZEdhdGUgPSBuZXcgRmxvYXQ2NEFycmF5KHRoaXMubWVtb3J5LmJsb2Nrcyk7XG4gICAgdGhpcy5yZWFkVmVjdG9yID0gZGVzdGluYXRpb25BcnJheSB8fCBuZXcgRmxvYXQ2NEFycmF5KHRoaXMubWVtb3J5LmJsb2Nrcyk7XG5cbiAgICB0aGlzLmxheWVyID0gbmV3IExheWVyLkxheWVyKHRoaXMubWVtb3J5LmJsb2NrV2lkdGggKyB0aGlzLm1lbW9yeS5ibG9ja3MgKiAzICsgSGVhZC5BRERJVElPTkFMX0lOUFVUX1ZBTFVFUyArIHRoaXMuc2hpZnRMZW5ndGgpO1xuXG4gICAgdGhpcy5tZW1vcnkuaGlkZGVuTGF5ZXIucHJvamVjdCh0aGlzLmxheWVyLCBMYXllci5MYXllci5jb25uZWN0aW9uVHlwZS5BTExfVE9fQUxMKTtcbiAgICB0aGlzLmxheWVyLnByb2plY3QodGhpcy5tZW1vcnkub3V0cHV0TGF5ZXIsIExheWVyLkxheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19BTEwpO1xuXG4gICAgdGhpcy5jaXJjdWxhbnRNYXRyaXggPSBVdGlscy5idWlsZENpcmN1bGFudE1hdHJpeCh0aGlzLm1lbW9yeS5ibG9ja3MpO1xuICB9XG5cbiAgcHJpdmF0ZSByZWFkUGFyYW1zKGFjdGl2YXRpb246IEZsb2F0NjRBcnJheSkge1xuXG4gICAgdGhpcy7Dn19rZXlTdHJlbmd0aCA9IFNxdWFzaC5TT0ZUUExVUyhhY3RpdmF0aW9uWzBdKTtcbiAgICB0aGlzLmdfaW50ZXJwb2xhdGlvbiA9IFNxdWFzaC5MT0dJU1RJQyhhY3RpdmF0aW9uWzFdKTtcbiAgICB0aGlzLllfZm9jdXMgPSBNYXRoLmxvZyhNYXRoLmV4cChhY3RpdmF0aW9uWzJdKzEpKSArIDE7Ly9TcXVhc2guU09GVFBMVVMoYWN0aXZhdGlvblsyXSkgKyAxO1xuXG4gICAgdmFyIHN0YXJ0QXQgPSAzO1xuICAgIGZvciAodmFyIGsgPSAwOyBrIDwgdGhpcy5rX2tleXMubGVuZ3RoOyBrKyspIHtcbiAgICAgIHRoaXMua19rZXlzW2tdID0gdGhpcy5sYXllci5saXN0W2sgKyBzdGFydEF0XS5hY3RpdmF0aW9uO1xuICAgIH1cblxuICAgIHN0YXJ0QXQgKz0gdGhpcy5rX2tleXMubGVuZ3RoO1xuICAgIGZvciAodmFyIGsgPSAwOyBrIDwgdGhpcy5hZGRHYXRlLmxlbmd0aDsgaysrKSB7XG4gICAgICB0aGlzLmFkZEdhdGVba10gPSB0aGlzLmxheWVyLmxpc3RbayArIHN0YXJ0QXRdLmRlcml2YXRpdmU7XG4gICAgfVxuXG4gICAgc3RhcnRBdCArPSB0aGlzLmFkZEdhdGUubGVuZ3RoO1xuICAgIGZvciAodmFyIGsgPSAwOyBrIDwgdGhpcy5lcmFzZUdhdGUubGVuZ3RoOyBrKyspIHtcbiAgICAgIHRoaXMuZXJhc2VHYXRlW2tdID0gU3F1YXNoLkxPR0lTVElDKHRoaXMubGF5ZXIubGlzdFtrICsgc3RhcnRBdF0uYWN0aXZhdGlvbik7XG4gICAgfVxuXG4gICAgc3RhcnRBdCArPSB0aGlzLmVyYXNlR2F0ZS5sZW5ndGg7XG4gICAgZm9yICh2YXIgayA9IDA7IGsgPCB0aGlzLnNoaWZ0TGVuZ3RoOyBrKyspIHtcbiAgICAgIHRoaXMuc19zaGlmdGluZ1ZlY3RvcltrXSA9IHRoaXMubGF5ZXIubGlzdFtrICsgc3RhcnRBdF0uYWN0aXZhdGlvbjtcbiAgICB9XG5cbiAgICB2YXIgTSA9IHRoaXMubWVtb3J5LmRhdGE7XG4gICAgXG4gICAgLy8gZm9jdXMgYnkgY29udGVudCwgb2J0YWlucyBhbiBhcnJheSBvZiBzaW1pbGFyaXR5IGluZGV4ZXMgZm9yIGVhY2ggbWVtb3J5QmxvY2tcbiAgICBmb3IgKHZhciBpID0gMDsgaSA8IE0ubGVuZ3RoOyBpKyspXG4gICAgICB0aGlzLndjX2ZvY3VzZWRXZWlnaHRzW2ldID0gVXRpbHMuZ2V0Q29zaW5lU2ltaWxhcml0eShNW2ldLCB0aGlzLmtfa2V5cyk7XG4gICAgXG4gICAgLy8gZm9jdXMgYnkgbG9jYXRpb24gKGludGVycG9sYXRpb24pXG4gICAgVXRpbHMuaW50ZXJwb2xhdGVBcnJheSh0aGlzLndjX2ZvY3VzZWRXZWlnaHRzLCB0aGlzLndfd2VpZ2h0aW5ncywgdGhpcy5nX2ludGVycG9sYXRpb24pO1xuICAgIFxuICAgIC8vIGNvbnZvbHV0aW9uYWwgc2hpZnRcbiAgICB0aGlzLmRvU2hpZnRpbmdzKCk7XG4gICAgIFxuICAgIC8vIHNoYXJwZW5pbmdcbiAgICBVdGlscy5zaGFycEFycmF5KHRoaXMud193ZWlnaHRpbmdzLCB0aGlzLndjX2ZvY3VzZWRXZWlnaHRzLCB0aGlzLllfZm9jdXMpO1xuICAgIFxuICAgIC8vIHNpbmNlIOKIkSB3ID0gMSwgd2UgaGF2ZSB0byBzb2Z0bWF4IHRoZSBhcnJheVxuICAgIFV0aWxzLnNvZnRNYXgodGhpcy53X3dlaWdodGluZ3MpO1xuICAgIFxuICAgIC8vLyB3ZSBnb3Qgd3QhXG4gIH1cblxuICBkb1NoaWZ0aW5ncygpIHtcbiAgICBVdGlscy5zb2Z0TWF4KHRoaXMuc19zaGlmdGluZ1ZlY3Rvcik7XG4gICAgXG4gICAgVXRpbHMudmVjdG9yU2hpZnRpbmcodGhpcy53Y19mb2N1c2VkV2VpZ2h0cywgdGhpcy5zX3NoaWZ0aW5nVmVjdG9yKTtcbiAgfVxuXG4gIGRvVGltZVN0ZXAoKSB7XG4gICAgdmFyIGFjdGl2YXRpb24gPSB0aGlzLmxheWVyLmFjdGl2YXRlKCk7XG5cbiAgICB0aGlzLnJlYWRQYXJhbXMoYWN0aXZhdGlvbik7XG4gICAgXG4gICAgLy8gcmVhZGluZ1xuICAgIGZvciAodmFyIGluZGV4ID0gMDsgaW5kZXggPCB0aGlzLm1lbW9yeS5ibG9ja3M7IGluZGV4KyspIHtcbiAgICAgIHRoaXMucmVhZFZlY3RvcltpbmRleF0gPSAwO1xuICAgICAgZm9yICh2YXIgY2VsbCA9IDA7IGNlbGwgPCB0aGlzLm1lbW9yeS5ibG9ja1dpZHRoOyBjZWxsKyspIHtcbiAgICAgICAgdGhpcy5yZWFkVmVjdG9yW2luZGV4XSArPSB0aGlzLm1lbW9yeS5kYXRhW2luZGV4XVtjZWxsXSAqIHRoaXMud193ZWlnaHRpbmdzW2luZGV4XTtcbiAgICAgIH1cbiAgICB9XG4gIH1cbn0iLCJpbXBvcnQgbmV0d29yayAgPSByZXF1aXJlKCcuLi9uZXR3b3JrJyk7XG5pbXBvcnQgdHJhaW5lciAgPSByZXF1aXJlKCcuLi90cmFpbmVyJyk7XG5pbXBvcnQgbGF5ZXIgID0gcmVxdWlyZSgnLi4vbGF5ZXInKTtcbmltcG9ydCBuZXVyb24gPSByZXF1aXJlKCcuLi9uZXVyb24nKTtcbi8vIE11bHRpbGF5ZXIgUGVyY2VwdHJvblxuZXhwb3J0IGNsYXNzIFBlcmNlcHRyb24gZXh0ZW5kcyBuZXR3b3JrLk5ldHdvcmsge1xuICB0cmFpbmVyOiB0cmFpbmVyLlRyYWluZXI7XG5cbiAgY29uc3RydWN0b3IoLi4uYXJnczogbnVtYmVyW10pIHtcblxuICAgIGlmIChhcmdzLmxlbmd0aCA8IDMpXG4gICAgICB0aHJvdyBcIkVycm9yOiBub3QgZW5vdWdoIGxheWVycyAobWluaW11bSAzKSAhIVwiO1xuXG4gICAgdmFyIGlucHV0cyA9IGFyZ3Muc2hpZnQoKTsgLy8gZmlyc3QgYXJndW1lbnRcbiAgICB2YXIgb3V0cHV0cyA9IGFyZ3MucG9wKCk7IC8vIGxhc3QgYXJndW1lbnRcbiAgICB2YXIgbGF5ZXJzID0gYXJnczsgLy8gYWxsIHRoZSBhcmd1bWVudHMgaW4gdGhlIG1pZGRsZVxuICBcbiAgICB2YXIgaW5wdXQgPSBuZXcgbGF5ZXIuTGF5ZXIoaW5wdXRzKTtcbiAgICB2YXIgaGlkZGVuID0gW107XG4gICAgdmFyIG91dHB1dCA9IG5ldyBsYXllci5MYXllcihvdXRwdXRzKTtcblxuICAgIHZhciBwcmV2aW91cyA9IGlucHV0O1xuICBcbiAgICAvLyBnZW5lcmF0ZSBoaWRkZW4gbGF5ZXJzXG4gICAgZm9yICh2YXIgbGV2ZWwgaW4gbGF5ZXJzKSB7XG4gICAgICB2YXIgc2l6ZSA9IGxheWVyc1tsZXZlbF07XG4gICAgICB2YXIgdGhlTGF5ZXIgPSBuZXcgbGF5ZXIuTGF5ZXIoc2l6ZSk7XG4gICAgICBoaWRkZW4ucHVzaCh0aGVMYXllcik7XG4gICAgICBwcmV2aW91cy5wcm9qZWN0KHRoZUxheWVyKTtcbiAgICAgIHByZXZpb3VzID0gdGhlTGF5ZXI7XG4gICAgfVxuICAgIHByZXZpb3VzLnByb2plY3Qob3V0cHV0KTtcbiAgXG4gICAgLy8gc2V0IGxheWVycyBvZiB0aGUgbmV1cmFsIG5ldHdvcmtcbiAgICAgIFxuICAgIHN1cGVyKHtcbiAgICAgIGlucHV0OiBpbnB1dCxcbiAgICAgIGhpZGRlbjogaGlkZGVuLFxuICAgICAgb3V0cHV0OiBvdXRwdXRcbiAgICB9KTtcbiAgXG4gICAgLy8gdHJhaW5lciBmb3IgdGhlIG5ldHdvcmtcbiAgICB0aGlzLnRyYWluZXIgPSBuZXcgdHJhaW5lci5UcmFpbmVyKHRoaXMpO1xuICB9XG59OyAiLCJpbXBvcnQgbmV1cm9uID0gcmVxdWlyZSgnLi9uZXVyb24nKTtcbmltcG9ydCBuZXR3b3JrID0gcmVxdWlyZSgnLi9uZXR3b3JrJyk7XG5pbXBvcnQgU3luYXB0aWMgPSByZXF1aXJlKCcuL3N5bmFwdGljJyk7XG5cbi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIExBWUVSXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuZXhwb3J0IGNsYXNzIExheWVyIHtcblx0XHRsaXN0OiBuZXVyb24uTmV1cm9uW10gPSBbXTtcblx0XHRsYWJlbDogc3RyaW5nID0gbnVsbDtcblx0XHRjb25uZWN0ZWR0byA9IFtdO1xuXHRcdHNpemUgPSAwO1xuXHRcdFxuXHRcdGN1cnJlbnRBY3RpdmF0aW9uOiBGbG9hdDY0QXJyYXk7XG5cblx0XHRjb25zdHJ1Y3RvcihzaXplOiBudW1iZXIsIGxhYmVsPzogc3RyaW5nKSB7XG5cdFx0XHR0aGlzLnNpemUgPSBzaXplIHwgMDtcblx0XHRcdHRoaXMubGlzdCA9IFtdO1xuXHRcdFx0dGhpcy5sYWJlbCA9IGxhYmVsIHx8IG51bGw7XG5cdFx0XHR0aGlzLmNvbm5lY3RlZHRvID0gW107XG5cdFx0XHRcblx0XHRcdHRoaXMuY3VycmVudEFjdGl2YXRpb24gPSBuZXcgRmxvYXQ2NEFycmF5KHNpemUpO1xuXG5cdFx0XHR3aGlsZSAoc2l6ZS0tKSB7XG5cdFx0XHRcdHZhciB0aGVOZXVyb24gPSBuZXcgbmV1cm9uLk5ldXJvbigpO1xuXHRcdFx0XHR0aGlzLmxpc3QucHVzaCh0aGVOZXVyb24pO1xuXHRcdFx0fVxuXHRcdH1cblxuXHRcblx0XHQvLyBhY3RpdmF0ZXMgYWxsIHRoZSBuZXVyb25zIGluIHRoZSBsYXllclxuXHRcdGFjdGl2YXRlKGlucHV0PzogU3luYXB0aWMuSU51bWVyaWNBcnJheSkgOiBGbG9hdDY0QXJyYXkge1xuXG5cdFx0XHRpZih0aGlzLmN1cnJlbnRBY3RpdmF0aW9uLmxlbmd0aCAhPSB0aGlzLmxpc3QubGVuZ3RoKVxuXHRcdFx0XHR0aGlzLmN1cnJlbnRBY3RpdmF0aW9uID0gbmV3IEZsb2F0NjRBcnJheSh0aGlzLmxpc3QubGVuZ3RoKTtcblxuXHRcdFx0dmFyIGFjdGl2YXRpb25JbmRleCA9IDA7XG5cblx0XHRcdGlmICh0eXBlb2YgaW5wdXQgIT0gJ3VuZGVmaW5lZCcpIHtcblx0XHRcdFx0aWYgKGlucHV0Lmxlbmd0aCAhPSB0aGlzLnNpemUpXG5cdFx0XHRcdFx0dGhyb3cgXCJJTlBVVCBzaXplIGFuZCBMQVlFUiBzaXplIG11c3QgYmUgdGhlIHNhbWUgdG8gYWN0aXZhdGUhXCI7XG5cblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5saXN0KSB7XG5cdFx0XHRcdFx0dGhpcy5jdXJyZW50QWN0aXZhdGlvblthY3RpdmF0aW9uSW5kZXgrK10gPSB0aGlzLmxpc3RbaWRdLmFjdGl2YXRlKGlucHV0W2lkXSk7XG5cdFx0XHRcdH1cblx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMubGlzdCkge1xuXHRcdFx0XHRcdHRoaXMuY3VycmVudEFjdGl2YXRpb25bYWN0aXZhdGlvbkluZGV4KytdID0gdGhpcy5saXN0W2lkXS5hY3RpdmF0ZSgpO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0XHRcblx0XHRcdHJldHVybiB0aGlzLmN1cnJlbnRBY3RpdmF0aW9uO1xuXHRcdH1cblxuXHRcdC8vIHByb3BhZ2F0ZXMgdGhlIGVycm9yIG9uIGFsbCB0aGUgbmV1cm9ucyBvZiB0aGUgbGF5ZXJcblx0XHRwcm9wYWdhdGUocmF0ZTogbnVtYmVyLCB0YXJnZXQ/IDogU3luYXB0aWMuSU51bWVyaWNBcnJheSkge1xuXHRcdFx0aWYgKHR5cGVvZiB0YXJnZXQgIT0gJ3VuZGVmaW5lZCcpIHtcblx0XHRcdFx0aWYgKHRhcmdldC5sZW5ndGggIT0gdGhpcy5zaXplKVxuXHRcdFx0XHRcdHRocm93IFwiVEFSR0VUIHNpemUgYW5kIExBWUVSIHNpemUgbXVzdCBiZSB0aGUgc2FtZSB0byBwcm9wYWdhdGUhXCI7XG5cblx0XHRcdFx0Zm9yICh2YXIgaWQgPSB0aGlzLmxpc3QubGVuZ3RoIC0gMTsgaWQgPj0gMDsgaWQtLSkge1xuXHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaWRdO1xuXHRcdFx0XHRcdG5ldXJvbi5wcm9wYWdhdGUocmF0ZSwgdGFyZ2V0W2lkXSk7XG5cdFx0XHRcdH1cblx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdGZvciAodmFyIGlkID0gdGhpcy5saXN0Lmxlbmd0aCAtIDE7IGlkID49IDA7IGlkLS0pIHtcblx0XHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5saXN0W2lkXTtcblx0XHRcdFx0XHRuZXVyb24ucHJvcGFnYXRlKHJhdGUpO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFx0Ly8gcHJvamVjdHMgYSBjb25uZWN0aW9uIGZyb20gdGhpcyBsYXllciB0byBhbm90aGVyIG9uZVxuXHRcdHByb2plY3QobGF5ZXIgOiBuZXR3b3JrLk5ldHdvcmsgfCBMYXllciwgdHlwZT8sIHdlaWdodHM/IDogU3luYXB0aWMuSU51bWVyaWNBcnJheSkge1xuXG5cdFx0XHRpZiAobGF5ZXIgaW5zdGFuY2VvZiBuZXR3b3JrLk5ldHdvcmspXG5cdFx0XHRcdGxheWVyID0gKDxuZXR3b3JrLk5ldHdvcms+bGF5ZXIpLmxheWVycy5pbnB1dDtcblxuXHRcdFx0aWYgKGxheWVyIGluc3RhbmNlb2YgTGF5ZXIpIHtcblx0XHRcdFx0aWYgKCF0aGlzLmNvbm5lY3RlZChsYXllcikpXG5cdFx0XHRcdFx0cmV0dXJuIG5ldyBMYXllci5MYXllckNvbm5lY3Rpb24odGhpcywgbGF5ZXIsIHR5cGUsIHdlaWdodHMpO1xuXHRcdFx0fSBlbHNlXG5cdFx0XHRcdHRocm93IFwiSW52YWxpZCBhcmd1bWVudCwgeW91IGNhbiBvbmx5IHByb2plY3QgY29ubmVjdGlvbnMgdG8gTEFZRVJTIGFuZCBORVRXT1JLUyFcIjtcblxuXG5cdFx0fVxuXG5cdFx0Ly8gZ2F0ZXMgYSBjb25uZWN0aW9uIGJldHdlbm4gdHdvIGxheWVyc1xuXHRcdGdhdGUoY29ubmVjdGlvbiwgdHlwZSkge1xuXG5cdFx0XHRpZiAodHlwZSA9PSBMYXllci5nYXRlVHlwZS5JTlBVVCkge1xuXHRcdFx0XHRpZiAoY29ubmVjdGlvbi50by5zaXplICE9IHRoaXMuc2l6ZSlcblx0XHRcdFx0XHR0aHJvdyBcIkdBVEVSIGxheWVyIGFuZCBDT05ORUNUSU9OLlRPIGxheWVyIG11c3QgYmUgdGhlIHNhbWUgc2l6ZSBpbiBvcmRlciB0byBnYXRlIVwiO1xuXG5cdFx0XHRcdGZvciAodmFyIGlkIGluIGNvbm5lY3Rpb24udG8ubGlzdCkge1xuXHRcdFx0XHRcdHZhciBuZXVyb24gPSBjb25uZWN0aW9uLnRvLmxpc3RbaWRdO1xuXHRcdFx0XHRcdHZhciBnYXRlciA9IHRoaXMubGlzdFtpZF07XG5cdFx0XHRcdFx0Zm9yICh2YXIgaW5wdXQgaW4gbmV1cm9uLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHRcdFx0dmFyIGdhdGVkID0gbmV1cm9uLmNvbm5lY3Rpb25zLmlucHV0c1tpbnB1dF07XG5cdFx0XHRcdFx0XHRpZiAoZ2F0ZWQuSUQgaW4gY29ubmVjdGlvbi5jb25uZWN0aW9ucylcblx0XHRcdFx0XHRcdFx0Z2F0ZXIuZ2F0ZShnYXRlZCk7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9XG5cdFx0XHR9IGVsc2UgaWYgKHR5cGUgPT0gTGF5ZXIuZ2F0ZVR5cGUuT1VUUFVUKSB7XG5cdFx0XHRcdGlmIChjb25uZWN0aW9uLmZyb20uc2l6ZSAhPSB0aGlzLnNpemUpXG5cdFx0XHRcdFx0dGhyb3cgXCJHQVRFUiBsYXllciBhbmQgQ09OTkVDVElPTi5GUk9NIGxheWVyIG11c3QgYmUgdGhlIHNhbWUgc2l6ZSBpbiBvcmRlciB0byBnYXRlIVwiO1xuXG5cdFx0XHRcdGZvciAodmFyIGlkIGluIGNvbm5lY3Rpb24uZnJvbS5saXN0KSB7XG5cdFx0XHRcdFx0dmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24uZnJvbS5saXN0W2lkXTtcblx0XHRcdFx0XHR2YXIgZ2F0ZXIgPSB0aGlzLmxpc3RbaWRdO1xuXHRcdFx0XHRcdGZvciAodmFyIHByb2plY3RlZCBpbiBuZXVyb24uY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG5cdFx0XHRcdFx0XHR2YXIgZ2F0ZWQgPSBuZXVyb24uY29ubmVjdGlvbnMucHJvamVjdGVkW3Byb2plY3RlZF07XG5cdFx0XHRcdFx0XHRpZiAoZ2F0ZWQuSUQgaW4gY29ubmVjdGlvbi5jb25uZWN0aW9ucylcblx0XHRcdFx0XHRcdFx0Z2F0ZXIuZ2F0ZShnYXRlZCk7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9XG5cdFx0XHR9IGVsc2UgaWYgKHR5cGUgPT0gTGF5ZXIuZ2F0ZVR5cGUuT05FX1RPX09ORSkge1xuXHRcdFx0XHRpZiAoY29ubmVjdGlvbi5zaXplICE9IHRoaXMuc2l6ZSlcblx0XHRcdFx0XHR0aHJvdyBcIlRoZSBudW1iZXIgb2YgR0FURVIgVU5JVFMgbXVzdCBiZSB0aGUgc2FtZSBhcyB0aGUgbnVtYmVyIG9mIENPTk5FQ1RJT05TIHRvIGdhdGUhXCI7XG5cblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gY29ubmVjdGlvbi5saXN0KSB7XG5cdFx0XHRcdFx0dmFyIGdhdGVyID0gdGhpcy5saXN0W2lkXTtcblx0XHRcdFx0XHR2YXIgZ2F0ZWQgPSBjb25uZWN0aW9uLmxpc3RbaWRdO1xuXHRcdFx0XHRcdGdhdGVyLmdhdGUoZ2F0ZWQpO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0XHRjb25uZWN0aW9uLmdhdGVkZnJvbS5wdXNoKHsgbGF5ZXI6IHRoaXMsIHR5cGU6IHR5cGUgfSk7XG5cdFx0fVxuXG5cdFx0Ly8gdHJ1ZSBvciBmYWxzZSB3aGV0aGVyIHRoZSB3aG9sZSBsYXllciBpcyBzZWxmLWNvbm5lY3RlZCBvciBub3Rcblx0XHRzZWxmY29ubmVjdGVkKCk6IGJvb2xlYW4ge1xuXG5cdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmxpc3QpIHtcblx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubGlzdFtpZF07XG5cdFx0XHRcdGlmICghbmV1cm9uLnNlbGZjb25uZWN0ZWQoKSlcblx0XHRcdFx0XHRyZXR1cm4gZmFsc2U7XG5cdFx0XHR9XG5cdFx0XHRyZXR1cm4gdHJ1ZTtcblx0XHR9XG5cblx0XHQvLyB0cnVlIG9mIGZhbHNlIHdoZXRoZXIgdGhlIGxheWVyIGlzIGNvbm5lY3RlZCB0byBhbm90aGVyIGxheWVyIChwYXJhbWV0ZXIpIG9yIG5vdFxuXHRcdGNvbm5lY3RlZChsYXllcikge1xuXHRcdFx0Ly8gQ2hlY2sgaWYgQUxMIHRvIEFMTCBjb25uZWN0aW9uXG5cdFx0XHR2YXIgY29ubmVjdGlvbnMgPSAwO1xuXHRcdFx0Zm9yICh2YXIgaGVyZSBpbiB0aGlzLmxpc3QpIHtcblx0XHRcdFx0Zm9yICh2YXIgdGhlcmUgaW4gbGF5ZXIubGlzdCkge1xuXHRcdFx0XHRcdHZhciBmcm9tID0gdGhpcy5saXN0W2hlcmVdO1xuXHRcdFx0XHRcdHZhciB0byA9IGxheWVyLmxpc3RbdGhlcmVdO1xuXHRcdFx0XHRcdHZhciBjb25uZWN0ZWQgPSBmcm9tLmNvbm5lY3RlZCh0byk7XG5cdFx0XHRcdFx0aWYgKGNvbm5lY3RlZCAmJiBjb25uZWN0ZWQudHlwZSA9PSAncHJvamVjdGVkJylcblx0XHRcdFx0XHRcdGNvbm5lY3Rpb25zKys7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHRcdGlmIChjb25uZWN0aW9ucyA9PSB0aGlzLnNpemUgKiBsYXllci5zaXplKVxuXHRcdFx0XHRyZXR1cm4gTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTDtcblxuXHRcdFx0Ly8gQ2hlY2sgaWYgT05FIHRvIE9ORSBjb25uZWN0aW9uXG5cdFx0XHRjb25uZWN0aW9ucyA9IDA7XG5cdFx0XHRmb3IgKHZhciBuZXVyb24gaW4gdGhpcy5saXN0KSB7XG5cdFx0XHRcdHZhciBmcm9tID0gdGhpcy5saXN0W25ldXJvbl07XG5cdFx0XHRcdHZhciB0byA9IGxheWVyLmxpc3RbbmV1cm9uXTtcblx0XHRcdFx0dmFyIGNvbm5lY3RlZCA9IGZyb20uY29ubmVjdGVkKHRvKTtcblx0XHRcdFx0aWYgKGNvbm5lY3RlZCAmJiBjb25uZWN0ZWQudHlwZSA9PSAncHJvamVjdGVkJylcblx0XHRcdFx0XHRjb25uZWN0aW9ucysrO1xuXHRcdFx0fVxuXHRcdFx0aWYgKGNvbm5lY3Rpb25zID09IHRoaXMuc2l6ZSlcblx0XHRcdFx0cmV0dXJuIExheWVyLmNvbm5lY3Rpb25UeXBlLk9ORV9UT19PTkU7XG5cdFx0fVxuXG5cdFx0Ly8gY2xlYXJzIGFsbCB0aGUgbmV1b3JucyBpbiB0aGUgbGF5ZXJcblx0XHRjbGVhcigpIHtcblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMubGlzdCkge1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5saXN0W2lkXTtcblx0XHRcdFx0bmV1cm9uLmNsZWFyKCk7XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFx0Ly8gcmVzZXRzIGFsbCB0aGUgbmV1cm9ucyBpbiB0aGUgbGF5ZXJcblx0XHRyZXNldCgpIHtcblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMubGlzdCkge1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5saXN0W2lkXTtcblx0XHRcdFx0bmV1cm9uLnJlc2V0KCk7XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFx0Ly8gcmV0dXJucyBhbGwgdGhlIG5ldXJvbnMgaW4gdGhlIGxheWVyIChhcnJheSlcblx0XHRuZXVyb25zKCkgOiBuZXVyb24uTmV1cm9uW10ge1xuXHRcdFx0cmV0dXJuIHRoaXMubGlzdDtcblx0XHR9XG5cblx0XHQvLyBhZGRzIGEgbmV1cm9uIHRvIHRoZSBsYXllclxuXHRcdGFkZChuZXVyb24pIHtcblx0XHRcdG5ldXJvbiA9IG5ldXJvbiB8fCBuZXcgbmV1cm9uLk5ldXJvbigpO1xuXHRcdFx0dGhpcy5uZXVyb25zW25ldXJvbi5JRF0gPSBuZXVyb247XG5cdFx0XHR0aGlzLmxpc3QucHVzaChuZXVyb24pO1xuXHRcdFx0dGhpcy5zaXplKys7XG5cdFx0fVxuXG5cdFx0c2V0KG9wdGlvbnMpIHtcblx0XHRcdG9wdGlvbnMgPSBvcHRpb25zIHx8IHt9O1xuXG5cdFx0XHRmb3IgKHZhciBpIGluIHRoaXMubGlzdCkge1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5saXN0W2ldO1xuXHRcdFx0XHRpZiAob3B0aW9ucy5sYWJlbClcblx0XHRcdFx0XHRuZXVyb24ubGFiZWwgPSBvcHRpb25zLmxhYmVsICsgJ18nICsgbmV1cm9uLklEO1xuXHRcdFx0XHRpZiAob3B0aW9ucy5zcXVhc2gpXG5cdFx0XHRcdFx0bmV1cm9uLnNxdWFzaCA9IG9wdGlvbnMuc3F1YXNoO1xuXHRcdFx0XHRpZiAob3B0aW9ucy5iaWFzKVxuXHRcdFx0XHRcdG5ldXJvbi5iaWFzID0gb3B0aW9ucy5iaWFzO1xuXHRcdFx0fVxuXHRcdFx0cmV0dXJuIHRoaXM7XG5cdFx0fVxuXHR9XG5cblxuZXhwb3J0IG1vZHVsZSBMYXllciB7XG5cdGV4cG9ydCB2YXIgbGF5ZXJRdHkgPSAwO1xuXHRleHBvcnQgZnVuY3Rpb24gdWlkKCkge1xuXHRcdHJldHVybiBsYXllclF0eSsrO1xuXHR9XG5cdFxuXHQvLyB0eXBlcyBvZiBjb25uZWN0aW9uc1xuXHRleHBvcnQgdmFyIGNvbm5lY3Rpb25UeXBlID0ge1xuXHRcdEFMTF9UT19BTEw6IFwiQUxMIFRPIEFMTFwiLFxuXHRcdE9ORV9UT19PTkU6IFwiT05FIFRPIE9ORVwiLFxuXHRcdEFMTF9UT19FTFNFOiBcIkFMTCBUTyBFTFNFXCJcblx0fTtcblxuXHQvLyB0eXBlcyBvZiBnYXRlc1xuXHRleHBvcnQgdmFyIGdhdGVUeXBlID0ge1xuXHRcdElOUFVUOiBcIklOUFVUXCIsXG5cdFx0T1VUUFVUOiBcIk9VVFBVVFwiLFxuXHRcdE9ORV9UT19PTkU6IFwiT05FIFRPIE9ORVwiXG5cdH07XG5cblx0Ly8gcmVwcmVzZW50cyBhIGNvbm5lY3Rpb24gZnJvbSBvbmUgbGF5ZXIgdG8gYW5vdGhlciwgYW5kIGtlZXBzIHRyYWNrIG9mIGl0cyB3ZWlnaHQgYW5kIGdhaW5cblx0ZXhwb3J0IGNsYXNzIExheWVyQ29ubmVjdGlvbiB7XG5cdFx0SUQgPSB1aWQoKTtcblx0XHRmcm9tOiBMYXllcjtcblx0XHR0bzogTGF5ZXI7XG5cdFx0c2VsZmNvbm5lY3Rpb24gOiBib29sZWFuID0gZmFsc2U7XG5cdFx0dHlwZTogc3RyaW5nO1xuXHRcdGNvbm5lY3Rpb25zOiBTeW5hcHRpYy5EaWN0aW9uYXJ5PG5ldXJvbi5OZXVyb24uQ29ubmVjdGlvbj47XG5cdFx0bGlzdDogbmV1cm9uLk5ldXJvbi5Db25uZWN0aW9uW107XG5cdFx0c2l6ZSA9IDA7XG5cdFx0Z2F0ZWRmcm9tID0gW107XG5cblx0XHRjb25zdHJ1Y3Rvcihmcm9tTGF5ZXIsIHRvTGF5ZXIsIHR5cGUsIHdlaWdodHMpIHtcblx0XHRcdHRoaXMuZnJvbSA9IGZyb21MYXllcjtcblx0XHRcdHRoaXMudG8gPSB0b0xheWVyO1xuXHRcdFx0dGhpcy5zZWxmY29ubmVjdGlvbiA9IHRvTGF5ZXIgPT0gZnJvbUxheWVyO1xuXHRcdFx0dGhpcy50eXBlID0gdHlwZTtcblx0XHRcdHRoaXMuY29ubmVjdGlvbnMgPSB7fTtcblx0XHRcdHRoaXMubGlzdCA9IFtdO1xuXHRcdFx0dGhpcy5zaXplID0gMDtcblx0XHRcdHRoaXMuZ2F0ZWRmcm9tID0gW107XG5cblxuXHRcdFx0aWYgKHR5cGVvZiB0aGlzLnR5cGUgPT0gJ3VuZGVmaW5lZCcpIHtcblx0XHRcdFx0aWYgKGZyb21MYXllciA9PSB0b0xheWVyKVxuXHRcdFx0XHRcdHRoaXMudHlwZSA9IExheWVyLmNvbm5lY3Rpb25UeXBlLk9ORV9UT19PTkU7XG5cdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHR0aGlzLnR5cGUgPSBMYXllci5jb25uZWN0aW9uVHlwZS5BTExfVE9fQUxMO1xuXHRcdFx0fVxuXG5cdFx0XHRpZiAodGhpcy50eXBlID09IExheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19BTEwgfHxcblx0XHRcdFx0dGhpcy50eXBlID09IExheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19FTFNFKSB7XG5cdFx0XHRcdGZvciAodmFyIGhlcmUgaW4gdGhpcy5mcm9tLmxpc3QpIHtcblx0XHRcdFx0XHRmb3IgKHZhciB0aGVyZSBpbiB0aGlzLnRvLmxpc3QpIHtcblx0XHRcdFx0XHRcdHZhciBmcm9tID0gdGhpcy5mcm9tLmxpc3RbaGVyZV07XG5cdFx0XHRcdFx0XHR2YXIgdG8gPSB0aGlzLnRvLmxpc3RbdGhlcmVdO1xuXHRcdFx0XHRcdFx0aWYgKHRoaXMudHlwZSA9PSBMYXllci5jb25uZWN0aW9uVHlwZS5BTExfVE9fRUxTRSAmJiBmcm9tID09IHRvKVxuXHRcdFx0XHRcdFx0XHRjb250aW51ZTtcblx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uID0gZnJvbS5wcm9qZWN0KHRvLCB3ZWlnaHRzKTtcblxuXHRcdFx0XHRcdFx0dGhpcy5jb25uZWN0aW9uc1tjb25uZWN0aW9uLklEXSA9IGNvbm5lY3Rpb247XG5cdFx0XHRcdFx0XHR0aGlzLnNpemUgPSB0aGlzLmxpc3QucHVzaChjb25uZWN0aW9uKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdH0gZWxzZSBpZiAodGhpcy50eXBlID09IExheWVyLmNvbm5lY3Rpb25UeXBlLk9ORV9UT19PTkUpIHtcblxuXHRcdFx0XHRmb3IgKHZhciBuZXVyb24gaW4gdGhpcy5mcm9tLmxpc3QpIHtcblx0XHRcdFx0XHR2YXIgZnJvbSA9IHRoaXMuZnJvbS5saXN0W25ldXJvbl07XG5cdFx0XHRcdFx0dmFyIHRvID0gdGhpcy50by5saXN0W25ldXJvbl07XG5cdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBmcm9tLnByb2plY3QodG8sIHdlaWdodHMpO1xuXG5cdFx0XHRcdFx0dGhpcy5jb25uZWN0aW9uc1tjb25uZWN0aW9uLklEXSA9IGNvbm5lY3Rpb247XG5cdFx0XHRcdFx0dGhpcy5zaXplID0gdGhpcy5saXN0LnB1c2goY29ubmVjdGlvbik7XG5cdFx0XHRcdH1cblx0XHRcdH1cblxuXHRcdFx0ZnJvbUxheWVyLmNvbm5lY3RlZHRvLnB1c2godGhpcyk7XG5cdFx0fVxuXHR9XG59IiwiaW1wb3J0IGxheWVyID0gcmVxdWlyZSgnLi9sYXllcicpO1xuaW1wb3J0IFNxdWFzaCA9IHJlcXVpcmUoJy4vc3F1YXNoJyk7XG5pbXBvcnQgU3luYXB0aWMgPSByZXF1aXJlKCcuL3N5bmFwdGljJyk7XG5pbXBvcnQgX25ldXJvbiA9IHJlcXVpcmUoJy4vbmV1cm9uJyk7XG5cbi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBORVRXT1JLXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuXG5kZWNsYXJlIGZ1bmN0aW9uIGVzY2FwZShhOiBzdHJpbmcpOiBzdHJpbmc7XG5cblxuZXhwb3J0IGNsYXNzIE5ldHdvcmsge1xuXHRvcHRpbWl6ZWQgPSBudWxsO1xuXHRsYXllcnMgPSB7XG5cdFx0aW5wdXQ6IG51bGwsXG5cdFx0aGlkZGVuOiB7fSxcblx0XHRvdXRwdXQ6IG51bGxcblx0fTtcblx0Y29uc3RydWN0b3IobGF5ZXJzPykge1xuXHRcdGlmICh0eXBlb2YgbGF5ZXJzICE9ICd1bmRlZmluZWQnKSB7XG5cdFx0XHR0aGlzLmxheWVycyA9IGxheWVycyB8fCB7XG5cdFx0XHRcdGlucHV0OiBudWxsLFxuXHRcdFx0XHRoaWRkZW46IHt9LFxuXHRcdFx0XHRvdXRwdXQ6IG51bGxcblx0XHRcdH07XG5cdFx0fVxuXHR9XG5cblx0Ly8gZmVlZC1mb3J3YXJkIGFjdGl2YXRpb24gb2YgYWxsIHRoZSBsYXllcnMgdG8gcHJvZHVjZSBhbiBvdXB1dFxuXHRhY3RpdmF0ZShpbnB1dCA6IFN5bmFwdGljLklOdW1lcmljQXJyYXkpIHtcblxuXHRcdGlmICh0aGlzLm9wdGltaXplZCA9PT0gZmFsc2UpIHtcblx0XHRcdHRoaXMubGF5ZXJzLmlucHV0LmFjdGl2YXRlKGlucHV0KTtcblx0XHRcdGZvciAodmFyIGxheWVyIGluIHRoaXMubGF5ZXJzLmhpZGRlbilcblx0XHRcdFx0dGhpcy5sYXllcnMuaGlkZGVuW2xheWVyXS5hY3RpdmF0ZSgpO1xuXHRcdFx0cmV0dXJuIHRoaXMubGF5ZXJzLm91dHB1dC5hY3RpdmF0ZSgpO1xuXHRcdH1cblx0XHRlbHNlIHtcblx0XHRcdGlmICh0aGlzLm9wdGltaXplZCA9PSBudWxsKVxuXHRcdFx0XHR0aGlzLm9wdGltaXplKCk7XG5cdFx0XHRyZXR1cm4gdGhpcy5vcHRpbWl6ZWQuYWN0aXZhdGUoaW5wdXQpO1xuXHRcdH1cblx0fVxuXG5cdC8vIGJhY2stcHJvcGFnYXRlIHRoZSBlcnJvciB0aHJ1IHRoZSBuZXR3b3JrXG5cdHByb3BhZ2F0ZShyYXRlOiBudW1iZXIsIHRhcmdldD86IFN5bmFwdGljLklOdW1lcmljQXJyYXkpIHtcblxuXHRcdGlmICh0aGlzLm9wdGltaXplZCA9PT0gZmFsc2UpIHtcblx0XHRcdHRoaXMubGF5ZXJzLm91dHB1dC5wcm9wYWdhdGUocmF0ZSwgdGFyZ2V0KTtcblx0XHRcdHZhciByZXZlcnNlID0gW107XG5cdFx0XHRmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pXG5cdFx0XHRcdHJldmVyc2UucHVzaCh0aGlzLmxheWVycy5oaWRkZW5bbGF5ZXJdKTtcblx0XHRcdHJldmVyc2UucmV2ZXJzZSgpO1xuXHRcdFx0Zm9yICh2YXIgbGF5ZXIgaW4gcmV2ZXJzZSlcblx0XHRcdFx0cmV2ZXJzZVtsYXllcl0ucHJvcGFnYXRlKHJhdGUpO1xuXHRcdH1cblx0XHRlbHNlIHtcblx0XHRcdGlmICh0aGlzLm9wdGltaXplZCA9PSBudWxsKVxuXHRcdFx0XHR0aGlzLm9wdGltaXplKCk7XG5cdFx0XHR0aGlzLm9wdGltaXplZC5wcm9wYWdhdGUocmF0ZSwgdGFyZ2V0KTtcblx0XHR9XG5cdH1cblxuXHQvLyBwcm9qZWN0IGEgY29ubmVjdGlvbiB0byBhbm90aGVyIHVuaXQgKGVpdGhlciBhIG5ldHdvcmsgb3IgYSBsYXllcilcblx0cHJvamVjdCh1bml0LCB0eXBlLCB3ZWlnaHRzKSB7XG5cblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplZC5yZXNldCgpO1xuXG5cdFx0aWYgKHVuaXQgaW5zdGFuY2VvZiBOZXR3b3JrKVxuXHRcdFx0cmV0dXJuIHRoaXMubGF5ZXJzLm91dHB1dC5wcm9qZWN0KHVuaXQubGF5ZXJzLmlucHV0LCB0eXBlLCB3ZWlnaHRzKTtcblxuXHRcdGlmICh1bml0IGluc3RhbmNlb2YgbGF5ZXIuTGF5ZXIpXG5cdFx0XHRyZXR1cm4gdGhpcy5sYXllcnMub3V0cHV0LnByb2plY3QodW5pdCwgdHlwZSwgd2VpZ2h0cyk7XG5cblx0XHR0aHJvdyBcIkludmFsaWQgYXJndW1lbnQsIHlvdSBjYW4gb25seSBwcm9qZWN0IGNvbm5lY3Rpb25zIHRvIExBWUVSUyBhbmQgTkVUV09SS1MhXCI7XG5cdH1cblxuXHQvLyBsZXQgdGhpcyBuZXR3b3JrIGdhdGUgYSBjb25uZWN0aW9uXG5cdGdhdGUoY29ubmVjdGlvbiwgdHlwZSkge1xuXHRcdGlmICh0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cdFx0dGhpcy5sYXllcnMub3V0cHV0LmdhdGUoY29ubmVjdGlvbiwgdHlwZSk7XG5cdH1cblxuXHQvLyBjbGVhciBhbGwgZWxlZ2liaWxpdHkgdHJhY2VzIGFuZCBleHRlbmRlZCBlbGVnaWJpbGl0eSB0cmFjZXMgKHRoZSBuZXR3b3JrIGZvcmdldHMgaXRzIGNvbnRleHQsIGJ1dCBub3Qgd2hhdCB3YXMgdHJhaW5lZClcblx0Y2xlYXIoKSB7XG5cblx0XHR0aGlzLnJlc3RvcmUoKTtcblxuXHRcdHZhciBpbnB1dExheWVyID0gdGhpcy5sYXllcnMuaW5wdXQsXG5cdFx0XHRvdXRwdXRMYXllciA9IHRoaXMubGF5ZXJzLm91dHB1dDtcblxuXHRcdGlucHV0TGF5ZXIuY2xlYXIoKTtcblx0XHRmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pIHtcblx0XHRcdHZhciBoaWRkZW5MYXllciA9IHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl07XG5cdFx0XHRoaWRkZW5MYXllci5jbGVhcigpO1xuXHRcdH1cblx0XHRvdXRwdXRMYXllci5jbGVhcigpO1xuXG5cdFx0aWYgKHRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcblx0fVxuXG5cdC8vIHJlc2V0IGFsbCB3ZWlnaHRzIGFuZCBjbGVhciBhbGwgdHJhY2VzIChlbmRzIHVwIGxpa2UgYSBuZXcgbmV0d29yaylcblx0cmVzZXQoKSB7XG5cblx0XHR0aGlzLnJlc3RvcmUoKTtcblxuXHRcdHZhciBpbnB1dExheWVyID0gdGhpcy5sYXllcnMuaW5wdXQsXG5cdFx0XHRvdXRwdXRMYXllciA9IHRoaXMubGF5ZXJzLm91dHB1dDtcblxuXHRcdGlucHV0TGF5ZXIucmVzZXQoKTtcblx0XHRmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pIHtcblx0XHRcdHZhciBoaWRkZW5MYXllciA9IHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl07XG5cdFx0XHRoaWRkZW5MYXllci5yZXNldCgpO1xuXHRcdH1cblx0XHRvdXRwdXRMYXllci5yZXNldCgpO1xuXG5cdFx0aWYgKHRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcblx0fVxuXG5cdC8vIGhhcmRjb2RlcyB0aGUgYmVoYXZpb3VyIG9mIHRoZSB3aG9sZSBuZXR3b3JrIGludG8gYSBzaW5nbGUgb3B0aW1pemVkIGZ1bmN0aW9uXG5cdG9wdGltaXplKCkge1xuXG5cdFx0dmFyIHRoYXQgPSB0aGlzO1xuXHRcdHZhciBvcHRpbWl6ZWQ6IFN5bmFwdGljLklDb21waWxlZFBhcmFtZXRlcnMgPSB7fTtcblx0XHR2YXIgbmV1cm9ucyA9IHRoaXMubmV1cm9ucygpO1xuXG5cdFx0Zm9yICh2YXIgaSBpbiBuZXVyb25zKSB7XG5cdFx0XHR2YXIgbmV1cm9uID0gbmV1cm9uc1tpXS5uZXVyb247XG5cdFx0XHR2YXIgbGF5ZXIgPSBuZXVyb25zW2ldLmxheWVyO1xuXHRcdFx0Lypcblx0XHRcdEZJWE1FOiBkb2VzIHRoaXMgd29ya2VkIG9uY2U/XG5cdFx0XHRcblx0XHRcdHdoaWxlIChuZXVyb24ubmV1cm9uKVxuXHRcdFx0XHRuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuXHRcdFx0Ki9cblxuXHRcdFx0b3B0aW1pemVkID0gbmV1cm9uLm9wdGltaXplKG9wdGltaXplZCwgbGF5ZXIpO1xuXHRcdH1cblxuXHRcdGZvciAodmFyIGkgaW4gb3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlcylcblx0XHRcdG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXNbaV0ucmV2ZXJzZSgpO1xuXHRcdG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMucmV2ZXJzZSgpO1xuXG5cdFx0dmFyIGhhcmRjb2RlID0gXCJcIjtcblx0XHRoYXJkY29kZSArPSBcInZhciBGID0gRmxvYXQ2NEFycmF5ID8gbmV3IEZsb2F0NjRBcnJheShcIiArIG9wdGltaXplZC5tZW1vcnkgK1xuXHRcdFwiKSA6IFtdOyBcIjtcblx0XHRmb3IgKHZhciBpIGluIG9wdGltaXplZC52YXJpYWJsZXMpXG5cdFx0XHRoYXJkY29kZSArPSBcIkZbXCIgKyBvcHRpbWl6ZWQudmFyaWFibGVzW2ldLmlkICsgXCJdID0gXCIgKyAob3B0aW1pemVkLnZhcmlhYmxlc1tcblx0XHRcdFx0aV0udmFsdWUgfHwgMCkgKyBcIjsgXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgYWN0aXZhdGUgPSBmdW5jdGlvbihpbnB1dCl7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJpbmZsdWVuY2VzID0gW107XCI7XG5cdFx0Zm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQuaW5wdXRzKVxuXHRcdFx0aGFyZGNvZGUgKz0gXCJGW1wiICsgb3B0aW1pemVkLmlucHV0c1tpXSArIFwiXSA9IGlucHV0W1wiICsgaSArIFwiXTsgXCI7XG5cdFx0Zm9yICh2YXIgY3VycmVudExheWVyIGluIG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlcykge1xuXHRcdFx0aWYgKG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdLmxlbmd0aCA+IDApIHtcblx0XHRcdFx0Zm9yICh2YXIgY3VycmVudE5ldXJvbiBpbiBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXSkge1xuXHRcdFx0XHRcdGhhcmRjb2RlICs9IG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdW2N1cnJlbnROZXVyb25dLmpvaW4oXCIgXCIpO1xuXHRcdFx0XHRcdGhhcmRjb2RlICs9IG9wdGltaXplZC50cmFjZV9zZW50ZW5jZXNbY3VycmVudExheWVyXVtjdXJyZW50TmV1cm9uXS5qb2luKFwiIFwiKTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdH1cblx0XHRoYXJkY29kZSArPSBcIiB2YXIgb3V0cHV0ID0gW107IFwiXG5cdFx0Zm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQub3V0cHV0cylcblx0XHRcdGhhcmRjb2RlICs9IFwib3V0cHV0W1wiICsgaSArIFwiXSA9IEZbXCIgKyBvcHRpbWl6ZWQub3V0cHV0c1tpXSArIFwiXTsgXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJyZXR1cm4gb3V0cHV0OyB9OyBcIlxuXHRcdGhhcmRjb2RlICs9IFwidmFyIHByb3BhZ2F0ZSA9IGZ1bmN0aW9uKHJhdGUsIHRhcmdldCl7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJGW1wiICsgb3B0aW1pemVkLnZhcmlhYmxlcy5yYXRlLmlkICsgXCJdID0gcmF0ZTsgXCI7XG5cdFx0Zm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQudGFyZ2V0cylcblx0XHRcdGhhcmRjb2RlICs9IFwiRltcIiArIG9wdGltaXplZC50YXJnZXRzW2ldICsgXCJdID0gdGFyZ2V0W1wiICsgaSArIFwiXTsgXCI7XG5cdFx0Zm9yICh2YXIgY3VycmVudExheWVyIGluIG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMpXG5cdFx0XHRmb3IgKHZhciBjdXJyZW50TmV1cm9uIGluIG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXSlcblx0XHRcdFx0aGFyZGNvZGUgKz0gb3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdW2N1cnJlbnROZXVyb25dLmpvaW4oXCIgXCIpICsgXCIgXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCIgfTtcXG5cIjtcblx0XHRoYXJkY29kZSArPVxuXHRcdFwidmFyIG93bmVyc2hpcCA9IGZ1bmN0aW9uKG1lbW9yeUJ1ZmZlcil7XFxuRiA9IG1lbW9yeUJ1ZmZlcjtcXG50aGlzLm1lbW9yeSA9IEY7XFxufTtcXG5cIjtcblx0XHRoYXJkY29kZSArPVxuXHRcdFwicmV0dXJuIHtcXG5tZW1vcnk6IEYsXFxuYWN0aXZhdGU6IGFjdGl2YXRlLFxcbnByb3BhZ2F0ZTogcHJvcGFnYXRlLFxcbm93bmVyc2hpcDogb3duZXJzaGlwXFxufTtcIjtcblx0XHRoYXJkY29kZSA9IGhhcmRjb2RlLnNwbGl0KFwiO1wiKS5qb2luKFwiO1xcblwiKTtcblxuXHRcdHZhciBjb25zdHJ1Y3RvciA9IG5ldyBGdW5jdGlvbihoYXJkY29kZSk7XG5cblx0XHR2YXIgbmV0d29yayA9IGNvbnN0cnVjdG9yKCk7XG5cblx0XHRuZXR3b3JrLmRhdGEgPSB7XG5cdFx0XHR2YXJpYWJsZXM6IG9wdGltaXplZC52YXJpYWJsZXMsXG5cdFx0XHRhY3RpdmF0ZTogb3B0aW1pemVkLmFjdGl2YXRpb25fc2VudGVuY2VzLFxuXHRcdFx0cHJvcGFnYXRlOiBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzLFxuXHRcdFx0dHJhY2U6IG9wdGltaXplZC50cmFjZV9zZW50ZW5jZXMsXG5cdFx0XHRpbnB1dHM6IG9wdGltaXplZC5pbnB1dHMsXG5cdFx0XHRvdXRwdXRzOiBvcHRpbWl6ZWQub3V0cHV0cyxcblx0XHRcdGNoZWNrX2FjdGl2YXRpb246IHRoaXMuYWN0aXZhdGUsXG5cdFx0XHRjaGVja19wcm9wYWdhdGlvbjogdGhpcy5wcm9wYWdhdGVcblx0XHR9XG5cblx0XHRuZXR3b3JrLnJlc2V0ID0gZnVuY3Rpb24oKSB7XG5cdFx0XHRpZiAodGhhdC5vcHRpbWl6ZWQpIHtcblx0XHRcdFx0dGhhdC5vcHRpbWl6ZWQgPSBudWxsO1xuXHRcdFx0XHR0aGF0LmFjdGl2YXRlID0gbmV0d29yay5kYXRhLmNoZWNrX2FjdGl2YXRpb247XG5cdFx0XHRcdHRoYXQucHJvcGFnYXRlID0gbmV0d29yay5kYXRhLmNoZWNrX3Byb3BhZ2F0aW9uO1xuXHRcdFx0fVxuXHRcdH1cblxuXHRcdHRoaXMub3B0aW1pemVkID0gbmV0d29yaztcblx0XHR0aGlzLmFjdGl2YXRlID0gbmV0d29yay5hY3RpdmF0ZTtcblx0XHR0aGlzLnByb3BhZ2F0ZSA9IG5ldHdvcmsucHJvcGFnYXRlO1xuXHR9XG5cblx0Ly8gcmVzdG9yZXMgYWxsIHRoZSB2YWx1ZXMgZnJvbSB0aGUgb3B0aW1pemVkIG5ldHdvcmsgdGhlIHRoZWlyIHJlc3BlY3RpdmUgb2JqZWN0cyBpbiBvcmRlciB0byBtYW5pcHVsYXRlIHRoZSBuZXR3b3JrXG5cdHJlc3RvcmUoKSB7XG5cdFx0aWYgKCF0aGlzLm9wdGltaXplZClcblx0XHRcdHJldHVybjtcblxuXHRcdHZhciBvcHRpbWl6ZWQgPSB0aGlzLm9wdGltaXplZDtcblxuXHRcdHZhciBnZXRWYWx1ZSA9IGZ1bmN0aW9uKC4uLmFyZ3M6IGFueVtdKSB7XG5cdFx0XHR2YXIgdW5pdCA9IGFyZ3Muc2hpZnQoKTtcblx0XHRcdHZhciBwcm9wID0gYXJncy5wb3AoKTtcblxuXHRcdFx0dmFyIGlkID0gcHJvcCArICdfJztcblx0XHRcdGZvciAodmFyIHByb3BlcnR5IGluIGFyZ3MpXG5cdFx0XHRcdGlkICs9IGFyZ3NbcHJvcGVydHldICsgJ18nO1xuXHRcdFx0aWQgKz0gdW5pdC5JRDtcblxuXHRcdFx0dmFyIG1lbW9yeSA9IG9wdGltaXplZC5tZW1vcnk7XG5cdFx0XHR2YXIgdmFyaWFibGVzID0gb3B0aW1pemVkLmRhdGEudmFyaWFibGVzO1xuXG5cdFx0XHRpZiAoaWQgaW4gdmFyaWFibGVzKVxuXHRcdFx0XHRyZXR1cm4gbWVtb3J5W3ZhcmlhYmxlc1tpZF0uaWRdO1xuXHRcdFx0cmV0dXJuIDA7XG5cdFx0fVxuXG5cdFx0dmFyIGxpc3QgPSB0aGlzLm5ldXJvbnMoKTtcblxuXHRcdC8vIGxpbmsgaWQncyB0byBwb3NpdGlvbnMgaW4gdGhlIGFycmF5XG5cdFx0dmFyIGlkcyA9IHt9O1xuXHRcdGZvciAodmFyIGkgaW4gbGlzdCkge1xuXHRcdFx0dmFyIG5ldXJvbiA9IGxpc3RbaV0ubmV1cm9uO1xuXHRcdFx0Lypcblx0XHRcdEZJWE1FOiBkb2VzIHRoaXMgd29ya2VkIG9uY2U/XG5cdFx0XHRcblx0XHRcdHdoaWxlIChuZXVyb24ubmV1cm9uKVxuXHRcdFx0XHRuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuXHRcdFx0Ki9cblxuXHRcdFx0bmV1cm9uLnN0YXRlID0gZ2V0VmFsdWUobmV1cm9uLCAnc3RhdGUnKTtcblx0XHRcdG5ldXJvbi5vbGQgPSBnZXRWYWx1ZShuZXVyb24sICdvbGQnKTtcblx0XHRcdG5ldXJvbi5hY3RpdmF0aW9uID0gZ2V0VmFsdWUobmV1cm9uLCAnYWN0aXZhdGlvbicpO1xuXHRcdFx0bmV1cm9uLmJpYXMgPSBnZXRWYWx1ZShuZXVyb24sICdiaWFzJyk7XG5cblx0XHRcdGZvciAodmFyIGlucHV0IGluIG5ldXJvbi50cmFjZS5lbGVnaWJpbGl0eSlcblx0XHRcdFx0bmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0XSA9IGdldFZhbHVlKG5ldXJvbiwgJ3RyYWNlJyxcblx0XHRcdFx0XHQnZWxlZ2liaWxpdHknLCBpbnB1dCk7XG5cblx0XHRcdGZvciAodmFyIGdhdGVkIGluIG5ldXJvbi50cmFjZS5leHRlbmRlZClcblx0XHRcdFx0Zm9yICh2YXIgaW5wdXQgaW4gbmV1cm9uLnRyYWNlLmV4dGVuZGVkW2dhdGVkXSlcblx0XHRcdFx0XHRuZXVyb24udHJhY2UuZXh0ZW5kZWRbZ2F0ZWRdW2lucHV0XSA9IGdldFZhbHVlKG5ldXJvbiwgJ3RyYWNlJyxcblx0XHRcdFx0XHRcdCdleHRlbmRlZCcsIGdhdGVkLCBpbnB1dCk7XG5cdFx0fVxuXG5cdFx0Ly8gZ2V0IGNvbm5lY3Rpb25zXG5cdFx0Zm9yICh2YXIgaSBpbiBsaXN0KSB7XG5cdFx0XHR2YXIgbmV1cm9uID0gbGlzdFtpXS5uZXVyb247XG5cdFx0XHQvKlxuXHRcdFx0RklYTUU6IGRvZXMgdGhpcyB3b3JrZWQgb25jZT9cblx0XHRcdFxuXHRcdFx0d2hpbGUgKG5ldXJvbi5uZXVyb24pXG5cdFx0XHRcdG5ldXJvbiA9IG5ldXJvbi5uZXVyb247XG5cdFx0XHQqL1xuXG5cdFx0XHRmb3IgKHZhciBqIGluIG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpIHtcblx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBuZXVyb24uY29ubmVjdGlvbnMucHJvamVjdGVkW2pdO1xuXHRcdFx0XHRjb25uZWN0aW9uLndlaWdodCA9IGdldFZhbHVlKGNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdFx0Y29ubmVjdGlvbi5nYWluID0gZ2V0VmFsdWUoY29ubmVjdGlvbiwgJ2dhaW4nKTtcblx0XHRcdH1cblx0XHR9XG5cdH1cblxuXHQvLyByZXR1cm5zIGFsbCB0aGUgbmV1cm9ucyBpbiB0aGUgbmV0d29ya1xuXHRuZXVyb25zKCk6IE5ldHdvcmsuSU5ldHdvcmtOZXVyb25bXSB7XG5cdFx0dmFyIG5ldXJvbnM6IE5ldHdvcmsuSU5ldHdvcmtOZXVyb25bXSA9IFtdO1xuXG5cdFx0dmFyIGlucHV0TGF5ZXIgPSB0aGlzLmxheWVycy5pbnB1dC5uZXVyb25zKCksXG5cdFx0XHRvdXRwdXRMYXllciA9IHRoaXMubGF5ZXJzLm91dHB1dC5uZXVyb25zKCk7XG5cblx0XHRmb3IgKHZhciBuZXVyb24gaW4gaW5wdXRMYXllcilcblx0XHRcdG5ldXJvbnMucHVzaCh7XG5cdFx0XHRcdG5ldXJvbjogaW5wdXRMYXllcltuZXVyb25dLFxuXHRcdFx0XHRsYXllcjogJ2lucHV0J1xuXHRcdFx0fSk7XG5cblx0XHRmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pIHtcblx0XHRcdHZhciBoaWRkZW5MYXllciA9IHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl0ubmV1cm9ucygpO1xuXHRcdFx0Zm9yICh2YXIgbmV1cm9uIGluIGhpZGRlbkxheWVyKVxuXHRcdFx0XHRuZXVyb25zLnB1c2goe1xuXHRcdFx0XHRcdG5ldXJvbjogaGlkZGVuTGF5ZXJbbmV1cm9uXSxcblx0XHRcdFx0XHRsYXllcjogbGF5ZXJcblx0XHRcdFx0fSk7XG5cdFx0fVxuXHRcdGZvciAodmFyIG5ldXJvbiBpbiBvdXRwdXRMYXllcilcblx0XHRcdG5ldXJvbnMucHVzaCh7XG5cdFx0XHRcdG5ldXJvbjogb3V0cHV0TGF5ZXJbbmV1cm9uXSxcblx0XHRcdFx0bGF5ZXI6ICdvdXRwdXQnXG5cdFx0XHR9KTtcblxuXHRcdHJldHVybiBuZXVyb25zO1xuXHR9XG5cblx0Ly8gcmV0dXJucyBudW1iZXIgb2YgaW5wdXRzIG9mIHRoZSBuZXR3b3JrXG5cdGlucHV0cygpOiBudW1iZXIge1xuXHRcdHJldHVybiB0aGlzLmxheWVycy5pbnB1dC5zaXplO1xuXHR9XG5cblx0Ly8gcmV0dXJucyBudW1iZXIgb2Ygb3V0cHV0cyBvZiBodGUgbmV0d29ya1xuXHRvdXRwdXRzKCk6IG51bWJlciB7XG5cdFx0cmV0dXJuIHRoaXMubGF5ZXJzLm91dHB1dC5zaXplO1xuXHR9XG5cblx0Ly8gc2V0cyB0aGUgbGF5ZXJzIG9mIHRoZSBuZXR3b3JrXG5cdHNldChsYXllcnMpIHtcblxuXHRcdHRoaXMubGF5ZXJzID0gbGF5ZXJzO1xuXHRcdGlmICh0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cdH1cblxuXHRzZXRPcHRpbWl6ZShib29sKSB7XG5cdFx0dGhpcy5yZXN0b3JlKCk7XG5cdFx0aWYgKHRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcblx0XHR0aGlzLm9wdGltaXplZCA9IGJvb2wgPyBudWxsIDogZmFsc2U7XG5cdH1cblxuXHQvLyByZXR1cm5zIGEganNvbiB0aGF0IHJlcHJlc2VudHMgYWxsIHRoZSBuZXVyb25zIGFuZCBjb25uZWN0aW9ucyBvZiB0aGUgbmV0d29ya1xuXHR0b0pTT04oaWdub3JlVHJhY2VzKSB7XG5cblx0XHR0aGlzLnJlc3RvcmUoKTtcblxuXHRcdHZhciBsaXN0ID0gdGhpcy5uZXVyb25zKCk7XG5cdFx0dmFyIG5ldXJvbnMgPSBbXTtcblx0XHR2YXIgY29ubmVjdGlvbnMgPSBbXTtcblxuXHRcdC8vIGxpbmsgaWQncyB0byBwb3NpdGlvbnMgaW4gdGhlIGFycmF5XG5cdFx0dmFyIGlkcyA9IHt9O1xuXHRcdGZvciAodmFyIGkgaW4gbGlzdCkge1xuXHRcdFx0dmFyIG5ldXJvbiA9IGxpc3RbaV0ubmV1cm9uO1xuXHRcdFx0Lypcblx0XHRcdEZJWE1FOiBkb2VzIHRoaXMgd29ya2VkIG9uY2U/XG5cdFx0XHRcblx0XHRcdHdoaWxlIChuZXVyb24ubmV1cm9uKVxuXHRcdFx0XHRuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuXHRcdFx0Ki9cblxuXHRcdFx0aWRzW25ldXJvbi5JRF0gPSBpO1xuXG5cdFx0XHR2YXIgY29weSA9IHtcblx0XHRcdFx0dHJhY2U6IHtcblx0XHRcdFx0XHRlbGVnaWJpbGl0eToge30sXG5cdFx0XHRcdFx0ZXh0ZW5kZWQ6IHt9XG5cdFx0XHRcdH0sXG5cdFx0XHRcdHN0YXRlOiBuZXVyb24uc3RhdGUsXG5cdFx0XHRcdG9sZDogbmV1cm9uLm9sZCxcblx0XHRcdFx0YWN0aXZhdGlvbjogbmV1cm9uLmFjdGl2YXRpb24sXG5cdFx0XHRcdGJpYXM6IG5ldXJvbi5iaWFzLFxuXHRcdFx0XHRsYXllcjogbGlzdFtpXS5sYXllcixcblx0XHRcdFx0c3F1YXNoOiBudWxsXG5cdFx0XHR9O1xuXG5cdFx0XHRjb3B5LnNxdWFzaCA9IG5ldXJvbi5zcXVhc2ggPT0gU3F1YXNoLkxPR0lTVElDID8gXCJMT0dJU1RJQ1wiIDpcblx0XHRcdFx0bmV1cm9uLnNxdWFzaCA9PSBTcXVhc2guVEFOSCA/IFwiVEFOSFwiIDpcblx0XHRcdFx0XHRuZXVyb24uc3F1YXNoID09IFNxdWFzaC5JREVOVElUWSA/IFwiSURFTlRJVFlcIiA6XG5cdFx0XHRcdFx0XHRuZXVyb24uc3F1YXNoID09IFNxdWFzaC5ITElNID8gXCJITElNXCIgOlxuXHRcdFx0XHRcdFx0XHRudWxsO1xuXG5cdFx0XHRuZXVyb25zLnB1c2goY29weSk7XG5cdFx0fVxuXG5cdFx0aWYgKCFpZ25vcmVUcmFjZXMpXG5cdFx0XHRmb3IgKHZhciBpIGluIG5ldXJvbnMpIHtcblx0XHRcdFx0dmFyIGNvcGllZE5ldXJvbiA9IG5ldXJvbnNbaV07XG5cblx0XHRcdFx0Zm9yICh2YXIgaW5wdXQgaW4gbmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5KVxuXHRcdFx0XHRcdGNvcGllZE5ldXJvbi50cmFjZS5lbGVnaWJpbGl0eVtpbnB1dF0gPSBuZXVyb24udHJhY2UuZWxlZ2liaWxpdHlbaW5wdXRdO1xuXG5cdFx0XHRcdGZvciAodmFyIGdhdGVkIGluIG5ldXJvbi50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHRcdGNvcGllZE5ldXJvbi50cmFjZS5leHRlbmRlZFtnYXRlZF0gPSB7fTtcblx0XHRcdFx0XHRmb3IgKHZhciBpbnB1dCBpbiBuZXVyb24udHJhY2UuZXh0ZW5kZWRbZ2F0ZWRdKVxuXHRcdFx0XHRcdFx0Y29waWVkTmV1cm9uLnRyYWNlLmV4dGVuZGVkW2lkc1tnYXRlZF1dW2lucHV0XSA9IG5ldXJvbi50cmFjZS5leHRlbmRlZFtcblx0XHRcdFx0XHRcdGdhdGVkXVtpbnB1dF07XG5cdFx0XHRcdH1cblx0XHRcdH1cblxuXHRcdC8vIGdldCBjb25uZWN0aW9uc1xuXHRcdGZvciAodmFyIGkgaW4gbGlzdCkge1xuXHRcdFx0dmFyIG5ldXJvbiA9IGxpc3RbaV0ubmV1cm9uO1xuXHRcdFx0XHRcblx0XHRcdC8qXG5cdFx0XHRGSVhNRTogZG9lcyB0aGlzIHdvcmtlZCBvbmNlP1xuXHRcdFx0XG5cdFx0XHR3aGlsZSAobmV1cm9uLm5ldXJvbilcblx0XHRcdFx0bmV1cm9uID0gbmV1cm9uLm5ldXJvbjtcblx0XHRcdCovXG5cblx0XHRcdGZvciAodmFyIGogaW4gbmV1cm9uLmNvbm5lY3Rpb25zLnByb2plY3RlZCkge1xuXHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbal07XG5cdFx0XHRcdGNvbm5lY3Rpb25zLnB1c2goe1xuXHRcdFx0XHRcdGZyb206IGlkc1tjb25uZWN0aW9uLmZyb20uSURdLFxuXHRcdFx0XHRcdHRvOiBpZHNbY29ubmVjdGlvbi50by5JRF0sXG5cdFx0XHRcdFx0d2VpZ2h0OiBjb25uZWN0aW9uLndlaWdodCxcblx0XHRcdFx0XHRnYXRlcjogY29ubmVjdGlvbi5nYXRlciA/IGlkc1tjb25uZWN0aW9uLmdhdGVyLklEXSA6IG51bGwsXG5cdFx0XHRcdH0pO1xuXHRcdFx0fVxuXHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGVkKCkpXG5cdFx0XHRcdGNvbm5lY3Rpb25zLnB1c2goe1xuXHRcdFx0XHRcdGZyb206IGlkc1tuZXVyb24uSURdLFxuXHRcdFx0XHRcdHRvOiBpZHNbbmV1cm9uLklEXSxcblx0XHRcdFx0XHR3ZWlnaHQ6IG5ldXJvbi5zZWxmY29ubmVjdGlvbi53ZWlnaHQsXG5cdFx0XHRcdFx0Z2F0ZXI6IG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA/IGlkc1tuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXJcblx0XHRcdFx0XHRcdC5JRF0gOiBudWxsLFxuXHRcdFx0XHR9KTtcblx0XHR9XG5cblx0XHRyZXR1cm4ge1xuXHRcdFx0bmV1cm9uczogbmV1cm9ucyxcblx0XHRcdGNvbm5lY3Rpb25zOiBjb25uZWN0aW9uc1xuXHRcdH1cblx0fVxuICBcblx0Ly8gZXhwb3J0IHRoZSB0b3BvbG9neSBpbnRvIGRvdCBsYW5ndWFnZSB3aGljaCBjYW4gYmUgdmlzdWFsaXplZCBhcyBncmFwaHMgdXNpbmcgZG90XG5cdC8qIGV4YW1wbGU6IC4uLiBjb25zb2xlLmxvZyhuZXQudG9Eb3RMYW5nKCkpO1xuXHRcdFx0XHQkIG5vZGUgZXhhbXBsZS5qcyA+IGV4YW1wbGUuZG90XG5cdFx0XHRcdCQgZG90IGV4YW1wbGUuZG90IC1UcG5nID4gb3V0LnBuZ1xuXHQqL1xuXHR0b0RvdChlZGdlY29ubmVjdGlvbikge1xuXHRcdGlmICghIHR5cGVvZiBlZGdlY29ubmVjdGlvbilcblx0XHRcdGVkZ2Vjb25uZWN0aW9uID0gZmFsc2U7XG5cdFx0dmFyIGNvZGUgPSBcImRpZ3JhcGggbm4ge1xcbiAgICByYW5rZGlyID0gQlRcXG5cIjtcblx0XHR2YXIgbGF5ZXJzID0gW3RoaXMubGF5ZXJzLmlucHV0XS5jb25jYXQodGhpcy5sYXllcnMuaGlkZGVuLCB0aGlzLmxheWVycy5vdXRwdXQpO1xuXHRcdGZvciAodmFyIGxheWVyIGluIGxheWVycykge1xuXHRcdFx0Zm9yICh2YXIgdG8gaW4gbGF5ZXJzW2xheWVyXS5jb25uZWN0ZWR0bykgeyAvLyBwcm9qZWN0aW9uc1xuXHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IGxheWVyc1tsYXllcl0uY29ubmVjdGVkdG9bdG9dO1xuXHRcdFx0XHR2YXIgbGF5ZXJ0byA9IGNvbm5lY3Rpb24udG87XG5cdFx0XHRcdHZhciBzaXplID0gY29ubmVjdGlvbi5zaXplO1xuXHRcdFx0XHR2YXIgbGF5ZXJJRCA9IGxheWVycy5pbmRleE9mKGxheWVyc1tsYXllcl0pO1xuXHRcdFx0XHR2YXIgbGF5ZXJ0b0lEID0gbGF5ZXJzLmluZGV4T2YobGF5ZXJ0byk7XG5cdFx0XHRcdC8qIGh0dHA6Ly9zdGFja292ZXJmbG93LmNvbS9xdWVzdGlvbnMvMjY4NDU1NDAvY29ubmVjdC1lZGdlcy13aXRoLWdyYXBoLWRvdFxuKiBET1QgZG9lcyBub3Qgc3VwcG9ydCBlZGdlLXRvLWVkZ2UgY29ubmVjdGlvbnNcbiogVGhpcyB3b3JrYXJvdW5kIHByb2R1Y2VzIHNvbWV3aGF0IHdlaXJkIGdyYXBocyAuLi5cblx0XHRcdFx0Ki9cblx0XHRcdFx0aWYgKGVkZ2Vjb25uZWN0aW9uKSB7XG5cdFx0XHRcdFx0aWYgKGNvbm5lY3Rpb24uZ2F0ZWRmcm9tLmxlbmd0aCkge1xuXHRcdFx0XHRcdFx0dmFyIGZha2VOb2RlID0gXCJmYWtlXCIgKyBsYXllcklEICsgXCJfXCIgKyBsYXllcnRvSUQ7XG5cdFx0XHRcdFx0XHRjb2RlICs9IFwiICAgIFwiICsgZmFrZU5vZGUgK1xuXHRcdFx0XHRcdFx0XCIgW2xhYmVsID0gXFxcIlxcXCIsIHNoYXBlID0gcG9pbnQsIHdpZHRoID0gMC4wMSwgaGVpZ2h0ID0gMC4wMV1cXG5cIjtcblx0XHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBsYXllcklEICsgXCIgLT4gXCIgKyBmYWtlTm9kZSArIFwiIFtsYWJlbCA9IFwiICsgc2l6ZSArIFwiLCBhcnJvd2hlYWQgPSBub25lXVxcblwiO1xuXHRcdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGZha2VOb2RlICsgXCIgLT4gXCIgKyBsYXllcnRvSUQgKyBcIlxcblwiO1xuXHRcdFx0XHRcdH0gZWxzZVxuXHRcdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGxheWVySUQgKyBcIiAtPiBcIiArIGxheWVydG9JRCArIFwiIFtsYWJlbCA9IFwiICsgc2l6ZSArIFwiXVxcblwiO1xuXHRcdFx0XHRcdGZvciAodmFyIGZyb20gaW4gY29ubmVjdGlvbi5nYXRlZGZyb20pIHsgLy8gZ2F0aW5nc1xuXHRcdFx0XHRcdFx0dmFyIGxheWVyZnJvbSA9IGNvbm5lY3Rpb24uZ2F0ZWRmcm9tW2Zyb21dLmxheWVyO1xuXHRcdFx0XHRcdFx0dmFyIHR5cGUgPSBjb25uZWN0aW9uLmdhdGVkZnJvbVtmcm9tXS50eXBlO1xuXHRcdFx0XHRcdFx0dmFyIGxheWVyZnJvbUlEID0gbGF5ZXJzLmluZGV4T2YobGF5ZXJmcm9tKTtcblx0XHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBsYXllcmZyb21JRCArIFwiIC0+IFwiICsgZmFrZU5vZGUgKyBcIiBbY29sb3IgPSBibHVlXVxcblwiO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0XHRjb2RlICs9IFwiICAgIFwiICsgbGF5ZXJJRCArIFwiIC0+IFwiICsgbGF5ZXJ0b0lEICsgXCIgW2xhYmVsID0gXCIgKyBzaXplICsgXCJdXFxuXCI7XG5cdFx0XHRcdFx0Zm9yICh2YXIgZnJvbSBpbiBjb25uZWN0aW9uLmdhdGVkZnJvbSkgeyAvLyBnYXRpbmdzXG5cdFx0XHRcdFx0XHR2YXIgbGF5ZXJmcm9tID0gY29ubmVjdGlvbi5nYXRlZGZyb21bZnJvbV0ubGF5ZXI7XG5cdFx0XHRcdFx0XHR2YXIgdHlwZSA9IGNvbm5lY3Rpb24uZ2F0ZWRmcm9tW2Zyb21dLnR5cGU7XG5cdFx0XHRcdFx0XHR2YXIgbGF5ZXJmcm9tSUQgPSBsYXllcnMuaW5kZXhPZihsYXllcmZyb20pO1xuXHRcdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGxheWVyZnJvbUlEICsgXCIgLT4gXCIgKyBsYXllcnRvSUQgKyBcIiBbY29sb3IgPSBibHVlXVxcblwiO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdH1cblx0XHRjb2RlICs9IFwifVxcblwiO1xuXHRcdHJldHVybiB7XG5cdFx0XHRjb2RlOiBjb2RlLFxuXHRcdFx0bGluazogXCJodHRwczovL2NoYXJ0Lmdvb2dsZWFwaXMuY29tL2NoYXJ0P2NobD1cIiArIGVzY2FwZShjb2RlLnJlcGxhY2UoXCIvIC9nXCIsIFwiK1wiKSkgKyBcIiZjaHQ9Z3ZcIlxuXHRcdH1cblx0fVxuXG5cdC8vIHJldHVybnMgYSBmdW5jdGlvbiB0aGF0IHdvcmtzIGFzIHRoZSBhY3RpdmF0aW9uIG9mIHRoZSBuZXR3b3JrIGFuZCBjYW4gYmUgdXNlZCB3aXRob3V0IGRlcGVuZGluZyBvbiB0aGUgbGlicmFyeVxuXHRzdGFuZGFsb25lKCkge1xuXHRcdGlmICghdGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplKCk7XG5cblx0XHR2YXIgZGF0YSA9IHRoaXMub3B0aW1pemVkLmRhdGE7XG5cblx0XHQvLyBidWlsZCBhY3RpdmF0aW9uIGZ1bmN0aW9uXG5cdFx0dmFyIGFjdGl2YXRpb24gPSBcImZ1bmN0aW9uIChpbnB1dCkge1xcblwiO1xuXG5cdFx0Ly8gYnVpbGQgaW5wdXRzXG5cdFx0Zm9yICh2YXIgaSBpbiBkYXRhLmlucHV0cylcblx0XHRcdGFjdGl2YXRpb24gKz0gXCJGW1wiICsgZGF0YS5pbnB1dHNbaV0gKyBcIl0gPSBpbnB1dFtcIiArIGkgKyBcIl07XFxuXCI7XG5cblx0XHQvLyBidWlsZCBuZXR3b3JrIGFjdGl2YXRpb25cblx0XHRmb3IgKHZhciBuZXVyb24gaW4gZGF0YS5hY3RpdmF0ZSkgeyAvLyBzaG91bGRuJ3QgdGhpcyBiZSBsYXllcj9cblx0XHRcdGZvciAodmFyIHNlbnRlbmNlIGluIGRhdGEuYWN0aXZhdGVbbmV1cm9uXSlcblx0XHRcdFx0YWN0aXZhdGlvbiArPSBkYXRhLmFjdGl2YXRlW25ldXJvbl1bc2VudGVuY2VdICsgXCJcXG5cIjtcblx0XHR9XG5cblx0XHQvLyBidWlsZCBvdXRwdXRzXG5cdFx0YWN0aXZhdGlvbiArPSBcInZhciBvdXRwdXQgPSBbXTtcXG5cIjtcblx0XHRmb3IgKHZhciBpIGluIGRhdGEub3V0cHV0cylcblx0XHRcdGFjdGl2YXRpb24gKz0gXCJvdXRwdXRbXCIgKyBpICsgXCJdID0gRltcIiArIGRhdGEub3V0cHV0c1tpXSArIFwiXTtcXG5cIjtcblx0XHRhY3RpdmF0aW9uICs9IFwicmV0dXJuIG91dHB1dDtcXG59XCI7XG5cblx0XHQvLyByZWZlcmVuY2UgYWxsIHRoZSBwb3NpdGlvbnMgaW4gbWVtb3J5XG5cdFx0dmFyIG1lbW9yeSA9IGFjdGl2YXRpb24ubWF0Y2goL0ZcXFsoXFxkKylcXF0vZyk7XG5cdFx0dmFyIGRpbWVuc2lvbiA9IDA7XG5cdFx0dmFyIGlkcyA9IHt9O1xuXHRcdGZvciAodmFyIGFkZHJlc3MgaW4gbWVtb3J5KSB7XG5cdFx0XHR2YXIgdG1wID0gbWVtb3J5W2FkZHJlc3NdLm1hdGNoKC9cXGQrLylbMF07XG5cdFx0XHRpZiAoISh0bXAgaW4gaWRzKSkge1xuXHRcdFx0XHRpZHNbdG1wXSA9IGRpbWVuc2lvbisrO1xuXHRcdFx0fVxuXHRcdH1cblx0XHR2YXIgaGFyZGNvZGUgPSBcIkYgPSB7XFxuXCI7XG5cdFx0Zm9yICh2YXIgaSBpbiBpZHMpXG5cdFx0XHRoYXJkY29kZSArPSBpZHNbaV0gKyBcIjogXCIgKyB0aGlzLm9wdGltaXplZC5tZW1vcnlbaV0gKyBcIixcXG5cIjtcblx0XHRoYXJkY29kZSA9IGhhcmRjb2RlLnN1YnN0cmluZygwLCBoYXJkY29kZS5sZW5ndGggLSAyKSArIFwiXFxufTtcXG5cIjtcblx0XHRoYXJkY29kZSA9IFwidmFyIHJ1biA9IFwiICsgYWN0aXZhdGlvbi5yZXBsYWNlKC9GXFxbKFxcZCspXS9nLCBmdW5jdGlvbihcblx0XHRcdGluZGV4KSB7XG5cdFx0XHRyZXR1cm4gJ0ZbJyArIGlkc1tpbmRleC5tYXRjaCgvXFxkKy8pWzBdXSArICddJ1xuXHRcdH0pLnJlcGxhY2UoXCJ7XFxuXCIsIFwie1xcblwiICsgaGFyZGNvZGUgKyBcIlwiKSArIFwiO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwicmV0dXJuIHJ1blwiO1xuXG5cdFx0Ly8gcmV0dXJuIHN0YW5kYWxvbmUgZnVuY3Rpb25cblx0XHRyZXR1cm4gbmV3IEZ1bmN0aW9uKGhhcmRjb2RlKSgpO1xuXHR9XG5cblx0d29ya2VyKCkge1xuXHRcdGlmICghdGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplKCk7XG5cblx0XHR2YXIgaGFyZGNvZGUgPSBcInZhciBpbnB1dHMgPSBcIiArIHRoaXMub3B0aW1pemVkLmRhdGEuaW5wdXRzLmxlbmd0aCArXG5cdFx0XHRcIjtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcInZhciBvdXRwdXRzID0gXCIgKyB0aGlzLm9wdGltaXplZC5kYXRhLm91dHB1dHMubGVuZ3RoICtcblx0XHRcIjtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcInZhciBGID0gbnVsbDtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcInZhciBhY3RpdmF0ZSA9IFwiICsgdGhpcy5vcHRpbWl6ZWQuYWN0aXZhdGUudG9TdHJpbmcoKSArXG5cdFx0XCI7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgcHJvcGFnYXRlID0gXCIgKyB0aGlzLm9wdGltaXplZC5wcm9wYWdhdGUudG9TdHJpbmcoKSArXG5cdFx0XCI7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJvbm1lc3NhZ2UgPSBmdW5jdGlvbihlKXtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcIkYgPSBlLmRhdGEubWVtb3J5QnVmZmVyO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwiaWYgKGUuZGF0YS5hY3Rpb24gPT0gJ2FjdGl2YXRlJyl7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJpZiAoZS5kYXRhLmlucHV0Lmxlbmd0aCA9PSBpbnB1dHMpe1xcblwiO1xuXHRcdGhhcmRjb2RlICs9XG5cdFx0XCJwb3N0TWVzc2FnZSggeyBhY3Rpb246ICdhY3RpdmF0ZScsIG91dHB1dDogYWN0aXZhdGUoZS5kYXRhLmlucHV0KSwgbWVtb3J5QnVmZmVyOiBGIH0sIFtGLmJ1ZmZlcl0pO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwifVxcbn1cXG5lbHNlIGlmIChlLmRhdGEuYWN0aW9uID09ICdwcm9wYWdhdGUnKXtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcInByb3BhZ2F0ZShlLmRhdGEucmF0ZSwgZS5kYXRhLnRhcmdldCk7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz1cblx0XHRcInBvc3RNZXNzYWdlKHsgYWN0aW9uOiAncHJvcGFnYXRlJywgbWVtb3J5QnVmZmVyOiBGIH0sIFtGLmJ1ZmZlcl0pO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwifVxcbn1cXG5cIjtcblxuXHRcdHZhciBibG9iID0gbmV3IEJsb2IoW2hhcmRjb2RlXSk7XG5cdFx0dmFyIGJsb2JVUkwgPSAoPGFueT53aW5kb3cpLlVSTC5jcmVhdGVPYmplY3RVUkwoYmxvYik7XG5cblx0XHRyZXR1cm4gbmV3IFdvcmtlcihibG9iVVJMKTtcblx0fVxuXG5cdC8vIHJldHVybnMgYSBjb3B5IG9mIHRoZSBuZXR3b3JrXG5cdGNsb25lKGlnbm9yZVRyYWNlcykge1xuXHRcdHJldHVybiBOZXR3b3JrLmZyb21KU09OKHRoaXMudG9KU09OKGlnbm9yZVRyYWNlcykpO1xuXHR9XG5cblx0c3RhdGljIGZyb21KU09OKGpzb24pIHtcblxuXHRcdHZhciBuZXVyb25zID0gW107XG5cblx0XHR2YXIgbGF5ZXJzID0ge1xuXHRcdFx0aW5wdXQ6IG5ldyBsYXllci5MYXllcigwKSxcblx0XHRcdGhpZGRlbjogW10sXG5cdFx0XHRvdXRwdXQ6IG5ldyBsYXllci5MYXllcigwKVxuXHRcdH1cblx0XHRcblxuXHRcdGZvciAodmFyIGkgaW4ganNvbi5uZXVyb25zKSB7XG5cdFx0XHR2YXIgY29uZmlnID0ganNvbi5uZXVyb25zW2ldO1xuXG5cdFx0XHR2YXIgbmV1cm9uID0gbmV3IF9uZXVyb24uTmV1cm9uKCk7XG5cdFx0XHRuZXVyb24udHJhY2UuZWxlZ2liaWxpdHkgPSBjb25maWcudHJhY2UuZWxlZ2liaWxpdHk7XG5cdFx0XHRuZXVyb24udHJhY2UuZXh0ZW5kZWQgPSBjb25maWcudHJhY2UuZXh0ZW5kZWQ7XG5cdFx0XHRuZXVyb24uc3RhdGUgPSBjb25maWcuc3RhdGU7XG5cdFx0XHRuZXVyb24ub2xkID0gY29uZmlnLm9sZDtcblx0XHRcdG5ldXJvbi5hY3RpdmF0aW9uID0gY29uZmlnLmFjdGl2YXRpb247XG5cdFx0XHRuZXVyb24uYmlhcyA9IGNvbmZpZy5iaWFzO1xuXHRcdFx0bmV1cm9uLnNxdWFzaCA9IGNvbmZpZy5zcXVhc2ggaW4gU3F1YXNoID8gU3F1YXNoW2NvbmZpZy5zcXVhc2hdIDpcblx0XHRcdFx0U3F1YXNoLkxPR0lTVElDO1xuXHRcdFx0bmV1cm9ucy5wdXNoKG5ldXJvbik7XG5cblx0XHRcdGlmIChjb25maWcubGF5ZXIgPT0gJ2lucHV0Jylcblx0XHRcdFx0bGF5ZXJzLmlucHV0LmFkZChuZXVyb24pO1xuXHRcdFx0ZWxzZSBpZiAoY29uZmlnLmxheWVyID09ICdvdXRwdXQnKVxuXHRcdFx0XHRsYXllcnMub3V0cHV0LmFkZChuZXVyb24pO1xuXHRcdFx0ZWxzZSB7XG5cdFx0XHRcdGlmICh0eXBlb2YgbGF5ZXJzLmhpZGRlbltjb25maWcubGF5ZXJdID09ICd1bmRlZmluZWQnKVxuXHRcdFx0XHRcdGxheWVycy5oaWRkZW5bY29uZmlnLmxheWVyXSA9IG5ldyBsYXllci5MYXllcigwKTtcblx0XHRcdFx0bGF5ZXJzLmhpZGRlbltjb25maWcubGF5ZXJdLmFkZChuZXVyb24pO1xuXHRcdFx0fVxuXHRcdH1cblxuXHRcdGZvciAodmFyIGkgaW4ganNvbi5jb25uZWN0aW9ucykge1xuXHRcdFx0dmFyIGNvbmZpZyA9IGpzb24uY29ubmVjdGlvbnNbaV07XG5cdFx0XHR2YXIgZnJvbSA9IG5ldXJvbnNbY29uZmlnLmZyb21dO1xuXHRcdFx0dmFyIHRvID0gbmV1cm9uc1tjb25maWcudG9dO1xuXHRcdFx0dmFyIHdlaWdodCA9IGNvbmZpZy53ZWlnaHRcblx0XHRcdHZhciBnYXRlciA9IG5ldXJvbnNbY29uZmlnLmdhdGVyXTtcblxuXHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBmcm9tLnByb2plY3QodG8sIHdlaWdodCk7XG5cdFx0XHRpZiAoZ2F0ZXIpXG5cdFx0XHRcdGdhdGVyLmdhdGUoY29ubmVjdGlvbik7XG5cdFx0fVxuXG5cdFx0cmV0dXJuIG5ldyBOZXR3b3JrKGxheWVycyk7XG5cdH1cbn1cblxuZXhwb3J0IG1vZHVsZSBOZXR3b3JrIHtcblx0ZXhwb3J0IGludGVyZmFjZSBJTmV0d29ya05ldXJvbiB7XG5cdFx0bmV1cm9uOiBfbmV1cm9uLk5ldXJvbjtcblx0XHRsYXllcjogc3RyaW5nO1xuXHR9XG59IiwiLy8vIDxyZWZlcmVuY2UgcGF0aD1cInN5bmFwdGljLnRzXCIgLz5cblxuaW1wb3J0IFN5bmFwdGljID0gcmVxdWlyZSgnLi9zeW5hcHRpYycpO1xuaW1wb3J0IFNxdWFzaCA9IHJlcXVpcmUoJy4vc3F1YXNoJyk7XG5cbi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgTkVVUk9OXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuXG4vKiBUUyBDSEFOR0VTOlxuXG5cdE5vdyBOZXVyb24uY29ubmVjdGVkKG5ldXJvbikgcmV0dXJucyBudWxsIGluc3RlYWQgb2YgZmFsc2VcblxuKi9cblxuZXhwb3J0IGNsYXNzIE5ldXJvbiB7XG5cdElEID0gTmV1cm9uLnVpZCgpO1xuXHRsYWJlbCA9IG51bGw7XG5cdGNvbm5lY3Rpb25zOiBOZXVyb24uSU5ldXJvbkNvbm5lY3Rpb25zID0ge1xuXHRcdGlucHV0czoge30sXG5cdFx0cHJvamVjdGVkOiB7fSxcblx0XHRnYXRlZDoge31cblx0fTtcblx0ZXJyb3IgPSB7XG5cdFx0cmVzcG9uc2liaWxpdHk6IDAsXG5cdFx0cHJvamVjdGVkOiAwLFxuXHRcdGdhdGVkOiAwXG5cdH07XG5cdHRyYWNlID0ge1xuXHRcdGVsZWdpYmlsaXR5OiB7fSxcblx0XHRleHRlbmRlZDoge30sXG5cdFx0aW5mbHVlbmNlczoge31cblx0fTtcblx0c3RhdGUgPSAwO1xuXHRvbGQgPSAwO1xuXHRhY3RpdmF0aW9uID0gMDtcblx0c2VsZmNvbm5lY3Rpb24gPSBuZXcgTmV1cm9uLkNvbm5lY3Rpb24odGhpcywgdGhpcywgMCk7IC8vIHdlaWdodCA9IDAgLT4gbm90IGNvbm5lY3RlZFxuXHRzcXVhc2ggPSBTcXVhc2guTE9HSVNUSUM7XG5cdG5laWdoYm9vcnMgPSB7fTtcblx0YmlhcyA9IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xO1xuXHRkZXJpdmF0aXZlID0gMDtcbiAgXG5cdC8vIGFjdGl2YXRlIHRoZSBuZXVyb25cblx0YWN0aXZhdGUoaW5wdXQ/OiBudW1iZXIpIHtcblx0XHQvLyBhY3RpdmF0aW9uIGZyb20gZW52aXJvbWVudCAoZm9yIGlucHV0IG5ldXJvbnMpXG5cdFx0aWYgKHR5cGVvZiBpbnB1dCAhPSAndW5kZWZpbmVkJykge1xuXHRcdFx0dGhpcy5hY3RpdmF0aW9uID0gaW5wdXQ7XG5cdFx0XHR0aGlzLmRlcml2YXRpdmUgPSAwO1xuXHRcdFx0dGhpcy5iaWFzID0gMDtcblx0XHRcdHJldHVybiB0aGlzLmFjdGl2YXRpb247XG5cdFx0fVxuXG5cdFx0Ly8gb2xkIHN0YXRlXG5cdFx0dGhpcy5vbGQgPSB0aGlzLnN0YXRlO1xuXG5cdFx0Ly8gZXEuIDE1XG5cdFx0dGhpcy5zdGF0ZSA9IHRoaXMuc2VsZmNvbm5lY3Rpb24uZ2FpbiAqIHRoaXMuc2VsZmNvbm5lY3Rpb24ud2VpZ2h0ICpcblx0XHR0aGlzLnN0YXRlICsgdGhpcy5iaWFzO1xuXG5cdFx0Zm9yICh2YXIgaSBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0dmFyIHRoZUlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cdFx0XHR0aGlzLnN0YXRlICs9IHRoZUlucHV0LmZyb20uYWN0aXZhdGlvbiAqIHRoZUlucHV0LndlaWdodCAqIHRoZUlucHV0LmdhaW47XG5cdFx0fVxuXG5cdFx0Ly8gZXEuIDE2XG5cdFx0dGhpcy5hY3RpdmF0aW9uID0gdGhpcy5zcXVhc2godGhpcy5zdGF0ZSk7XG5cblx0XHQvLyBmJyhzKVxuXHRcdHRoaXMuZGVyaXZhdGl2ZSA9IHRoaXMuc3F1YXNoKHRoaXMuc3RhdGUsIHRydWUpO1xuXG5cdFx0Ly8gdXBkYXRlIHRyYWNlc1xuXHRcdHZhciBpbmZsdWVuY2VzID0gW107XG5cdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0Ly8gZXh0ZW5kZWQgZWxlZ2liaWxpdHkgdHJhY2Vcblx0XHRcdHZhciB4dHJhY2UgPSB0aGlzLnRyYWNlLmV4dGVuZGVkW2lkXTtcblx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXG5cdFx0XHQvLyBpZiBnYXRlZCBuZXVyb24ncyBzZWxmY29ubmVjdGlvbiBpcyBnYXRlZCBieSB0aGlzIHVuaXQsIHRoZSBpbmZsdWVuY2Uga2VlcHMgdHJhY2sgb2YgdGhlIG5ldXJvbidzIG9sZCBzdGF0ZVxuXHRcdFx0dmFyIGluZmx1ZW5jZSA9IG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzID8gbmV1cm9uLm9sZCA6IDA7XG5cblx0XHRcdC8vIGluZGV4IHJ1bnMgb3ZlciBhbGwgdGhlIGluY29taW5nIGNvbm5lY3Rpb25zIHRvIHRoZSBnYXRlZCBuZXVyb24gdGhhdCBhcmUgZ2F0ZWQgYnkgdGhpcyB1bml0XG5cdFx0XHRmb3IgKHZhciBpbmNvbWluZyBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXSkgeyAvLyBjYXB0dXJlcyB0aGUgZWZmZWN0IHRoYXQgaGFzIGFuIGlucHV0IGNvbm5lY3Rpb24gdG8gdGhpcyB1bml0LCBvbiBhIG5ldXJvbiB0aGF0IGlzIGdhdGVkIGJ5IHRoaXMgdW5pdFxuXHRcdFx0XHRpbmZsdWVuY2UgKz0gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF1baW5jb21pbmddLndlaWdodCAqXG5cdFx0XHRcdHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdW2luY29taW5nXS5mcm9tLmFjdGl2YXRpb247XG5cdFx0XHR9XG5cdFx0XHRpbmZsdWVuY2VzW25ldXJvbi5JRF0gPSBpbmZsdWVuY2U7XG5cdFx0fVxuXG5cdFx0Zm9yICh2YXIgaSBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0dmFyIHRoZUlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cblx0XHRcdC8vIGVsZWdpYmlsaXR5IHRyYWNlIC0gRXEuIDE3XG5cdFx0XHR0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RoZUlucHV0LklEXSA9IHRoaXMuc2VsZmNvbm5lY3Rpb24uZ2FpbiAqIHRoaXMuc2VsZmNvbm5lY3Rpb25cblx0XHRcdC53ZWlnaHQgKiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RoZUlucHV0LklEXSArIHRoZUlucHV0LmdhaW4gKiB0aGVJbnB1dC5mcm9tXG5cdFx0XHQuYWN0aXZhdGlvbjtcblxuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHQvLyBleHRlbmRlZCBlbGVnaWJpbGl0eSB0cmFjZVxuXHRcdFx0XHR2YXIgeHRyYWNlID0gdGhpcy50cmFjZS5leHRlbmRlZFtpZF07XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHR2YXIgaW5mbHVlbmNlID0gaW5mbHVlbmNlc1tuZXVyb24uSURdO1xuXG5cdFx0XHRcdC8vIGVxLiAxOFxuXHRcdFx0XHR4dHJhY2VbdGhlSW5wdXQuSURdID0gbmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhaW4gKiBuZXVyb24uc2VsZmNvbm5lY3Rpb25cblx0XHRcdFx0LndlaWdodCAqIHh0cmFjZVt0aGVJbnB1dC5JRF0gKyB0aGlzLmRlcml2YXRpdmUgKiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W1xuXHRcdFx0XHR0aGVJbnB1dC5JRF0gKiBpbmZsdWVuY2U7XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFx0Ly8gIHVwZGF0ZSBnYXRlZCBjb25uZWN0aW9uJ3MgZ2FpbnNcblx0XHRmb3IgKHZhciBjb25uZWN0aW9uIGluIHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWQpIHtcblx0XHRcdHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWRbY29ubmVjdGlvbl0uZ2FpbiA9IHRoaXMuYWN0aXZhdGlvbjtcblx0XHR9XG5cblx0XHRyZXR1cm4gdGhpcy5hY3RpdmF0aW9uO1xuXHR9XG5cblx0Ly8gYmFjay1wcm9wYWdhdGUgdGhlIGVycm9yXG5cdHByb3BhZ2F0ZShyYXRlOiBudW1iZXIsIHRhcmdldD86IG51bWJlcikge1xuXHRcdC8vIGVycm9yIGFjY3VtdWxhdG9yXG5cdFx0dmFyIGVycm9yID0gMDtcblxuXHRcdC8vIHdoZXRoZXIgb3Igbm90IHRoaXMgbmV1cm9uIGlzIGluIHRoZSBvdXRwdXQgbGF5ZXJcblx0XHR2YXIgaXNPdXRwdXQgPSB0eXBlb2YgdGFyZ2V0ICE9ICd1bmRlZmluZWQnICYmIHRhcmdldCAhPSBudWxsO1xuXG5cdFx0Ly8gb3V0cHV0IG5ldXJvbnMgZ2V0IHRoZWlyIGVycm9yIGZyb20gdGhlIGVudmlyb21lbnRcblx0XHRpZiAoaXNPdXRwdXQpXG5cdFx0XHR0aGlzLmVycm9yLnJlc3BvbnNpYmlsaXR5ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgPSB0YXJnZXQgLSB0aGlzLmFjdGl2YXRpb247IC8vIEVxLiAxMFxuICAgIFxuXHRcdGVsc2UgLy8gdGhlIHJlc3Qgb2YgdGhlIG5ldXJvbiBjb21wdXRlIHRoZWlyIGVycm9yIHJlc3BvbnNpYmlsaXRpZXMgYnkgYmFja3Byb3BhZ2F0aW9uXG5cdFx0e1xuXHRcdFx0Ly8gZXJyb3IgcmVzcG9uc2liaWxpdGllcyBmcm9tIGFsbCB0aGUgY29ubmVjdGlvbnMgcHJvamVjdGVkIGZyb20gdGhpcyBuZXVyb25cblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG5cdFx0XHRcdHZhciBjb25uZWN0aW9uID0gdGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbaWRdO1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gY29ubmVjdGlvbi50bztcblx0XHRcdFx0Ly8gRXEuIDIxXG5cdFx0XHRcdGVycm9yICs9IG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSAqIGNvbm5lY3Rpb24uZ2FpbiAqIGNvbm5lY3Rpb24ud2VpZ2h0O1xuXHRcdFx0fVxuXG5cdFx0XHQvLyBwcm9qZWN0ZWQgZXJyb3IgcmVzcG9uc2liaWxpdHlcblx0XHRcdHRoaXMuZXJyb3IucHJvamVjdGVkID0gdGhpcy5kZXJpdmF0aXZlICogZXJyb3I7XG5cblx0XHRcdGVycm9yID0gMDtcblx0XHRcdC8vIGVycm9yIHJlc3BvbnNpYmlsaXRpZXMgZnJvbSBhbGwgdGhlIGNvbm5lY3Rpb25zIGdhdGVkIGJ5IHRoaXMgbmV1cm9uXG5cdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdOyAvLyBnYXRlZCBuZXVyb25cblx0XHRcdFx0dmFyIGluZmx1ZW5jZSA9IG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzID8gbmV1cm9uLm9sZCA6IDA7IC8vIGlmIGdhdGVkIG5ldXJvbidzIHNlbGZjb25uZWN0aW9uIGlzIGdhdGVkIGJ5IHRoaXMgbmV1cm9uXG5cblx0XHRcdFx0Ly8gaW5kZXggcnVucyBvdmVyIGFsbCB0aGUgY29ubmVjdGlvbnMgdG8gdGhlIGdhdGVkIG5ldXJvbiB0aGF0IGFyZSBnYXRlZCBieSB0aGlzIG5ldXJvblxuXHRcdFx0XHRmb3IgKHZhciBpbnB1dCBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbaWRdKSB7IC8vIGNhcHR1cmVzIHRoZSBlZmZlY3QgdGhhdCB0aGUgaW5wdXQgY29ubmVjdGlvbiBvZiB0aGlzIG5ldXJvbiBoYXZlLCBvbiBhIG5ldXJvbiB3aGljaCBpdHMgaW5wdXQvcyBpcy9hcmUgZ2F0ZWQgYnkgdGhpcyBuZXVyb25cblx0XHRcdFx0XHRpbmZsdWVuY2UgKz0gdGhpcy50cmFjZS5pbmZsdWVuY2VzW2lkXVtpbnB1dF0ud2VpZ2h0ICogdGhpcy50cmFjZS5pbmZsdWVuY2VzW1xuXHRcdFx0XHRcdG5ldXJvbi5JRF1baW5wdXRdLmZyb20uYWN0aXZhdGlvbjtcblx0XHRcdFx0fVxuXHRcdFx0XHQvLyBlcS4gMjJcblx0XHRcdFx0ZXJyb3IgKz0gbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5ICogaW5mbHVlbmNlO1xuXHRcdFx0fVxuXG5cdFx0XHQvLyBnYXRlZCBlcnJvciByZXNwb25zaWJpbGl0eVxuXHRcdFx0dGhpcy5lcnJvci5nYXRlZCA9IHRoaXMuZGVyaXZhdGl2ZSAqIGVycm9yO1xuXG5cdFx0XHQvLyBlcnJvciByZXNwb25zaWJpbGl0eSAtIEVxLiAyM1xuXHRcdFx0dGhpcy5lcnJvci5yZXNwb25zaWJpbGl0eSA9IHRoaXMuZXJyb3IucHJvamVjdGVkICsgdGhpcy5lcnJvci5nYXRlZDtcblx0XHR9XG5cblx0XHQvLyBsZWFybmluZyByYXRlXG5cdFx0cmF0ZSA9IHJhdGUgfHwgLjE7XG5cblx0XHQvLyBhZGp1c3QgYWxsIHRoZSBuZXVyb24ncyBpbmNvbWluZyBjb25uZWN0aW9uc1xuXHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHR2YXIgdGhlSW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cblx0XHRcdC8vIEVxLiAyNFxuXHRcdFx0dmFyIGdyYWRpZW50ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgKiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RoZUlucHV0LklEXTtcblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdGdyYWRpZW50ICs9IG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSAqIHRoaXMudHJhY2UuZXh0ZW5kZWRbXG5cdFx0XHRcdG5ldXJvbi5JRF1bdGhlSW5wdXQuSURdO1xuXHRcdFx0fVxuXHRcdFx0dGhlSW5wdXQud2VpZ2h0ICs9IHJhdGUgKiBncmFkaWVudDsgLy8gYWRqdXN0IHdlaWdodHMgLSBha2EgbGVhcm5cblx0XHR9XG5cblx0XHQvLyBhZGp1c3QgYmlhc1xuXHRcdHRoaXMuYmlhcyArPSByYXRlICogdGhpcy5lcnJvci5yZXNwb25zaWJpbGl0eTtcblx0fVxuXG5cdHByb2plY3QobmV1cm9uLCB3ZWlnaHQ/OiBudW1iZXIpOiBOZXVyb24uQ29ubmVjdGlvbiB7XG5cdFx0Ly8gc2VsZi1jb25uZWN0aW9uXG5cdFx0aWYgKG5ldXJvbiA9PSB0aGlzKSB7XG5cdFx0XHR0aGlzLnNlbGZjb25uZWN0aW9uLndlaWdodCA9IDE7XG5cdFx0XHRyZXR1cm4gdGhpcy5zZWxmY29ubmVjdGlvbjtcblx0XHR9XG5cblx0XHQvLyBjaGVjayBpZiBjb25uZWN0aW9uIGFscmVhZHkgZXhpc3RzXG5cdFx0dmFyIGNvbm5lY3RlZCA9IHRoaXMuY29ubmVjdGVkKG5ldXJvbik7XG5cdFx0aWYgKGNvbm5lY3RlZCAmJiBjb25uZWN0ZWQudHlwZSA9PSBcInByb2plY3RlZFwiKSB7XG5cdFx0XHQvLyB1cGRhdGUgY29ubmVjdGlvblxuXHRcdFx0aWYgKHR5cGVvZiB3ZWlnaHQgIT0gJ3VuZGVmaW5lZCcpXG5cdFx0XHRcdGNvbm5lY3RlZC5jb25uZWN0aW9uLndlaWdodCA9IHdlaWdodDtcblx0XHRcdC8vIHJldHVybiBleGlzdGluZyBjb25uZWN0aW9uXG5cdFx0XHRyZXR1cm4gY29ubmVjdGVkLmNvbm5lY3Rpb247XG5cdFx0fSBlbHNlIHtcblx0XHRcdC8vIGNyZWF0ZSBhIG5ldyBjb25uZWN0aW9uXG5cdFx0XHR2YXIgY29ubmVjdGlvbiA9IG5ldyBOZXVyb24uQ29ubmVjdGlvbih0aGlzLCBuZXVyb24sIHdlaWdodCk7XG5cdFx0fVxuXG5cdFx0Ly8gcmVmZXJlbmNlIGFsbCB0aGUgY29ubmVjdGlvbnMgYW5kIHRyYWNlc1xuXHRcdHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcblx0XHR0aGlzLm5laWdoYm9vcnNbbmV1cm9uLklEXSA9IG5ldXJvbjtcblx0XHRuZXVyb24uY29ubmVjdGlvbnMuaW5wdXRzW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcblx0XHRuZXVyb24udHJhY2UuZWxlZ2liaWxpdHlbY29ubmVjdGlvbi5JRF0gPSAwO1xuXG5cdFx0Zm9yICh2YXIgaWQgaW4gbmV1cm9uLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHR2YXIgdHJhY2UgPSBuZXVyb24udHJhY2UuZXh0ZW5kZWRbaWRdO1xuXHRcdFx0dHJhY2VbY29ubmVjdGlvbi5JRF0gPSAwO1xuXHRcdH1cblxuXHRcdHJldHVybiBjb25uZWN0aW9uO1xuXHR9XG5cblx0Z2F0ZShjb25uZWN0aW9uKSB7XG5cdFx0Ly8gYWRkIGNvbm5lY3Rpb24gdG8gZ2F0ZWQgbGlzdFxuXHRcdHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWRbY29ubmVjdGlvbi5JRF0gPSBjb25uZWN0aW9uO1xuXG5cdFx0dmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24udG87XG5cdFx0aWYgKCEobmV1cm9uLklEIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpKSB7XG5cdFx0XHQvLyBleHRlbmRlZCB0cmFjZVxuXHRcdFx0dGhpcy5uZWlnaGJvb3JzW25ldXJvbi5JRF0gPSBuZXVyb247XG5cdFx0XHR2YXIgeHRyYWNlID0gdGhpcy50cmFjZS5leHRlbmRlZFtuZXVyb24uSURdID0ge307XG5cdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHR2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cdFx0XHRcdHh0cmFjZVtpbnB1dC5JRF0gPSAwO1xuXHRcdFx0fVxuXHRcdH1cblxuXHRcdC8vIGtlZXAgdHJhY2tcblx0XHRpZiAobmV1cm9uLklEIGluIHRoaXMudHJhY2UuaW5mbHVlbmNlcylcblx0XHRcdHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdLnB1c2goY29ubmVjdGlvbik7XG5cdFx0ZWxzZVxuXHRcdFx0dGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0gPSBbY29ubmVjdGlvbl07XG5cblx0XHQvLyBzZXQgZ2F0ZXJcblx0XHRjb25uZWN0aW9uLmdhdGVyID0gdGhpcztcblx0fVxuICBcblx0Ly8gcmV0dXJucyB0cnVlIG9yIGZhbHNlIHdoZXRoZXIgdGhlIG5ldXJvbiBpcyBzZWxmLWNvbm5lY3RlZCBvciBub3Rcblx0c2VsZmNvbm5lY3RlZCgpIHtcblx0XHRyZXR1cm4gdGhpcy5zZWxmY29ubmVjdGlvbi53ZWlnaHQgIT09IDA7XG5cdH1cblxuXHQvLyByZXR1cm5zIHRydWUgb3IgZmFsc2Ugd2hldGhlciB0aGUgbmV1cm9uIGlzIGNvbm5lY3RlZCB0byBhbm90aGVyIG5ldXJvbiAocGFyYW1ldGVyKVxuXHRjb25uZWN0ZWQobmV1cm9uKSB7XG5cdFx0dmFyIHJlc3VsdDoge1xuXHRcdFx0dHlwZTogc3RyaW5nO1xuXHRcdFx0Y29ubmVjdGlvbjogTmV1cm9uLkNvbm5lY3Rpb247XG5cdFx0fSA9IHtcblx0XHRcdFx0dHlwZTogbnVsbCxcblx0XHRcdFx0Y29ubmVjdGlvbjogbnVsbFxuXHRcdFx0fTtcblxuXHRcdGlmICh0aGlzID09IG5ldXJvbikge1xuXHRcdFx0aWYgKHRoaXMuc2VsZmNvbm5lY3RlZCgpKSB7XG5cdFx0XHRcdHJlc3VsdC50eXBlID0gJ3NlbGZjb25uZWN0aW9uJztcblx0XHRcdFx0cmVzdWx0LmNvbm5lY3Rpb24gPSB0aGlzLnNlbGZjb25uZWN0aW9uO1xuXHRcdFx0XHRyZXR1cm4gcmVzdWx0O1xuXHRcdFx0fSBlbHNlXG5cdFx0XHRcdHJldHVybiBudWxsO1xuXHRcdH1cblxuXHRcdGZvciAodmFyIHR5cGUgaW4gdGhpcy5jb25uZWN0aW9ucykge1xuXHRcdFx0Zm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zW3R5cGVdKSB7XG5cdFx0XHRcdHZhciBjb25uZWN0aW9uID0gdGhpcy5jb25uZWN0aW9uc1t0eXBlXVtjb25uZWN0aW9uXTtcblx0XHRcdFx0aWYgKGNvbm5lY3Rpb24udG8gPT0gbmV1cm9uKSB7XG5cdFx0XHRcdFx0cmVzdWx0LnR5cGUgPSB0eXBlO1xuXHRcdFx0XHRcdHJlc3VsdC5jb25uZWN0aW9uID0gY29ubmVjdGlvbjtcblx0XHRcdFx0XHRyZXR1cm4gcmVzdWx0O1xuXHRcdFx0XHR9IGVsc2UgaWYgKGNvbm5lY3Rpb24uZnJvbSA9PSBuZXVyb24pIHtcblx0XHRcdFx0XHRyZXN1bHQudHlwZSA9IHR5cGU7XG5cdFx0XHRcdFx0cmVzdWx0LmNvbm5lY3Rpb24gPSBjb25uZWN0aW9uO1xuXHRcdFx0XHRcdHJldHVybiByZXN1bHQ7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHR9XG5cblx0XHRyZXR1cm4gbnVsbDtcblx0fVxuXG5cdC8vIGNsZWFycyBhbGwgdGhlIHRyYWNlcyAodGhlIG5ldXJvbiBmb3JnZXRzIGl0J3MgY29udGV4dCwgYnV0IHRoZSBjb25uZWN0aW9ucyByZW1haW4gaW50YWN0KVxuXHRjbGVhcigpIHtcblxuXHRcdGZvciAodmFyIHRyYWNlIGluIHRoaXMudHJhY2UuZWxlZ2liaWxpdHkpXG5cdFx0XHR0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RyYWNlXSA9IDA7XG5cblx0XHRmb3IgKHZhciB0cmFjZSBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKVxuXHRcdFx0Zm9yICh2YXIgZXh0ZW5kZWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZFt0cmFjZV0pXG5cdFx0XHRcdHRoaXMudHJhY2UuZXh0ZW5kZWRbdHJhY2VdW2V4dGVuZGVkXSA9IDA7XG5cblx0XHR0aGlzLmVycm9yLnJlc3BvbnNpYmlsaXR5ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgPSB0aGlzLmVycm9yLmdhdGVkID0gMDtcblx0fVxuXG5cdC8vIGFsbCB0aGUgY29ubmVjdGlvbnMgYXJlIHJhbmRvbWl6ZWQgYW5kIHRoZSB0cmFjZXMgYXJlIGNsZWFyZWRcblx0cmVzZXQoKSB7XG5cdFx0dGhpcy5jbGVhcigpO1xuXG5cdFx0Zm9yICh2YXIgdHlwZSBpbiB0aGlzLmNvbm5lY3Rpb25zKVxuXHRcdFx0Zm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zW3R5cGVdKVxuXHRcdFx0XHR0aGlzLmNvbm5lY3Rpb25zW3R5cGVdW2Nvbm5lY3Rpb25dLndlaWdodCA9IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xO1xuXHRcdHRoaXMuYmlhcyA9IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xO1xuXG5cdFx0dGhpcy5vbGQgPSB0aGlzLnN0YXRlID0gdGhpcy5hY3RpdmF0aW9uID0gMDtcblx0fVxuXHRcblxuICBcblxuXHQvLyBoYXJkY29kZXMgdGhlIGJlaGF2aW91ciBvZiB0aGUgbmV1cm9uIGludG8gYW4gb3B0aW1pemVkIGZ1bmN0aW9uXG5cdG9wdGltaXplKG9wdGltaXplZCwgbGF5ZXIpOiBTeW5hcHRpYy5JQ29tcGlsZWRQYXJhbWV0ZXJzIHtcblxuXHRcdG9wdGltaXplZCA9IG9wdGltaXplZCB8fCB7fTtcblx0XHR2YXIgdGhhdCA9IHRoaXM7XG5cdFx0dmFyIHN0b3JlX2FjdGl2YXRpb24gPSBbXTtcblx0XHR2YXIgc3RvcmVfdHJhY2UgPSBbXTtcblx0XHR2YXIgc3RvcmVfcHJvcGFnYXRpb24gPSBbXTtcblx0XHR2YXIgdmFySUQgPSBvcHRpbWl6ZWQubWVtb3J5IHx8IDA7XG5cdFx0dmFyIG5ldXJvbnMgPSBvcHRpbWl6ZWQubmV1cm9ucyB8fCAxO1xuXHRcdHZhciBpbnB1dHMgPSBvcHRpbWl6ZWQuaW5wdXRzIHx8IFtdO1xuXHRcdHZhciB0YXJnZXRzID0gb3B0aW1pemVkLnRhcmdldHMgfHwgW107XG5cdFx0dmFyIG91dHB1dHMgPSBvcHRpbWl6ZWQub3V0cHV0cyB8fCBbXTtcblx0XHR2YXIgdmFyaWFibGVzID0gb3B0aW1pemVkLnZhcmlhYmxlcyB8fCB7fTtcblx0XHR2YXIgYWN0aXZhdGlvbl9zZW50ZW5jZXMgPSBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXMgfHwgW107XG5cdFx0dmFyIHRyYWNlX3NlbnRlbmNlcyA9IG9wdGltaXplZC50cmFjZV9zZW50ZW5jZXMgfHwgW107XG5cdFx0dmFyIHByb3BhZ2F0aW9uX3NlbnRlbmNlcyA9IG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMgfHwgW107XG5cdFx0dmFyIGxheWVycyA9IG9wdGltaXplZC5sYXllcnMgfHwgeyBfX2NvdW50OiAwLCBfX25ldXJvbjogMCB9O1xuXG5cdFx0Ly8gYWxsb2NhdGUgc2VudGVuY2VzXG5cdFx0dmFyIGFsbG9jYXRlID0gZnVuY3Rpb24oc3RvcmUpIHtcblx0XHRcdHZhciBhbGxvY2F0ZWQgPSBsYXllciBpbiBsYXllcnMgJiYgc3RvcmVbbGF5ZXJzLl9fY291bnRdO1xuXHRcdFx0aWYgKCFhbGxvY2F0ZWQpIHtcblx0XHRcdFx0bGF5ZXJzLl9fY291bnQgPSBzdG9yZS5wdXNoKFtdKSAtIDE7XG5cdFx0XHRcdGxheWVyc1tsYXllcl0gPSBsYXllcnMuX19jb3VudDtcblx0XHRcdH1cblx0XHR9XG5cdFx0YWxsb2NhdGUoYWN0aXZhdGlvbl9zZW50ZW5jZXMpO1xuXHRcdGFsbG9jYXRlKHRyYWNlX3NlbnRlbmNlcyk7XG5cdFx0YWxsb2NhdGUocHJvcGFnYXRpb25fc2VudGVuY2VzKTtcblx0XHR2YXIgY3VycmVudExheWVyID0gbGF5ZXJzLl9fY291bnQ7XG5cblx0XHQvLyBnZXQvcmVzZXJ2ZSBzcGFjZSBpbiBtZW1vcnkgYnkgY3JlYXRpbmcgYSB1bmlxdWUgSUQgZm9yIGEgdmFyaWFibGVsXG5cdFx0dmFyIGdldFZhciA9IGZ1bmN0aW9uKC4uLmFyZ3M6IGFueVtdKSB7XG5cdFx0XHR2YXIgaWQ7XG5cdFx0XHRpZiAoYXJncy5sZW5ndGggPT0gMSkge1xuXG5cdFx0XHRcdGlmIChhcmdzWzBdID09ICd0YXJnZXQnKSB7XG5cdFx0XHRcdFx0aWQgPSAndGFyZ2V0XycgKyB0YXJnZXRzLmxlbmd0aDtcblx0XHRcdFx0XHR0YXJnZXRzLnB1c2godmFySUQpO1xuXHRcdFx0XHR9IGVsc2Vcblx0XHRcdFx0XHRpZCA9IGFyZ3NbMF07XG5cdFx0XHRcdGlmIChpZCBpbiB2YXJpYWJsZXMpXG5cdFx0XHRcdFx0cmV0dXJuIHZhcmlhYmxlc1tpZF07XG5cdFx0XHRcdHJldHVybiB2YXJpYWJsZXNbaWRdID0ge1xuXHRcdFx0XHRcdHZhbHVlOiAwLFxuXHRcdFx0XHRcdGlkOiB2YXJJRCsrXG5cdFx0XHRcdH07XG5cdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHR2YXIgZXh0ZW5kZWQgPSBhcmdzLmxlbmd0aCA+IDI7XG5cdFx0XHRcdGlmIChleHRlbmRlZClcblx0XHRcdFx0XHR2YXIgdmFsdWUgPSBhcmdzLnBvcCgpO1xuXG5cdFx0XHRcdHZhciB1bml0ID0gYXJncy5zaGlmdCgpO1xuXHRcdFx0XHR2YXIgcHJvcCA9IGFyZ3MucG9wKCk7XG5cblx0XHRcdFx0aWYgKCFleHRlbmRlZClcblx0XHRcdFx0XHR2YXIgdmFsdWUgPSB1bml0W3Byb3BdO1xuXG5cdFx0XHRcdGlkID0gcHJvcCArICdfJztcblx0XHRcdFx0Zm9yICh2YXIgcHJvcGVydHkgaW4gYXJncylcblx0XHRcdFx0XHRpZCArPSBhcmdzW3Byb3BlcnR5XSArICdfJztcblx0XHRcdFx0aWQgKz0gdW5pdC5JRDtcblx0XHRcdFx0aWYgKGlkIGluIHZhcmlhYmxlcylcblx0XHRcdFx0XHRyZXR1cm4gdmFyaWFibGVzW2lkXTtcblxuXHRcdFx0XHRyZXR1cm4gdmFyaWFibGVzW2lkXSA9IHtcblx0XHRcdFx0XHR2YWx1ZTogdmFsdWUsXG5cdFx0XHRcdFx0aWQ6IHZhcklEKytcblx0XHRcdFx0fTtcblx0XHRcdH1cblx0XHR9O1xuXG5cdFx0Ly8gYnVpbGQgc2VudGVuY2Vcblx0XHR2YXIgYnVpbGRTZW50ZW5jZSA9IGZ1bmN0aW9uKC4uLmFyZ3M6IGFueVtdKSB7XG5cdFx0XHR2YXIgc3RvcmUgPSBhcmdzLnBvcCgpO1xuXHRcdFx0dmFyIHNlbnRlbmNlID0gXCJcIjtcblx0XHRcdGZvciAodmFyIGkgaW4gYXJncylcblx0XHRcdFx0aWYgKHR5cGVvZiBhcmdzW2ldID09ICdzdHJpbmcnKVxuXHRcdFx0XHRcdHNlbnRlbmNlICs9IGFyZ3NbaV07XG5cdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRzZW50ZW5jZSArPSAnRlsnICsgYXJnc1tpXS5pZCArICddJztcblxuXHRcdFx0c3RvcmUucHVzaChzZW50ZW5jZSArICc7Jyk7XG5cdFx0fVxuXG5cdFx0Ly8gaGVscGVyIHRvIGNoZWNrIGlmIGFuIG9iamVjdCBpcyBlbXB0eVxuXHRcdHZhciBpc0VtcHR5ID0gZnVuY3Rpb24ob2JqKSB7XG5cdFx0XHRmb3IgKHZhciBwcm9wIGluIG9iaikge1xuXHRcdFx0XHRpZiAob2JqLmhhc093blByb3BlcnR5KHByb3ApKVxuXHRcdFx0XHRcdHJldHVybiBmYWxzZTtcblx0XHRcdH1cblx0XHRcdHJldHVybiB0cnVlO1xuXHRcdH07XG5cblx0XHQvLyBjaGFyYWN0ZXJpc3RpY3Mgb2YgdGhlIG5ldXJvblxuXHRcdHZhciBub1Byb2plY3Rpb25zID0gaXNFbXB0eSh0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZCk7XG5cdFx0dmFyIG5vR2F0ZXMgPSBpc0VtcHR5KHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWQpO1xuXHRcdHZhciBpc0lucHV0ID0gbGF5ZXIgPT0gJ2lucHV0JyA/IHRydWUgOiBpc0VtcHR5KHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKTtcblx0XHR2YXIgaXNPdXRwdXQgPSBsYXllciA9PSAnb3V0cHV0JyA/IHRydWUgOiBub1Byb2plY3Rpb25zICYmIG5vR2F0ZXM7XG5cblx0XHQvLyBvcHRpbWl6ZSBuZXVyb24ncyBiZWhhdmlvdXJcblx0XHR2YXIgcmF0ZSA9IGdldFZhcigncmF0ZScpO1xuXHRcdHZhciBhY3RpdmF0aW9uID0gZ2V0VmFyKHRoaXMsICdhY3RpdmF0aW9uJyk7XG5cdFx0aWYgKGlzSW5wdXQpXG5cdFx0XHRpbnB1dHMucHVzaChhY3RpdmF0aW9uLmlkKTtcblx0XHRlbHNlIHtcblx0XHRcdGFjdGl2YXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0ucHVzaChzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdHRyYWNlX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdLnB1c2goc3RvcmVfdHJhY2UpO1xuXHRcdFx0cHJvcGFnYXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0ucHVzaChzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHR2YXIgb2xkID0gZ2V0VmFyKHRoaXMsICdvbGQnKTtcblx0XHRcdHZhciBzdGF0ZSA9IGdldFZhcih0aGlzLCAnc3RhdGUnKTtcblx0XHRcdHZhciBiaWFzID0gZ2V0VmFyKHRoaXMsICdiaWFzJyk7XG5cdFx0XHRpZiAodGhpcy5zZWxmY29ubmVjdGlvbi5nYXRlcilcblx0XHRcdFx0dmFyIHNlbGZfZ2FpbiA9IGdldFZhcih0aGlzLnNlbGZjb25uZWN0aW9uLCAnZ2FpbicpO1xuXHRcdFx0aWYgKHRoaXMuc2VsZmNvbm5lY3RlZCgpKVxuXHRcdFx0XHR2YXIgc2VsZl93ZWlnaHQgPSBnZXRWYXIodGhpcy5zZWxmY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuXHRcdFx0YnVpbGRTZW50ZW5jZShvbGQsICcgPSAnLCBzdGF0ZSwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRpZiAodGhpcy5zZWxmY29ubmVjdGVkKCkpXG5cdFx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0aW9uLmdhdGVyKVxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2Uoc3RhdGUsICcgPSAnLCBzZWxmX2dhaW4sICcgKiAnLCBzZWxmX3dlaWdodCwgJyAqICcsXG5cdFx0XHRcdFx0XHRzdGF0ZSwgJyArICcsIGJpYXMsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShzdGF0ZSwgJyA9ICcsIHNlbGZfd2VpZ2h0LCAnICogJywgc3RhdGUsICcgKyAnLFxuXHRcdFx0XHRcdFx0Ymlhcywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRlbHNlXG5cdFx0XHRcdGJ1aWxkU2VudGVuY2Uoc3RhdGUsICcgPSAnLCBiaWFzLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdGZvciAodmFyIGkgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cdFx0XHRcdHZhciBpbnB1dF9hY3RpdmF0aW9uID0gZ2V0VmFyKGlucHV0LmZyb20sICdhY3RpdmF0aW9uJyk7XG5cdFx0XHRcdHZhciBpbnB1dF93ZWlnaHQgPSBnZXRWYXIoaW5wdXQsICd3ZWlnaHQnKTtcblx0XHRcdFx0aWYgKGlucHV0LmdhdGVyKVxuXHRcdFx0XHRcdHZhciBpbnB1dF9nYWluID0gZ2V0VmFyKGlucHV0LCAnZ2FpbicpO1xuXHRcdFx0XHRpZiAodGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV0uZ2F0ZXIpXG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShzdGF0ZSwgJyArPSAnLCBpbnB1dF9hY3RpdmF0aW9uLCAnICogJyxcblx0XHRcdFx0XHRcdGlucHV0X3dlaWdodCwgJyAqICcsIGlucHV0X2dhaW4sIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShzdGF0ZSwgJyArPSAnLCBpbnB1dF9hY3RpdmF0aW9uLCAnICogJyxcblx0XHRcdFx0XHRcdGlucHV0X3dlaWdodCwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHR9XG5cdFx0XHR2YXIgZGVyaXZhdGl2ZSA9IGdldFZhcih0aGlzLCAnZGVyaXZhdGl2ZScpO1xuXHRcdFx0c3dpdGNoICh0aGlzLnNxdWFzaCkge1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5MT0dJU1RJQzpcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSAoMSAvICgxICsgTWF0aC5leHAoLScsIHN0YXRlLCAnKSkpJyxcblx0XHRcdFx0XHRcdHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZGVyaXZhdGl2ZSwgJyA9ICcsIGFjdGl2YXRpb24sICcgKiAoMSAtICcsXG5cdFx0XHRcdFx0XHRhY3RpdmF0aW9uLCAnKScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJyZWFrO1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5UQU5IOlxuXHRcdFx0XHRcdHZhciBlUCA9IGdldFZhcignYXV4Jyk7XG5cdFx0XHRcdFx0dmFyIGVOID0gZ2V0VmFyKCdhdXhfMicpO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZVAsICcgPSBNYXRoLmV4cCgnLCBzdGF0ZSwgJyknLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVOLCAnID0gMSAvICcsIGVQLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSAoJywgZVAsICcgLSAnLCBlTiwgJykgLyAoJywgZVAsICcgKyAnLCBlTiwgJyknLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGRlcml2YXRpdmUsICcgPSAxIC0gKCcsIGFjdGl2YXRpb24sICcgKiAnLCBhY3RpdmF0aW9uLCAnKScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJyZWFrO1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5JREVOVElUWTpcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSAnLCBzdGF0ZSwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShkZXJpdmF0aXZlLCAnID0gMScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJyZWFrO1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5ITElNOlxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoYWN0aXZhdGlvbiwgJyA9ICsoJywgc3RhdGUsICcgPiAwKScsXG5cdFx0XHRcdFx0XHRzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGRlcml2YXRpdmUsICcgPSAxJywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnJlYWs7XG5cdFx0XHR9XG5cblxuXHRcdFx0dmFyIGluZmx1ZW5jZXMgPSBbXTtcblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0Ly8gY2FsY3VsYXRlIGV4dGVuZGVkIGVsZWdpYmlsaXR5IHRyYWNlcyBpbiBhZHZhbmNlXG4gICAgICAgIFxuXHRcdFx0XHR2YXIgeHRyYWNlID0gdGhpcy50cmFjZS5leHRlbmRlZFtpZF07XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHR2YXIgaW5mbHVlbmNlID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0dmFyIG5ldXJvbl9vbGQgPSBnZXRWYXIobmV1cm9uLCAnb2xkJyk7XG5cdFx0XHRcdHZhciBpbml0aWFsaXplZCA9IGZhbHNlO1xuXHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyID09IHRoaXMpIHtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9ICcsIG5ldXJvbl9vbGQsIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRpbml0aWFsaXplZCA9IHRydWU7XG5cdFx0XHRcdH1cblx0XHRcdFx0Zm9yICh2YXIgaW5jb21pbmcgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0pIHtcblx0XHRcdFx0XHR2YXIgaW5jb21pbmdfd2VpZ2h0ID0gZ2V0VmFyKHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdXG5cdFx0XHRcdFx0W2luY29taW5nXSwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdHZhciBpbmNvbWluZ19hY3RpdmF0aW9uID0gZ2V0VmFyKHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdXG5cdFx0XHRcdFx0W2luY29taW5nXS5mcm9tLCAnYWN0aXZhdGlvbicpO1xuXG5cdFx0XHRcdFx0aWYgKGluaXRpYWxpemVkKVxuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgKz0gJywgaW5jb21pbmdfd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdFx0aW5jb21pbmdfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdGVsc2Uge1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAnLCBpbmNvbWluZ193ZWlnaHQsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRpbmNvbWluZ19hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRpbml0aWFsaXplZCA9IHRydWU7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9XG5cblx0XHRcdFx0aW5mbHVlbmNlcy5wdXNoKG5ldXJvbi5JRCk7XG5cdFx0XHRcdGJ1aWxkU2VudGVuY2UoXCJpbmZsdWVuY2VzW1wiICsgKGluZmx1ZW5jZXMubGVuZ3RoIC0gMSkgKyBcIl0gPSBcIiwgaW5mbHVlbmNlLCBzdG9yZV90cmFjZSk7XG5cdFx0XHR9XG5cblx0XHRcdGZvciAodmFyIGkgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cdFx0XHRcdGlmIChpbnB1dC5nYXRlcilcblx0XHRcdFx0XHR2YXIgaW5wdXRfZ2FpbiA9IGdldFZhcihpbnB1dCwgJ2dhaW4nKTtcblx0XHRcdFx0dmFyIGlucHV0X2FjdGl2YXRpb24gPSBnZXRWYXIoaW5wdXQuZnJvbSwgJ2FjdGl2YXRpb24nKTtcblx0XHRcdFx0dmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzLnRyYWNlXG5cdFx0XHRcdFx0LmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG5cdFx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0ZWQoKSkge1xuXHRcdFx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0aW9uLmdhdGVyKSB7XG5cdFx0XHRcdFx0XHRpZiAoaW5wdXQuZ2F0ZXIpXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBzZWxmX2dhaW4sICcgKiAnLCBzZWxmX3dlaWdodCxcblx0XHRcdFx0XHRcdFx0XHQnICogJywgdHJhY2UsICcgKyAnLCBpbnB1dF9nYWluLCAnICogJywgaW5wdXRfYWN0aXZhdGlvbixcblx0XHRcdFx0XHRcdFx0XHRzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBzZWxmX2dhaW4sICcgKiAnLCBzZWxmX3dlaWdodCxcblx0XHRcdFx0XHRcdFx0XHQnICogJywgdHJhY2UsICcgKyAnLCBpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0XHRcdGlmIChpbnB1dC5nYXRlcilcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZSh0cmFjZSwgJyA9ICcsIHNlbGZfd2VpZ2h0LCAnICogJywgdHJhY2UsICcgKyAnLFxuXHRcdFx0XHRcdFx0XHRcdGlucHV0X2dhaW4sICcgKiAnLCBpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBzZWxmX3dlaWdodCwgJyAqICcsIHRyYWNlLCAnICsgJyxcblx0XHRcdFx0XHRcdFx0XHRpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRcdGlmIChpbnB1dC5nYXRlcilcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBpbnB1dF9nYWluLCAnICogJywgaW5wdXRfYWN0aXZhdGlvbixcblx0XHRcdFx0XHRcdFx0c3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdH1cblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHRcdC8vIGV4dGVuZGVkIGVsZWdpYmlsaXR5IHRyYWNlXG5cdFx0XHRcdFx0dmFyIHh0cmFjZSA9IHRoaXMudHJhY2UuZXh0ZW5kZWRbaWRdO1xuXHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHRcdHZhciBpbmZsdWVuY2UgPSBnZXRWYXIoJ2F1eCcpO1xuXHRcdFx0XHRcdHZhciBuZXVyb25fb2xkID0gZ2V0VmFyKG5ldXJvbiwgJ29sZCcpO1xuXG5cdFx0XHRcdFx0dmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzLnRyYWNlXG5cdFx0XHRcdFx0XHQuZWxlZ2liaWxpdHlbaW5wdXQuSURdKTtcblx0XHRcdFx0XHR2YXIgeHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdleHRlbmRlZCcsIG5ldXJvbi5JRCwgaW5wdXQuSUQsXG5cdFx0XHRcdFx0XHR0aGlzLnRyYWNlLmV4dGVuZGVkW25ldXJvbi5JRF1baW5wdXQuSURdKTtcblx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0ZWQoKSlcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fc2VsZl93ZWlnaHQgPSBnZXRWYXIobmV1cm9uLnNlbGZjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlcilcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fc2VsZl9nYWluID0gZ2V0VmFyKG5ldXJvbi5zZWxmY29ubmVjdGlvbiwgJ2dhaW4nKTtcblx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0ZWQoKSlcblx0XHRcdFx0XHRcdGlmIChuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIpXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoeHRyYWNlLCAnID0gJywgbmV1cm9uX3NlbGZfZ2FpbiwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0bmV1cm9uX3NlbGZfd2VpZ2h0LCAnICogJywgeHRyYWNlLCAnICsgJywgZGVyaXZhdGl2ZSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0dHJhY2UsICcgKiAnLCBcImluZmx1ZW5jZXNbXCIgKyBpbmZsdWVuY2VzLmluZGV4T2YobmV1cm9uLklEKSArIFwiXVwiLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoeHRyYWNlLCAnID0gJywgbmV1cm9uX3NlbGZfd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHR4dHJhY2UsICcgKyAnLCBkZXJpdmF0aXZlLCAnICogJywgdHJhY2UsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdFwiaW5mbHVlbmNlc1tcIiArIGluZmx1ZW5jZXMuaW5kZXhPZihuZXVyb24uSUQpICsgXCJdXCIsIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHh0cmFjZSwgJyA9ICcsIGRlcml2YXRpdmUsICcgKiAnLCB0cmFjZSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFwiaW5mbHVlbmNlc1tcIiArIGluZmx1ZW5jZXMuaW5kZXhPZihuZXVyb24uSUQpICsgXCJdXCIsIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdFx0Zm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zLmdhdGVkKSB7XG5cdFx0XHRcdHZhciBnYXRlZF9nYWluID0gZ2V0VmFyKHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWRbY29ubmVjdGlvbl0sICdnYWluJyk7XG5cdFx0XHRcdGJ1aWxkU2VudGVuY2UoZ2F0ZWRfZ2FpbiwgJyA9ICcsIGFjdGl2YXRpb24sIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0fVxuXHRcdH1cblx0XHRpZiAoIWlzSW5wdXQpIHtcblx0XHRcdHZhciByZXNwb25zaWJpbGl0eSA9IGdldFZhcih0aGlzLCAnZXJyb3InLCAncmVzcG9uc2liaWxpdHknLCB0aGlzLmVycm9yXG5cdFx0XHRcdC5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRpZiAoaXNPdXRwdXQpIHtcblx0XHRcdFx0dmFyIHRhcmdldCA9IGdldFZhcigndGFyZ2V0Jyk7XG5cdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgPSAnLCB0YXJnZXQsICcgLSAnLCBhY3RpdmF0aW9uLFxuXHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0XHR2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cdFx0XHRcdFx0dmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzLnRyYWNlXG5cdFx0XHRcdFx0XHQuZWxlZ2liaWxpdHlbaW5wdXQuSURdKTtcblx0XHRcdFx0XHR2YXIgaW5wdXRfd2VpZ2h0ID0gZ2V0VmFyKGlucHV0LCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbnB1dF93ZWlnaHQsICcgKz0gJywgcmF0ZSwgJyAqICgnLCByZXNwb25zaWJpbGl0eSxcblx0XHRcdFx0XHRcdCcgKiAnLCB0cmFjZSwgJyknLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdH1cblx0XHRcdFx0b3V0cHV0cy5wdXNoKGFjdGl2YXRpb24uaWQpO1xuXHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0aWYgKCFub1Byb2plY3Rpb25zICYmICFub0dhdGVzKSB7XG5cdFx0XHRcdFx0dmFyIGVycm9yID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZCkge1xuXHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZFtpZF07XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gY29ubmVjdGlvbi50bztcblx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX3dlaWdodCA9IGdldFZhcihjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcblx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdGlmIChjb25uZWN0aW9uLmdhdGVyKSB7XG5cdFx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX2dhaW4gPSBnZXRWYXIoY29ubmVjdGlvbiwgJ2dhaW4nKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShlcnJvciwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdGNvbm5lY3Rpb25fZ2FpbiwgJyAqICcsIGNvbm5lY3Rpb25fd2VpZ2h0LFxuXHRcdFx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdH0gZWxzZVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVycm9yLCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0Y29ubmVjdGlvbl93ZWlnaHQsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0dmFyIHByb2plY3RlZCA9IGdldFZhcih0aGlzLCAnZXJyb3InLCAncHJvamVjdGVkJywgdGhpcy5lcnJvci5wcm9qZWN0ZWQpO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocHJvamVjdGVkLCAnID0gJywgZGVyaXZhdGl2ZSwgJyAqICcsIGVycm9yLFxuXHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZXJyb3IsICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHRcdFx0dmFyIGluZmx1ZW5jZSA9IGdldFZhcignYXV4XzInKTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fb2xkID0gZ2V0VmFyKG5ldXJvbiwgJ29sZCcpO1xuXHRcdFx0XHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzKVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9ICcsIG5ldXJvbl9vbGQsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0Zm9yICh2YXIgaW5mbHVlbmNlSW5wdXQgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0pIHtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXVtpbmZsdWVuY2VJbnB1dF07XG5cdFx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX3dlaWdodCA9IGdldFZhcihjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHRcdHZhciBuZXVyb25fYWN0aXZhdGlvbiA9IGdldFZhcihjb25uZWN0aW9uLmZyb20sICdhY3RpdmF0aW9uJyk7XG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnICs9ICcsIGNvbm5lY3Rpb25fd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHRuZXVyb25fYWN0aXZhdGlvbiwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG5cdFx0XHRcdFx0XHRcdCdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVycm9yLCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdGluZmx1ZW5jZSwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHR2YXIgZ2F0ZWQgPSBnZXRWYXIodGhpcywgJ2Vycm9yJywgJ2dhdGVkJywgdGhpcy5lcnJvci5nYXRlZCk7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShnYXRlZCwgJyA9ICcsIGRlcml2YXRpdmUsICcgKiAnLCBlcnJvcixcblx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnID0gJywgcHJvamVjdGVkLCAnICsgJywgZ2F0ZWQsXG5cdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0XHRcdHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcblx0XHRcdFx0XHRcdHZhciBncmFkaWVudCA9IGdldFZhcignYXV4Jyk7XG5cdFx0XHRcdFx0XHR2YXIgdHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2VsZWdpYmlsaXR5JywgaW5wdXQuSUQsIHRoaXNcblx0XHRcdFx0XHRcdFx0LnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGdyYWRpZW50LCAnID0gJywgcHJvamVjdGVkLCAnICogJywgdHJhY2UsXG5cdFx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdFx0XHRcdHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuXHRcdFx0XHRcdFx0XHRcdCdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRcdFx0XHRcdHZhciB4dHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2V4dGVuZGVkJywgbmV1cm9uLklELFxuXHRcdFx0XHRcdFx0XHRcdGlucHV0LklELCB0aGlzLnRyYWNlLmV4dGVuZGVkW25ldXJvbi5JRF1baW5wdXQuSURdKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShncmFkaWVudCwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdHh0cmFjZSwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0dmFyIGlucHV0X3dlaWdodCA9IGdldFZhcihpbnB1dCwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbnB1dF93ZWlnaHQsICcgKz0gJywgcmF0ZSwgJyAqICcsIGdyYWRpZW50LFxuXHRcdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXG5cdFx0XHRcdH0gZWxzZSBpZiAobm9HYXRlcykge1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG5cdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkW2lkXTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb24gPSBjb25uZWN0aW9uLnRvO1xuXHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fd2VpZ2h0ID0gZ2V0VmFyKGNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuXHRcdFx0XHRcdFx0XHQncmVzcG9uc2liaWxpdHknLCBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkpO1xuXHRcdFx0XHRcdFx0aWYgKGNvbm5lY3Rpb24uZ2F0ZXIpIHtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fZ2FpbiA9IGdldFZhcihjb25uZWN0aW9uLCAnZ2FpbicpO1xuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSxcblx0XHRcdFx0XHRcdFx0XHQnICogJywgY29ubmVjdGlvbl9nYWluLCAnICogJywgY29ubmVjdGlvbl93ZWlnaHQsXG5cdFx0XHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fSBlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LFxuXHRcdFx0XHRcdFx0XHRcdCcgKiAnLCBjb25uZWN0aW9uX3dlaWdodCwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICo9ICcsIGRlcml2YXRpdmUsXG5cdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0XHRcdHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcblx0XHRcdFx0XHRcdHZhciB0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZWxlZ2liaWxpdHknLCBpbnB1dC5JRCwgdGhpc1xuXHRcdFx0XHRcdFx0XHQudHJhY2UuZWxlZ2liaWxpdHlbaW5wdXQuSURdKTtcblx0XHRcdFx0XHRcdHZhciBpbnB1dF93ZWlnaHQgPSBnZXRWYXIoaW5wdXQsICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5wdXRfd2VpZ2h0LCAnICs9ICcsIHJhdGUsICcgKiAoJyxcblx0XHRcdFx0XHRcdFx0cmVzcG9uc2liaWxpdHksICcgKiAnLCB0cmFjZSwgJyknLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9IGVsc2UgaWYgKG5vUHJvamVjdGlvbnMpIHtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0XHRcdHZhciBpbmZsdWVuY2UgPSBnZXRWYXIoJ2F1eCcpO1xuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9vbGQgPSBnZXRWYXIobmV1cm9uLCAnb2xkJyk7XG5cdFx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyID09IHRoaXMpXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnID0gJywgbmV1cm9uX29sZCwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9IDAnLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHRmb3IgKHZhciBpbmZsdWVuY2VJbnB1dCBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXSkge1xuXHRcdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdW2luZmx1ZW5jZUlucHV0XTtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fd2VpZ2h0ID0gZ2V0VmFyKGNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9hY3RpdmF0aW9uID0gZ2V0VmFyKGNvbm5lY3Rpb24uZnJvbSwgJ2FjdGl2YXRpb24nKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgKz0gJywgY29ubmVjdGlvbl93ZWlnaHQsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdG5ldXJvbl9hY3RpdmF0aW9uLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcblx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LFxuXHRcdFx0XHRcdFx0XHQnICogJywgaW5mbHVlbmNlLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgKj0gJywgZGVyaXZhdGl2ZSxcblx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaWRdO1xuXHRcdFx0XHRcdFx0dmFyIGdyYWRpZW50ID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZ3JhZGllbnQsICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG5cdFx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdFx0dmFyIHh0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZXh0ZW5kZWQnLCBuZXVyb24uSUQsXG5cdFx0XHRcdFx0XHRcdFx0aW5wdXQuSUQsIHRoaXMudHJhY2UuZXh0ZW5kZWRbbmV1cm9uLklEXVtpbnB1dC5JRF0pO1xuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGdyYWRpZW50LCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0eHRyYWNlLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHR2YXIgaW5wdXRfd2VpZ2h0ID0gZ2V0VmFyKGlucHV0LCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGlucHV0X3dlaWdodCwgJyArPSAnLCByYXRlLCAnICogJywgZ3JhZGllbnQsXG5cdFx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHRcdGJ1aWxkU2VudGVuY2UoYmlhcywgJyArPSAnLCByYXRlLCAnICogJywgcmVzcG9uc2liaWxpdHksXG5cdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHR9XG5cdFx0cmV0dXJuIHtcblx0XHRcdG1lbW9yeTogdmFySUQsXG5cdFx0XHRuZXVyb25zOiBuZXVyb25zICsgMSxcblx0XHRcdGlucHV0czogaW5wdXRzLFxuXHRcdFx0b3V0cHV0czogb3V0cHV0cyxcblx0XHRcdHRhcmdldHM6IHRhcmdldHMsXG5cdFx0XHR2YXJpYWJsZXM6IHZhcmlhYmxlcyxcblx0XHRcdGFjdGl2YXRpb25fc2VudGVuY2VzOiBhY3RpdmF0aW9uX3NlbnRlbmNlcyxcblx0XHRcdHRyYWNlX3NlbnRlbmNlczogdHJhY2Vfc2VudGVuY2VzLFxuXHRcdFx0cHJvcGFnYXRpb25fc2VudGVuY2VzOiBwcm9wYWdhdGlvbl9zZW50ZW5jZXMsXG5cdFx0XHRsYXllcnM6IGxheWVyc1xuXHRcdH1cblx0fVxufVxuXG5leHBvcnQgbW9kdWxlIE5ldXJvbiB7XG5cblx0ZXhwb3J0IGludGVyZmFjZSBJTmV1cm9uQ29ubmVjdGlvbnMge1xuXHRcdGlucHV0czogU3luYXB0aWMuRGljdGlvbmFyeTxOZXVyb24uQ29ubmVjdGlvbj47XG5cdFx0cHJvamVjdGVkOiB7fTtcblx0XHRnYXRlZDoge307XG5cdH1cblxuXHRleHBvcnQgY2xhc3MgQ29ubmVjdGlvbiB7XG5cdFx0SUQgPSBDb25uZWN0aW9uLnVpZCgpO1xuXHRcdGZyb207XG5cdFx0dG87XG5cdFx0Z2FpbjogbnVtYmVyID0gMTtcblx0XHR3ZWlnaHQ6IG51bWJlciA9IDA7XG5cdFx0Z2F0ZXI6IGFueSA9IG51bGw7XG5cdFx0Y29uc3RydWN0b3IoZnJvbSwgdG8sIHdlaWdodD86IG51bWJlcikge1xuXHRcdFx0aWYgKCFmcm9tIHx8ICF0bylcblx0XHRcdFx0dGhyb3cgXCJDb25uZWN0aW9uIEVycm9yOiBJbnZhbGlkIG5ldXJvbnNcIjtcblx0XHRcdHRoaXMuZnJvbSA9IGZyb207XG5cdFx0XHR0aGlzLnRvID0gdG87XG5cdFx0XHR0aGlzLndlaWdodCA9IHR5cGVvZiB3ZWlnaHQgPT0gJ3VuZGVmaW5lZCcgfHwgaXNOYU4od2VpZ2h0KSA/IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xIDpcblx0XHRcdFx0d2VpZ2h0O1xuXHRcdH1cblx0fVxuXG5cdGV4cG9ydCB2YXIgbmV1cm9uUXR5ID0gMDtcblx0ZXhwb3J0IGZ1bmN0aW9uIHVpZCgpOiBudW1iZXIge1xuXHRcdHJldHVybiBuZXVyb25RdHkrKztcblx0fVxuXG5cdGV4cG9ydCBmdW5jdGlvbiBxdWFudGl0eSgpIHtcblx0XHRyZXR1cm4ge1xuXHRcdFx0bmV1cm9uczogbmV1cm9uUXR5LFxuXHRcdFx0Y29ubmVjdGlvbnM6IENvbm5lY3Rpb24uY29ubmVjdGlvblF0eVxuXHRcdH1cblx0fVxufVxuXG5leHBvcnQgbW9kdWxlIE5ldXJvbi5Db25uZWN0aW9uIHtcblx0ZXhwb3J0IHZhciBjb25uZWN0aW9uUXR5ID0gMDtcblx0ZXhwb3J0IGZ1bmN0aW9uIHVpZCgpOiBudW1iZXIge1xuXHRcdHJldHVybiBjb25uZWN0aW9uUXR5Kys7XG5cdH1cbn0iLCJpbXBvcnQgU3luYXB0aWMgPSByZXF1aXJlKCcuL3N5bmFwdGljJyk7XG5cbi8vIHNxdWFzaGluZyBmdW5jdGlvbnNcblxuZXhwb3J0IGZ1bmN0aW9uIExPR0lTVElDKHg6IG51bWJlciwgZGVyaXZhdGU/OiBib29sZWFuKSB7XG5cdGlmICghZGVyaXZhdGUpXG5cdFx0cmV0dXJuIDEgLyAoMSArIE1hdGguZXhwKC14KSk7XG5cdHZhciBmeCA9IExPR0lTVElDKHgpO1xuXHRyZXR1cm4gZnggKiAoMSAtIGZ4KTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIFRBTkgoeDogbnVtYmVyLCBkZXJpdmF0ZT86IGJvb2xlYW4pIHtcblx0aWYgKGRlcml2YXRlKVxuXHRcdHJldHVybiAxIC0gTWF0aC5wb3coVEFOSCh4KSwgMik7XG5cdHZhciBlUCA9IE1hdGguZXhwKHgpO1xuXHR2YXIgZU4gPSAxIC8gZVA7XG5cdHJldHVybiAoZVAgLSBlTikgLyAoZVAgKyBlTik7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBJREVOVElUWSh4OiBudW1iZXIsIGRlcml2YXRlPzogYm9vbGVhbikge1xuXHRyZXR1cm4gZGVyaXZhdGUgPyAxIDogeDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIEhMSU0oeDogbnVtYmVyLCBkZXJpdmF0ZT86IGJvb2xlYW4pIHtcblx0cmV0dXJuIGRlcml2YXRlID8gMSA6ICsoeCA+IDApO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gU09GVFBMVVMoeDogbnVtYmVyLCBkZXJpdmF0ZT86IGJvb2xlYW4pIHtcblx0aWYgKGRlcml2YXRlKVxuXHRcdHJldHVybiAxIC0gMSAvIDEgKyBNYXRoLmV4cCh4KTtcblx0cmV0dXJuIE1hdGgubG9nKDEgKyBNYXRoLmV4cCh4KSk7XG59IiwiLypcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFNZTkFQVElDXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxuXG5TeW5hcHRpYyBpcyBhIGphdmFzY3JpcHQgbmV1cmFsIG5ldHdvcmsgbGlicmFyeSBmb3Igbm9kZS5qcyBhbmQgdGhlIGJyb3dzZXIsIGl0cyBnZW5lcmFsaXplZFxuYWxnb3JpdGhtIGlzIGFyY2hpdGVjdHVyZS1mcmVlLCBzbyB5b3UgY2FuIGJ1aWxkIGFuZCB0cmFpbiBiYXNpY2FsbHkgYW55IHR5cGUgb2YgZmlyc3Qgb3JkZXJcbm9yIGV2ZW4gc2Vjb25kIG9yZGVyIG5ldXJhbCBuZXR3b3JrIGFyY2hpdGVjdHVyZXMuXG5cbmh0dHA6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvUmVjdXJyZW50X25ldXJhbF9uZXR3b3JrI1NlY29uZF9PcmRlcl9SZWN1cnJlbnRfTmV1cmFsX05ldHdvcmtcblxuVGhlIGxpYnJhcnkgaW5jbHVkZXMgYSBmZXcgYnVpbHQtaW4gYXJjaGl0ZWN0dXJlcyBsaWtlIG11bHRpbGF5ZXIgcGVyY2VwdHJvbnMsIG11bHRpbGF5ZXJcbmxvbmctc2hvcnQgdGVybSBtZW1vcnkgbmV0d29ya3MgKExTVE0pIG9yIGxpcXVpZCBzdGF0ZSBtYWNoaW5lcywgYW5kIGEgdHJhaW5lciBjYXBhYmxlIG9mXG50cmFpbmluZyBhbnkgZ2l2ZW4gbmV0d29yaywgYW5kIGluY2x1ZGVzIGJ1aWx0LWluIHRyYWluaW5nIHRhc2tzL3Rlc3RzIGxpa2Ugc29sdmluZyBhbiBYT1IsXG5wYXNzaW5nIGEgRGlzdHJhY3RlZCBTZXF1ZW5jZSBSZWNhbGwgdGVzdCBvciBhbiBFbWJlZGVkIFJlYmVyIEdyYW1tYXIgdGVzdC5cblxuVGhlIGFsZ29yaXRobSBpbXBsZW1lbnRlZCBieSB0aGlzIGxpYnJhcnkgaGFzIGJlZW4gdGFrZW4gZnJvbSBEZXJlayBELiBNb25uZXIncyBwYXBlcjpcblxuQSBnZW5lcmFsaXplZCBMU1RNLWxpa2UgdHJhaW5pbmcgYWxnb3JpdGhtIGZvciBzZWNvbmQtb3JkZXIgcmVjdXJyZW50IG5ldXJhbCBuZXR3b3Jrc1xuaHR0cDovL3d3dy5vdmVyY29tcGxldGUubmV0L3BhcGVycy9ubjIwMTIucGRmXG5cblRoZXJlIGFyZSByZWZlcmVuY2VzIHRvIHRoZSBlcXVhdGlvbnMgaW4gdGhhdCBwYXBlciBjb21tZW50ZWQgdGhyb3VnaCB0aGUgc291cmNlIGNvZGUuXG5cblxuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cblxuXG5pbXBvcnQgbmV0d29yayA9IHJlcXVpcmUoJy4vbmV0d29yaycpO1xuaW1wb3J0IGxheWVyID0gcmVxdWlyZSgnLi9sYXllcicpO1xuaW1wb3J0IG5ldXJvbiA9IHJlcXVpcmUoJy4vbmV1cm9uJyk7XG5pbXBvcnQgdHJhaW5lciA9IHJlcXVpcmUoJy4vdHJhaW5lcicpO1xuaW1wb3J0IGFyY2hpdGVjdCA9IHJlcXVpcmUoJy4vYXJjaGl0ZWN0Jyk7XG5pbXBvcnQgc3F1YXNoID0gcmVxdWlyZSgnLi9zcXVhc2gnKTtcblxuZGVjbGFyZSB2YXIgd2luZG93O1xuXG5tb2R1bGUgU3luYXB0aWMge1xuXHRleHBvcnQgaW50ZXJmYWNlIERpY3Rpb25hcnk8VD4ge1xuXHRcdFtpZDogc3RyaW5nXSA6IFQ7XG5cdH1cblx0XG5cdHZhciBvbGRTeW5hcHRpYyA9IHR5cGVvZiB3aW5kb3cgIT0gXCJ1bmRlZmluZWRcIiAmJiB3aW5kb3cgJiYgd2luZG93WydTeW5hcHRpYyddO1xuXHRcblx0ZXhwb3J0IGZ1bmN0aW9uIG5pbmphKCkge1xuICAgICAgd2luZG93WydzeW5hcHRpYyddID0gb2xkU3luYXB0aWM7IFxuICAgICAgcmV0dXJuIFN5bmFwdGljO1xuXHR9XG5cdFxuXHRleHBvcnQgaW50ZXJmYWNlIElDb21waWxlZFBhcmFtZXRlcnMge1x0XG5cdFx0bWVtb3J5PzogYW55O1xuXHRcdG5ldXJvbnM/OiBudW1iZXI7XG5cdFx0aW5wdXRzPzogYW55W107XG5cdFx0b3V0cHV0cz86IGFueVtdO1xuXHRcdHRhcmdldHM/OiBhbnlbXTtcblx0XHR2YXJpYWJsZXM/OiBhbnk7XG5cdFx0YWN0aXZhdGlvbl9zZW50ZW5jZXM/OiBhbnlbXTtcblx0XHR0cmFjZV9zZW50ZW5jZXM/OiBhbnlbXTtcblx0XHRwcm9wYWdhdGlvbl9zZW50ZW5jZXM/OiBhbnlbXTtcblx0XHRsYXllcnM/OiBhbnk7XG5cdH1cblx0XG5cdGV4cG9ydCBpbnRlcmZhY2UgSU51bWVyaWNBcnJheSB7XG5cdCAgW2luZGV4OiBudW1iZXJdIDogbnVtYmVyO1xuXHQgIGxlbmd0aCA6IG51bWJlcjtcblx0fVxuXHRcblx0ZXhwb3J0IHZhciBOZXVyb24gPSBuZXVyb24uTmV1cm9uO1xuXHRleHBvcnQgdmFyIExheWVyID0gbGF5ZXIuTGF5ZXI7XG5cdGV4cG9ydCB2YXIgTmV0d29yayA9IG5ldHdvcmsuTmV0d29yaztcblx0ZXhwb3J0IHZhciBUcmFpbmVyID0gdHJhaW5lci5UcmFpbmVyO1xuXHRleHBvcnQgdmFyIFNxdWFzaCA9IHNxdWFzaDtcblx0ZXhwb3J0IHZhciBBcmNoaXRlY3QgPSBhcmNoaXRlY3Q7XG59XG5cbmV4cG9ydCA9IFN5bmFwdGljO1xuXG5pZih0eXBlb2Ygd2luZG93ICE9IFwidW5kZWZpbmVkXCIpIFxuXHR3aW5kb3dbJ3N5bmFwdGljJ10gPSBTeW5hcHRpYztcbiIsImltcG9ydCBuZXQgPSByZXF1aXJlKCcuL25ldHdvcmsnKTtcblxuLyoqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBUUkFJTkVSXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuXG5leHBvcnQgY2xhc3MgVHJhaW5lciB7XG4gIG5ldHdvcms6IG5ldC5OZXR3b3JrO1xuICByYXRlOiBhbnkgPSAuMjtcbiAgaXRlcmF0aW9ucyA9IDEwMDAwMDtcbiAgZXJyb3IgPSAuMDA1O1xuICBjb3N0OiBUcmFpbmVyLklUcmFpbmVyQ29zdEZuO1xuICBzY2hlZHVsZTogYW55O1xuXG4gIGNvbnN0cnVjdG9yKG5ldHdvcms6IG5ldC5OZXR3b3JrLCBvcHRpb25zPzogYW55KSB7XG4gICAgb3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG4gICAgdGhpcy5uZXR3b3JrID0gbmV0d29yaztcbiAgICB0aGlzLnJhdGUgPSBvcHRpb25zLnJhdGUgfHwgLjI7XG4gICAgdGhpcy5pdGVyYXRpb25zID0gb3B0aW9ucy5pdGVyYXRpb25zIHx8IDEwMDAwMDtcbiAgICB0aGlzLmVycm9yID0gb3B0aW9ucy5lcnJvciB8fCAuMDA1XG4gICAgdGhpcy5jb3N0ID0gb3B0aW9ucy5jb3N0IHx8IFRyYWluZXIuY29zdC5DUk9TU19FTlRST1BZO1xuICB9XG5cbiAgLy8gdHJhaW5zIGFueSBnaXZlbiBzZXQgdG8gYSBuZXR3b3JrXG4gIHRyYWluKHNldCwgb3B0aW9ucykge1xuXG4gICAgdmFyIGVycm9yID0gMTtcbiAgICB2YXIgaXRlcmF0aW9ucyA9IDAsIGJ1Y2tldFNpemUgPSAwO1xuICAgIHZhciBhYm9ydF90cmFpbmluZyA9IGZhbHNlO1xuICAgIHZhciBpbnB1dCwgb3V0cHV0LCB0YXJnZXQsIGN1cnJlbnRSYXRlO1xuXG4gICAgdmFyIHN0YXJ0ID0gRGF0ZS5ub3coKTtcblxuICAgIGlmIChvcHRpb25zKSB7XG4gICAgICBpZiAob3B0aW9ucy5zaHVmZmxlKSB7XG4gICAgICAgIC8vKyBKb25hcyBSYW9uaSBTb2FyZXMgU2lsdmFcbiAgICAgICAgLy9AIGh0dHA6Ly9qc2Zyb21oZWxsLmNvbS9hcnJheS9zaHVmZmxlIFt2MS4wXVxuICAgICAgICBmdW5jdGlvbiBzaHVmZmxlKG8pIHsgLy92MS4wXG4gICAgICAgICAgZm9yICh2YXIgaiwgeCwgaSA9IG8ubGVuZ3RoOyBpOyBqID0gTWF0aC5mbG9vcihNYXRoLnJhbmRvbSgpICogaSksIHggPSBvWy0taV0sIG9baV0gPSBvW2pdLCBvW2pdID0geCk7XG4gICAgICAgICAgcmV0dXJuIG87XG4gICAgICAgIH07XG4gICAgICB9XG4gICAgICBpZiAob3B0aW9ucy5pdGVyYXRpb25zKVxuICAgICAgICB0aGlzLml0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnM7XG4gICAgICBpZiAob3B0aW9ucy5lcnJvcilcbiAgICAgICAgdGhpcy5lcnJvciA9IG9wdGlvbnMuZXJyb3I7XG4gICAgICBpZiAob3B0aW9ucy5yYXRlKVxuICAgICAgICB0aGlzLnJhdGUgPSBvcHRpb25zLnJhdGU7XG4gICAgICBpZiAob3B0aW9ucy5jb3N0KVxuICAgICAgICB0aGlzLmNvc3QgPSBvcHRpb25zLmNvc3Q7XG4gICAgICBpZiAob3B0aW9ucy5zY2hlZHVsZSlcbiAgICAgICAgdGhpcy5zY2hlZHVsZSA9IG9wdGlvbnMuc2NoZWR1bGU7XG4gICAgICBpZiAob3B0aW9ucy5jdXN0b21Mb2cpIHtcbiAgICAgICAgLy8gZm9yIGJhY2t3YXJkIGNvbXBhdGliaWxpdHkgd2l0aCBjb2RlIHRoYXQgdXNlZCBjdXN0b21Mb2dcbiAgICAgICAgY29uc29sZS5sb2coJ0RlcHJlY2F0ZWQ6IHVzZSBzY2hlZHVsZSBpbnN0ZWFkIG9mIGN1c3RvbUxvZycpXG4gICAgICAgIHRoaXMuc2NoZWR1bGUgPSBvcHRpb25zLmN1c3RvbUxvZztcbiAgICAgIH1cbiAgICB9XG5cbiAgICBjdXJyZW50UmF0ZSA9IHRoaXMucmF0ZTtcbiAgICBpZiAoQXJyYXkuaXNBcnJheSh0aGlzLnJhdGUpKSB7XG4gICAgICBidWNrZXRTaXplID0gTWF0aC5mbG9vcih0aGlzLml0ZXJhdGlvbnMgLyB0aGlzLnJhdGUubGVuZ3RoKTtcbiAgICB9XG5cblxuICAgIHdoaWxlICghYWJvcnRfdHJhaW5pbmcgJiYgaXRlcmF0aW9ucyA8IHRoaXMuaXRlcmF0aW9ucyAmJiBlcnJvciA+IHRoaXMuZXJyb3IpIHtcbiAgICAgIGVycm9yID0gMDtcblxuICAgICAgaWYgKGJ1Y2tldFNpemUgPiAwKSB7XG4gICAgICAgIHZhciBjdXJyZW50QnVja2V0ID0gTWF0aC5mbG9vcihpdGVyYXRpb25zIC8gYnVja2V0U2l6ZSk7XG4gICAgICAgIGN1cnJlbnRSYXRlID0gdGhpcy5yYXRlW2N1cnJlbnRCdWNrZXRdO1xuICAgICAgfVxuXG4gICAgICBmb3IgKHZhciB0cmFpbiBpbiBzZXQpIHtcbiAgICAgICAgaW5wdXQgPSBzZXRbdHJhaW5dLmlucHV0O1xuICAgICAgICB0YXJnZXQgPSBzZXRbdHJhaW5dLm91dHB1dDtcblxuICAgICAgICBvdXRwdXQgPSB0aGlzLm5ldHdvcmsuYWN0aXZhdGUoaW5wdXQpO1xuICAgICAgICB0aGlzLm5ldHdvcmsucHJvcGFnYXRlKGN1cnJlbnRSYXRlLCB0YXJnZXQpO1xuXG4gICAgICAgIGVycm9yICs9IHRoaXMuY29zdCh0YXJnZXQsIG91dHB1dCk7XG4gICAgICB9XG5cbiAgICAgIC8vIGNoZWNrIGVycm9yXG4gICAgICBpdGVyYXRpb25zKys7XG4gICAgICBlcnJvciAvPSBzZXQubGVuZ3RoO1xuXG4gICAgICBpZiAob3B0aW9ucykge1xuICAgICAgICBpZiAodGhpcy5zY2hlZHVsZSAmJiB0aGlzLnNjaGVkdWxlLmV2ZXJ5ICYmIGl0ZXJhdGlvbnMgJSB0aGlzLnNjaGVkdWxlLmV2ZXJ5ID09IDApIHtcblxuICAgICAgICAgIGFib3J0X3RyYWluaW5nID0gdGhpcy5zY2hlZHVsZS5kbyh7XG4gICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgICAgICAgcmF0ZTogY3VycmVudFJhdGVcbiAgICAgICAgICB9KTtcbiAgICAgXG4gICAgICAgIH0gZWxzZSBpZiAob3B0aW9ucy5sb2cgJiYgaXRlcmF0aW9ucyAlIG9wdGlvbnMubG9nID09IDApIHtcbiAgICAgICAgICBjb25zb2xlLmxvZygnaXRlcmF0aW9ucycsIGl0ZXJhdGlvbnMsICdlcnJvcicsIGVycm9yLCAncmF0ZScsIGN1cnJlbnRSYXRlKTtcbiAgICAgICAgfTtcbiAgICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSlcbiAgICAgICAgICBzaHVmZmxlKHNldCk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdmFyIHJlc3VsdHMgPSB7XG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0XG4gICAgfVxuXG4gICAgcmV0dXJuIHJlc3VsdHM7XG4gIH1cblxuICAvLyB0cmFpbnMgYW55IGdpdmVuIHNldCB0byBhIG5ldHdvcmsgdXNpbmcgYSBXZWJXb3JrZXJcbiAgd29ya2VyVHJhaW4oc2V0LCBjYWxsYmFjaywgb3B0aW9ucykge1xuXG4gICAgdmFyIHRoYXQgPSB0aGlzO1xuICAgIHZhciBlcnJvciA9IDE7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSAwLCBidWNrZXRTaXplID0gMDtcbiAgICB2YXIgaW5wdXQsIG91dHB1dCwgdGFyZ2V0LCBjdXJyZW50UmF0ZTtcbiAgICB2YXIgbGVuZ3RoID0gc2V0Lmxlbmd0aDtcbiAgICB2YXIgYWJvcnRfdHJhaW5pbmcgPSBmYWxzZTtcblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICBpZiAob3B0aW9ucykge1xuICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSkge1xuICAgICAgICAvLysgSm9uYXMgUmFvbmkgU29hcmVzIFNpbHZhXG4gICAgICAgIC8vQCBodHRwOi8vanNmcm9taGVsbC5jb20vYXJyYXkvc2h1ZmZsZSBbdjEuMF1cbiAgICAgICAgZnVuY3Rpb24gc2h1ZmZsZShvKSB7IC8vdjEuMFxuICAgICAgICAgIGZvciAodmFyIGosIHgsIGkgPSBvLmxlbmd0aDsgaTsgaiA9IE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqXG4gICAgICAgICAgICBpKSwgeCA9IG9bLS1pXSwgb1tpXSA9IG9bal0sIG9bal0gPSB4KTtcbiAgICAgICAgICByZXR1cm4gbztcbiAgICAgICAgfTtcbiAgICAgIH1cbiAgICAgIGlmIChvcHRpb25zLml0ZXJhdGlvbnMpXG4gICAgICAgIHRoaXMuaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucztcbiAgICAgIGlmIChvcHRpb25zLmVycm9yKVxuICAgICAgICB0aGlzLmVycm9yID0gb3B0aW9ucy5lcnJvcjtcbiAgICAgIGlmIChvcHRpb25zLnJhdGUpXG4gICAgICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZTtcbiAgICAgIGlmIChvcHRpb25zLmNvc3QpXG4gICAgICAgIHRoaXMuY29zdCA9IG9wdGlvbnMuY29zdDtcbiAgICAgIGlmIChvcHRpb25zLnNjaGVkdWxlKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5zY2hlZHVsZTtcbiAgICAgIGlmIChvcHRpb25zLmN1c3RvbUxvZylcbiAgICAgICAgLy8gZm9yIGJhY2t3YXJkIGNvbXBhdGliaWxpdHkgd2l0aCBjb2RlIHRoYXQgdXNlZCBjdXN0b21Mb2dcbiAgICAgICAgY29uc29sZS5sb2coJ0RlcHJlY2F0ZWQ6IHVzZSBzY2hlZHVsZSBpbnN0ZWFkIG9mIGN1c3RvbUxvZycpXG4gICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5jdXN0b21Mb2c7XG4gICAgfVxuXG4gICAgLy8gZHluYW1pYyBsZWFybmluZyByYXRlXG4gICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGU7XG4gICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5yYXRlKSkge1xuICAgICAgYnVja2V0U2l6ZSA9IE1hdGguZmxvb3IodGhpcy5pdGVyYXRpb25zIC8gdGhpcy5yYXRlLmxlbmd0aCk7XG4gICAgfVxuXG4gICAgLy8gY3JlYXRlIGEgd29ya2VyXG4gICAgdmFyIHdvcmtlciA9IHRoaXMubmV0d29yay53b3JrZXIoKTtcblxuICAgIC8vIGFjdGl2YXRlIHRoZSBuZXR3b3JrXG4gICAgZnVuY3Rpb24gYWN0aXZhdGVXb3JrZXIoaW5wdXQpIHtcbiAgICAgIHdvcmtlci5wb3N0TWVzc2FnZSh7XG4gICAgICAgIGFjdGlvbjogXCJhY3RpdmF0ZVwiLFxuICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgIG1lbW9yeUJ1ZmZlcjogdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnlcbiAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyBiYWNrcHJvcGFnYXRlIHRoZSBuZXR3b3JrXG4gICAgZnVuY3Rpb24gcHJvcGFnYXRlV29ya2VyKHRhcmdldCkge1xuICAgICAgaWYgKGJ1Y2tldFNpemUgPiAwKSB7XG4gICAgICAgIHZhciBjdXJyZW50QnVja2V0ID0gTWF0aC5mbG9vcihpdGVyYXRpb25zIC8gYnVja2V0U2l6ZSk7XG4gICAgICAgIGN1cnJlbnRSYXRlID0gdGhpcy5yYXRlW2N1cnJlbnRCdWNrZXRdO1xuICAgICAgfVxuICAgICAgd29ya2VyLnBvc3RNZXNzYWdlKHtcbiAgICAgICAgYWN0aW9uOiBcInByb3BhZ2F0ZVwiLFxuICAgICAgICB0YXJnZXQ6IHRhcmdldCxcbiAgICAgICAgcmF0ZTogY3VycmVudFJhdGUsXG4gICAgICAgIG1lbW9yeUJ1ZmZlcjogdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnlcbiAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyB0cmFpbiB0aGUgd29ya2VyXG4gICAgd29ya2VyLm9ubWVzc2FnZSA9IGZ1bmN0aW9uKGUpIHtcbiAgICAgIC8vIGdpdmUgY29udHJvbCBvZiB0aGUgbWVtb3J5IGJhY2sgdG8gdGhlIG5ldHdvcmtcbiAgICAgIHRoYXQubmV0d29yay5vcHRpbWl6ZWQub3duZXJzaGlwKGUuZGF0YS5tZW1vcnlCdWZmZXIpO1xuXG4gICAgICBpZiAoZS5kYXRhLmFjdGlvbiA9PSBcInByb3BhZ2F0ZVwiKSB7XG4gICAgICAgIGlmIChpbmRleCA+PSBsZW5ndGgpIHtcbiAgICAgICAgICBpbmRleCA9IDA7XG4gICAgICAgICAgaXRlcmF0aW9ucysrO1xuICAgICAgICAgIGVycm9yIC89IHNldC5sZW5ndGg7XG5cbiAgICAgICAgICAvLyBsb2dcbiAgICAgICAgICBpZiAob3B0aW9ucykge1xuICAgICAgICAgICAgaWYgKHRoaXMuc2NoZWR1bGUgJiYgdGhpcy5zY2hlZHVsZS5ldmVyeSAmJiBpdGVyYXRpb25zICUgdGhpcy5zY2hlZHVsZS5ldmVyeSA9PSAwKVxuICAgICAgICAgICAgICBhYm9ydF90cmFpbmluZyA9IHRoaXMuc2NoZWR1bGUuZG8oe1xuICAgICAgICAgICAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zXG4gICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgZWxzZSBpZiAob3B0aW9ucy5sb2cgJiYgaXRlcmF0aW9ucyAlIG9wdGlvbnMubG9nID09IDApIHtcbiAgICAgICAgICAgICAgY29uc29sZS5sb2coJ2l0ZXJhdGlvbnMnLCBpdGVyYXRpb25zLCAnZXJyb3InLCBlcnJvcik7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSlcbiAgICAgICAgICAgICAgc2h1ZmZsZShzZXQpO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIGlmICghYWJvcnRfdHJhaW5pbmcgJiYgaXRlcmF0aW9ucyA8IHRoYXQuaXRlcmF0aW9ucyAmJiBlcnJvciA+IHRoYXQuZXJyb3IpIHtcbiAgICAgICAgICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAvLyBjYWxsYmFja1xuICAgICAgICAgICAgY2FsbGJhY2soe1xuICAgICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICAgIGl0ZXJhdGlvbnM6IGl0ZXJhdGlvbnMsXG4gICAgICAgICAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydFxuICAgICAgICAgICAgfSlcbiAgICAgICAgICB9XG4gICAgICAgICAgZXJyb3IgPSAwO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChlLmRhdGEuYWN0aW9uID09IFwiYWN0aXZhdGVcIikge1xuICAgICAgICBlcnJvciArPSB0aGF0LmNvc3Qoc2V0W2luZGV4XS5vdXRwdXQsIGUuZGF0YS5vdXRwdXQpO1xuICAgICAgICBwcm9wYWdhdGVXb3JrZXIoc2V0W2luZGV4XS5vdXRwdXQpO1xuICAgICAgICBpbmRleCsrO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIGtpY2sgaXRcbiAgICB2YXIgaW5kZXggPSAwO1xuICAgIHZhciBpdGVyYXRpb25zID0gMDtcbiAgICBhY3RpdmF0ZVdvcmtlcihzZXRbaW5kZXhdLmlucHV0KTtcbiAgfVxuXG4gIC8vIHRyYWlucyBhbiBYT1IgdG8gdGhlIG5ldHdvcmtcbiAgWE9SKG9wdGlvbnMpIHtcblxuICAgIGlmICh0aGlzLm5ldHdvcmsuaW5wdXRzKCkgIT0gMiB8fCB0aGlzLm5ldHdvcmsub3V0cHV0cygpICE9IDEpXG4gICAgICB0aHJvdyBcIkVycm9yOiBJbmNvbXBhdGlibGUgbmV0d29yayAoMiBpbnB1dHMsIDEgb3V0cHV0KVwiO1xuXG4gICAgdmFyIGRlZmF1bHRzID0ge1xuICAgICAgaXRlcmF0aW9uczogMTAwMDAwLFxuICAgICAgbG9nOiBmYWxzZSxcbiAgICAgIHNodWZmbGU6IHRydWUsXG4gICAgICBjb3N0OiBUcmFpbmVyLmNvc3QuTVNFXG4gICAgfVxuXG4gICAgaWYgKG9wdGlvbnMpXG4gICAgICBmb3IgKHZhciBpIGluIG9wdGlvbnMpXG4gICAgICAgIGRlZmF1bHRzW2ldID0gb3B0aW9uc1tpXTtcblxuICAgIHJldHVybiB0aGlzLnRyYWluKFt7XG4gICAgICBpbnB1dDogWzAsIDBdLFxuICAgICAgb3V0cHV0OiBbMF1cbiAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMSwgMF0sXG4gICAgICAgIG91dHB1dDogWzFdXG4gICAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMCwgMV0sXG4gICAgICAgIG91dHB1dDogWzFdXG4gICAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMSwgMV0sXG4gICAgICAgIG91dHB1dDogWzBdXG4gICAgICB9XSwgZGVmYXVsdHMpO1xuICB9XG5cbiAgLy8gdHJhaW5zIHRoZSBuZXR3b3JrIHRvIHBhc3MgYSBEaXN0cmFjdGVkIFNlcXVlbmNlIFJlY2FsbCB0ZXN0XG4gIERTUihvcHRpb25zKSB7XG4gICAgb3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG5cbiAgICB2YXIgdGFyZ2V0cyA9IG9wdGlvbnMudGFyZ2V0cyB8fCBbMiwgNCwgNywgOF07XG4gICAgdmFyIGRpc3RyYWN0b3JzID0gb3B0aW9ucy5kaXN0cmFjdG9ycyB8fCBbMywgNSwgNiwgOV07XG4gICAgdmFyIHByb21wdHMgPSBvcHRpb25zLnByb21wdHMgfHwgWzAsIDFdO1xuICAgIHZhciBsZW5ndGggPSBvcHRpb25zLmxlbmd0aCB8fCAyNDtcbiAgICB2YXIgY3JpdGVyaW9uID0gb3B0aW9ucy5zdWNjZXNzIHx8IDAuOTU7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnMgfHwgMTAwMDAwO1xuICAgIHZhciByYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4xO1xuICAgIHZhciBsb2cgPSBvcHRpb25zLmxvZyB8fCAwO1xuICAgIHZhciBzY2hlZHVsZSA9IG9wdGlvbnMuc2NoZWR1bGUgfHwge307XG4gICAgdmFyIGNvcnJlY3QgPSAwO1xuICAgIHZhciBpID0gMDtcbiAgICB2YXIgc3VjY2VzcyA9IDA7XG4gICAgdmFyIHRyaWFsID0gaSA9IGNvcnJlY3QgPSBqID0gc3VjY2VzcyA9IDAsXG4gICAgICBlcnJvciA9IDEsXG4gICAgICBzeW1ib2xzID0gdGFyZ2V0cy5sZW5ndGggKyBkaXN0cmFjdG9ycy5sZW5ndGggKyBwcm9tcHRzLmxlbmd0aDtcblxuICAgIHZhciBub1JlcGVhdCA9IGZ1bmN0aW9uKHJhbmdlLCBhdm9pZCkge1xuICAgICAgdmFyIG51bWJlciA9IE1hdGgucmFuZG9tKCkgKiByYW5nZSB8IDA7XG4gICAgICB2YXIgdXNlZCA9IGZhbHNlO1xuICAgICAgZm9yICh2YXIgaSBpbiBhdm9pZClcbiAgICAgICAgaWYgKG51bWJlciA9PSBhdm9pZFtpXSlcbiAgICAgICAgICB1c2VkID0gdHJ1ZTtcbiAgICAgIHJldHVybiB1c2VkID8gbm9SZXBlYXQocmFuZ2UsIGF2b2lkKSA6IG51bWJlcjtcbiAgICB9XG5cbiAgICB2YXIgZXF1YWwgPSBmdW5jdGlvbihwcmVkaWN0aW9uLCBvdXRwdXQpIHtcbiAgICAgIGZvciAodmFyIGkgaW4gcHJlZGljdGlvbilcbiAgICAgICAgaWYgKE1hdGgucm91bmQocHJlZGljdGlvbltpXSkgIT0gb3V0cHV0W2ldKVxuICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICB3aGlsZSAodHJpYWwgPCBpdGVyYXRpb25zICYmIChzdWNjZXNzIDwgY3JpdGVyaW9uIHx8IHRyaWFsICUgMTAwMCAhPSAwKSkge1xuICAgICAgLy8gZ2VuZXJhdGUgc2VxdWVuY2VcbiAgICAgIHZhciBzZXF1ZW5jZSA9IFtdLFxuICAgICAgICBzZXF1ZW5jZUxlbmd0aCA9IGxlbmd0aCAtIHByb21wdHMubGVuZ3RoO1xuICAgICAgZm9yIChpID0gMDsgaSA8IHNlcXVlbmNlTGVuZ3RoOyBpKyspIHtcbiAgICAgICAgdmFyIGFueSA9IE1hdGgucmFuZG9tKCkgKiBkaXN0cmFjdG9ycy5sZW5ndGggfCAwO1xuICAgICAgICBzZXF1ZW5jZS5wdXNoKGRpc3RyYWN0b3JzW2FueV0pO1xuICAgICAgfVxuICAgICAgdmFyIGluZGV4ZXMgPSBbXSxcbiAgICAgICAgcG9zaXRpb25zID0gW107XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgcHJvbXB0cy5sZW5ndGg7IGkrKykge1xuICAgICAgICBpbmRleGVzLnB1c2goTWF0aC5yYW5kb20oKSAqIHRhcmdldHMubGVuZ3RoIHwgMCk7XG4gICAgICAgIHBvc2l0aW9ucy5wdXNoKG5vUmVwZWF0KHNlcXVlbmNlTGVuZ3RoLCBwb3NpdGlvbnMpKTtcbiAgICAgIH1cbiAgICAgIHBvc2l0aW9ucyA9IHBvc2l0aW9ucy5zb3J0KCk7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgcHJvbXB0cy5sZW5ndGg7IGkrKykge1xuICAgICAgICBzZXF1ZW5jZVtwb3NpdGlvbnNbaV1dID0gdGFyZ2V0c1tpbmRleGVzW2ldXTtcbiAgICAgICAgc2VxdWVuY2UucHVzaChwcm9tcHRzW2ldKTtcbiAgICAgIH1cblxuICAgICAgLy90cmFpbiBzZXF1ZW5jZVxuICAgICAgdmFyIGRpc3RyYWN0b3JzQ29ycmVjdDtcbiAgICAgIHZhciB0YXJnZXRzQ29ycmVjdCA9IGRpc3RyYWN0b3JzQ29ycmVjdCA9IDA7XG4gICAgICBlcnJvciA9IDA7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgbGVuZ3RoOyBpKyspIHtcbiAgICAgICAgLy8gZ2VuZXJhdGUgaW5wdXQgZnJvbSBzZXF1ZW5jZVxuICAgICAgICB2YXIgaW5wdXQgPSBbXTtcbiAgICAgICAgZm9yIChqID0gMDsgaiA8IHN5bWJvbHM7IGorKylcbiAgICAgICAgICBpbnB1dFtqXSA9IDA7XG4gICAgICAgIGlucHV0W3NlcXVlbmNlW2ldXSA9IDE7XG5cbiAgICAgICAgLy8gZ2VuZXJhdGUgdGFyZ2V0IG91dHB1dFxuICAgICAgICB2YXIgb3V0cHV0ID0gW107XG4gICAgICAgIGZvciAoaiA9IDA7IGogPCB0YXJnZXRzLmxlbmd0aDsgaisrKVxuICAgICAgICAgIG91dHB1dFtqXSA9IDA7XG5cbiAgICAgICAgaWYgKGkgPj0gc2VxdWVuY2VMZW5ndGgpIHtcbiAgICAgICAgICB2YXIgaW5kZXggPSBpIC0gc2VxdWVuY2VMZW5ndGg7XG4gICAgICAgICAgb3V0cHV0W2luZGV4ZXNbaW5kZXhdXSA9IDE7XG4gICAgICAgIH1cblxuICAgICAgICAvLyBjaGVjayByZXN1bHRcbiAgICAgICAgdmFyIHByZWRpY3Rpb24gPSB0aGlzLm5ldHdvcmsuYWN0aXZhdGUoaW5wdXQpO1xuXG4gICAgICAgIGlmIChlcXVhbChwcmVkaWN0aW9uLCBvdXRwdXQpKVxuICAgICAgICAgIGlmIChpIDwgc2VxdWVuY2VMZW5ndGgpXG4gICAgICAgICAgICBkaXN0cmFjdG9yc0NvcnJlY3QrKztcbiAgICAgICAgICBlbHNlXG4gICAgICAgICAgICB0YXJnZXRzQ29ycmVjdCsrO1xuICAgICAgICBlbHNlIHtcbiAgICAgICAgICB0aGlzLm5ldHdvcmsucHJvcGFnYXRlKHJhdGUsIG91dHB1dCk7XG4gICAgICAgIH1cblxuICAgICAgICB2YXIgZGVsdGEgPSAwO1xuICAgICAgICBmb3IgKHZhciBqIGluIHByZWRpY3Rpb24pXG4gICAgICAgICAgZGVsdGEgKz0gTWF0aC5wb3cob3V0cHV0W2pdIC0gcHJlZGljdGlvbltqXSwgMik7XG4gICAgICAgIGVycm9yICs9IGRlbHRhIC8gdGhpcy5uZXR3b3JrLm91dHB1dHMoKTtcblxuICAgICAgICBpZiAoZGlzdHJhY3RvcnNDb3JyZWN0ICsgdGFyZ2V0c0NvcnJlY3QgPT0gbGVuZ3RoKVxuICAgICAgICAgIGNvcnJlY3QrKztcbiAgICAgIH1cblxuICAgICAgLy8gY2FsY3VsYXRlIGVycm9yXG4gICAgICBpZiAodHJpYWwgJSAxMDAwID09IDApXG4gICAgICAgIGNvcnJlY3QgPSAwO1xuICAgICAgdHJpYWwrKztcbiAgICAgIHZhciBkaXZpZGVFcnJvciA9IHRyaWFsICUgMTAwMDtcbiAgICAgIGRpdmlkZUVycm9yID0gZGl2aWRlRXJyb3IgPT0gMCA/IDEwMDAgOiBkaXZpZGVFcnJvcjtcbiAgICAgIHN1Y2Nlc3MgPSBjb3JyZWN0IC8gZGl2aWRlRXJyb3I7XG4gICAgICBlcnJvciAvPSBsZW5ndGg7XG5cbiAgICAgIC8vIGxvZ1xuICAgICAgaWYgKGxvZyAmJiB0cmlhbCAlIGxvZyA9PSAwKVxuICAgICAgICBjb25zb2xlLmxvZyhcIml0ZXJhdGlvbnM6XCIsIHRyaWFsLCBcIiBzdWNjZXNzOlwiLCBzdWNjZXNzLCBcIiBjb3JyZWN0OlwiLFxuICAgICAgICAgIGNvcnJlY3QsIFwiIHRpbWU6XCIsIERhdGUubm93KCkgLSBzdGFydCwgXCIgZXJyb3I6XCIsIGVycm9yKTtcbiAgICAgIGlmIChzY2hlZHVsZS5kbyAmJiBzY2hlZHVsZS5ldmVyeSAmJiB0cmlhbCAlIHNjaGVkdWxlLmV2ZXJ5ID09IDApIHtcbiAgICAgICAgc2NoZWR1bGUuZG8oe1xuICAgICAgICAgIGl0ZXJhdGlvbnM6IHRyaWFsLFxuICAgICAgICAgIHN1Y2Nlc3M6IHN1Y2Nlc3MsXG4gICAgICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydCxcbiAgICAgICAgICBjb3JyZWN0OiBjb3JyZWN0XG4gICAgICAgIH0pO1xuXG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgIGl0ZXJhdGlvbnM6IHRyaWFsLFxuICAgICAgc3VjY2Vzczogc3VjY2VzcyxcbiAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydFxuICAgIH1cbiAgfVxuXG4gIC8vIHRyYWluIHRoZSBuZXR3b3JrIHRvIGxlYXJuIGFuIEVtYmVkZWQgUmViZXIgR3JhbW1hclxuICBFUkcob3B0aW9ucykge1xuXG4gICAgb3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnMgfHwgMTUwMDAwO1xuICAgIHZhciBjcml0ZXJpb24gPSBvcHRpb25zLmVycm9yIHx8IC4wNTtcbiAgICB2YXIgcmF0ZSA9IG9wdGlvbnMucmF0ZSB8fCAuMTtcbiAgICB2YXIgbG9nID0gb3B0aW9ucy5sb2cgfHwgNTAwO1xuXG4gICAgLy8gZ3JhbWFyIG5vZGVcbiAgICB2YXIgTm9kZSA9IGZ1bmN0aW9uKCkge1xuICAgICAgdGhpcy5wYXRocyA9IFtdO1xuICAgIH1cbiAgICBOb2RlLnByb3RvdHlwZSA9IHtcbiAgICAgIGNvbm5lY3Q6IGZ1bmN0aW9uKG5vZGUsIHZhbHVlKSB7XG4gICAgICAgIHRoaXMucGF0aHMucHVzaCh7XG4gICAgICAgICAgbm9kZTogbm9kZSxcbiAgICAgICAgICB2YWx1ZTogdmFsdWVcbiAgICAgICAgfSk7XG4gICAgICAgIHJldHVybiB0aGlzO1xuICAgICAgfSxcbiAgICAgIGFueTogZnVuY3Rpb24oKSB7XG4gICAgICAgIGlmICh0aGlzLnBhdGhzLmxlbmd0aCA9PSAwKVxuICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgdmFyIGluZGV4ID0gTWF0aC5yYW5kb20oKSAqIHRoaXMucGF0aHMubGVuZ3RoIHwgMDtcbiAgICAgICAgcmV0dXJuIHRoaXMucGF0aHNbaW5kZXhdO1xuICAgICAgfSxcbiAgICAgIHRlc3Q6IGZ1bmN0aW9uKHZhbHVlKSB7XG4gICAgICAgIGZvciAodmFyIGkgaW4gdGhpcy5wYXRocylcbiAgICAgICAgICBpZiAodGhpcy5wYXRoc1tpXS52YWx1ZSA9PSB2YWx1ZSlcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnBhdGhzW2ldO1xuICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdmFyIHJlYmVyR3JhbW1hciA9IGZ1bmN0aW9uKCkge1xuXG4gICAgICAvLyBidWlsZCBhIHJlYmVyIGdyYW1tYXJcbiAgICAgIHZhciBvdXRwdXQgPSBuZXcgTm9kZSgpO1xuICAgICAgdmFyIG4xID0gKG5ldyBOb2RlKCkpLmNvbm5lY3Qob3V0cHV0LCBcIkVcIik7XG4gICAgICB2YXIgbjIgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuMSwgXCJTXCIpO1xuICAgICAgdmFyIG4zID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjEsIFwiVlwiKS5jb25uZWN0KG4yLCBcIlBcIik7XG4gICAgICB2YXIgbjQgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuMiwgXCJYXCIpXG4gICAgICBuNC5jb25uZWN0KG40LCBcIlNcIik7XG4gICAgICB2YXIgbjUgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuMywgXCJWXCIpXG4gICAgICBuNS5jb25uZWN0KG41LCBcIlRcIik7XG4gICAgICBuMi5jb25uZWN0KG41LCBcIlhcIilcbiAgICAgIHZhciBuNiA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG40LCBcIlRcIikuY29ubmVjdChuNSwgXCJQXCIpO1xuICAgICAgdmFyIGlucHV0ID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjYsIFwiQlwiKVxuXG4gICAgICByZXR1cm4ge1xuICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgIG91dHB1dDogb3V0cHV0XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8gYnVpbGQgYW4gZW1iZWRlZCByZWJlciBncmFtbWFyXG4gICAgdmFyIGVtYmVkZWRSZWJlckdyYW1tYXIgPSBmdW5jdGlvbigpIHtcbiAgICAgIHZhciByZWJlcjEgPSByZWJlckdyYW1tYXIoKTtcbiAgICAgIHZhciByZWJlcjIgPSByZWJlckdyYW1tYXIoKTtcblxuICAgICAgdmFyIG91dHB1dCA9IG5ldyBOb2RlKCk7XG4gICAgICB2YXIgbjEgPSAobmV3IE5vZGUpLmNvbm5lY3Qob3V0cHV0LCBcIkVcIik7XG4gICAgICByZWJlcjEub3V0cHV0LmNvbm5lY3QobjEsIFwiVFwiKTtcbiAgICAgIHJlYmVyMi5vdXRwdXQuY29ubmVjdChuMSwgXCJQXCIpO1xuICAgICAgdmFyIG4yID0gKG5ldyBOb2RlKS5jb25uZWN0KHJlYmVyMS5pbnB1dCwgXCJQXCIpLmNvbm5lY3QocmViZXIyLmlucHV0LFxuICAgICAgICBcIlRcIik7XG4gICAgICB2YXIgaW5wdXQgPSAobmV3IE5vZGUpLmNvbm5lY3QobjIsIFwiQlwiKTtcblxuICAgICAgcmV0dXJuIHtcbiAgICAgICAgaW5wdXQ6IGlucHV0LFxuICAgICAgICBvdXRwdXQ6IG91dHB1dFxuICAgICAgfVxuXG4gICAgfVxuXG4gICAgLy8gZ2VuZXJhdGUgYW4gRVJHIHNlcXVlbmNlXG4gICAgdmFyIGdlbmVyYXRlID0gZnVuY3Rpb24oKSB7XG4gICAgICB2YXIgbm9kZSA9IGVtYmVkZWRSZWJlckdyYW1tYXIoKS5pbnB1dDtcbiAgICAgIHZhciBuZXh0ID0gbm9kZS5hbnkoKTtcbiAgICAgIHZhciBzdHIgPSBcIlwiO1xuICAgICAgd2hpbGUgKG5leHQpIHtcbiAgICAgICAgc3RyICs9IG5leHQudmFsdWU7XG4gICAgICAgIG5leHQgPSBuZXh0Lm5vZGUuYW55KCk7XG4gICAgICB9XG4gICAgICByZXR1cm4gc3RyO1xuICAgIH1cblxuICAgIC8vIHRlc3QgaWYgYSBzdHJpbmcgbWF0Y2hlcyBhbiBlbWJlZGVkIHJlYmVyIGdyYW1tYXJcbiAgICB2YXIgdGVzdCA9IGZ1bmN0aW9uKHN0cikge1xuICAgICAgdmFyIG5vZGUgPSBlbWJlZGVkUmViZXJHcmFtbWFyKCkuaW5wdXQ7XG4gICAgICB2YXIgaSA9IDA7XG4gICAgICB2YXIgY2ggPSBzdHIuY2hhckF0KGkpO1xuICAgICAgd2hpbGUgKGkgPCBzdHIubGVuZ3RoKSB7XG4gICAgICAgIHZhciBuZXh0ID0gbm9kZS50ZXN0KGNoKTtcbiAgICAgICAgaWYgKCFuZXh0KVxuICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgbm9kZSA9IG5leHQubm9kZTtcbiAgICAgICAgY2ggPSBzdHIuY2hhckF0KCsraSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG5cbiAgICAvLyBoZWxwZXIgdG8gY2hlY2sgaWYgdGhlIG91dHB1dCBhbmQgdGhlIHRhcmdldCB2ZWN0b3JzIG1hdGNoXG4gICAgdmFyIGRpZmZlcmVudCA9IGZ1bmN0aW9uKGFycmF5MSwgYXJyYXkyKSB7XG4gICAgICB2YXIgbWF4MSA9IDA7XG4gICAgICB2YXIgaTEgPSAtMTtcbiAgICAgIHZhciBtYXgyID0gMDtcbiAgICAgIHZhciBpMiA9IC0xO1xuICAgICAgZm9yICh2YXIgaSBpbiBhcnJheTEpIHtcbiAgICAgICAgaWYgKGFycmF5MVtpXSA+IG1heDEpIHtcbiAgICAgICAgICBtYXgxID0gYXJyYXkxW2ldO1xuICAgICAgICAgIGkxID0gaTtcbiAgICAgICAgfVxuICAgICAgICBpZiAoYXJyYXkyW2ldID4gbWF4Mikge1xuICAgICAgICAgIG1heDIgPSBhcnJheTJbaV07XG4gICAgICAgICAgaTIgPSBpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIHJldHVybiBpMSAhPSBpMjtcbiAgICB9XG5cbiAgICB2YXIgaXRlcmF0aW9uID0gMDtcbiAgICB2YXIgZXJyb3IgPSAxO1xuICAgIHZhciB0YWJsZSA9IHtcbiAgICAgIFwiQlwiOiAwLFxuICAgICAgXCJQXCI6IDEsXG4gICAgICBcIlRcIjogMixcbiAgICAgIFwiWFwiOiAzLFxuICAgICAgXCJTXCI6IDQsXG4gICAgICBcIkVcIjogNVxuICAgIH1cblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG4gICAgd2hpbGUgKGl0ZXJhdGlvbiA8IGl0ZXJhdGlvbnMgJiYgZXJyb3IgPiBjcml0ZXJpb24pIHtcbiAgICAgIHZhciBpID0gMDtcbiAgICAgIGVycm9yID0gMDtcblxuICAgICAgLy8gRVJHIHNlcXVlbmNlIHRvIGxlYXJuXG4gICAgICB2YXIgc2VxdWVuY2UgPSBnZW5lcmF0ZSgpO1xuXG4gICAgICAvLyBpbnB1dFxuICAgICAgdmFyIHJlYWQgPSBzZXF1ZW5jZS5jaGFyQXQoaSk7XG4gICAgICAvLyB0YXJnZXRcbiAgICAgIHZhciBwcmVkaWN0ID0gc2VxdWVuY2UuY2hhckF0KGkgKyAxKTtcblxuICAgICAgLy8gdHJhaW5cbiAgICAgIHdoaWxlIChpIDwgc2VxdWVuY2UubGVuZ3RoIC0gMSkge1xuICAgICAgICB2YXIgaW5wdXQgPSBbXTtcbiAgICAgICAgdmFyIHRhcmdldCA9IFtdO1xuICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IDY7IGorKykge1xuICAgICAgICAgIGlucHV0W2pdID0gMDtcbiAgICAgICAgICB0YXJnZXRbal0gPSAwO1xuICAgICAgICB9XG4gICAgICAgIGlucHV0W3RhYmxlW3JlYWRdXSA9IDE7XG4gICAgICAgIHRhcmdldFt0YWJsZVtwcmVkaWN0XV0gPSAxO1xuXG4gICAgICAgIHZhciBvdXRwdXQgPSB0aGlzLm5ldHdvcmsuYWN0aXZhdGUoaW5wdXQpO1xuXG4gICAgICAgIGlmIChkaWZmZXJlbnQob3V0cHV0LCB0YXJnZXQpKVxuICAgICAgICAgIHRoaXMubmV0d29yay5wcm9wYWdhdGUocmF0ZSwgdGFyZ2V0KTtcblxuICAgICAgICByZWFkID0gc2VxdWVuY2UuY2hhckF0KCsraSk7XG4gICAgICAgIHByZWRpY3QgPSBzZXF1ZW5jZS5jaGFyQXQoaSArIDEpO1xuXG4gICAgICAgIHZhciBkZWx0YSA9IDA7XG4gICAgICAgIGZvciAodmFyIGsgaW4gb3V0cHV0KVxuICAgICAgICAgIGRlbHRhICs9IE1hdGgucG93KHRhcmdldFtrXSAtIG91dHB1dFtrXSwgMilcbiAgICAgICAgZGVsdGEgLz0gb3V0cHV0Lmxlbmd0aDtcblxuICAgICAgICBlcnJvciArPSBkZWx0YTtcbiAgICAgIH1cbiAgICAgIGVycm9yIC89IHNlcXVlbmNlLmxlbmd0aDtcbiAgICAgIGl0ZXJhdGlvbisrO1xuICAgICAgaWYgKGl0ZXJhdGlvbiAlIGxvZyA9PSAwKSB7XG4gICAgICAgIGNvbnNvbGUubG9nKFwiaXRlcmF0aW9uczpcIiwgaXRlcmF0aW9uLCBcIiB0aW1lOlwiLCBEYXRlLm5vdygpIC0gc3RhcnQsXG4gICAgICAgICAgXCIgZXJyb3I6XCIsIGVycm9yKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4ge1xuICAgICAgaXRlcmF0aW9uczogaXRlcmF0aW9uLFxuICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0LFxuICAgICAgdGVzdDogdGVzdCxcbiAgICAgIGdlbmVyYXRlOiBnZW5lcmF0ZVxuICAgIH1cbiAgfVxuXG59XG5cbmV4cG9ydCBtb2R1bGUgVHJhaW5lciB7XG4gIC8vIEJ1aWx0LWluIGNvc3QgZnVuY3Rpb25zXG4gIFxuICBleHBvcnQgaW50ZXJmYWNlIElUcmFpbmVyQ29zdEZuIHtcbiAgICAodGFyZ2V0LCBvdXRwdXQpOiBudW1iZXI7XG4gIH1cblxuICBleHBvcnQgdmFyIGNvc3QgPSB7XG4gICAgLy8gRXEuIDlcbiAgICBDUk9TU19FTlRST1BZOiBmdW5jdGlvbih0YXJnZXQsIG91dHB1dCkge1xuICAgICAgdmFyIGNyb3NzZW50cm9weSA9IDA7XG4gICAgICBmb3IgKHZhciBpIGluIG91dHB1dClcbiAgICAgICAgY3Jvc3NlbnRyb3B5IC09ICh0YXJnZXRbaV0gKiBNYXRoLmxvZyhvdXRwdXRbaV0gKyAxZS0xNSkpICsgKCgxIC0gdGFyZ2V0W2ldKSAqIE1hdGgubG9nKCgxICsgMWUtMTUpIC0gb3V0cHV0W2ldKSk7IC8vICsxZS0xNSBpcyBhIHRpbnkgcHVzaCBhd2F5IHRvIGF2b2lkIE1hdGgubG9nKDApXG4gICAgICByZXR1cm4gY3Jvc3NlbnRyb3B5O1xuICAgIH0sXG4gICAgTVNFOiBmdW5jdGlvbih0YXJnZXQsIG91dHB1dCkge1xuICAgICAgdmFyIG1zZSA9IDA7XG4gICAgICBmb3IgKHZhciBpIGluIG91dHB1dClcbiAgICAgICAgbXNlICs9IE1hdGgucG93KHRhcmdldFtpXSAtIG91dHB1dFtpXSwgMik7XG4gICAgICByZXR1cm4gbXNlIC8gb3V0cHV0Lmxlbmd0aDtcbiAgICB9XG4gIH1cbn0iXX0=
var synaptic = synaptic || Synaptic;var Neuron = synaptic.Neuron, Layer = synaptic.Layer, Network = synaptic.Network, Trainer = synaptic.Trainer, Architect = synaptic.Architect;