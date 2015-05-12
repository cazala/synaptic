/*

The MIT License (MIT)

Copyright (c) 2014 Juan Cazala - juancazala.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE



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
(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
/*

The MIT License (MIT)

Copyright (c) 2014 Juan Cazala - juancazala.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE



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

var Synaptic = {
    Neuron: require('./neuron'),
    Layer: require('./layer'),
    Network: require('./network'),
    Trainer: require('./trainer'),
    Architect: require('./architect')
};

// CommonJS & AMD
if (typeof define !== 'undefined' && define.amd)
{
  define([], function(){ return Synaptic });
}

// Node.js
if (typeof module !== 'undefined' && module.exports)
{
  module.exports = Synaptic;
}

// Browser
if (typeof window == 'object')
{
  (function(){ 
    var oldSynaptic = window['synaptic'];
    Synaptic.ninja = function(){ 
      window['synaptic'] = oldSynaptic; 
      return Synaptic;
    };	
  })();

  window['synaptic'] = Synaptic;
}

},{"./architect":2,"./layer":3,"./network":4,"./neuron":5,"./trainer":6}],2:[function(require,module,exports){
// import
var Layer = require('./layer'),
    Network = require('./network'),
    Trainer = require('./trainer');

/*******************************************************************************************
                                        ARCHITECT
*******************************************************************************************/

// Colection of useful built-in architectures
var Architect = {

  // Multilayer Perceptron
  Perceptron: function Perceptron() {

    var args = Array.prototype.slice.call(arguments); // convert arguments to Array
    if (args.length < 3)
      throw "Error: not enough layers (minimum 3) !!";

    var inputs = args.shift(); // first argument
    var outputs = args.pop(); // last argument
    var layers = args; // all the arguments in the middle

    var input = new Layer(inputs);
    var hidden = [];
    var output = new Layer(outputs);

    var previous = input;

    // generate hidden layers
    for (level in layers) {
      var size = layers[level];
      var layer = new Layer(size);
      hidden.push(layer);
      previous.project(layer);
      previous = layer;
    }
    previous.project(output);

    // set layers of the neural network
    this.set({
      input: input,
      hidden: hidden,
      output: output
    });

    // trainer for the network
    this.trainer = new Trainer(this);
  },

  // Multilayer Long Short-Term Memory
  LSTM: function LSTM() {

    var args = Array.prototype.slice.call(arguments); // convert arguments to array
    if (args.length < 3)
      throw "Error: not enough layers (minimum 3) !!";

    var last = args.pop();
    var option = {
      peepholes: Layer.connectionType.ALL_TO_ALL,
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
    } else
      var outputs = last;

    var inputs = args.shift();
    var layers = args;

    var inputLayer = new Layer(inputs);
    var hiddenLayers = [];
    var outputLayer = new Layer(outputs);

    var previous = null;

    // generate layers
    for (var layer in layers) {
      // generate memory blocks (memory cell and respective gates)
      var size = layers[layer];

      var inputGate = new Layer(size).set({
        bias: 1
      });
      var forgetGate = new Layer(size).set({
        bias: 1
      });
      var memoryCell = new Layer(size);
      var outputGate = new Layer(size).set({
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
        memoryCell.project(memoryCell, Layer.connectionType.ALL_TO_ELSE);

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
      inputGate.gate(input, Layer.gateType.INPUT);
      forgetGate.gate(self, Layer.gateType.ONE_TO_ONE);
      outputGate.gate(output, Layer.gateType.OUTPUT);
      if (previous != null)
        inputGate.gate(cell, Layer.gateType.INPUT);

      previous = memoryCell;
    }

    // input to output direct connection
    if (option.intoout)
      inputLayer.project(outputLayer);

    // set the layers of the neural network
    this.set({
      input: inputLayer,
      hidden: hiddenLayers,
      output: outputLayer
    });

    // trainer
    this.trainer = new Trainer(this);
  },

  // Liquid State Machine
  Liquid: function Liquid(inputs, hidden, outputs, connections, gates) {

    // create layers
    var inputLayer = new Layer(inputs);
    var hiddenLayer = new Layer(hidden);
    var outputLayer = new Layer(outputs);

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
      var connection = Math.random() * connectionList.length | 0;
      // let the gater gate the connection
      neurons[gater].gate(connectionList[connection]);
    }

    // connect the layers
    inputLayer.project(hiddenLayer);
    hiddenLayer.project(outputLayer);

    // set the layers of the network
    this.set({
      input: inputLayer,
      hidden: [hiddenLayer],
      output: outputLayer
    });

    // trainer
    this.trainer = new Trainer(this);
  },

  Hopfield: function Hopfield(size)
  {
    var inputLayer = new Layer(size);
    var outputLayer = new Layer(size);

    inputLayer.project(outputLayer, Layer.connectionType.ALL_TO_ALL);

    this.set({
      input: inputLayer,
      hidden: [],
      output: outputLayer
    });

    var trainer = new Trainer(this);

    var proto = Architect.Hopfield.prototype;

    proto.learn = proto.learn || function(patterns)
    {
      var set = [];
      for (var p in patterns)
        set.push({
          input: patterns[p],
          output: patterns[p]
        });

      return trainer.train(set, {
        iterations: 500000,
        error: .00005,
        rate: 1
      });
    }

    proto.feed = proto.feed || function(pattern)
    {
      var output = this.activate(pattern);

      var pattern = [];
      for (var i in output)
        pattern[i] = output[i] > .5 ? 1 : 0;

      return pattern;
    }
  }
}

// Extend prototype chain (so every architectures is an instance of Network)
for (var architecture in Architect) {
  Architect[architecture].prototype = new Network();
  Architect[architecture].prototype.constructor = Architect[architecture];
}

// export
if (module) module.exports = Architect; 


},{"./layer":3,"./network":4,"./trainer":6}],3:[function(require,module,exports){
// import
var Neuron = require('./neuron');

/*******************************************************************************************
                                            LAYER
*******************************************************************************************/

function Layer(size, label) {
  this.size = size | 0;
  this.list = [];
  this.label = label || null;
  this.connectedto = [];

  while (size--) {
    var neuron = new Neuron();
    this.list.push(neuron);
  }
}

Layer.prototype = {

  // activates all the neurons in the layer
  activate: function(input) {

    var activations = [];

    if (typeof input != 'undefined') {
      if (input.length != this.size)
        throw "INPUT size and LAYER size must be the same to activate!";

      for (var id in this.list) {
        var neuron = this.list[id];
        var activation = neuron.activate(input[id]);
        activations.push(activation);
      }
    } else {
      for (var id in this.list) {
        var neuron = this.list[id];
        var activation = neuron.activate();
        activations.push(activation);
      }
    }
    return activations;
  },

  // propagates the error on all the neurons of the layer
  propagate: function(rate, target) {

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
  },

  // projects a connection from this layer to another one
  project: function(layer, type, weights) {

    if (layer instanceof require('./network'))
      layer = layer.layers.input;

    if (layer instanceof Layer) {
      if (!this.connected(layer))
        return new Layer.connection(this, layer, type, weights);
    } else
      throw "Invalid argument, you can only project connections to LAYERS and NETWORKS!";


  },

  // gates a connection betwenn two layers
  gate: function(connection, type) {

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
    connection.gatedfrom.push({layer: this, type: type});
  },

  // true or false whether the whole layer is self-connected or not
  selfconnected: function() {

    for (var id in this.list) {
      var neuron = this.list[id];
      if (!neuron.selfconnected())
        return false;
    }
    return true;
  },

  // true of false whether the layer is connected to another layer (parameter) or not
  connected: function(layer) {
    // Check if ALL to ALL connection
    var connections = 0;
    for (var here in this.list) {
      for (var there in layer.list) {
        var from = this.list[here];
        var to = layer.list[there];
        var connected = from.connected(to);
        if (connected.type == 'projected')
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
      if (connected.type == 'projected')
        connections++;
    }
    if (connections == this.size)
      return Layer.connectionType.ONE_TO_ONE;
  },

  // clears all the neuorns in the layer
  clear: function() {
    for (var id in this.list) {
      var neuron = this.list[id];
      neuron.clear();
    }
  },

  // resets all the neurons in the layer
  reset: function() {
    for (var id in this.list) {
      var neuron = this.list[id];
      neuron.reset();
    }
  },

  // returns all the neurons in the layer (array)
  neurons: function() {
    return this.list;
  },

  // adds a neuron to the layer
  add: function(neuron) {
    this.neurons[neuron.ID] = neuron || new Neuron();
    this.list.push(neuron);
    this.size++;
  },

  set: function(options) {
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

// represents a connection from one layer to another, and keeps track of its weight and gain
Layer.connection = function LayerConnection(fromLayer, toLayer, type, weights) {
  this.ID = Layer.connection.uid();
  this.from = fromLayer;
  this.to = toLayer;
  this.selfconnection = toLayer == fromLayer;
  this.type = type;
  this.connections = {};
  this.list = [];
  this.size = 0;
  this.gatedfrom = [];

  if (typeof this.type == 'undefined')
  {
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
        if(this.type == Layer.connectionType.ALL_TO_ELSE && from == to)
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

// types of connections
Layer.connectionType = {};
Layer.connectionType.ALL_TO_ALL = "ALL TO ALL";
Layer.connectionType.ONE_TO_ONE = "ONE TO ONE";
Layer.connectionType.ALL_TO_ELSE = "ALL TO ELSE";

// types of gates
Layer.gateType = {};
Layer.gateType.INPUT = "INPUT";
Layer.gateType.OUTPUT = "OUTPUT";
Layer.gateType.ONE_TO_ONE = "ONE TO ONE";

(function() {
  var connections = 0;
  Layer.connection.uid = function() {
    return connections++;
  }
})();

// export
if (module) module.exports = Layer;


},{"./network":4,"./neuron":5}],4:[function(require,module,exports){
// import
var Neuron = require('./neuron'),
    Layer = require('./layer');

/*******************************************************************************************
                                         NETWORK
*******************************************************************************************/

function Network(layers) {
  if (typeof layers != 'undefined') {
    this.layers = layers || {
      input: null,
      hidden: {},
      output: null
    };
    this.optimized = null;
  }
}
Network.prototype = {

  // feed-forward activation of all the layers to produce an ouput
  activate: function(input) {

    if (this.optimized === false)
    {
      this.layers.input.activate(input);
      for (var layer in this.layers.hidden)
        this.layers.hidden[layer].activate();
      return this.layers.output.activate();
    } 
    else 
    {
      if (this.optimized == null)
        this.optimize();
      return this.optimized.activate(input);
    }
  },

  // back-propagate the error thru the network
  propagate: function(rate, target) {

    if (this.optimized === false)
    {
      this.layers.output.propagate(rate, target);
      var reverse = [];
      for (var layer in this.layers.hidden)
        reverse.push(this.layers.hidden[layer]);
      reverse.reverse();
      for (var layer in reverse)
        reverse[layer].propagate(rate);
    } 
    else 
    {
      if (this.optimized == null)
        this.optimize();
      this.optimized.propagate(rate, target);
    }
  },

  // project a connection to another unit (either a network or a layer)
  project: function(unit, type, weights) {

    if (this.optimized)
      this.optimized.reset();

    if (unit instanceof Network)
      return this.layers.output.project(unit.layers.input, type, weights);

    if (unit instanceof Layer)
      return this.layers.output.project(unit, type, weights);

    throw "Invalid argument, you can only project connections to LAYERS and NETWORKS!";
  },

  // let this network gate a connection
  gate: function(connection, type) {
    if (this.optimized)
      this.optimized.reset();
    this.layers.output.gate(connection, type);
  },

  // clear all elegibility traces and extended elegibility traces (the network forgets its context, but not what was trained)
  clear: function() {

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
  },

  // reset all weights and clear all traces (ends up like a new network)
  reset: function() {

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
  },

  // hardcodes the behaviour of the whole network into a single optimized function
  optimize: function() {

    var that = this;
    var optimized = {};
    var neurons = this.neurons();

    for (var i in neurons) {
      var neuron = neurons[i].neuron;
      var layer = neurons[i].layer;
      while (neuron.neuron)
        neuron = neuron.neuron;
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
  },

  // restores all the values from the optimized network the their respective objects in order to manipulate the network
  restore: function() {
    if (!this.optimized)
      return;

    var optimized = this.optimized;

    var getValue = function() {
      var args = Array.prototype.slice.call(arguments);

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
      while (neuron.neuron)
        neuron = neuron.neuron;

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
      while (neuron.neuron)
        neuron = neuron.neuron;

      for (var j in neuron.connections.projected) {
        var connection = neuron.connections.projected[j];
        connection.weight = getValue(connection, 'weight');
        connection.gain = getValue(connection, 'gain');
      }
    }
  },

  // returns all the neurons in the network
  neurons: function() {

    var neurons = [];

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
  },

  // returns number of inputs of the network
  inputs: function() {
    return this.layers.input.size;
  },

  // returns number of outputs of hte network
  outputs: function() {
    return this.layers.output.size;
  },

  // sets the layers of the network
  set: function(layers) {

    this.layers = layers;
    if (this.optimized)
      this.optimized.reset();
  },

  setOptimize: function(bool){
    this.restore();
    if (this.optimized)
      this.optimized.reset();
    this.optimized = bool? null : false;
  },

  // returns a json that represents all the neurons and connections of the network
  toJSON: function(ignoreTraces) {

    this.restore();

    var list = this.neurons();
    var neurons = [];
    var connections = [];

    // link id's to positions in the array
    var ids = {};
    for (var i in list) {
      var neuron = list[i].neuron;
      while (neuron.neuron)
        neuron = neuron.neuron;
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
        layer: list[i].layer
      };

      copy.squash = neuron.squash == Neuron.squash.LOGISTIC ? "LOGISTIC" :
        neuron.squash == Neuron.squash.TANH ? "TANH" :
        neuron.squash == Neuron.squash.IDENTITY ? "IDENTITY" :
        neuron.squash == Neuron.squash.HLIM ? "HLIM" :
        null;

      neurons.push(copy);
    }

    if (!ignoreTraces)
      for (var i in neurons) {
        var copy = neurons[i];

        for (var input in neuron.trace.elegibility)
          copy.trace.elegibility[input] = neuron.trace.elegibility[input];

        for (var gated in neuron.trace.extended) {
          copy.trace.extended[gated] = {};
          for (var input in neuron.trace.extended[gated])
            copy.trace.extended[ids[gated]][input] = neuron.trace.extended[
              gated][input];
        }
      }

    // get connections
    for (var i in list) {
      var neuron = list[i].neuron;
      while (neuron.neuron)
        neuron = neuron.neuron;

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
  },
  
  // export the topology into dot language which can be visualized as graphs using dot
  /* example: ... console.log(net.toDotLang());
              $ node example.js > example.dot
              $ dot example.dot -Tpng > out.png
  */
  toDotLang: function(edgeconnection) {
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
        if ( edgeconnection) {
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
    return code;
  },

  // returns a function that works as the activation of the network and can be used without depending on the library
  standalone: function() {
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
  },

  worker: function() {
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
  },

  // returns a copy of the network
  clone: function(ignoreTraces) {
    return Network.fromJSON(this.toJSON(ignoreTraces));
  }
}

// rebuild a network that has been stored in a json using the method toJson()
Network.fromJSON = function(json) {

  var neurons = [];

  var layers = {
    input: new Layer(),
    hidden: [],
    output: new Layer()
  }

  for (var i in json.neurons) {
    var config = json.neurons[i];

    var neuron = new Neuron();
    neuron.trace.elegibility = config.trace.elegibility;
    neuron.trace.extended = config.trace.extended;
    neuron.state = config.state;
    neuron.old = config.old;
    neuron.activation = config.activation;
    neuron.bias = config.bias;
    neuron.squash = config.squash in Neuron.squash ? Neuron.squash[config.squash] :
      Neuron.squash.LOGISTIC;
    neurons.push(neuron);

    if (config.layer == 'input')
      layers.input.add(neuron);
    else if (config.layer == 'output')
      layers.output.add(neuron);
    else {
      if (typeof layers.hidden[config.layer] == 'undefined')
        layers.hidden[config.layer] = new Layer();
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

// export
if (module) module.exports = Network;


},{"./layer":3,"./neuron":5}],5:[function(require,module,exports){
/******************************************************************************************
                                         NEURON
*******************************************************************************************/

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
  this.selfconnection = new Neuron.connection(this, this, 0); // weight = 0 -> not connected
  this.squash = Neuron.squash.LOGISTIC;
  this.neighboors = {};
  this.bias = Math.random() * .2 - .1;
}

Neuron.prototype = {

  // activate the neuron
  activate: function(input) {
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
      var input = this.connections.inputs[i];
      this.state += input.from.activation * input.weight * input.gain;
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
      for (var incoming in this.trace.influences[neuron.ID]) { // captures the effect that has an input connection to this unit, on a neuron that is gated by this unit
        influence += this.trace.influences[neuron.ID][incoming].weight *
          this.trace.influences[neuron.ID][incoming].from.activation;
      }
      influences[neuron.ID] = influence;
    }
    
    for (var i in this.connections.inputs) {
      var input = this.connections.inputs[i];

      // elegibility trace - Eq. 17
      this.trace.elegibility[input.ID] = this.selfconnection.gain * this.selfconnection
        .weight * this.trace.elegibility[input.ID] + input.gain * input.from
        .activation;

      for (var id in this.trace.extended) {
        // extended elegibility trace
        var xtrace = this.trace.extended[id];
        var neuron = this.neighboors[id];
        var influence = influences[neuron.ID];

        // eq. 18
        xtrace[input.ID] = neuron.selfconnection.gain * neuron.selfconnection
          .weight * xtrace[input.ID] + this.derivative * this.trace.elegibility[
            input.ID] * influence;
      }
    }

    //  update gated connection's gains
    for (var connection in this.connections.gated) {
      this.connections.gated[connection].gain = this.activation;
    }

    return this.activation;
  },

  // back-propagate the error
  propagate: function(rate, target) {
    // error accumulator
    var error = 0;

    // whether or not this neuron is in the output layer
    var isOutput = typeof target != 'undefined';

    // output neurons get their error from the enviroment
    if (isOutput)
      this.error.responsibility = this.error.projected = target - this.activation; // Eq. 10
    
    else // the rest of the neuron compute their error responsibilities by backpropagation
    {
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
        for (var input in this.trace.influences[id]) { // captures the effect that the input connection of this neuron have, on a neuron which its input/s is/are gated by this neuron
          influence += this.trace.influences[id][input].weight * this.trace.influences[
            neuron.ID][input].from.activation;
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
      var input = this.connections.inputs[id];

      // Eq. 24
      var gradient = this.error.projected * this.trace.elegibility[input.ID];
      for (var id in this.trace.extended) {
        var neuron = this.neighboors[id];
        gradient += neuron.error.responsibility * this.trace.extended[
          neuron.ID][input.ID];
      }
      input.weight += rate * gradient; // adjust weights - aka learn
    }

    // adjust bias
    this.bias += rate * this.error.responsibility;
  },

  project: function(neuron, weight) {
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
    } else {
      // create a new connection
      var connection = new Neuron.connection(this, neuron, weight);
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
  },

  gate: function(connection) {
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
  },

  // returns true or false whether the neuron is self-connected or not
  selfconnected: function() {
    return this.selfconnection.weight !== 0;
  },

  // returns true or false whether the neuron is connected to another neuron (parameter)
  connected: function(neuron) {
    var result = {
      type: null,
      connection: false
    };

    if (this == neuron) {
      if (this.selfconnected()) {
        result.type = 'selfconnection';
        result.connection = this.selfconnection;
        return result;
      } else
        return false;
    }

    for (var type in this.connections) {
      for (var connection in this.connections[type]) {
        var connection = this.connections[type][connection];
        if (connection.to == neuron) {
          result.type = type;
          result.connection = connection;
          return result;
        } else if (connection.from == neuron) {
          result.type = type;
          result.connection = connection;
          return result;
        }
      }
    }

    return false;
  },

  // clears all the traces (the neuron forgets it's context, but the connections remain intact)
  clear: function() {

    for (var trace in this.trace.elegibility)
      this.trace.elegibility[trace] = 0;

    for (var trace in this.trace.extended)
      for (var extended in this.trace.extended[trace])
        this.trace.extended[trace][extended] = 0;

    this.error.responsibility = this.error.projected = this.error.gated = 0;
  },

  // all the connections are randomized and the traces are cleared
  reset: function() {
    this.clear();

    for (var type in this.connection)
      for (var connection in this.connection[type])
        this.connection[type][connection].weight = Math.random() * .2 - .1;
    this.bias = Math.random() * .2 - .1;

    this.old = this.state = this.activation = 0;
  },

  // hardcodes the behaviour of the neuron into an optimized function
  optimize: function(optimized, layer) {
    
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
    var allocate = function(store){
      var allocated = layer in layers && store[layers.__count];
      if (!allocated)
      {
        layers.__count = store.push([]) - 1;
        layers[layer] = layers.__count;
      }
    }
    allocate(activation_sentences);
    allocate(trace_sentences);
    allocate(propagation_sentences);
    var currentLayer = layers.__count;

    // get/reserve space in memory by creating a unique ID for a variablel
    var getVar = function() {
      var args = Array.prototype.slice.call(arguments);

      if (args.length == 1) {
        if (args[0] == 'target') {
          var id = 'target_' + targets.length;
          targets.push(varID);
        } else
          var id = args[0];
        if (id in variables)
          return variables[id];
        return variables[id] = {
          value: 0,
          id: varID++
        };
      } else {
        var extended = args.length > 2;
        if (extended)
          var value = args.pop();

        var unit = args.shift();
        var prop = args.pop();

        if (!extended)
          var value = unit[prop];

        var id = prop + '_';
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
    var buildSentence = function() {
      var args = Array.prototype.slice.call(arguments);
      var store = args.pop();
      var sentence = "";
      for (var i in args)
        if (typeof args[i] == 'string')
          sentence += args[i];
        else
          sentence += 'F[' + args[i].id + ']';

      store.push(sentence + ';');
    }

    // helper to check if an object is empty
    var isEmpty = function(obj) {
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
          buildSentence(state, ' = ', self_gain, ' * ', self_weight, ' * ',
            state, ' + ', bias, store_activation);
        else
          buildSentence(state, ' = ', self_weight, ' * ', state, ' + ',
            bias, store_activation);
      else
        buildSentence(state, ' = ', bias, store_activation);
      for (var i in this.connections.inputs) {
        var input = this.connections.inputs[i];
        var input_activation = getVar(input.from, 'activation');
        var input_weight = getVar(input, 'weight');
        if (input.gater)
          var input_gain = getVar(input, 'gain');
        if (this.connections.inputs[i].gater)
          buildSentence(state, ' += ', input_activation, ' * ',
            input_weight, ' * ', input_gain, store_activation);
        else
          buildSentence(state, ' += ', input_activation, ' * ',
            input_weight, store_activation);
      }
      var derivative = getVar(this, 'derivative');
      switch (this.squash) {
        case Neuron.squash.LOGISTIC:
          buildSentence(activation, ' = (1 / (1 + Math.exp(-', state, ')))',
            store_activation);
          buildSentence(derivative, ' = ', activation, ' * (1 - ',
            activation, ')', store_activation);
          break;
        case Neuron.squash.TANH:
          var eP = getVar('aux');
          var eN = getVar('aux_2');
          buildSentence(eP, ' = Math.exp(', state, ')', store_activation);
          buildSentence(eN, ' = 1 / ', eP, store_activation);
          buildSentence(activation, ' = (', eP, ' - ', eN, ') / (', eP, ' + ', eN, ')', store_activation);
          buildSentence(derivative, ' = 1 - (', activation, ' * ', activation, ')', store_activation);
          break;
        case Neuron.squash.IDENTITY:
          buildSentence(activation, ' = ', state, store_activation);
          buildSentence(derivative, ' = 1', store_activation);
          break;
        case Neuron.squash.HLIM:
          buildSentence(activation, ' = +(', state, ' > 0)',
            store_activation);
          buildSentence(derivative, ' = 1', store_activation);
          break;
      }

      influences = [];
      for (var id in this.trace.extended) {
        // calculate extended elegibility traces in advance
        
        var xtrace = this.trace.extended[id];
        var neuron = this.neighboors[id];
        var influence = getVar('aux');
        var neuron_old = getVar(neuron, 'old');
        var initialized = false;
        if (neuron.selfconnection.gater == this)
        {
          buildSentence(influence, ' = ', neuron_old, store_trace);
          initialized = true;
        }
        for (var incoming in this.trace.influences[neuron.ID]) {
          var incoming_weight = getVar(this.trace.influences[neuron.ID]
            [incoming], 'weight');
          var incoming_activation = getVar(this.trace.influences[neuron.ID]
            [incoming].from, 'activation');

          if (initialized)
            buildSentence(influence, ' += ', incoming_weight, ' * ',
              incoming_activation, store_trace);
          else {
            buildSentence(influence, ' = ', incoming_weight, ' * ',
              incoming_activation, store_trace);
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
              buildSentence(trace, ' = ', self_gain, ' * ', self_weight,
                ' * ', trace, ' + ', input_gain, ' * ', input_activation,
                store_trace);
            else
              buildSentence(trace, ' = ', self_gain, ' * ', self_weight,
                ' * ', trace, ' + ', input_activation, store_trace);
          } else {
            if (input.gater)
              buildSentence(trace, ' = ', self_weight, ' * ', trace, ' + ',
                input_gain, ' * ', input_activation, store_trace);
            else
              buildSentence(trace, ' = ', self_weight, ' * ', trace, ' + ',
                input_activation, store_trace);
          }
        } else {
          if (input.gater)
            buildSentence(trace, ' = ', input_gain, ' * ', input_activation,
              store_trace);
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
          var xtrace = getVar(this, 'trace', 'extended', neuron.ID, input.ID,
            this.trace.extended[neuron.ID][input.ID]);
          if (neuron.selfconnected())
            var neuron_self_weight = getVar(neuron.selfconnection, 'weight');
          if (neuron.selfconnection.gater)
            var neuron_self_gain = getVar(neuron.selfconnection, 'gain');
          if (neuron.selfconnected())
            if (neuron.selfconnection.gater)
              buildSentence(xtrace, ' = ', neuron_self_gain, ' * ',
                neuron_self_weight, ' * ', xtrace, ' + ', derivative, ' * ',
                trace, ' * ', "influences[" + influences.indexOf(neuron.ID) + "]", store_trace);
            else
              buildSentence(xtrace, ' = ', neuron_self_weight, ' * ',
                xtrace, ' + ', derivative, ' * ', trace, ' * ',
                "influences[" + influences.indexOf(neuron.ID) + "]", store_trace);
          else
            buildSentence(xtrace, ' = ', derivative, ' * ', trace, ' * ',
              "influences[" + influences.indexOf(neuron.ID) + "]", store_trace);
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
        buildSentence(responsibility, ' = ', target, ' - ', activation,
          store_propagation);
        for (var id in this.connections.inputs) {
          var input = this.connections.inputs[id];
          var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace
            .elegibility[input.ID]);
          var input_weight = getVar(input, 'weight');
          buildSentence(input_weight, ' += ', rate, ' * (', responsibility,
            ' * ', trace, ')', store_propagation);
        }
        outputs.push(activation.id);
      } else {
        if (!noProjections && !noGates) {
          var error = getVar('aux');
          for (var id in this.connections.projected) {
            var connection = this.connections.projected[id];
            var neuron = connection.to;
            var connection_weight = getVar(connection, 'weight');
            var neuron_responsibility = getVar(neuron, 'error',
              'responsibility', neuron.error.responsibility);
            if (connection.gater) {
              var connection_gain = getVar(connection, 'gain');
              buildSentence(error, ' += ', neuron_responsibility, ' * ',
                connection_gain, ' * ', connection_weight,
                store_propagation);
            } else
              buildSentence(error, ' += ', neuron_responsibility, ' * ',
                connection_weight, store_propagation);
          }
          var projected = getVar(this, 'error', 'projected', this.error.projected);
          buildSentence(projected, ' = ', derivative, ' * ', error,
            store_propagation);
          buildSentence(error, ' = 0', store_propagation);
          for (var id in this.trace.extended) {
            var neuron = this.neighboors[id];
            var influence = getVar('aux_2');
            var neuron_old = getVar(neuron, 'old');
            if (neuron.selfconnection.gater == this)
              buildSentence(influence, ' = ', neuron_old, store_propagation);
            else
              buildSentence(influence, ' = 0', store_propagation);
            for (var input in this.trace.influences[neuron.ID]) {
              var connection = this.trace.influences[neuron.ID][input];
              var connection_weight = getVar(connection, 'weight');
              var neuron_activation = getVar(connection.from, 'activation');
              buildSentence(influence, ' += ', connection_weight, ' * ',
                neuron_activation, store_propagation);
            }
            var neuron_responsibility = getVar(neuron, 'error',
              'responsibility', neuron.error.responsibility);
            buildSentence(error, ' += ', neuron_responsibility, ' * ',
              influence, store_propagation);
          }
          var gated = getVar(this, 'error', 'gated', this.error.gated);
          buildSentence(gated, ' = ', derivative, ' * ', error,
            store_propagation);
          buildSentence(responsibility, ' = ', projected, ' + ', gated,
            store_propagation);
          for (var id in this.connections.inputs) {
            var input = this.connections.inputs[id];
            var gradient = getVar('aux');
            var trace = getVar(this, 'trace', 'elegibility', input.ID, this
              .trace.elegibility[input.ID]);
            buildSentence(gradient, ' = ', projected, ' * ', trace,
              store_propagation);
            for (var id in this.trace.extended) {
              var neuron = this.neighboors[id];
              var neuron_responsibility = getVar(neuron, 'error',
                'responsibility', neuron.error.responsibility);
              var xtrace = getVar(this, 'trace', 'extended', neuron.ID,
                input.ID, this.trace.extended[neuron.ID][input.ID]);
              buildSentence(gradient, ' += ', neuron_responsibility, ' * ',
                xtrace, store_propagation);
            }
            var input_weight = getVar(input, 'weight');
            buildSentence(input_weight, ' += ', rate, ' * ', gradient,
              store_propagation);
          }

        } else if (noGates) {
          buildSentence(responsibility, ' = 0', store_propagation);
          for (var id in this.connections.projected) {
            var connection = this.connections.projected[id];
            var neuron = connection.to;
            var connection_weight = getVar(connection, 'weight');
            var neuron_responsibility = getVar(neuron, 'error',
              'responsibility', neuron.error.responsibility);
            if (connection.gater) {
              var connection_gain = getVar(connection, 'gain');
              buildSentence(responsibility, ' += ', neuron_responsibility,
                ' * ', connection_gain, ' * ', connection_weight,
                store_propagation);
            } else
              buildSentence(responsibility, ' += ', neuron_responsibility,
                ' * ', connection_weight, store_propagation);
          }
          buildSentence(responsibility, ' *= ', derivative,
            store_propagation);
          for (var id in this.connections.inputs) {
            var input = this.connections.inputs[id];
            var trace = getVar(this, 'trace', 'elegibility', input.ID, this
              .trace.elegibility[input.ID]);
            var input_weight = getVar(input, 'weight');
            buildSentence(input_weight, ' += ', rate, ' * (',
              responsibility, ' * ', trace, ')', store_propagation);
          }
        } else if (noProjections) {
          buildSentence(responsibility, ' = 0', store_propagation);
          for (var id in this.trace.extended) {
            var neuron = this.neighboors[id];
            var influence = getVar('aux');
            var neuron_old = getVar(neuron, 'old');
            if (neuron.selfconnection.gater == this)
              buildSentence(influence, ' = ', neuron_old, store_propagation);
            else
              buildSentence(influence, ' = 0', store_propagation);
            for (var input in this.trace.influences[neuron.ID]) {
              var connection = this.trace.influences[neuron.ID][input];
              var connection_weight = getVar(connection, 'weight');
              var neuron_activation = getVar(connection.from, 'activation');
              buildSentence(influence, ' += ', connection_weight, ' * ',
                neuron_activation, store_propagation);
            }
            var neuron_responsibility = getVar(neuron, 'error',
              'responsibility', neuron.error.responsibility);
            buildSentence(responsibility, ' += ', neuron_responsibility,
              ' * ', influence, store_propagation);
          }
          buildSentence(responsibility, ' *= ', derivative,
            store_propagation);
          for (var id in this.connections.inputs) {
            var input = this.connections.inputs[id];
            var gradient = getVar('aux');
            buildSentence(gradient, ' = 0', store_propagation);
            for (var id in this.trace.extended) {
              var neuron = this.neighboors[id];
              var neuron_responsibility = getVar(neuron, 'error',
                'responsibility', neuron.error.responsibility);
              var xtrace = getVar(this, 'trace', 'extended', neuron.ID,
                input.ID, this.trace.extended[neuron.ID][input.ID]);
              buildSentence(gradient, ' += ', neuron_responsibility, ' * ',
                xtrace, store_propagation);
            }
            var input_weight = getVar(input, 'weight');
            buildSentence(input_weight, ' += ', rate, ' * ', gradient,
              store_propagation);
          }
        }
      }
      buildSentence(bias, ' += ', rate, ' * ', responsibility,
        store_propagation);
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
    }
  }
}


// represents a connection between two neurons
Neuron.connection = function Connection(from, to, weight) {

  if (!from || !to)
    throw "Connection Error: Invalid neurons";

  this.ID = Neuron.connection.uid();
  this.from = from;
  this.to = to;
  this.weight = typeof weight == 'undefined' ? Math.random() * .2 - .1 :
    weight;
  this.gain = 1;
  this.gater = null;
}


// squashing functions
Neuron.squash = {};

// eq. 5 & 5'
Neuron.squash.LOGISTIC = function(x, derivate) {
  if (!derivate)
    return 1 / (1 + Math.exp(-x));
  var fx = Neuron.squash.LOGISTIC(x);
  return fx * (1 - fx);
};
Neuron.squash.TANH = function(x, derivate) {
  if (derivate)
    return 1 - Math.pow(Neuron.squash.TANH(x), 2);
  var eP = Math.exp(x);
  var eN = 1 / eP;
  return (eP - eN) / (eP + eN);
};
Neuron.squash.IDENTITY = function(x, derivate) {
  return derivate ? 1 : x;
};
Neuron.squash.HLIM = function(x, derivate) {
  return derivate ? 1 : +(x > 0);
};

// unique ID's
(function() {
  var neurons = 0;
  var connections = 0;
  Neuron.uid = function() {
    return neurons++;
  }
  Neuron.connection.uid = function() {
    return connections++;
  }
  Neuron.quantity = function() {
    return {
      neurons: neurons,
      connections: connections
    }
  }
})();

// export
if (module) module.exports = Neuron;


},{}],6:[function(require,module,exports){
/*******************************************************************************************
                                        TRAINER
*******************************************************************************************/

function Trainer(network, options) {
  options = options || {};
  this.network = network;
  this.rate = options.rate || .2;
  this.iterations = options.iterations || 100000;
  this.error = options.error || .005
  this.cost = options.cost || Trainer.cost.CROSS_ENTROPY;
}

Trainer.prototype = {

  // trains any given set to a network
  train: function(set, options) {

    var error = 1;
    var iterations = bucketSize = 0;
    var abort_training = false;
    var input, output, target, currentRate;

    var start = Date.now();

    if (options) {
      if (options.shuffle) {
        //+ Jonas Raoni Soares Silva
        //@ http://jsfromhell.com/array/shuffle [v1.0]
        function shuffle(o) { //v1.0
          for (var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
          return o;
        };
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
      if (options.customLog){
        // for backward compatibility with code that used customLog
        console.log('Deprecated: use schedule instead of customLog')
        this.schedule = options.customLog;
      }
    }

    currentRate = this.rate;
    if(Array.isArray(this.rate)) {
      bucketSize = Math.floor(this.iterations / this.rate.length);
    }


    while (!abort_training && iterations < this.iterations && error > this.error) {
      error = 0;

      if(bucketSize > 0) {
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
        };
        if (options.shuffle)
          shuffle(set);
      }
    }

    var results = {
      error: error,
      iterations: iterations,
      time: Date.now() - start
    }

    return results;
  },

  // trains any given set to a network using a WebWorker
  workerTrain: function(set, callback, options) {

    var that = this;
    var error = 1;
    var iterations = bucketSize = 0;
    var input, output, target, currentRate;
    var length = set.length;

    var start = Date.now();

    if (options) {
      if (options.shuffle) {
        //+ Jonas Raoni Soares Silva
        //@ http://jsfromhell.com/array/shuffle [v1.0]
        function shuffle(o) { //v1.0
          for (var j, x, i = o.length; i; j = Math.floor(Math.random() *
              i), x = o[--i], o[i] = o[j], o[j] = x);
          return o;
        };
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
        console.log('Deprecated: use schedule instead of customLog')
        this.schedule = options.customLog;
    }

    // dynamic learning rate
    currentRate = this.rate;
    if(Array.isArray(this.rate)) {
      bucketSize = Math.floor(this.iterations / this.rate.length);
    }

    // create a worker
    var worker = this.network.worker();

    // activate the network
    function activateWorker(input)
    {
        worker.postMessage({ 
            action: "activate",
            input: input,
            memoryBuffer: that.network.optimized.memory
        }, [that.network.optimized.memory.buffer]);
    }

    // backpropagate the network
    function propagateWorker(target){
        if(bucketSize > 0) {
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
    worker.onmessage = function(e){
        // give control of the memory back to the network
        that.network.optimized.ownership(e.data.memoryBuffer);

        if (e.data.action == "propagate")
        {
            if (index >= length)
            {
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
                  };
                  if (options.shuffle)
                    shuffle(set);
                }

                if (!abort_training && iterations < that.iterations && error > that.error)
                {
                    activateWorker(set[index].input);
                } else {
                    // callback
                    callback({
                      error: error,
                      iterations: iterations,
                      time: Date.now() - start
                    })
                }
                error = 0;
            } else {
                activateWorker(set[index].input);
            }
        }

        if (e.data.action == "activate")
        {
            error += that.cost(set[index].output, e.data.output);
            propagateWorker(set[index].output); 
            index++;
        }
    }

    // kick it
    var index = 0;
    var iterations = 0;
    activateWorker(set[index].input);
  },

  // trains an XOR to the network
  XOR: function(options) {

    if (this.network.inputs() != 2 || this.network.outputs() != 1)
      throw "Error: Incompatible network (2 inputs, 1 output)";

    var defaults = {
      iterations: 100000,
      log: false,
      shuffle: true,
      cost: Trainer.cost.MSE
    }

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
  },

  // trains the network to pass a Distracted Sequence Recall test
  DSR: function(options) {
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

    var trial = correct = i = j = success = 0,
      error = 1,
      symbols = targets.length + distractors.length + prompts.length;

    var noRepeat = function(range, avoid) {
      var number = Math.random() * range | 0;
      var used = false;
      for (var i in avoid)
        if (number == avoid[i])
          used = true;
      return used ? noRepeat(range, avoid) : number;
    }

    var equal = function(prediction, output) {
      for (var i in prediction)
        if (Math.round(prediction[i]) != output[i])
          return false;
      return true;
    }

    var start = Date.now();

    while (trial < iterations && (success < criterion || trial % 1000 != 0)) {
      // generate sequence
      var sequence = [],
        sequenceLength = length - prompts.length;
      for (i = 0; i < sequenceLength; i++) {
        var any = Math.random() * distractors.length | 0;
        sequence.push(distractors[any]);
      }
      var indexes = [],
        positions = [];
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
        console.log("iterations:", trial, " success:", success, " correct:",
          correct, " time:", Date.now() - start, " error:", error);
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
    }
  },

  // train the network to learn an Embeded Reber Grammar
  ERG: function(options) {

    options = options || {};
    var iterations = options.iterations || 150000;
    var criterion = options.error || .05;
    var rate = options.rate || .1;
    var log = options.log || 500;

    // gramar node
    var Node = function() {
      this.paths = [];
    }
    Node.prototype = {
      connect: function(node, value) {
        this.paths.push({
          node: node,
          value: value
        });
        return this;
      },
      any: function() {
        if (this.paths.length == 0)
          return false;
        var index = Math.random() * this.paths.length | 0;
        return this.paths[index];
      },
      test: function(value) {
        for (var i in this.paths)
          if (this.paths[i].value == value)
            return this.paths[i];
        return false;
      }
    }

    var reberGrammar = function() {

      // build a reber grammar
      var output = new Node();
      var n1 = (new Node()).connect(output, "E");
      var n2 = (new Node()).connect(n1, "S");
      var n3 = (new Node()).connect(n1, "V").connect(n2, "P");
      var n4 = (new Node()).connect(n2, "X")
      n4.connect(n4, "S");
      var n5 = (new Node()).connect(n3, "V")
      n5.connect(n5, "T");
      n2.connect(n5, "X")
      var n6 = (new Node()).connect(n4, "T").connect(n5, "P");
      var input = (new Node()).connect(n6, "B")

      return {
        input: input,
        output: output
      }
    }

    // build an embeded reber grammar
    var embededReberGrammar = function() {
      var reber1 = reberGrammar();
      var reber2 = reberGrammar();

      var output = new Node();
      var n1 = (new Node).connect(output, "E");
      reber1.output.connect(n1, "T");
      reber2.output.connect(n1, "P");
      var n2 = (new Node).connect(reber1.input, "P").connect(reber2.input,
        "T");
      var input = (new Node).connect(n2, "B");

      return {
        input: input,
        output: output
      }

    }

    // generate an ERG sequence
    var generate = function() {
      var node = embededReberGrammar().input;
      var next = node.any();
      var str = "";
      while (next) {
        str += next.value;
        next = next.node.any();
      }
      return str;
    }

    // test if a string matches an embeded reber grammar
    var test = function(str) {
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
    }

    // helper to check if the output and the target vectors match
    var different = function(array1, array2) {
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
    }

    var iteration = 0;
    var error = 1;
    var table = {
      "B": 0,
      "P": 1,
      "T": 2,
      "X": 3,
      "S": 4,
      "E": 5
    }

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
          delta += Math.pow(target[k] - output[k], 2)
        delta /= output.length;

        error += delta;
      }
      error /= sequence.length;
      iteration++;
      if (iteration % log == 0) {
        console.log("iterations:", iteration, " time:", Date.now() - start,
          " error:", error);
      }
    }

    return {
      iterations: iteration,
      error: error,
      time: Date.now() - start,
      test: test,
      generate: generate
    }
  }
};

// Built-in cost functions
Trainer.cost = {
  // Eq. 9
  CROSS_ENTROPY: function(target, output)
  {
    var crossentropy = 0;
    for (var i in output)
      crossentropy -= (target[i] * Math.log(output[i]+1e-15)) + ((1-target[i]) * Math.log((1+1e-15)-output[i])); // +1e-15 is a tiny push away to avoid Math.log(0)
    return crossentropy;
  },
  MSE: function(target, output)
  {
    var mse = 0;
    for (var i in output)
      mse += Math.pow(target[i] - output[i], 2);
    return mse / output.length;
  }
}

// export
if (module) module.exports = Trainer;


},{}]},{},[1])
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyaWZ5L25vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJzcmMvc3luYXB0aWMuanMiLCJzcmMvYXJjaGl0ZWN0LmpzIiwic3JjL2xheWVyLmpzIiwic3JjL25ldHdvcmsuanMiLCJzcmMvbmV1cm9uLmpzIiwic3JjL3RyYWluZXIuanMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7QUNBQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUNwRkE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUNsUkE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7O0FDcFJBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTs7QUNwbEJBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQ2p5QkE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBIiwiZmlsZSI6ImdlbmVyYXRlZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzQ29udGVudCI6WyIoZnVuY3Rpb24gZSh0LG4scil7ZnVuY3Rpb24gcyhvLHUpe2lmKCFuW29dKXtpZighdFtvXSl7dmFyIGE9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtpZighdSYmYSlyZXR1cm4gYShvLCEwKTtpZihpKXJldHVybiBpKG8sITApO3ZhciBmPW5ldyBFcnJvcihcIkNhbm5vdCBmaW5kIG1vZHVsZSAnXCIrbytcIidcIik7dGhyb3cgZi5jb2RlPVwiTU9EVUxFX05PVF9GT1VORFwiLGZ9dmFyIGw9bltvXT17ZXhwb3J0czp7fX07dFtvXVswXS5jYWxsKGwuZXhwb3J0cyxmdW5jdGlvbihlKXt2YXIgbj10W29dWzFdW2VdO3JldHVybiBzKG4/bjplKX0sbCxsLmV4cG9ydHMsZSx0LG4scil9cmV0dXJuIG5bb10uZXhwb3J0c312YXIgaT10eXBlb2YgcmVxdWlyZT09XCJmdW5jdGlvblwiJiZyZXF1aXJlO2Zvcih2YXIgbz0wO288ci5sZW5ndGg7bysrKXMocltvXSk7cmV0dXJuIHN9KSIsIi8qXG5cblRoZSBNSVQgTGljZW5zZSAoTUlUKVxuXG5Db3B5cmlnaHQgKGMpIDIwMTQgSnVhbiBDYXphbGEgLSBqdWFuY2F6YWxhLmNvbVxuXG5QZXJtaXNzaW9uIGlzIGhlcmVieSBncmFudGVkLCBmcmVlIG9mIGNoYXJnZSwgdG8gYW55IHBlcnNvbiBvYnRhaW5pbmcgYSBjb3B5XG5vZiB0aGlzIHNvZnR3YXJlIGFuZCBhc3NvY2lhdGVkIGRvY3VtZW50YXRpb24gZmlsZXMgKHRoZSBcIlNvZnR3YXJlXCIpLCB0byBkZWFsXG5pbiB0aGUgU29mdHdhcmUgd2l0aG91dCByZXN0cmljdGlvbiwgaW5jbHVkaW5nIHdpdGhvdXQgbGltaXRhdGlvbiB0aGUgcmlnaHRzXG50byB1c2UsIGNvcHksIG1vZGlmeSwgbWVyZ2UsIHB1Ymxpc2gsIGRpc3RyaWJ1dGUsIHN1YmxpY2Vuc2UsIGFuZC9vciBzZWxsXG5jb3BpZXMgb2YgdGhlIFNvZnR3YXJlLCBhbmQgdG8gcGVybWl0IHBlcnNvbnMgdG8gd2hvbSB0aGUgU29mdHdhcmUgaXNcbmZ1cm5pc2hlZCB0byBkbyBzbywgc3ViamVjdCB0byB0aGUgZm9sbG93aW5nIGNvbmRpdGlvbnM6XG5cblRoZSBhYm92ZSBjb3B5cmlnaHQgbm90aWNlIGFuZCB0aGlzIHBlcm1pc3Npb24gbm90aWNlIHNoYWxsIGJlIGluY2x1ZGVkIGluXG5hbGwgY29waWVzIG9yIHN1YnN0YW50aWFsIHBvcnRpb25zIG9mIHRoZSBTb2Z0d2FyZS5cblxuVEhFIFNPRlRXQVJFIElTIFBST1ZJREVEIFwiQVMgSVNcIiwgV0lUSE9VVCBXQVJSQU5UWSBPRiBBTlkgS0lORCwgRVhQUkVTUyBPUlxuSU1QTElFRCwgSU5DTFVESU5HIEJVVCBOT1QgTElNSVRFRCBUTyBUSEUgV0FSUkFOVElFUyBPRiBNRVJDSEFOVEFCSUxJVFksXG5GSVRORVNTIEZPUiBBIFBBUlRJQ1VMQVIgUFVSUE9TRSBBTkQgTk9OSU5GUklOR0VNRU5ULiBJTiBOTyBFVkVOVCBTSEFMTCBUSEVcbkFVVEhPUlMgT1IgQ09QWVJJR0hUIEhPTERFUlMgQkUgTElBQkxFIEZPUiBBTlkgQ0xBSU0sIERBTUFHRVMgT1IgT1RIRVJcbkxJQUJJTElUWSwgV0hFVEhFUiBJTiBBTiBBQ1RJT04gT0YgQ09OVFJBQ1QsIFRPUlQgT1IgT1RIRVJXSVNFLCBBUklTSU5HIEZST00sXG5PVVQgT0YgT1IgSU4gQ09OTkVDVElPTiBXSVRIIFRIRSBTT0ZUV0FSRSBPUiBUSEUgVVNFIE9SIE9USEVSIERFQUxJTkdTIElOXG5USEUgU09GVFdBUkVcblxuXG5cbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFNZTkFQVElDXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxuXG5TeW5hcHRpYyBpcyBhIGphdmFzY3JpcHQgbmV1cmFsIG5ldHdvcmsgbGlicmFyeSBmb3Igbm9kZS5qcyBhbmQgdGhlIGJyb3dzZXIsIGl0cyBnZW5lcmFsaXplZFxuYWxnb3JpdGhtIGlzIGFyY2hpdGVjdHVyZS1mcmVlLCBzbyB5b3UgY2FuIGJ1aWxkIGFuZCB0cmFpbiBiYXNpY2FsbHkgYW55IHR5cGUgb2YgZmlyc3Qgb3JkZXJcbm9yIGV2ZW4gc2Vjb25kIG9yZGVyIG5ldXJhbCBuZXR3b3JrIGFyY2hpdGVjdHVyZXMuXG5cbmh0dHA6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvUmVjdXJyZW50X25ldXJhbF9uZXR3b3JrI1NlY29uZF9PcmRlcl9SZWN1cnJlbnRfTmV1cmFsX05ldHdvcmtcblxuVGhlIGxpYnJhcnkgaW5jbHVkZXMgYSBmZXcgYnVpbHQtaW4gYXJjaGl0ZWN0dXJlcyBsaWtlIG11bHRpbGF5ZXIgcGVyY2VwdHJvbnMsIG11bHRpbGF5ZXJcbmxvbmctc2hvcnQgdGVybSBtZW1vcnkgbmV0d29ya3MgKExTVE0pIG9yIGxpcXVpZCBzdGF0ZSBtYWNoaW5lcywgYW5kIGEgdHJhaW5lciBjYXBhYmxlIG9mXG50cmFpbmluZyBhbnkgZ2l2ZW4gbmV0d29yaywgYW5kIGluY2x1ZGVzIGJ1aWx0LWluIHRyYWluaW5nIHRhc2tzL3Rlc3RzIGxpa2Ugc29sdmluZyBhbiBYT1IsXG5wYXNzaW5nIGEgRGlzdHJhY3RlZCBTZXF1ZW5jZSBSZWNhbGwgdGVzdCBvciBhbiBFbWJlZGVkIFJlYmVyIEdyYW1tYXIgdGVzdC5cblxuVGhlIGFsZ29yaXRobSBpbXBsZW1lbnRlZCBieSB0aGlzIGxpYnJhcnkgaGFzIGJlZW4gdGFrZW4gZnJvbSBEZXJlayBELiBNb25uZXIncyBwYXBlcjpcblxuQSBnZW5lcmFsaXplZCBMU1RNLWxpa2UgdHJhaW5pbmcgYWxnb3JpdGhtIGZvciBzZWNvbmQtb3JkZXIgcmVjdXJyZW50IG5ldXJhbCBuZXR3b3Jrc1xuaHR0cDovL3d3dy5vdmVyY29tcGxldGUubmV0L3BhcGVycy9ubjIwMTIucGRmXG5cblRoZXJlIGFyZSByZWZlcmVuY2VzIHRvIHRoZSBlcXVhdGlvbnMgaW4gdGhhdCBwYXBlciBjb21tZW50ZWQgdGhyb3VnaCB0aGUgc291cmNlIGNvZGUuXG5cblxuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cbnZhciBTeW5hcHRpYyA9IHtcbiAgICBOZXVyb246IHJlcXVpcmUoJy4vbmV1cm9uJyksXG4gICAgTGF5ZXI6IHJlcXVpcmUoJy4vbGF5ZXInKSxcbiAgICBOZXR3b3JrOiByZXF1aXJlKCcuL25ldHdvcmsnKSxcbiAgICBUcmFpbmVyOiByZXF1aXJlKCcuL3RyYWluZXInKSxcbiAgICBBcmNoaXRlY3Q6IHJlcXVpcmUoJy4vYXJjaGl0ZWN0Jylcbn07XG5cbi8vIENvbW1vbkpTICYgQU1EXG5pZiAodHlwZW9mIGRlZmluZSAhPT0gJ3VuZGVmaW5lZCcgJiYgZGVmaW5lLmFtZClcbntcbiAgZGVmaW5lKFtdLCBmdW5jdGlvbigpeyByZXR1cm4gU3luYXB0aWMgfSk7XG59XG5cbi8vIE5vZGUuanNcbmlmICh0eXBlb2YgbW9kdWxlICE9PSAndW5kZWZpbmVkJyAmJiBtb2R1bGUuZXhwb3J0cylcbntcbiAgbW9kdWxlLmV4cG9ydHMgPSBTeW5hcHRpYztcbn1cblxuLy8gQnJvd3NlclxuaWYgKHR5cGVvZiB3aW5kb3cgPT0gJ29iamVjdCcpXG57XG4gIChmdW5jdGlvbigpeyBcbiAgICB2YXIgb2xkU3luYXB0aWMgPSB3aW5kb3dbJ3N5bmFwdGljJ107XG4gICAgU3luYXB0aWMubmluamEgPSBmdW5jdGlvbigpeyBcbiAgICAgIHdpbmRvd1snc3luYXB0aWMnXSA9IG9sZFN5bmFwdGljOyBcbiAgICAgIHJldHVybiBTeW5hcHRpYztcbiAgICB9O1x0XG4gIH0pKCk7XG5cbiAgd2luZG93WydzeW5hcHRpYyddID0gU3luYXB0aWM7XG59XG4iLCIvLyBpbXBvcnRcbnZhciBMYXllciA9IHJlcXVpcmUoJy4vbGF5ZXInKSxcbiAgICBOZXR3b3JrID0gcmVxdWlyZSgnLi9uZXR3b3JrJyksXG4gICAgVHJhaW5lciA9IHJlcXVpcmUoJy4vdHJhaW5lcicpO1xuXG4vKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIEFSQ0hJVEVDVFxuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKi9cblxuLy8gQ29sZWN0aW9uIG9mIHVzZWZ1bCBidWlsdC1pbiBhcmNoaXRlY3R1cmVzXG52YXIgQXJjaGl0ZWN0ID0ge1xuXG4gIC8vIE11bHRpbGF5ZXIgUGVyY2VwdHJvblxuICBQZXJjZXB0cm9uOiBmdW5jdGlvbiBQZXJjZXB0cm9uKCkge1xuXG4gICAgdmFyIGFyZ3MgPSBBcnJheS5wcm90b3R5cGUuc2xpY2UuY2FsbChhcmd1bWVudHMpOyAvLyBjb252ZXJ0IGFyZ3VtZW50cyB0byBBcnJheVxuICAgIGlmIChhcmdzLmxlbmd0aCA8IDMpXG4gICAgICB0aHJvdyBcIkVycm9yOiBub3QgZW5vdWdoIGxheWVycyAobWluaW11bSAzKSAhIVwiO1xuXG4gICAgdmFyIGlucHV0cyA9IGFyZ3Muc2hpZnQoKTsgLy8gZmlyc3QgYXJndW1lbnRcbiAgICB2YXIgb3V0cHV0cyA9IGFyZ3MucG9wKCk7IC8vIGxhc3QgYXJndW1lbnRcbiAgICB2YXIgbGF5ZXJzID0gYXJnczsgLy8gYWxsIHRoZSBhcmd1bWVudHMgaW4gdGhlIG1pZGRsZVxuXG4gICAgdmFyIGlucHV0ID0gbmV3IExheWVyKGlucHV0cyk7XG4gICAgdmFyIGhpZGRlbiA9IFtdO1xuICAgIHZhciBvdXRwdXQgPSBuZXcgTGF5ZXIob3V0cHV0cyk7XG5cbiAgICB2YXIgcHJldmlvdXMgPSBpbnB1dDtcblxuICAgIC8vIGdlbmVyYXRlIGhpZGRlbiBsYXllcnNcbiAgICBmb3IgKGxldmVsIGluIGxheWVycykge1xuICAgICAgdmFyIHNpemUgPSBsYXllcnNbbGV2ZWxdO1xuICAgICAgdmFyIGxheWVyID0gbmV3IExheWVyKHNpemUpO1xuICAgICAgaGlkZGVuLnB1c2gobGF5ZXIpO1xuICAgICAgcHJldmlvdXMucHJvamVjdChsYXllcik7XG4gICAgICBwcmV2aW91cyA9IGxheWVyO1xuICAgIH1cbiAgICBwcmV2aW91cy5wcm9qZWN0KG91dHB1dCk7XG5cbiAgICAvLyBzZXQgbGF5ZXJzIG9mIHRoZSBuZXVyYWwgbmV0d29ya1xuICAgIHRoaXMuc2V0KHtcbiAgICAgIGlucHV0OiBpbnB1dCxcbiAgICAgIGhpZGRlbjogaGlkZGVuLFxuICAgICAgb3V0cHV0OiBvdXRwdXRcbiAgICB9KTtcblxuICAgIC8vIHRyYWluZXIgZm9yIHRoZSBuZXR3b3JrXG4gICAgdGhpcy50cmFpbmVyID0gbmV3IFRyYWluZXIodGhpcyk7XG4gIH0sXG5cbiAgLy8gTXVsdGlsYXllciBMb25nIFNob3J0LVRlcm0gTWVtb3J5XG4gIExTVE06IGZ1bmN0aW9uIExTVE0oKSB7XG5cbiAgICB2YXIgYXJncyA9IEFycmF5LnByb3RvdHlwZS5zbGljZS5jYWxsKGFyZ3VtZW50cyk7IC8vIGNvbnZlcnQgYXJndW1lbnRzIHRvIGFycmF5XG4gICAgaWYgKGFyZ3MubGVuZ3RoIDwgMylcbiAgICAgIHRocm93IFwiRXJyb3I6IG5vdCBlbm91Z2ggbGF5ZXJzIChtaW5pbXVtIDMpICEhXCI7XG5cbiAgICB2YXIgbGFzdCA9IGFyZ3MucG9wKCk7XG4gICAgdmFyIG9wdGlvbiA9IHtcbiAgICAgIHBlZXBob2xlczogTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTCxcbiAgICAgIGhpZGRlbnRvaGlkZGVuOiBmYWxzZSxcbiAgICAgIG91dHRvaGlkZGVuOiBmYWxzZSxcbiAgICAgIG91dHRvZ2F0ZXM6IGZhbHNlLFxuICAgICAgaW50b291dDogdHJ1ZSxcbiAgICB9O1xuICAgIGlmICh0eXBlb2YgbGFzdCAhPSAnbnVtYmVyJykge1xuICAgICAgdmFyIG91dHB1dHMgPSBhcmdzLnBvcCgpO1xuICAgICAgaWYgKGxhc3QuaGFzT3duUHJvcGVydHkoJ3BlZXBob2xlcycpKVxuICAgICAgICBvcHRpb24ucGVlcGhvbGVzID0gbGFzdC5wZWVwaG9sZXM7XG4gICAgICBpZiAobGFzdC5oYXNPd25Qcm9wZXJ0eSgnaGlkZGVudG9oaWRkZW4nKSlcbiAgICAgICAgb3B0aW9uLmhpZGRlbnRvaGlkZGVuID0gbGFzdC5oaWRkZW50b2hpZGRlbjtcbiAgICAgIGlmIChsYXN0Lmhhc093blByb3BlcnR5KCdvdXR0b2hpZGRlbicpKVxuICAgICAgICBvcHRpb24ub3V0dG9oaWRkZW4gPSBsYXN0Lm91dHRvaGlkZGVuO1xuICAgICAgaWYgKGxhc3QuaGFzT3duUHJvcGVydHkoJ291dHRvZ2F0ZXMnKSlcbiAgICAgICAgb3B0aW9uLm91dHRvZ2F0ZXMgPSBsYXN0Lm91dHRvZ2F0ZXM7XG4gICAgICBpZiAobGFzdC5oYXNPd25Qcm9wZXJ0eSgnaW50b291dCcpKVxuICAgICAgICBvcHRpb24uaW50b291dCA9IGxhc3QuaW50b291dDtcbiAgICB9IGVsc2VcbiAgICAgIHZhciBvdXRwdXRzID0gbGFzdDtcblxuICAgIHZhciBpbnB1dHMgPSBhcmdzLnNoaWZ0KCk7XG4gICAgdmFyIGxheWVycyA9IGFyZ3M7XG5cbiAgICB2YXIgaW5wdXRMYXllciA9IG5ldyBMYXllcihpbnB1dHMpO1xuICAgIHZhciBoaWRkZW5MYXllcnMgPSBbXTtcbiAgICB2YXIgb3V0cHV0TGF5ZXIgPSBuZXcgTGF5ZXIob3V0cHV0cyk7XG5cbiAgICB2YXIgcHJldmlvdXMgPSBudWxsO1xuXG4gICAgLy8gZ2VuZXJhdGUgbGF5ZXJzXG4gICAgZm9yICh2YXIgbGF5ZXIgaW4gbGF5ZXJzKSB7XG4gICAgICAvLyBnZW5lcmF0ZSBtZW1vcnkgYmxvY2tzIChtZW1vcnkgY2VsbCBhbmQgcmVzcGVjdGl2ZSBnYXRlcylcbiAgICAgIHZhciBzaXplID0gbGF5ZXJzW2xheWVyXTtcblxuICAgICAgdmFyIGlucHV0R2F0ZSA9IG5ldyBMYXllcihzaXplKS5zZXQoe1xuICAgICAgICBiaWFzOiAxXG4gICAgICB9KTtcbiAgICAgIHZhciBmb3JnZXRHYXRlID0gbmV3IExheWVyKHNpemUpLnNldCh7XG4gICAgICAgIGJpYXM6IDFcbiAgICAgIH0pO1xuICAgICAgdmFyIG1lbW9yeUNlbGwgPSBuZXcgTGF5ZXIoc2l6ZSk7XG4gICAgICB2YXIgb3V0cHV0R2F0ZSA9IG5ldyBMYXllcihzaXplKS5zZXQoe1xuICAgICAgICBiaWFzOiAxXG4gICAgICB9KTtcblxuICAgICAgaGlkZGVuTGF5ZXJzLnB1c2goaW5wdXRHYXRlKTtcbiAgICAgIGhpZGRlbkxheWVycy5wdXNoKGZvcmdldEdhdGUpO1xuICAgICAgaGlkZGVuTGF5ZXJzLnB1c2gobWVtb3J5Q2VsbCk7XG4gICAgICBoaWRkZW5MYXllcnMucHVzaChvdXRwdXRHYXRlKTtcblxuICAgICAgLy8gY29ubmVjdGlvbnMgZnJvbSBpbnB1dCBsYXllclxuICAgICAgdmFyIGlucHV0ID0gaW5wdXRMYXllci5wcm9qZWN0KG1lbW9yeUNlbGwpO1xuICAgICAgaW5wdXRMYXllci5wcm9qZWN0KGlucHV0R2F0ZSk7XG4gICAgICBpbnB1dExheWVyLnByb2plY3QoZm9yZ2V0R2F0ZSk7XG4gICAgICBpbnB1dExheWVyLnByb2plY3Qob3V0cHV0R2F0ZSk7XG5cbiAgICAgIC8vIGNvbm5lY3Rpb25zIGZyb20gcHJldmlvdXMgbWVtb3J5LWJsb2NrIGxheWVyIHRvIHRoaXMgb25lXG4gICAgICBpZiAocHJldmlvdXMgIT0gbnVsbCkge1xuICAgICAgICB2YXIgY2VsbCA9IHByZXZpb3VzLnByb2plY3QobWVtb3J5Q2VsbCk7XG4gICAgICAgIHByZXZpb3VzLnByb2plY3QoaW5wdXRHYXRlKTtcbiAgICAgICAgcHJldmlvdXMucHJvamVjdChmb3JnZXRHYXRlKTtcbiAgICAgICAgcHJldmlvdXMucHJvamVjdChvdXRwdXRHYXRlKTtcbiAgICAgIH1cblxuICAgICAgLy8gY29ubmVjdGlvbnMgZnJvbSBtZW1vcnkgY2VsbFxuICAgICAgdmFyIG91dHB1dCA9IG1lbW9yeUNlbGwucHJvamVjdChvdXRwdXRMYXllcik7XG5cbiAgICAgIC8vIHNlbGYtY29ubmVjdGlvblxuICAgICAgdmFyIHNlbGYgPSBtZW1vcnlDZWxsLnByb2plY3QobWVtb3J5Q2VsbCk7XG5cbiAgICAgIC8vIGhpZGRlbiB0byBoaWRkZW4gcmVjdXJyZW50IGNvbm5lY3Rpb25cbiAgICAgIGlmIChvcHRpb24uaGlkZGVudG9oaWRkZW4pXG4gICAgICAgIG1lbW9yeUNlbGwucHJvamVjdChtZW1vcnlDZWxsLCBMYXllci5jb25uZWN0aW9uVHlwZS5BTExfVE9fRUxTRSk7XG5cbiAgICAgIC8vIG91dCB0byBoaWRkZW4gcmVjdXJyZW50IGNvbm5lY3Rpb25cbiAgICAgIGlmIChvcHRpb24ub3V0dG9oaWRkZW4pXG4gICAgICAgIG91dHB1dExheWVyLnByb2plY3QobWVtb3J5Q2VsbCk7XG5cbiAgICAgIC8vIG91dCB0byBnYXRlcyByZWN1cnJlbnQgY29ubmVjdGlvblxuICAgICAgaWYgKG9wdGlvbi5vdXR0b2dhdGVzKSB7XG4gICAgICAgIG91dHB1dExheWVyLnByb2plY3QoaW5wdXRHYXRlKTtcbiAgICAgICAgb3V0cHV0TGF5ZXIucHJvamVjdChvdXRwdXRHYXRlKTtcbiAgICAgICAgb3V0cHV0TGF5ZXIucHJvamVjdChmb3JnZXRHYXRlKTtcbiAgICAgIH1cbiAgICAgIFxuICAgICAgLy8gcGVlcGhvbGVzXG4gICAgICBtZW1vcnlDZWxsLnByb2plY3QoaW5wdXRHYXRlLCBvcHRpb24ucGVlcGhvbGVzKTtcbiAgICAgIG1lbW9yeUNlbGwucHJvamVjdChmb3JnZXRHYXRlLCBvcHRpb24ucGVlcGhvbGVzKTtcbiAgICAgIG1lbW9yeUNlbGwucHJvamVjdChvdXRwdXRHYXRlLCBvcHRpb24ucGVlcGhvbGVzKTtcblxuICAgICAgLy8gZ2F0ZXNcbiAgICAgIGlucHV0R2F0ZS5nYXRlKGlucHV0LCBMYXllci5nYXRlVHlwZS5JTlBVVCk7XG4gICAgICBmb3JnZXRHYXRlLmdhdGUoc2VsZiwgTGF5ZXIuZ2F0ZVR5cGUuT05FX1RPX09ORSk7XG4gICAgICBvdXRwdXRHYXRlLmdhdGUob3V0cHV0LCBMYXllci5nYXRlVHlwZS5PVVRQVVQpO1xuICAgICAgaWYgKHByZXZpb3VzICE9IG51bGwpXG4gICAgICAgIGlucHV0R2F0ZS5nYXRlKGNlbGwsIExheWVyLmdhdGVUeXBlLklOUFVUKTtcblxuICAgICAgcHJldmlvdXMgPSBtZW1vcnlDZWxsO1xuICAgIH1cblxuICAgIC8vIGlucHV0IHRvIG91dHB1dCBkaXJlY3QgY29ubmVjdGlvblxuICAgIGlmIChvcHRpb24uaW50b291dClcbiAgICAgIGlucHV0TGF5ZXIucHJvamVjdChvdXRwdXRMYXllcik7XG5cbiAgICAvLyBzZXQgdGhlIGxheWVycyBvZiB0aGUgbmV1cmFsIG5ldHdvcmtcbiAgICB0aGlzLnNldCh7XG4gICAgICBpbnB1dDogaW5wdXRMYXllcixcbiAgICAgIGhpZGRlbjogaGlkZGVuTGF5ZXJzLFxuICAgICAgb3V0cHV0OiBvdXRwdXRMYXllclxuICAgIH0pO1xuXG4gICAgLy8gdHJhaW5lclxuICAgIHRoaXMudHJhaW5lciA9IG5ldyBUcmFpbmVyKHRoaXMpO1xuICB9LFxuXG4gIC8vIExpcXVpZCBTdGF0ZSBNYWNoaW5lXG4gIExpcXVpZDogZnVuY3Rpb24gTGlxdWlkKGlucHV0cywgaGlkZGVuLCBvdXRwdXRzLCBjb25uZWN0aW9ucywgZ2F0ZXMpIHtcblxuICAgIC8vIGNyZWF0ZSBsYXllcnNcbiAgICB2YXIgaW5wdXRMYXllciA9IG5ldyBMYXllcihpbnB1dHMpO1xuICAgIHZhciBoaWRkZW5MYXllciA9IG5ldyBMYXllcihoaWRkZW4pO1xuICAgIHZhciBvdXRwdXRMYXllciA9IG5ldyBMYXllcihvdXRwdXRzKTtcblxuICAgIC8vIG1ha2UgY29ubmVjdGlvbnMgYW5kIGdhdGVzIHJhbmRvbWx5IGFtb25nIHRoZSBuZXVyb25zXG4gICAgdmFyIG5ldXJvbnMgPSBoaWRkZW5MYXllci5uZXVyb25zKCk7XG4gICAgdmFyIGNvbm5lY3Rpb25MaXN0ID0gW107XG5cbiAgICBmb3IgKHZhciBpID0gMDsgaSA8IGNvbm5lY3Rpb25zOyBpKyspIHtcbiAgICAgIC8vIGNvbm5lY3QgdHdvIHJhbmRvbSBuZXVyb25zXG4gICAgICB2YXIgZnJvbSA9IE1hdGgucmFuZG9tKCkgKiBuZXVyb25zLmxlbmd0aCB8IDA7XG4gICAgICB2YXIgdG8gPSBNYXRoLnJhbmRvbSgpICogbmV1cm9ucy5sZW5ndGggfCAwO1xuICAgICAgdmFyIGNvbm5lY3Rpb24gPSBuZXVyb25zW2Zyb21dLnByb2plY3QobmV1cm9uc1t0b10pO1xuICAgICAgY29ubmVjdGlvbkxpc3QucHVzaChjb25uZWN0aW9uKTtcbiAgICB9XG5cbiAgICBmb3IgKHZhciBqID0gMDsgaiA8IGdhdGVzOyBqKyspIHtcbiAgICAgIC8vIHBpY2sgYSByYW5kb20gZ2F0ZXIgbmV1cm9uXG4gICAgICB2YXIgZ2F0ZXIgPSBNYXRoLnJhbmRvbSgpICogbmV1cm9ucy5sZW5ndGggfCAwO1xuICAgICAgLy8gcGljayBhIHJhbmRvbSBjb25uZWN0aW9uIHRvIGdhdGVcbiAgICAgIHZhciBjb25uZWN0aW9uID0gTWF0aC5yYW5kb20oKSAqIGNvbm5lY3Rpb25MaXN0Lmxlbmd0aCB8IDA7XG4gICAgICAvLyBsZXQgdGhlIGdhdGVyIGdhdGUgdGhlIGNvbm5lY3Rpb25cbiAgICAgIG5ldXJvbnNbZ2F0ZXJdLmdhdGUoY29ubmVjdGlvbkxpc3RbY29ubmVjdGlvbl0pO1xuICAgIH1cblxuICAgIC8vIGNvbm5lY3QgdGhlIGxheWVyc1xuICAgIGlucHV0TGF5ZXIucHJvamVjdChoaWRkZW5MYXllcik7XG4gICAgaGlkZGVuTGF5ZXIucHJvamVjdChvdXRwdXRMYXllcik7XG5cbiAgICAvLyBzZXQgdGhlIGxheWVycyBvZiB0aGUgbmV0d29ya1xuICAgIHRoaXMuc2V0KHtcbiAgICAgIGlucHV0OiBpbnB1dExheWVyLFxuICAgICAgaGlkZGVuOiBbaGlkZGVuTGF5ZXJdLFxuICAgICAgb3V0cHV0OiBvdXRwdXRMYXllclxuICAgIH0pO1xuXG4gICAgLy8gdHJhaW5lclxuICAgIHRoaXMudHJhaW5lciA9IG5ldyBUcmFpbmVyKHRoaXMpO1xuICB9LFxuXG4gIEhvcGZpZWxkOiBmdW5jdGlvbiBIb3BmaWVsZChzaXplKVxuICB7XG4gICAgdmFyIGlucHV0TGF5ZXIgPSBuZXcgTGF5ZXIoc2l6ZSk7XG4gICAgdmFyIG91dHB1dExheWVyID0gbmV3IExheWVyKHNpemUpO1xuXG4gICAgaW5wdXRMYXllci5wcm9qZWN0KG91dHB1dExheWVyLCBMYXllci5jb25uZWN0aW9uVHlwZS5BTExfVE9fQUxMKTtcblxuICAgIHRoaXMuc2V0KHtcbiAgICAgIGlucHV0OiBpbnB1dExheWVyLFxuICAgICAgaGlkZGVuOiBbXSxcbiAgICAgIG91dHB1dDogb3V0cHV0TGF5ZXJcbiAgICB9KTtcblxuICAgIHZhciB0cmFpbmVyID0gbmV3IFRyYWluZXIodGhpcyk7XG5cbiAgICB2YXIgcHJvdG8gPSBBcmNoaXRlY3QuSG9wZmllbGQucHJvdG90eXBlO1xuXG4gICAgcHJvdG8ubGVhcm4gPSBwcm90by5sZWFybiB8fCBmdW5jdGlvbihwYXR0ZXJucylcbiAgICB7XG4gICAgICB2YXIgc2V0ID0gW107XG4gICAgICBmb3IgKHZhciBwIGluIHBhdHRlcm5zKVxuICAgICAgICBzZXQucHVzaCh7XG4gICAgICAgICAgaW5wdXQ6IHBhdHRlcm5zW3BdLFxuICAgICAgICAgIG91dHB1dDogcGF0dGVybnNbcF1cbiAgICAgICAgfSk7XG5cbiAgICAgIHJldHVybiB0cmFpbmVyLnRyYWluKHNldCwge1xuICAgICAgICBpdGVyYXRpb25zOiA1MDAwMDAsXG4gICAgICAgIGVycm9yOiAuMDAwMDUsXG4gICAgICAgIHJhdGU6IDFcbiAgICAgIH0pO1xuICAgIH1cblxuICAgIHByb3RvLmZlZWQgPSBwcm90by5mZWVkIHx8IGZ1bmN0aW9uKHBhdHRlcm4pXG4gICAge1xuICAgICAgdmFyIG91dHB1dCA9IHRoaXMuYWN0aXZhdGUocGF0dGVybik7XG5cbiAgICAgIHZhciBwYXR0ZXJuID0gW107XG4gICAgICBmb3IgKHZhciBpIGluIG91dHB1dClcbiAgICAgICAgcGF0dGVybltpXSA9IG91dHB1dFtpXSA+IC41ID8gMSA6IDA7XG5cbiAgICAgIHJldHVybiBwYXR0ZXJuO1xuICAgIH1cbiAgfVxufVxuXG4vLyBFeHRlbmQgcHJvdG90eXBlIGNoYWluIChzbyBldmVyeSBhcmNoaXRlY3R1cmVzIGlzIGFuIGluc3RhbmNlIG9mIE5ldHdvcmspXG5mb3IgKHZhciBhcmNoaXRlY3R1cmUgaW4gQXJjaGl0ZWN0KSB7XG4gIEFyY2hpdGVjdFthcmNoaXRlY3R1cmVdLnByb3RvdHlwZSA9IG5ldyBOZXR3b3JrKCk7XG4gIEFyY2hpdGVjdFthcmNoaXRlY3R1cmVdLnByb3RvdHlwZS5jb25zdHJ1Y3RvciA9IEFyY2hpdGVjdFthcmNoaXRlY3R1cmVdO1xufVxuXG4vLyBleHBvcnRcbmlmIChtb2R1bGUpIG1vZHVsZS5leHBvcnRzID0gQXJjaGl0ZWN0OyBcblxuIiwiLy8gaW1wb3J0XG52YXIgTmV1cm9uID0gcmVxdWlyZSgnLi9uZXVyb24nKTtcblxuLyoqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgTEFZRVJcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cbmZ1bmN0aW9uIExheWVyKHNpemUsIGxhYmVsKSB7XG4gIHRoaXMuc2l6ZSA9IHNpemUgfCAwO1xuICB0aGlzLmxpc3QgPSBbXTtcbiAgdGhpcy5sYWJlbCA9IGxhYmVsIHx8IG51bGw7XG4gIHRoaXMuY29ubmVjdGVkdG8gPSBbXTtcblxuICB3aGlsZSAoc2l6ZS0tKSB7XG4gICAgdmFyIG5ldXJvbiA9IG5ldyBOZXVyb24oKTtcbiAgICB0aGlzLmxpc3QucHVzaChuZXVyb24pO1xuICB9XG59XG5cbkxheWVyLnByb3RvdHlwZSA9IHtcblxuICAvLyBhY3RpdmF0ZXMgYWxsIHRoZSBuZXVyb25zIGluIHRoZSBsYXllclxuICBhY3RpdmF0ZTogZnVuY3Rpb24oaW5wdXQpIHtcblxuICAgIHZhciBhY3RpdmF0aW9ucyA9IFtdO1xuXG4gICAgaWYgKHR5cGVvZiBpbnB1dCAhPSAndW5kZWZpbmVkJykge1xuICAgICAgaWYgKGlucHV0Lmxlbmd0aCAhPSB0aGlzLnNpemUpXG4gICAgICAgIHRocm93IFwiSU5QVVQgc2l6ZSBhbmQgTEFZRVIgc2l6ZSBtdXN0IGJlIHRoZSBzYW1lIHRvIGFjdGl2YXRlIVwiO1xuXG4gICAgICBmb3IgKHZhciBpZCBpbiB0aGlzLmxpc3QpIHtcbiAgICAgICAgdmFyIG5ldXJvbiA9IHRoaXMubGlzdFtpZF07XG4gICAgICAgIHZhciBhY3RpdmF0aW9uID0gbmV1cm9uLmFjdGl2YXRlKGlucHV0W2lkXSk7XG4gICAgICAgIGFjdGl2YXRpb25zLnB1c2goYWN0aXZhdGlvbik7XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIGZvciAodmFyIGlkIGluIHRoaXMubGlzdCkge1xuICAgICAgICB2YXIgbmV1cm9uID0gdGhpcy5saXN0W2lkXTtcbiAgICAgICAgdmFyIGFjdGl2YXRpb24gPSBuZXVyb24uYWN0aXZhdGUoKTtcbiAgICAgICAgYWN0aXZhdGlvbnMucHVzaChhY3RpdmF0aW9uKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGFjdGl2YXRpb25zO1xuICB9LFxuXG4gIC8vIHByb3BhZ2F0ZXMgdGhlIGVycm9yIG9uIGFsbCB0aGUgbmV1cm9ucyBvZiB0aGUgbGF5ZXJcbiAgcHJvcGFnYXRlOiBmdW5jdGlvbihyYXRlLCB0YXJnZXQpIHtcblxuICAgIGlmICh0eXBlb2YgdGFyZ2V0ICE9ICd1bmRlZmluZWQnKSB7XG4gICAgICBpZiAodGFyZ2V0Lmxlbmd0aCAhPSB0aGlzLnNpemUpXG4gICAgICAgIHRocm93IFwiVEFSR0VUIHNpemUgYW5kIExBWUVSIHNpemUgbXVzdCBiZSB0aGUgc2FtZSB0byBwcm9wYWdhdGUhXCI7XG5cbiAgICAgIGZvciAodmFyIGlkID0gdGhpcy5saXN0Lmxlbmd0aCAtIDE7IGlkID49IDA7IGlkLS0pIHtcbiAgICAgICAgdmFyIG5ldXJvbiA9IHRoaXMubGlzdFtpZF07XG4gICAgICAgIG5ldXJvbi5wcm9wYWdhdGUocmF0ZSwgdGFyZ2V0W2lkXSk7XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIGZvciAodmFyIGlkID0gdGhpcy5saXN0Lmxlbmd0aCAtIDE7IGlkID49IDA7IGlkLS0pIHtcbiAgICAgICAgdmFyIG5ldXJvbiA9IHRoaXMubGlzdFtpZF07XG4gICAgICAgIG5ldXJvbi5wcm9wYWdhdGUocmF0ZSk7XG4gICAgICB9XG4gICAgfVxuICB9LFxuXG4gIC8vIHByb2plY3RzIGEgY29ubmVjdGlvbiBmcm9tIHRoaXMgbGF5ZXIgdG8gYW5vdGhlciBvbmVcbiAgcHJvamVjdDogZnVuY3Rpb24obGF5ZXIsIHR5cGUsIHdlaWdodHMpIHtcblxuICAgIGlmIChsYXllciBpbnN0YW5jZW9mIHJlcXVpcmUoJy4vbmV0d29yaycpKVxuICAgICAgbGF5ZXIgPSBsYXllci5sYXllcnMuaW5wdXQ7XG5cbiAgICBpZiAobGF5ZXIgaW5zdGFuY2VvZiBMYXllcikge1xuICAgICAgaWYgKCF0aGlzLmNvbm5lY3RlZChsYXllcikpXG4gICAgICAgIHJldHVybiBuZXcgTGF5ZXIuY29ubmVjdGlvbih0aGlzLCBsYXllciwgdHlwZSwgd2VpZ2h0cyk7XG4gICAgfSBlbHNlXG4gICAgICB0aHJvdyBcIkludmFsaWQgYXJndW1lbnQsIHlvdSBjYW4gb25seSBwcm9qZWN0IGNvbm5lY3Rpb25zIHRvIExBWUVSUyBhbmQgTkVUV09SS1MhXCI7XG5cblxuICB9LFxuXG4gIC8vIGdhdGVzIGEgY29ubmVjdGlvbiBiZXR3ZW5uIHR3byBsYXllcnNcbiAgZ2F0ZTogZnVuY3Rpb24oY29ubmVjdGlvbiwgdHlwZSkge1xuXG4gICAgaWYgKHR5cGUgPT0gTGF5ZXIuZ2F0ZVR5cGUuSU5QVVQpIHtcbiAgICAgIGlmIChjb25uZWN0aW9uLnRvLnNpemUgIT0gdGhpcy5zaXplKVxuICAgICAgICB0aHJvdyBcIkdBVEVSIGxheWVyIGFuZCBDT05ORUNUSU9OLlRPIGxheWVyIG11c3QgYmUgdGhlIHNhbWUgc2l6ZSBpbiBvcmRlciB0byBnYXRlIVwiO1xuXG4gICAgICBmb3IgKHZhciBpZCBpbiBjb25uZWN0aW9uLnRvLmxpc3QpIHtcbiAgICAgICAgdmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24udG8ubGlzdFtpZF07XG4gICAgICAgIHZhciBnYXRlciA9IHRoaXMubGlzdFtpZF07XG4gICAgICAgIGZvciAodmFyIGlucHV0IGluIG5ldXJvbi5jb25uZWN0aW9ucy5pbnB1dHMpIHtcbiAgICAgICAgICB2YXIgZ2F0ZWQgPSBuZXVyb24uY29ubmVjdGlvbnMuaW5wdXRzW2lucHV0XTtcbiAgICAgICAgICBpZiAoZ2F0ZWQuSUQgaW4gY29ubmVjdGlvbi5jb25uZWN0aW9ucylcbiAgICAgICAgICAgIGdhdGVyLmdhdGUoZ2F0ZWQpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfSBlbHNlIGlmICh0eXBlID09IExheWVyLmdhdGVUeXBlLk9VVFBVVCkge1xuICAgICAgaWYgKGNvbm5lY3Rpb24uZnJvbS5zaXplICE9IHRoaXMuc2l6ZSlcbiAgICAgICAgdGhyb3cgXCJHQVRFUiBsYXllciBhbmQgQ09OTkVDVElPTi5GUk9NIGxheWVyIG11c3QgYmUgdGhlIHNhbWUgc2l6ZSBpbiBvcmRlciB0byBnYXRlIVwiO1xuXG4gICAgICBmb3IgKHZhciBpZCBpbiBjb25uZWN0aW9uLmZyb20ubGlzdCkge1xuICAgICAgICB2YXIgbmV1cm9uID0gY29ubmVjdGlvbi5mcm9tLmxpc3RbaWRdO1xuICAgICAgICB2YXIgZ2F0ZXIgPSB0aGlzLmxpc3RbaWRdO1xuICAgICAgICBmb3IgKHZhciBwcm9qZWN0ZWQgaW4gbmV1cm9uLmNvbm5lY3Rpb25zLnByb2plY3RlZCkge1xuICAgICAgICAgIHZhciBnYXRlZCA9IG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbcHJvamVjdGVkXTtcbiAgICAgICAgICBpZiAoZ2F0ZWQuSUQgaW4gY29ubmVjdGlvbi5jb25uZWN0aW9ucylcbiAgICAgICAgICAgIGdhdGVyLmdhdGUoZ2F0ZWQpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfSBlbHNlIGlmICh0eXBlID09IExheWVyLmdhdGVUeXBlLk9ORV9UT19PTkUpIHtcbiAgICAgIGlmIChjb25uZWN0aW9uLnNpemUgIT0gdGhpcy5zaXplKVxuICAgICAgICB0aHJvdyBcIlRoZSBudW1iZXIgb2YgR0FURVIgVU5JVFMgbXVzdCBiZSB0aGUgc2FtZSBhcyB0aGUgbnVtYmVyIG9mIENPTk5FQ1RJT05TIHRvIGdhdGUhXCI7XG5cbiAgICAgIGZvciAodmFyIGlkIGluIGNvbm5lY3Rpb24ubGlzdCkge1xuICAgICAgICB2YXIgZ2F0ZXIgPSB0aGlzLmxpc3RbaWRdO1xuICAgICAgICB2YXIgZ2F0ZWQgPSBjb25uZWN0aW9uLmxpc3RbaWRdO1xuICAgICAgICBnYXRlci5nYXRlKGdhdGVkKTtcbiAgICAgIH1cbiAgICB9XG4gICAgY29ubmVjdGlvbi5nYXRlZGZyb20ucHVzaCh7bGF5ZXI6IHRoaXMsIHR5cGU6IHR5cGV9KTtcbiAgfSxcblxuICAvLyB0cnVlIG9yIGZhbHNlIHdoZXRoZXIgdGhlIHdob2xlIGxheWVyIGlzIHNlbGYtY29ubmVjdGVkIG9yIG5vdFxuICBzZWxmY29ubmVjdGVkOiBmdW5jdGlvbigpIHtcblxuICAgIGZvciAodmFyIGlkIGluIHRoaXMubGlzdCkge1xuICAgICAgdmFyIG5ldXJvbiA9IHRoaXMubGlzdFtpZF07XG4gICAgICBpZiAoIW5ldXJvbi5zZWxmY29ubmVjdGVkKCkpXG4gICAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gICAgcmV0dXJuIHRydWU7XG4gIH0sXG5cbiAgLy8gdHJ1ZSBvZiBmYWxzZSB3aGV0aGVyIHRoZSBsYXllciBpcyBjb25uZWN0ZWQgdG8gYW5vdGhlciBsYXllciAocGFyYW1ldGVyKSBvciBub3RcbiAgY29ubmVjdGVkOiBmdW5jdGlvbihsYXllcikge1xuICAgIC8vIENoZWNrIGlmIEFMTCB0byBBTEwgY29ubmVjdGlvblxuICAgIHZhciBjb25uZWN0aW9ucyA9IDA7XG4gICAgZm9yICh2YXIgaGVyZSBpbiB0aGlzLmxpc3QpIHtcbiAgICAgIGZvciAodmFyIHRoZXJlIGluIGxheWVyLmxpc3QpIHtcbiAgICAgICAgdmFyIGZyb20gPSB0aGlzLmxpc3RbaGVyZV07XG4gICAgICAgIHZhciB0byA9IGxheWVyLmxpc3RbdGhlcmVdO1xuICAgICAgICB2YXIgY29ubmVjdGVkID0gZnJvbS5jb25uZWN0ZWQodG8pO1xuICAgICAgICBpZiAoY29ubmVjdGVkLnR5cGUgPT0gJ3Byb2plY3RlZCcpXG4gICAgICAgICAgY29ubmVjdGlvbnMrKztcbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKGNvbm5lY3Rpb25zID09IHRoaXMuc2l6ZSAqIGxheWVyLnNpemUpXG4gICAgICByZXR1cm4gTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTDtcblxuICAgIC8vIENoZWNrIGlmIE9ORSB0byBPTkUgY29ubmVjdGlvblxuICAgIGNvbm5lY3Rpb25zID0gMDtcbiAgICBmb3IgKHZhciBuZXVyb24gaW4gdGhpcy5saXN0KSB7XG4gICAgICB2YXIgZnJvbSA9IHRoaXMubGlzdFtuZXVyb25dO1xuICAgICAgdmFyIHRvID0gbGF5ZXIubGlzdFtuZXVyb25dO1xuICAgICAgdmFyIGNvbm5lY3RlZCA9IGZyb20uY29ubmVjdGVkKHRvKTtcbiAgICAgIGlmIChjb25uZWN0ZWQudHlwZSA9PSAncHJvamVjdGVkJylcbiAgICAgICAgY29ubmVjdGlvbnMrKztcbiAgICB9XG4gICAgaWYgKGNvbm5lY3Rpb25zID09IHRoaXMuc2l6ZSlcbiAgICAgIHJldHVybiBMYXllci5jb25uZWN0aW9uVHlwZS5PTkVfVE9fT05FO1xuICB9LFxuXG4gIC8vIGNsZWFycyBhbGwgdGhlIG5ldW9ybnMgaW4gdGhlIGxheWVyXG4gIGNsZWFyOiBmdW5jdGlvbigpIHtcbiAgICBmb3IgKHZhciBpZCBpbiB0aGlzLmxpc3QpIHtcbiAgICAgIHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaWRdO1xuICAgICAgbmV1cm9uLmNsZWFyKCk7XG4gICAgfVxuICB9LFxuXG4gIC8vIHJlc2V0cyBhbGwgdGhlIG5ldXJvbnMgaW4gdGhlIGxheWVyXG4gIHJlc2V0OiBmdW5jdGlvbigpIHtcbiAgICBmb3IgKHZhciBpZCBpbiB0aGlzLmxpc3QpIHtcbiAgICAgIHZhciBuZXVyb24gPSB0aGlzLmxpc3RbaWRdO1xuICAgICAgbmV1cm9uLnJlc2V0KCk7XG4gICAgfVxuICB9LFxuXG4gIC8vIHJldHVybnMgYWxsIHRoZSBuZXVyb25zIGluIHRoZSBsYXllciAoYXJyYXkpXG4gIG5ldXJvbnM6IGZ1bmN0aW9uKCkge1xuICAgIHJldHVybiB0aGlzLmxpc3Q7XG4gIH0sXG5cbiAgLy8gYWRkcyBhIG5ldXJvbiB0byB0aGUgbGF5ZXJcbiAgYWRkOiBmdW5jdGlvbihuZXVyb24pIHtcbiAgICB0aGlzLm5ldXJvbnNbbmV1cm9uLklEXSA9IG5ldXJvbiB8fCBuZXcgTmV1cm9uKCk7XG4gICAgdGhpcy5saXN0LnB1c2gobmV1cm9uKTtcbiAgICB0aGlzLnNpemUrKztcbiAgfSxcblxuICBzZXQ6IGZ1bmN0aW9uKG9wdGlvbnMpIHtcbiAgICBvcHRpb25zID0gb3B0aW9ucyB8fCB7fTtcblxuICAgIGZvciAodmFyIGkgaW4gdGhpcy5saXN0KSB7XG4gICAgICB2YXIgbmV1cm9uID0gdGhpcy5saXN0W2ldO1xuICAgICAgaWYgKG9wdGlvbnMubGFiZWwpXG4gICAgICAgIG5ldXJvbi5sYWJlbCA9IG9wdGlvbnMubGFiZWwgKyAnXycgKyBuZXVyb24uSUQ7XG4gICAgICBpZiAob3B0aW9ucy5zcXVhc2gpXG4gICAgICAgIG5ldXJvbi5zcXVhc2ggPSBvcHRpb25zLnNxdWFzaDtcbiAgICAgIGlmIChvcHRpb25zLmJpYXMpXG4gICAgICAgIG5ldXJvbi5iaWFzID0gb3B0aW9ucy5iaWFzO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcztcbiAgfVxufVxuXG4vLyByZXByZXNlbnRzIGEgY29ubmVjdGlvbiBmcm9tIG9uZSBsYXllciB0byBhbm90aGVyLCBhbmQga2VlcHMgdHJhY2sgb2YgaXRzIHdlaWdodCBhbmQgZ2FpblxuTGF5ZXIuY29ubmVjdGlvbiA9IGZ1bmN0aW9uIExheWVyQ29ubmVjdGlvbihmcm9tTGF5ZXIsIHRvTGF5ZXIsIHR5cGUsIHdlaWdodHMpIHtcbiAgdGhpcy5JRCA9IExheWVyLmNvbm5lY3Rpb24udWlkKCk7XG4gIHRoaXMuZnJvbSA9IGZyb21MYXllcjtcbiAgdGhpcy50byA9IHRvTGF5ZXI7XG4gIHRoaXMuc2VsZmNvbm5lY3Rpb24gPSB0b0xheWVyID09IGZyb21MYXllcjtcbiAgdGhpcy50eXBlID0gdHlwZTtcbiAgdGhpcy5jb25uZWN0aW9ucyA9IHt9O1xuICB0aGlzLmxpc3QgPSBbXTtcbiAgdGhpcy5zaXplID0gMDtcbiAgdGhpcy5nYXRlZGZyb20gPSBbXTtcblxuICBpZiAodHlwZW9mIHRoaXMudHlwZSA9PSAndW5kZWZpbmVkJylcbiAge1xuICAgIGlmIChmcm9tTGF5ZXIgPT0gdG9MYXllcilcbiAgICAgIHRoaXMudHlwZSA9IExheWVyLmNvbm5lY3Rpb25UeXBlLk9ORV9UT19PTkU7XG4gICAgZWxzZVxuICAgICAgdGhpcy50eXBlID0gTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTDtcbiAgfVxuXG4gIGlmICh0aGlzLnR5cGUgPT0gTGF5ZXIuY29ubmVjdGlvblR5cGUuQUxMX1RPX0FMTCB8fFxuICAgICAgdGhpcy50eXBlID09IExheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19FTFNFKSB7XG4gICAgZm9yICh2YXIgaGVyZSBpbiB0aGlzLmZyb20ubGlzdCkge1xuICAgICAgZm9yICh2YXIgdGhlcmUgaW4gdGhpcy50by5saXN0KSB7XG4gICAgICAgIHZhciBmcm9tID0gdGhpcy5mcm9tLmxpc3RbaGVyZV07XG4gICAgICAgIHZhciB0byA9IHRoaXMudG8ubGlzdFt0aGVyZV07XG4gICAgICAgIGlmKHRoaXMudHlwZSA9PSBMYXllci5jb25uZWN0aW9uVHlwZS5BTExfVE9fRUxTRSAmJiBmcm9tID09IHRvKVxuICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICB2YXIgY29ubmVjdGlvbiA9IGZyb20ucHJvamVjdCh0bywgd2VpZ2h0cyk7XG5cbiAgICAgICAgdGhpcy5jb25uZWN0aW9uc1tjb25uZWN0aW9uLklEXSA9IGNvbm5lY3Rpb247XG4gICAgICAgIHRoaXMuc2l6ZSA9IHRoaXMubGlzdC5wdXNoKGNvbm5lY3Rpb24pO1xuICAgICAgfVxuICAgIH1cbiAgfSBlbHNlIGlmICh0aGlzLnR5cGUgPT0gTGF5ZXIuY29ubmVjdGlvblR5cGUuT05FX1RPX09ORSkge1xuXG4gICAgZm9yICh2YXIgbmV1cm9uIGluIHRoaXMuZnJvbS5saXN0KSB7XG4gICAgICB2YXIgZnJvbSA9IHRoaXMuZnJvbS5saXN0W25ldXJvbl07XG4gICAgICB2YXIgdG8gPSB0aGlzLnRvLmxpc3RbbmV1cm9uXTtcbiAgICAgIHZhciBjb25uZWN0aW9uID0gZnJvbS5wcm9qZWN0KHRvLCB3ZWlnaHRzKTtcblxuICAgICAgdGhpcy5jb25uZWN0aW9uc1tjb25uZWN0aW9uLklEXSA9IGNvbm5lY3Rpb247XG4gICAgICB0aGlzLnNpemUgPSB0aGlzLmxpc3QucHVzaChjb25uZWN0aW9uKTtcbiAgICB9XG4gIH1cbiAgXG4gIGZyb21MYXllci5jb25uZWN0ZWR0by5wdXNoKHRoaXMpO1xufVxuXG4vLyB0eXBlcyBvZiBjb25uZWN0aW9uc1xuTGF5ZXIuY29ubmVjdGlvblR5cGUgPSB7fTtcbkxheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19BTEwgPSBcIkFMTCBUTyBBTExcIjtcbkxheWVyLmNvbm5lY3Rpb25UeXBlLk9ORV9UT19PTkUgPSBcIk9ORSBUTyBPTkVcIjtcbkxheWVyLmNvbm5lY3Rpb25UeXBlLkFMTF9UT19FTFNFID0gXCJBTEwgVE8gRUxTRVwiO1xuXG4vLyB0eXBlcyBvZiBnYXRlc1xuTGF5ZXIuZ2F0ZVR5cGUgPSB7fTtcbkxheWVyLmdhdGVUeXBlLklOUFVUID0gXCJJTlBVVFwiO1xuTGF5ZXIuZ2F0ZVR5cGUuT1VUUFVUID0gXCJPVVRQVVRcIjtcbkxheWVyLmdhdGVUeXBlLk9ORV9UT19PTkUgPSBcIk9ORSBUTyBPTkVcIjtcblxuKGZ1bmN0aW9uKCkge1xuICB2YXIgY29ubmVjdGlvbnMgPSAwO1xuICBMYXllci5jb25uZWN0aW9uLnVpZCA9IGZ1bmN0aW9uKCkge1xuICAgIHJldHVybiBjb25uZWN0aW9ucysrO1xuICB9XG59KSgpO1xuXG4vLyBleHBvcnRcbmlmIChtb2R1bGUpIG1vZHVsZS5leHBvcnRzID0gTGF5ZXI7XG5cbiIsIi8vIGltcG9ydFxudmFyIE5ldXJvbiA9IHJlcXVpcmUoJy4vbmV1cm9uJyksXG4gICAgTGF5ZXIgPSByZXF1aXJlKCcuL2xheWVyJyk7XG5cbi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIE5FVFdPUktcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cbmZ1bmN0aW9uIE5ldHdvcmsobGF5ZXJzKSB7XG4gIGlmICh0eXBlb2YgbGF5ZXJzICE9ICd1bmRlZmluZWQnKSB7XG4gICAgdGhpcy5sYXllcnMgPSBsYXllcnMgfHwge1xuICAgICAgaW5wdXQ6IG51bGwsXG4gICAgICBoaWRkZW46IHt9LFxuICAgICAgb3V0cHV0OiBudWxsXG4gICAgfTtcbiAgICB0aGlzLm9wdGltaXplZCA9IG51bGw7XG4gIH1cbn1cbk5ldHdvcmsucHJvdG90eXBlID0ge1xuXG4gIC8vIGZlZWQtZm9yd2FyZCBhY3RpdmF0aW9uIG9mIGFsbCB0aGUgbGF5ZXJzIHRvIHByb2R1Y2UgYW4gb3VwdXRcbiAgYWN0aXZhdGU6IGZ1bmN0aW9uKGlucHV0KSB7XG5cbiAgICBpZiAodGhpcy5vcHRpbWl6ZWQgPT09IGZhbHNlKVxuICAgIHtcbiAgICAgIHRoaXMubGF5ZXJzLmlucHV0LmFjdGl2YXRlKGlucHV0KTtcbiAgICAgIGZvciAodmFyIGxheWVyIGluIHRoaXMubGF5ZXJzLmhpZGRlbilcbiAgICAgICAgdGhpcy5sYXllcnMuaGlkZGVuW2xheWVyXS5hY3RpdmF0ZSgpO1xuICAgICAgcmV0dXJuIHRoaXMubGF5ZXJzLm91dHB1dC5hY3RpdmF0ZSgpO1xuICAgIH0gXG4gICAgZWxzZSBcbiAgICB7XG4gICAgICBpZiAodGhpcy5vcHRpbWl6ZWQgPT0gbnVsbClcbiAgICAgICAgdGhpcy5vcHRpbWl6ZSgpO1xuICAgICAgcmV0dXJuIHRoaXMub3B0aW1pemVkLmFjdGl2YXRlKGlucHV0KTtcbiAgICB9XG4gIH0sXG5cbiAgLy8gYmFjay1wcm9wYWdhdGUgdGhlIGVycm9yIHRocnUgdGhlIG5ldHdvcmtcbiAgcHJvcGFnYXRlOiBmdW5jdGlvbihyYXRlLCB0YXJnZXQpIHtcblxuICAgIGlmICh0aGlzLm9wdGltaXplZCA9PT0gZmFsc2UpXG4gICAge1xuICAgICAgdGhpcy5sYXllcnMub3V0cHV0LnByb3BhZ2F0ZShyYXRlLCB0YXJnZXQpO1xuICAgICAgdmFyIHJldmVyc2UgPSBbXTtcbiAgICAgIGZvciAodmFyIGxheWVyIGluIHRoaXMubGF5ZXJzLmhpZGRlbilcbiAgICAgICAgcmV2ZXJzZS5wdXNoKHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl0pO1xuICAgICAgcmV2ZXJzZS5yZXZlcnNlKCk7XG4gICAgICBmb3IgKHZhciBsYXllciBpbiByZXZlcnNlKVxuICAgICAgICByZXZlcnNlW2xheWVyXS5wcm9wYWdhdGUocmF0ZSk7XG4gICAgfSBcbiAgICBlbHNlIFxuICAgIHtcbiAgICAgIGlmICh0aGlzLm9wdGltaXplZCA9PSBudWxsKVxuICAgICAgICB0aGlzLm9wdGltaXplKCk7XG4gICAgICB0aGlzLm9wdGltaXplZC5wcm9wYWdhdGUocmF0ZSwgdGFyZ2V0KTtcbiAgICB9XG4gIH0sXG5cbiAgLy8gcHJvamVjdCBhIGNvbm5lY3Rpb24gdG8gYW5vdGhlciB1bml0IChlaXRoZXIgYSBuZXR3b3JrIG9yIGEgbGF5ZXIpXG4gIHByb2plY3Q6IGZ1bmN0aW9uKHVuaXQsIHR5cGUsIHdlaWdodHMpIHtcblxuICAgIGlmICh0aGlzLm9wdGltaXplZClcbiAgICAgIHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cbiAgICBpZiAodW5pdCBpbnN0YW5jZW9mIE5ldHdvcmspXG4gICAgICByZXR1cm4gdGhpcy5sYXllcnMub3V0cHV0LnByb2plY3QodW5pdC5sYXllcnMuaW5wdXQsIHR5cGUsIHdlaWdodHMpO1xuXG4gICAgaWYgKHVuaXQgaW5zdGFuY2VvZiBMYXllcilcbiAgICAgIHJldHVybiB0aGlzLmxheWVycy5vdXRwdXQucHJvamVjdCh1bml0LCB0eXBlLCB3ZWlnaHRzKTtcblxuICAgIHRocm93IFwiSW52YWxpZCBhcmd1bWVudCwgeW91IGNhbiBvbmx5IHByb2plY3QgY29ubmVjdGlvbnMgdG8gTEFZRVJTIGFuZCBORVRXT1JLUyFcIjtcbiAgfSxcblxuICAvLyBsZXQgdGhpcyBuZXR3b3JrIGdhdGUgYSBjb25uZWN0aW9uXG4gIGdhdGU6IGZ1bmN0aW9uKGNvbm5lY3Rpb24sIHR5cGUpIHtcbiAgICBpZiAodGhpcy5vcHRpbWl6ZWQpXG4gICAgICB0aGlzLm9wdGltaXplZC5yZXNldCgpO1xuICAgIHRoaXMubGF5ZXJzLm91dHB1dC5nYXRlKGNvbm5lY3Rpb24sIHR5cGUpO1xuICB9LFxuXG4gIC8vIGNsZWFyIGFsbCBlbGVnaWJpbGl0eSB0cmFjZXMgYW5kIGV4dGVuZGVkIGVsZWdpYmlsaXR5IHRyYWNlcyAodGhlIG5ldHdvcmsgZm9yZ2V0cyBpdHMgY29udGV4dCwgYnV0IG5vdCB3aGF0IHdhcyB0cmFpbmVkKVxuICBjbGVhcjogZnVuY3Rpb24oKSB7XG5cbiAgICB0aGlzLnJlc3RvcmUoKTtcblxuICAgIHZhciBpbnB1dExheWVyID0gdGhpcy5sYXllcnMuaW5wdXQsXG4gICAgICBvdXRwdXRMYXllciA9IHRoaXMubGF5ZXJzLm91dHB1dDtcblxuICAgIGlucHV0TGF5ZXIuY2xlYXIoKTtcbiAgICBmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pIHtcbiAgICAgIHZhciBoaWRkZW5MYXllciA9IHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl07XG4gICAgICBoaWRkZW5MYXllci5jbGVhcigpO1xuICAgIH1cbiAgICBvdXRwdXRMYXllci5jbGVhcigpO1xuXG4gICAgaWYgKHRoaXMub3B0aW1pemVkKVxuICAgICAgdGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcbiAgfSxcblxuICAvLyByZXNldCBhbGwgd2VpZ2h0cyBhbmQgY2xlYXIgYWxsIHRyYWNlcyAoZW5kcyB1cCBsaWtlIGEgbmV3IG5ldHdvcmspXG4gIHJlc2V0OiBmdW5jdGlvbigpIHtcblxuICAgIHRoaXMucmVzdG9yZSgpO1xuXG4gICAgdmFyIGlucHV0TGF5ZXIgPSB0aGlzLmxheWVycy5pbnB1dCxcbiAgICAgIG91dHB1dExheWVyID0gdGhpcy5sYXllcnMub3V0cHV0O1xuXG4gICAgaW5wdXRMYXllci5yZXNldCgpO1xuICAgIGZvciAodmFyIGxheWVyIGluIHRoaXMubGF5ZXJzLmhpZGRlbikge1xuICAgICAgdmFyIGhpZGRlbkxheWVyID0gdGhpcy5sYXllcnMuaGlkZGVuW2xheWVyXTtcbiAgICAgIGhpZGRlbkxheWVyLnJlc2V0KCk7XG4gICAgfVxuICAgIG91dHB1dExheWVyLnJlc2V0KCk7XG5cbiAgICBpZiAodGhpcy5vcHRpbWl6ZWQpXG4gICAgICB0aGlzLm9wdGltaXplZC5yZXNldCgpO1xuICB9LFxuXG4gIC8vIGhhcmRjb2RlcyB0aGUgYmVoYXZpb3VyIG9mIHRoZSB3aG9sZSBuZXR3b3JrIGludG8gYSBzaW5nbGUgb3B0aW1pemVkIGZ1bmN0aW9uXG4gIG9wdGltaXplOiBmdW5jdGlvbigpIHtcblxuICAgIHZhciB0aGF0ID0gdGhpcztcbiAgICB2YXIgb3B0aW1pemVkID0ge307XG4gICAgdmFyIG5ldXJvbnMgPSB0aGlzLm5ldXJvbnMoKTtcblxuICAgIGZvciAodmFyIGkgaW4gbmV1cm9ucykge1xuICAgICAgdmFyIG5ldXJvbiA9IG5ldXJvbnNbaV0ubmV1cm9uO1xuICAgICAgdmFyIGxheWVyID0gbmV1cm9uc1tpXS5sYXllcjtcbiAgICAgIHdoaWxlIChuZXVyb24ubmV1cm9uKVxuICAgICAgICBuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuICAgICAgb3B0aW1pemVkID0gbmV1cm9uLm9wdGltaXplKG9wdGltaXplZCwgbGF5ZXIpO1xuICAgIH1cbiAgICBmb3IgKHZhciBpIGluIG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMpXG4gICAgICBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzW2ldLnJldmVyc2UoKTtcbiAgICBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzLnJldmVyc2UoKTtcblxuICAgIHZhciBoYXJkY29kZSA9IFwiXCI7XG4gICAgaGFyZGNvZGUgKz0gXCJ2YXIgRiA9IEZsb2F0NjRBcnJheSA/IG5ldyBGbG9hdDY0QXJyYXkoXCIgKyBvcHRpbWl6ZWQubWVtb3J5ICtcbiAgICAgIFwiKSA6IFtdOyBcIjtcbiAgICBmb3IgKHZhciBpIGluIG9wdGltaXplZC52YXJpYWJsZXMpXG4gICAgICBoYXJkY29kZSArPSBcIkZbXCIgKyBvcHRpbWl6ZWQudmFyaWFibGVzW2ldLmlkICsgXCJdID0gXCIgKyAob3B0aW1pemVkLnZhcmlhYmxlc1tcbiAgICAgICAgaV0udmFsdWUgfHwgMCkgKyBcIjsgXCI7XG4gICAgaGFyZGNvZGUgKz0gXCJ2YXIgYWN0aXZhdGUgPSBmdW5jdGlvbihpbnB1dCl7XFxuXCI7XG4gICAgaGFyZGNvZGUgKz0gXCJpbmZsdWVuY2VzID0gW107XCI7XG4gICAgZm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQuaW5wdXRzKVxuICAgICAgaGFyZGNvZGUgKz0gXCJGW1wiICsgb3B0aW1pemVkLmlucHV0c1tpXSArIFwiXSA9IGlucHV0W1wiICsgaSArIFwiXTsgXCI7XG4gICAgZm9yICh2YXIgY3VycmVudExheWVyIGluIG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlcykge1xuICAgICAgaWYgKG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdLmxlbmd0aCA+IDApIHtcbiAgICAgICAgZm9yICh2YXIgY3VycmVudE5ldXJvbiBpbiBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXSkge1xuICAgICAgICAgIGhhcmRjb2RlICs9IG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdW2N1cnJlbnROZXVyb25dLmpvaW4oXCIgXCIpO1xuICAgICAgICAgIGhhcmRjb2RlICs9IG9wdGltaXplZC50cmFjZV9zZW50ZW5jZXNbY3VycmVudExheWVyXVtjdXJyZW50TmV1cm9uXS5qb2luKFwiIFwiKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICBoYXJkY29kZSArPSBcIiB2YXIgb3V0cHV0ID0gW107IFwiXG4gICAgZm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQub3V0cHV0cylcbiAgICAgIGhhcmRjb2RlICs9IFwib3V0cHV0W1wiICsgaSArIFwiXSA9IEZbXCIgKyBvcHRpbWl6ZWQub3V0cHV0c1tpXSArIFwiXTsgXCI7XG4gICAgaGFyZGNvZGUgKz0gXCJyZXR1cm4gb3V0cHV0OyB9OyBcIlxuICAgIGhhcmRjb2RlICs9IFwidmFyIHByb3BhZ2F0ZSA9IGZ1bmN0aW9uKHJhdGUsIHRhcmdldCl7XFxuXCI7XG4gICAgaGFyZGNvZGUgKz0gXCJGW1wiICsgb3B0aW1pemVkLnZhcmlhYmxlcy5yYXRlLmlkICsgXCJdID0gcmF0ZTsgXCI7XG4gICAgZm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQudGFyZ2V0cylcbiAgICAgIGhhcmRjb2RlICs9IFwiRltcIiArIG9wdGltaXplZC50YXJnZXRzW2ldICsgXCJdID0gdGFyZ2V0W1wiICsgaSArIFwiXTsgXCI7XG4gICAgZm9yICh2YXIgY3VycmVudExheWVyIGluIG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMpXG4gICAgICBmb3IgKHZhciBjdXJyZW50TmV1cm9uIGluIG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXSlcbiAgICAgICAgaGFyZGNvZGUgKz0gb3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdW2N1cnJlbnROZXVyb25dLmpvaW4oXCIgXCIpICsgXCIgXCI7XG4gICAgaGFyZGNvZGUgKz0gXCIgfTtcXG5cIjtcbiAgICBoYXJkY29kZSArPVxuICAgICAgXCJ2YXIgb3duZXJzaGlwID0gZnVuY3Rpb24obWVtb3J5QnVmZmVyKXtcXG5GID0gbWVtb3J5QnVmZmVyO1xcbnRoaXMubWVtb3J5ID0gRjtcXG59O1xcblwiO1xuICAgIGhhcmRjb2RlICs9XG4gICAgICBcInJldHVybiB7XFxubWVtb3J5OiBGLFxcbmFjdGl2YXRlOiBhY3RpdmF0ZSxcXG5wcm9wYWdhdGU6IHByb3BhZ2F0ZSxcXG5vd25lcnNoaXA6IG93bmVyc2hpcFxcbn07XCI7XG4gICAgaGFyZGNvZGUgPSBoYXJkY29kZS5zcGxpdChcIjtcIikuam9pbihcIjtcXG5cIik7XG5cbiAgICB2YXIgY29uc3RydWN0b3IgPSBuZXcgRnVuY3Rpb24oaGFyZGNvZGUpO1xuXG4gICAgdmFyIG5ldHdvcmsgPSBjb25zdHJ1Y3RvcigpO1xuICAgIG5ldHdvcmsuZGF0YSA9IHtcbiAgICAgIHZhcmlhYmxlczogb3B0aW1pemVkLnZhcmlhYmxlcyxcbiAgICAgIGFjdGl2YXRlOiBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXMsXG4gICAgICBwcm9wYWdhdGU6IG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMsXG4gICAgICB0cmFjZTogb3B0aW1pemVkLnRyYWNlX3NlbnRlbmNlcyxcbiAgICAgIGlucHV0czogb3B0aW1pemVkLmlucHV0cyxcbiAgICAgIG91dHB1dHM6IG9wdGltaXplZC5vdXRwdXRzLFxuICAgICAgY2hlY2tfYWN0aXZhdGlvbjogdGhpcy5hY3RpdmF0ZSxcbiAgICAgIGNoZWNrX3Byb3BhZ2F0aW9uOiB0aGlzLnByb3BhZ2F0ZVxuICAgIH1cblxuICAgIG5ldHdvcmsucmVzZXQgPSBmdW5jdGlvbigpIHtcbiAgICAgIGlmICh0aGF0Lm9wdGltaXplZCkge1xuICAgICAgICB0aGF0Lm9wdGltaXplZCA9IG51bGw7XG4gICAgICAgIHRoYXQuYWN0aXZhdGUgPSBuZXR3b3JrLmRhdGEuY2hlY2tfYWN0aXZhdGlvbjtcbiAgICAgICAgdGhhdC5wcm9wYWdhdGUgPSBuZXR3b3JrLmRhdGEuY2hlY2tfcHJvcGFnYXRpb247XG4gICAgICB9XG4gICAgfVxuXG4gICAgdGhpcy5vcHRpbWl6ZWQgPSBuZXR3b3JrO1xuICAgIHRoaXMuYWN0aXZhdGUgPSBuZXR3b3JrLmFjdGl2YXRlO1xuICAgIHRoaXMucHJvcGFnYXRlID0gbmV0d29yay5wcm9wYWdhdGU7XG4gIH0sXG5cbiAgLy8gcmVzdG9yZXMgYWxsIHRoZSB2YWx1ZXMgZnJvbSB0aGUgb3B0aW1pemVkIG5ldHdvcmsgdGhlIHRoZWlyIHJlc3BlY3RpdmUgb2JqZWN0cyBpbiBvcmRlciB0byBtYW5pcHVsYXRlIHRoZSBuZXR3b3JrXG4gIHJlc3RvcmU6IGZ1bmN0aW9uKCkge1xuICAgIGlmICghdGhpcy5vcHRpbWl6ZWQpXG4gICAgICByZXR1cm47XG5cbiAgICB2YXIgb3B0aW1pemVkID0gdGhpcy5vcHRpbWl6ZWQ7XG5cbiAgICB2YXIgZ2V0VmFsdWUgPSBmdW5jdGlvbigpIHtcbiAgICAgIHZhciBhcmdzID0gQXJyYXkucHJvdG90eXBlLnNsaWNlLmNhbGwoYXJndW1lbnRzKTtcblxuICAgICAgdmFyIHVuaXQgPSBhcmdzLnNoaWZ0KCk7XG4gICAgICB2YXIgcHJvcCA9IGFyZ3MucG9wKCk7XG5cbiAgICAgIHZhciBpZCA9IHByb3AgKyAnXyc7XG4gICAgICBmb3IgKHZhciBwcm9wZXJ0eSBpbiBhcmdzKVxuICAgICAgICBpZCArPSBhcmdzW3Byb3BlcnR5XSArICdfJztcbiAgICAgIGlkICs9IHVuaXQuSUQ7XG5cbiAgICAgIHZhciBtZW1vcnkgPSBvcHRpbWl6ZWQubWVtb3J5O1xuICAgICAgdmFyIHZhcmlhYmxlcyA9IG9wdGltaXplZC5kYXRhLnZhcmlhYmxlcztcblxuICAgICAgaWYgKGlkIGluIHZhcmlhYmxlcylcbiAgICAgICAgcmV0dXJuIG1lbW9yeVt2YXJpYWJsZXNbaWRdLmlkXTtcbiAgICAgIHJldHVybiAwO1xuICAgIH1cblxuICAgIHZhciBsaXN0ID0gdGhpcy5uZXVyb25zKCk7XG5cbiAgICAvLyBsaW5rIGlkJ3MgdG8gcG9zaXRpb25zIGluIHRoZSBhcnJheVxuICAgIHZhciBpZHMgPSB7fTtcbiAgICBmb3IgKHZhciBpIGluIGxpc3QpIHtcbiAgICAgIHZhciBuZXVyb24gPSBsaXN0W2ldLm5ldXJvbjtcbiAgICAgIHdoaWxlIChuZXVyb24ubmV1cm9uKVxuICAgICAgICBuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuXG4gICAgICBuZXVyb24uc3RhdGUgPSBnZXRWYWx1ZShuZXVyb24sICdzdGF0ZScpO1xuICAgICAgbmV1cm9uLm9sZCA9IGdldFZhbHVlKG5ldXJvbiwgJ29sZCcpO1xuICAgICAgbmV1cm9uLmFjdGl2YXRpb24gPSBnZXRWYWx1ZShuZXVyb24sICdhY3RpdmF0aW9uJyk7XG4gICAgICBuZXVyb24uYmlhcyA9IGdldFZhbHVlKG5ldXJvbiwgJ2JpYXMnKTtcblxuICAgICAgZm9yICh2YXIgaW5wdXQgaW4gbmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5KVxuICAgICAgICBuZXVyb24udHJhY2UuZWxlZ2liaWxpdHlbaW5wdXRdID0gZ2V0VmFsdWUobmV1cm9uLCAndHJhY2UnLFxuICAgICAgICAgICdlbGVnaWJpbGl0eScsIGlucHV0KTtcblxuICAgICAgZm9yICh2YXIgZ2F0ZWQgaW4gbmV1cm9uLnRyYWNlLmV4dGVuZGVkKVxuICAgICAgICBmb3IgKHZhciBpbnB1dCBpbiBuZXVyb24udHJhY2UuZXh0ZW5kZWRbZ2F0ZWRdKVxuICAgICAgICAgIG5ldXJvbi50cmFjZS5leHRlbmRlZFtnYXRlZF1baW5wdXRdID0gZ2V0VmFsdWUobmV1cm9uLCAndHJhY2UnLFxuICAgICAgICAgICAgJ2V4dGVuZGVkJywgZ2F0ZWQsIGlucHV0KTtcbiAgICB9XG5cbiAgICAvLyBnZXQgY29ubmVjdGlvbnNcbiAgICBmb3IgKHZhciBpIGluIGxpc3QpIHtcbiAgICAgIHZhciBuZXVyb24gPSBsaXN0W2ldLm5ldXJvbjtcbiAgICAgIHdoaWxlIChuZXVyb24ubmV1cm9uKVxuICAgICAgICBuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuXG4gICAgICBmb3IgKHZhciBqIGluIG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpIHtcbiAgICAgICAgdmFyIGNvbm5lY3Rpb24gPSBuZXVyb24uY29ubmVjdGlvbnMucHJvamVjdGVkW2pdO1xuICAgICAgICBjb25uZWN0aW9uLndlaWdodCA9IGdldFZhbHVlKGNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcbiAgICAgICAgY29ubmVjdGlvbi5nYWluID0gZ2V0VmFsdWUoY29ubmVjdGlvbiwgJ2dhaW4nKTtcbiAgICAgIH1cbiAgICB9XG4gIH0sXG5cbiAgLy8gcmV0dXJucyBhbGwgdGhlIG5ldXJvbnMgaW4gdGhlIG5ldHdvcmtcbiAgbmV1cm9uczogZnVuY3Rpb24oKSB7XG5cbiAgICB2YXIgbmV1cm9ucyA9IFtdO1xuXG4gICAgdmFyIGlucHV0TGF5ZXIgPSB0aGlzLmxheWVycy5pbnB1dC5uZXVyb25zKCksXG4gICAgICBvdXRwdXRMYXllciA9IHRoaXMubGF5ZXJzLm91dHB1dC5uZXVyb25zKCk7XG5cbiAgICBmb3IgKHZhciBuZXVyb24gaW4gaW5wdXRMYXllcilcbiAgICAgIG5ldXJvbnMucHVzaCh7XG4gICAgICAgIG5ldXJvbjogaW5wdXRMYXllcltuZXVyb25dLFxuICAgICAgICBsYXllcjogJ2lucHV0J1xuICAgICAgfSk7XG5cbiAgICBmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pIHtcbiAgICAgIHZhciBoaWRkZW5MYXllciA9IHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl0ubmV1cm9ucygpO1xuICAgICAgZm9yICh2YXIgbmV1cm9uIGluIGhpZGRlbkxheWVyKVxuICAgICAgICBuZXVyb25zLnB1c2goe1xuICAgICAgICAgIG5ldXJvbjogaGlkZGVuTGF5ZXJbbmV1cm9uXSxcbiAgICAgICAgICBsYXllcjogbGF5ZXJcbiAgICAgICAgfSk7XG4gICAgfVxuICAgIGZvciAodmFyIG5ldXJvbiBpbiBvdXRwdXRMYXllcilcbiAgICAgIG5ldXJvbnMucHVzaCh7XG4gICAgICAgIG5ldXJvbjogb3V0cHV0TGF5ZXJbbmV1cm9uXSxcbiAgICAgICAgbGF5ZXI6ICdvdXRwdXQnXG4gICAgICB9KTtcblxuICAgIHJldHVybiBuZXVyb25zO1xuICB9LFxuXG4gIC8vIHJldHVybnMgbnVtYmVyIG9mIGlucHV0cyBvZiB0aGUgbmV0d29ya1xuICBpbnB1dHM6IGZ1bmN0aW9uKCkge1xuICAgIHJldHVybiB0aGlzLmxheWVycy5pbnB1dC5zaXplO1xuICB9LFxuXG4gIC8vIHJldHVybnMgbnVtYmVyIG9mIG91dHB1dHMgb2YgaHRlIG5ldHdvcmtcbiAgb3V0cHV0czogZnVuY3Rpb24oKSB7XG4gICAgcmV0dXJuIHRoaXMubGF5ZXJzLm91dHB1dC5zaXplO1xuICB9LFxuXG4gIC8vIHNldHMgdGhlIGxheWVycyBvZiB0aGUgbmV0d29ya1xuICBzZXQ6IGZ1bmN0aW9uKGxheWVycykge1xuXG4gICAgdGhpcy5sYXllcnMgPSBsYXllcnM7XG4gICAgaWYgKHRoaXMub3B0aW1pemVkKVxuICAgICAgdGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcbiAgfSxcblxuICBzZXRPcHRpbWl6ZTogZnVuY3Rpb24oYm9vbCl7XG4gICAgdGhpcy5yZXN0b3JlKCk7XG4gICAgaWYgKHRoaXMub3B0aW1pemVkKVxuICAgICAgdGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcbiAgICB0aGlzLm9wdGltaXplZCA9IGJvb2w/IG51bGwgOiBmYWxzZTtcbiAgfSxcblxuICAvLyByZXR1cm5zIGEganNvbiB0aGF0IHJlcHJlc2VudHMgYWxsIHRoZSBuZXVyb25zIGFuZCBjb25uZWN0aW9ucyBvZiB0aGUgbmV0d29ya1xuICB0b0pTT046IGZ1bmN0aW9uKGlnbm9yZVRyYWNlcykge1xuXG4gICAgdGhpcy5yZXN0b3JlKCk7XG5cbiAgICB2YXIgbGlzdCA9IHRoaXMubmV1cm9ucygpO1xuICAgIHZhciBuZXVyb25zID0gW107XG4gICAgdmFyIGNvbm5lY3Rpb25zID0gW107XG5cbiAgICAvLyBsaW5rIGlkJ3MgdG8gcG9zaXRpb25zIGluIHRoZSBhcnJheVxuICAgIHZhciBpZHMgPSB7fTtcbiAgICBmb3IgKHZhciBpIGluIGxpc3QpIHtcbiAgICAgIHZhciBuZXVyb24gPSBsaXN0W2ldLm5ldXJvbjtcbiAgICAgIHdoaWxlIChuZXVyb24ubmV1cm9uKVxuICAgICAgICBuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuICAgICAgaWRzW25ldXJvbi5JRF0gPSBpO1xuXG4gICAgICB2YXIgY29weSA9IHtcbiAgICAgICAgdHJhY2U6IHtcbiAgICAgICAgICBlbGVnaWJpbGl0eToge30sXG4gICAgICAgICAgZXh0ZW5kZWQ6IHt9XG4gICAgICAgIH0sXG4gICAgICAgIHN0YXRlOiBuZXVyb24uc3RhdGUsXG4gICAgICAgIG9sZDogbmV1cm9uLm9sZCxcbiAgICAgICAgYWN0aXZhdGlvbjogbmV1cm9uLmFjdGl2YXRpb24sXG4gICAgICAgIGJpYXM6IG5ldXJvbi5iaWFzLFxuICAgICAgICBsYXllcjogbGlzdFtpXS5sYXllclxuICAgICAgfTtcblxuICAgICAgY29weS5zcXVhc2ggPSBuZXVyb24uc3F1YXNoID09IE5ldXJvbi5zcXVhc2guTE9HSVNUSUMgPyBcIkxPR0lTVElDXCIgOlxuICAgICAgICBuZXVyb24uc3F1YXNoID09IE5ldXJvbi5zcXVhc2guVEFOSCA/IFwiVEFOSFwiIDpcbiAgICAgICAgbmV1cm9uLnNxdWFzaCA9PSBOZXVyb24uc3F1YXNoLklERU5USVRZID8gXCJJREVOVElUWVwiIDpcbiAgICAgICAgbmV1cm9uLnNxdWFzaCA9PSBOZXVyb24uc3F1YXNoLkhMSU0gPyBcIkhMSU1cIiA6XG4gICAgICAgIG51bGw7XG5cbiAgICAgIG5ldXJvbnMucHVzaChjb3B5KTtcbiAgICB9XG5cbiAgICBpZiAoIWlnbm9yZVRyYWNlcylcbiAgICAgIGZvciAodmFyIGkgaW4gbmV1cm9ucykge1xuICAgICAgICB2YXIgY29weSA9IG5ldXJvbnNbaV07XG5cbiAgICAgICAgZm9yICh2YXIgaW5wdXQgaW4gbmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5KVxuICAgICAgICAgIGNvcHkudHJhY2UuZWxlZ2liaWxpdHlbaW5wdXRdID0gbmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0XTtcblxuICAgICAgICBmb3IgKHZhciBnYXRlZCBpbiBuZXVyb24udHJhY2UuZXh0ZW5kZWQpIHtcbiAgICAgICAgICBjb3B5LnRyYWNlLmV4dGVuZGVkW2dhdGVkXSA9IHt9O1xuICAgICAgICAgIGZvciAodmFyIGlucHV0IGluIG5ldXJvbi50cmFjZS5leHRlbmRlZFtnYXRlZF0pXG4gICAgICAgICAgICBjb3B5LnRyYWNlLmV4dGVuZGVkW2lkc1tnYXRlZF1dW2lucHV0XSA9IG5ldXJvbi50cmFjZS5leHRlbmRlZFtcbiAgICAgICAgICAgICAgZ2F0ZWRdW2lucHV0XTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgLy8gZ2V0IGNvbm5lY3Rpb25zXG4gICAgZm9yICh2YXIgaSBpbiBsaXN0KSB7XG4gICAgICB2YXIgbmV1cm9uID0gbGlzdFtpXS5uZXVyb247XG4gICAgICB3aGlsZSAobmV1cm9uLm5ldXJvbilcbiAgICAgICAgbmV1cm9uID0gbmV1cm9uLm5ldXJvbjtcblxuICAgICAgZm9yICh2YXIgaiBpbiBuZXVyb24uY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG4gICAgICAgIHZhciBjb25uZWN0aW9uID0gbmV1cm9uLmNvbm5lY3Rpb25zLnByb2plY3RlZFtqXTtcbiAgICAgICAgY29ubmVjdGlvbnMucHVzaCh7XG4gICAgICAgICAgZnJvbTogaWRzW2Nvbm5lY3Rpb24uZnJvbS5JRF0sXG4gICAgICAgICAgdG86IGlkc1tjb25uZWN0aW9uLnRvLklEXSxcbiAgICAgICAgICB3ZWlnaHQ6IGNvbm5lY3Rpb24ud2VpZ2h0LFxuICAgICAgICAgIGdhdGVyOiBjb25uZWN0aW9uLmdhdGVyID8gaWRzW2Nvbm5lY3Rpb24uZ2F0ZXIuSURdIDogbnVsbCxcbiAgICAgICAgfSk7XG4gICAgICB9XG4gICAgICBpZiAobmV1cm9uLnNlbGZjb25uZWN0ZWQoKSlcbiAgICAgICAgY29ubmVjdGlvbnMucHVzaCh7XG4gICAgICAgICAgZnJvbTogaWRzW25ldXJvbi5JRF0sXG4gICAgICAgICAgdG86IGlkc1tuZXVyb24uSURdLFxuICAgICAgICAgIHdlaWdodDogbmV1cm9uLnNlbGZjb25uZWN0aW9uLndlaWdodCxcbiAgICAgICAgICBnYXRlcjogbmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyID8gaWRzW25ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlclxuICAgICAgICAgICAgLklEXSA6IG51bGwsXG4gICAgICAgIH0pO1xuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICBuZXVyb25zOiBuZXVyb25zLFxuICAgICAgY29ubmVjdGlvbnM6IGNvbm5lY3Rpb25zXG4gICAgfVxuICB9LFxuICBcbiAgLy8gZXhwb3J0IHRoZSB0b3BvbG9neSBpbnRvIGRvdCBsYW5ndWFnZSB3aGljaCBjYW4gYmUgdmlzdWFsaXplZCBhcyBncmFwaHMgdXNpbmcgZG90XG4gIC8qIGV4YW1wbGU6IC4uLiBjb25zb2xlLmxvZyhuZXQudG9Eb3RMYW5nKCkpO1xuICAgICAgICAgICAgICAkIG5vZGUgZXhhbXBsZS5qcyA+IGV4YW1wbGUuZG90XG4gICAgICAgICAgICAgICQgZG90IGV4YW1wbGUuZG90IC1UcG5nID4gb3V0LnBuZ1xuICAqL1xuICB0b0RvdExhbmc6IGZ1bmN0aW9uKGVkZ2Vjb25uZWN0aW9uKSB7XG4gICAgaWYgKCEgdHlwZW9mIGVkZ2Vjb25uZWN0aW9uKVxuICAgICAgZWRnZWNvbm5lY3Rpb24gPSBmYWxzZTtcbiAgICB2YXIgY29kZSA9IFwiZGlncmFwaCBubiB7XFxuICAgIHJhbmtkaXIgPSBCVFxcblwiO1xuICAgIHZhciBsYXllcnMgPSBbdGhpcy5sYXllcnMuaW5wdXRdLmNvbmNhdCh0aGlzLmxheWVycy5oaWRkZW4sIHRoaXMubGF5ZXJzLm91dHB1dCk7XG4gICAgZm9yICh2YXIgbGF5ZXIgaW4gbGF5ZXJzKSB7XG4gICAgICBmb3IgKHZhciB0byBpbiBsYXllcnNbbGF5ZXJdLmNvbm5lY3RlZHRvKSB7IC8vIHByb2plY3Rpb25zXG4gICAgICAgIHZhciBjb25uZWN0aW9uID0gbGF5ZXJzW2xheWVyXS5jb25uZWN0ZWR0b1t0b107XG4gICAgICAgIHZhciBsYXllcnRvID0gY29ubmVjdGlvbi50bztcbiAgICAgICAgdmFyIHNpemUgPSBjb25uZWN0aW9uLnNpemU7XG4gICAgICAgIHZhciBsYXllcklEID0gbGF5ZXJzLmluZGV4T2YobGF5ZXJzW2xheWVyXSk7XG4gICAgICAgIHZhciBsYXllcnRvSUQgPSBsYXllcnMuaW5kZXhPZihsYXllcnRvKTtcbiAgICAgICAgLyogaHR0cDovL3N0YWNrb3ZlcmZsb3cuY29tL3F1ZXN0aW9ucy8yNjg0NTU0MC9jb25uZWN0LWVkZ2VzLXdpdGgtZ3JhcGgtZG90XG4gICAgICAgICAqIERPVCBkb2VzIG5vdCBzdXBwb3J0IGVkZ2UtdG8tZWRnZSBjb25uZWN0aW9uc1xuICAgICAgICAgKiBUaGlzIHdvcmthcm91bmQgcHJvZHVjZXMgc29tZXdoYXQgd2VpcmQgZ3JhcGhzIC4uLlxuICAgICAgICAqL1xuICAgICAgICBpZiAoIGVkZ2Vjb25uZWN0aW9uKSB7XG4gICAgICAgICAgaWYgKGNvbm5lY3Rpb24uZ2F0ZWRmcm9tLmxlbmd0aCkge1xuICAgICAgICAgICAgdmFyIGZha2VOb2RlID0gXCJmYWtlXCIgKyBsYXllcklEICsgXCJfXCIgKyBsYXllcnRvSUQ7XG4gICAgICAgICAgICBjb2RlICs9IFwiICAgIFwiICsgZmFrZU5vZGUgK1xuICAgICAgICAgICAgICBcIiBbbGFiZWwgPSBcXFwiXFxcIiwgc2hhcGUgPSBwb2ludCwgd2lkdGggPSAwLjAxLCBoZWlnaHQgPSAwLjAxXVxcblwiO1xuICAgICAgICAgICAgY29kZSArPSBcIiAgICBcIiArIGxheWVySUQgKyBcIiAtPiBcIiArIGZha2VOb2RlICsgXCIgW2xhYmVsID0gXCIgKyBzaXplICsgXCIsIGFycm93aGVhZCA9IG5vbmVdXFxuXCI7XG4gICAgICAgICAgICBjb2RlICs9IFwiICAgIFwiICsgZmFrZU5vZGUgKyBcIiAtPiBcIiArIGxheWVydG9JRCArIFwiXFxuXCI7XG4gICAgICAgICAgfSBlbHNlXG4gICAgICAgICAgICBjb2RlICs9IFwiICAgIFwiICsgbGF5ZXJJRCArIFwiIC0+IFwiICsgbGF5ZXJ0b0lEICsgXCIgW2xhYmVsID0gXCIgKyBzaXplICsgXCJdXFxuXCI7XG4gICAgICAgICAgZm9yICh2YXIgZnJvbSBpbiBjb25uZWN0aW9uLmdhdGVkZnJvbSkgeyAvLyBnYXRpbmdzXG4gICAgICAgICAgICB2YXIgbGF5ZXJmcm9tID0gY29ubmVjdGlvbi5nYXRlZGZyb21bZnJvbV0ubGF5ZXI7XG4gICAgICAgICAgICB2YXIgdHlwZSA9IGNvbm5lY3Rpb24uZ2F0ZWRmcm9tW2Zyb21dLnR5cGU7XG4gICAgICAgICAgICB2YXIgbGF5ZXJmcm9tSUQgPSBsYXllcnMuaW5kZXhPZihsYXllcmZyb20pO1xuICAgICAgICAgICAgY29kZSArPSBcIiAgICBcIiArIGxheWVyZnJvbUlEICsgXCIgLT4gXCIgKyBmYWtlTm9kZSArIFwiIFtjb2xvciA9IGJsdWVdXFxuXCI7XG4gICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIGNvZGUgKz0gXCIgICAgXCIgKyBsYXllcklEICsgXCIgLT4gXCIgKyBsYXllcnRvSUQgKyBcIiBbbGFiZWwgPSBcIiArIHNpemUgKyBcIl1cXG5cIjtcbiAgICAgICAgICBmb3IgKHZhciBmcm9tIGluIGNvbm5lY3Rpb24uZ2F0ZWRmcm9tKSB7IC8vIGdhdGluZ3NcbiAgICAgICAgICAgIHZhciBsYXllcmZyb20gPSBjb25uZWN0aW9uLmdhdGVkZnJvbVtmcm9tXS5sYXllcjtcbiAgICAgICAgICAgIHZhciB0eXBlID0gY29ubmVjdGlvbi5nYXRlZGZyb21bZnJvbV0udHlwZTtcbiAgICAgICAgICAgIHZhciBsYXllcmZyb21JRCA9IGxheWVycy5pbmRleE9mKGxheWVyZnJvbSk7XG4gICAgICAgICAgICBjb2RlICs9IFwiICAgIFwiICsgbGF5ZXJmcm9tSUQgKyBcIiAtPiBcIiArIGxheWVydG9JRCArIFwiIFtjb2xvciA9IGJsdWVdXFxuXCI7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICAgIGNvZGUgKz0gXCJ9XFxuXCI7XG4gICAgcmV0dXJuIGNvZGU7XG4gIH0sXG5cbiAgLy8gcmV0dXJucyBhIGZ1bmN0aW9uIHRoYXQgd29ya3MgYXMgdGhlIGFjdGl2YXRpb24gb2YgdGhlIG5ldHdvcmsgYW5kIGNhbiBiZSB1c2VkIHdpdGhvdXQgZGVwZW5kaW5nIG9uIHRoZSBsaWJyYXJ5XG4gIHN0YW5kYWxvbmU6IGZ1bmN0aW9uKCkge1xuICAgIGlmICghdGhpcy5vcHRpbWl6ZWQpXG4gICAgICB0aGlzLm9wdGltaXplKCk7XG5cbiAgICB2YXIgZGF0YSA9IHRoaXMub3B0aW1pemVkLmRhdGE7XG5cbiAgICAvLyBidWlsZCBhY3RpdmF0aW9uIGZ1bmN0aW9uXG4gICAgdmFyIGFjdGl2YXRpb24gPSBcImZ1bmN0aW9uIChpbnB1dCkge1xcblwiO1xuXG4gICAgLy8gYnVpbGQgaW5wdXRzXG4gICAgZm9yICh2YXIgaSBpbiBkYXRhLmlucHV0cylcbiAgICAgIGFjdGl2YXRpb24gKz0gXCJGW1wiICsgZGF0YS5pbnB1dHNbaV0gKyBcIl0gPSBpbnB1dFtcIiArIGkgKyBcIl07XFxuXCI7XG5cbiAgICAvLyBidWlsZCBuZXR3b3JrIGFjdGl2YXRpb25cbiAgICBmb3IgKHZhciBuZXVyb24gaW4gZGF0YS5hY3RpdmF0ZSkgeyAvLyBzaG91bGRuJ3QgdGhpcyBiZSBsYXllcj9cbiAgICAgIGZvciAodmFyIHNlbnRlbmNlIGluIGRhdGEuYWN0aXZhdGVbbmV1cm9uXSlcbiAgICAgICAgYWN0aXZhdGlvbiArPSBkYXRhLmFjdGl2YXRlW25ldXJvbl1bc2VudGVuY2VdICsgXCJcXG5cIjtcbiAgICB9XG5cbiAgICAvLyBidWlsZCBvdXRwdXRzXG4gICAgYWN0aXZhdGlvbiArPSBcInZhciBvdXRwdXQgPSBbXTtcXG5cIjtcbiAgICBmb3IgKHZhciBpIGluIGRhdGEub3V0cHV0cylcbiAgICAgIGFjdGl2YXRpb24gKz0gXCJvdXRwdXRbXCIgKyBpICsgXCJdID0gRltcIiArIGRhdGEub3V0cHV0c1tpXSArIFwiXTtcXG5cIjtcbiAgICBhY3RpdmF0aW9uICs9IFwicmV0dXJuIG91dHB1dDtcXG59XCI7XG5cbiAgICAvLyByZWZlcmVuY2UgYWxsIHRoZSBwb3NpdGlvbnMgaW4gbWVtb3J5XG4gICAgdmFyIG1lbW9yeSA9IGFjdGl2YXRpb24ubWF0Y2goL0ZcXFsoXFxkKylcXF0vZyk7XG4gICAgdmFyIGRpbWVuc2lvbiA9IDA7XG4gICAgdmFyIGlkcyA9IHt9O1xuICAgIGZvciAodmFyIGFkZHJlc3MgaW4gbWVtb3J5KSB7XG4gICAgICB2YXIgdG1wID0gbWVtb3J5W2FkZHJlc3NdLm1hdGNoKC9cXGQrLylbMF07XG4gICAgICBpZiAoISh0bXAgaW4gaWRzKSkge1xuICAgICAgICBpZHNbdG1wXSA9IGRpbWVuc2lvbisrO1xuICAgICAgfVxuICAgIH1cbiAgICB2YXIgaGFyZGNvZGUgPSBcIkYgPSB7XFxuXCI7XG4gICAgZm9yICh2YXIgaSBpbiBpZHMpXG4gICAgICBoYXJkY29kZSArPSBpZHNbaV0gKyBcIjogXCIgKyB0aGlzLm9wdGltaXplZC5tZW1vcnlbaV0gKyBcIixcXG5cIjtcbiAgICBoYXJkY29kZSA9IGhhcmRjb2RlLnN1YnN0cmluZygwLCBoYXJkY29kZS5sZW5ndGggLSAyKSArIFwiXFxufTtcXG5cIjtcbiAgICBoYXJkY29kZSA9IFwidmFyIHJ1biA9IFwiICsgYWN0aXZhdGlvbi5yZXBsYWNlKC9GXFxbKFxcZCspXS9nLCBmdW5jdGlvbihcbiAgICAgIGluZGV4KSB7XG4gICAgICByZXR1cm4gJ0ZbJyArIGlkc1tpbmRleC5tYXRjaCgvXFxkKy8pWzBdXSArICddJ1xuICAgIH0pLnJlcGxhY2UoXCJ7XFxuXCIsIFwie1xcblwiICsgaGFyZGNvZGUgKyBcIlwiKSArIFwiO1xcblwiO1xuICAgIGhhcmRjb2RlICs9IFwicmV0dXJuIHJ1blwiO1xuXG4gICAgLy8gcmV0dXJuIHN0YW5kYWxvbmUgZnVuY3Rpb25cbiAgICByZXR1cm4gbmV3IEZ1bmN0aW9uKGhhcmRjb2RlKSgpO1xuICB9LFxuXG4gIHdvcmtlcjogZnVuY3Rpb24oKSB7XG4gICAgaWYgKCF0aGlzLm9wdGltaXplZClcbiAgICAgIHRoaXMub3B0aW1pemUoKTtcblxuICAgIHZhciBoYXJkY29kZSA9IFwidmFyIGlucHV0cyA9IFwiICsgdGhpcy5vcHRpbWl6ZWQuZGF0YS5pbnB1dHMubGVuZ3RoICtcbiAgICAgIFwiO1xcblwiO1xuICAgIGhhcmRjb2RlICs9IFwidmFyIG91dHB1dHMgPSBcIiArIHRoaXMub3B0aW1pemVkLmRhdGEub3V0cHV0cy5sZW5ndGggK1xuICAgICAgXCI7XFxuXCI7XG4gICAgaGFyZGNvZGUgKz0gXCJ2YXIgRiA9IG51bGw7XFxuXCI7XG4gICAgaGFyZGNvZGUgKz0gXCJ2YXIgYWN0aXZhdGUgPSBcIiArIHRoaXMub3B0aW1pemVkLmFjdGl2YXRlLnRvU3RyaW5nKCkgK1xuICAgICAgXCI7XFxuXCI7XG4gICAgaGFyZGNvZGUgKz0gXCJ2YXIgcHJvcGFnYXRlID0gXCIgKyB0aGlzLm9wdGltaXplZC5wcm9wYWdhdGUudG9TdHJpbmcoKSArXG4gICAgICBcIjtcXG5cIjtcbiAgICBoYXJkY29kZSArPSBcIm9ubWVzc2FnZSA9IGZ1bmN0aW9uKGUpe1xcblwiO1xuICAgIGhhcmRjb2RlICs9IFwiRiA9IGUuZGF0YS5tZW1vcnlCdWZmZXI7XFxuXCI7XG4gICAgaGFyZGNvZGUgKz0gXCJpZiAoZS5kYXRhLmFjdGlvbiA9PSAnYWN0aXZhdGUnKXtcXG5cIjtcbiAgICBoYXJkY29kZSArPSBcImlmIChlLmRhdGEuaW5wdXQubGVuZ3RoID09IGlucHV0cyl7XFxuXCI7XG4gICAgaGFyZGNvZGUgKz1cbiAgICAgIFwicG9zdE1lc3NhZ2UoIHsgYWN0aW9uOiAnYWN0aXZhdGUnLCBvdXRwdXQ6IGFjdGl2YXRlKGUuZGF0YS5pbnB1dCksIG1lbW9yeUJ1ZmZlcjogRiB9LCBbRi5idWZmZXJdKTtcXG5cIjtcbiAgICBoYXJkY29kZSArPSBcIn1cXG59XFxuZWxzZSBpZiAoZS5kYXRhLmFjdGlvbiA9PSAncHJvcGFnYXRlJyl7XFxuXCI7XG4gICAgaGFyZGNvZGUgKz0gXCJwcm9wYWdhdGUoZS5kYXRhLnJhdGUsIGUuZGF0YS50YXJnZXQpO1xcblwiO1xuICAgIGhhcmRjb2RlICs9XG4gICAgICBcInBvc3RNZXNzYWdlKHsgYWN0aW9uOiAncHJvcGFnYXRlJywgbWVtb3J5QnVmZmVyOiBGIH0sIFtGLmJ1ZmZlcl0pO1xcblwiO1xuICAgIGhhcmRjb2RlICs9IFwifVxcbn1cXG5cIjtcblxuICAgIHZhciBibG9iID0gbmV3IEJsb2IoW2hhcmRjb2RlXSk7XG4gICAgdmFyIGJsb2JVUkwgPSB3aW5kb3cuVVJMLmNyZWF0ZU9iamVjdFVSTChibG9iKTtcblxuICAgIHJldHVybiBuZXcgV29ya2VyKGJsb2JVUkwpO1xuICB9LFxuXG4gIC8vIHJldHVybnMgYSBjb3B5IG9mIHRoZSBuZXR3b3JrXG4gIGNsb25lOiBmdW5jdGlvbihpZ25vcmVUcmFjZXMpIHtcbiAgICByZXR1cm4gTmV0d29yay5mcm9tSlNPTih0aGlzLnRvSlNPTihpZ25vcmVUcmFjZXMpKTtcbiAgfVxufVxuXG4vLyByZWJ1aWxkIGEgbmV0d29yayB0aGF0IGhhcyBiZWVuIHN0b3JlZCBpbiBhIGpzb24gdXNpbmcgdGhlIG1ldGhvZCB0b0pzb24oKVxuTmV0d29yay5mcm9tSlNPTiA9IGZ1bmN0aW9uKGpzb24pIHtcblxuICB2YXIgbmV1cm9ucyA9IFtdO1xuXG4gIHZhciBsYXllcnMgPSB7XG4gICAgaW5wdXQ6IG5ldyBMYXllcigpLFxuICAgIGhpZGRlbjogW10sXG4gICAgb3V0cHV0OiBuZXcgTGF5ZXIoKVxuICB9XG5cbiAgZm9yICh2YXIgaSBpbiBqc29uLm5ldXJvbnMpIHtcbiAgICB2YXIgY29uZmlnID0ganNvbi5uZXVyb25zW2ldO1xuXG4gICAgdmFyIG5ldXJvbiA9IG5ldyBOZXVyb24oKTtcbiAgICBuZXVyb24udHJhY2UuZWxlZ2liaWxpdHkgPSBjb25maWcudHJhY2UuZWxlZ2liaWxpdHk7XG4gICAgbmV1cm9uLnRyYWNlLmV4dGVuZGVkID0gY29uZmlnLnRyYWNlLmV4dGVuZGVkO1xuICAgIG5ldXJvbi5zdGF0ZSA9IGNvbmZpZy5zdGF0ZTtcbiAgICBuZXVyb24ub2xkID0gY29uZmlnLm9sZDtcbiAgICBuZXVyb24uYWN0aXZhdGlvbiA9IGNvbmZpZy5hY3RpdmF0aW9uO1xuICAgIG5ldXJvbi5iaWFzID0gY29uZmlnLmJpYXM7XG4gICAgbmV1cm9uLnNxdWFzaCA9IGNvbmZpZy5zcXVhc2ggaW4gTmV1cm9uLnNxdWFzaCA/IE5ldXJvbi5zcXVhc2hbY29uZmlnLnNxdWFzaF0gOlxuICAgICAgTmV1cm9uLnNxdWFzaC5MT0dJU1RJQztcbiAgICBuZXVyb25zLnB1c2gobmV1cm9uKTtcblxuICAgIGlmIChjb25maWcubGF5ZXIgPT0gJ2lucHV0JylcbiAgICAgIGxheWVycy5pbnB1dC5hZGQobmV1cm9uKTtcbiAgICBlbHNlIGlmIChjb25maWcubGF5ZXIgPT0gJ291dHB1dCcpXG4gICAgICBsYXllcnMub3V0cHV0LmFkZChuZXVyb24pO1xuICAgIGVsc2Uge1xuICAgICAgaWYgKHR5cGVvZiBsYXllcnMuaGlkZGVuW2NvbmZpZy5sYXllcl0gPT0gJ3VuZGVmaW5lZCcpXG4gICAgICAgIGxheWVycy5oaWRkZW5bY29uZmlnLmxheWVyXSA9IG5ldyBMYXllcigpO1xuICAgICAgbGF5ZXJzLmhpZGRlbltjb25maWcubGF5ZXJdLmFkZChuZXVyb24pO1xuICAgIH1cbiAgfVxuXG4gIGZvciAodmFyIGkgaW4ganNvbi5jb25uZWN0aW9ucykge1xuICAgIHZhciBjb25maWcgPSBqc29uLmNvbm5lY3Rpb25zW2ldO1xuICAgIHZhciBmcm9tID0gbmV1cm9uc1tjb25maWcuZnJvbV07XG4gICAgdmFyIHRvID0gbmV1cm9uc1tjb25maWcudG9dO1xuICAgIHZhciB3ZWlnaHQgPSBjb25maWcud2VpZ2h0XG4gICAgdmFyIGdhdGVyID0gbmV1cm9uc1tjb25maWcuZ2F0ZXJdO1xuXG4gICAgdmFyIGNvbm5lY3Rpb24gPSBmcm9tLnByb2plY3QodG8sIHdlaWdodCk7XG4gICAgaWYgKGdhdGVyKVxuICAgICAgZ2F0ZXIuZ2F0ZShjb25uZWN0aW9uKTtcbiAgfVxuXG4gIHJldHVybiBuZXcgTmV0d29yayhsYXllcnMpO1xufVxuXG4vLyBleHBvcnRcbmlmIChtb2R1bGUpIG1vZHVsZS5leHBvcnRzID0gTmV0d29yaztcblxuIiwiLyoqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBORVVST05cbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cbmZ1bmN0aW9uIE5ldXJvbigpIHtcbiAgdGhpcy5JRCA9IE5ldXJvbi51aWQoKTtcbiAgdGhpcy5sYWJlbCA9IG51bGw7XG4gIHRoaXMuY29ubmVjdGlvbnMgPSB7XG4gICAgaW5wdXRzOiB7fSxcbiAgICBwcm9qZWN0ZWQ6IHt9LFxuICAgIGdhdGVkOiB7fVxuICB9O1xuICB0aGlzLmVycm9yID0ge1xuICAgIHJlc3BvbnNpYmlsaXR5OiAwLFxuICAgIHByb2plY3RlZDogMCxcbiAgICBnYXRlZDogMFxuICB9O1xuICB0aGlzLnRyYWNlID0ge1xuICAgIGVsZWdpYmlsaXR5OiB7fSxcbiAgICBleHRlbmRlZDoge30sXG4gICAgaW5mbHVlbmNlczoge31cbiAgfTtcbiAgdGhpcy5zdGF0ZSA9IDA7XG4gIHRoaXMub2xkID0gMDtcbiAgdGhpcy5hY3RpdmF0aW9uID0gMDtcbiAgdGhpcy5zZWxmY29ubmVjdGlvbiA9IG5ldyBOZXVyb24uY29ubmVjdGlvbih0aGlzLCB0aGlzLCAwKTsgLy8gd2VpZ2h0ID0gMCAtPiBub3QgY29ubmVjdGVkXG4gIHRoaXMuc3F1YXNoID0gTmV1cm9uLnNxdWFzaC5MT0dJU1RJQztcbiAgdGhpcy5uZWlnaGJvb3JzID0ge307XG4gIHRoaXMuYmlhcyA9IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xO1xufVxuXG5OZXVyb24ucHJvdG90eXBlID0ge1xuXG4gIC8vIGFjdGl2YXRlIHRoZSBuZXVyb25cbiAgYWN0aXZhdGU6IGZ1bmN0aW9uKGlucHV0KSB7XG4gICAgLy8gYWN0aXZhdGlvbiBmcm9tIGVudmlyb21lbnQgKGZvciBpbnB1dCBuZXVyb25zKVxuICAgIGlmICh0eXBlb2YgaW5wdXQgIT0gJ3VuZGVmaW5lZCcpIHtcbiAgICAgIHRoaXMuYWN0aXZhdGlvbiA9IGlucHV0O1xuICAgICAgdGhpcy5kZXJpdmF0aXZlID0gMDtcbiAgICAgIHRoaXMuYmlhcyA9IDA7XG4gICAgICByZXR1cm4gdGhpcy5hY3RpdmF0aW9uO1xuICAgIH1cblxuICAgIC8vIG9sZCBzdGF0ZVxuICAgIHRoaXMub2xkID0gdGhpcy5zdGF0ZTtcblxuICAgIC8vIGVxLiAxNVxuICAgIHRoaXMuc3RhdGUgPSB0aGlzLnNlbGZjb25uZWN0aW9uLmdhaW4gKiB0aGlzLnNlbGZjb25uZWN0aW9uLndlaWdodCAqXG4gICAgICB0aGlzLnN0YXRlICsgdGhpcy5iaWFzO1xuXG4gICAgZm9yICh2YXIgaSBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuICAgICAgdmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG4gICAgICB0aGlzLnN0YXRlICs9IGlucHV0LmZyb20uYWN0aXZhdGlvbiAqIGlucHV0LndlaWdodCAqIGlucHV0LmdhaW47XG4gICAgfVxuXG4gICAgLy8gZXEuIDE2XG4gICAgdGhpcy5hY3RpdmF0aW9uID0gdGhpcy5zcXVhc2godGhpcy5zdGF0ZSk7XG5cbiAgICAvLyBmJyhzKVxuICAgIHRoaXMuZGVyaXZhdGl2ZSA9IHRoaXMuc3F1YXNoKHRoaXMuc3RhdGUsIHRydWUpO1xuXG4gICAgLy8gdXBkYXRlIHRyYWNlc1xuICAgIHZhciBpbmZsdWVuY2VzID0gW107XG4gICAgZm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuICAgICAgLy8gZXh0ZW5kZWQgZWxlZ2liaWxpdHkgdHJhY2VcbiAgICAgIHZhciB4dHJhY2UgPSB0aGlzLnRyYWNlLmV4dGVuZGVkW2lkXTtcbiAgICAgIHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXG4gICAgICAvLyBpZiBnYXRlZCBuZXVyb24ncyBzZWxmY29ubmVjdGlvbiBpcyBnYXRlZCBieSB0aGlzIHVuaXQsIHRoZSBpbmZsdWVuY2Uga2VlcHMgdHJhY2sgb2YgdGhlIG5ldXJvbidzIG9sZCBzdGF0ZVxuICAgICAgdmFyIGluZmx1ZW5jZSA9IG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzID8gbmV1cm9uLm9sZCA6IDA7XG5cbiAgICAgIC8vIGluZGV4IHJ1bnMgb3ZlciBhbGwgdGhlIGluY29taW5nIGNvbm5lY3Rpb25zIHRvIHRoZSBnYXRlZCBuZXVyb24gdGhhdCBhcmUgZ2F0ZWQgYnkgdGhpcyB1bml0XG4gICAgICBmb3IgKHZhciBpbmNvbWluZyBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXSkgeyAvLyBjYXB0dXJlcyB0aGUgZWZmZWN0IHRoYXQgaGFzIGFuIGlucHV0IGNvbm5lY3Rpb24gdG8gdGhpcyB1bml0LCBvbiBhIG5ldXJvbiB0aGF0IGlzIGdhdGVkIGJ5IHRoaXMgdW5pdFxuICAgICAgICBpbmZsdWVuY2UgKz0gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF1baW5jb21pbmddLndlaWdodCAqXG4gICAgICAgICAgdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF1baW5jb21pbmddLmZyb20uYWN0aXZhdGlvbjtcbiAgICAgIH1cbiAgICAgIGluZmx1ZW5jZXNbbmV1cm9uLklEXSA9IGluZmx1ZW5jZTtcbiAgICB9XG4gICAgXG4gICAgZm9yICh2YXIgaSBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuICAgICAgdmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cbiAgICAgIC8vIGVsZWdpYmlsaXR5IHRyYWNlIC0gRXEuIDE3XG4gICAgICB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0LklEXSA9IHRoaXMuc2VsZmNvbm5lY3Rpb24uZ2FpbiAqIHRoaXMuc2VsZmNvbm5lY3Rpb25cbiAgICAgICAgLndlaWdodCAqIHRoaXMudHJhY2UuZWxlZ2liaWxpdHlbaW5wdXQuSURdICsgaW5wdXQuZ2FpbiAqIGlucHV0LmZyb21cbiAgICAgICAgLmFjdGl2YXRpb247XG5cbiAgICAgIGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcbiAgICAgICAgLy8gZXh0ZW5kZWQgZWxlZ2liaWxpdHkgdHJhY2VcbiAgICAgICAgdmFyIHh0cmFjZSA9IHRoaXMudHJhY2UuZXh0ZW5kZWRbaWRdO1xuICAgICAgICB2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcbiAgICAgICAgdmFyIGluZmx1ZW5jZSA9IGluZmx1ZW5jZXNbbmV1cm9uLklEXTtcblxuICAgICAgICAvLyBlcS4gMThcbiAgICAgICAgeHRyYWNlW2lucHV0LklEXSA9IG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYWluICogbmV1cm9uLnNlbGZjb25uZWN0aW9uXG4gICAgICAgICAgLndlaWdodCAqIHh0cmFjZVtpbnB1dC5JRF0gKyB0aGlzLmRlcml2YXRpdmUgKiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W1xuICAgICAgICAgICAgaW5wdXQuSURdICogaW5mbHVlbmNlO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vICB1cGRhdGUgZ2F0ZWQgY29ubmVjdGlvbidzIGdhaW5zXG4gICAgZm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zLmdhdGVkKSB7XG4gICAgICB0aGlzLmNvbm5lY3Rpb25zLmdhdGVkW2Nvbm5lY3Rpb25dLmdhaW4gPSB0aGlzLmFjdGl2YXRpb247XG4gICAgfVxuXG4gICAgcmV0dXJuIHRoaXMuYWN0aXZhdGlvbjtcbiAgfSxcblxuICAvLyBiYWNrLXByb3BhZ2F0ZSB0aGUgZXJyb3JcbiAgcHJvcGFnYXRlOiBmdW5jdGlvbihyYXRlLCB0YXJnZXQpIHtcbiAgICAvLyBlcnJvciBhY2N1bXVsYXRvclxuICAgIHZhciBlcnJvciA9IDA7XG5cbiAgICAvLyB3aGV0aGVyIG9yIG5vdCB0aGlzIG5ldXJvbiBpcyBpbiB0aGUgb3V0cHV0IGxheWVyXG4gICAgdmFyIGlzT3V0cHV0ID0gdHlwZW9mIHRhcmdldCAhPSAndW5kZWZpbmVkJztcblxuICAgIC8vIG91dHB1dCBuZXVyb25zIGdldCB0aGVpciBlcnJvciBmcm9tIHRoZSBlbnZpcm9tZW50XG4gICAgaWYgKGlzT3V0cHV0KVxuICAgICAgdGhpcy5lcnJvci5yZXNwb25zaWJpbGl0eSA9IHRoaXMuZXJyb3IucHJvamVjdGVkID0gdGFyZ2V0IC0gdGhpcy5hY3RpdmF0aW9uOyAvLyBFcS4gMTBcbiAgICBcbiAgICBlbHNlIC8vIHRoZSByZXN0IG9mIHRoZSBuZXVyb24gY29tcHV0ZSB0aGVpciBlcnJvciByZXNwb25zaWJpbGl0aWVzIGJ5IGJhY2twcm9wYWdhdGlvblxuICAgIHtcbiAgICAgIC8vIGVycm9yIHJlc3BvbnNpYmlsaXRpZXMgZnJvbSBhbGwgdGhlIGNvbm5lY3Rpb25zIHByb2plY3RlZCBmcm9tIHRoaXMgbmV1cm9uXG4gICAgICBmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZCkge1xuICAgICAgICB2YXIgY29ubmVjdGlvbiA9IHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkW2lkXTtcbiAgICAgICAgdmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24udG87XG4gICAgICAgIC8vIEVxLiAyMVxuICAgICAgICBlcnJvciArPSBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkgKiBjb25uZWN0aW9uLmdhaW4gKiBjb25uZWN0aW9uLndlaWdodDtcbiAgICAgIH1cblxuICAgICAgLy8gcHJvamVjdGVkIGVycm9yIHJlc3BvbnNpYmlsaXR5XG4gICAgICB0aGlzLmVycm9yLnByb2plY3RlZCA9IHRoaXMuZGVyaXZhdGl2ZSAqIGVycm9yO1xuXG4gICAgICBlcnJvciA9IDA7XG4gICAgICAvLyBlcnJvciByZXNwb25zaWJpbGl0aWVzIGZyb20gYWxsIHRoZSBjb25uZWN0aW9ucyBnYXRlZCBieSB0aGlzIG5ldXJvblxuICAgICAgZm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuICAgICAgICB2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTsgLy8gZ2F0ZWQgbmV1cm9uXG4gICAgICAgIHZhciBpbmZsdWVuY2UgPSBuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIgPT0gdGhpcyA/IG5ldXJvbi5vbGQgOiAwOyAvLyBpZiBnYXRlZCBuZXVyb24ncyBzZWxmY29ubmVjdGlvbiBpcyBnYXRlZCBieSB0aGlzIG5ldXJvblxuXG4gICAgICAgIC8vIGluZGV4IHJ1bnMgb3ZlciBhbGwgdGhlIGNvbm5lY3Rpb25zIHRvIHRoZSBnYXRlZCBuZXVyb24gdGhhdCBhcmUgZ2F0ZWQgYnkgdGhpcyBuZXVyb25cbiAgICAgICAgZm9yICh2YXIgaW5wdXQgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzW2lkXSkgeyAvLyBjYXB0dXJlcyB0aGUgZWZmZWN0IHRoYXQgdGhlIGlucHV0IGNvbm5lY3Rpb24gb2YgdGhpcyBuZXVyb24gaGF2ZSwgb24gYSBuZXVyb24gd2hpY2ggaXRzIGlucHV0L3MgaXMvYXJlIGdhdGVkIGJ5IHRoaXMgbmV1cm9uXG4gICAgICAgICAgaW5mbHVlbmNlICs9IHRoaXMudHJhY2UuaW5mbHVlbmNlc1tpZF1baW5wdXRdLndlaWdodCAqIHRoaXMudHJhY2UuaW5mbHVlbmNlc1tcbiAgICAgICAgICAgIG5ldXJvbi5JRF1baW5wdXRdLmZyb20uYWN0aXZhdGlvbjtcbiAgICAgICAgfVxuICAgICAgICAvLyBlcS4gMjJcbiAgICAgICAgZXJyb3IgKz0gbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5ICogaW5mbHVlbmNlO1xuICAgICAgfVxuXG4gICAgICAvLyBnYXRlZCBlcnJvciByZXNwb25zaWJpbGl0eVxuICAgICAgdGhpcy5lcnJvci5nYXRlZCA9IHRoaXMuZGVyaXZhdGl2ZSAqIGVycm9yO1xuXG4gICAgICAvLyBlcnJvciByZXNwb25zaWJpbGl0eSAtIEVxLiAyM1xuICAgICAgdGhpcy5lcnJvci5yZXNwb25zaWJpbGl0eSA9IHRoaXMuZXJyb3IucHJvamVjdGVkICsgdGhpcy5lcnJvci5nYXRlZDtcbiAgICB9XG5cbiAgICAvLyBsZWFybmluZyByYXRlXG4gICAgcmF0ZSA9IHJhdGUgfHwgLjE7XG5cbiAgICAvLyBhZGp1c3QgYWxsIHRoZSBuZXVyb24ncyBpbmNvbWluZyBjb25uZWN0aW9uc1xuICAgIGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG4gICAgICB2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cbiAgICAgIC8vIEVxLiAyNFxuICAgICAgdmFyIGdyYWRpZW50ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgKiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0LklEXTtcbiAgICAgIGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcbiAgICAgICAgdmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG4gICAgICAgIGdyYWRpZW50ICs9IG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSAqIHRoaXMudHJhY2UuZXh0ZW5kZWRbXG4gICAgICAgICAgbmV1cm9uLklEXVtpbnB1dC5JRF07XG4gICAgICB9XG4gICAgICBpbnB1dC53ZWlnaHQgKz0gcmF0ZSAqIGdyYWRpZW50OyAvLyBhZGp1c3Qgd2VpZ2h0cyAtIGFrYSBsZWFyblxuICAgIH1cblxuICAgIC8vIGFkanVzdCBiaWFzXG4gICAgdGhpcy5iaWFzICs9IHJhdGUgKiB0aGlzLmVycm9yLnJlc3BvbnNpYmlsaXR5O1xuICB9LFxuXG4gIHByb2plY3Q6IGZ1bmN0aW9uKG5ldXJvbiwgd2VpZ2h0KSB7XG4gICAgLy8gc2VsZi1jb25uZWN0aW9uXG4gICAgaWYgKG5ldXJvbiA9PSB0aGlzKSB7XG4gICAgICB0aGlzLnNlbGZjb25uZWN0aW9uLndlaWdodCA9IDE7XG4gICAgICByZXR1cm4gdGhpcy5zZWxmY29ubmVjdGlvbjtcbiAgICB9XG5cbiAgICAvLyBjaGVjayBpZiBjb25uZWN0aW9uIGFscmVhZHkgZXhpc3RzXG4gICAgdmFyIGNvbm5lY3RlZCA9IHRoaXMuY29ubmVjdGVkKG5ldXJvbik7XG4gICAgaWYgKGNvbm5lY3RlZCAmJiBjb25uZWN0ZWQudHlwZSA9PSBcInByb2plY3RlZFwiKSB7XG4gICAgICAvLyB1cGRhdGUgY29ubmVjdGlvblxuICAgICAgaWYgKHR5cGVvZiB3ZWlnaHQgIT0gJ3VuZGVmaW5lZCcpXG4gICAgICAgIGNvbm5lY3RlZC5jb25uZWN0aW9uLndlaWdodCA9IHdlaWdodDtcbiAgICAgIC8vIHJldHVybiBleGlzdGluZyBjb25uZWN0aW9uXG4gICAgICByZXR1cm4gY29ubmVjdGVkLmNvbm5lY3Rpb247XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIGNyZWF0ZSBhIG5ldyBjb25uZWN0aW9uXG4gICAgICB2YXIgY29ubmVjdGlvbiA9IG5ldyBOZXVyb24uY29ubmVjdGlvbih0aGlzLCBuZXVyb24sIHdlaWdodCk7XG4gICAgfVxuXG4gICAgLy8gcmVmZXJlbmNlIGFsbCB0aGUgY29ubmVjdGlvbnMgYW5kIHRyYWNlc1xuICAgIHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcbiAgICB0aGlzLm5laWdoYm9vcnNbbmV1cm9uLklEXSA9IG5ldXJvbjtcbiAgICBuZXVyb24uY29ubmVjdGlvbnMuaW5wdXRzW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcbiAgICBuZXVyb24udHJhY2UuZWxlZ2liaWxpdHlbY29ubmVjdGlvbi5JRF0gPSAwO1xuXG4gICAgZm9yICh2YXIgaWQgaW4gbmV1cm9uLnRyYWNlLmV4dGVuZGVkKSB7XG4gICAgICB2YXIgdHJhY2UgPSBuZXVyb24udHJhY2UuZXh0ZW5kZWRbaWRdO1xuICAgICAgdHJhY2VbY29ubmVjdGlvbi5JRF0gPSAwO1xuICAgIH1cblxuICAgIHJldHVybiBjb25uZWN0aW9uO1xuICB9LFxuXG4gIGdhdGU6IGZ1bmN0aW9uKGNvbm5lY3Rpb24pIHtcbiAgICAvLyBhZGQgY29ubmVjdGlvbiB0byBnYXRlZCBsaXN0XG4gICAgdGhpcy5jb25uZWN0aW9ucy5nYXRlZFtjb25uZWN0aW9uLklEXSA9IGNvbm5lY3Rpb247XG5cbiAgICB2YXIgbmV1cm9uID0gY29ubmVjdGlvbi50bztcbiAgICBpZiAoIShuZXVyb24uSUQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkpIHtcbiAgICAgIC8vIGV4dGVuZGVkIHRyYWNlXG4gICAgICB0aGlzLm5laWdoYm9vcnNbbmV1cm9uLklEXSA9IG5ldXJvbjtcbiAgICAgIHZhciB4dHJhY2UgPSB0aGlzLnRyYWNlLmV4dGVuZGVkW25ldXJvbi5JRF0gPSB7fTtcbiAgICAgIGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG4gICAgICAgIHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcbiAgICAgICAgeHRyYWNlW2lucHV0LklEXSA9IDA7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8ga2VlcCB0cmFja1xuICAgIGlmIChuZXVyb24uSUQgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzKVxuICAgICAgdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0ucHVzaChjb25uZWN0aW9uKTtcbiAgICBlbHNlXG4gICAgICB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXSA9IFtjb25uZWN0aW9uXTtcblxuICAgIC8vIHNldCBnYXRlclxuICAgIGNvbm5lY3Rpb24uZ2F0ZXIgPSB0aGlzO1xuICB9LFxuXG4gIC8vIHJldHVybnMgdHJ1ZSBvciBmYWxzZSB3aGV0aGVyIHRoZSBuZXVyb24gaXMgc2VsZi1jb25uZWN0ZWQgb3Igbm90XG4gIHNlbGZjb25uZWN0ZWQ6IGZ1bmN0aW9uKCkge1xuICAgIHJldHVybiB0aGlzLnNlbGZjb25uZWN0aW9uLndlaWdodCAhPT0gMDtcbiAgfSxcblxuICAvLyByZXR1cm5zIHRydWUgb3IgZmFsc2Ugd2hldGhlciB0aGUgbmV1cm9uIGlzIGNvbm5lY3RlZCB0byBhbm90aGVyIG5ldXJvbiAocGFyYW1ldGVyKVxuICBjb25uZWN0ZWQ6IGZ1bmN0aW9uKG5ldXJvbikge1xuICAgIHZhciByZXN1bHQgPSB7XG4gICAgICB0eXBlOiBudWxsLFxuICAgICAgY29ubmVjdGlvbjogZmFsc2VcbiAgICB9O1xuXG4gICAgaWYgKHRoaXMgPT0gbmV1cm9uKSB7XG4gICAgICBpZiAodGhpcy5zZWxmY29ubmVjdGVkKCkpIHtcbiAgICAgICAgcmVzdWx0LnR5cGUgPSAnc2VsZmNvbm5lY3Rpb24nO1xuICAgICAgICByZXN1bHQuY29ubmVjdGlvbiA9IHRoaXMuc2VsZmNvbm5lY3Rpb247XG4gICAgICAgIHJldHVybiByZXN1bHQ7XG4gICAgICB9IGVsc2VcbiAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cblxuICAgIGZvciAodmFyIHR5cGUgaW4gdGhpcy5jb25uZWN0aW9ucykge1xuICAgICAgZm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zW3R5cGVdKSB7XG4gICAgICAgIHZhciBjb25uZWN0aW9uID0gdGhpcy5jb25uZWN0aW9uc1t0eXBlXVtjb25uZWN0aW9uXTtcbiAgICAgICAgaWYgKGNvbm5lY3Rpb24udG8gPT0gbmV1cm9uKSB7XG4gICAgICAgICAgcmVzdWx0LnR5cGUgPSB0eXBlO1xuICAgICAgICAgIHJlc3VsdC5jb25uZWN0aW9uID0gY29ubmVjdGlvbjtcbiAgICAgICAgICByZXR1cm4gcmVzdWx0O1xuICAgICAgICB9IGVsc2UgaWYgKGNvbm5lY3Rpb24uZnJvbSA9PSBuZXVyb24pIHtcbiAgICAgICAgICByZXN1bHQudHlwZSA9IHR5cGU7XG4gICAgICAgICAgcmVzdWx0LmNvbm5lY3Rpb24gPSBjb25uZWN0aW9uO1xuICAgICAgICAgIHJldHVybiByZXN1bHQ7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gZmFsc2U7XG4gIH0sXG5cbiAgLy8gY2xlYXJzIGFsbCB0aGUgdHJhY2VzICh0aGUgbmV1cm9uIGZvcmdldHMgaXQncyBjb250ZXh0LCBidXQgdGhlIGNvbm5lY3Rpb25zIHJlbWFpbiBpbnRhY3QpXG4gIGNsZWFyOiBmdW5jdGlvbigpIHtcblxuICAgIGZvciAodmFyIHRyYWNlIGluIHRoaXMudHJhY2UuZWxlZ2liaWxpdHkpXG4gICAgICB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RyYWNlXSA9IDA7XG5cbiAgICBmb3IgKHZhciB0cmFjZSBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKVxuICAgICAgZm9yICh2YXIgZXh0ZW5kZWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZFt0cmFjZV0pXG4gICAgICAgIHRoaXMudHJhY2UuZXh0ZW5kZWRbdHJhY2VdW2V4dGVuZGVkXSA9IDA7XG5cbiAgICB0aGlzLmVycm9yLnJlc3BvbnNpYmlsaXR5ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgPSB0aGlzLmVycm9yLmdhdGVkID0gMDtcbiAgfSxcblxuICAvLyBhbGwgdGhlIGNvbm5lY3Rpb25zIGFyZSByYW5kb21pemVkIGFuZCB0aGUgdHJhY2VzIGFyZSBjbGVhcmVkXG4gIHJlc2V0OiBmdW5jdGlvbigpIHtcbiAgICB0aGlzLmNsZWFyKCk7XG5cbiAgICBmb3IgKHZhciB0eXBlIGluIHRoaXMuY29ubmVjdGlvbilcbiAgICAgIGZvciAodmFyIGNvbm5lY3Rpb24gaW4gdGhpcy5jb25uZWN0aW9uW3R5cGVdKVxuICAgICAgICB0aGlzLmNvbm5lY3Rpb25bdHlwZV1bY29ubmVjdGlvbl0ud2VpZ2h0ID0gTWF0aC5yYW5kb20oKSAqIC4yIC0gLjE7XG4gICAgdGhpcy5iaWFzID0gTWF0aC5yYW5kb20oKSAqIC4yIC0gLjE7XG5cbiAgICB0aGlzLm9sZCA9IHRoaXMuc3RhdGUgPSB0aGlzLmFjdGl2YXRpb24gPSAwO1xuICB9LFxuXG4gIC8vIGhhcmRjb2RlcyB0aGUgYmVoYXZpb3VyIG9mIHRoZSBuZXVyb24gaW50byBhbiBvcHRpbWl6ZWQgZnVuY3Rpb25cbiAgb3B0aW1pemU6IGZ1bmN0aW9uKG9wdGltaXplZCwgbGF5ZXIpIHtcbiAgICBcbiAgICBvcHRpbWl6ZWQgPSBvcHRpbWl6ZWQgfHwge307XG4gICAgdmFyIHRoYXQgPSB0aGlzO1xuICAgIHZhciBzdG9yZV9hY3RpdmF0aW9uID0gW107XG4gICAgdmFyIHN0b3JlX3RyYWNlID0gW107XG4gICAgdmFyIHN0b3JlX3Byb3BhZ2F0aW9uID0gW107XG4gICAgdmFyIHZhcklEID0gb3B0aW1pemVkLm1lbW9yeSB8fCAwO1xuICAgIHZhciBuZXVyb25zID0gb3B0aW1pemVkLm5ldXJvbnMgfHwgMTtcbiAgICB2YXIgaW5wdXRzID0gb3B0aW1pemVkLmlucHV0cyB8fCBbXTtcbiAgICB2YXIgdGFyZ2V0cyA9IG9wdGltaXplZC50YXJnZXRzIHx8IFtdO1xuICAgIHZhciBvdXRwdXRzID0gb3B0aW1pemVkLm91dHB1dHMgfHwgW107XG4gICAgdmFyIHZhcmlhYmxlcyA9IG9wdGltaXplZC52YXJpYWJsZXMgfHwge307XG4gICAgdmFyIGFjdGl2YXRpb25fc2VudGVuY2VzID0gb3B0aW1pemVkLmFjdGl2YXRpb25fc2VudGVuY2VzIHx8IFtdO1xuICAgIHZhciB0cmFjZV9zZW50ZW5jZXMgPSBvcHRpbWl6ZWQudHJhY2Vfc2VudGVuY2VzIHx8IFtdO1xuICAgIHZhciBwcm9wYWdhdGlvbl9zZW50ZW5jZXMgPSBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzIHx8IFtdO1xuICAgIHZhciBsYXllcnMgPSBvcHRpbWl6ZWQubGF5ZXJzIHx8IHsgX19jb3VudDogMCwgX19uZXVyb246IDAgfTtcblxuICAgIC8vIGFsbG9jYXRlIHNlbnRlbmNlc1xuICAgIHZhciBhbGxvY2F0ZSA9IGZ1bmN0aW9uKHN0b3JlKXtcbiAgICAgIHZhciBhbGxvY2F0ZWQgPSBsYXllciBpbiBsYXllcnMgJiYgc3RvcmVbbGF5ZXJzLl9fY291bnRdO1xuICAgICAgaWYgKCFhbGxvY2F0ZWQpXG4gICAgICB7XG4gICAgICAgIGxheWVycy5fX2NvdW50ID0gc3RvcmUucHVzaChbXSkgLSAxO1xuICAgICAgICBsYXllcnNbbGF5ZXJdID0gbGF5ZXJzLl9fY291bnQ7XG4gICAgICB9XG4gICAgfVxuICAgIGFsbG9jYXRlKGFjdGl2YXRpb25fc2VudGVuY2VzKTtcbiAgICBhbGxvY2F0ZSh0cmFjZV9zZW50ZW5jZXMpO1xuICAgIGFsbG9jYXRlKHByb3BhZ2F0aW9uX3NlbnRlbmNlcyk7XG4gICAgdmFyIGN1cnJlbnRMYXllciA9IGxheWVycy5fX2NvdW50O1xuXG4gICAgLy8gZ2V0L3Jlc2VydmUgc3BhY2UgaW4gbWVtb3J5IGJ5IGNyZWF0aW5nIGEgdW5pcXVlIElEIGZvciBhIHZhcmlhYmxlbFxuICAgIHZhciBnZXRWYXIgPSBmdW5jdGlvbigpIHtcbiAgICAgIHZhciBhcmdzID0gQXJyYXkucHJvdG90eXBlLnNsaWNlLmNhbGwoYXJndW1lbnRzKTtcblxuICAgICAgaWYgKGFyZ3MubGVuZ3RoID09IDEpIHtcbiAgICAgICAgaWYgKGFyZ3NbMF0gPT0gJ3RhcmdldCcpIHtcbiAgICAgICAgICB2YXIgaWQgPSAndGFyZ2V0XycgKyB0YXJnZXRzLmxlbmd0aDtcbiAgICAgICAgICB0YXJnZXRzLnB1c2godmFySUQpO1xuICAgICAgICB9IGVsc2VcbiAgICAgICAgICB2YXIgaWQgPSBhcmdzWzBdO1xuICAgICAgICBpZiAoaWQgaW4gdmFyaWFibGVzKVxuICAgICAgICAgIHJldHVybiB2YXJpYWJsZXNbaWRdO1xuICAgICAgICByZXR1cm4gdmFyaWFibGVzW2lkXSA9IHtcbiAgICAgICAgICB2YWx1ZTogMCxcbiAgICAgICAgICBpZDogdmFySUQrK1xuICAgICAgICB9O1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdmFyIGV4dGVuZGVkID0gYXJncy5sZW5ndGggPiAyO1xuICAgICAgICBpZiAoZXh0ZW5kZWQpXG4gICAgICAgICAgdmFyIHZhbHVlID0gYXJncy5wb3AoKTtcblxuICAgICAgICB2YXIgdW5pdCA9IGFyZ3Muc2hpZnQoKTtcbiAgICAgICAgdmFyIHByb3AgPSBhcmdzLnBvcCgpO1xuXG4gICAgICAgIGlmICghZXh0ZW5kZWQpXG4gICAgICAgICAgdmFyIHZhbHVlID0gdW5pdFtwcm9wXTtcblxuICAgICAgICB2YXIgaWQgPSBwcm9wICsgJ18nO1xuICAgICAgICBmb3IgKHZhciBwcm9wZXJ0eSBpbiBhcmdzKVxuICAgICAgICAgIGlkICs9IGFyZ3NbcHJvcGVydHldICsgJ18nO1xuICAgICAgICBpZCArPSB1bml0LklEO1xuICAgICAgICBpZiAoaWQgaW4gdmFyaWFibGVzKVxuICAgICAgICAgIHJldHVybiB2YXJpYWJsZXNbaWRdO1xuXG4gICAgICAgIHJldHVybiB2YXJpYWJsZXNbaWRdID0ge1xuICAgICAgICAgIHZhbHVlOiB2YWx1ZSxcbiAgICAgICAgICBpZDogdmFySUQrK1xuICAgICAgICB9O1xuICAgICAgfVxuICAgIH07XG5cbiAgICAvLyBidWlsZCBzZW50ZW5jZVxuICAgIHZhciBidWlsZFNlbnRlbmNlID0gZnVuY3Rpb24oKSB7XG4gICAgICB2YXIgYXJncyA9IEFycmF5LnByb3RvdHlwZS5zbGljZS5jYWxsKGFyZ3VtZW50cyk7XG4gICAgICB2YXIgc3RvcmUgPSBhcmdzLnBvcCgpO1xuICAgICAgdmFyIHNlbnRlbmNlID0gXCJcIjtcbiAgICAgIGZvciAodmFyIGkgaW4gYXJncylcbiAgICAgICAgaWYgKHR5cGVvZiBhcmdzW2ldID09ICdzdHJpbmcnKVxuICAgICAgICAgIHNlbnRlbmNlICs9IGFyZ3NbaV07XG4gICAgICAgIGVsc2VcbiAgICAgICAgICBzZW50ZW5jZSArPSAnRlsnICsgYXJnc1tpXS5pZCArICddJztcblxuICAgICAgc3RvcmUucHVzaChzZW50ZW5jZSArICc7Jyk7XG4gICAgfVxuXG4gICAgLy8gaGVscGVyIHRvIGNoZWNrIGlmIGFuIG9iamVjdCBpcyBlbXB0eVxuICAgIHZhciBpc0VtcHR5ID0gZnVuY3Rpb24ob2JqKSB7XG4gICAgICBmb3IgKHZhciBwcm9wIGluIG9iaikge1xuICAgICAgICBpZiAob2JqLmhhc093blByb3BlcnR5KHByb3ApKVxuICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIH1cbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH07XG5cbiAgICAvLyBjaGFyYWN0ZXJpc3RpY3Mgb2YgdGhlIG5ldXJvblxuICAgIHZhciBub1Byb2plY3Rpb25zID0gaXNFbXB0eSh0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZCk7XG4gICAgdmFyIG5vR2F0ZXMgPSBpc0VtcHR5KHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWQpO1xuICAgIHZhciBpc0lucHV0ID0gbGF5ZXIgPT0gJ2lucHV0JyA/IHRydWUgOiBpc0VtcHR5KHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKTtcbiAgICB2YXIgaXNPdXRwdXQgPSBsYXllciA9PSAnb3V0cHV0JyA/IHRydWUgOiBub1Byb2plY3Rpb25zICYmIG5vR2F0ZXM7XG5cbiAgICAvLyBvcHRpbWl6ZSBuZXVyb24ncyBiZWhhdmlvdXJcbiAgICB2YXIgcmF0ZSA9IGdldFZhcigncmF0ZScpO1xuICAgIHZhciBhY3RpdmF0aW9uID0gZ2V0VmFyKHRoaXMsICdhY3RpdmF0aW9uJyk7XG4gICAgaWYgKGlzSW5wdXQpXG4gICAgICBpbnB1dHMucHVzaChhY3RpdmF0aW9uLmlkKTtcbiAgICBlbHNlIHtcbiAgICAgIGFjdGl2YXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0ucHVzaChzdG9yZV9hY3RpdmF0aW9uKTtcbiAgICAgIHRyYWNlX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdLnB1c2goc3RvcmVfdHJhY2UpO1xuICAgICAgcHJvcGFnYXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0ucHVzaChzdG9yZV9wcm9wYWdhdGlvbik7XG4gICAgICB2YXIgb2xkID0gZ2V0VmFyKHRoaXMsICdvbGQnKTtcbiAgICAgIHZhciBzdGF0ZSA9IGdldFZhcih0aGlzLCAnc3RhdGUnKTtcbiAgICAgIHZhciBiaWFzID0gZ2V0VmFyKHRoaXMsICdiaWFzJyk7XG4gICAgICBpZiAodGhpcy5zZWxmY29ubmVjdGlvbi5nYXRlcilcbiAgICAgICAgdmFyIHNlbGZfZ2FpbiA9IGdldFZhcih0aGlzLnNlbGZjb25uZWN0aW9uLCAnZ2FpbicpO1xuICAgICAgaWYgKHRoaXMuc2VsZmNvbm5lY3RlZCgpKVxuICAgICAgICB2YXIgc2VsZl93ZWlnaHQgPSBnZXRWYXIodGhpcy5zZWxmY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuICAgICAgYnVpbGRTZW50ZW5jZShvbGQsICcgPSAnLCBzdGF0ZSwgc3RvcmVfYWN0aXZhdGlvbik7XG4gICAgICBpZiAodGhpcy5zZWxmY29ubmVjdGVkKCkpXG4gICAgICAgIGlmICh0aGlzLnNlbGZjb25uZWN0aW9uLmdhdGVyKVxuICAgICAgICAgIGJ1aWxkU2VudGVuY2Uoc3RhdGUsICcgPSAnLCBzZWxmX2dhaW4sICcgKiAnLCBzZWxmX3dlaWdodCwgJyAqICcsXG4gICAgICAgICAgICBzdGF0ZSwgJyArICcsIGJpYXMsIHN0b3JlX2FjdGl2YXRpb24pO1xuICAgICAgICBlbHNlXG4gICAgICAgICAgYnVpbGRTZW50ZW5jZShzdGF0ZSwgJyA9ICcsIHNlbGZfd2VpZ2h0LCAnICogJywgc3RhdGUsICcgKyAnLFxuICAgICAgICAgICAgYmlhcywgc3RvcmVfYWN0aXZhdGlvbik7XG4gICAgICBlbHNlXG4gICAgICAgIGJ1aWxkU2VudGVuY2Uoc3RhdGUsICcgPSAnLCBiaWFzLCBzdG9yZV9hY3RpdmF0aW9uKTtcbiAgICAgIGZvciAodmFyIGkgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcbiAgICAgICAgdmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG4gICAgICAgIHZhciBpbnB1dF9hY3RpdmF0aW9uID0gZ2V0VmFyKGlucHV0LmZyb20sICdhY3RpdmF0aW9uJyk7XG4gICAgICAgIHZhciBpbnB1dF93ZWlnaHQgPSBnZXRWYXIoaW5wdXQsICd3ZWlnaHQnKTtcbiAgICAgICAgaWYgKGlucHV0LmdhdGVyKVxuICAgICAgICAgIHZhciBpbnB1dF9nYWluID0gZ2V0VmFyKGlucHV0LCAnZ2FpbicpO1xuICAgICAgICBpZiAodGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV0uZ2F0ZXIpXG4gICAgICAgICAgYnVpbGRTZW50ZW5jZShzdGF0ZSwgJyArPSAnLCBpbnB1dF9hY3RpdmF0aW9uLCAnICogJyxcbiAgICAgICAgICAgIGlucHV0X3dlaWdodCwgJyAqICcsIGlucHV0X2dhaW4sIHN0b3JlX2FjdGl2YXRpb24pO1xuICAgICAgICBlbHNlXG4gICAgICAgICAgYnVpbGRTZW50ZW5jZShzdGF0ZSwgJyArPSAnLCBpbnB1dF9hY3RpdmF0aW9uLCAnICogJyxcbiAgICAgICAgICAgIGlucHV0X3dlaWdodCwgc3RvcmVfYWN0aXZhdGlvbik7XG4gICAgICB9XG4gICAgICB2YXIgZGVyaXZhdGl2ZSA9IGdldFZhcih0aGlzLCAnZGVyaXZhdGl2ZScpO1xuICAgICAgc3dpdGNoICh0aGlzLnNxdWFzaCkge1xuICAgICAgICBjYXNlIE5ldXJvbi5zcXVhc2guTE9HSVNUSUM6XG4gICAgICAgICAgYnVpbGRTZW50ZW5jZShhY3RpdmF0aW9uLCAnID0gKDEgLyAoMSArIE1hdGguZXhwKC0nLCBzdGF0ZSwgJykpKScsXG4gICAgICAgICAgICBzdG9yZV9hY3RpdmF0aW9uKTtcbiAgICAgICAgICBidWlsZFNlbnRlbmNlKGRlcml2YXRpdmUsICcgPSAnLCBhY3RpdmF0aW9uLCAnICogKDEgLSAnLFxuICAgICAgICAgICAgYWN0aXZhdGlvbiwgJyknLCBzdG9yZV9hY3RpdmF0aW9uKTtcbiAgICAgICAgICBicmVhaztcbiAgICAgICAgY2FzZSBOZXVyb24uc3F1YXNoLlRBTkg6XG4gICAgICAgICAgdmFyIGVQID0gZ2V0VmFyKCdhdXgnKTtcbiAgICAgICAgICB2YXIgZU4gPSBnZXRWYXIoJ2F1eF8yJyk7XG4gICAgICAgICAgYnVpbGRTZW50ZW5jZShlUCwgJyA9IE1hdGguZXhwKCcsIHN0YXRlLCAnKScsIHN0b3JlX2FjdGl2YXRpb24pO1xuICAgICAgICAgIGJ1aWxkU2VudGVuY2UoZU4sICcgPSAxIC8gJywgZVAsIHN0b3JlX2FjdGl2YXRpb24pO1xuICAgICAgICAgIGJ1aWxkU2VudGVuY2UoYWN0aXZhdGlvbiwgJyA9ICgnLCBlUCwgJyAtICcsIGVOLCAnKSAvICgnLCBlUCwgJyArICcsIGVOLCAnKScsIHN0b3JlX2FjdGl2YXRpb24pO1xuICAgICAgICAgIGJ1aWxkU2VudGVuY2UoZGVyaXZhdGl2ZSwgJyA9IDEgLSAoJywgYWN0aXZhdGlvbiwgJyAqICcsIGFjdGl2YXRpb24sICcpJywgc3RvcmVfYWN0aXZhdGlvbik7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIGNhc2UgTmV1cm9uLnNxdWFzaC5JREVOVElUWTpcbiAgICAgICAgICBidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSAnLCBzdGF0ZSwgc3RvcmVfYWN0aXZhdGlvbik7XG4gICAgICAgICAgYnVpbGRTZW50ZW5jZShkZXJpdmF0aXZlLCAnID0gMScsIHN0b3JlX2FjdGl2YXRpb24pO1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICBjYXNlIE5ldXJvbi5zcXVhc2guSExJTTpcbiAgICAgICAgICBidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSArKCcsIHN0YXRlLCAnID4gMCknLFxuICAgICAgICAgICAgc3RvcmVfYWN0aXZhdGlvbik7XG4gICAgICAgICAgYnVpbGRTZW50ZW5jZShkZXJpdmF0aXZlLCAnID0gMScsIHN0b3JlX2FjdGl2YXRpb24pO1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgfVxuXG4gICAgICBpbmZsdWVuY2VzID0gW107XG4gICAgICBmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG4gICAgICAgIC8vIGNhbGN1bGF0ZSBleHRlbmRlZCBlbGVnaWJpbGl0eSB0cmFjZXMgaW4gYWR2YW5jZVxuICAgICAgICBcbiAgICAgICAgdmFyIHh0cmFjZSA9IHRoaXMudHJhY2UuZXh0ZW5kZWRbaWRdO1xuICAgICAgICB2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcbiAgICAgICAgdmFyIGluZmx1ZW5jZSA9IGdldFZhcignYXV4Jyk7XG4gICAgICAgIHZhciBuZXVyb25fb2xkID0gZ2V0VmFyKG5ldXJvbiwgJ29sZCcpO1xuICAgICAgICB2YXIgaW5pdGlhbGl6ZWQgPSBmYWxzZTtcbiAgICAgICAgaWYgKG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzKVxuICAgICAgICB7XG4gICAgICAgICAgYnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAnLCBuZXVyb25fb2xkLCBzdG9yZV90cmFjZSk7XG4gICAgICAgICAgaW5pdGlhbGl6ZWQgPSB0cnVlO1xuICAgICAgICB9XG4gICAgICAgIGZvciAodmFyIGluY29taW5nIGluIHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdKSB7XG4gICAgICAgICAgdmFyIGluY29taW5nX3dlaWdodCA9IGdldFZhcih0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXVxuICAgICAgICAgICAgW2luY29taW5nXSwgJ3dlaWdodCcpO1xuICAgICAgICAgIHZhciBpbmNvbWluZ19hY3RpdmF0aW9uID0gZ2V0VmFyKHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdXG4gICAgICAgICAgICBbaW5jb21pbmddLmZyb20sICdhY3RpdmF0aW9uJyk7XG5cbiAgICAgICAgICBpZiAoaW5pdGlhbGl6ZWQpXG4gICAgICAgICAgICBidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyArPSAnLCBpbmNvbWluZ193ZWlnaHQsICcgKiAnLFxuICAgICAgICAgICAgICBpbmNvbWluZ19hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG4gICAgICAgICAgZWxzZSB7XG4gICAgICAgICAgICBidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9ICcsIGluY29taW5nX3dlaWdodCwgJyAqICcsXG4gICAgICAgICAgICAgIGluY29taW5nX2FjdGl2YXRpb24sIHN0b3JlX3RyYWNlKTtcbiAgICAgICAgICAgIGluaXRpYWxpemVkID0gdHJ1ZTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgXG4gICAgICAgIGluZmx1ZW5jZXMucHVzaChuZXVyb24uSUQpO1xuICAgICAgICBidWlsZFNlbnRlbmNlKFwiaW5mbHVlbmNlc1tcIiArIChpbmZsdWVuY2VzLmxlbmd0aCAtIDEpICsgXCJdID0gXCIsIGluZmx1ZW5jZSwgc3RvcmVfdHJhY2UpO1xuICAgICAgfVxuICAgICAgXG4gICAgICBmb3IgKHZhciBpIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG4gICAgICAgIHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2ldO1xuICAgICAgICBpZiAoaW5wdXQuZ2F0ZXIpXG4gICAgICAgICAgdmFyIGlucHV0X2dhaW4gPSBnZXRWYXIoaW5wdXQsICdnYWluJyk7XG4gICAgICAgIHZhciBpbnB1dF9hY3RpdmF0aW9uID0gZ2V0VmFyKGlucHV0LmZyb20sICdhY3RpdmF0aW9uJyk7XG4gICAgICAgIHZhciB0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZWxlZ2liaWxpdHknLCBpbnB1dC5JRCwgdGhpcy50cmFjZVxuICAgICAgICAgIC5lbGVnaWJpbGl0eVtpbnB1dC5JRF0pO1xuICAgICAgICBpZiAodGhpcy5zZWxmY29ubmVjdGVkKCkpIHtcbiAgICAgICAgICBpZiAodGhpcy5zZWxmY29ubmVjdGlvbi5nYXRlcikge1xuICAgICAgICAgICAgaWYgKGlucHV0LmdhdGVyKVxuICAgICAgICAgICAgICBidWlsZFNlbnRlbmNlKHRyYWNlLCAnID0gJywgc2VsZl9nYWluLCAnICogJywgc2VsZl93ZWlnaHQsXG4gICAgICAgICAgICAgICAgJyAqICcsIHRyYWNlLCAnICsgJywgaW5wdXRfZ2FpbiwgJyAqICcsIGlucHV0X2FjdGl2YXRpb24sXG4gICAgICAgICAgICAgICAgc3RvcmVfdHJhY2UpO1xuICAgICAgICAgICAgZWxzZVxuICAgICAgICAgICAgICBidWlsZFNlbnRlbmNlKHRyYWNlLCAnID0gJywgc2VsZl9nYWluLCAnICogJywgc2VsZl93ZWlnaHQsXG4gICAgICAgICAgICAgICAgJyAqICcsIHRyYWNlLCAnICsgJywgaW5wdXRfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpZiAoaW5wdXQuZ2F0ZXIpXG4gICAgICAgICAgICAgIGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBzZWxmX3dlaWdodCwgJyAqICcsIHRyYWNlLCAnICsgJyxcbiAgICAgICAgICAgICAgICBpbnB1dF9nYWluLCAnICogJywgaW5wdXRfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuICAgICAgICAgICAgZWxzZVxuICAgICAgICAgICAgICBidWlsZFNlbnRlbmNlKHRyYWNlLCAnID0gJywgc2VsZl93ZWlnaHQsICcgKiAnLCB0cmFjZSwgJyArICcsXG4gICAgICAgICAgICAgICAgaW5wdXRfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBpZiAoaW5wdXQuZ2F0ZXIpXG4gICAgICAgICAgICBidWlsZFNlbnRlbmNlKHRyYWNlLCAnID0gJywgaW5wdXRfZ2FpbiwgJyAqICcsIGlucHV0X2FjdGl2YXRpb24sXG4gICAgICAgICAgICAgIHN0b3JlX3RyYWNlKTtcbiAgICAgICAgICBlbHNlXG4gICAgICAgICAgICBidWlsZFNlbnRlbmNlKHRyYWNlLCAnID0gJywgaW5wdXRfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuICAgICAgICB9XG4gICAgICAgIGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcbiAgICAgICAgICAvLyBleHRlbmRlZCBlbGVnaWJpbGl0eSB0cmFjZVxuICAgICAgICAgIHZhciB4dHJhY2UgPSB0aGlzLnRyYWNlLmV4dGVuZGVkW2lkXTtcbiAgICAgICAgICB2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcbiAgICAgICAgICB2YXIgaW5mbHVlbmNlID0gZ2V0VmFyKCdhdXgnKTtcbiAgICAgICAgICB2YXIgbmV1cm9uX29sZCA9IGdldFZhcihuZXVyb24sICdvbGQnKTtcblxuICAgICAgICAgIHZhciB0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZWxlZ2liaWxpdHknLCBpbnB1dC5JRCwgdGhpcy50cmFjZVxuICAgICAgICAgICAgLmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG4gICAgICAgICAgdmFyIHh0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZXh0ZW5kZWQnLCBuZXVyb24uSUQsIGlucHV0LklELFxuICAgICAgICAgICAgdGhpcy50cmFjZS5leHRlbmRlZFtuZXVyb24uSURdW2lucHV0LklEXSk7XG4gICAgICAgICAgaWYgKG5ldXJvbi5zZWxmY29ubmVjdGVkKCkpXG4gICAgICAgICAgICB2YXIgbmV1cm9uX3NlbGZfd2VpZ2h0ID0gZ2V0VmFyKG5ldXJvbi5zZWxmY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuICAgICAgICAgIGlmIChuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIpXG4gICAgICAgICAgICB2YXIgbmV1cm9uX3NlbGZfZ2FpbiA9IGdldFZhcihuZXVyb24uc2VsZmNvbm5lY3Rpb24sICdnYWluJyk7XG4gICAgICAgICAgaWYgKG5ldXJvbi5zZWxmY29ubmVjdGVkKCkpXG4gICAgICAgICAgICBpZiAobmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyKVxuICAgICAgICAgICAgICBidWlsZFNlbnRlbmNlKHh0cmFjZSwgJyA9ICcsIG5ldXJvbl9zZWxmX2dhaW4sICcgKiAnLFxuICAgICAgICAgICAgICAgIG5ldXJvbl9zZWxmX3dlaWdodCwgJyAqICcsIHh0cmFjZSwgJyArICcsIGRlcml2YXRpdmUsICcgKiAnLFxuICAgICAgICAgICAgICAgIHRyYWNlLCAnICogJywgXCJpbmZsdWVuY2VzW1wiICsgaW5mbHVlbmNlcy5pbmRleE9mKG5ldXJvbi5JRCkgKyBcIl1cIiwgc3RvcmVfdHJhY2UpO1xuICAgICAgICAgICAgZWxzZVxuICAgICAgICAgICAgICBidWlsZFNlbnRlbmNlKHh0cmFjZSwgJyA9ICcsIG5ldXJvbl9zZWxmX3dlaWdodCwgJyAqICcsXG4gICAgICAgICAgICAgICAgeHRyYWNlLCAnICsgJywgZGVyaXZhdGl2ZSwgJyAqICcsIHRyYWNlLCAnICogJyxcbiAgICAgICAgICAgICAgICBcImluZmx1ZW5jZXNbXCIgKyBpbmZsdWVuY2VzLmluZGV4T2YobmV1cm9uLklEKSArIFwiXVwiLCBzdG9yZV90cmFjZSk7XG4gICAgICAgICAgZWxzZVxuICAgICAgICAgICAgYnVpbGRTZW50ZW5jZSh4dHJhY2UsICcgPSAnLCBkZXJpdmF0aXZlLCAnICogJywgdHJhY2UsICcgKiAnLFxuICAgICAgICAgICAgICBcImluZmx1ZW5jZXNbXCIgKyBpbmZsdWVuY2VzLmluZGV4T2YobmV1cm9uLklEKSArIFwiXVwiLCBzdG9yZV90cmFjZSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGZvciAodmFyIGNvbm5lY3Rpb24gaW4gdGhpcy5jb25uZWN0aW9ucy5nYXRlZCkge1xuICAgICAgICB2YXIgZ2F0ZWRfZ2FpbiA9IGdldFZhcih0aGlzLmNvbm5lY3Rpb25zLmdhdGVkW2Nvbm5lY3Rpb25dLCAnZ2FpbicpO1xuICAgICAgICBidWlsZFNlbnRlbmNlKGdhdGVkX2dhaW4sICcgPSAnLCBhY3RpdmF0aW9uLCBzdG9yZV9hY3RpdmF0aW9uKTtcbiAgICAgIH1cbiAgICB9XG4gICAgaWYgKCFpc0lucHV0KSB7XG4gICAgICB2YXIgcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIodGhpcywgJ2Vycm9yJywgJ3Jlc3BvbnNpYmlsaXR5JywgdGhpcy5lcnJvclxuICAgICAgICAucmVzcG9uc2liaWxpdHkpO1xuICAgICAgaWYgKGlzT3V0cHV0KSB7XG4gICAgICAgIHZhciB0YXJnZXQgPSBnZXRWYXIoJ3RhcmdldCcpO1xuICAgICAgICBidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnID0gJywgdGFyZ2V0LCAnIC0gJywgYWN0aXZhdGlvbixcbiAgICAgICAgICBzdG9yZV9wcm9wYWdhdGlvbik7XG4gICAgICAgIGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG4gICAgICAgICAgdmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaWRdO1xuICAgICAgICAgIHZhciB0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZWxlZ2liaWxpdHknLCBpbnB1dC5JRCwgdGhpcy50cmFjZVxuICAgICAgICAgICAgLmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG4gICAgICAgICAgdmFyIGlucHV0X3dlaWdodCA9IGdldFZhcihpbnB1dCwgJ3dlaWdodCcpO1xuICAgICAgICAgIGJ1aWxkU2VudGVuY2UoaW5wdXRfd2VpZ2h0LCAnICs9ICcsIHJhdGUsICcgKiAoJywgcmVzcG9uc2liaWxpdHksXG4gICAgICAgICAgICAnICogJywgdHJhY2UsICcpJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuICAgICAgICB9XG4gICAgICAgIG91dHB1dHMucHVzaChhY3RpdmF0aW9uLmlkKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGlmICghbm9Qcm9qZWN0aW9ucyAmJiAhbm9HYXRlcykge1xuICAgICAgICAgIHZhciBlcnJvciA9IGdldFZhcignYXV4Jyk7XG4gICAgICAgICAgZm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpIHtcbiAgICAgICAgICAgIHZhciBjb25uZWN0aW9uID0gdGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbaWRdO1xuICAgICAgICAgICAgdmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24udG87XG4gICAgICAgICAgICB2YXIgY29ubmVjdGlvbl93ZWlnaHQgPSBnZXRWYXIoY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuICAgICAgICAgICAgdmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG4gICAgICAgICAgICAgICdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG4gICAgICAgICAgICBpZiAoY29ubmVjdGlvbi5nYXRlcikge1xuICAgICAgICAgICAgICB2YXIgY29ubmVjdGlvbl9nYWluID0gZ2V0VmFyKGNvbm5lY3Rpb24sICdnYWluJyk7XG4gICAgICAgICAgICAgIGJ1aWxkU2VudGVuY2UoZXJyb3IsICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LCAnICogJyxcbiAgICAgICAgICAgICAgICBjb25uZWN0aW9uX2dhaW4sICcgKiAnLCBjb25uZWN0aW9uX3dlaWdodCxcbiAgICAgICAgICAgICAgICBzdG9yZV9wcm9wYWdhdGlvbik7XG4gICAgICAgICAgICB9IGVsc2VcbiAgICAgICAgICAgICAgYnVpbGRTZW50ZW5jZShlcnJvciwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuICAgICAgICAgICAgICAgIGNvbm5lY3Rpb25fd2VpZ2h0LCBzdG9yZV9wcm9wYWdhdGlvbik7XG4gICAgICAgICAgfVxuICAgICAgICAgIHZhciBwcm9qZWN0ZWQgPSBnZXRWYXIodGhpcywgJ2Vycm9yJywgJ3Byb2plY3RlZCcsIHRoaXMuZXJyb3IucHJvamVjdGVkKTtcbiAgICAgICAgICBidWlsZFNlbnRlbmNlKHByb2plY3RlZCwgJyA9ICcsIGRlcml2YXRpdmUsICcgKiAnLCBlcnJvcixcbiAgICAgICAgICAgIHN0b3JlX3Byb3BhZ2F0aW9uKTtcbiAgICAgICAgICBidWlsZFNlbnRlbmNlKGVycm9yLCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcbiAgICAgICAgICBmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG4gICAgICAgICAgICB2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcbiAgICAgICAgICAgIHZhciBpbmZsdWVuY2UgPSBnZXRWYXIoJ2F1eF8yJyk7XG4gICAgICAgICAgICB2YXIgbmV1cm9uX29sZCA9IGdldFZhcihuZXVyb24sICdvbGQnKTtcbiAgICAgICAgICAgIGlmIChuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIgPT0gdGhpcylcbiAgICAgICAgICAgICAgYnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAnLCBuZXVyb25fb2xkLCBzdG9yZV9wcm9wYWdhdGlvbik7XG4gICAgICAgICAgICBlbHNlXG4gICAgICAgICAgICAgIGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcbiAgICAgICAgICAgIGZvciAodmFyIGlucHV0IGluIHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdKSB7XG4gICAgICAgICAgICAgIHZhciBjb25uZWN0aW9uID0gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF1baW5wdXRdO1xuICAgICAgICAgICAgICB2YXIgY29ubmVjdGlvbl93ZWlnaHQgPSBnZXRWYXIoY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuICAgICAgICAgICAgICB2YXIgbmV1cm9uX2FjdGl2YXRpb24gPSBnZXRWYXIoY29ubmVjdGlvbi5mcm9tLCAnYWN0aXZhdGlvbicpO1xuICAgICAgICAgICAgICBidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyArPSAnLCBjb25uZWN0aW9uX3dlaWdodCwgJyAqICcsXG4gICAgICAgICAgICAgICAgbmV1cm9uX2FjdGl2YXRpb24sIHN0b3JlX3Byb3BhZ2F0aW9uKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuICAgICAgICAgICAgICAncmVzcG9uc2liaWxpdHknLCBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkpO1xuICAgICAgICAgICAgYnVpbGRTZW50ZW5jZShlcnJvciwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuICAgICAgICAgICAgICBpbmZsdWVuY2UsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgdmFyIGdhdGVkID0gZ2V0VmFyKHRoaXMsICdlcnJvcicsICdnYXRlZCcsIHRoaXMuZXJyb3IuZ2F0ZWQpO1xuICAgICAgICAgIGJ1aWxkU2VudGVuY2UoZ2F0ZWQsICcgPSAnLCBkZXJpdmF0aXZlLCAnICogJywgZXJyb3IsXG4gICAgICAgICAgICBzdG9yZV9wcm9wYWdhdGlvbik7XG4gICAgICAgICAgYnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyA9ICcsIHByb2plY3RlZCwgJyArICcsIGdhdGVkLFxuICAgICAgICAgICAgc3RvcmVfcHJvcGFnYXRpb24pO1xuICAgICAgICAgIGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG4gICAgICAgICAgICB2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG4gICAgICAgICAgICB2YXIgZ3JhZGllbnQgPSBnZXRWYXIoJ2F1eCcpO1xuICAgICAgICAgICAgdmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzXG4gICAgICAgICAgICAgIC50cmFjZS5lbGVnaWJpbGl0eVtpbnB1dC5JRF0pO1xuICAgICAgICAgICAgYnVpbGRTZW50ZW5jZShncmFkaWVudCwgJyA9ICcsIHByb2plY3RlZCwgJyAqICcsIHRyYWNlLFxuICAgICAgICAgICAgICBzdG9yZV9wcm9wYWdhdGlvbik7XG4gICAgICAgICAgICBmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG4gICAgICAgICAgICAgIHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuICAgICAgICAgICAgICB2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcbiAgICAgICAgICAgICAgICAncmVzcG9uc2liaWxpdHknLCBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkpO1xuICAgICAgICAgICAgICB2YXIgeHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdleHRlbmRlZCcsIG5ldXJvbi5JRCxcbiAgICAgICAgICAgICAgICBpbnB1dC5JRCwgdGhpcy50cmFjZS5leHRlbmRlZFtuZXVyb24uSURdW2lucHV0LklEXSk7XG4gICAgICAgICAgICAgIGJ1aWxkU2VudGVuY2UoZ3JhZGllbnQsICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LCAnICogJyxcbiAgICAgICAgICAgICAgICB4dHJhY2UsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHZhciBpbnB1dF93ZWlnaHQgPSBnZXRWYXIoaW5wdXQsICd3ZWlnaHQnKTtcbiAgICAgICAgICAgIGJ1aWxkU2VudGVuY2UoaW5wdXRfd2VpZ2h0LCAnICs9ICcsIHJhdGUsICcgKiAnLCBncmFkaWVudCxcbiAgICAgICAgICAgICAgc3RvcmVfcHJvcGFnYXRpb24pO1xuICAgICAgICAgIH1cblxuICAgICAgICB9IGVsc2UgaWYgKG5vR2F0ZXMpIHtcbiAgICAgICAgICBidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcbiAgICAgICAgICBmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZCkge1xuICAgICAgICAgICAgdmFyIGNvbm5lY3Rpb24gPSB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZFtpZF07XG4gICAgICAgICAgICB2YXIgbmV1cm9uID0gY29ubmVjdGlvbi50bztcbiAgICAgICAgICAgIHZhciBjb25uZWN0aW9uX3dlaWdodCA9IGdldFZhcihjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG4gICAgICAgICAgICB2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcbiAgICAgICAgICAgICAgJ3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcbiAgICAgICAgICAgIGlmIChjb25uZWN0aW9uLmdhdGVyKSB7XG4gICAgICAgICAgICAgIHZhciBjb25uZWN0aW9uX2dhaW4gPSBnZXRWYXIoY29ubmVjdGlvbiwgJ2dhaW4nKTtcbiAgICAgICAgICAgICAgYnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksXG4gICAgICAgICAgICAgICAgJyAqICcsIGNvbm5lY3Rpb25fZ2FpbiwgJyAqICcsIGNvbm5lY3Rpb25fd2VpZ2h0LFxuICAgICAgICAgICAgICAgIHN0b3JlX3Byb3BhZ2F0aW9uKTtcbiAgICAgICAgICAgIH0gZWxzZVxuICAgICAgICAgICAgICBidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSxcbiAgICAgICAgICAgICAgICAnICogJywgY29ubmVjdGlvbl93ZWlnaHQsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgYnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyAqPSAnLCBkZXJpdmF0aXZlLFxuICAgICAgICAgICAgc3RvcmVfcHJvcGFnYXRpb24pO1xuICAgICAgICAgIGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG4gICAgICAgICAgICB2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG4gICAgICAgICAgICB2YXIgdHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2VsZWdpYmlsaXR5JywgaW5wdXQuSUQsIHRoaXNcbiAgICAgICAgICAgICAgLnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG4gICAgICAgICAgICB2YXIgaW5wdXRfd2VpZ2h0ID0gZ2V0VmFyKGlucHV0LCAnd2VpZ2h0Jyk7XG4gICAgICAgICAgICBidWlsZFNlbnRlbmNlKGlucHV0X3dlaWdodCwgJyArPSAnLCByYXRlLCAnICogKCcsXG4gICAgICAgICAgICAgIHJlc3BvbnNpYmlsaXR5LCAnICogJywgdHJhY2UsICcpJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIGlmIChub1Byb2plY3Rpb25zKSB7XG4gICAgICAgICAgYnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyA9IDAnLCBzdG9yZV9wcm9wYWdhdGlvbik7XG4gICAgICAgICAgZm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuICAgICAgICAgICAgdmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG4gICAgICAgICAgICB2YXIgaW5mbHVlbmNlID0gZ2V0VmFyKCdhdXgnKTtcbiAgICAgICAgICAgIHZhciBuZXVyb25fb2xkID0gZ2V0VmFyKG5ldXJvbiwgJ29sZCcpO1xuICAgICAgICAgICAgaWYgKG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzKVxuICAgICAgICAgICAgICBidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9ICcsIG5ldXJvbl9vbGQsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcbiAgICAgICAgICAgIGVsc2VcbiAgICAgICAgICAgICAgYnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuICAgICAgICAgICAgZm9yICh2YXIgaW5wdXQgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0pIHtcbiAgICAgICAgICAgICAgdmFyIGNvbm5lY3Rpb24gPSB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXVtpbnB1dF07XG4gICAgICAgICAgICAgIHZhciBjb25uZWN0aW9uX3dlaWdodCA9IGdldFZhcihjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG4gICAgICAgICAgICAgIHZhciBuZXVyb25fYWN0aXZhdGlvbiA9IGdldFZhcihjb25uZWN0aW9uLmZyb20sICdhY3RpdmF0aW9uJyk7XG4gICAgICAgICAgICAgIGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnICs9ICcsIGNvbm5lY3Rpb25fd2VpZ2h0LCAnICogJyxcbiAgICAgICAgICAgICAgICBuZXVyb25fYWN0aXZhdGlvbiwgc3RvcmVfcHJvcGFnYXRpb24pO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG4gICAgICAgICAgICAgICdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG4gICAgICAgICAgICBidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSxcbiAgICAgICAgICAgICAgJyAqICcsIGluZmx1ZW5jZSwgc3RvcmVfcHJvcGFnYXRpb24pO1xuICAgICAgICAgIH1cbiAgICAgICAgICBidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICo9ICcsIGRlcml2YXRpdmUsXG4gICAgICAgICAgICBzdG9yZV9wcm9wYWdhdGlvbik7XG4gICAgICAgICAgZm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcbiAgICAgICAgICAgIHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcbiAgICAgICAgICAgIHZhciBncmFkaWVudCA9IGdldFZhcignYXV4Jyk7XG4gICAgICAgICAgICBidWlsZFNlbnRlbmNlKGdyYWRpZW50LCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcbiAgICAgICAgICAgIGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcbiAgICAgICAgICAgICAgdmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG4gICAgICAgICAgICAgIHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuICAgICAgICAgICAgICAgICdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG4gICAgICAgICAgICAgIHZhciB4dHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2V4dGVuZGVkJywgbmV1cm9uLklELFxuICAgICAgICAgICAgICAgIGlucHV0LklELCB0aGlzLnRyYWNlLmV4dGVuZGVkW25ldXJvbi5JRF1baW5wdXQuSURdKTtcbiAgICAgICAgICAgICAgYnVpbGRTZW50ZW5jZShncmFkaWVudCwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuICAgICAgICAgICAgICAgIHh0cmFjZSwgc3RvcmVfcHJvcGFnYXRpb24pO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdmFyIGlucHV0X3dlaWdodCA9IGdldFZhcihpbnB1dCwgJ3dlaWdodCcpO1xuICAgICAgICAgICAgYnVpbGRTZW50ZW5jZShpbnB1dF93ZWlnaHQsICcgKz0gJywgcmF0ZSwgJyAqICcsIGdyYWRpZW50LFxuICAgICAgICAgICAgICBzdG9yZV9wcm9wYWdhdGlvbik7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgICBidWlsZFNlbnRlbmNlKGJpYXMsICcgKz0gJywgcmF0ZSwgJyAqICcsIHJlc3BvbnNpYmlsaXR5LFxuICAgICAgICBzdG9yZV9wcm9wYWdhdGlvbik7XG4gICAgfVxuICAgIHJldHVybiB7XG4gICAgICBtZW1vcnk6IHZhcklELFxuICAgICAgbmV1cm9uczogbmV1cm9ucyArIDEsXG4gICAgICBpbnB1dHM6IGlucHV0cyxcbiAgICAgIG91dHB1dHM6IG91dHB1dHMsXG4gICAgICB0YXJnZXRzOiB0YXJnZXRzLFxuICAgICAgdmFyaWFibGVzOiB2YXJpYWJsZXMsXG4gICAgICBhY3RpdmF0aW9uX3NlbnRlbmNlczogYWN0aXZhdGlvbl9zZW50ZW5jZXMsXG4gICAgICB0cmFjZV9zZW50ZW5jZXM6IHRyYWNlX3NlbnRlbmNlcyxcbiAgICAgIHByb3BhZ2F0aW9uX3NlbnRlbmNlczogcHJvcGFnYXRpb25fc2VudGVuY2VzLFxuICAgICAgbGF5ZXJzOiBsYXllcnNcbiAgICB9XG4gIH1cbn1cblxuXG4vLyByZXByZXNlbnRzIGEgY29ubmVjdGlvbiBiZXR3ZWVuIHR3byBuZXVyb25zXG5OZXVyb24uY29ubmVjdGlvbiA9IGZ1bmN0aW9uIENvbm5lY3Rpb24oZnJvbSwgdG8sIHdlaWdodCkge1xuXG4gIGlmICghZnJvbSB8fCAhdG8pXG4gICAgdGhyb3cgXCJDb25uZWN0aW9uIEVycm9yOiBJbnZhbGlkIG5ldXJvbnNcIjtcblxuICB0aGlzLklEID0gTmV1cm9uLmNvbm5lY3Rpb24udWlkKCk7XG4gIHRoaXMuZnJvbSA9IGZyb207XG4gIHRoaXMudG8gPSB0bztcbiAgdGhpcy53ZWlnaHQgPSB0eXBlb2Ygd2VpZ2h0ID09ICd1bmRlZmluZWQnID8gTWF0aC5yYW5kb20oKSAqIC4yIC0gLjEgOlxuICAgIHdlaWdodDtcbiAgdGhpcy5nYWluID0gMTtcbiAgdGhpcy5nYXRlciA9IG51bGw7XG59XG5cblxuLy8gc3F1YXNoaW5nIGZ1bmN0aW9uc1xuTmV1cm9uLnNxdWFzaCA9IHt9O1xuXG4vLyBlcS4gNSAmIDUnXG5OZXVyb24uc3F1YXNoLkxPR0lTVElDID0gZnVuY3Rpb24oeCwgZGVyaXZhdGUpIHtcbiAgaWYgKCFkZXJpdmF0ZSlcbiAgICByZXR1cm4gMSAvICgxICsgTWF0aC5leHAoLXgpKTtcbiAgdmFyIGZ4ID0gTmV1cm9uLnNxdWFzaC5MT0dJU1RJQyh4KTtcbiAgcmV0dXJuIGZ4ICogKDEgLSBmeCk7XG59O1xuTmV1cm9uLnNxdWFzaC5UQU5IID0gZnVuY3Rpb24oeCwgZGVyaXZhdGUpIHtcbiAgaWYgKGRlcml2YXRlKVxuICAgIHJldHVybiAxIC0gTWF0aC5wb3coTmV1cm9uLnNxdWFzaC5UQU5IKHgpLCAyKTtcbiAgdmFyIGVQID0gTWF0aC5leHAoeCk7XG4gIHZhciBlTiA9IDEgLyBlUDtcbiAgcmV0dXJuIChlUCAtIGVOKSAvIChlUCArIGVOKTtcbn07XG5OZXVyb24uc3F1YXNoLklERU5USVRZID0gZnVuY3Rpb24oeCwgZGVyaXZhdGUpIHtcbiAgcmV0dXJuIGRlcml2YXRlID8gMSA6IHg7XG59O1xuTmV1cm9uLnNxdWFzaC5ITElNID0gZnVuY3Rpb24oeCwgZGVyaXZhdGUpIHtcbiAgcmV0dXJuIGRlcml2YXRlID8gMSA6ICsoeCA+IDApO1xufTtcblxuLy8gdW5pcXVlIElEJ3NcbihmdW5jdGlvbigpIHtcbiAgdmFyIG5ldXJvbnMgPSAwO1xuICB2YXIgY29ubmVjdGlvbnMgPSAwO1xuICBOZXVyb24udWlkID0gZnVuY3Rpb24oKSB7XG4gICAgcmV0dXJuIG5ldXJvbnMrKztcbiAgfVxuICBOZXVyb24uY29ubmVjdGlvbi51aWQgPSBmdW5jdGlvbigpIHtcbiAgICByZXR1cm4gY29ubmVjdGlvbnMrKztcbiAgfVxuICBOZXVyb24ucXVhbnRpdHkgPSBmdW5jdGlvbigpIHtcbiAgICByZXR1cm4ge1xuICAgICAgbmV1cm9uczogbmV1cm9ucyxcbiAgICAgIGNvbm5lY3Rpb25zOiBjb25uZWN0aW9uc1xuICAgIH1cbiAgfVxufSkoKTtcblxuLy8gZXhwb3J0XG5pZiAobW9kdWxlKSBtb2R1bGUuZXhwb3J0cyA9IE5ldXJvbjtcblxuIiwiLyoqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBUUkFJTkVSXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuXG5mdW5jdGlvbiBUcmFpbmVyKG5ldHdvcmssIG9wdGlvbnMpIHtcbiAgb3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG4gIHRoaXMubmV0d29yayA9IG5ldHdvcms7XG4gIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZSB8fCAuMjtcbiAgdGhpcy5pdGVyYXRpb25zID0gb3B0aW9ucy5pdGVyYXRpb25zIHx8IDEwMDAwMDtcbiAgdGhpcy5lcnJvciA9IG9wdGlvbnMuZXJyb3IgfHwgLjAwNVxuICB0aGlzLmNvc3QgPSBvcHRpb25zLmNvc3QgfHwgVHJhaW5lci5jb3N0LkNST1NTX0VOVFJPUFk7XG59XG5cblRyYWluZXIucHJvdG90eXBlID0ge1xuXG4gIC8vIHRyYWlucyBhbnkgZ2l2ZW4gc2V0IHRvIGEgbmV0d29ya1xuICB0cmFpbjogZnVuY3Rpb24oc2V0LCBvcHRpb25zKSB7XG5cbiAgICB2YXIgZXJyb3IgPSAxO1xuICAgIHZhciBpdGVyYXRpb25zID0gYnVja2V0U2l6ZSA9IDA7XG4gICAgdmFyIGFib3J0X3RyYWluaW5nID0gZmFsc2U7XG4gICAgdmFyIGlucHV0LCBvdXRwdXQsIHRhcmdldCwgY3VycmVudFJhdGU7XG5cbiAgICB2YXIgc3RhcnQgPSBEYXRlLm5vdygpO1xuXG4gICAgaWYgKG9wdGlvbnMpIHtcbiAgICAgIGlmIChvcHRpb25zLnNodWZmbGUpIHtcbiAgICAgICAgLy8rIEpvbmFzIFJhb25pIFNvYXJlcyBTaWx2YVxuICAgICAgICAvL0AgaHR0cDovL2pzZnJvbWhlbGwuY29tL2FycmF5L3NodWZmbGUgW3YxLjBdXG4gICAgICAgIGZ1bmN0aW9uIHNodWZmbGUobykgeyAvL3YxLjBcbiAgICAgICAgICBmb3IgKHZhciBqLCB4LCBpID0gby5sZW5ndGg7IGk7IGogPSBNYXRoLmZsb29yKE1hdGgucmFuZG9tKCkgKiBpKSwgeCA9IG9bLS1pXSwgb1tpXSA9IG9bal0sIG9bal0gPSB4KTtcbiAgICAgICAgICByZXR1cm4gbztcbiAgICAgICAgfTtcbiAgICAgIH1cbiAgICAgIGlmIChvcHRpb25zLml0ZXJhdGlvbnMpXG4gICAgICAgIHRoaXMuaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucztcbiAgICAgIGlmIChvcHRpb25zLmVycm9yKVxuICAgICAgICB0aGlzLmVycm9yID0gb3B0aW9ucy5lcnJvcjtcbiAgICAgIGlmIChvcHRpb25zLnJhdGUpXG4gICAgICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZTtcbiAgICAgIGlmIChvcHRpb25zLmNvc3QpXG4gICAgICAgIHRoaXMuY29zdCA9IG9wdGlvbnMuY29zdDtcbiAgICAgIGlmIChvcHRpb25zLnNjaGVkdWxlKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5zY2hlZHVsZTtcbiAgICAgIGlmIChvcHRpb25zLmN1c3RvbUxvZyl7XG4gICAgICAgIC8vIGZvciBiYWNrd2FyZCBjb21wYXRpYmlsaXR5IHdpdGggY29kZSB0aGF0IHVzZWQgY3VzdG9tTG9nXG4gICAgICAgIGNvbnNvbGUubG9nKCdEZXByZWNhdGVkOiB1c2Ugc2NoZWR1bGUgaW5zdGVhZCBvZiBjdXN0b21Mb2cnKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5jdXN0b21Mb2c7XG4gICAgICB9XG4gICAgfVxuXG4gICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGU7XG4gICAgaWYoQXJyYXkuaXNBcnJheSh0aGlzLnJhdGUpKSB7XG4gICAgICBidWNrZXRTaXplID0gTWF0aC5mbG9vcih0aGlzLml0ZXJhdGlvbnMgLyB0aGlzLnJhdGUubGVuZ3RoKTtcbiAgICB9XG5cblxuICAgIHdoaWxlICghYWJvcnRfdHJhaW5pbmcgJiYgaXRlcmF0aW9ucyA8IHRoaXMuaXRlcmF0aW9ucyAmJiBlcnJvciA+IHRoaXMuZXJyb3IpIHtcbiAgICAgIGVycm9yID0gMDtcblxuICAgICAgaWYoYnVja2V0U2l6ZSA+IDApIHtcbiAgICAgICAgdmFyIGN1cnJlbnRCdWNrZXQgPSBNYXRoLmZsb29yKGl0ZXJhdGlvbnMgLyBidWNrZXRTaXplKTtcbiAgICAgICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGVbY3VycmVudEJ1Y2tldF07XG4gICAgICB9XG5cbiAgICAgIGZvciAodmFyIHRyYWluIGluIHNldCkge1xuICAgICAgICBpbnB1dCA9IHNldFt0cmFpbl0uaW5wdXQ7XG4gICAgICAgIHRhcmdldCA9IHNldFt0cmFpbl0ub3V0cHV0O1xuXG4gICAgICAgIG91dHB1dCA9IHRoaXMubmV0d29yay5hY3RpdmF0ZShpbnB1dCk7XG4gICAgICAgIHRoaXMubmV0d29yay5wcm9wYWdhdGUoY3VycmVudFJhdGUsIHRhcmdldCk7XG5cbiAgICAgICAgZXJyb3IgKz0gdGhpcy5jb3N0KHRhcmdldCwgb3V0cHV0KTtcbiAgICAgIH1cblxuICAgICAgLy8gY2hlY2sgZXJyb3JcbiAgICAgIGl0ZXJhdGlvbnMrKztcbiAgICAgIGVycm9yIC89IHNldC5sZW5ndGg7XG5cbiAgICAgIGlmIChvcHRpb25zKSB7XG4gICAgICAgIGlmICh0aGlzLnNjaGVkdWxlICYmIHRoaXMuc2NoZWR1bGUuZXZlcnkgJiYgaXRlcmF0aW9ucyAlXG4gICAgICAgICAgdGhpcy5zY2hlZHVsZS5ldmVyeSA9PSAwKVxuICAgICAgICAgIGFib3J0X3RyYWluaW5nID0gdGhpcy5zY2hlZHVsZS5kbyh7XG4gICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgICAgICAgcmF0ZTogY3VycmVudFJhdGVcbiAgICAgICAgICB9KTtcbiAgICAgICAgZWxzZSBpZiAob3B0aW9ucy5sb2cgJiYgaXRlcmF0aW9ucyAlIG9wdGlvbnMubG9nID09IDApIHtcbiAgICAgICAgICBjb25zb2xlLmxvZygnaXRlcmF0aW9ucycsIGl0ZXJhdGlvbnMsICdlcnJvcicsIGVycm9yLCAncmF0ZScsIGN1cnJlbnRSYXRlKTtcbiAgICAgICAgfTtcbiAgICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSlcbiAgICAgICAgICBzaHVmZmxlKHNldCk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdmFyIHJlc3VsdHMgPSB7XG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0XG4gICAgfVxuXG4gICAgcmV0dXJuIHJlc3VsdHM7XG4gIH0sXG5cbiAgLy8gdHJhaW5zIGFueSBnaXZlbiBzZXQgdG8gYSBuZXR3b3JrIHVzaW5nIGEgV2ViV29ya2VyXG4gIHdvcmtlclRyYWluOiBmdW5jdGlvbihzZXQsIGNhbGxiYWNrLCBvcHRpb25zKSB7XG5cbiAgICB2YXIgdGhhdCA9IHRoaXM7XG4gICAgdmFyIGVycm9yID0gMTtcbiAgICB2YXIgaXRlcmF0aW9ucyA9IGJ1Y2tldFNpemUgPSAwO1xuICAgIHZhciBpbnB1dCwgb3V0cHV0LCB0YXJnZXQsIGN1cnJlbnRSYXRlO1xuICAgIHZhciBsZW5ndGggPSBzZXQubGVuZ3RoO1xuXG4gICAgdmFyIHN0YXJ0ID0gRGF0ZS5ub3coKTtcblxuICAgIGlmIChvcHRpb25zKSB7XG4gICAgICBpZiAob3B0aW9ucy5zaHVmZmxlKSB7XG4gICAgICAgIC8vKyBKb25hcyBSYW9uaSBTb2FyZXMgU2lsdmFcbiAgICAgICAgLy9AIGh0dHA6Ly9qc2Zyb21oZWxsLmNvbS9hcnJheS9zaHVmZmxlIFt2MS4wXVxuICAgICAgICBmdW5jdGlvbiBzaHVmZmxlKG8pIHsgLy92MS4wXG4gICAgICAgICAgZm9yICh2YXIgaiwgeCwgaSA9IG8ubGVuZ3RoOyBpOyBqID0gTWF0aC5mbG9vcihNYXRoLnJhbmRvbSgpICpcbiAgICAgICAgICAgICAgaSksIHggPSBvWy0taV0sIG9baV0gPSBvW2pdLCBvW2pdID0geCk7XG4gICAgICAgICAgcmV0dXJuIG87XG4gICAgICAgIH07XG4gICAgICB9XG4gICAgICBpZiAob3B0aW9ucy5pdGVyYXRpb25zKVxuICAgICAgICB0aGlzLml0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnM7XG4gICAgICBpZiAob3B0aW9ucy5lcnJvcilcbiAgICAgICAgdGhpcy5lcnJvciA9IG9wdGlvbnMuZXJyb3I7XG4gICAgICBpZiAob3B0aW9ucy5yYXRlKVxuICAgICAgICB0aGlzLnJhdGUgPSBvcHRpb25zLnJhdGU7XG4gICAgICBpZiAob3B0aW9ucy5jb3N0KVxuICAgICAgICB0aGlzLmNvc3QgPSBvcHRpb25zLmNvc3Q7XG4gICAgICBpZiAob3B0aW9ucy5zY2hlZHVsZSlcbiAgICAgICAgdGhpcy5zY2hlZHVsZSA9IG9wdGlvbnMuc2NoZWR1bGU7XG4gICAgICBpZiAob3B0aW9ucy5jdXN0b21Mb2cpXG4gICAgICAgIC8vIGZvciBiYWNrd2FyZCBjb21wYXRpYmlsaXR5IHdpdGggY29kZSB0aGF0IHVzZWQgY3VzdG9tTG9nXG4gICAgICAgIGNvbnNvbGUubG9nKCdEZXByZWNhdGVkOiB1c2Ugc2NoZWR1bGUgaW5zdGVhZCBvZiBjdXN0b21Mb2cnKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5jdXN0b21Mb2c7XG4gICAgfVxuXG4gICAgLy8gZHluYW1pYyBsZWFybmluZyByYXRlXG4gICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGU7XG4gICAgaWYoQXJyYXkuaXNBcnJheSh0aGlzLnJhdGUpKSB7XG4gICAgICBidWNrZXRTaXplID0gTWF0aC5mbG9vcih0aGlzLml0ZXJhdGlvbnMgLyB0aGlzLnJhdGUubGVuZ3RoKTtcbiAgICB9XG5cbiAgICAvLyBjcmVhdGUgYSB3b3JrZXJcbiAgICB2YXIgd29ya2VyID0gdGhpcy5uZXR3b3JrLndvcmtlcigpO1xuXG4gICAgLy8gYWN0aXZhdGUgdGhlIG5ldHdvcmtcbiAgICBmdW5jdGlvbiBhY3RpdmF0ZVdvcmtlcihpbnB1dClcbiAgICB7XG4gICAgICAgIHdvcmtlci5wb3N0TWVzc2FnZSh7IFxuICAgICAgICAgICAgYWN0aW9uOiBcImFjdGl2YXRlXCIsXG4gICAgICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgICAgICBtZW1vcnlCdWZmZXI6IHRoYXQubmV0d29yay5vcHRpbWl6ZWQubWVtb3J5XG4gICAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyBiYWNrcHJvcGFnYXRlIHRoZSBuZXR3b3JrXG4gICAgZnVuY3Rpb24gcHJvcGFnYXRlV29ya2VyKHRhcmdldCl7XG4gICAgICAgIGlmKGJ1Y2tldFNpemUgPiAwKSB7XG4gICAgICAgICAgdmFyIGN1cnJlbnRCdWNrZXQgPSBNYXRoLmZsb29yKGl0ZXJhdGlvbnMgLyBidWNrZXRTaXplKTtcbiAgICAgICAgICBjdXJyZW50UmF0ZSA9IHRoaXMucmF0ZVtjdXJyZW50QnVja2V0XTtcbiAgICAgICAgfVxuICAgICAgICB3b3JrZXIucG9zdE1lc3NhZ2UoeyBcbiAgICAgICAgICAgIGFjdGlvbjogXCJwcm9wYWdhdGVcIixcbiAgICAgICAgICAgIHRhcmdldDogdGFyZ2V0LFxuICAgICAgICAgICAgcmF0ZTogY3VycmVudFJhdGUsXG4gICAgICAgICAgICBtZW1vcnlCdWZmZXI6IHRoYXQubmV0d29yay5vcHRpbWl6ZWQubWVtb3J5XG4gICAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyB0cmFpbiB0aGUgd29ya2VyXG4gICAgd29ya2VyLm9ubWVzc2FnZSA9IGZ1bmN0aW9uKGUpe1xuICAgICAgICAvLyBnaXZlIGNvbnRyb2wgb2YgdGhlIG1lbW9yeSBiYWNrIHRvIHRoZSBuZXR3b3JrXG4gICAgICAgIHRoYXQubmV0d29yay5vcHRpbWl6ZWQub3duZXJzaGlwKGUuZGF0YS5tZW1vcnlCdWZmZXIpO1xuXG4gICAgICAgIGlmIChlLmRhdGEuYWN0aW9uID09IFwicHJvcGFnYXRlXCIpXG4gICAgICAgIHtcbiAgICAgICAgICAgIGlmIChpbmRleCA+PSBsZW5ndGgpXG4gICAgICAgICAgICB7XG4gICAgICAgICAgICAgICAgaW5kZXggPSAwO1xuICAgICAgICAgICAgICAgIGl0ZXJhdGlvbnMrKztcbiAgICAgICAgICAgICAgICBlcnJvciAvPSBzZXQubGVuZ3RoO1xuXG4gICAgICAgICAgICAgICAgLy8gbG9nXG4gICAgICAgICAgICAgICAgaWYgKG9wdGlvbnMpIHtcbiAgICAgICAgICAgICAgICAgIGlmICh0aGlzLnNjaGVkdWxlICYmIHRoaXMuc2NoZWR1bGUuZXZlcnkgJiYgaXRlcmF0aW9ucyAlIHRoaXMuc2NoZWR1bGUuZXZlcnkgPT0gMClcbiAgICAgICAgICAgICAgICAgICAgYWJvcnRfdHJhaW5pbmcgPSB0aGlzLnNjaGVkdWxlLmRvKHtcbiAgICAgICAgICAgICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICAgICAgICAgICAgaXRlcmF0aW9uczogaXRlcmF0aW9uc1xuICAgICAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgICAgICAgICAgIGVsc2UgaWYgKG9wdGlvbnMubG9nICYmIGl0ZXJhdGlvbnMgJSBvcHRpb25zLmxvZyA9PSAwKSB7XG4gICAgICAgICAgICAgICAgICAgIGNvbnNvbGUubG9nKCdpdGVyYXRpb25zJywgaXRlcmF0aW9ucywgJ2Vycm9yJywgZXJyb3IpO1xuICAgICAgICAgICAgICAgICAgfTtcbiAgICAgICAgICAgICAgICAgIGlmIChvcHRpb25zLnNodWZmbGUpXG4gICAgICAgICAgICAgICAgICAgIHNodWZmbGUoc2V0KTtcbiAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICBpZiAoIWFib3J0X3RyYWluaW5nICYmIGl0ZXJhdGlvbnMgPCB0aGF0Lml0ZXJhdGlvbnMgJiYgZXJyb3IgPiB0aGF0LmVycm9yKVxuICAgICAgICAgICAgICAgIHtcbiAgICAgICAgICAgICAgICAgICAgYWN0aXZhdGVXb3JrZXIoc2V0W2luZGV4XS5pbnB1dCk7XG4gICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgLy8gY2FsbGJhY2tcbiAgICAgICAgICAgICAgICAgICAgY2FsbGJhY2soe1xuICAgICAgICAgICAgICAgICAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgICAgICAgICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgICAgICAgICAgICAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydFxuICAgICAgICAgICAgICAgICAgICB9KVxuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBlcnJvciA9IDA7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICAgICAgICAgICAgfVxuICAgICAgICB9XG5cbiAgICAgICAgaWYgKGUuZGF0YS5hY3Rpb24gPT0gXCJhY3RpdmF0ZVwiKVxuICAgICAgICB7XG4gICAgICAgICAgICBlcnJvciArPSB0aGF0LmNvc3Qoc2V0W2luZGV4XS5vdXRwdXQsIGUuZGF0YS5vdXRwdXQpO1xuICAgICAgICAgICAgcHJvcGFnYXRlV29ya2VyKHNldFtpbmRleF0ub3V0cHV0KTsgXG4gICAgICAgICAgICBpbmRleCsrO1xuICAgICAgICB9XG4gICAgfVxuXG4gICAgLy8ga2ljayBpdFxuICAgIHZhciBpbmRleCA9IDA7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSAwO1xuICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICB9LFxuXG4gIC8vIHRyYWlucyBhbiBYT1IgdG8gdGhlIG5ldHdvcmtcbiAgWE9SOiBmdW5jdGlvbihvcHRpb25zKSB7XG5cbiAgICBpZiAodGhpcy5uZXR3b3JrLmlucHV0cygpICE9IDIgfHwgdGhpcy5uZXR3b3JrLm91dHB1dHMoKSAhPSAxKVxuICAgICAgdGhyb3cgXCJFcnJvcjogSW5jb21wYXRpYmxlIG5ldHdvcmsgKDIgaW5wdXRzLCAxIG91dHB1dClcIjtcblxuICAgIHZhciBkZWZhdWx0cyA9IHtcbiAgICAgIGl0ZXJhdGlvbnM6IDEwMDAwMCxcbiAgICAgIGxvZzogZmFsc2UsXG4gICAgICBzaHVmZmxlOiB0cnVlLFxuICAgICAgY29zdDogVHJhaW5lci5jb3N0Lk1TRVxuICAgIH1cblxuICAgIGlmIChvcHRpb25zKVxuICAgICAgZm9yICh2YXIgaSBpbiBvcHRpb25zKVxuICAgICAgICBkZWZhdWx0c1tpXSA9IG9wdGlvbnNbaV07XG5cbiAgICByZXR1cm4gdGhpcy50cmFpbihbe1xuICAgICAgaW5wdXQ6IFswLCAwXSxcbiAgICAgIG91dHB1dDogWzBdXG4gICAgfSwge1xuICAgICAgaW5wdXQ6IFsxLCAwXSxcbiAgICAgIG91dHB1dDogWzFdXG4gICAgfSwge1xuICAgICAgaW5wdXQ6IFswLCAxXSxcbiAgICAgIG91dHB1dDogWzFdXG4gICAgfSwge1xuICAgICAgaW5wdXQ6IFsxLCAxXSxcbiAgICAgIG91dHB1dDogWzBdXG4gICAgfV0sIGRlZmF1bHRzKTtcbiAgfSxcblxuICAvLyB0cmFpbnMgdGhlIG5ldHdvcmsgdG8gcGFzcyBhIERpc3RyYWN0ZWQgU2VxdWVuY2UgUmVjYWxsIHRlc3RcbiAgRFNSOiBmdW5jdGlvbihvcHRpb25zKSB7XG4gICAgb3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG5cbiAgICB2YXIgdGFyZ2V0cyA9IG9wdGlvbnMudGFyZ2V0cyB8fCBbMiwgNCwgNywgOF07XG4gICAgdmFyIGRpc3RyYWN0b3JzID0gb3B0aW9ucy5kaXN0cmFjdG9ycyB8fCBbMywgNSwgNiwgOV07XG4gICAgdmFyIHByb21wdHMgPSBvcHRpb25zLnByb21wdHMgfHwgWzAsIDFdO1xuICAgIHZhciBsZW5ndGggPSBvcHRpb25zLmxlbmd0aCB8fCAyNDtcbiAgICB2YXIgY3JpdGVyaW9uID0gb3B0aW9ucy5zdWNjZXNzIHx8IDAuOTU7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnMgfHwgMTAwMDAwO1xuICAgIHZhciByYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4xO1xuICAgIHZhciBsb2cgPSBvcHRpb25zLmxvZyB8fCAwO1xuICAgIHZhciBzY2hlZHVsZSA9IG9wdGlvbnMuc2NoZWR1bGUgfHwge307XG5cbiAgICB2YXIgdHJpYWwgPSBjb3JyZWN0ID0gaSA9IGogPSBzdWNjZXNzID0gMCxcbiAgICAgIGVycm9yID0gMSxcbiAgICAgIHN5bWJvbHMgPSB0YXJnZXRzLmxlbmd0aCArIGRpc3RyYWN0b3JzLmxlbmd0aCArIHByb21wdHMubGVuZ3RoO1xuXG4gICAgdmFyIG5vUmVwZWF0ID0gZnVuY3Rpb24ocmFuZ2UsIGF2b2lkKSB7XG4gICAgICB2YXIgbnVtYmVyID0gTWF0aC5yYW5kb20oKSAqIHJhbmdlIHwgMDtcbiAgICAgIHZhciB1c2VkID0gZmFsc2U7XG4gICAgICBmb3IgKHZhciBpIGluIGF2b2lkKVxuICAgICAgICBpZiAobnVtYmVyID09IGF2b2lkW2ldKVxuICAgICAgICAgIHVzZWQgPSB0cnVlO1xuICAgICAgcmV0dXJuIHVzZWQgPyBub1JlcGVhdChyYW5nZSwgYXZvaWQpIDogbnVtYmVyO1xuICAgIH1cblxuICAgIHZhciBlcXVhbCA9IGZ1bmN0aW9uKHByZWRpY3Rpb24sIG91dHB1dCkge1xuICAgICAgZm9yICh2YXIgaSBpbiBwcmVkaWN0aW9uKVxuICAgICAgICBpZiAoTWF0aC5yb3VuZChwcmVkaWN0aW9uW2ldKSAhPSBvdXRwdXRbaV0pXG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuXG4gICAgdmFyIHN0YXJ0ID0gRGF0ZS5ub3coKTtcblxuICAgIHdoaWxlICh0cmlhbCA8IGl0ZXJhdGlvbnMgJiYgKHN1Y2Nlc3MgPCBjcml0ZXJpb24gfHwgdHJpYWwgJSAxMDAwICE9IDApKSB7XG4gICAgICAvLyBnZW5lcmF0ZSBzZXF1ZW5jZVxuICAgICAgdmFyIHNlcXVlbmNlID0gW10sXG4gICAgICAgIHNlcXVlbmNlTGVuZ3RoID0gbGVuZ3RoIC0gcHJvbXB0cy5sZW5ndGg7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgc2VxdWVuY2VMZW5ndGg7IGkrKykge1xuICAgICAgICB2YXIgYW55ID0gTWF0aC5yYW5kb20oKSAqIGRpc3RyYWN0b3JzLmxlbmd0aCB8IDA7XG4gICAgICAgIHNlcXVlbmNlLnB1c2goZGlzdHJhY3RvcnNbYW55XSk7XG4gICAgICB9XG4gICAgICB2YXIgaW5kZXhlcyA9IFtdLFxuICAgICAgICBwb3NpdGlvbnMgPSBbXTtcbiAgICAgIGZvciAoaSA9IDA7IGkgPCBwcm9tcHRzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIGluZGV4ZXMucHVzaChNYXRoLnJhbmRvbSgpICogdGFyZ2V0cy5sZW5ndGggfCAwKTtcbiAgICAgICAgcG9zaXRpb25zLnB1c2gobm9SZXBlYXQoc2VxdWVuY2VMZW5ndGgsIHBvc2l0aW9ucykpO1xuICAgICAgfVxuICAgICAgcG9zaXRpb25zID0gcG9zaXRpb25zLnNvcnQoKTtcbiAgICAgIGZvciAoaSA9IDA7IGkgPCBwcm9tcHRzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIHNlcXVlbmNlW3Bvc2l0aW9uc1tpXV0gPSB0YXJnZXRzW2luZGV4ZXNbaV1dO1xuICAgICAgICBzZXF1ZW5jZS5wdXNoKHByb21wdHNbaV0pO1xuICAgICAgfVxuXG4gICAgICAvL3RyYWluIHNlcXVlbmNlXG4gICAgICB2YXIgdGFyZ2V0c0NvcnJlY3QgPSBkaXN0cmFjdG9yc0NvcnJlY3QgPSAwO1xuICAgICAgZXJyb3IgPSAwO1xuICAgICAgZm9yIChpID0gMDsgaSA8IGxlbmd0aDsgaSsrKSB7XG4gICAgICAgIC8vIGdlbmVyYXRlIGlucHV0IGZyb20gc2VxdWVuY2VcbiAgICAgICAgdmFyIGlucHV0ID0gW107XG4gICAgICAgIGZvciAoaiA9IDA7IGogPCBzeW1ib2xzOyBqKyspXG4gICAgICAgICAgaW5wdXRbal0gPSAwO1xuICAgICAgICBpbnB1dFtzZXF1ZW5jZVtpXV0gPSAxO1xuXG4gICAgICAgIC8vIGdlbmVyYXRlIHRhcmdldCBvdXRwdXRcbiAgICAgICAgdmFyIG91dHB1dCA9IFtdO1xuICAgICAgICBmb3IgKGogPSAwOyBqIDwgdGFyZ2V0cy5sZW5ndGg7IGorKylcbiAgICAgICAgICBvdXRwdXRbal0gPSAwO1xuXG4gICAgICAgIGlmIChpID49IHNlcXVlbmNlTGVuZ3RoKSB7XG4gICAgICAgICAgdmFyIGluZGV4ID0gaSAtIHNlcXVlbmNlTGVuZ3RoO1xuICAgICAgICAgIG91dHB1dFtpbmRleGVzW2luZGV4XV0gPSAxO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gY2hlY2sgcmVzdWx0XG4gICAgICAgIHZhciBwcmVkaWN0aW9uID0gdGhpcy5uZXR3b3JrLmFjdGl2YXRlKGlucHV0KTtcblxuICAgICAgICBpZiAoZXF1YWwocHJlZGljdGlvbiwgb3V0cHV0KSlcbiAgICAgICAgICBpZiAoaSA8IHNlcXVlbmNlTGVuZ3RoKVxuICAgICAgICAgICAgZGlzdHJhY3RvcnNDb3JyZWN0Kys7XG4gICAgICAgICAgZWxzZVxuICAgICAgICAgICAgdGFyZ2V0c0NvcnJlY3QrKztcbiAgICAgICAgZWxzZSB7XG4gICAgICAgICAgdGhpcy5uZXR3b3JrLnByb3BhZ2F0ZShyYXRlLCBvdXRwdXQpO1xuICAgICAgICB9XG5cbiAgICAgICAgdmFyIGRlbHRhID0gMDtcbiAgICAgICAgZm9yICh2YXIgaiBpbiBwcmVkaWN0aW9uKVxuICAgICAgICAgIGRlbHRhICs9IE1hdGgucG93KG91dHB1dFtqXSAtIHByZWRpY3Rpb25bal0sIDIpO1xuICAgICAgICBlcnJvciArPSBkZWx0YSAvIHRoaXMubmV0d29yay5vdXRwdXRzKCk7XG5cbiAgICAgICAgaWYgKGRpc3RyYWN0b3JzQ29ycmVjdCArIHRhcmdldHNDb3JyZWN0ID09IGxlbmd0aClcbiAgICAgICAgICBjb3JyZWN0Kys7XG4gICAgICB9XG5cbiAgICAgIC8vIGNhbGN1bGF0ZSBlcnJvclxuICAgICAgaWYgKHRyaWFsICUgMTAwMCA9PSAwKVxuICAgICAgICBjb3JyZWN0ID0gMDtcbiAgICAgIHRyaWFsKys7XG4gICAgICB2YXIgZGl2aWRlRXJyb3IgPSB0cmlhbCAlIDEwMDA7XG4gICAgICBkaXZpZGVFcnJvciA9IGRpdmlkZUVycm9yID09IDAgPyAxMDAwIDogZGl2aWRlRXJyb3I7XG4gICAgICBzdWNjZXNzID0gY29ycmVjdCAvIGRpdmlkZUVycm9yO1xuICAgICAgZXJyb3IgLz0gbGVuZ3RoO1xuXG4gICAgICAvLyBsb2dcbiAgICAgIGlmIChsb2cgJiYgdHJpYWwgJSBsb2cgPT0gMClcbiAgICAgICAgY29uc29sZS5sb2coXCJpdGVyYXRpb25zOlwiLCB0cmlhbCwgXCIgc3VjY2VzczpcIiwgc3VjY2VzcywgXCIgY29ycmVjdDpcIixcbiAgICAgICAgICBjb3JyZWN0LCBcIiB0aW1lOlwiLCBEYXRlLm5vdygpIC0gc3RhcnQsIFwiIGVycm9yOlwiLCBlcnJvcik7XG4gICAgICBpZiAoc2NoZWR1bGUuZG8gJiYgc2NoZWR1bGUuZXZlcnkgJiYgdHJpYWwgJSBzY2hlZHVsZS5ldmVyeSA9PSAwKVxuICAgICAgICBzY2hlZHVsZS5kbyh7XG4gICAgICAgICAgaXRlcmF0aW9uczogdHJpYWwsXG4gICAgICAgICAgc3VjY2Vzczogc3VjY2VzcyxcbiAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0LFxuICAgICAgICAgIGNvcnJlY3Q6IGNvcnJlY3RcbiAgICAgICAgfSk7XG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgIGl0ZXJhdGlvbnM6IHRyaWFsLFxuICAgICAgc3VjY2Vzczogc3VjY2VzcyxcbiAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydFxuICAgIH1cbiAgfSxcblxuICAvLyB0cmFpbiB0aGUgbmV0d29yayB0byBsZWFybiBhbiBFbWJlZGVkIFJlYmVyIEdyYW1tYXJcbiAgRVJHOiBmdW5jdGlvbihvcHRpb25zKSB7XG5cbiAgICBvcHRpb25zID0gb3B0aW9ucyB8fCB7fTtcbiAgICB2YXIgaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucyB8fCAxNTAwMDA7XG4gICAgdmFyIGNyaXRlcmlvbiA9IG9wdGlvbnMuZXJyb3IgfHwgLjA1O1xuICAgIHZhciByYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4xO1xuICAgIHZhciBsb2cgPSBvcHRpb25zLmxvZyB8fCA1MDA7XG5cbiAgICAvLyBncmFtYXIgbm9kZVxuICAgIHZhciBOb2RlID0gZnVuY3Rpb24oKSB7XG4gICAgICB0aGlzLnBhdGhzID0gW107XG4gICAgfVxuICAgIE5vZGUucHJvdG90eXBlID0ge1xuICAgICAgY29ubmVjdDogZnVuY3Rpb24obm9kZSwgdmFsdWUpIHtcbiAgICAgICAgdGhpcy5wYXRocy5wdXNoKHtcbiAgICAgICAgICBub2RlOiBub2RlLFxuICAgICAgICAgIHZhbHVlOiB2YWx1ZVxuICAgICAgICB9KTtcbiAgICAgICAgcmV0dXJuIHRoaXM7XG4gICAgICB9LFxuICAgICAgYW55OiBmdW5jdGlvbigpIHtcbiAgICAgICAgaWYgKHRoaXMucGF0aHMubGVuZ3RoID09IDApXG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICB2YXIgaW5kZXggPSBNYXRoLnJhbmRvbSgpICogdGhpcy5wYXRocy5sZW5ndGggfCAwO1xuICAgICAgICByZXR1cm4gdGhpcy5wYXRoc1tpbmRleF07XG4gICAgICB9LFxuICAgICAgdGVzdDogZnVuY3Rpb24odmFsdWUpIHtcbiAgICAgICAgZm9yICh2YXIgaSBpbiB0aGlzLnBhdGhzKVxuICAgICAgICAgIGlmICh0aGlzLnBhdGhzW2ldLnZhbHVlID09IHZhbHVlKVxuICAgICAgICAgICAgcmV0dXJuIHRoaXMucGF0aHNbaV07XG4gICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB2YXIgcmViZXJHcmFtbWFyID0gZnVuY3Rpb24oKSB7XG5cbiAgICAgIC8vIGJ1aWxkIGEgcmViZXIgZ3JhbW1hclxuICAgICAgdmFyIG91dHB1dCA9IG5ldyBOb2RlKCk7XG4gICAgICB2YXIgbjEgPSAobmV3IE5vZGUoKSkuY29ubmVjdChvdXRwdXQsIFwiRVwiKTtcbiAgICAgIHZhciBuMiA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4xLCBcIlNcIik7XG4gICAgICB2YXIgbjMgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuMSwgXCJWXCIpLmNvbm5lY3QobjIsIFwiUFwiKTtcbiAgICAgIHZhciBuNCA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4yLCBcIlhcIilcbiAgICAgIG40LmNvbm5lY3QobjQsIFwiU1wiKTtcbiAgICAgIHZhciBuNSA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4zLCBcIlZcIilcbiAgICAgIG41LmNvbm5lY3QobjUsIFwiVFwiKTtcbiAgICAgIG4yLmNvbm5lY3QobjUsIFwiWFwiKVxuICAgICAgdmFyIG42ID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjQsIFwiVFwiKS5jb25uZWN0KG41LCBcIlBcIik7XG4gICAgICB2YXIgaW5wdXQgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuNiwgXCJCXCIpXG5cbiAgICAgIHJldHVybiB7XG4gICAgICAgIGlucHV0OiBpbnB1dCxcbiAgICAgICAgb3V0cHV0OiBvdXRwdXRcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBidWlsZCBhbiBlbWJlZGVkIHJlYmVyIGdyYW1tYXJcbiAgICB2YXIgZW1iZWRlZFJlYmVyR3JhbW1hciA9IGZ1bmN0aW9uKCkge1xuICAgICAgdmFyIHJlYmVyMSA9IHJlYmVyR3JhbW1hcigpO1xuICAgICAgdmFyIHJlYmVyMiA9IHJlYmVyR3JhbW1hcigpO1xuXG4gICAgICB2YXIgb3V0cHV0ID0gbmV3IE5vZGUoKTtcbiAgICAgIHZhciBuMSA9IChuZXcgTm9kZSkuY29ubmVjdChvdXRwdXQsIFwiRVwiKTtcbiAgICAgIHJlYmVyMS5vdXRwdXQuY29ubmVjdChuMSwgXCJUXCIpO1xuICAgICAgcmViZXIyLm91dHB1dC5jb25uZWN0KG4xLCBcIlBcIik7XG4gICAgICB2YXIgbjIgPSAobmV3IE5vZGUpLmNvbm5lY3QocmViZXIxLmlucHV0LCBcIlBcIikuY29ubmVjdChyZWJlcjIuaW5wdXQsXG4gICAgICAgIFwiVFwiKTtcbiAgICAgIHZhciBpbnB1dCA9IChuZXcgTm9kZSkuY29ubmVjdChuMiwgXCJCXCIpO1xuXG4gICAgICByZXR1cm4ge1xuICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgIG91dHB1dDogb3V0cHV0XG4gICAgICB9XG5cbiAgICB9XG5cbiAgICAvLyBnZW5lcmF0ZSBhbiBFUkcgc2VxdWVuY2VcbiAgICB2YXIgZ2VuZXJhdGUgPSBmdW5jdGlvbigpIHtcbiAgICAgIHZhciBub2RlID0gZW1iZWRlZFJlYmVyR3JhbW1hcigpLmlucHV0O1xuICAgICAgdmFyIG5leHQgPSBub2RlLmFueSgpO1xuICAgICAgdmFyIHN0ciA9IFwiXCI7XG4gICAgICB3aGlsZSAobmV4dCkge1xuICAgICAgICBzdHIgKz0gbmV4dC52YWx1ZTtcbiAgICAgICAgbmV4dCA9IG5leHQubm9kZS5hbnkoKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBzdHI7XG4gICAgfVxuXG4gICAgLy8gdGVzdCBpZiBhIHN0cmluZyBtYXRjaGVzIGFuIGVtYmVkZWQgcmViZXIgZ3JhbW1hclxuICAgIHZhciB0ZXN0ID0gZnVuY3Rpb24oc3RyKSB7XG4gICAgICB2YXIgbm9kZSA9IGVtYmVkZWRSZWJlckdyYW1tYXIoKS5pbnB1dDtcbiAgICAgIHZhciBpID0gMDtcbiAgICAgIHZhciBjaCA9IHN0ci5jaGFyQXQoaSk7XG4gICAgICB3aGlsZSAoaSA8IHN0ci5sZW5ndGgpIHtcbiAgICAgICAgdmFyIG5leHQgPSBub2RlLnRlc3QoY2gpO1xuICAgICAgICBpZiAoIW5leHQpXG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICBub2RlID0gbmV4dC5ub2RlO1xuICAgICAgICBjaCA9IHN0ci5jaGFyQXQoKytpKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIC8vIGhlbHBlciB0byBjaGVjayBpZiB0aGUgb3V0cHV0IGFuZCB0aGUgdGFyZ2V0IHZlY3RvcnMgbWF0Y2hcbiAgICB2YXIgZGlmZmVyZW50ID0gZnVuY3Rpb24oYXJyYXkxLCBhcnJheTIpIHtcbiAgICAgIHZhciBtYXgxID0gMDtcbiAgICAgIHZhciBpMSA9IC0xO1xuICAgICAgdmFyIG1heDIgPSAwO1xuICAgICAgdmFyIGkyID0gLTE7XG4gICAgICBmb3IgKHZhciBpIGluIGFycmF5MSkge1xuICAgICAgICBpZiAoYXJyYXkxW2ldID4gbWF4MSkge1xuICAgICAgICAgIG1heDEgPSBhcnJheTFbaV07XG4gICAgICAgICAgaTEgPSBpO1xuICAgICAgICB9XG4gICAgICAgIGlmIChhcnJheTJbaV0gPiBtYXgyKSB7XG4gICAgICAgICAgbWF4MiA9IGFycmF5MltpXTtcbiAgICAgICAgICBpMiA9IGk7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgcmV0dXJuIGkxICE9IGkyO1xuICAgIH1cblxuICAgIHZhciBpdGVyYXRpb24gPSAwO1xuICAgIHZhciBlcnJvciA9IDE7XG4gICAgdmFyIHRhYmxlID0ge1xuICAgICAgXCJCXCI6IDAsXG4gICAgICBcIlBcIjogMSxcbiAgICAgIFwiVFwiOiAyLFxuICAgICAgXCJYXCI6IDMsXG4gICAgICBcIlNcIjogNCxcbiAgICAgIFwiRVwiOiA1XG4gICAgfVxuXG4gICAgdmFyIHN0YXJ0ID0gRGF0ZS5ub3coKTtcbiAgICB3aGlsZSAoaXRlcmF0aW9uIDwgaXRlcmF0aW9ucyAmJiBlcnJvciA+IGNyaXRlcmlvbikge1xuICAgICAgdmFyIGkgPSAwO1xuICAgICAgZXJyb3IgPSAwO1xuXG4gICAgICAvLyBFUkcgc2VxdWVuY2UgdG8gbGVhcm5cbiAgICAgIHZhciBzZXF1ZW5jZSA9IGdlbmVyYXRlKCk7XG5cbiAgICAgIC8vIGlucHV0XG4gICAgICB2YXIgcmVhZCA9IHNlcXVlbmNlLmNoYXJBdChpKTtcbiAgICAgIC8vIHRhcmdldFxuICAgICAgdmFyIHByZWRpY3QgPSBzZXF1ZW5jZS5jaGFyQXQoaSArIDEpO1xuXG4gICAgICAvLyB0cmFpblxuICAgICAgd2hpbGUgKGkgPCBzZXF1ZW5jZS5sZW5ndGggLSAxKSB7XG4gICAgICAgIHZhciBpbnB1dCA9IFtdO1xuICAgICAgICB2YXIgdGFyZ2V0ID0gW107XG4gICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgNjsgaisrKSB7XG4gICAgICAgICAgaW5wdXRbal0gPSAwO1xuICAgICAgICAgIHRhcmdldFtqXSA9IDA7XG4gICAgICAgIH1cbiAgICAgICAgaW5wdXRbdGFibGVbcmVhZF1dID0gMTtcbiAgICAgICAgdGFyZ2V0W3RhYmxlW3ByZWRpY3RdXSA9IDE7XG5cbiAgICAgICAgdmFyIG91dHB1dCA9IHRoaXMubmV0d29yay5hY3RpdmF0ZShpbnB1dCk7XG5cbiAgICAgICAgaWYgKGRpZmZlcmVudChvdXRwdXQsIHRhcmdldCkpXG4gICAgICAgICAgdGhpcy5uZXR3b3JrLnByb3BhZ2F0ZShyYXRlLCB0YXJnZXQpO1xuXG4gICAgICAgIHJlYWQgPSBzZXF1ZW5jZS5jaGFyQXQoKytpKTtcbiAgICAgICAgcHJlZGljdCA9IHNlcXVlbmNlLmNoYXJBdChpICsgMSk7XG5cbiAgICAgICAgdmFyIGRlbHRhID0gMDtcbiAgICAgICAgZm9yICh2YXIgayBpbiBvdXRwdXQpXG4gICAgICAgICAgZGVsdGEgKz0gTWF0aC5wb3codGFyZ2V0W2tdIC0gb3V0cHV0W2tdLCAyKVxuICAgICAgICBkZWx0YSAvPSBvdXRwdXQubGVuZ3RoO1xuXG4gICAgICAgIGVycm9yICs9IGRlbHRhO1xuICAgICAgfVxuICAgICAgZXJyb3IgLz0gc2VxdWVuY2UubGVuZ3RoO1xuICAgICAgaXRlcmF0aW9uKys7XG4gICAgICBpZiAoaXRlcmF0aW9uICUgbG9nID09IDApIHtcbiAgICAgICAgY29uc29sZS5sb2coXCJpdGVyYXRpb25zOlwiLCBpdGVyYXRpb24sIFwiIHRpbWU6XCIsIERhdGUubm93KCkgLSBzdGFydCxcbiAgICAgICAgICBcIiBlcnJvcjpcIiwgZXJyb3IpO1xuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb24sXG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICB0aW1lOiBEYXRlLm5vdygpIC0gc3RhcnQsXG4gICAgICB0ZXN0OiB0ZXN0LFxuICAgICAgZ2VuZXJhdGU6IGdlbmVyYXRlXG4gICAgfVxuICB9XG59O1xuXG4vLyBCdWlsdC1pbiBjb3N0IGZ1bmN0aW9uc1xuVHJhaW5lci5jb3N0ID0ge1xuICAvLyBFcS4gOVxuICBDUk9TU19FTlRST1BZOiBmdW5jdGlvbih0YXJnZXQsIG91dHB1dClcbiAge1xuICAgIHZhciBjcm9zc2VudHJvcHkgPSAwO1xuICAgIGZvciAodmFyIGkgaW4gb3V0cHV0KVxuICAgICAgY3Jvc3NlbnRyb3B5IC09ICh0YXJnZXRbaV0gKiBNYXRoLmxvZyhvdXRwdXRbaV0rMWUtMTUpKSArICgoMS10YXJnZXRbaV0pICogTWF0aC5sb2coKDErMWUtMTUpLW91dHB1dFtpXSkpOyAvLyArMWUtMTUgaXMgYSB0aW55IHB1c2ggYXdheSB0byBhdm9pZCBNYXRoLmxvZygwKVxuICAgIHJldHVybiBjcm9zc2VudHJvcHk7XG4gIH0sXG4gIE1TRTogZnVuY3Rpb24odGFyZ2V0LCBvdXRwdXQpXG4gIHtcbiAgICB2YXIgbXNlID0gMDtcbiAgICBmb3IgKHZhciBpIGluIG91dHB1dClcbiAgICAgIG1zZSArPSBNYXRoLnBvdyh0YXJnZXRbaV0gLSBvdXRwdXRbaV0sIDIpO1xuICAgIHJldHVybiBtc2UgLyBvdXRwdXQubGVuZ3RoO1xuICB9XG59XG5cbi8vIGV4cG9ydFxuaWYgKG1vZHVsZSkgbW9kdWxlLmV4cG9ydHMgPSBUcmFpbmVyO1xuXG4iXX0=
var Neuron = synaptic.Neuron, Layer = synaptic.Layer, Network = synaptic.Network, Trainer = synaptic.Trainer, Architect = synaptic.Architect;