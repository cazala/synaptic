/*!
 * The MIT License (MIT)
 * 
 * Copyright (c) 2016 Juan Cazala - juancazala.com
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE
 * 
 * 
 * 
 * ********************************************************************************************
 *                                   SYNAPTIC (v1.0.8)
 * ********************************************************************************************
 * 
 * Synaptic is a javascript neural network library for node.js and the browser, its generalized
 * algorithm is architecture-free, so you can build and train basically any type of first order
 * or even second order neural network architectures.
 * 
 * http://en.wikipedia.org/wiki/Recurrent_neural_network#Second_Order_Recurrent_Neural_Network
 * 
 * The library includes a few built-in architectures like multilayer perceptrons, multilayer
 * long-short term memory networks (LSTM) or liquid state machines, and a trainer capable of
 * training any given network, and includes built-in training tasks/tests like solving an XOR,
 * passing a Distracted Sequence Recall test or an Embeded Reber Grammar test.
 * 
 * The algorithm implemented by this library has been taken from Derek D. Monner's paper:
 * 
 * 
 * A generalized LSTM-like training algorithm for second-order recurrent neural networks
 * http://www.overcomplete.net/papers/nn2012.pdf
 * 
 * There are references to the equations in that paper commented through the source code.
 * 
 */
(function webpackUniversalModuleDefinition(root, factory) {
	if(typeof exports === 'object' && typeof module === 'object')
		module.exports = factory();
	else if(typeof define === 'function' && define.amd)
		define([], factory);
	else {
		var a = factory();
		for(var i in a) (typeof exports === 'object' ? exports : root)[i] = a[i];
	}
})(this, function() {
return /******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};

/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {

/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId])
/******/ 			return installedModules[moduleId].exports;

/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			exports: {},
/******/ 			id: moduleId,
/******/ 			loaded: false
/******/ 		};

/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);

/******/ 		// Flag the module as loaded
/******/ 		module.loaded = true;

/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}


/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;

/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;

/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";

/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(0);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ function(module, exports, __webpack_require__) {

	module.exports = __webpack_require__(1);


/***/ },
/* 1 */
/***/ function(module, exports, __webpack_require__) {

	var Synaptic = {};

	Synaptic.Neuron = __webpack_require__(2);
	Synaptic.Layer = __webpack_require__(3);
	Synaptic.Network = __webpack_require__(4);
	Synaptic.Trainer = __webpack_require__(11);
	Synaptic.Architect = __webpack_require__(12);

	module.exports = Synaptic;

	// Browser
	if (typeof window == 'object') {
	  //noinspection CommaExpressionJS
	  Synaptic.ninja = ((oldSynaptic = window['synaptic']) =>
	      () => (window['synaptic'] = oldSynaptic, Synaptic))();

	  window['synaptic'] = Synaptic;
	}


/***/ },
/* 2 */
/***/ function(module, exports) {

	/******************************************************************************************
	 NEURON
	 *******************************************************************************************/

	class Neuron {
	  constructor() {
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

	  // activate the neuron
	  activate(input) {
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
	    const influences = [];
	    for (var id in this.trace.extended) {
	      // extended elegibility trace
	      var neuron = this.neighboors[id];

	      // if gated neuron's selfconnection is gated by this unit, the influence keeps track of the neuron's old state
	      var influence = neuron.selfconnection.gater == this ? neuron.old : 0;

	      // index runs over all the incoming connections to the gated neuron that are gated by this unit
	      for (let incoming in this.trace.influences[neuron.ID]) { // captures the effect that has an input connection to this unit, on a neuron that is gated by this unit
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
	        const xtrace = this.trace.extended[id];
	        var neuron = this.neighboors[id];
	        var influence = influences[neuron.ID];

	        // eq. 18
	        xtrace[input.ID] = neuron.selfconnection.gain * neuron.selfconnection
	            .weight * xtrace[input.ID] + this.derivative * this.trace.elegibility[
	            input.ID] * influence;
	      }
	    }

	    //  update gated connection's gains
	    for (let connection in this.connections.gated) {
	      this.connections.gated[connection].gain = this.activation;
	    }

	    return this.activation;
	  }

	  // back-propagate the error
	  propagate(rate, target) {
	    // error accumulator
	    let error = 0;

	    // whether or not this neuron is in the output layer
	    const isOutput = typeof target != 'undefined';

	    // output neurons get their error from the enviroment
	    if (isOutput)
	      this.error.responsibility = this.error.projected = target - this.activation; // Eq. 10

	    else // the rest of the neuron compute their error responsibilities by backpropagation
	    {
	      // error responsibilities from all the connections projected from this neuron
	      for (var id in this.connections.projected) {
	        const connection = this.connections.projected[id];
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
	        let influence = neuron.selfconnection.gater == this ? neuron.old : 0; // if gated neuron's selfconnection is gated by this neuron

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
	      let gradient = this.error.projected * this.trace.elegibility[input.ID];
	      for (var id in this.trace.extended) {
	        var neuron = this.neighboors[id];
	        gradient += neuron.error.responsibility * this.trace.extended[
	            neuron.ID][input.ID];
	      }
	      input.weight += rate * gradient; // adjust weights - aka learn
	    }

	    // adjust bias
	    this.bias += rate * this.error.responsibility;
	  }

	  project(neuron, weight) {
	    // self-connection
	    if (neuron == this) {
	      this.selfconnection.weight = 1;
	      return this.selfconnection;
	    }

	    // check if connection already exists
	    const connected = this.connected(neuron);
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

	    for (let id in neuron.trace.extended) {
	      const trace = neuron.trace.extended[id];
	      trace[connection.ID] = 0;
	    }

	    return connection;
	  }

	  gate(connection) {
	    // add connection to gated list
	    this.connections.gated[connection.ID] = connection;

	    const neuron = connection.to;
	    if (!(neuron.ID in this.trace.extended)) {
	      // extended trace
	      this.neighboors[neuron.ID] = neuron;
	      const xtrace = this.trace.extended[neuron.ID] = {};
	      for (let id in this.connections.inputs) {
	        const input = this.connections.inputs[id];
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
	  }

	  // returns true or false whether the neuron is self-connected or not
	  selfconnected() {
	    return this.selfconnection.weight !== 0;
	  }

	  // returns true or false whether the neuron is connected to another neuron (parameter)
	  connected(neuron) {
	    const result = {
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

	    for (let type in this.connections) {
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
	  }

	  // clears all the traces (the neuron forgets it's context, but the connections remain intact)
	  clear() {

	    for (var trace in this.trace.elegibility)
	      this.trace.elegibility[trace] = 0;

	    for (var trace in this.trace.extended)
	      for (let extended in this.trace.extended[trace])
	        this.trace.extended[trace][extended] = 0;

	    this.error.responsibility = this.error.projected = this.error.gated = 0;
	  }

	  // all the connections are randomized and the traces are cleared
	  reset() {
	    this.clear();

	    for (let type in this.connections)
	      for (let connection in this.connections[type])
	        this.connections[type][connection].weight = Math.random() * .2 - .1;
	    this.bias = Math.random() * .2 - .1;

	    this.old = this.state = this.activation = 0;
	  }

	  // hardcodes the behaviour of the neuron into an optimized function
	  optimize(optimized = {}, layer) {
	    const store_activation = [];
	    const store_trace = [];
	    const store_propagation = [];
	    let varID = optimized.memory || 0;
	    const neurons = optimized.neurons || 1;
	    const inputs = optimized.inputs || [];
	    const targets = optimized.targets || [];
	    const outputs = optimized.outputs || [];
	    const variables = optimized.variables || {};
	    const activation_sentences = optimized.activation_sentences || [];
	    const trace_sentences = optimized.trace_sentences || [];
	    const propagation_sentences = optimized.propagation_sentences || [];
	    const layers = optimized.layers || {__count: 0, __neuron: 0};

	    // allocate sentences
	    const allocate = store => {
	      const allocated = layer in layers && store[layers.__count];
	      if (!allocated) {
	        layers.__count = store.push([]) - 1;
	        layers[layer] = layers.__count;
	      }
	    };
	    allocate(activation_sentences);
	    allocate(trace_sentences);
	    allocate(propagation_sentences);
	    const currentLayer = layers.__count;

	    // get/reserve space in memory by creating a unique ID for a variablel
	    const getVar = function () {
	      const args = Array.prototype.slice.call(arguments);

	      if (args.length == 1) {
	        if (args[0] == 'target') {
	          var id = `target_${targets.length}`;
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
	        const extended = args.length > 2;
	        if (extended)
	          var value = args.pop();

	        const unit = args.shift();
	        const prop = args.pop();

	        if (!extended)
	          var value = unit[prop];

	        var id = `${prop}_`;
	        for (let property in args)
	          id += `${args[property]}_`;
	        id += unit.ID;
	        if (id in variables)
	          return variables[id];

	        return variables[id] = {
	          value,
	          id: varID++
	        };
	      }
	    };

	    // build sentence
	    const buildSentence = function () {
	      const args = Array.prototype.slice.call(arguments);
	      const store = args.pop();
	      let sentence = "";
	      for (let i in args)
	        if (typeof args[i] == 'string')
	          sentence += args[i];
	        else
	          sentence += `F[${args[i].id}]`;

	      store.push(`${sentence};`);
	    };

	    // helper to check if an object is empty
	    const isEmpty = obj => {
	      for (let prop in obj) {
	        if (obj.hasOwnProperty(prop))
	          return false;
	      }
	      return true;
	    };

	    // characteristics of the neuron
	    const noProjections = isEmpty(this.connections.projected);
	    const noGates = isEmpty(this.connections.gated);
	    const isInput = layer == 'input' ? true : isEmpty(this.connections.inputs);
	    const isOutput = layer == 'output' ? true : noProjections && noGates;

	    // optimize neuron's behaviour
	    const rate = getVar('rate');
	    const activation = getVar(this, 'activation');
	    if (isInput)
	      inputs.push(activation.id);
	    else {
	      activation_sentences[currentLayer].push(store_activation);
	      trace_sentences[currentLayer].push(store_trace);
	      propagation_sentences[currentLayer].push(store_propagation);
	      const old = getVar(this, 'old');
	      const state = getVar(this, 'state');
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
	          const eP = getVar('aux');
	          const eN = getVar('aux_2');
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
	          buildSentence(activation, ' = +(', state, ' > 0)', store_activation);
	          buildSentence(derivative, ' = 1', store_activation);
	        case Neuron.squash.RELU:
	          buildSentence(activation, ' = ', state, ' > 0 ? ', state, ' : 0', store_activation);
	          buildSentence(derivative, ' = ', state, ' > 0 ? 1 : 0', store_activation);
	          break;
	      }

	      for (var id in this.trace.extended) {
	        // calculate extended elegibility traces in advance

	        var neuron = this.neighboors[id];
	        var influence = getVar(`influences[${neuron.ID}]`);
	        var neuron_old = getVar(neuron, 'old');
	        let initialized = false;
	        if (neuron.selfconnection.gater == this) {
	          buildSentence(influence, ' = ', neuron_old, store_trace);
	          initialized = true;
	        }
	        for (let incoming in this.trace.influences[neuron.ID]) {
	          const incoming_weight = getVar(this.trace.influences[neuron.ID]
	            [incoming], 'weight');
	          const incoming_activation = getVar(this.trace.influences[neuron.ID]
	            [incoming].from, 'activation');

	          if (initialized)
	            buildSentence(influence, ' += ', incoming_weight, ' * ', incoming_activation, store_trace);
	          else {
	            buildSentence(influence, ' = ', incoming_weight, ' * ', incoming_activation, store_trace);
	            initialized = true;
	          }
	        }
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
	          var neuron = this.neighboors[id];
	          var influence = getVar(`influences[${neuron.ID}]`);

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
	                trace, ' * ', influence, store_trace);
	            else
	              buildSentence(xtrace, ' = ', neuron_self_weight, ' * ',
	                xtrace, ' + ', derivative, ' * ', trace, ' * ',
	                influence, store_trace);
	          else
	            buildSentence(xtrace, ' = ', derivative, ' * ', trace, ' * ',
	              influence, store_trace);
	        }
	      }
	      for (var connection in this.connections.gated) {
	        const gated_gain = getVar(this.connections.gated[connection], 'gain');
	        buildSentence(gated_gain, ' = ', activation, store_activation);
	      }
	    }
	    if (!isInput) {
	      const responsibility = getVar(this, 'error', 'responsibility', this.error
	        .responsibility);
	      if (isOutput) {
	        const target = getVar('target');
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
	          const error = getVar('aux');
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
	          const projected = getVar(this, 'error', 'projected', this.error.projected);
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
	          const gated = getVar(this, 'error', 'gated', this.error.gated);
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
	      inputs,
	      outputs,
	      targets,
	      variables,
	      activation_sentences,
	      trace_sentences,
	      propagation_sentences,
	      layers
	    }
	  }
	}
	// represents a connection between two neurons
	Neuron.connection = class NeuronConnection {
	  constructor(from, to, weight) {

	    if (!from || !to)
	      throw new Error("Connection Error: Invalid neurons");

	    this.ID = Neuron.connection.uid();
	    this.from = from;
	    this.to = to;
	    this.weight = typeof weight == 'undefined' ? Math.random() * .2 - .1 :
	      weight;
	    this.gain = 1;
	    this.gater = null;
	  }
	}


	// squashing functions
	Neuron.squash = {};

	function registerSquash(name, fn) {
	  Neuron.squash[name] = fn;
	  fn.squashType = [name];
	}

	// eq. 5 & 5'
	registerSquash('LOGISTIC', (x, derivate) => {
	  if (!derivate)
	    return 1 / (1 + Math.exp(-x));
	  const fx = Neuron.squash.LOGISTIC(x);
	  return fx * (1 - fx);
	});
	registerSquash('TANH', (x, derivate) => {
	  if (derivate)
	    return 1 - Math.pow(Neuron.squash.TANH(x), 2);
	  const eP = Math.exp(x);
	  const eN = 1 / eP;
	  return (eP - eN) / (eP + eN);
	});
	registerSquash('IDENTITY', (x, derivate) => derivate ? 1 : x);
	registerSquash('HLIM', (x, derivate) => derivate ? 1 : x > 0 ? 1 : 0);
	registerSquash('RELU', (x, derivate) => {
	  if (derivate)
	    return x > 0 ? 1 : 0;
	  return x > 0 ? x : 0;
	});

	// unique ID's
	((() => {
	  let neurons = 0;
	  let connections = 0;
	  Neuron.uid = () => neurons++
	  Neuron.connection.uid = () => connections++
	  Neuron.quantity = () => ({
	    neurons,
	    connections
	  })
	}))();

	module.exports = Neuron;


/***/ },
/* 3 */
/***/ function(module, exports, __webpack_require__) {

	const Neuron = __webpack_require__(2);

	/*******************************************************************************************
	 LAYER
	 *******************************************************************************************/

	class Layer {
	  constructor(size, label) {
	    this.label = label || null;
	    this.connectedTo = [];

	    if (Array.isArray(size)) {
	      const neurons = size;
	      this.size = neurons.length;
	      this.list = neurons.slice(); // cloning the array
	    } else {
	      this.size = size | 0;
	      this.list = [];

	      while (size--) {
	        const neuron = new Neuron();
	        this.list.push(neuron);
	      }
	    }
	  }

	  // activates all the neurons in the layer
	  activate(input) {

	    const activations = [];

	    if (typeof input != 'undefined') {
	      if (input.length != this.size)
	        throw new Error("INPUT size and LAYER size must be the same to activate!");

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
	  }

	  // propagates the error on all the neurons of the layer
	  propagate(rate, target) {

	    if (typeof target != 'undefined') {
	      if (target.length != this.size)
	        throw new Error("TARGET size and LAYER size must be the same to propagate!");

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
	  project(layer, type, weights) {
	    if (layer && layer.layers && layer.layers.input)
	      return this.project(layer.layers.input, type, weights);

	    if (layer instanceof Layer) {
	      if (!this.connected(layer))
	        return new Layer.connection(this, layer, type, weights);
	    } else {
	      throw new Error("Invalid argument, you can only project connections to LAYERS and NETWORKS!");
	    }
	  }

	  // gates a connection betwenn two layers
	  gate(connection, type) {

	    if (type == Layer.gateType.INPUT) {
	      if (connection.to.size != this.size)
	        throw new Error("GATER layer and CONNECTION.TO layer must be the same size in order to gate!");

	      for (var id in connection.to.list) {
	        var neuron = connection.to.list[id];
	        var gater = this.list[id];
	        for (let input in neuron.connections.inputs) {
	          var gated = neuron.connections.inputs[input];
	          if (gated.ID in connection.connections)
	            gater.gate(gated);
	        }
	      }
	    } else if (type == Layer.gateType.OUTPUT) {
	      if (connection.from.size != this.size)
	        throw new Error("GATER layer and CONNECTION.FROM layer must be the same size in order to gate!");

	      for (var id in connection.from.list) {
	        var neuron = connection.from.list[id];
	        var gater = this.list[id];
	        for (let projected in neuron.connections.projected) {
	          var gated = neuron.connections.projected[projected];
	          if (gated.ID in connection.connections)
	            gater.gate(gated);
	        }
	      }
	    } else if (type == Layer.gateType.ONE_TO_ONE) {
	      if (connection.size != this.size)
	        throw new Error("The number of GATER UNITS must be the same as the number of CONNECTIONS to gate!");

	      for (var id in connection.list) {
	        var gater = this.list[id];
	        var gated = connection.list[id];
	        gater.gate(gated);
	      }
	    }
	    connection.gatedfrom.push({layer: this, type});
	  }

	  // true or false whether the whole layer is self-connected or not
	  selfconnected() {

	    for (let id in this.list) {
	      const neuron = this.list[id];
	      if (!neuron.selfconnected())
	        return false;
	    }
	    return true;
	  }

	  // true of false whether the layer is connected to another layer (parameter) or not
	  connected(layer) {
	    // Check if ALL to ALL connection
	    let connections = 0;
	    for (let here in this.list) {
	      for (let there in layer.list) {
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
	    for (let neuron in this.list) {
	      var from = this.list[neuron];
	      var to = layer.list[neuron];
	      var connected = from.connected(to);
	      if (connected.type == 'projected')
	        connections++;
	    }
	    if (connections == this.size)
	      return Layer.connectionType.ONE_TO_ONE;
	  }

	  // clears all the neuorns in the layer
	  clear() {
	    for (let id in this.list) {
	      const neuron = this.list[id];
	      neuron.clear();
	    }
	  }

	  // resets all the neurons in the layer
	  reset() {
	    for (let id in this.list) {
	      const neuron = this.list[id];
	      neuron.reset();
	    }
	  }

	  // returns all the neurons in the layer (array)
	  neurons() {
	    return this.list;
	  }


	  set(options = {}) {
	    for (let i in this.list) {
	      const neuron = this.list[i];
	      if (options.label)
	        neuron.label = `${options.label}_${neuron.ID}`;
	      if (options.squash)
	        neuron.squash = options.squash;
	      if (options.bias)
	        neuron.bias = options.bias;
	    }
	    return this;
	  }
	}

	// represents a connection from one layer to another, and keeps track of its weight and gain
	Layer.connection = class LayerConnection {
	  constructor(fromLayer, toLayer, type, weights) {
	    this.ID = Layer.connection.uid();
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
	      for (let here in this.from.list) {
	        for (let there in this.to.list) {
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

	      for (let neuron in this.from.list) {
	        var from = this.from.list[neuron];
	        var to = this.to.list[neuron];
	        var connection = from.project(to, weights);

	        this.connections[connection.ID] = connection;
	        this.size = this.list.push(connection);
	      }
	    }

	    fromLayer.connectedTo.push(this);
	  }
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

	((() => {
	  let connections = 0;
	  Layer.connection.uid = () => connections++
	}))();

	module.exports = Layer;

/***/ },
/* 4 */
/***/ function(module, exports, __webpack_require__) {

	/* WEBPACK VAR INJECTION */(function(global) {const Network = __webpack_require__(5);

	class SynapticNetwork extends Network {
	}

	// features enabling
	if (global.Worker) {
	  // this code should be kept on
	  const WorkerProxyNetwork = __webpack_require__(6);
	  // Return a HTML5 WebWorker specialized on training the network stored in `memory`.
	  // Train based on the given dataSet and options.
	  // The worker returns the updated `memory` when done.
	  SynapticNetwork.prototype.worker = function () {
	    return WorkerProxyNetwork.fromJSON(this.toJSON());
	  }
	}

	module.exports = SynapticNetwork;
	/* WEBPACK VAR INJECTION */}.call(exports, (function() { return this; }())))

/***/ },
/* 5 */
/***/ function(module, exports, __webpack_require__) {

	const Layer = __webpack_require__(3);
	const Neuron = __webpack_require__(2);

	/*******************************************************************************************
	 NETWORK
	 *******************************************************************************************/

	class Network {
	  constructor(layers) {
	    this.layers = layers;
	    this.optimized = null;
	  }

	  // feed-forward activation of all the layers to produce an ouput
	  activate(input) {
	    if (this.optimized === false) {
	      this.layers.input.activate(input);
	      for (let layer in this.layers.hidden)
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
	  propagate(rate, target) {

	    if (this.optimized === false) {
	      this.layers.output.propagate(rate, target);
	      const reverse = [];
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

	    if (unit instanceof Layer)
	      return this.layers.output.project(unit, type, weights);

	    throw new Error("Invalid argument, you can only project connections to LAYERS and NETWORKS!");
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

	    const inputLayer = this.layers.input, outputLayer = this.layers.output;

	    inputLayer.clear();
	    for (let layer in this.layers.hidden) {
	      const hiddenLayer = this.layers.hidden[layer];
	      hiddenLayer.clear();
	    }
	    outputLayer.clear();

	    if (this.optimized)
	      this.optimized.reset();
	  }

	  // reset all weights and clear all traces (ends up like a new network)
	  reset() {

	    this.restore();

	    const inputLayer = this.layers.input, outputLayer = this.layers.output;

	    inputLayer.reset();
	    for (let layer in this.layers.hidden) {
	      const hiddenLayer = this.layers.hidden[layer];
	      hiddenLayer.reset();
	    }
	    outputLayer.reset();

	    if (this.optimized)
	      this.optimized.reset();
	  }

	  // hardcodes the behaviour of the whole network into a single optimized function
	  optimize() {

	    const that = this;
	    let optimized = {};
	    const neurons = this.neurons();

	    for (var i in neurons) {
	      let neuron = neurons[i].neuron;
	      const layer = neurons[i].layer;
	      while (neuron.neuron)
	        neuron = neuron.neuron;
	      optimized = neuron.optimize(optimized, layer);
	    }
	    for (var i in optimized.propagation_sentences)
	      optimized.propagation_sentences[i].reverse();
	    optimized.propagation_sentences.reverse();

	    let hardcode = "";
	    hardcode += `var F = Float64Array ? new Float64Array(${optimized.memory}) : []; `;
	    for (var i in optimized.variables)
	      hardcode += `F[${optimized.variables[i].id}] = ${optimized.variables[
	        i].value || 0}; `;
	    hardcode += "var activate = function(input){\n";
	    for (var i in optimized.inputs)
	      hardcode += `F[${optimized.inputs[i]}] = input[${i}]; `;
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
	      hardcode += `output[${i}] = F[${optimized.outputs[i]}]; `;
	    hardcode += "return output; }; "
	    hardcode += "var propagate = function(rate, target){\n";
	    hardcode += `F[${optimized.variables.rate.id}] = rate; `;
	    for (var i in optimized.targets)
	      hardcode += `F[${optimized.targets[i]}] = target[${i}]; `;
	    for (var currentLayer in optimized.propagation_sentences)
	      for (var currentNeuron in optimized.propagation_sentences[currentLayer])
	        hardcode += `${optimized.propagation_sentences[currentLayer][currentNeuron].join(" ")} `;
	    hardcode += " };\n";
	    hardcode +=
	      "var ownership = function(memoryBuffer){\nF = memoryBuffer;\nthis.memory = F;\n};\n";
	    hardcode +=
	      "return {\nmemory: F,\nactivate: activate,\npropagate: propagate,\nownership: ownership\n};";
	    hardcode = hardcode.split(";").join(";\n");

	    const constructor = new Function(hardcode);

	    const network = constructor();
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

	    network.reset = () => {
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

	    const optimized = this.optimized;

	    const getValue = function () {
	      const args = Array.prototype.slice.call(arguments);

	      const unit = args.shift();
	      const prop = args.pop();

	      let id = `${prop}_`;
	      for (let property in args)
	        id += `${args[property]}_`;
	      id += unit.ID;

	      const memory = optimized.memory;
	      const variables = optimized.data.variables;

	      if (id in variables)
	        return memory[variables[id].id];
	      return 0;
	    };

	    const list = this.neurons();

	    // link id's to positions in the array
	    const ids = {};
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

	      for (let gated in neuron.trace.extended)
	        for (var input in neuron.trace.extended[gated])
	          neuron.trace.extended[gated][input] = getValue(neuron, 'trace',
	            'extended', gated, input);
	    }

	    // get connections
	    for (var i in list) {
	      var neuron = list[i].neuron;
	      while (neuron.neuron)
	        neuron = neuron.neuron;

	      for (let j in neuron.connections.projected) {
	        const connection = neuron.connections.projected[j];
	        connection.weight = getValue(connection, 'weight');
	        connection.gain = getValue(connection, 'gain');
	      }
	    }
	  }

	  // returns all the neurons in the network
	  neurons() {

	    const neurons = [];

	    const inputLayer = this.layers.input.neurons(), outputLayer = this.layers.output.neurons();

	    for (var neuron in inputLayer)
	      neurons.push({
	        neuron: inputLayer[neuron],
	        layer: 'input'
	      });

	    for (let layer in this.layers.hidden) {
	      const hiddenLayer = this.layers.hidden[layer].neurons();
	      for (var neuron in hiddenLayer)
	        neurons.push({
	          neuron: hiddenLayer[neuron],
	          layer
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
	  inputs() {
	    return this.layers.input.size;
	  }

	  // returns number of outputs of hte network
	  outputs() {
	    return this.layers.output.size;
	  }

	  // sets the layers of the network
	  set(layers) {
	    console.warn('This method is deprecated! Use super(layers) in constructor instead');
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

	    const list = this.neurons();
	    const neurons = [];
	    const connections = [];

	    // link id's to positions in the array
	    const ids = {};
	    for (var i in list) {
	      var neuron = list[i].neuron;
	      while (neuron.neuron)
	        neuron = neuron.neuron;
	      ids[neuron.ID] = i;

	      const copy = {
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

	      copy.squash = neuron.squash.squashType || neuron.squash.toString();

	      neurons.push(copy);
	    }

	    // get connections
	    for (var i in list) {
	      var neuron = list[i].neuron;
	      while (neuron.neuron)
	        neuron = neuron.neuron;

	      for (let j in neuron.connections.projected) {
	        const connection = neuron.connections.projected[j];
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
	          gater: neuron.selfconnection.gater ? ids[neuron.selfconnection.gater.ID] : null,
	        });
	    }

	    return {
	      neurons,
	      connections
	    }
	  }

	  // export the topology into dot language which can be visualized as graphs using dot
	  /* example: ... console.log(net.toDotLang());
	   $ node example.js > example.dot
	   $ dot example.dot -Tpng > out.png
	   */
	  toDot(edgeConnection) {
	    if (!typeof edgeConnection)
	      edgeConnection = false;
	    let code = "digraph nn {\n    rankdir = BT\n";
	    const layers = [this.layers.input].concat(this.layers.hidden, this.layers.output);
	    for (let layer in layers) {
	      for (let to in layers[layer].connectedTo) { // projections
	        const connection = layers[layer].connectedTo[to];
	        const layerTo = connection.to;
	        const size = connection.size;
	        const layerID = layers.indexOf(layers[layer]);
	        const layerToID = layers.indexOf(layerTo);
	        /* http://stackoverflow.com/questions/26845540/connect-edges-with-graph-dot
	         * DOT does not support edge-to-edge connections
	         * This workaround produces somewhat weird graphs ...
	         */
	        if (edgeConnection) {
	          if (connection.gatedfrom.length) {
	            var fakeNode = `fake${layerID}_${layerToID}`;
	            code += `    ${fakeNode} [label = "", shape = point, width = 0.01, height = 0.01]\n`;
	            code += `    ${layerID} -> ${fakeNode} [label = ${size}, arrowhead = none]\n`;
	            code += `    ${fakeNode} -> ${layerToID}\n`;
	          } else
	            code += `    ${layerID} -> ${layerToID} [label = ${size}]\n`;
	          for (var from in connection.gatedfrom) { // gatings
	            var layerfrom = connection.gatedfrom[from].layer;
	            var layerfromID = layers.indexOf(layerfrom);
	            code += `    ${layerfromID} -> ${fakeNode} [color = blue]\n`;
	          }
	        } else {
	          code += `    ${layerID} -> ${layerToID} [label = ${size}]\n`;
	          for (var from in connection.gatedfrom) { // gatings
	            var layerfrom = connection.gatedfrom[from].layer;
	            var layerfromID = layers.indexOf(layerfrom);
	            code += `    ${layerfromID} -> ${layerToID} [color = blue]\n`;
	          }
	        }
	      }
	    }
	    code += "}\n";
	    return {
	      code,
	      link: `https://chart.googleapis.com/chart?chl=${escape(code.replace("/ /g", "+"))}&cht=gv`
	    }
	  }

	  // returns a function that works as the activation of the network and can be used without depending on the library
	  standalone() {
	    if (!this.optimized)
	      this.optimize();

	    const data = this.optimized.data;

	    // build activation function
	    let activation = "function (input) {\n";

	    // build inputs
	    for (var i in data.inputs)
	      activation += `F[${data.inputs[i]}] = input[${i}];\n`;

	    // build network activation
	    for (let neuron in data.activate) { // shouldn't this be layer?
	      for (let sentence in data.activate[neuron])
	        activation += `${data.activate[neuron][sentence].join('')}\n`;
	    }

	    // build outputs
	    activation += "var output = [];\n";
	    for (var i in data.outputs)
	      activation += `output[${i}] = F[${data.outputs[i]}];\n`;
	    activation += "return output;\n}";

	    // reference all the positions in memory
	    const memory = activation.match(/F\[(\d+)\]/g);
	    let dimension = 0;
	    const ids = {};
	    for (let address in memory) {
	      const tmp = memory[address].match(/\d+/)[0];
	      if (!(tmp in ids)) {
	        ids[tmp] = dimension++;
	      }
	    }
	    let hardcode = "F = {\n";
	    for (var i in ids)
	      hardcode += `${ids[i]}: ${this.optimized.memory[i]},\n`;
	    hardcode = `${hardcode.substring(0, hardcode.length - 2)}\n};\n`;
	    hardcode = `var run = ${activation.replace(/F\[(\d+)]/g, index => 'F[' + ids[index.match(/\d+/)[0]] + ']').replace("{\n", "{\n" + hardcode + "")};\n`;
	    hardcode += "return run";

	    // return standalone function
	    return new Function(hardcode)();
	  }

	  // returns a copy of the network
	  clone() {
	    return this.constructor.fromJSON(this.toJSON());
	  }
	}

	// rebuild a network that has been stored in a json using the method toJSON()
	Network.fromJSON = function (json) {

	  const neurons = [];

	  const layers = {
	    input: [],
	    hidden: [],
	    output: []
	  };

	  for (var i in json.neurons) {
	    var config = json.neurons[i];

	    const neuron = new Neuron();
	    neuron.trace.elegibility = {};
	    neuron.trace.extended = {};
	    neuron.state = config.state;
	    neuron.old = config.old;
	    neuron.activation = config.activation;
	    neuron.bias = config.bias;
	    if (Neuron.squash[config.squash] instanceof Function && Neuron.squash[config.squash].length === 1) {
	      neuron.squash = Neuron.squash[config.squash];
	    }

	    if (!neuron.squash) {
	      neuron.squash = Neuron.squash.LOGISTIC;
	    }

	    neurons.push(neuron);

	    if (config.layer == 'input')
	      layers.input.push(neuron);
	    else if (config.layer == 'output')
	      layers.output.push(neuron);
	    else {
	      if (typeof layers.hidden[config.layer] == 'undefined')
	        layers.hidden[config.layer] = [];
	      layers.hidden[config.layer].push(neuron);
	    }
	  }

	  for (var i in json.connections) {
	    var config = json.connections[i];
	    const from = neurons[config.from];
	    const to = neurons[config.to];
	    const weight = config.weight;
	    const gater = neurons[config.gater];

	    const connection = from.project(to, weight);
	    if (gater)
	      gater.gate(connection);
	  }

	  return new this({
	    input: new Layer(layers.input),
	    hidden: layers.hidden.map(layer => new Layer(layer)),
	    output: new Layer(layers.output)
	  });
	};

	module.exports = Network;


/***/ },
/* 6 */
/***/ function(module, exports, __webpack_require__) {

	var Network = __webpack_require__(5);
	var constants = __webpack_require__(7);
	var WorkerNetworkProxyAPI = __webpack_require__(8);

	const workerNetworkConstructorInternalKey = {};

	const workerNetworkProxyKey = '_workerNetworkProxyAPI';
	const getGetterName = key => `get${key[0].toUpperCase()}${key.slice(1)}`;
	const getSetterName = key => `set${key[0].toUpperCase()}${key.slice(1)}`;

	class WorkerProxyNetwork extends Network {
	  constructor(key, workerNetworkProxyAPI) {
	    if (key !== workerNetworkConstructorInternalKey)
	      throw new Error(
	        `cannot initiate ${this.name} directly as it is constructed in an async manner. ` +
	        `Instead of that use ${this.name}.fromJSON() or ${this.name}.construct`);

	    super();
	    this[workerNetworkProxyKey] = workerNetworkProxyAPI;
	  }
	}

	module.exports = WorkerProxyNetwork;

	WorkerProxyNetwork.construct = function (proxy) { return new this(workerNetworkConstructorInternalKey, proxy) };

	WorkerProxyNetwork.fromJSON = function (jsonAlikeObject, proxyAPI = new WorkerNetworkProxyAPI()) {
	  return proxyAPI.callMethod(constants.FROM_JSON, [jsonAlikeObject])
	    .then(() => this.construct(proxyAPI))
	};


	const prototypeKeys = Object.getOwnPropertyNames(Network.prototype);

	const methodKeys = prototypeKeys.filter(key => (Network.prototype[key] instanceof Function));
	const propertyKeys = prototypeKeys.filter(key => !(Network.prototype[key] instanceof Function));

	const createDef = (key, value, length) => {
	  Object.defineProperty(value, 'name', {value: key});
	  if (length !== undefined)
	    Object.defineProperty(value, 'length', {value: length});
	  return {key, descriptor: {value}};
	};

	const definitions = [
	  ...methodKeys
	    .map(key =>
	      createDef(key,
	        function (...args) { return this[workerNetworkProxyKey].callMethod(key, args) }, Network.prototype[key].length)),
	  ...propertyKeys
	    .map(key => ({
	      key,
	      descriptor: {
	        get() {
	          throw new Error(`cannot get property ${this.constructor.name}#${key} directly as actual data reflected is stored in a worker. `
	            + `To get it use method ${this.constructor.name}#${getGetterName(key)} with a signature () => Promise<value>`)
	        },
	        set() {
	          throw new Error(`cannot set property ${this.constructor.name}#${key} directly as actual data reflected is stored in a worker. `
	            + `To set it use method ${this.constructor.name}#${getSetterName(key)} with a signature (value) => Promise<>`)
	        }
	      }
	    })),
	  ...propertyKeys
	    .map(key => createDef(getGetterName(key),
	      function () { return this[workerNetworkProxyKey].callMethod(constants.GET, [key]) })),
	  ...propertyKeys
	    .map(key => createDef(getSetterName(key),
	      function (value) { return this[workerNetworkProxyKey].callMethod(constants.SET, [key, value]) }))
	];

	for (const definition of definitions)
	  Object.defineProperty(WorkerProxyNetwork.prototype, definition.key, definition.descriptor);

/***/ },
/* 7 */
/***/ function(module, exports) {

	module.exports.GET = '%get';
	module.exports.SET = '%set';
	module.exports.FROM_JSON = '%fromJSON';

/***/ },
/* 8 */
/***/ function(module, exports, __webpack_require__) {

	var WorkerNetworkWorker = __webpack_require__(9);

	module.exports = class WorkerNetworkProxyAPI {
	  constructor() {
	    var worker = this.worker = new WorkerNetworkWorker();
	    this.callbacks = {};
	    this.nextCallbackId = 1;

	    worker.onmessage = ({data: {response: [error, data] = [], callbackId} = {}}) =>
	      this.callbacks[callbackId](error, data);
	  }

	  subscribe(method, args, callback) {
	    const callbackId = this.nextCallbackId++;
	    this.callbacks[callbackId] = callback;
	    this.worker.postMessage({method, args, callbackId});
	    return callbackId;
	  }

	  unsubscribe(callbackId) {
	    return delete this.callbacks[callbackId];
	  }

	  callMethod(method, args) {
	    return new Promise((resolve, reject) => {
	      const callbackId = this.subscribe(method, args, (err, data) => {
	        this.unsubscribe(callbackId);
	        err ? reject(err) : resolve(data);
	      });
	    })
	  };
	};


/***/ },
/* 9 */
/***/ function(module, exports, __webpack_require__) {

	module.exports = function() {
		return __webpack_require__(10)("/*!\n * The MIT License (MIT)\n * \n * Copyright (c) 2016 Juan Cazala - juancazala.com\n * \n * Permission is hereby granted, free of charge, to any person obtaining a copy\n * of this software and associated documentation files (the \"Software\"), to deal\n * in the Software without restriction, including without limitation the rights\n * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n * copies of the Software, and to permit persons to whom the Software is\n * furnished to do so, subject to the following conditions:\n * \n * The above copyright notice and this permission notice shall be included in\n * all copies or substantial portions of the Software.\n * \n * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\n * THE SOFTWARE\n * \n * \n * \n * ********************************************************************************************\n *                                   SYNAPTIC (v1.0.8)\n * ********************************************************************************************\n * \n * Synaptic is a javascript neural network library for node.js and the browser, its generalized\n * algorithm is architecture-free, so you can build and train basically any type of first order\n * or even second order neural network architectures.\n * \n * http://en.wikipedia.org/wiki/Recurrent_neural_network#Second_Order_Recurrent_Neural_Network\n * \n * The library includes a few built-in architectures like multilayer perceptrons, multilayer\n * long-short term memory networks (LSTM) or liquid state machines, and a trainer capable of\n * training any given network, and includes built-in training tasks/tests like solving an XOR,\n * passing a Distracted Sequence Recall test or an Embeded Reber Grammar test.\n * \n * The algorithm implemented by this library has been taken from Derek D. Monner's paper:\n * \n * \n * A generalized LSTM-like training algorithm for second-order recurrent neural networks\n * http://www.overcomplete.net/papers/nn2012.pdf\n * \n * There are references to the equations in that paper commented through the source code.\n * \n */\n/******/ (function(modules) { // webpackBootstrap\n/******/ \t// The module cache\n/******/ \tvar installedModules = {};\n\n/******/ \t// The require function\n/******/ \tfunction __webpack_require__(moduleId) {\n\n/******/ \t\t// Check if module is in cache\n/******/ \t\tif(installedModules[moduleId])\n/******/ \t\t\treturn installedModules[moduleId].exports;\n\n/******/ \t\t// Create a new module (and put it into the cache)\n/******/ \t\tvar module = installedModules[moduleId] = {\n/******/ \t\t\texports: {},\n/******/ \t\t\tid: moduleId,\n/******/ \t\t\tloaded: false\n/******/ \t\t};\n\n/******/ \t\t// Execute the module function\n/******/ \t\tmodules[moduleId].call(module.exports, module, module.exports, __webpack_require__);\n\n/******/ \t\t// Flag the module as loaded\n/******/ \t\tmodule.loaded = true;\n\n/******/ \t\t// Return the exports of the module\n/******/ \t\treturn module.exports;\n/******/ \t}\n\n\n/******/ \t// expose the modules object (__webpack_modules__)\n/******/ \t__webpack_require__.m = modules;\n\n/******/ \t// expose the module cache\n/******/ \t__webpack_require__.c = installedModules;\n\n/******/ \t// __webpack_public_path__\n/******/ \t__webpack_require__.p = \"\";\n\n/******/ \t// Load entry module and return exports\n/******/ \treturn __webpack_require__(0);\n/******/ })\n/************************************************************************/\n/******/ ([\n/* 0 */\n/***/ function(module, exports, __webpack_require__) {\n\n\tvar WorkerNetwork = __webpack_require__(1);\n\tvar constants = __webpack_require__(4);\n\n\tonmessage = function (e) {\n\t  const {method, args, callbackId} = e.data;\n\t  try {\n\t    var response = callMethod(method, args);\n\t    postMessage({response: [null, response], callbackId})\n\t  } catch (e) {\n\t    postMessage({response: [e.message], callbackId})\n\t  }\n\t};\n\n\tconst callMethod = (function() {\n\t  var instance;\n\n\t  return function callMethod(method, args) {\n\t    var result;\n\t    switch (method) {\n\t      case constants.GET:\n\t        result = instance[args[0]];\n\t        break;\n\t      case constants.SET:\n\t        instance[method][args[0]] = args[1];\n\t        break;\n\t      case constants.FROM_JSON:\n\t        instance = WorkerNetwork.fromJSON(args[0]);\n\t        break;\n\t      default:\n\t        result = instance[method](...args);\n\t        break;\n\t    }\n\n\n\t    return result;\n\t  }\n\t})();\n\n\n\n/***/ },\n/* 1 */\n/***/ function(module, exports, __webpack_require__) {\n\n\tconst Layer = __webpack_require__(2);\n\tconst Neuron = __webpack_require__(3);\n\n\t/*******************************************************************************************\n\t NETWORK\n\t *******************************************************************************************/\n\n\tclass Network {\n\t  constructor(layers) {\n\t    this.layers = layers;\n\t    this.optimized = null;\n\t  }\n\n\t  // feed-forward activation of all the layers to produce an ouput\n\t  activate(input) {\n\t    if (this.optimized === false) {\n\t      this.layers.input.activate(input);\n\t      for (let layer in this.layers.hidden)\n\t        this.layers.hidden[layer].activate();\n\t      return this.layers.output.activate();\n\t    }\n\t    else {\n\t      if (this.optimized == null)\n\t        this.optimize();\n\t      return this.optimized.activate(input);\n\t    }\n\t  }\n\n\t  // back-propagate the error thru the network\n\t  propagate(rate, target) {\n\n\t    if (this.optimized === false) {\n\t      this.layers.output.propagate(rate, target);\n\t      const reverse = [];\n\t      for (var layer in this.layers.hidden)\n\t        reverse.push(this.layers.hidden[layer]);\n\t      reverse.reverse();\n\t      for (var layer in reverse)\n\t        reverse[layer].propagate(rate);\n\t    }\n\t    else {\n\t      if (this.optimized == null)\n\t        this.optimize();\n\t      this.optimized.propagate(rate, target);\n\t    }\n\t  }\n\n\t  // project a connection to another unit (either a network or a layer)\n\t  project(unit, type, weights) {\n\n\t    if (this.optimized)\n\t      this.optimized.reset();\n\n\t    if (unit instanceof Network)\n\t      return this.layers.output.project(unit.layers.input, type, weights);\n\n\t    if (unit instanceof Layer)\n\t      return this.layers.output.project(unit, type, weights);\n\n\t    throw new Error(\"Invalid argument, you can only project connections to LAYERS and NETWORKS!\");\n\t  }\n\n\t  // let this network gate a connection\n\t  gate(connection, type) {\n\t    if (this.optimized)\n\t      this.optimized.reset();\n\t    this.layers.output.gate(connection, type);\n\t  }\n\n\t  // clear all elegibility traces and extended elegibility traces (the network forgets its context, but not what was trained)\n\t  clear() {\n\n\t    this.restore();\n\n\t    const inputLayer = this.layers.input, outputLayer = this.layers.output;\n\n\t    inputLayer.clear();\n\t    for (let layer in this.layers.hidden) {\n\t      const hiddenLayer = this.layers.hidden[layer];\n\t      hiddenLayer.clear();\n\t    }\n\t    outputLayer.clear();\n\n\t    if (this.optimized)\n\t      this.optimized.reset();\n\t  }\n\n\t  // reset all weights and clear all traces (ends up like a new network)\n\t  reset() {\n\n\t    this.restore();\n\n\t    const inputLayer = this.layers.input, outputLayer = this.layers.output;\n\n\t    inputLayer.reset();\n\t    for (let layer in this.layers.hidden) {\n\t      const hiddenLayer = this.layers.hidden[layer];\n\t      hiddenLayer.reset();\n\t    }\n\t    outputLayer.reset();\n\n\t    if (this.optimized)\n\t      this.optimized.reset();\n\t  }\n\n\t  // hardcodes the behaviour of the whole network into a single optimized function\n\t  optimize() {\n\n\t    const that = this;\n\t    let optimized = {};\n\t    const neurons = this.neurons();\n\n\t    for (var i in neurons) {\n\t      let neuron = neurons[i].neuron;\n\t      const layer = neurons[i].layer;\n\t      while (neuron.neuron)\n\t        neuron = neuron.neuron;\n\t      optimized = neuron.optimize(optimized, layer);\n\t    }\n\t    for (var i in optimized.propagation_sentences)\n\t      optimized.propagation_sentences[i].reverse();\n\t    optimized.propagation_sentences.reverse();\n\n\t    let hardcode = \"\";\n\t    hardcode += `var F = Float64Array ? new Float64Array(${optimized.memory}) : []; `;\n\t    for (var i in optimized.variables)\n\t      hardcode += `F[${optimized.variables[i].id}] = ${optimized.variables[\n\t        i].value || 0}; `;\n\t    hardcode += \"var activate = function(input){\\n\";\n\t    for (var i in optimized.inputs)\n\t      hardcode += `F[${optimized.inputs[i]}] = input[${i}]; `;\n\t    for (var currentLayer in optimized.activation_sentences) {\n\t      if (optimized.activation_sentences[currentLayer].length > 0) {\n\t        for (var currentNeuron in optimized.activation_sentences[currentLayer]) {\n\t          hardcode += optimized.activation_sentences[currentLayer][currentNeuron].join(\" \");\n\t          hardcode += optimized.trace_sentences[currentLayer][currentNeuron].join(\" \");\n\t        }\n\t      }\n\t    }\n\t    hardcode += \" var output = []; \"\n\t    for (var i in optimized.outputs)\n\t      hardcode += `output[${i}] = F[${optimized.outputs[i]}]; `;\n\t    hardcode += \"return output; }; \"\n\t    hardcode += \"var propagate = function(rate, target){\\n\";\n\t    hardcode += `F[${optimized.variables.rate.id}] = rate; `;\n\t    for (var i in optimized.targets)\n\t      hardcode += `F[${optimized.targets[i]}] = target[${i}]; `;\n\t    for (var currentLayer in optimized.propagation_sentences)\n\t      for (var currentNeuron in optimized.propagation_sentences[currentLayer])\n\t        hardcode += `${optimized.propagation_sentences[currentLayer][currentNeuron].join(\" \")} `;\n\t    hardcode += \" };\\n\";\n\t    hardcode +=\n\t      \"var ownership = function(memoryBuffer){\\nF = memoryBuffer;\\nthis.memory = F;\\n};\\n\";\n\t    hardcode +=\n\t      \"return {\\nmemory: F,\\nactivate: activate,\\npropagate: propagate,\\nownership: ownership\\n};\";\n\t    hardcode = hardcode.split(\";\").join(\";\\n\");\n\n\t    const constructor = new Function(hardcode);\n\n\t    const network = constructor();\n\t    network.data = {\n\t      variables: optimized.variables,\n\t      activate: optimized.activation_sentences,\n\t      propagate: optimized.propagation_sentences,\n\t      trace: optimized.trace_sentences,\n\t      inputs: optimized.inputs,\n\t      outputs: optimized.outputs,\n\t      check_activation: this.activate,\n\t      check_propagation: this.propagate\n\t    }\n\n\t    network.reset = () => {\n\t      if (that.optimized) {\n\t        that.optimized = null;\n\t        that.activate = network.data.check_activation;\n\t        that.propagate = network.data.check_propagation;\n\t      }\n\t    }\n\n\t    this.optimized = network;\n\t    this.activate = network.activate;\n\t    this.propagate = network.propagate;\n\t  }\n\n\t  // restores all the values from the optimized network the their respective objects in order to manipulate the network\n\t  restore() {\n\t    if (!this.optimized)\n\t      return;\n\n\t    const optimized = this.optimized;\n\n\t    const getValue = function () {\n\t      const args = Array.prototype.slice.call(arguments);\n\n\t      const unit = args.shift();\n\t      const prop = args.pop();\n\n\t      let id = `${prop}_`;\n\t      for (let property in args)\n\t        id += `${args[property]}_`;\n\t      id += unit.ID;\n\n\t      const memory = optimized.memory;\n\t      const variables = optimized.data.variables;\n\n\t      if (id in variables)\n\t        return memory[variables[id].id];\n\t      return 0;\n\t    };\n\n\t    const list = this.neurons();\n\n\t    // link id's to positions in the array\n\t    const ids = {};\n\t    for (var i in list) {\n\t      var neuron = list[i].neuron;\n\t      while (neuron.neuron)\n\t        neuron = neuron.neuron;\n\n\t      neuron.state = getValue(neuron, 'state');\n\t      neuron.old = getValue(neuron, 'old');\n\t      neuron.activation = getValue(neuron, 'activation');\n\t      neuron.bias = getValue(neuron, 'bias');\n\n\t      for (var input in neuron.trace.elegibility)\n\t        neuron.trace.elegibility[input] = getValue(neuron, 'trace',\n\t          'elegibility', input);\n\n\t      for (let gated in neuron.trace.extended)\n\t        for (var input in neuron.trace.extended[gated])\n\t          neuron.trace.extended[gated][input] = getValue(neuron, 'trace',\n\t            'extended', gated, input);\n\t    }\n\n\t    // get connections\n\t    for (var i in list) {\n\t      var neuron = list[i].neuron;\n\t      while (neuron.neuron)\n\t        neuron = neuron.neuron;\n\n\t      for (let j in neuron.connections.projected) {\n\t        const connection = neuron.connections.projected[j];\n\t        connection.weight = getValue(connection, 'weight');\n\t        connection.gain = getValue(connection, 'gain');\n\t      }\n\t    }\n\t  }\n\n\t  // returns all the neurons in the network\n\t  neurons() {\n\n\t    const neurons = [];\n\n\t    const inputLayer = this.layers.input.neurons(), outputLayer = this.layers.output.neurons();\n\n\t    for (var neuron in inputLayer)\n\t      neurons.push({\n\t        neuron: inputLayer[neuron],\n\t        layer: 'input'\n\t      });\n\n\t    for (let layer in this.layers.hidden) {\n\t      const hiddenLayer = this.layers.hidden[layer].neurons();\n\t      for (var neuron in hiddenLayer)\n\t        neurons.push({\n\t          neuron: hiddenLayer[neuron],\n\t          layer\n\t        });\n\t    }\n\t    for (var neuron in outputLayer)\n\t      neurons.push({\n\t        neuron: outputLayer[neuron],\n\t        layer: 'output'\n\t      });\n\n\t    return neurons;\n\t  }\n\n\t  // returns number of inputs of the network\n\t  inputs() {\n\t    return this.layers.input.size;\n\t  }\n\n\t  // returns number of outputs of hte network\n\t  outputs() {\n\t    return this.layers.output.size;\n\t  }\n\n\t  // sets the layers of the network\n\t  set(layers) {\n\t    console.warn('This method is deprecated! Use super(layers) in constructor instead');\n\t    this.layers = layers;\n\t    if (this.optimized)\n\t      this.optimized.reset();\n\t  }\n\n\t  setOptimize(bool) {\n\t    this.restore();\n\t    if (this.optimized)\n\t      this.optimized.reset();\n\t    this.optimized = bool ? null : false;\n\t  }\n\n\t  // returns a json that represents all the neurons and connections of the network\n\t  toJSON(ignoreTraces) {\n\n\t    this.restore();\n\n\t    const list = this.neurons();\n\t    const neurons = [];\n\t    const connections = [];\n\n\t    // link id's to positions in the array\n\t    const ids = {};\n\t    for (var i in list) {\n\t      var neuron = list[i].neuron;\n\t      while (neuron.neuron)\n\t        neuron = neuron.neuron;\n\t      ids[neuron.ID] = i;\n\n\t      const copy = {\n\t        trace: {\n\t          elegibility: {},\n\t          extended: {}\n\t        },\n\t        state: neuron.state,\n\t        old: neuron.old,\n\t        activation: neuron.activation,\n\t        bias: neuron.bias,\n\t        layer: list[i].layer\n\t      };\n\n\t      copy.squash = neuron.squash.squashType || neuron.squash.toString();\n\n\t      neurons.push(copy);\n\t    }\n\n\t    // get connections\n\t    for (var i in list) {\n\t      var neuron = list[i].neuron;\n\t      while (neuron.neuron)\n\t        neuron = neuron.neuron;\n\n\t      for (let j in neuron.connections.projected) {\n\t        const connection = neuron.connections.projected[j];\n\t        connections.push({\n\t          from: ids[connection.from.ID],\n\t          to: ids[connection.to.ID],\n\t          weight: connection.weight,\n\t          gater: connection.gater ? ids[connection.gater.ID] : null,\n\t        });\n\t      }\n\t      if (neuron.selfconnected())\n\t        connections.push({\n\t          from: ids[neuron.ID],\n\t          to: ids[neuron.ID],\n\t          weight: neuron.selfconnection.weight,\n\t          gater: neuron.selfconnection.gater ? ids[neuron.selfconnection.gater.ID] : null,\n\t        });\n\t    }\n\n\t    return {\n\t      neurons,\n\t      connections\n\t    }\n\t  }\n\n\t  // export the topology into dot language which can be visualized as graphs using dot\n\t  /* example: ... console.log(net.toDotLang());\n\t   $ node example.js > example.dot\n\t   $ dot example.dot -Tpng > out.png\n\t   */\n\t  toDot(edgeConnection) {\n\t    if (!typeof edgeConnection)\n\t      edgeConnection = false;\n\t    let code = \"digraph nn {\\n    rankdir = BT\\n\";\n\t    const layers = [this.layers.input].concat(this.layers.hidden, this.layers.output);\n\t    for (let layer in layers) {\n\t      for (let to in layers[layer].connectedTo) { // projections\n\t        const connection = layers[layer].connectedTo[to];\n\t        const layerTo = connection.to;\n\t        const size = connection.size;\n\t        const layerID = layers.indexOf(layers[layer]);\n\t        const layerToID = layers.indexOf(layerTo);\n\t        /* http://stackoverflow.com/questions/26845540/connect-edges-with-graph-dot\n\t         * DOT does not support edge-to-edge connections\n\t         * This workaround produces somewhat weird graphs ...\n\t         */\n\t        if (edgeConnection) {\n\t          if (connection.gatedfrom.length) {\n\t            var fakeNode = `fake${layerID}_${layerToID}`;\n\t            code += `    ${fakeNode} [label = \"\", shape = point, width = 0.01, height = 0.01]\\n`;\n\t            code += `    ${layerID} -> ${fakeNode} [label = ${size}, arrowhead = none]\\n`;\n\t            code += `    ${fakeNode} -> ${layerToID}\\n`;\n\t          } else\n\t            code += `    ${layerID} -> ${layerToID} [label = ${size}]\\n`;\n\t          for (var from in connection.gatedfrom) { // gatings\n\t            var layerfrom = connection.gatedfrom[from].layer;\n\t            var layerfromID = layers.indexOf(layerfrom);\n\t            code += `    ${layerfromID} -> ${fakeNode} [color = blue]\\n`;\n\t          }\n\t        } else {\n\t          code += `    ${layerID} -> ${layerToID} [label = ${size}]\\n`;\n\t          for (var from in connection.gatedfrom) { // gatings\n\t            var layerfrom = connection.gatedfrom[from].layer;\n\t            var layerfromID = layers.indexOf(layerfrom);\n\t            code += `    ${layerfromID} -> ${layerToID} [color = blue]\\n`;\n\t          }\n\t        }\n\t      }\n\t    }\n\t    code += \"}\\n\";\n\t    return {\n\t      code,\n\t      link: `https://chart.googleapis.com/chart?chl=${escape(code.replace(\"/ /g\", \"+\"))}&cht=gv`\n\t    }\n\t  }\n\n\t  // returns a function that works as the activation of the network and can be used without depending on the library\n\t  standalone() {\n\t    if (!this.optimized)\n\t      this.optimize();\n\n\t    const data = this.optimized.data;\n\n\t    // build activation function\n\t    let activation = \"function (input) {\\n\";\n\n\t    // build inputs\n\t    for (var i in data.inputs)\n\t      activation += `F[${data.inputs[i]}] = input[${i}];\\n`;\n\n\t    // build network activation\n\t    for (let neuron in data.activate) { // shouldn't this be layer?\n\t      for (let sentence in data.activate[neuron])\n\t        activation += `${data.activate[neuron][sentence].join('')}\\n`;\n\t    }\n\n\t    // build outputs\n\t    activation += \"var output = [];\\n\";\n\t    for (var i in data.outputs)\n\t      activation += `output[${i}] = F[${data.outputs[i]}];\\n`;\n\t    activation += \"return output;\\n}\";\n\n\t    // reference all the positions in memory\n\t    const memory = activation.match(/F\\[(\\d+)\\]/g);\n\t    let dimension = 0;\n\t    const ids = {};\n\t    for (let address in memory) {\n\t      const tmp = memory[address].match(/\\d+/)[0];\n\t      if (!(tmp in ids)) {\n\t        ids[tmp] = dimension++;\n\t      }\n\t    }\n\t    let hardcode = \"F = {\\n\";\n\t    for (var i in ids)\n\t      hardcode += `${ids[i]}: ${this.optimized.memory[i]},\\n`;\n\t    hardcode = `${hardcode.substring(0, hardcode.length - 2)}\\n};\\n`;\n\t    hardcode = `var run = ${activation.replace(/F\\[(\\d+)]/g, index => 'F[' + ids[index.match(/\\d+/)[0]] + ']').replace(\"{\\n\", \"{\\n\" + hardcode + \"\")};\\n`;\n\t    hardcode += \"return run\";\n\n\t    // return standalone function\n\t    return new Function(hardcode)();\n\t  }\n\n\t  // returns a copy of the network\n\t  clone() {\n\t    return this.constructor.fromJSON(this.toJSON());\n\t  }\n\t}\n\n\t// rebuild a network that has been stored in a json using the method toJSON()\n\tNetwork.fromJSON = function (json) {\n\n\t  const neurons = [];\n\n\t  const layers = {\n\t    input: [],\n\t    hidden: [],\n\t    output: []\n\t  };\n\n\t  for (var i in json.neurons) {\n\t    var config = json.neurons[i];\n\n\t    const neuron = new Neuron();\n\t    neuron.trace.elegibility = {};\n\t    neuron.trace.extended = {};\n\t    neuron.state = config.state;\n\t    neuron.old = config.old;\n\t    neuron.activation = config.activation;\n\t    neuron.bias = config.bias;\n\t    if (Neuron.squash[config.squash] instanceof Function && Neuron.squash[config.squash].length === 1) {\n\t      neuron.squash = Neuron.squash[config.squash];\n\t    }\n\n\t    if (!neuron.squash) {\n\t      neuron.squash = Neuron.squash.LOGISTIC;\n\t    }\n\n\t    neurons.push(neuron);\n\n\t    if (config.layer == 'input')\n\t      layers.input.push(neuron);\n\t    else if (config.layer == 'output')\n\t      layers.output.push(neuron);\n\t    else {\n\t      if (typeof layers.hidden[config.layer] == 'undefined')\n\t        layers.hidden[config.layer] = [];\n\t      layers.hidden[config.layer].push(neuron);\n\t    }\n\t  }\n\n\t  for (var i in json.connections) {\n\t    var config = json.connections[i];\n\t    const from = neurons[config.from];\n\t    const to = neurons[config.to];\n\t    const weight = config.weight;\n\t    const gater = neurons[config.gater];\n\n\t    const connection = from.project(to, weight);\n\t    if (gater)\n\t      gater.gate(connection);\n\t  }\n\n\t  return new this({\n\t    input: new Layer(layers.input),\n\t    hidden: layers.hidden.map(layer => new Layer(layer)),\n\t    output: new Layer(layers.output)\n\t  });\n\t};\n\n\tmodule.exports = Network;\n\n\n/***/ },\n/* 2 */\n/***/ function(module, exports, __webpack_require__) {\n\n\tconst Neuron = __webpack_require__(3);\n\n\t/*******************************************************************************************\n\t LAYER\n\t *******************************************************************************************/\n\n\tclass Layer {\n\t  constructor(size, label) {\n\t    this.label = label || null;\n\t    this.connectedTo = [];\n\n\t    if (Array.isArray(size)) {\n\t      const neurons = size;\n\t      this.size = neurons.length;\n\t      this.list = neurons.slice(); // cloning the array\n\t    } else {\n\t      this.size = size | 0;\n\t      this.list = [];\n\n\t      while (size--) {\n\t        const neuron = new Neuron();\n\t        this.list.push(neuron);\n\t      }\n\t    }\n\t  }\n\n\t  // activates all the neurons in the layer\n\t  activate(input) {\n\n\t    const activations = [];\n\n\t    if (typeof input != 'undefined') {\n\t      if (input.length != this.size)\n\t        throw new Error(\"INPUT size and LAYER size must be the same to activate!\");\n\n\t      for (var id in this.list) {\n\t        var neuron = this.list[id];\n\t        var activation = neuron.activate(input[id]);\n\t        activations.push(activation);\n\t      }\n\t    } else {\n\t      for (var id in this.list) {\n\t        var neuron = this.list[id];\n\t        var activation = neuron.activate();\n\t        activations.push(activation);\n\t      }\n\t    }\n\t    return activations;\n\t  }\n\n\t  // propagates the error on all the neurons of the layer\n\t  propagate(rate, target) {\n\n\t    if (typeof target != 'undefined') {\n\t      if (target.length != this.size)\n\t        throw new Error(\"TARGET size and LAYER size must be the same to propagate!\");\n\n\t      for (var id = this.list.length - 1; id >= 0; id--) {\n\t        var neuron = this.list[id];\n\t        neuron.propagate(rate, target[id]);\n\t      }\n\t    } else {\n\t      for (var id = this.list.length - 1; id >= 0; id--) {\n\t        var neuron = this.list[id];\n\t        neuron.propagate(rate);\n\t      }\n\t    }\n\t  }\n\n\t  // projects a connection from this layer to another one\n\t  project(layer, type, weights) {\n\t    if (layer && layer.layers && layer.layers.input)\n\t      return this.project(layer.layers.input, type, weights);\n\n\t    if (layer instanceof Layer) {\n\t      if (!this.connected(layer))\n\t        return new Layer.connection(this, layer, type, weights);\n\t    } else {\n\t      throw new Error(\"Invalid argument, you can only project connections to LAYERS and NETWORKS!\");\n\t    }\n\t  }\n\n\t  // gates a connection betwenn two layers\n\t  gate(connection, type) {\n\n\t    if (type == Layer.gateType.INPUT) {\n\t      if (connection.to.size != this.size)\n\t        throw new Error(\"GATER layer and CONNECTION.TO layer must be the same size in order to gate!\");\n\n\t      for (var id in connection.to.list) {\n\t        var neuron = connection.to.list[id];\n\t        var gater = this.list[id];\n\t        for (let input in neuron.connections.inputs) {\n\t          var gated = neuron.connections.inputs[input];\n\t          if (gated.ID in connection.connections)\n\t            gater.gate(gated);\n\t        }\n\t      }\n\t    } else if (type == Layer.gateType.OUTPUT) {\n\t      if (connection.from.size != this.size)\n\t        throw new Error(\"GATER layer and CONNECTION.FROM layer must be the same size in order to gate!\");\n\n\t      for (var id in connection.from.list) {\n\t        var neuron = connection.from.list[id];\n\t        var gater = this.list[id];\n\t        for (let projected in neuron.connections.projected) {\n\t          var gated = neuron.connections.projected[projected];\n\t          if (gated.ID in connection.connections)\n\t            gater.gate(gated);\n\t        }\n\t      }\n\t    } else if (type == Layer.gateType.ONE_TO_ONE) {\n\t      if (connection.size != this.size)\n\t        throw new Error(\"The number of GATER UNITS must be the same as the number of CONNECTIONS to gate!\");\n\n\t      for (var id in connection.list) {\n\t        var gater = this.list[id];\n\t        var gated = connection.list[id];\n\t        gater.gate(gated);\n\t      }\n\t    }\n\t    connection.gatedfrom.push({layer: this, type});\n\t  }\n\n\t  // true or false whether the whole layer is self-connected or not\n\t  selfconnected() {\n\n\t    for (let id in this.list) {\n\t      const neuron = this.list[id];\n\t      if (!neuron.selfconnected())\n\t        return false;\n\t    }\n\t    return true;\n\t  }\n\n\t  // true of false whether the layer is connected to another layer (parameter) or not\n\t  connected(layer) {\n\t    // Check if ALL to ALL connection\n\t    let connections = 0;\n\t    for (let here in this.list) {\n\t      for (let there in layer.list) {\n\t        var from = this.list[here];\n\t        var to = layer.list[there];\n\t        var connected = from.connected(to);\n\t        if (connected.type == 'projected')\n\t          connections++;\n\t      }\n\t    }\n\t    if (connections == this.size * layer.size)\n\t      return Layer.connectionType.ALL_TO_ALL;\n\n\t    // Check if ONE to ONE connection\n\t    connections = 0;\n\t    for (let neuron in this.list) {\n\t      var from = this.list[neuron];\n\t      var to = layer.list[neuron];\n\t      var connected = from.connected(to);\n\t      if (connected.type == 'projected')\n\t        connections++;\n\t    }\n\t    if (connections == this.size)\n\t      return Layer.connectionType.ONE_TO_ONE;\n\t  }\n\n\t  // clears all the neuorns in the layer\n\t  clear() {\n\t    for (let id in this.list) {\n\t      const neuron = this.list[id];\n\t      neuron.clear();\n\t    }\n\t  }\n\n\t  // resets all the neurons in the layer\n\t  reset() {\n\t    for (let id in this.list) {\n\t      const neuron = this.list[id];\n\t      neuron.reset();\n\t    }\n\t  }\n\n\t  // returns all the neurons in the layer (array)\n\t  neurons() {\n\t    return this.list;\n\t  }\n\n\n\t  set(options = {}) {\n\t    for (let i in this.list) {\n\t      const neuron = this.list[i];\n\t      if (options.label)\n\t        neuron.label = `${options.label}_${neuron.ID}`;\n\t      if (options.squash)\n\t        neuron.squash = options.squash;\n\t      if (options.bias)\n\t        neuron.bias = options.bias;\n\t    }\n\t    return this;\n\t  }\n\t}\n\n\t// represents a connection from one layer to another, and keeps track of its weight and gain\n\tLayer.connection = class LayerConnection {\n\t  constructor(fromLayer, toLayer, type, weights) {\n\t    this.ID = Layer.connection.uid();\n\t    this.from = fromLayer;\n\t    this.to = toLayer;\n\t    this.selfconnection = toLayer == fromLayer;\n\t    this.type = type;\n\t    this.connections = {};\n\t    this.list = [];\n\t    this.size = 0;\n\t    this.gatedfrom = [];\n\n\t    if (typeof this.type == 'undefined') {\n\t      if (fromLayer == toLayer)\n\t        this.type = Layer.connectionType.ONE_TO_ONE;\n\t      else\n\t        this.type = Layer.connectionType.ALL_TO_ALL;\n\t    }\n\n\t    if (this.type == Layer.connectionType.ALL_TO_ALL ||\n\t      this.type == Layer.connectionType.ALL_TO_ELSE) {\n\t      for (let here in this.from.list) {\n\t        for (let there in this.to.list) {\n\t          var from = this.from.list[here];\n\t          var to = this.to.list[there];\n\t          if (this.type == Layer.connectionType.ALL_TO_ELSE && from == to)\n\t            continue;\n\t          var connection = from.project(to, weights);\n\n\t          this.connections[connection.ID] = connection;\n\t          this.size = this.list.push(connection);\n\t        }\n\t      }\n\t    } else if (this.type == Layer.connectionType.ONE_TO_ONE) {\n\n\t      for (let neuron in this.from.list) {\n\t        var from = this.from.list[neuron];\n\t        var to = this.to.list[neuron];\n\t        var connection = from.project(to, weights);\n\n\t        this.connections[connection.ID] = connection;\n\t        this.size = this.list.push(connection);\n\t      }\n\t    }\n\n\t    fromLayer.connectedTo.push(this);\n\t  }\n\t}\n\n\t// types of connections\n\tLayer.connectionType = {};\n\tLayer.connectionType.ALL_TO_ALL = \"ALL TO ALL\";\n\tLayer.connectionType.ONE_TO_ONE = \"ONE TO ONE\";\n\tLayer.connectionType.ALL_TO_ELSE = \"ALL TO ELSE\";\n\n\t// types of gates\n\tLayer.gateType = {};\n\tLayer.gateType.INPUT = \"INPUT\";\n\tLayer.gateType.OUTPUT = \"OUTPUT\";\n\tLayer.gateType.ONE_TO_ONE = \"ONE TO ONE\";\n\n\t((() => {\n\t  let connections = 0;\n\t  Layer.connection.uid = () => connections++\n\t}))();\n\n\tmodule.exports = Layer;\n\n/***/ },\n/* 3 */\n/***/ function(module, exports) {\n\n\t/******************************************************************************************\n\t NEURON\n\t *******************************************************************************************/\n\n\tclass Neuron {\n\t  constructor() {\n\t    this.ID = Neuron.uid();\n\t    this.label = null;\n\t    this.connections = {\n\t      inputs: {},\n\t      projected: {},\n\t      gated: {}\n\t    };\n\t    this.error = {\n\t      responsibility: 0,\n\t      projected: 0,\n\t      gated: 0\n\t    };\n\t    this.trace = {\n\t      elegibility: {},\n\t      extended: {},\n\t      influences: {}\n\t    };\n\t    this.state = 0;\n\t    this.old = 0;\n\t    this.activation = 0;\n\t    this.selfconnection = new Neuron.connection(this, this, 0); // weight = 0 -> not connected\n\t    this.squash = Neuron.squash.LOGISTIC;\n\t    this.neighboors = {};\n\t    this.bias = Math.random() * .2 - .1;\n\t  }\n\n\t  // activate the neuron\n\t  activate(input) {\n\t    // activation from enviroment (for input neurons)\n\t    if (typeof input != 'undefined') {\n\t      this.activation = input;\n\t      this.derivative = 0;\n\t      this.bias = 0;\n\t      return this.activation;\n\t    }\n\n\t    // old state\n\t    this.old = this.state;\n\n\t    // eq. 15\n\t    this.state = this.selfconnection.gain * this.selfconnection.weight *\n\t      this.state + this.bias;\n\n\t    for (var i in this.connections.inputs) {\n\t      var input = this.connections.inputs[i];\n\t      this.state += input.from.activation * input.weight * input.gain;\n\t    }\n\n\t    // eq. 16\n\t    this.activation = this.squash(this.state);\n\n\t    // f'(s)\n\t    this.derivative = this.squash(this.state, true);\n\n\t    // update traces\n\t    const influences = [];\n\t    for (var id in this.trace.extended) {\n\t      // extended elegibility trace\n\t      var neuron = this.neighboors[id];\n\n\t      // if gated neuron's selfconnection is gated by this unit, the influence keeps track of the neuron's old state\n\t      var influence = neuron.selfconnection.gater == this ? neuron.old : 0;\n\n\t      // index runs over all the incoming connections to the gated neuron that are gated by this unit\n\t      for (let incoming in this.trace.influences[neuron.ID]) { // captures the effect that has an input connection to this unit, on a neuron that is gated by this unit\n\t        influence += this.trace.influences[neuron.ID][incoming].weight *\n\t          this.trace.influences[neuron.ID][incoming].from.activation;\n\t      }\n\t      influences[neuron.ID] = influence;\n\t    }\n\n\t    for (var i in this.connections.inputs) {\n\t      var input = this.connections.inputs[i];\n\n\t      // elegibility trace - Eq. 17\n\t      this.trace.elegibility[input.ID] = this.selfconnection.gain * this.selfconnection\n\t          .weight * this.trace.elegibility[input.ID] + input.gain * input.from\n\t          .activation;\n\n\t      for (var id in this.trace.extended) {\n\t        // extended elegibility trace\n\t        const xtrace = this.trace.extended[id];\n\t        var neuron = this.neighboors[id];\n\t        var influence = influences[neuron.ID];\n\n\t        // eq. 18\n\t        xtrace[input.ID] = neuron.selfconnection.gain * neuron.selfconnection\n\t            .weight * xtrace[input.ID] + this.derivative * this.trace.elegibility[\n\t            input.ID] * influence;\n\t      }\n\t    }\n\n\t    //  update gated connection's gains\n\t    for (let connection in this.connections.gated) {\n\t      this.connections.gated[connection].gain = this.activation;\n\t    }\n\n\t    return this.activation;\n\t  }\n\n\t  // back-propagate the error\n\t  propagate(rate, target) {\n\t    // error accumulator\n\t    let error = 0;\n\n\t    // whether or not this neuron is in the output layer\n\t    const isOutput = typeof target != 'undefined';\n\n\t    // output neurons get their error from the enviroment\n\t    if (isOutput)\n\t      this.error.responsibility = this.error.projected = target - this.activation; // Eq. 10\n\n\t    else // the rest of the neuron compute their error responsibilities by backpropagation\n\t    {\n\t      // error responsibilities from all the connections projected from this neuron\n\t      for (var id in this.connections.projected) {\n\t        const connection = this.connections.projected[id];\n\t        var neuron = connection.to;\n\t        // Eq. 21\n\t        error += neuron.error.responsibility * connection.gain * connection.weight;\n\t      }\n\n\t      // projected error responsibility\n\t      this.error.projected = this.derivative * error;\n\n\t      error = 0;\n\t      // error responsibilities from all the connections gated by this neuron\n\t      for (var id in this.trace.extended) {\n\t        var neuron = this.neighboors[id]; // gated neuron\n\t        let influence = neuron.selfconnection.gater == this ? neuron.old : 0; // if gated neuron's selfconnection is gated by this neuron\n\n\t        // index runs over all the connections to the gated neuron that are gated by this neuron\n\t        for (var input in this.trace.influences[id]) { // captures the effect that the input connection of this neuron have, on a neuron which its input/s is/are gated by this neuron\n\t          influence += this.trace.influences[id][input].weight * this.trace.influences[\n\t              neuron.ID][input].from.activation;\n\t        }\n\t        // eq. 22\n\t        error += neuron.error.responsibility * influence;\n\t      }\n\n\t      // gated error responsibility\n\t      this.error.gated = this.derivative * error;\n\n\t      // error responsibility - Eq. 23\n\t      this.error.responsibility = this.error.projected + this.error.gated;\n\t    }\n\n\t    // learning rate\n\t    rate = rate || .1;\n\n\t    // adjust all the neuron's incoming connections\n\t    for (var id in this.connections.inputs) {\n\t      var input = this.connections.inputs[id];\n\n\t      // Eq. 24\n\t      let gradient = this.error.projected * this.trace.elegibility[input.ID];\n\t      for (var id in this.trace.extended) {\n\t        var neuron = this.neighboors[id];\n\t        gradient += neuron.error.responsibility * this.trace.extended[\n\t            neuron.ID][input.ID];\n\t      }\n\t      input.weight += rate * gradient; // adjust weights - aka learn\n\t    }\n\n\t    // adjust bias\n\t    this.bias += rate * this.error.responsibility;\n\t  }\n\n\t  project(neuron, weight) {\n\t    // self-connection\n\t    if (neuron == this) {\n\t      this.selfconnection.weight = 1;\n\t      return this.selfconnection;\n\t    }\n\n\t    // check if connection already exists\n\t    const connected = this.connected(neuron);\n\t    if (connected && connected.type == \"projected\") {\n\t      // update connection\n\t      if (typeof weight != 'undefined')\n\t        connected.connection.weight = weight;\n\t      // return existing connection\n\t      return connected.connection;\n\t    } else {\n\t      // create a new connection\n\t      var connection = new Neuron.connection(this, neuron, weight);\n\t    }\n\n\t    // reference all the connections and traces\n\t    this.connections.projected[connection.ID] = connection;\n\t    this.neighboors[neuron.ID] = neuron;\n\t    neuron.connections.inputs[connection.ID] = connection;\n\t    neuron.trace.elegibility[connection.ID] = 0;\n\n\t    for (let id in neuron.trace.extended) {\n\t      const trace = neuron.trace.extended[id];\n\t      trace[connection.ID] = 0;\n\t    }\n\n\t    return connection;\n\t  }\n\n\t  gate(connection) {\n\t    // add connection to gated list\n\t    this.connections.gated[connection.ID] = connection;\n\n\t    const neuron = connection.to;\n\t    if (!(neuron.ID in this.trace.extended)) {\n\t      // extended trace\n\t      this.neighboors[neuron.ID] = neuron;\n\t      const xtrace = this.trace.extended[neuron.ID] = {};\n\t      for (let id in this.connections.inputs) {\n\t        const input = this.connections.inputs[id];\n\t        xtrace[input.ID] = 0;\n\t      }\n\t    }\n\n\t    // keep track\n\t    if (neuron.ID in this.trace.influences)\n\t      this.trace.influences[neuron.ID].push(connection);\n\t    else\n\t      this.trace.influences[neuron.ID] = [connection];\n\n\t    // set gater\n\t    connection.gater = this;\n\t  }\n\n\t  // returns true or false whether the neuron is self-connected or not\n\t  selfconnected() {\n\t    return this.selfconnection.weight !== 0;\n\t  }\n\n\t  // returns true or false whether the neuron is connected to another neuron (parameter)\n\t  connected(neuron) {\n\t    const result = {\n\t      type: null,\n\t      connection: false\n\t    };\n\n\t    if (this == neuron) {\n\t      if (this.selfconnected()) {\n\t        result.type = 'selfconnection';\n\t        result.connection = this.selfconnection;\n\t        return result;\n\t      } else\n\t        return false;\n\t    }\n\n\t    for (let type in this.connections) {\n\t      for (var connection in this.connections[type]) {\n\t        var connection = this.connections[type][connection];\n\t        if (connection.to == neuron) {\n\t          result.type = type;\n\t          result.connection = connection;\n\t          return result;\n\t        } else if (connection.from == neuron) {\n\t          result.type = type;\n\t          result.connection = connection;\n\t          return result;\n\t        }\n\t      }\n\t    }\n\n\t    return false;\n\t  }\n\n\t  // clears all the traces (the neuron forgets it's context, but the connections remain intact)\n\t  clear() {\n\n\t    for (var trace in this.trace.elegibility)\n\t      this.trace.elegibility[trace] = 0;\n\n\t    for (var trace in this.trace.extended)\n\t      for (let extended in this.trace.extended[trace])\n\t        this.trace.extended[trace][extended] = 0;\n\n\t    this.error.responsibility = this.error.projected = this.error.gated = 0;\n\t  }\n\n\t  // all the connections are randomized and the traces are cleared\n\t  reset() {\n\t    this.clear();\n\n\t    for (let type in this.connections)\n\t      for (let connection in this.connections[type])\n\t        this.connections[type][connection].weight = Math.random() * .2 - .1;\n\t    this.bias = Math.random() * .2 - .1;\n\n\t    this.old = this.state = this.activation = 0;\n\t  }\n\n\t  // hardcodes the behaviour of the neuron into an optimized function\n\t  optimize(optimized = {}, layer) {\n\t    const store_activation = [];\n\t    const store_trace = [];\n\t    const store_propagation = [];\n\t    let varID = optimized.memory || 0;\n\t    const neurons = optimized.neurons || 1;\n\t    const inputs = optimized.inputs || [];\n\t    const targets = optimized.targets || [];\n\t    const outputs = optimized.outputs || [];\n\t    const variables = optimized.variables || {};\n\t    const activation_sentences = optimized.activation_sentences || [];\n\t    const trace_sentences = optimized.trace_sentences || [];\n\t    const propagation_sentences = optimized.propagation_sentences || [];\n\t    const layers = optimized.layers || {__count: 0, __neuron: 0};\n\n\t    // allocate sentences\n\t    const allocate = store => {\n\t      const allocated = layer in layers && store[layers.__count];\n\t      if (!allocated) {\n\t        layers.__count = store.push([]) - 1;\n\t        layers[layer] = layers.__count;\n\t      }\n\t    };\n\t    allocate(activation_sentences);\n\t    allocate(trace_sentences);\n\t    allocate(propagation_sentences);\n\t    const currentLayer = layers.__count;\n\n\t    // get/reserve space in memory by creating a unique ID for a variablel\n\t    const getVar = function () {\n\t      const args = Array.prototype.slice.call(arguments);\n\n\t      if (args.length == 1) {\n\t        if (args[0] == 'target') {\n\t          var id = `target_${targets.length}`;\n\t          targets.push(varID);\n\t        } else\n\t          var id = args[0];\n\t        if (id in variables)\n\t          return variables[id];\n\t        return variables[id] = {\n\t          value: 0,\n\t          id: varID++\n\t        };\n\t      } else {\n\t        const extended = args.length > 2;\n\t        if (extended)\n\t          var value = args.pop();\n\n\t        const unit = args.shift();\n\t        const prop = args.pop();\n\n\t        if (!extended)\n\t          var value = unit[prop];\n\n\t        var id = `${prop}_`;\n\t        for (let property in args)\n\t          id += `${args[property]}_`;\n\t        id += unit.ID;\n\t        if (id in variables)\n\t          return variables[id];\n\n\t        return variables[id] = {\n\t          value,\n\t          id: varID++\n\t        };\n\t      }\n\t    };\n\n\t    // build sentence\n\t    const buildSentence = function () {\n\t      const args = Array.prototype.slice.call(arguments);\n\t      const store = args.pop();\n\t      let sentence = \"\";\n\t      for (let i in args)\n\t        if (typeof args[i] == 'string')\n\t          sentence += args[i];\n\t        else\n\t          sentence += `F[${args[i].id}]`;\n\n\t      store.push(`${sentence};`);\n\t    };\n\n\t    // helper to check if an object is empty\n\t    const isEmpty = obj => {\n\t      for (let prop in obj) {\n\t        if (obj.hasOwnProperty(prop))\n\t          return false;\n\t      }\n\t      return true;\n\t    };\n\n\t    // characteristics of the neuron\n\t    const noProjections = isEmpty(this.connections.projected);\n\t    const noGates = isEmpty(this.connections.gated);\n\t    const isInput = layer == 'input' ? true : isEmpty(this.connections.inputs);\n\t    const isOutput = layer == 'output' ? true : noProjections && noGates;\n\n\t    // optimize neuron's behaviour\n\t    const rate = getVar('rate');\n\t    const activation = getVar(this, 'activation');\n\t    if (isInput)\n\t      inputs.push(activation.id);\n\t    else {\n\t      activation_sentences[currentLayer].push(store_activation);\n\t      trace_sentences[currentLayer].push(store_trace);\n\t      propagation_sentences[currentLayer].push(store_propagation);\n\t      const old = getVar(this, 'old');\n\t      const state = getVar(this, 'state');\n\t      var bias = getVar(this, 'bias');\n\t      if (this.selfconnection.gater)\n\t        var self_gain = getVar(this.selfconnection, 'gain');\n\t      if (this.selfconnected())\n\t        var self_weight = getVar(this.selfconnection, 'weight');\n\t      buildSentence(old, ' = ', state, store_activation);\n\t      if (this.selfconnected())\n\t        if (this.selfconnection.gater)\n\t          buildSentence(state, ' = ', self_gain, ' * ', self_weight, ' * ',\n\t            state, ' + ', bias, store_activation);\n\t        else\n\t          buildSentence(state, ' = ', self_weight, ' * ', state, ' + ',\n\t            bias, store_activation);\n\t      else\n\t        buildSentence(state, ' = ', bias, store_activation);\n\t      for (var i in this.connections.inputs) {\n\t        var input = this.connections.inputs[i];\n\t        var input_activation = getVar(input.from, 'activation');\n\t        var input_weight = getVar(input, 'weight');\n\t        if (input.gater)\n\t          var input_gain = getVar(input, 'gain');\n\t        if (this.connections.inputs[i].gater)\n\t          buildSentence(state, ' += ', input_activation, ' * ',\n\t            input_weight, ' * ', input_gain, store_activation);\n\t        else\n\t          buildSentence(state, ' += ', input_activation, ' * ',\n\t            input_weight, store_activation);\n\t      }\n\t      var derivative = getVar(this, 'derivative');\n\t      switch (this.squash) {\n\t        case Neuron.squash.LOGISTIC:\n\t          buildSentence(activation, ' = (1 / (1 + Math.exp(-', state, ')))',\n\t            store_activation);\n\t          buildSentence(derivative, ' = ', activation, ' * (1 - ',\n\t            activation, ')', store_activation);\n\t          break;\n\t        case Neuron.squash.TANH:\n\t          const eP = getVar('aux');\n\t          const eN = getVar('aux_2');\n\t          buildSentence(eP, ' = Math.exp(', state, ')', store_activation);\n\t          buildSentence(eN, ' = 1 / ', eP, store_activation);\n\t          buildSentence(activation, ' = (', eP, ' - ', eN, ') / (', eP, ' + ', eN, ')', store_activation);\n\t          buildSentence(derivative, ' = 1 - (', activation, ' * ', activation, ')', store_activation);\n\t          break;\n\t        case Neuron.squash.IDENTITY:\n\t          buildSentence(activation, ' = ', state, store_activation);\n\t          buildSentence(derivative, ' = 1', store_activation);\n\t          break;\n\t        case Neuron.squash.HLIM:\n\t          buildSentence(activation, ' = +(', state, ' > 0)', store_activation);\n\t          buildSentence(derivative, ' = 1', store_activation);\n\t        case Neuron.squash.RELU:\n\t          buildSentence(activation, ' = ', state, ' > 0 ? ', state, ' : 0', store_activation);\n\t          buildSentence(derivative, ' = ', state, ' > 0 ? 1 : 0', store_activation);\n\t          break;\n\t      }\n\n\t      for (var id in this.trace.extended) {\n\t        // calculate extended elegibility traces in advance\n\n\t        var neuron = this.neighboors[id];\n\t        var influence = getVar(`influences[${neuron.ID}]`);\n\t        var neuron_old = getVar(neuron, 'old');\n\t        let initialized = false;\n\t        if (neuron.selfconnection.gater == this) {\n\t          buildSentence(influence, ' = ', neuron_old, store_trace);\n\t          initialized = true;\n\t        }\n\t        for (let incoming in this.trace.influences[neuron.ID]) {\n\t          const incoming_weight = getVar(this.trace.influences[neuron.ID]\n\t            [incoming], 'weight');\n\t          const incoming_activation = getVar(this.trace.influences[neuron.ID]\n\t            [incoming].from, 'activation');\n\n\t          if (initialized)\n\t            buildSentence(influence, ' += ', incoming_weight, ' * ', incoming_activation, store_trace);\n\t          else {\n\t            buildSentence(influence, ' = ', incoming_weight, ' * ', incoming_activation, store_trace);\n\t            initialized = true;\n\t          }\n\t        }\n\t      }\n\n\t      for (var i in this.connections.inputs) {\n\t        var input = this.connections.inputs[i];\n\t        if (input.gater)\n\t          var input_gain = getVar(input, 'gain');\n\t        var input_activation = getVar(input.from, 'activation');\n\t        var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace\n\t          .elegibility[input.ID]);\n\t        if (this.selfconnected()) {\n\t          if (this.selfconnection.gater) {\n\t            if (input.gater)\n\t              buildSentence(trace, ' = ', self_gain, ' * ', self_weight,\n\t                ' * ', trace, ' + ', input_gain, ' * ', input_activation,\n\t                store_trace);\n\t            else\n\t              buildSentence(trace, ' = ', self_gain, ' * ', self_weight,\n\t                ' * ', trace, ' + ', input_activation, store_trace);\n\t          } else {\n\t            if (input.gater)\n\t              buildSentence(trace, ' = ', self_weight, ' * ', trace, ' + ',\n\t                input_gain, ' * ', input_activation, store_trace);\n\t            else\n\t              buildSentence(trace, ' = ', self_weight, ' * ', trace, ' + ',\n\t                input_activation, store_trace);\n\t          }\n\t        } else {\n\t          if (input.gater)\n\t            buildSentence(trace, ' = ', input_gain, ' * ', input_activation,\n\t              store_trace);\n\t          else\n\t            buildSentence(trace, ' = ', input_activation, store_trace);\n\t        }\n\t        for (var id in this.trace.extended) {\n\t          // extended elegibility trace\n\t          var neuron = this.neighboors[id];\n\t          var influence = getVar(`influences[${neuron.ID}]`);\n\n\t          var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace\n\t            .elegibility[input.ID]);\n\t          var xtrace = getVar(this, 'trace', 'extended', neuron.ID, input.ID,\n\t            this.trace.extended[neuron.ID][input.ID]);\n\t          if (neuron.selfconnected())\n\t            var neuron_self_weight = getVar(neuron.selfconnection, 'weight');\n\t          if (neuron.selfconnection.gater)\n\t            var neuron_self_gain = getVar(neuron.selfconnection, 'gain');\n\t          if (neuron.selfconnected())\n\t            if (neuron.selfconnection.gater)\n\t              buildSentence(xtrace, ' = ', neuron_self_gain, ' * ',\n\t                neuron_self_weight, ' * ', xtrace, ' + ', derivative, ' * ',\n\t                trace, ' * ', influence, store_trace);\n\t            else\n\t              buildSentence(xtrace, ' = ', neuron_self_weight, ' * ',\n\t                xtrace, ' + ', derivative, ' * ', trace, ' * ',\n\t                influence, store_trace);\n\t          else\n\t            buildSentence(xtrace, ' = ', derivative, ' * ', trace, ' * ',\n\t              influence, store_trace);\n\t        }\n\t      }\n\t      for (var connection in this.connections.gated) {\n\t        const gated_gain = getVar(this.connections.gated[connection], 'gain');\n\t        buildSentence(gated_gain, ' = ', activation, store_activation);\n\t      }\n\t    }\n\t    if (!isInput) {\n\t      const responsibility = getVar(this, 'error', 'responsibility', this.error\n\t        .responsibility);\n\t      if (isOutput) {\n\t        const target = getVar('target');\n\t        buildSentence(responsibility, ' = ', target, ' - ', activation,\n\t          store_propagation);\n\t        for (var id in this.connections.inputs) {\n\t          var input = this.connections.inputs[id];\n\t          var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace\n\t            .elegibility[input.ID]);\n\t          var input_weight = getVar(input, 'weight');\n\t          buildSentence(input_weight, ' += ', rate, ' * (', responsibility,\n\t            ' * ', trace, ')', store_propagation);\n\t        }\n\t        outputs.push(activation.id);\n\t      } else {\n\t        if (!noProjections && !noGates) {\n\t          const error = getVar('aux');\n\t          for (var id in this.connections.projected) {\n\t            var connection = this.connections.projected[id];\n\t            var neuron = connection.to;\n\t            var connection_weight = getVar(connection, 'weight');\n\t            var neuron_responsibility = getVar(neuron, 'error',\n\t              'responsibility', neuron.error.responsibility);\n\t            if (connection.gater) {\n\t              var connection_gain = getVar(connection, 'gain');\n\t              buildSentence(error, ' += ', neuron_responsibility, ' * ',\n\t                connection_gain, ' * ', connection_weight,\n\t                store_propagation);\n\t            } else\n\t              buildSentence(error, ' += ', neuron_responsibility, ' * ',\n\t                connection_weight, store_propagation);\n\t          }\n\t          const projected = getVar(this, 'error', 'projected', this.error.projected);\n\t          buildSentence(projected, ' = ', derivative, ' * ', error,\n\t            store_propagation);\n\t          buildSentence(error, ' = 0', store_propagation);\n\t          for (var id in this.trace.extended) {\n\t            var neuron = this.neighboors[id];\n\t            var influence = getVar('aux_2');\n\t            var neuron_old = getVar(neuron, 'old');\n\t            if (neuron.selfconnection.gater == this)\n\t              buildSentence(influence, ' = ', neuron_old, store_propagation);\n\t            else\n\t              buildSentence(influence, ' = 0', store_propagation);\n\t            for (var input in this.trace.influences[neuron.ID]) {\n\t              var connection = this.trace.influences[neuron.ID][input];\n\t              var connection_weight = getVar(connection, 'weight');\n\t              var neuron_activation = getVar(connection.from, 'activation');\n\t              buildSentence(influence, ' += ', connection_weight, ' * ',\n\t                neuron_activation, store_propagation);\n\t            }\n\t            var neuron_responsibility = getVar(neuron, 'error',\n\t              'responsibility', neuron.error.responsibility);\n\t            buildSentence(error, ' += ', neuron_responsibility, ' * ',\n\t              influence, store_propagation);\n\t          }\n\t          const gated = getVar(this, 'error', 'gated', this.error.gated);\n\t          buildSentence(gated, ' = ', derivative, ' * ', error,\n\t            store_propagation);\n\t          buildSentence(responsibility, ' = ', projected, ' + ', gated,\n\t            store_propagation);\n\t          for (var id in this.connections.inputs) {\n\t            var input = this.connections.inputs[id];\n\t            var gradient = getVar('aux');\n\t            var trace = getVar(this, 'trace', 'elegibility', input.ID, this\n\t              .trace.elegibility[input.ID]);\n\t            buildSentence(gradient, ' = ', projected, ' * ', trace,\n\t              store_propagation);\n\t            for (var id in this.trace.extended) {\n\t              var neuron = this.neighboors[id];\n\t              var neuron_responsibility = getVar(neuron, 'error',\n\t                'responsibility', neuron.error.responsibility);\n\t              var xtrace = getVar(this, 'trace', 'extended', neuron.ID,\n\t                input.ID, this.trace.extended[neuron.ID][input.ID]);\n\t              buildSentence(gradient, ' += ', neuron_responsibility, ' * ',\n\t                xtrace, store_propagation);\n\t            }\n\t            var input_weight = getVar(input, 'weight');\n\t            buildSentence(input_weight, ' += ', rate, ' * ', gradient,\n\t              store_propagation);\n\t          }\n\n\t        } else if (noGates) {\n\t          buildSentence(responsibility, ' = 0', store_propagation);\n\t          for (var id in this.connections.projected) {\n\t            var connection = this.connections.projected[id];\n\t            var neuron = connection.to;\n\t            var connection_weight = getVar(connection, 'weight');\n\t            var neuron_responsibility = getVar(neuron, 'error',\n\t              'responsibility', neuron.error.responsibility);\n\t            if (connection.gater) {\n\t              var connection_gain = getVar(connection, 'gain');\n\t              buildSentence(responsibility, ' += ', neuron_responsibility,\n\t                ' * ', connection_gain, ' * ', connection_weight,\n\t                store_propagation);\n\t            } else\n\t              buildSentence(responsibility, ' += ', neuron_responsibility,\n\t                ' * ', connection_weight, store_propagation);\n\t          }\n\t          buildSentence(responsibility, ' *= ', derivative,\n\t            store_propagation);\n\t          for (var id in this.connections.inputs) {\n\t            var input = this.connections.inputs[id];\n\t            var trace = getVar(this, 'trace', 'elegibility', input.ID, this\n\t              .trace.elegibility[input.ID]);\n\t            var input_weight = getVar(input, 'weight');\n\t            buildSentence(input_weight, ' += ', rate, ' * (',\n\t              responsibility, ' * ', trace, ')', store_propagation);\n\t          }\n\t        } else if (noProjections) {\n\t          buildSentence(responsibility, ' = 0', store_propagation);\n\t          for (var id in this.trace.extended) {\n\t            var neuron = this.neighboors[id];\n\t            var influence = getVar('aux');\n\t            var neuron_old = getVar(neuron, 'old');\n\t            if (neuron.selfconnection.gater == this)\n\t              buildSentence(influence, ' = ', neuron_old, store_propagation);\n\t            else\n\t              buildSentence(influence, ' = 0', store_propagation);\n\t            for (var input in this.trace.influences[neuron.ID]) {\n\t              var connection = this.trace.influences[neuron.ID][input];\n\t              var connection_weight = getVar(connection, 'weight');\n\t              var neuron_activation = getVar(connection.from, 'activation');\n\t              buildSentence(influence, ' += ', connection_weight, ' * ',\n\t                neuron_activation, store_propagation);\n\t            }\n\t            var neuron_responsibility = getVar(neuron, 'error',\n\t              'responsibility', neuron.error.responsibility);\n\t            buildSentence(responsibility, ' += ', neuron_responsibility,\n\t              ' * ', influence, store_propagation);\n\t          }\n\t          buildSentence(responsibility, ' *= ', derivative,\n\t            store_propagation);\n\t          for (var id in this.connections.inputs) {\n\t            var input = this.connections.inputs[id];\n\t            var gradient = getVar('aux');\n\t            buildSentence(gradient, ' = 0', store_propagation);\n\t            for (var id in this.trace.extended) {\n\t              var neuron = this.neighboors[id];\n\t              var neuron_responsibility = getVar(neuron, 'error',\n\t                'responsibility', neuron.error.responsibility);\n\t              var xtrace = getVar(this, 'trace', 'extended', neuron.ID,\n\t                input.ID, this.trace.extended[neuron.ID][input.ID]);\n\t              buildSentence(gradient, ' += ', neuron_responsibility, ' * ',\n\t                xtrace, store_propagation);\n\t            }\n\t            var input_weight = getVar(input, 'weight');\n\t            buildSentence(input_weight, ' += ', rate, ' * ', gradient,\n\t              store_propagation);\n\t          }\n\t        }\n\t      }\n\t      buildSentence(bias, ' += ', rate, ' * ', responsibility,\n\t        store_propagation);\n\t    }\n\t    return {\n\t      memory: varID,\n\t      neurons: neurons + 1,\n\t      inputs,\n\t      outputs,\n\t      targets,\n\t      variables,\n\t      activation_sentences,\n\t      trace_sentences,\n\t      propagation_sentences,\n\t      layers\n\t    }\n\t  }\n\t}\n\t// represents a connection between two neurons\n\tNeuron.connection = class NeuronConnection {\n\t  constructor(from, to, weight) {\n\n\t    if (!from || !to)\n\t      throw new Error(\"Connection Error: Invalid neurons\");\n\n\t    this.ID = Neuron.connection.uid();\n\t    this.from = from;\n\t    this.to = to;\n\t    this.weight = typeof weight == 'undefined' ? Math.random() * .2 - .1 :\n\t      weight;\n\t    this.gain = 1;\n\t    this.gater = null;\n\t  }\n\t}\n\n\n\t// squashing functions\n\tNeuron.squash = {};\n\n\tfunction registerSquash(name, fn) {\n\t  Neuron.squash[name] = fn;\n\t  fn.squashType = [name];\n\t}\n\n\t// eq. 5 & 5'\n\tregisterSquash('LOGISTIC', (x, derivate) => {\n\t  if (!derivate)\n\t    return 1 / (1 + Math.exp(-x));\n\t  const fx = Neuron.squash.LOGISTIC(x);\n\t  return fx * (1 - fx);\n\t});\n\tregisterSquash('TANH', (x, derivate) => {\n\t  if (derivate)\n\t    return 1 - Math.pow(Neuron.squash.TANH(x), 2);\n\t  const eP = Math.exp(x);\n\t  const eN = 1 / eP;\n\t  return (eP - eN) / (eP + eN);\n\t});\n\tregisterSquash('IDENTITY', (x, derivate) => derivate ? 1 : x);\n\tregisterSquash('HLIM', (x, derivate) => derivate ? 1 : x > 0 ? 1 : 0);\n\tregisterSquash('RELU', (x, derivate) => {\n\t  if (derivate)\n\t    return x > 0 ? 1 : 0;\n\t  return x > 0 ? x : 0;\n\t});\n\n\t// unique ID's\n\t((() => {\n\t  let neurons = 0;\n\t  let connections = 0;\n\t  Neuron.uid = () => neurons++\n\t  Neuron.connection.uid = () => connections++\n\t  Neuron.quantity = () => ({\n\t    neurons,\n\t    connections\n\t  })\n\t}))();\n\n\tmodule.exports = Neuron;\n\n\n/***/ },\n/* 4 */\n/***/ function(module, exports) {\n\n\tmodule.exports.GET = '%get';\n\tmodule.exports.SET = '%set';\n\tmodule.exports.FROM_JSON = '%fromJSON';\n\n/***/ }\n/******/ ]);", __webpack_require__.p + "c275304e08386603d60b.worker.js");
	};

/***/ },
/* 10 */
/***/ function(module, exports) {

	// http://stackoverflow.com/questions/10343913/how-to-create-a-web-worker-from-a-string

	var URL = window.URL || window.webkitURL;
	module.exports = function(content, url) {
		try {
			try {
				var blob;
				try { // BlobBuilder = Deprecated, but widely implemented
					var BlobBuilder = window.BlobBuilder || window.WebKitBlobBuilder || window.MozBlobBuilder || window.MSBlobBuilder;
					blob = new BlobBuilder();
					blob.append(content);
					blob = blob.getBlob();
				} catch(e) { // The proposed API
					blob = new Blob([content]);
				}
				return new Worker(URL.createObjectURL(blob));
			} catch(e) {
				return new Worker('data:application/javascript,' + encodeURIComponent(content));
			}
		} catch(e) {
			return new Worker(url);
		}
	}

/***/ },
/* 11 */
/***/ function(module, exports) {

	/*******************************************************************************************
	                                        TRAINER
	*******************************************************************************************/

	class Trainer {
	  constructor(network, options={}) {
	    this.network = network;
	    this.rate = options.rate || .2;
	    this.iterations = options.iterations || 100000;
	    this.error = options.error || .005;
	    this.cost = options.cost || null;
	    this.crossValidate = options.crossValidate || null;
	  }

	  // trains any given set to a network
	  train(set, options) {

	    let error = 1;
	    let iterations = bucketSize = 0;
	    let abort = false;
	    let currentRate;
	    const cost = options && options.cost || this.cost || Trainer.cost.MSE;
	    let crossValidate = false, testSet, trainSet;

	    const start = Date.now();

	    if (options) {
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
	      if (this.crossValidate || options.crossValidate) {
	        if(!this.crossValidate) this.crossValidate = {};
	        crossValidate = true;
	        if (options.crossValidate.testSize)
	          this.crossValidate.testSize = options.crossValidate.testSize;
	        if (options.crossValidate.testError)
	          this.crossValidate.testError = options.crossValidate.testError;
	      }
	    }

	    currentRate = this.rate;
	    if(Array.isArray(this.rate)) {
	      var bucketSize = Math.floor(this.iterations / this.rate.length);
	    }

	    if(crossValidate) {
	      const numTrain = Math.ceil((1 - this.crossValidate.testSize) * set.length);
	      trainSet = set.slice(0, numTrain);
	      testSet = set.slice(numTrain);
	    }

	    var lastError = 0;
	    while ((!abort && iterations < this.iterations && error > this.error)) {
	      if (crossValidate && error <= this.crossValidate.testError) {
	        break;
	      }

	      let currentSetSize = set.length;
	      error = 0;
	      iterations++;

	      if(bucketSize > 0) {
	        const currentBucket = Math.floor(iterations / bucketSize);
	        currentRate = this.rate[currentBucket] || currentRate;
	      }
	      
	      if(typeof this.rate === 'function') {
	        currentRate = this.rate(iterations, lastError);
	      }

	      if (crossValidate) {
	        this._trainSet(trainSet, currentRate, cost);
	        error += this.test(testSet).error;
	        currentSetSize = 1;
	      } else {
	        error += this._trainSet(set, currentRate, cost);
	        currentSetSize = set.length;
	      }

	      // check error
	      error /= currentSetSize;
	      lastError = error;

	      if (options) {
	        if (this.schedule && this.schedule.every && iterations %
	          this.schedule.every == 0)
	          abort = this.schedule.do({ error, iterations, rate: currentRate });
	        else if (options.log && iterations % options.log == 0) {
	          console.log('iterations', iterations, 'error', error, 'rate', currentRate);
	        };
	        if (options.shuffle)
	          shuffle(set);
	      }
	    }

	    const results = {
	      error,
	      iterations,
	      time: Date.now() - start
	    };

	    return results;
	  }

	  // trains any given set to a network, using a WebWorker (only for the browser). Returns a Promise of the results.
	  trainAsync(set, options) {
	    const train = this.workerTrain.bind(this);
	    return new Promise((resolve, reject) => {
	      try {
	        train(set, resolve, options, true)
	      } catch(e) {
	        reject(e)
	      }
	    })
	  }

	  // preforms one training epoch and returns the error (private function used in this.train)
	  _trainSet(set, currentRate, costFunction) {
	    let errorSum = 0;
	    for (let train in set) {
	      const input = set[train].input;
	      const target = set[train].output;

	      const output = this.network.activate(input);
	      this.network.propagate(currentRate, target);

	      errorSum += costFunction(target, output);
	    }
	    return errorSum;
	  }

	  // tests a set and returns the error and elapsed time
	  test(set, options) {

	    let error = 0;
	    let input, output, target;
	    const cost = options && options.cost || this.cost || Trainer.cost.MSE;

	    const start = Date.now();

	    for (let test in set) {
	      input = set[test].input;
	      target = set[test].output;
	      output = this.network.activate(input);
	      error += cost(target, output);
	    }

	    error /= set.length;

	    const results = {
	      error,
	      time: Date.now() - start
	    };

	    return results;
	  }

	  // trains any given set to a network using a WebWorker [deprecated: use trainAsync instead]
	  workerTrain(set, callback, options, suppressWarning) {

	    if (!suppressWarning) {
	      console.warn('Deprecated: do not use `workerTrain`, use `trainAsync` instead.')
	    }
	    const that = this;

	    if (!this.network.optimized)
	      this.network.optimize();

	    // Create a new worker
	    const worker = this.network.worker(this.network.optimized.memory, set, options);

	    // train the worker
	    worker.onmessage = e => {
	      switch(e.data.action) {
	          case 'done':
	            const iterations = e.data.message.iterations;
	            const error = e.data.message.error;
	            const time = e.data.message.time;

	            that.network.optimized.ownership(e.data.memoryBuffer);

	            // Done callback
	            callback({
	              error,
	              iterations,
	              time
	            });

	            // Delete the worker and all its associated memory
	            worker.terminate();
	          break;

	          case 'log':
	            console.log(e.data.message);

	          case 'schedule':
	            if (options && options.schedule && typeof options.schedule.do === 'function') {
	              const scheduled = options.schedule.do;
	              scheduled(e.data.message)
	            }
	          break;
	      }
	    };

	    // Start the worker
	    worker.postMessage({action: 'startTraining'});
	  }

	  // trains an XOR to the network
	  XOR(options) {

	    if (this.network.inputs() != 2 || this.network.outputs() != 1)
	      throw new Error("Incompatible network (2 inputs, 1 output)");

	    const defaults = {
	      iterations: 100000,
	      log: false,
	      shuffle: true,
	      cost: Trainer.cost.MSE
	    };

	    if (options)
	      for (let i in options)
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
	  }

	  // trains the network to pass a Distracted Sequence Recall test
	  DSR(options={}) {
	    const targets = options.targets || [2, 4, 7, 8];
	    const distractors = options.distractors || [3, 5, 6, 9];
	    const prompts = options.prompts || [0, 1];
	    const length = options.length || 24;
	    const criterion = options.success || 0.95;
	    const iterations = options.iterations || 100000;
	    const rate = options.rate || .1;
	    const log = options.log || 0;
	    const schedule = options.schedule || {};
	    const cost = options.cost || this.cost || Trainer.cost.CROSS_ENTROPY;

	    let trial, correct, i, j, success;
	    trial = correct = i = j = success = 0;
	    let error = 1;
	    const symbols = targets.length + distractors.length + prompts.length;

	    const noRepeat = (range, avoid) => {
	      const number = Math.random() * range | 0;
	      let used = false;
	      for (let i in avoid)
	        if (number == avoid[i])
	          used = true;
	      return used ? noRepeat(range, avoid) : number;
	    };

	    const equal = (prediction, output) => {
	      for (let i in prediction)
	        if (Math.round(prediction[i]) != output[i])
	          return false;
	      return true;
	    };

	    const start = Date.now();

	    while (trial < iterations && (success < criterion || trial % 1000 != 0)) {
	      // generate sequence
	      const sequence = [], sequenceLength = length - prompts.length;
	      for (i = 0; i < sequenceLength; i++) {
	        const any = Math.random() * distractors.length | 0;
	        sequence.push(distractors[any]);
	      }
	      const indexes = [];
	      let positions = [];
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
	      let distractorsCorrect;
	      let targetsCorrect = distractorsCorrect = 0;
	      error = 0;
	      for (i = 0; i < length; i++) {
	        // generate input from sequence
	        const input = [];
	        for (j = 0; j < symbols; j++)
	          input[j] = 0;
	        input[sequence[i]] = 1;

	        // generate target output
	        const output = [];
	        for (j = 0; j < targets.length; j++)
	          output[j] = 0;

	        if (i >= sequenceLength) {
	          const index = i - sequenceLength;
	          output[indexes[index]] = 1;
	        }

	        // check result
	        const prediction = this.network.activate(input);

	        if (equal(prediction, output))
	          if (i < sequenceLength)
	            distractorsCorrect++;
	          else
	            targetsCorrect++;
	        else {
	          this.network.propagate(rate, output);
	        }

	        error += cost(output, prediction);

	        if (distractorsCorrect + targetsCorrect == length)
	          correct++;
	      }

	      // calculate error
	      if (trial % 1000 == 0)
	        correct = 0;
	      trial++;
	      let divideError = trial % 1000;
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
	          success,
	          error,
	          time: Date.now() - start,
	          correct
	        });
	    }

	    return {
	      iterations: trial,
	      success,
	      error,
	      time: Date.now() - start
	    }
	  }

	  // train the network to learn an Embeded Reber Grammar
	  ERG(options={}) {
	    const iterations = options.iterations || 150000;
	    const criterion = options.error || .05;
	    const rate = options.rate || .1;
	    const log = options.log || 500;
	    const cost = options.cost || this.cost || Trainer.cost.CROSS_ENTROPY;

	    // gramar node
	    class Node {
	      constructor() {
	        this.paths = [];
	      }

	      connect(node, value) {
	        this.paths.push({
	          node,
	          value
	        });
	        return this;
	      }

	      any() {
	        if (this.paths.length == 0)
	          return false;
	        const index = Math.random() * this.paths.length | 0;
	        return this.paths[index];
	      }

	      test(value) {
	        for (let i in this.paths)
	          if (this.paths[i].value == value)
	            return this.paths[i];
	        return false;
	      }
	    }

	    const reberGrammar = () => {

	      // build a reber grammar
	      const output = new Node();
	      const n1 = (new Node()).connect(output, "E");
	      const n2 = (new Node()).connect(n1, "S");
	      const n3 = (new Node()).connect(n1, "V").connect(n2, "P");
	      const n4 = (new Node()).connect(n2, "X");
	      n4.connect(n4, "S");
	      const n5 = (new Node()).connect(n3, "V");
	      n5.connect(n5, "T");
	      n2.connect(n5, "X");
	      const n6 = (new Node()).connect(n4, "T").connect(n5, "P");
	      const input = (new Node()).connect(n6, "B");

	      return {
	        input,
	        output
	      }
	    };

	    // build an embeded reber grammar
	    const embededReberGrammar = () => {
	      const reber1 = reberGrammar();
	      const reber2 = reberGrammar();

	      const output = new Node();
	      const n1 = (new Node).connect(output, "E");
	      reber1.output.connect(n1, "T");
	      reber2.output.connect(n1, "P");
	      const n2 = (new Node).connect(reber1.input, "P").connect(reber2.input,
	        "T");
	      const input = (new Node).connect(n2, "B");

	      return {
	        input,
	        output
	      }

	    };

	    // generate an ERG sequence
	    const generate = () => {
	      const node = embededReberGrammar().input;
	      let next = node.any();
	      let str = "";
	      while (next) {
	        str += next.value;
	        next = next.node.any();
	      }
	      return str;
	    };

	    // test if a string matches an embeded reber grammar
	    const test = str => {
	      let node = embededReberGrammar().input;
	      let i = 0;
	      let ch = str.charAt(i);
	      while (i < str.length) {
	        const next = node.test(ch);
	        if (!next)
	          return false;
	        node = next.node;
	        ch = str.charAt(++i);
	      }
	      return true;
	    };

	    // helper to check if the output and the target vectors match
	    const different = (array1, array2) => {
	      let max1 = 0;
	      let i1 = -1;
	      let max2 = 0;
	      let i2 = -1;
	      for (let i in array1) {
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

	    let iteration = 0;
	    let error = 1;
	    const table = {
	      "B": 0,
	      "P": 1,
	      "T": 2,
	      "X": 3,
	      "S": 4,
	      "E": 5
	    };

	    const start = Date.now();
	    while (iteration < iterations && error > criterion) {
	      let i = 0;
	      error = 0;

	      // ERG sequence to learn
	      const sequence = generate();

	      // input
	      let read = sequence.charAt(i);
	      // target
	      let predict = sequence.charAt(i + 1);

	      // train
	      while (i < sequence.length - 1) {
	        const input = [];
	        const target = [];
	        for (let j = 0; j < 6; j++) {
	          input[j] = 0;
	          target[j] = 0;
	        }
	        input[table[read]] = 1;
	        target[table[predict]] = 1;

	        const output = this.network.activate(input);

	        if (different(output, target))
	          this.network.propagate(rate, target);

	        read = sequence.charAt(++i);
	        predict = sequence.charAt(i + 1);

	        error += cost(target, output);
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
	      error,
	      time: Date.now() - start,
	      test,
	      generate
	    }
	  }

	  timingTask(options) {

	    if (this.network.inputs() != 2 || this.network.outputs() != 1)
	      throw new Error("Invalid Network: must have 2 inputs and one output");

	    if (typeof options == 'undefined')
	      options = {};

	    // helper
	    function getSamples (trainingSize, testSize){

	      // sample size
	      const size = trainingSize + testSize;

	      // generate samples
	      let t = 0;
	      const set = [];
	      for (let i = 0; i < size; i++) {
	        set.push({ input: [0,0], output: [0] });
	      }
	      while(t < size - 20) {
	          let n = Math.round(Math.random() * 20);
	          set[t].input[0] = 1;
	          for (let j = t; j <= t + n; j++){
	              set[j].input[1] = n / 20;
	              set[j].output[0] = 0.5;
	          }
	          t += n;
	          n = Math.round(Math.random() * 20);
	          for (let k = t+1; k <= (t + n) &&  k < size; k++)
	              set[k].input[1] = set[t].input[1];
	          t += n;
	      }

	      // separate samples between train and test sets
	      const trainingSet = []; const testSet = [];
	      for (let l = 0; l < size; l++)
	          (l < trainingSize ? trainingSet : testSet).push(set[l]);

	      // return samples
	      return {
	          train: trainingSet,
	          test: testSet
	      }
	    }

	    const iterations = options.iterations || 200;
	    const error = options.error || .005;
	    const rate = options.rate || [.03, .02];
	    const log = options.log === false ? false : options.log || 10;
	    const cost = options.cost || this.cost || Trainer.cost.MSE;
	    const trainingSamples = options.trainSamples || 7000;
	    const testSamples = options.trainSamples || 1000;

	    // samples for training and testing
	    const samples = getSamples(trainingSamples, testSamples);

	    // train
	    const result = this.train(samples.train, {
	      rate,
	      log,
	      iterations,
	      error,
	      cost
	    });

	    return {
	      train: result,
	      test: this.test(samples.test)
	    }
	  }
	}

	// Built-in cost functions
	Trainer.cost = {
	  // Eq. 9
	  CROSS_ENTROPY(target, output) {
	    let crossentropy = 0;
	    for (let i in output)
	      crossentropy -= (target[i] * Math.log(output[i]+1e-15)) + ((1-target[i]) * Math.log((1+1e-15)-output[i])); // +1e-15 is a tiny push away to avoid Math.log(0)
	    return crossentropy;
	  },
	  MSE(target, output) {
	    let mse = 0;
	    for (let i in output)
	      mse += Math.pow(target[i] - output[i], 2);
	    return mse / output.length;
	  },
	  BINARY(target, output) {
	    let misses = 0;
	    for (let i in output)
	      misses += Math.round(target[i] * 2) != Math.round(output[i] * 2);
	    return misses;
	  }
	};

	//+ Jonas Raoni Soares Silva
	//@ http://jsfromhell.com/array/shuffle [v1.0]
	function shuffle(o) { //v1.0
	  for (let j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
	  return o;
	};

	module.exports = Trainer;

/***/ },
/* 12 */
/***/ function(module, exports, __webpack_require__) {

	module.exports = {
	  Hopfield: __webpack_require__(13),
	  LSTM: __webpack_require__(14),
	  Liquid: __webpack_require__(15),
	  Perceptron: __webpack_require__(16)
	};

/***/ },
/* 13 */
/***/ function(module, exports, __webpack_require__) {

	const Network = __webpack_require__(4);
	const Layer = __webpack_require__(3);
	const Trainer = __webpack_require__(11);

	module.exports = class Hopfield extends Network {
	  constructor(size) {
	    const inputLayer = new Layer(size);
	    const outputLayer = new Layer(size);

	    inputLayer.project(outputLayer, Layer.connectionType.ALL_TO_ALL);

	    super({
	      input: inputLayer,
	      hidden: [],
	      output: outputLayer
	    });

	    this.trainer = new Trainer(this);
	  }

	  learn(patterns) {
	    const set = [];
	    for (let p in patterns)
	      set.push({
	        input: patterns[p],
	        output: patterns[p]
	      });

	    return this.trainer.train(set, {
	      iterations: 500000,
	      error: .00005,
	      rate: 1
	    });
	  }

	  feed(pattern) {
	    const output = this.activate(pattern);

	    pattern = [];
	    for (let i in output)
	      pattern[i] = output[i] > .5 ? 1 : 0;

	    return pattern;
	  }
	}

/***/ },
/* 14 */
/***/ function(module, exports, __webpack_require__) {

	// Multilayer Long Short-Term Memory
	const Network = __webpack_require__(4);
	const Layer = __webpack_require__(3);
	const Trainer = __webpack_require__(11);

	module.exports = class LSTM extends Network {
	  constructor(...args) {
	    if (args.length < 3)
	      throw new Error("not enough layers (minimum 3) !!");

	    const last = args.pop();
	    const option = {
	      peepholes: Layer.connectionType.ALL_TO_ALL,
	      hiddenToHidden: false,
	      outputToHidden: false,
	      outputToGates: false,
	      inputToOutput: true,
	    };
	    if (typeof last != 'number') {
	      var outputs = args.pop();
	      if (last.hasOwnProperty('peepholes'))
	        option.peepholes = last.peepholes;
	      if (last.hasOwnProperty('hiddenToHidden'))
	        option.hiddenToHidden = last.hiddenToHidden;
	      if (last.hasOwnProperty('outputToHidden'))
	        option.outputToHidden = last.outputToHidden;
	      if (last.hasOwnProperty('outputToGates'))
	        option.outputToGates = last.outputToGates;
	      if (last.hasOwnProperty('inputToOutput'))
	        option.inputToOutput = last.inputToOutput;
	    } else
	      var outputs = last;

	    const inputs = args.shift();
	    const layers = args;

	    const inputLayer = new Layer(inputs);
	    const hiddenLayers = [];
	    const outputLayer = new Layer(outputs);

	    let previous = null;

	    // generate layers
	    for (let layer in layers) {
	      // generate memory blocks (memory cell and respective gates)
	      const size = layers[layer];

	      const inputGate = new Layer(size).set({
	        bias: 1
	      });
	      const forgetGate = new Layer(size).set({
	        bias: 1
	      });
	      const memoryCell = new Layer(size);
	      const outputGate = new Layer(size).set({
	        bias: 1
	      });

	      hiddenLayers.push(inputGate);
	      hiddenLayers.push(forgetGate);
	      hiddenLayers.push(memoryCell);
	      hiddenLayers.push(outputGate);

	      // connections from input layer
	      const input = inputLayer.project(memoryCell);
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
	      const output = memoryCell.project(outputLayer);

	      // self-connection
	      const self = memoryCell.project(memoryCell);

	      // hidden to hidden recurrent connection
	      if (option.hiddenToHidden)
	        memoryCell.project(memoryCell, Layer.connectionType.ALL_TO_ELSE);

	      // out to hidden recurrent connection
	      if (option.outputToHidden)
	        outputLayer.project(memoryCell);

	      // out to gates recurrent connection
	      if (option.outputToGates) {
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
	    if (option.inputToOutput)
	      inputLayer.project(outputLayer);

	    // set the layers of the neural network
	    super({
	      input: inputLayer,
	      hidden: hiddenLayers,
	      output: outputLayer
	    });

	    // trainer
	    this.trainer = new Trainer(this);
	  }
	}

/***/ },
/* 15 */
/***/ function(module, exports, __webpack_require__) {

	// Liquid State Machine
	const Network = __webpack_require__(4);
	const Layer = __webpack_require__(3);
	const Trainer = __webpack_require__(11);

	module.exports = class Liquid extends Network {
	  constructor(inputs, hidden, outputs, connections, gates) {
	    // create layers
	    const inputLayer = new Layer(inputs);
	    const hiddenLayer = new Layer(hidden);
	    const outputLayer = new Layer(outputs);

	    // make connections and gates randomly among the neurons
	    const neurons = hiddenLayer.neurons();
	    const connectionList = [];

	    for (let i = 0; i < connections; i++) {
	      // connect two random neurons
	      const from = Math.random() * neurons.length | 0;
	      const to = Math.random() * neurons.length | 0;
	      var connection = neurons[from].project(neurons[to]);
	      connectionList.push(connection);
	    }

	    for (let j = 0; j < gates; j++) {
	      // pick a random gater neuron
	      const gater = Math.random() * neurons.length | 0;
	      // pick a random connection to gate
	      var connection = Math.random() * connectionList.length | 0;
	      // let the gater gate the connection
	      neurons[gater].gate(connectionList[connection]);
	    }

	    // connect the layers
	    inputLayer.project(hiddenLayer);
	    hiddenLayer.project(outputLayer);

	    // set the layers of the network
	    super({
	      input: inputLayer,
	      hidden: [hiddenLayer],
	      output: outputLayer
	    });

	    // trainer
	    this.trainer = new Trainer(this);
	  }
	}

/***/ },
/* 16 */
/***/ function(module, exports, __webpack_require__) {

	// Multilayer Perceptron
	const Network = __webpack_require__(4);
	const Layer = __webpack_require__(3);
	const Trainer = __webpack_require__(11);

	module.exports = class Perceptron extends Network {
	  constructor(...args) {
	    if (args.length < 3)
	      throw new Error("not enough layers (minimum 3) !!");

	    const inputs = args.shift(); // first argument
	    const outputs = args.pop(); // last argument
	    const layers = args; // all the arguments in the middle

	    const input = new Layer(inputs);
	    const hidden = [];
	    const output = new Layer(outputs);

	    let previous = input;

	    // generate hidden layers
	    for (let level in layers) {
	      const size = layers[level];
	      const layer = new Layer(size);
	      hidden.push(layer);
	      previous.project(layer);
	      previous = layer;
	    }
	    previous.project(output);

	    // set layers of the neural network
	    super({
	      input,
	      hidden,
	      output
	    });

	    // trainer for the network
	    this.trainer = new Trainer(this);
	  }
	};

/***/ }
/******/ ])
});
;