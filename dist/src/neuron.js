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
        this.state = this.selfconnection.gain * this.selfconnection.weight * this.state + this.bias;
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
            for (var incoming in this.trace.influences[neuron.ID]) {
                influence += this.trace.influences[neuron.ID][incoming].weight * this.trace.influences[neuron.ID][incoming].from.activation;
            }
            influences[neuron.ID] = influence;
        }
        for (var i in this.connections.inputs) {
            var theInput = this.connections.inputs[i];
            // elegibility trace - Eq. 17
            this.trace.elegibility[theInput.ID] = this.selfconnection.gain * this.selfconnection.weight * this.trace.elegibility[theInput.ID] + theInput.gain * theInput.from.activation;
            for (var id in this.trace.extended) {
                // extended elegibility trace
                var xtrace = this.trace.extended[id];
                var neuron = this.neighboors[id];
                var influence = influences[neuron.ID];
                // eq. 18
                xtrace[theInput.ID] = neuron.selfconnection.gain * neuron.selfconnection.weight * xtrace[theInput.ID] + this.derivative * this.trace.elegibility[theInput.ID] * influence;
            }
        }
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
            for (var id in this.connections.projected) {
                var connection = this.connections.projected[id];
                var neuron = connection.to;
                // Eq. 21
                error += neuron.error.responsibility * connection.gain * connection.weight;
            }
            // projected error responsibility
            this.error.projected = this.derivative * error;
            error = 0;
            for (var id in this.trace.extended) {
                var neuron = this.neighboors[id]; // gated neuron
                var influence = neuron.selfconnection.gater == this ? neuron.old : 0; // if gated neuron's selfconnection is gated by this neuron
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
                var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace.elegibility[input.ID]);
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
                    var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace.elegibility[input.ID]);
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
            var responsibility = getVar(this, 'error', 'responsibility', this.error.responsibility);
            if (isOutput) {
                var target = getVar('target');
                buildSentence(responsibility, ' = ', target, ' - ', activation, store_propagation);
                for (var id in this.connections.inputs) {
                    var input = this.connections.inputs[id];
                    var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace.elegibility[input.ID]);
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
                        var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace.elegibility[input.ID]);
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
                        var trace = getVar(this, 'trace', 'elegibility', input.ID, this.trace.elegibility[input.ID]);
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
            this.weight = typeof weight == 'undefined' || isNaN(weight) ? Math.random() * .2 - .1 : weight;
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

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9uZXVyb24udHMiXSwibmFtZXMiOlsiTmV1cm9uIiwiTmV1cm9uLmNvbnN0cnVjdG9yIiwiTmV1cm9uLnJlYWRJbmNvbW1pbmdDb25uZWN0aW9ucyIsIk5ldXJvbi51cGRhdGVUcmFjZXMiLCJOZXVyb24uYWN0aXZhdGUiLCJOZXVyb24ucHJvcGFnYXRlIiwiTmV1cm9uLnByb2plY3QiLCJOZXVyb24uZ2F0ZSIsIk5ldXJvbi5zZWxmY29ubmVjdGVkIiwiTmV1cm9uLmNvbm5lY3RlZCIsIk5ldXJvbi5jbGVhciIsIk5ldXJvbi5yZXNldCIsIk5ldXJvbi5vcHRpbWl6ZSIsIk5ldXJvbi5Db25uZWN0aW9uIiwiTmV1cm9uLkNvbm5lY3Rpb24uY29uc3RydWN0b3IiLCJOZXVyb24udWlkIiwiTmV1cm9uLnF1YW50aXR5IiwiTmV1cm9uLkNvbm5lY3Rpb24udWlkIl0sIm1hcHBpbmdzIjoiQUFBQSxvQ0FBb0M7QUFHcEMsSUFBTyxNQUFNLFdBQVcsVUFBVSxDQUFDLENBQUM7QUFFcEMsQUFVQTs7NEZBUjRGO0FBRTVGOzs7O0VBSUU7SUFFVyxNQUFNO0lBNkJsQkEsU0E3QllBLE1BQU1BO1FBQ2xCQyxnQkFBV0EsR0FBR0EsSUFBSUEsQ0FBQ0E7UUFFbkJBLE9BQUVBLEdBQUdBLE1BQU1BLENBQUNBLEdBQUdBLEVBQUVBLENBQUNBO1FBQ2xCQSxVQUFLQSxHQUFHQSxJQUFJQSxDQUFDQTtRQUNiQSxnQkFBV0EsR0FBOEJBO1lBQ3hDQSxNQUFNQSxFQUFFQSxFQUFFQTtZQUNWQSxTQUFTQSxFQUFFQSxFQUFFQTtZQUNiQSxLQUFLQSxFQUFFQSxFQUFFQTtTQUNUQSxDQUFDQTtRQUNGQSxVQUFLQSxHQUFHQTtZQUNQQSxjQUFjQSxFQUFFQSxDQUFDQTtZQUNqQkEsU0FBU0EsRUFBRUEsQ0FBQ0E7WUFDWkEsS0FBS0EsRUFBRUEsQ0FBQ0E7U0FDUkEsQ0FBQ0E7UUFDRkEsVUFBS0EsR0FBR0E7WUFDUEEsV0FBV0EsRUFBRUEsRUFBRUE7WUFDZkEsUUFBUUEs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LElBQUlBLENBQUNBLFVBQVVBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBRWpDQSxBQUNBQSw4R0FEOEdBO2dCQUMxR0EsU0FBU0EsR0FBR0EsTUFBTUEsQ0FBQ0EsY0FBY0EsQ0FBQ0EsS0FBS0EsSUFBSUEsSUFBSUEsR0FBR0EsTUFBTUEsQ0FBQ0EsR0FBR0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7WUFHckVBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLFFBQVFBLElBQUlBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFVBQVVBLENBQUNBLE1BQU1BLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLENBQUNBLENBQUNBO2dCQUN2REEsU0FBU0EsSUFBSUEsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsVUFBVUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0EsUUFBUUEsQ0FBQ0EsQ0FBQ0EsTUFBTUEsR0FDOURBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFVBQVVBLENBQUNBLE1BQU1BLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLFFBQVFBLENBQUNBLENBQUNBLElBQUlBLENBQUNBLFVBQVVBLENBQUNBO1lBQzVEQSxDQUFDQTtZQUNEQSxVQUFVQSxDQUFDQSxNQUFNQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxTQUFTQSxDQUFDQTtRQUNuQ0EsQ0FBQ0E7UUFFREEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsSUFBSUEsSUFBSUEsQ0FBQ0EsV0FBV0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDdkNBLElBQUlBLFFBQVFBLEdBQUdBLElBQUlBLENBQUNBLFdBQVdBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBLENBQUNBO1lBRTFDQSxBQUNBQSw2QkFENkJBO1lBQzdCQSxJQUFJQSxDQUFDQSxLQUFLQSxDQUFDQSxXQUFXQSxDQUFDQSxRQUFRQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxJQUFJQSxDQUFDQSxjQUFjQSxDQUFDQSxJQUFJQSxHQUFHQSxJQUFJQSxDQUFDQSxjQUFjQSxDQUNuRkEsTUFBTUEsR0FBR0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsV0FBV0EsQ0FBQ0EsUUFBUUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsUUFBUUEsQ0FBQ0EsSUFBSUEsR0FBR0EsUUFBUUEsQ0FBQ0EsSUFBSUEsQ0FDNUVBLFVBQVVBLENBQUNBO1lBRVpBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLEVBQUVBLElBQUlBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFFBQVFBLENBQUNBLENBQUNBLENBQUNBO2dCQUNwQ0EsQUFDQUEsNkJBRDZCQTtvQkFDekJBLE1BQU1BLEdBQUdBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFFBQVFBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO2dCQUNyQ0EsSUFBSUEsTUFBTUEsR0FBR0EsSUFBSUEsQ0FBQ0EsVUFBVUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0E7Z0JBQ2pDQSxJQUFJQSxTQUFTQSxHQUFHQSxVQUFVQSxDQUFDQSxNQUFNQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQTtnQkFFdENBLEFBQ0FBLFNBRFNBO2dCQUNUQSxNQUFNQSxDQUFDQSxRQUFRQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxNQUFNQSxDQUFDQSxjQUFjQSxDQUFDQSxJQUFJQSxHQUFHQSxNQUFNQSxDQUFDQSxjQUFjQSxDQUN2RUEsTUFBTUEsR0FBR0EsTUFBTUEsQ0FBQ0EsUUFBUUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsSUFBSUEsQ0FBQ0EsVUFBVUEsR0FBR0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsV0FBV0EsQ0FDeEVBLFFBQVFBLENBQUNBLEVBQUVBLENBQUNBLEdBQUdBLFNBQVNBLENBQUNBO1lBQzFCQSxDQUFDQTtRQUNGQSxDQUFDQTtRQUdEQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxVQUFVQSxJQUFJQSxJQUFJQSxDQUFDQSxXQUFXQSxDQUFDQSxLQUFLQSxDQUFDQSxD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QUNBLEtBQUtBLENBQUNBLENBQUNBLElBQUlBLENBQUNBLFVBQVVBLENBQUNBO2dCQUNuQ0EsQ0FBQ0E7Z0JBQ0RBLEFBQ0FBLFNBRFNBO2dCQUNUQSxLQUFLQSxJQUFJQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxjQUFjQSxHQUFHQSxTQUFTQSxDQUFDQTtZQUNsREEsQ0FBQ0E7WUFFREEsQUFDQUEsNkJBRDZCQTtZQUM3QkEsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsS0FBS0EsR0FBR0EsSUFBSUEsQ0FBQ0EsVUFBVUEsR0FBR0EsS0FBS0EsQ0FBQ0E7WUFFM0NBLEFBQ0FBLGdDQURnQ0E7WUFDaENBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLGNBQWNBLEdBQUdBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFNBQVNBLEdBQUdBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLEtBQUtBLENBQUNBO1FBQ3JFQSxDQUFDQTtRQUVEQSxBQUNBQSxnQkFEZ0JBO1FBQ2hCQSxJQUFJQSxHQUFHQSxJQUFJQSxJQUFJQSxFQUFFQSxDQUFDQTtRQUdsQkEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsSUFBSUEsSUFBSUEsQ0FBQ0EsV0FBV0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDeENBLElBQUlBLFFBQVFBLEdBQUdBLElBQUlBLENBQUNBLFdBQVdBLENBQUNBLE1BQU1BLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBRTNDQSxBQUNBQSxTQURTQTtnQkFDTEEsUUFBUUEsR0FBR0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsU0FBU0EsR0FBR0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsV0FBV0EsQ0FBQ0EsUUFBUUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0E7WUFDMUVBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLEVBQUVBLElBQUlBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFFBQVFBLENBQUNBLENBQUNBLENBQUNBO2dCQUNwQ0EsSUFBSUEsTUFBTUEsR0FBR0EsSUFBSUEsQ0FBQ0EsVUFBVUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0E7Z0JBQ2pDQSxRQUFRQSxJQUFJQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxjQUFjQSxHQUFHQSxJQUFJQSxDQUFDQSxLQUFLQSxDQUFDQSxRQUFRQSxDQUM3REEsTUFBTUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0EsUUFBUUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0E7WUFDekJBLENBQUNBO1lBQ0RBLFFBQVFBLENBQUNBLE1BQU1BLElBQUlBLElBQUlBLEdBQUdBLFFBQVFBLEVBQUVBLDZCQUE2QkE7UUFDbEVBLENBQUNBLEdBRG1DQTtRQUdwQ0EsQUFDQUEsY0FEY0E7UUFDZEEsSUFBSUEsQ0FBQ0EsSUFBSUEsSUFBSUEsSUFBSUEsR0FBR0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsY0FBY0EsQ0FBQ0E7SUFDL0NBLENBQUNBO0lBRURMLHdCQUFPQSxHQUFQQSxVQUFRQSxNQUFNQSxFQUFFQSxNQUFlQTtRQUM5Qk0sQUFDQUEsa0JBRGtCQTtRQUNsQkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsTUFBTUEsSUFBSUEsSUFBSUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDcEJBLElBQUlBLENBQUNBLGNBQWNBLENBQUNBLE1BQU1BLEdBQUdBLENBQUNBLENBQUNBO1lBQy9CQSxNQUFNQSxDQUFDQSxJQUFJQSxDQUFDQSxjQUFjQSxDQUFDQTtRQUM1QkEsQ0FBQ0E7UUFFREEsQUFDQUEscUNBRHFDQTtZQUNqQ0EsU0FBU0EsR0FBR0EsSUFBSUEsQ0FBQ0EsU0FBU0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0E7UUFDdkNBLEVBQUVBLENBQUNBLENBQUNBLFNBQVNBLElBQUlBLFNBQVNBLENBQUNBLElBQUlBLElBQUlBLFdBQVdBLENBQUNBLENBQUNBLENBQUNBO1lBQ2hEQSxBQUNBQSxvQkFEb0JBO1lBQ3BCQSxFQUFFQSxDQUFDQSxDQUFDQSxPQUFPQSxNQUFNQSxJQUFJQSxXQUFXQSxDQUFDQTtnQkFDaENBLFNBQVNBLENBQUNBLFVBQVVBLENBQUNBLE1BQU1BLEdBQUdBLE1BQU1BLENBQUNBO1lBQ3RDQSxBQUNBQSw2QkFENkJBO1lBQzdCQSxNQUFNQSxDQUFDQSxTQUFTQSxDQUFDQSxVQUFVQSxDQUFDQTtRQUM3QkEsQ0FBQ0E7UUFBQ0EsSUFBSUEsQ0FBQ0EsQ0FBQ0E7WUFDUEEsQUFDQUEsMEJBRDBCQTtnQkFDdEJBLFVBQVVBLEdBQUdBLElBQUlBLE1BQU1BLENBQUNBLFVBQVVBLENBQUNBLElBQUlBLEVBQUVBLE1BQU1BLEVBQUVBLE1BQU1BLENBQUNBLENBQUNBO1FBQzlEQSxDQUFDQTtRQUVEQSxBQUNBQSwyQ0FEMkNBO1FBQzNDQSxJQUFJQSxDQUFDQSxXQUFXQSxDQUFDQSxTQUFTQSxDQUFDQSxVQUFVQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxVQUFVQSxDQUFDQTtRQUN2REEsSUFBSUEsQ0FBQ0EsVUFBVUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsTUFBTUEsQ0FBQ0E7UUFDcENBLE1BQU1BLENBQUNBLFdBQVdBLENBQUNBLE1BQU1BLENBQUNBLFVBQVVBLENBQUNBLEVBQUVBLENBQUNBLEdBQUdBLFVBQVVBLENBQUNBO1FBQ3REQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxXQUFXQSxDQUFDQSxVQUFVQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQTtRQUU1Q0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsSUFBSUEsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsUUFBUUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDdENBLElBQUlBLEtBQUtBLEdBQUdBLE1BQU1BLENBQUNBLEtBQUtBLENBQUNBLFFBQVFBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBQ3RDQSxLQUFLQSxDQUFDQSxVQUFVQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQTtRQUMxQkEsQ0FBQ0E7UUFFREEsTUFBTUEsQ0FBQ0EsVUFBVUEsQ0FBQ0E7SUFD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QUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxLQUFLQSxJQUFJQSxJQUFJQSxDQUFDQSxLQUFLQSxDQUFDQSxRQUFRQSxDQUFDQTtZQUNyQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsUUFBUUEsSUFBSUEsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsUUFBUUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsQ0FBQ0E7Z0JBQy9DQSxJQUFJQSxDQUFDQSxLQUFLQSxDQUFDQSxRQUFRQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQSxRQUFRQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQTtRQUUzQ0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsY0FBY0EsR0FBR0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsU0FBU0EsR0FBR0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsS0FBS0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7SUFDekVBLENBQUNBO0lBRURWLGdFQUFnRUE7SUFDaEVBLHNCQUFLQSxHQUFMQTtRQUNDVyxJQUFJQSxDQUFDQSxLQUFLQSxFQUFFQSxDQUFDQTtRQUViQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxJQUFJQSxJQUFJQSxJQUFJQSxDQUFDQSxXQUFXQSxDQUFDQTtZQUNqQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsVUFBVUEsSUFBSUEsSUFBSUEsQ0FBQ0EsV0FBV0EsQ0FBQ0EsSUFBSUEsQ0FBQ0EsQ0FBQ0E7Z0JBQzdDQSxJQUFJQSxDQUFDQSxXQUFXQSxDQUFDQSxJQUFJQSxDQUFDQSxDQUFDQSxVQUFVQSxDQUFDQSxDQUFDQSxNQUFNQSxHQUFHQSxJQUFJQSxDQUFDQSxNQUFNQSxFQUFFQSxHQUFHQSxFQUFFQSxHQUFHQSxFQUFFQSxDQUFDQTtRQUN0RUEsSUFBSUEsQ0FBQ0EsSUFBSUEsR0FBR0EsSUFBSUEsQ0FBQ0EsTUFBTUEsRUFBRUEsR0FBR0EsRUFBRUEsR0FBR0EsRUFBRUEsQ0FBQ0E7UUFFcENBLElBQUlBLENBQUNBLEdBQUdBLEdBQUdBLElBQUlBLENBQUNBLEtBQUtBLEdBQUdBLElBQUlBLENBQUNBLFVBQVVBLEdBQUdBLENBQUNBLENBQUNBO0lBQzdDQSxDQUFDQTtJQUtEWCxtRUFBbUVBO0lBQ25FQSx5QkFBUUEsR0FBUkEsVUFBU0EsU0FBU0EsRUFBRUEsS0FBS0E7UUFFeEJZLFNBQVNBLEdBQUdBLFNBQVNBLElBQUlBLEVBQUVBLENBQUNBO1FBQzVCQSxJQUFJQSxJQUFJQSxHQUFHQSxJQUFJQSxDQUFDQTtRQUNoQkEsSUFBSUEsZ0JBQWdCQSxHQUFHQSxFQUFFQSxDQUFDQTtRQUMxQkEsSUFBSUEsV0FBV0EsR0FBR0EsRUFBRUEsQ0FBQ0E7UUFDckJBLElBQUlBLGlCQUFpQkEsR0FBR0EsRUFBRUEsQ0FBQ0E7UUFDM0JBLElBQUlBLEtBQUtBLEdBQUdBLFNBQVNBLENBQUNBLE1BQU1BLElBQUlBLENBQUNBLENBQUNBO1FBQ2xDQSxJQUFJQSxPQUFPQSxHQUFHQSxTQUFTQSxDQUFDQSxPQUFPQSxJQUFJQSxDQUFDQSxDQUFDQTtRQUNyQ0EsSUFBSUEsTUFBTUEsR0FBR0EsU0FBU0EsQ0FBQ0EsTUFBTUEsSUFBSUEsRUFBRUEsQ0FBQ0E7UUFDcENBLElBQUlBLE9BQU9BLEdBQUdBLFNBQVNBLENBQUNBLE9BQU9BLElBQUlBLEVBQUVBLENBQUNBO1FBQ3RDQSxJQUFJQSxPQUFPQSxHQUFHQSxTQUFTQSxDQUFDQSxPQUFPQSxJQUFJQSxFQUFFQSxDQUFDQTtRQUN0Q0EsSUFBSUEsU0FBU0EsR0FBR0EsU0FBU0EsQ0FBQ0EsU0FBU0EsSUFBSUEsRUFBRUEsQ0FBQ0E7UUFDMUNBLElBQUlBLG9CQUFvQkEsR0FBR0EsU0FBU0EsQ0FBQ0Esb0JBQW9CQSxJQUFJQSxFQUFFQSxDQUFDQTtRQUNoRUEsSUFBSUEsZUFBZUEsR0FBR0EsU0FBU0EsQ0FBQ0EsZUFBZUEsSUFBSUEsRUFBRUEsQ0FBQ0E7UUFDdERBLElBQUlBLHFCQUFxQkEsR0FBR0EsU0FBU0EsQ0FBQ0EscUJBQXFCQSxJQUFJQSxFQUFFQSxDQUFDQTtRQUNsRUEsSUFBSUEsTUFBTUEsR0FBR0EsU0FBU0EsQ0FBQ0EsTUFBTUEsSUFBSUEsRUFBRUEsT0FBT0EsRUFBRUEsQ0FBQ0EsRUFBRUEsUUFBUUEsRUFBRUEsQ0FBQ0EsRUFBRUEsQ0FBQ0E7UUFFN0RBLEFBQ0FBLHFCQURxQkE7WUFDakJBLFFBQVFBLEdBQUdBLFVBQVNBLEtBQUtBO1lBQzVCLElBQUksU0FBUyxHQUFHLEtBQUssSUFBSSxNQUFNLElBQUksS0FBSyxDQUFDLE1BQU0sQ0FBQyxPQUFPLENBQUMsQ0FBQztZQUN6RCxFQUFFLENBQUMsQ0FBQyxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUM7Z0JBQ2hCLE1BQU0sQ0FBQyxPQUFPLEdBQUcsS0FBSyxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUM7Z0JBQ3BDLE1BQU0sQ0FBQyxLQUFLLENBQUMsR0FBRyxNQUFNLENBQUMsT0FBTyxDQUFDO1lBQ2hDLENBQUM7UUFDRixDQUFDLENBQUFBO1FBQ0RBLFFBQVFBLENBQUNBLG9CQUFvQkEsQ0FBQ0EsQ0FBQ0E7UUFDL0JBLFFBQVFBLENBQUNBLGVBQWVBLENBQUNBLENBQUNBO1FBQzFCQSxRQUFRQSxDQUFDQSxxQkFBcUJBLENBQUNBLENBQUNBO1FBQ2hDQSxJQUFJQSxZQUFZQSxHQUFHQSxNQUFNQSxDQUFDQSxPQUFPQSxDQUFDQTtRQUVsQ0EsQUFDQUEsc0VBRHNFQTtZQUNsRUEsTUFBTUEsR0FBR0E7WUFBUyxjQUFjO2lCQUFkLFdBQWMsQ0FBZCxzQkFBYyxDQUFkLElBQWM7Z0JBQWQsNkJBQWM7O1lBQ25DLElBQUksRUFBRSxDQUFDO1lBQ1AsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLE1BQU0sSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUV0QixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLElBQUksUUFBUSxDQUFDLENBQUMsQ0FBQztvQkFDekIsRUFBRSxHQUFHLFNBQVMsR0FBRyxPQUFPLENBQUMsTUFBTSxDQUFDO29CQUNoQyxPQUFPLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO2dCQUNyQixDQUFDO2dCQUFDLElBQUk7b0JBQ0wsRUFBRSxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDZCxFQUFFLENBQUMsQ0FBQyxFQUFFLElBQUksU0FBUyxDQUFDO29CQUNuQixNQUFNLENBQUMsU0FBUyxDQUFDLEVBQUUsQ0FBQyxDQUFDO2dCQUN0QixNQUFNLENBQUMsU0FBUyxDQUFDLEVBQUUsQ0FBQyxHQUFHO29CQUN0QixLQUFLLEVBQUUsQ0FBQztvQkFDUixFQUFFLEVBQUUsS0FBSyxFQUFFO2lCQUNYLENBQUM7WUFDSCxDQUFDO1lBQUMsSUFBSSxDQUFDLENBQUM7Z0JBQ1AsSUFBSSxRQUFRLEdBQUcsSUFBSSxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUM7Z0JBQy9CLEVBQUUsQ0FBQyxDQUFDLFFBQVEsQ0FBQztvQkFDWixJQUFJLEtBQUssR0FBRyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUM7Z0JBRXhCLElBQUksSUFBSSxHQUFHLElBQUksQ0FBQyxLQUFLLEVBQUUsQ0FBQztnQkFDeEIsSUFBSSxJQUFJLEdBQUcsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDO2dCQUV0QixFQUFFLENBQUMsQ0FBQyxDQUFDLFFBQVEsQ0FBQztvQkFDYixJQUFJLEtBQUssR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7Z0JBRXhCLEVBQUUsR0FBRyxJQUFJLEdBQUcsR0FBRyxDQUFDO2dCQUNoQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsUUFBUSxJQUFJLElBQUksQ0FBQztvQkFDekIsRUFBRSxJQUFJLElBQUksQ0FBQyxRQUFRLENBQUMsR0FBRyxHQUFHLENBQUM7Z0JBQzVCLEVBQUUsSUFBSSxJQUFJLENBQUMsRUFBRSxDQUFDO2dCQUNkLEVBQUUsQ0FBQyxDQUFDLEVBQUUsSUFBSSxTQUFTLENBQUM7b0JBQ25CLE1BQU0sQ0FBQyxTQUFTLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBRXRCLE1BQU0sQ0FBQyxTQUFTLENBQUMsRUFBRSxDQUFDLEdBQUc7b0JBQ3RCLEtBQUssRUFBRSxLQUFLO29CQUNaLEVBQUUsRUFBRSxLQUFLLEVBQUU7aUJBQ1gsQ0FBQztZQUNILENBQUM7UUFDRixDQUFDLENBQUNBO1FBRUZBLEFBQ0FBLGlCQURpQkE7WUFDYkEsYUFBYUEsR0FBR0E7WUFBUyxjQUFjO2lCQUFkLFdBQWMsQ0FBZCxzQkFBYyxDQUFkLElBQWM7Z0JBQWQsNkJBQWM7O1lBQzFDLElBQUksS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQztZQUN2QixJQUFJLFFBQVEsR0FBRyxFQUFFLENBQUM7WUFDbEIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUM7Z0JBQ2xCLEVBQUUsQ0FBQyxDQUFDLE9BQU8sSUFBSSxDQUFDLENBQUMsQ0FBQyxJQUFJLFFBQVEsQ0FBQztvQkFDOUIsUUFBUSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDckIsSUFBSTtvQkFDSCxRQUFRLElBQUksSUFBSSxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsR0FBRyxDQUFDO1lBRXRDLEtBQUssQ0FBQyxJQUFJLENBQUMsUUFBUSxHQUFHLEdBQUcsQ0FBQyxDQUFDO1FBQzVCLENBQUMsQ0FBQUE7UUFFREEsQUFDQUEsd0NBRHdDQTtZQUNwQ0EsT0FBT0EsR0FBR0EsVUFBU0EsR0FBR0E7WUFDekIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLElBQUksSUFBSSxHQUFHLENBQUMsQ0FBQyxDQUFDO2dCQUN0QixFQUFFLENBQUMsQ0FBQyxHQUFHLENBQUMsY0FBYyxDQUFDLElBQUksQ0FBQyxDQUFDO29CQUM1QixNQUFNLENBQUMsS0FBSyxDQUFDO1lBQ2YsQ0FBQztZQUNELE1BQU0sQ0FBQyxJQUFJLENBQUM7UUFDYixDQUFDLENBQUNBO1FBRUZBLEFBQ0FBLGdDQURnQ0E7WUFDNUJBLGFBQWFBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLENBQUNBLFdBQVdBLENBQUNBLFNBQVNBLENBQUNBLENBQUNBO1FBQ3hEQSxJQUFJQSxPQUFPQSxHQUFHQSxPQUFPQSxDQUFDQSxJQUFJQSxDQUFDQSxXQUFXQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQTtRQUM5Q0EsSUFBSUEsT0FBT0EsR0FBR0EsS0FBS0EsSUFBSUEsT0FBT0EsR0FBR0EsSUFBSUEsR0FBR0EsT0FBT0EsQ0FBQ0EsSUFBSUEsQ0FBQ0EsV0FBV0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0E7UUFDekVBLElBQUlBLFFBQVFBLEdBQUdBLEtBQUtBLElBQUlBLFFBQVFBLEdBQUdBLElBQUlBLEdBQUdBLGFBQWFBLElBQUlBLE9BQU9BLENBQUNBO1FBRW5FQSxBQUNBQSw4QkFEOEJBO1lBQzFCQSxJQUFJQSxHQUFHQSxNQUFNQSxDQUFDQSxNQUFNQSxDQUFDQSxDQUFDQTtRQUMxQkEsSUFBSUEsVUFBVUEsR0FBR0EsTUFBTUEsQ0FBQ0EsSUFBSUEsRUFBRUEsWUFBWUEsQ0FBQ0EsQ0FBQ0E7UUFDNUNBLEVBQUVBLENBQUNBLENBQUNBLE9BQU9BLENBQUNBO1lBQ1hBLE1BQU1BLENBQUNBLElBQUlBLENBQUNBLFVBQVVBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1FBQzVCQSxJQUFJQSxDQUFDQSxDQUFDQTtZQUNMQSxvQkFBb0JBLENBQUNBLFlBQVlBLENBQUNBLENBQUNBLElBQUlBLENBQUNBLGdCQUFnQkEsQ0FBQ0EsQ0FBQ0E7WUFDMURBLGVBQWVBLENBQUNBLFlBQVlBLENBQUNBLENBQUNBLElBQUlBLENBQUNBLFdBQVdBLENBQUNBLENBQUNBO1lBQ2hEQSxxQkFBcUJBLENBQUNBLFlBQVlBLENBQUNBLENBQUNBLElBQUlBLENBQUNBLGlCQUFpQkEsQ0FBQ0EsQ0FBQ0E7WUFDNURBLElBQUlBLEdBQUdBLEdBQUdBLE1BQU1BLENBQUNBLElBQUlBLEVBQUVBLEtBQUtBLENBQUNBLENBQUNBO1lBQzlCQSxJQUFJQSxLQUFLQSxHQUFHQSxNQUFNQSxDQUFDQSxJQUFJQSxFQUFFQSxPQUFPQSxDQUFDQSxDQUFDQTtZQUNsQ0EsSUFBSUEsSUFBSUEsR0FBR0EsTUFBTUEsQ0FBQ0EsSUFBSUEsRUFBRUEsTUFBTUEsQ0FBQ0EsQ0FBQ0E7WUFDaENBLEVBQUVBLENBQUNBLENBQUNBLElBQUlBLENBQUNBLGNBQWNBLENBQUNBLEtBQUtBLENBQUNBO2dCQUM3QkEsSUFBSUEsU0FBU0EsR0FBR0EsTUFBTUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsY0FBY0EsRUFBRUEsTUFBTUEsQ0FBQ0EsQ0FBQ0E7WUFDckRBLEVBQUVBLENBQUNBLENBQUNBLElBQUlBLENBQUNBLGFBQWFBLEVBQUVBLENBQUNBO2dCQUN4QkEsSUFBSUEsV0FBV0EsR0FBR0EsTUFBTUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsY0FBY0EsRUFBRUEsUUFBUUEsQ0FBQ0EsQ0FBQ0E7WUFDekRBLGFBQWFBLENBQUNBLEdBQUdBLEVBQUVBLEtBQUtBLEVBQUVBLEtBQUtBLEVBQUVBLGdCQUFnQkEsQ0FBQ0EsQ0FBQ0E7WUFDbkRBLEVBQUVBLENBQUNBLENBQUNBLElBQUlBLENBQUNBLGFBQWFBLEVBQUVBLENBQUNBO2dCQUN4QkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsSUFBSUEsQ0FBQ0EsY0FBY0EsQ0FBQ0EsS0FBS0EsQ0FBQ0E7b0JBQzdCQSxhQUFhQSxDQUFDQSxLQUFLQSxFQUFFQSxLQUFLQSxFQUFFQSxTQUFTQSxFQUFFQSxLQUFLQSxFQUFFQSxXQUFXQSxFQUFFQSxLQUFLQSxFQUMvREEsS0FBS0EsRUFBRUEsS0FBS0EsRUFBRUEsSUFBSUEsRUFBRUEsZ0JBQWdCQSxDQUFDQSxDQUFDQTtnQkFDeENBLElBQUlBO29CQUNIQSxhQUFhQSxDQUFDQSxLQUFLQSxFQUFFQSxLQUFLQSxFQUFFQSxXQUFXQSxFQUFFQSxLQUFLQSxFQUFFQSxLQUFLQSxFQUFFQSxLQUFLQSxFQUMzREEsSUFBSUEsRUFBRUEsZ0JBQWdCQSxDQUFDQSxDQUFDQTtZQUMzQkEsSUFBSUE7Z0JBQ0hBLGFBQWFBLENBQUNBLEtBQUtBLEVBQUVBLEtBQUtBLEVBQUVBLElBQUlBLEVBQUVBLGdCQUFnQkEsQ0FBQ0EsQ0FBQ0E7WUFDckRBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLElBQUlBLElBQUlBLENBQUNBLFdBQVdBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBO2dCQUN2Q0EsSUFBSUEsS0FBS0EsR0FBR0EsSUFBSUEsQ0FBQ0EsV0FBV0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ3ZDQSxJQUFJQSxnQkFBZ0JBLEdBQUdBLE1BQU1BLENBQUNBLEtBQUtBLENBQUNBLElBQUlBLEVBQUVBLFlBQVlBLENBQUNBLENBQUNBO2dCQUN4REEsSUFBSUEsWUFBWUEsR0FBR0EsTUFBTUEsQ0FBQ0EsS0FBS0EsRUFBRUEsUUFBUUEsQ0FBQ0EsQ0FBQ0E7Z0JBQzNDQSxFQUFFQSxDQUFDQSxDQUFDQSxLQUFLQSxDQUFDQSxLQUFLQSxDQUFDQTtvQkFDZkEsSUFBSUEsVUFBVUEsR0FBR0EsTUFBTUEsQ0FBQ0EsS0FBS0EsRUFBRUEsTUFBTUEsQ0FBQ0EsQ0FBQ0E7Z0JBQ3hDQSxFQUFFQSxDQUFDQSxDQUFDQSxJQUFJQSxDQUFDQSxXQUFXQSxDQUFDQSxNQUFNQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxLQUFLQSxDQUFDQTtvQkFDcENBLGFBQWFBLENBQUNBLEtBQUtBLEVBQUVBLE1BQU1BLEVBQUVBLGdCQUFnQkEsRUFBRUEsS0FBS0EsRUFDbkRBLFlBQVlBLEVBQUVBLEtBQUtBLEVBQUVBLFVBQVVBLEVBQUVBLGdCQUFnQkEsQ0FBQ0EsQ0FBQ0E7Z0JBQ3JEQSxJQUFJQTtvQkFDSEEsYUFBYUEsQ0FBQ0EsS0FBS0EsRUFBRUEsTUFBTUEsRUFBRUEsZ0JBQWdCQSxFQUFFQSxLQUFLQSxFQUNuREEsWUFBWUEsRUFBRUEsZ0JBQWdCQSxDQUFDQSxDQUFDQTtZQUNuQ0EsQ0FBQ0E7WUFDREEsSUFBSUEsVUFBVUEsR0FBR0EsTUFBTUEsQ0FBQ0EsSUFBSUEsRUFBRUEsWUFBWUEsQ0FBQ0EsQ0FBQ0E7WUFDNUNBLE1BQU1BLENBQUNBLENBQUNBLElBQUlBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBO2dC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Q0E7NEJBQ2ZBLGFBQWFBLENBQUNBLEtBQUtBLEVBQUVBLEtBQUtBLEVBQUVBLFNBQVNBLEVBQUVBLEtBQUtBLEVBQUVBLFdBQVdBLEVBQ3hEQSxLQUFLQSxFQUFFQSxLQUFLQSxFQUFFQSxLQUFLQSxFQUFFQSxVQUFVQSxFQUFFQSxLQUFLQSxFQUFFQSxnQkFBZ0JBLEVBQ3hEQSxXQUFXQSxDQUFDQSxDQUFDQTt3QkFDZkEsSUFBSUE7NEJBQ0hBLGFBQWFBLENBQUNBLEtBQUtBLEVBQUVBLEtBQUtBLEVBQUVBLFNBQVNBLEVBQUVBLEtBQUtBLEVBQUVBLFdBQVdBLEVBQ3hEQSxLQUFLQSxFQUFFQSxLQUFLQSxFQUFFQSxLQUFLQSxFQUFFQSxnQkFBZ0JBLEVBQUVBLFdBQVdBLENBQUNBLENBQUNBO29CQUN2REEsQ0FBQ0E7b0JBQUNBLElBQUlBLENBQUNBLENBQUNBO3dCQUNQQSxFQUFFQSxDQUFDQSxDQUFDQSxLQUFLQSxDQUFDQSxLQUFLQSxDQUFDQTs0QkFDZkEsYUFBYUEsQ0FBQ0EsS0FBS0EsRUFBRUEsS0FBS0EsRUFBRUEsV0FBV0EsRUFBRUEsS0FBS0EsRUFBRUEsS0FBS0EsRUFBRUEsS0FBS0EsRUFDM0RBLFVBQVVBLEVBQUVBLEtBQUtBLEVBQUVBLGdCQUFnQkEsRUFBRUEsV0FBV0EsQ0FBQ0EsQ0FBQ0E7d0JBQ3BEQSxJQUFJQTs0QkFDSEEsYUFBYUEsQ0FBQ0EsS0FBS0EsRUFBRUEsS0FBS0EsRUFBRUEsV0FBV0EsRUFBRUEsS0FBS0EsRUFBRUEsS0FBS0EsRUFBRUEsS0FBS0EsRUFDM0RBLGdCQUFnQkEsRUFBRUEsV0FBV0EsQ0FBQ0EsQ0FBQ0E7b0JBQ2xDQSxDQUFDQTtnQkFDRkEsQ0FBQ0E7Z0JBQUNBLElBQUlBLENBQUNBLENBQUNBO29CQUNQQSxFQUFFQSxDQUFDQSxDQUFDQSxLQUFLQSxDQUFDQSxLQUFLQSxDQUFDQTt3QkFDZkEsYUFBYUEsQ0FBQ0EsS0FBS0EsRUFBRUEsS0FBS0EsRUFBRUEsVUFBVUEsRUFBRUEsS0FBS0EsRUFBRUEsZ0JBQWdCQSxFQUM5REEsV0FBV0Es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QVVBLEtBQVZBLGlCQUFVQSxRQUs5QkE7QUFBREEsQ0FBQ0EsRUFMYSxNQUFNLEdBQU4sY0FBTSxLQUFOLGNBQU0sUUFLbkIiLCJmaWxlIjoic3JjL25ldXJvbi5qcyIsInNvdXJjZXNDb250ZW50IjpbIi8vLyA8cmVmZXJlbmNlIHBhdGg9XCJzeW5hcHRpYy50c1wiIC8+XG5cbmltcG9ydCBTeW5hcHRpYyA9IHJlcXVpcmUoJy4vc3luYXB0aWMnKTtcbmltcG9ydCBTcXVhc2ggPSByZXF1aXJlKCcuL3NxdWFzaCcpO1xuXG4vKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIE5FVVJPTlxuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKi9cblxuLyogVFMgQ0hBTkdFUzpcblxuXHROb3cgTmV1cm9uLmNvbm5lY3RlZChuZXVyb24pIHJldHVybnMgbnVsbCBpbnN0ZWFkIG9mIGZhbHNlXG5cbiovXG5cbmV4cG9ydCBjbGFzcyBOZXVyb24ge1xuXHRvcHRpbWl6YWJsZSA9IHRydWU7XG5cblx0SUQgPSBOZXVyb24udWlkKCk7XG5cdGxhYmVsID0gbnVsbDtcblx0Y29ubmVjdGlvbnM6IE5ldXJvbi5JTmV1cm9uQ29ubmVjdGlvbnMgPSB7XG5cdFx0aW5wdXRzOiB7fSxcblx0XHRwcm9qZWN0ZWQ6IHt9LFxuXHRcdGdhdGVkOiB7fVxuXHR9O1xuXHRlcnJvciA9IHtcblx0XHRyZXNwb25zaWJpbGl0eTogMCxcblx0XHRwcm9qZWN0ZWQ6IDAsXG5cdFx0Z2F0ZWQ6IDBcblx0fTtcblx0dHJhY2UgPSB7XG5cdFx0ZWxlZ2liaWxpdHk6IHt9LFxuXHRcdGV4dGVuZGVkOiB7fSxcblx0XHRpbmZsdWVuY2VzOiB7fVxuXHR9O1xuXHRzdGF0ZSA9IDA7XG5cdG9sZCA9IDA7XG5cdGFjdGl2YXRpb24gPSAwO1xuXHRzZWxmY29ubmVjdGlvbiA9IG5ldyBOZXVyb24uQ29ubmVjdGlvbih0aGlzLCB0aGlzLCAwKTsgLy8gd2VpZ2h0ID0gMCAtPiBub3QgY29ubmVjdGVkXG5cdHNxdWFzaCA9IFNxdWFzaC5MT0dJU1RJQztcblx0bmVpZ2hib29ycyA9IHt9O1xuXHRiaWFzID0gTWF0aC5yYW5kb20oKSAqIC4yIC0gLjE7XG5cdGRlcml2YXRpdmUgPSAwO1xuXG5cdGNvbnN0cnVjdG9yKCkge1xuXG5cdH1cblxuXHRyZWFkSW5jb21taW5nQ29ubmVjdGlvbnMoaW5wdXQ/OiBudW1iZXIpIHtcblx0XHQvLyBhY3RpdmF0aW9uIGZyb20gZW52aXJvbWVudCAoZm9yIGlucHV0IG5ldXJvbnMpXG5cdFx0aWYgKHR5cGVvZiBpbnB1dCAhPSAndW5kZWZpbmVkJykge1xuXHRcdFx0dGhpcy5hY3RpdmF0aW9uID0gaW5wdXQ7XG5cdFx0XHR0aGlzLmRlcml2YXRpdmUgPSAwO1xuXHRcdFx0dGhpcy5iaWFzID0gMDtcblx0XHRcdHJldHVybiB0aGlzLmFjdGl2YXRpb247XG5cdFx0fVxuXHRcdFxuXHRcdC8vIG9sZCBzdGF0ZVxuXHRcdHRoaXMub2xkID0gdGhpcy5zdGF0ZTtcblxuXHRcdC8vIGVxLiAxNVxuXHRcdHRoaXMuc3RhdGUgPSB0aGlzLnNlbGZjb25uZWN0aW9uLmdhaW4gKiB0aGlzLnNlbGZjb25uZWN0aW9uLndlaWdodCAqXG5cdFx0dGhpcy5zdGF0ZSArIHRoaXMuYmlhcztcblxuXHRcdGZvciAodmFyIGkgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdHZhciB0aGVJbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2ldO1xuXHRcdFx0dGhpcy5zdGF0ZSArPSB0aGVJbnB1dC5mcm9tLmFjdGl2YXRpb24gKiB0aGVJbnB1dC53ZWlnaHQgKiB0aGVJbnB1dC5nYWluO1xuXHRcdH1cblxuXHRcdC8vIGVxLiAxNlxuXHRcdHRoaXMuYWN0aXZhdGlvbiA9IHRoaXMuc3F1YXNoKHRoaXMuc3RhdGUpO1xuXG5cdFx0Ly8gZicocylcblx0XHR0aGlzLmRlcml2YXRpdmUgPSB0aGlzLnNxdWFzaCh0aGlzLnN0YXRlLCB0cnVlKTtcblx0fVxuXG5cdHVwZGF0ZVRyYWNlcygpIHtcblx0XHQvLyB1cGRhdGUgdHJhY2VzXG5cdFx0dmFyIGluZmx1ZW5jZXMgPSBbXTtcblx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHQvLyBleHRlbmRlZCBlbGVnaWJpbGl0eSB0cmFjZVxuXHRcdFx0dmFyIHh0cmFjZSA9IHRoaXMudHJhY2UuZXh0ZW5kZWRbaWRdO1xuXHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cblx0XHRcdC8vIGlmIGdhdGVkIG5ldXJvbidzIHNlbGZjb25uZWN0aW9uIGlzIGdhdGVkIGJ5IHRoaXMgdW5pdCwgdGhlIGluZmx1ZW5jZSBrZWVwcyB0cmFjayBvZiB0aGUgbmV1cm9uJ3Mgb2xkIHN0YXRlXG5cdFx0XHR2YXIgaW5mbHVlbmNlID0gbmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyID09IHRoaXMgPyBuZXVyb24ub2xkIDogMDtcblxuXHRcdFx0Ly8gaW5kZXggcnVucyBvdmVyIGFsbCB0aGUgaW5jb21pbmcgY29ubmVjdGlvbnMgdG8gdGhlIGdhdGVkIG5ldXJvbiB0aGF0IGFyZSBnYXRlZCBieSB0aGlzIHVuaXRcblx0XHRcdGZvciAodmFyIGluY29taW5nIGluIHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdKSB7IC8vIGNhcHR1cmVzIHRoZSBlZmZlY3QgdGhhdCBoYXMgYW4gaW5wdXQgY29ubmVjdGlvbiB0byB0aGlzIHVuaXQsIG9uIGEgbmV1cm9uIHRoYXQgaXMgZ2F0ZWQgYnkgdGhpcyB1bml0XG5cdFx0XHRcdGluZmx1ZW5jZSArPSB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXVtpbmNvbWluZ10ud2VpZ2h0ICpcblx0XHRcdFx0dGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF1baW5jb21pbmddLmZyb20uYWN0aXZhdGlvbjtcblx0XHRcdH1cblx0XHRcdGluZmx1ZW5jZXNbbmV1cm9uLklEXSA9IGluZmx1ZW5jZTtcblx0XHR9XG5cblx0XHRmb3IgKHZhciBpIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHR2YXIgdGhlSW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpXTtcblxuXHRcdFx0Ly8gZWxlZ2liaWxpdHkgdHJhY2UgLSBFcS4gMTdcblx0XHRcdHRoaXMudHJhY2UuZWxlZ2liaWxpdHlbdGhlSW5wdXQuSURdID0gdGhpcy5zZWxmY29ubmVjdGlvbi5nYWluICogdGhpcy5zZWxmY29ubmVjdGlvblxuXHRcdFx0LndlaWdodCAqIHRoaXMudHJhY2UuZWxlZ2liaWxpdHlbdGhlSW5wdXQuSURdICsgdGhlSW5wdXQuZ2FpbiAqIHRoZUlucHV0LmZyb21cblx0XHRcdC5hY3RpdmF0aW9uO1xuXG5cdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdC8vIGV4dGVuZGVkIGVsZWdpYmlsaXR5IHRyYWNlXG5cdFx0XHRcdHZhciB4dHJhY2UgPSB0aGlzLnRyYWNlLmV4dGVuZGVkW2lkXTtcblx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdHZhciBpbmZsdWVuY2UgPSBpbmZsdWVuY2VzW25ldXJvbi5JRF07XG5cblx0XHRcdFx0Ly8gZXEuIDE4XG5cdFx0XHRcdHh0cmFjZVt0aGVJbnB1dC5JRF0gPSBuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2FpbiAqIG5ldXJvbi5zZWxmY29ubmVjdGlvblxuXHRcdFx0XHQud2VpZ2h0ICogeHRyYWNlW3RoZUlucHV0LklEXSArIHRoaXMuZGVyaXZhdGl2ZSAqIHRoaXMudHJhY2UuZWxlZ2liaWxpdHlbXG5cdFx0XHRcdHRoZUlucHV0LklEXSAqIGluZmx1ZW5jZTtcblx0XHRcdH1cblx0XHR9XG5cdFx0XG5cdFx0Ly8gIHVwZGF0ZSBnYXRlZCBjb25uZWN0aW9uJ3MgZ2FpbnNcblx0XHRmb3IgKHZhciBjb25uZWN0aW9uIGluIHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWQpIHtcblx0XHRcdHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWRbY29ubmVjdGlvbl0uZ2FpbiA9IHRoaXMuYWN0aXZhdGlvbjtcblx0XHR9XG5cdH1cbiAgXG4gIFxuXHQvLyBhY3RpdmF0ZSB0aGUgbmV1cm9uXG5cdGFjdGl2YXRlKGlucHV0PzogbnVtYmVyKSA6IG51bWJlciB7XG5cdFx0dGhpcy5yZWFkSW5jb21taW5nQ29ubmVjdGlvbnMoaW5wdXQpO1xuXG5cdFx0dGhpcy51cGRhdGVUcmFjZXMoKTtcblxuXHRcdHJldHVybiB0aGlzLmFjdGl2YXRpb247XG5cdH1cblxuXHQvLyBiYWNrLXByb3BhZ2F0ZSB0aGUgZXJyb3Jcblx0cHJvcGFnYXRlKHJhdGU6IG51bWJlciwgdGFyZ2V0PzogbnVtYmVyKSB7XG5cdFx0Ly8gZXJyb3IgYWNjdW11bGF0b3Jcblx0XHR2YXIgZXJyb3IgPSAwO1xuXG5cdFx0Ly8gd2hldGhlciBvciBub3QgdGhpcyBuZXVyb24gaXMgaW4gdGhlIG91dHB1dCBsYXllclxuXHRcdHZhciBpc091dHB1dCA9IHR5cGVvZiB0YXJnZXQgIT0gJ3VuZGVmaW5lZCcgJiYgdGFyZ2V0ICE9IG51bGw7XG5cblx0XHQvLyBvdXRwdXQgbmV1cm9ucyBnZXQgdGhlaXIgZXJyb3IgZnJvbSB0aGUgZW52aXJvbWVudFxuXHRcdGlmIChpc091dHB1dClcblx0XHRcdHRoaXMuZXJyb3IucmVzcG9uc2liaWxpdHkgPSB0aGlzLmVycm9yLnByb2plY3RlZCA9IHRhcmdldCAtIHRoaXMuYWN0aXZhdGlvbjsgLy8gRXEuIDEwXG4gICAgXG5cdFx0ZWxzZSAvLyB0aGUgcmVzdCBvZiB0aGUgbmV1cm9uIGNvbXB1dGUgdGhlaXIgZXJyb3IgcmVzcG9uc2liaWxpdGllcyBieSBiYWNrcHJvcGFnYXRpb25cblx0XHR7XG5cdFx0XHQvLyBlcnJvciByZXNwb25zaWJpbGl0aWVzIGZyb20gYWxsIHRoZSBjb25uZWN0aW9ucyBwcm9qZWN0ZWQgZnJvbSB0aGlzIG5ldXJvblxuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpIHtcblx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZFtpZF07XG5cdFx0XHRcdHZhciBuZXVyb24gPSBjb25uZWN0aW9uLnRvO1xuXHRcdFx0XHQvLyBFcS4gMjFcblx0XHRcdFx0ZXJyb3IgKz0gbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5ICogY29ubmVjdGlvbi5nYWluICogY29ubmVjdGlvbi53ZWlnaHQ7XG5cdFx0XHR9XG5cblx0XHRcdC8vIHByb2plY3RlZCBlcnJvciByZXNwb25zaWJpbGl0eVxuXHRcdFx0dGhpcy5lcnJvci5wcm9qZWN0ZWQgPSB0aGlzLmRlcml2YXRpdmUgKiBlcnJvcjtcblxuXHRcdFx0ZXJyb3IgPSAwO1xuXHRcdFx0Ly8gZXJyb3IgcmVzcG9uc2liaWxpdGllcyBmcm9tIGFsbCB0aGUgY29ubmVjdGlvbnMgZ2F0ZWQgYnkgdGhpcyBuZXVyb25cblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07IC8vIGdhdGVkIG5ldXJvblxuXHRcdFx0XHR2YXIgaW5mbHVlbmNlID0gbmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyID09IHRoaXMgPyBuZXVyb24ub2xkIDogMDsgLy8gaWYgZ2F0ZWQgbmV1cm9uJ3Mgc2VsZmNvbm5lY3Rpb24gaXMgZ2F0ZWQgYnkgdGhpcyBuZXVyb25cblxuXHRcdFx0XHQvLyBpbmRleCBydW5zIG92ZXIgYWxsIHRoZSBjb25uZWN0aW9ucyB0byB0aGUgZ2F0ZWQgbmV1cm9uIHRoYXQgYXJlIGdhdGVkIGJ5IHRoaXMgbmV1cm9uXG5cdFx0XHRcdGZvciAodmFyIGlucHV0IGluIHRoaXMudHJhY2UuaW5mbHVlbmNlc1tpZF0pIHsgLy8gY2FwdHVyZXMgdGhlIGVmZmVjdCB0aGF0IHRoZSBpbnB1dCBjb25uZWN0aW9uIG9mIHRoaXMgbmV1cm9uIGhhdmUsIG9uIGEgbmV1cm9uIHdoaWNoIGl0cyBpbnB1dC9zIGlzL2FyZSBnYXRlZCBieSB0aGlzIG5ldXJvblxuXHRcdFx0XHRcdGluZmx1ZW5jZSArPSB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbaWRdW2lucHV0XS53ZWlnaHQgKiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbXG5cdFx0XHRcdFx0bmV1cm9uLklEXVtpbnB1dF0uZnJvbS5hY3RpdmF0aW9uO1xuXHRcdFx0XHR9XG5cdFx0XHRcdC8vIGVxLiAyMlxuXHRcdFx0XHRlcnJvciArPSBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkgKiBpbmZsdWVuY2U7XG5cdFx0XHR9XG5cblx0XHRcdC8vIGdhdGVkIGVycm9yIHJlc3BvbnNpYmlsaXR5XG5cdFx0XHR0aGlzLmVycm9yLmdhdGVkID0gdGhpcy5kZXJpdmF0aXZlICogZXJyb3I7XG5cblx0XHRcdC8vIGVycm9yIHJlc3BvbnNpYmlsaXR5IC0gRXEuIDIzXG5cdFx0XHR0aGlzLmVycm9yLnJlc3BvbnNpYmlsaXR5ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgKyB0aGlzLmVycm9yLmdhdGVkO1xuXHRcdH1cblxuXHRcdC8vIGxlYXJuaW5nIHJhdGVcblx0XHRyYXRlID0gcmF0ZSB8fCAuMTtcblxuXHRcdC8vIGFkanVzdCBhbGwgdGhlIG5ldXJvbidzIGluY29taW5nIGNvbm5lY3Rpb25zXG5cdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdHZhciB0aGVJbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcblxuXHRcdFx0Ly8gRXEuIDI0XG5cdFx0XHR2YXIgZ3JhZGllbnQgPSB0aGlzLmVycm9yLnByb2plY3RlZCAqIHRoaXMudHJhY2UuZWxlZ2liaWxpdHlbdGhlSW5wdXQuSURdO1xuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0Z3JhZGllbnQgKz0gbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5ICogdGhpcy50cmFjZS5leHRlbmRlZFtcblx0XHRcdFx0bmV1cm9uLklEXVt0aGVJbnB1dC5JRF07XG5cdFx0XHR9XG5cdFx0XHR0aGVJbnB1dC53ZWlnaHQgKz0gcmF0ZSAqIGdyYWRpZW50OyAvLyBhZGp1c3Qgd2VpZ2h0cyAtIGFrYSBsZWFyblxuXHRcdH1cblxuXHRcdC8vIGFkanVzdCBiaWFzXG5cdFx0dGhpcy5iaWFzICs9IHJhdGUgKiB0aGlzLmVycm9yLnJlc3BvbnNpYmlsaXR5O1xuXHR9XG5cblx0cHJvamVjdChuZXVyb24sIHdlaWdodD86IG51bWJlcik6IE5ldXJvbi5Db25uZWN0aW9uIHtcblx0XHQvLyBzZWxmLWNvbm5lY3Rpb25cblx0XHRpZiAobmV1cm9uID09IHRoaXMpIHtcblx0XHRcdHRoaXMuc2VsZmNvbm5lY3Rpb24ud2VpZ2h0ID0gMTtcblx0XHRcdHJldHVybiB0aGlzLnNlbGZjb25uZWN0aW9uO1xuXHRcdH1cblxuXHRcdC8vIGNoZWNrIGlmIGNvbm5lY3Rpb24gYWxyZWFkeSBleGlzdHNcblx0XHR2YXIgY29ubmVjdGVkID0gdGhpcy5jb25uZWN0ZWQobmV1cm9uKTtcblx0XHRpZiAoY29ubmVjdGVkICYmIGNvbm5lY3RlZC50eXBlID09IFwicHJvamVjdGVkXCIpIHtcblx0XHRcdC8vIHVwZGF0ZSBjb25uZWN0aW9uXG5cdFx0XHRpZiAodHlwZW9mIHdlaWdodCAhPSAndW5kZWZpbmVkJylcblx0XHRcdFx0Y29ubmVjdGVkLmNvbm5lY3Rpb24ud2VpZ2h0ID0gd2VpZ2h0O1xuXHRcdFx0Ly8gcmV0dXJuIGV4aXN0aW5nIGNvbm5lY3Rpb25cblx0XHRcdHJldHVybiBjb25uZWN0ZWQuY29ubmVjdGlvbjtcblx0XHR9IGVsc2Uge1xuXHRcdFx0Ly8gY3JlYXRlIGEgbmV3IGNvbm5lY3Rpb25cblx0XHRcdHZhciBjb25uZWN0aW9uID0gbmV3IE5ldXJvbi5Db25uZWN0aW9uKHRoaXMsIG5ldXJvbiwgd2VpZ2h0KTtcblx0XHR9XG5cblx0XHQvLyByZWZlcmVuY2UgYWxsIHRoZSBjb25uZWN0aW9ucyBhbmQgdHJhY2VzXG5cdFx0dGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbY29ubmVjdGlvbi5JRF0gPSBjb25uZWN0aW9uO1xuXHRcdHRoaXMubmVpZ2hib29yc1tuZXVyb24uSURdID0gbmV1cm9uO1xuXHRcdG5ldXJvbi5jb25uZWN0aW9ucy5pbnB1dHNbY29ubmVjdGlvbi5JRF0gPSBjb25uZWN0aW9uO1xuXHRcdG5ldXJvbi50cmFjZS5lbGVnaWJpbGl0eVtjb25uZWN0aW9uLklEXSA9IDA7XG5cblx0XHRmb3IgKHZhciBpZCBpbiBuZXVyb24udHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdHZhciB0cmFjZSA9IG5ldXJvbi50cmFjZS5leHRlbmRlZFtpZF07XG5cdFx0XHR0cmFjZVtjb25uZWN0aW9uLklEXSA9IDA7XG5cdFx0fVxuXG5cdFx0cmV0dXJuIGNvbm5lY3Rpb247XG5cdH1cblxuXHRnYXRlKGNvbm5lY3Rpb24pIHtcblx0XHQvLyBhZGQgY29ubmVjdGlvbiB0byBnYXRlZCBsaXN0XG5cdFx0dGhpcy5jb25uZWN0aW9ucy5nYXRlZFtjb25uZWN0aW9uLklEXSA9IGNvbm5lY3Rpb247XG5cblx0XHR2YXIgbmV1cm9uID0gY29ubmVjdGlvbi50bztcblx0XHRpZiAoIShuZXVyb24uSUQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkpIHtcblx0XHRcdC8vIGV4dGVuZGVkIHRyYWNlXG5cdFx0XHR0aGlzLm5laWdoYm9vcnNbbmV1cm9uLklEXSA9IG5ldXJvbjtcblx0XHRcdHZhciB4dHJhY2UgPSB0aGlzLnRyYWNlLmV4dGVuZGVkW25ldXJvbi5JRF0gPSB7fTtcblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHRcdHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcblx0XHRcdFx0eHRyYWNlW2lucHV0LklEXSA9IDA7XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFx0Ly8ga2VlcCB0cmFja1xuXHRcdGlmIChuZXVyb24uSUQgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzKVxuXHRcdFx0dGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0ucHVzaChjb25uZWN0aW9uKTtcblx0XHRlbHNlXG5cdFx0XHR0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXSA9IFtjb25uZWN0aW9uXTtcblxuXHRcdC8vIHNldCBnYXRlclxuXHRcdGNvbm5lY3Rpb24uZ2F0ZXIgPSB0aGlzO1xuXHR9XG4gIFxuXHQvLyByZXR1cm5zIHRydWUgb3IgZmFsc2Ugd2hldGhlciB0aGUgbmV1cm9uIGlzIHNlbGYtY29ubmVjdGVkIG9yIG5vdFxuXHRzZWxmY29ubmVjdGVkKCkge1xuXHRcdHJldHVybiB0aGlzLnNlbGZjb25uZWN0aW9uLndlaWdodCAhPT0gMDtcblx0fVxuXG5cdC8vIHJldHVybnMgdHJ1ZSBvciBmYWxzZSB3aGV0aGVyIHRoZSBuZXVyb24gaXMgY29ubmVjdGVkIHRvIGFub3RoZXIgbmV1cm9uIChwYXJhbWV0ZXIpXG5cdGNvbm5lY3RlZChuZXVyb24pIHtcblx0XHR2YXIgcmVzdWx0OiB7XG5cdFx0XHR0eXBlOiBzdHJpbmc7XG5cdFx0XHRjb25uZWN0aW9uOiBOZXVyb24uQ29ubmVjdGlvbjtcblx0XHR9ID0ge1xuXHRcdFx0XHR0eXBlOiBudWxsLFxuXHRcdFx0XHRjb25uZWN0aW9uOiBudWxsXG5cdFx0XHR9O1xuXG5cdFx0aWYgKHRoaXMgPT0gbmV1cm9uKSB7XG5cdFx0XHRpZiAodGhpcy5zZWxmY29ubmVjdGVkKCkpIHtcblx0XHRcdFx0cmVzdWx0LnR5cGUgPSAnc2VsZmNvbm5lY3Rpb24nO1xuXHRcdFx0XHRyZXN1bHQuY29ubmVjdGlvbiA9IHRoaXMuc2VsZmNvbm5lY3Rpb247XG5cdFx0XHRcdHJldHVybiByZXN1bHQ7XG5cdFx0XHR9IGVsc2Vcblx0XHRcdFx0cmV0dXJuIG51bGw7XG5cdFx0fVxuXG5cdFx0Zm9yICh2YXIgdHlwZSBpbiB0aGlzLmNvbm5lY3Rpb25zKSB7XG5cdFx0XHRmb3IgKHZhciBjb25uZWN0aW9uIGluIHRoaXMuY29ubmVjdGlvbnNbdHlwZV0pIHtcblx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSB0aGlzLmNvbm5lY3Rpb25zW3R5cGVdW2Nvbm5lY3Rpb25dO1xuXHRcdFx0XHRpZiAoY29ubmVjdGlvbi50byA9PSBuZXVyb24pIHtcblx0XHRcdFx0XHRyZXN1bHQudHlwZSA9IHR5cGU7XG5cdFx0XHRcdFx0cmVzdWx0LmNvbm5lY3Rpb24gPSBjb25uZWN0aW9uO1xuXHRcdFx0XHRcdHJldHVybiByZXN1bHQ7XG5cdFx0XHRcdH0gZWxzZSBpZiAoY29ubmVjdGlvbi5mcm9tID09IG5ldXJvbikge1xuXHRcdFx0XHRcdHJlc3VsdC50eXBlID0gdHlwZTtcblx0XHRcdFx0XHRyZXN1bHQuY29ubmVjdGlvbiA9IGNvbm5lY3Rpb247XG5cdFx0XHRcdFx0cmV0dXJuIHJlc3VsdDtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdH1cblxuXHRcdHJldHVybiBudWxsO1xuXHR9XG5cblx0Ly8gY2xlYXJzIGFsbCB0aGUgdHJhY2VzICh0aGUgbmV1cm9uIGZvcmdldHMgaXQncyBjb250ZXh0LCBidXQgdGhlIGNvbm5lY3Rpb25zIHJlbWFpbiBpbnRhY3QpXG5cdGNsZWFyKCkge1xuXG5cdFx0Zm9yICh2YXIgdHJhY2UgaW4gdGhpcy50cmFjZS5lbGVnaWJpbGl0eSlcblx0XHRcdHRoaXMudHJhY2UuZWxlZ2liaWxpdHlbdHJhY2VdID0gMDtcblxuXHRcdGZvciAodmFyIHRyYWNlIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpXG5cdFx0XHRmb3IgKHZhciBleHRlbmRlZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkW3RyYWNlXSlcblx0XHRcdFx0dGhpcy50cmFjZS5leHRlbmRlZFt0cmFjZV1bZXh0ZW5kZWRdID0gMDtcblxuXHRcdHRoaXMuZXJyb3IucmVzcG9uc2liaWxpdHkgPSB0aGlzLmVycm9yLnByb2plY3RlZCA9IHRoaXMuZXJyb3IuZ2F0ZWQgPSAwO1xuXHR9XG5cblx0Ly8gYWxsIHRoZSBjb25uZWN0aW9ucyBhcmUgcmFuZG9taXplZCBhbmQgdGhlIHRyYWNlcyBhcmUgY2xlYXJlZFxuXHRyZXNldCgpIHtcblx0XHR0aGlzLmNsZWFyKCk7XG5cblx0XHRmb3IgKHZhciB0eXBlIGluIHRoaXMuY29ubmVjdGlvbnMpXG5cdFx0XHRmb3IgKHZhciBjb25uZWN0aW9uIGluIHRoaXMuY29ubmVjdGlvbnNbdHlwZV0pXG5cdFx0XHRcdHRoaXMuY29ubmVjdGlvbnNbdHlwZV1bY29ubmVjdGlvbl0ud2VpZ2h0ID0gTWF0aC5yYW5kb20oKSAqIC4yIC0gLjE7XG5cdFx0dGhpcy5iaWFzID0gTWF0aC5yYW5kb20oKSAqIC4yIC0gLjE7XG5cblx0XHR0aGlzLm9sZCA9IHRoaXMuc3RhdGUgPSB0aGlzLmFjdGl2YXRpb24gPSAwO1xuXHR9XG5cdFxuXG4gIFxuXG5cdC8vIGhhcmRjb2RlcyB0aGUgYmVoYXZpb3VyIG9mIHRoZSBuZXVyb24gaW50byBhbiBvcHRpbWl6ZWQgZnVuY3Rpb25cblx0b3B0aW1pemUob3B0aW1pemVkLCBsYXllcik6IFN5bmFwdGljLklDb21waWxlZFBhcmFtZXRlcnMge1xuXG5cdFx0b3B0aW1pemVkID0gb3B0aW1pemVkIHx8IHt9O1xuXHRcdHZhciB0aGF0ID0gdGhpcztcblx0XHR2YXIgc3RvcmVfYWN0aXZhdGlvbiA9IFtdO1xuXHRcdHZhciBzdG9yZV90cmFjZSA9IFtdO1xuXHRcdHZhciBzdG9yZV9wcm9wYWdhdGlvbiA9IFtdO1xuXHRcdHZhciB2YXJJRCA9IG9wdGltaXplZC5tZW1vcnkgfHwgMDtcblx0XHR2YXIgbmV1cm9ucyA9IG9wdGltaXplZC5uZXVyb25zIHx8IDE7XG5cdFx0dmFyIGlucHV0cyA9IG9wdGltaXplZC5pbnB1dHMgfHwgW107XG5cdFx0dmFyIHRhcmdldHMgPSBvcHRpbWl6ZWQudGFyZ2V0cyB8fCBbXTtcblx0XHR2YXIgb3V0cHV0cyA9IG9wdGltaXplZC5vdXRwdXRzIHx8IFtdO1xuXHRcdHZhciB2YXJpYWJsZXMgPSBvcHRpbWl6ZWQudmFyaWFibGVzIHx8IHt9O1xuXHRcdHZhciBhY3RpdmF0aW9uX3NlbnRlbmNlcyA9IG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlcyB8fCBbXTtcblx0XHR2YXIgdHJhY2Vfc2VudGVuY2VzID0gb3B0aW1pemVkLnRyYWNlX3NlbnRlbmNlcyB8fCBbXTtcblx0XHR2YXIgcHJvcGFnYXRpb25fc2VudGVuY2VzID0gb3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlcyB8fCBbXTtcblx0XHR2YXIgbGF5ZXJzID0gb3B0aW1pemVkLmxheWVycyB8fCB7IF9fY291bnQ6IDAsIF9fbmV1cm9uOiAwIH07XG5cblx0XHQvLyBhbGxvY2F0ZSBzZW50ZW5jZXNcblx0XHR2YXIgYWxsb2NhdGUgPSBmdW5jdGlvbihzdG9yZSkge1xuXHRcdFx0dmFyIGFsbG9jYXRlZCA9IGxheWVyIGluIGxheWVycyAmJiBzdG9yZVtsYXllcnMuX19jb3VudF07XG5cdFx0XHRpZiAoIWFsbG9jYXRlZCkge1xuXHRcdFx0XHRsYXllcnMuX19jb3VudCA9IHN0b3JlLnB1c2goW10pIC0gMTtcblx0XHRcdFx0bGF5ZXJzW2xheWVyXSA9IGxheWVycy5fX2NvdW50O1xuXHRcdFx0fVxuXHRcdH1cblx0XHRhbGxvY2F0ZShhY3RpdmF0aW9uX3NlbnRlbmNlcyk7XG5cdFx0YWxsb2NhdGUodHJhY2Vfc2VudGVuY2VzKTtcblx0XHRhbGxvY2F0ZShwcm9wYWdhdGlvbl9zZW50ZW5jZXMpO1xuXHRcdHZhciBjdXJyZW50TGF5ZXIgPSBsYXllcnMuX19jb3VudDtcblxuXHRcdC8vIGdldC9yZXNlcnZlIHNwYWNlIGluIG1lbW9yeSBieSBjcmVhdGluZyBhIHVuaXF1ZSBJRCBmb3IgYSB2YXJpYWJsZWxcblx0XHR2YXIgZ2V0VmFyID0gZnVuY3Rpb24oLi4uYXJnczogYW55W10pIHtcblx0XHRcdHZhciBpZDtcblx0XHRcdGlmIChhcmdzLmxlbmd0aCA9PSAxKSB7XG5cblx0XHRcdFx0aWYgKGFyZ3NbMF0gPT0gJ3RhcmdldCcpIHtcblx0XHRcdFx0XHRpZCA9ICd0YXJnZXRfJyArIHRhcmdldHMubGVuZ3RoO1xuXHRcdFx0XHRcdHRhcmdldHMucHVzaCh2YXJJRCk7XG5cdFx0XHRcdH0gZWxzZVxuXHRcdFx0XHRcdGlkID0gYXJnc1swXTtcblx0XHRcdFx0aWYgKGlkIGluIHZhcmlhYmxlcylcblx0XHRcdFx0XHRyZXR1cm4gdmFyaWFibGVzW2lkXTtcblx0XHRcdFx0cmV0dXJuIHZhcmlhYmxlc1tpZF0gPSB7XG5cdFx0XHRcdFx0dmFsdWU6IDAsXG5cdFx0XHRcdFx0aWQ6IHZhcklEKytcblx0XHRcdFx0fTtcblx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdHZhciBleHRlbmRlZCA9IGFyZ3MubGVuZ3RoID4gMjtcblx0XHRcdFx0aWYgKGV4dGVuZGVkKVxuXHRcdFx0XHRcdHZhciB2YWx1ZSA9IGFyZ3MucG9wKCk7XG5cblx0XHRcdFx0dmFyIHVuaXQgPSBhcmdzLnNoaWZ0KCk7XG5cdFx0XHRcdHZhciBwcm9wID0gYXJncy5wb3AoKTtcblxuXHRcdFx0XHRpZiAoIWV4dGVuZGVkKVxuXHRcdFx0XHRcdHZhciB2YWx1ZSA9IHVuaXRbcHJvcF07XG5cblx0XHRcdFx0aWQgPSBwcm9wICsgJ18nO1xuXHRcdFx0XHRmb3IgKHZhciBwcm9wZXJ0eSBpbiBhcmdzKVxuXHRcdFx0XHRcdGlkICs9IGFyZ3NbcHJvcGVydHldICsgJ18nO1xuXHRcdFx0XHRpZCArPSB1bml0LklEO1xuXHRcdFx0XHRpZiAoaWQgaW4gdmFyaWFibGVzKVxuXHRcdFx0XHRcdHJldHVybiB2YXJpYWJsZXNbaWRdO1xuXG5cdFx0XHRcdHJldHVybiB2YXJpYWJsZXNbaWRdID0ge1xuXHRcdFx0XHRcdHZhbHVlOiB2YWx1ZSxcblx0XHRcdFx0XHRpZDogdmFySUQrK1xuXHRcdFx0XHR9O1xuXHRcdFx0fVxuXHRcdH07XG5cblx0XHQvLyBidWlsZCBzZW50ZW5jZVxuXHRcdHZhciBidWlsZFNlbnRlbmNlID0gZnVuY3Rpb24oLi4uYXJnczogYW55W10pIHtcblx0XHRcdHZhciBzdG9yZSA9IGFyZ3MucG9wKCk7XG5cdFx0XHR2YXIgc2VudGVuY2UgPSBcIlwiO1xuXHRcdFx0Zm9yICh2YXIgaSBpbiBhcmdzKVxuXHRcdFx0XHRpZiAodHlwZW9mIGFyZ3NbaV0gPT0gJ3N0cmluZycpXG5cdFx0XHRcdFx0c2VudGVuY2UgKz0gYXJnc1tpXTtcblx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdHNlbnRlbmNlICs9ICdGWycgKyBhcmdzW2ldLmlkICsgJ10nO1xuXG5cdFx0XHRzdG9yZS5wdXNoKHNlbnRlbmNlICsgJzsnKTtcblx0XHR9XG5cblx0XHQvLyBoZWxwZXIgdG8gY2hlY2sgaWYgYW4gb2JqZWN0IGlzIGVtcHR5XG5cdFx0dmFyIGlzRW1wdHkgPSBmdW5jdGlvbihvYmopIHtcblx0XHRcdGZvciAodmFyIHByb3AgaW4gb2JqKSB7XG5cdFx0XHRcdGlmIChvYmouaGFzT3duUHJvcGVydHkocHJvcCkpXG5cdFx0XHRcdFx0cmV0dXJuIGZhbHNlO1xuXHRcdFx0fVxuXHRcdFx0cmV0dXJuIHRydWU7XG5cdFx0fTtcblxuXHRcdC8vIGNoYXJhY3RlcmlzdGljcyBvZiB0aGUgbmV1cm9uXG5cdFx0dmFyIG5vUHJvamVjdGlvbnMgPSBpc0VtcHR5KHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkKTtcblx0XHR2YXIgbm9HYXRlcyA9IGlzRW1wdHkodGhpcy5jb25uZWN0aW9ucy5nYXRlZCk7XG5cdFx0dmFyIGlzSW5wdXQgPSBsYXllciA9PSAnaW5wdXQnID8gdHJ1ZSA6IGlzRW1wdHkodGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpO1xuXHRcdHZhciBpc091dHB1dCA9IGxheWVyID09ICdvdXRwdXQnID8gdHJ1ZSA6IG5vUHJvamVjdGlvbnMgJiYgbm9HYXRlcztcblxuXHRcdC8vIG9wdGltaXplIG5ldXJvbidzIGJlaGF2aW91clxuXHRcdHZhciByYXRlID0gZ2V0VmFyKCdyYXRlJyk7XG5cdFx0dmFyIGFjdGl2YXRpb24gPSBnZXRWYXIodGhpcywgJ2FjdGl2YXRpb24nKTtcblx0XHRpZiAoaXNJbnB1dClcblx0XHRcdGlucHV0cy5wdXNoKGFjdGl2YXRpb24uaWQpO1xuXHRcdGVsc2Uge1xuXHRcdFx0YWN0aXZhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXS5wdXNoKHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0dHJhY2Vfc2VudGVuY2VzW2N1cnJlbnRMYXllcl0ucHVzaChzdG9yZV90cmFjZSk7XG5cdFx0XHRwcm9wYWdhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXS5wdXNoKHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdHZhciBvbGQgPSBnZXRWYXIodGhpcywgJ29sZCcpO1xuXHRcdFx0dmFyIHN0YXRlID0gZ2V0VmFyKHRoaXMsICdzdGF0ZScpO1xuXHRcdFx0dmFyIGJpYXMgPSBnZXRWYXIodGhpcywgJ2JpYXMnKTtcblx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0aW9uLmdhdGVyKVxuXHRcdFx0XHR2YXIgc2VsZl9nYWluID0gZ2V0VmFyKHRoaXMuc2VsZmNvbm5lY3Rpb24sICdnYWluJyk7XG5cdFx0XHRpZiAodGhpcy5zZWxmY29ubmVjdGVkKCkpXG5cdFx0XHRcdHZhciBzZWxmX3dlaWdodCA9IGdldFZhcih0aGlzLnNlbGZjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRidWlsZFNlbnRlbmNlKG9sZCwgJyA9ICcsIHN0YXRlLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0ZWQoKSlcblx0XHRcdFx0aWYgKHRoaXMuc2VsZmNvbm5lY3Rpb24uZ2F0ZXIpXG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShzdGF0ZSwgJyA9ICcsIHNlbGZfZ2FpbiwgJyAqICcsIHNlbGZfd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdHN0YXRlLCAnICsgJywgYmlhcywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHN0YXRlLCAnID0gJywgc2VsZl93ZWlnaHQsICcgKiAnLCBzdGF0ZSwgJyArICcsXG5cdFx0XHRcdFx0XHRiaWFzLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdGVsc2Vcblx0XHRcdFx0YnVpbGRTZW50ZW5jZShzdGF0ZSwgJyA9ICcsIGJpYXMsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0Zm9yICh2YXIgaSBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHR2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpXTtcblx0XHRcdFx0dmFyIGlucHV0X2FjdGl2YXRpb24gPSBnZXRWYXIoaW5wdXQuZnJvbSwgJ2FjdGl2YXRpb24nKTtcblx0XHRcdFx0dmFyIGlucHV0X3dlaWdodCA9IGdldFZhcihpbnB1dCwgJ3dlaWdodCcpO1xuXHRcdFx0XHRpZiAoaW5wdXQuZ2F0ZXIpXG5cdFx0XHRcdFx0dmFyIGlucHV0X2dhaW4gPSBnZXRWYXIoaW5wdXQsICdnYWluJyk7XG5cdFx0XHRcdGlmICh0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpXS5nYXRlcilcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHN0YXRlLCAnICs9ICcsIGlucHV0X2FjdGl2YXRpb24sICcgKiAnLFxuXHRcdFx0XHRcdFx0aW5wdXRfd2VpZ2h0LCAnICogJywgaW5wdXRfZ2Fpbiwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHN0YXRlLCAnICs9ICcsIGlucHV0X2FjdGl2YXRpb24sICcgKiAnLFxuXHRcdFx0XHRcdFx0aW5wdXRfd2VpZ2h0LCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdH1cblx0XHRcdHZhciBkZXJpdmF0aXZlID0gZ2V0VmFyKHRoaXMsICdkZXJpdmF0aXZlJyk7XG5cdFx0XHRzd2l0Y2ggKHRoaXMuc3F1YXNoKSB7XG5cdFx0XHRcdGNhc2UgU3F1YXNoLkxPR0lTVElDOlxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoYWN0aXZhdGlvbiwgJyA9ICgxIC8gKDEgKyBNYXRoLmV4cCgtJywgc3RhdGUsICcpKSknLFxuXHRcdFx0XHRcdFx0c3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShkZXJpdmF0aXZlLCAnID0gJywgYWN0aXZhdGlvbiwgJyAqICgxIC0gJyxcblx0XHRcdFx0XHRcdGFjdGl2YXRpb24sICcpJywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnJlYWs7XG5cdFx0XHRcdGNhc2UgU3F1YXNoLlRBTkg6XG5cdFx0XHRcdFx0dmFyIGVQID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0XHR2YXIgZU4gPSBnZXRWYXIoJ2F1eF8yJyk7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShlUCwgJyA9IE1hdGguZXhwKCcsIHN0YXRlLCAnKScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZU4sICcgPSAxIC8gJywgZVAsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoYWN0aXZhdGlvbiwgJyA9ICgnLCBlUCwgJyAtICcsIGVOLCAnKSAvICgnLCBlUCwgJyArICcsIGVOLCAnKScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZGVyaXZhdGl2ZSwgJyA9IDEgLSAoJywgYWN0aXZhdGlvbiwgJyAqICcsIGFjdGl2YXRpb24sICcpJywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnJlYWs7XG5cdFx0XHRcdGNhc2UgU3F1YXNoLklERU5USVRZOlxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoYWN0aXZhdGlvbiwgJyA9ICcsIHN0YXRlLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGRlcml2YXRpdmUsICcgPSAxJywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnJlYWs7XG5cdFx0XHRcdGNhc2UgU3F1YXNoLkhMSU06XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShhY3RpdmF0aW9uLCAnID0gKygnLCBzdGF0ZSwgJyA+IDApJyxcblx0XHRcdFx0XHRcdHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZGVyaXZhdGl2ZSwgJyA9IDEnLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRicmVhaztcblx0XHRcdH1cblxuXG5cdFx0XHR2YXIgaW5mbHVlbmNlcyA9IFtdO1xuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHQvLyBjYWxjdWxhdGUgZXh0ZW5kZWQgZWxlZ2liaWxpdHkgdHJhY2VzIGluIGFkdmFuY2VcbiAgICAgICAgXG5cdFx0XHRcdHZhciB4dHJhY2UgPSB0aGlzLnRyYWNlLmV4dGVuZGVkW2lkXTtcblx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdHZhciBpbmZsdWVuY2UgPSBnZXRWYXIoJ2F1eCcpO1xuXHRcdFx0XHR2YXIgbmV1cm9uX29sZCA9IGdldFZhcihuZXVyb24sICdvbGQnKTtcblx0XHRcdFx0dmFyIGluaXRpYWxpemVkID0gZmFsc2U7XG5cdFx0XHRcdGlmIChuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIgPT0gdGhpcykge1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnID0gJywgbmV1cm9uX29sZCwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdGluaXRpYWxpemVkID0gdHJ1ZTtcblx0XHRcdFx0fVxuXHRcdFx0XHRmb3IgKHZhciBpbmNvbWluZyBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXSkge1xuXHRcdFx0XHRcdHZhciBpbmNvbWluZ193ZWlnaHQgPSBnZXRWYXIodGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF1cblx0XHRcdFx0XHRbaW5jb21pbmddLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0dmFyIGluY29taW5nX2FjdGl2YXRpb24gPSBnZXRWYXIodGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF1cblx0XHRcdFx0XHRbaW5jb21pbmddLmZyb20sICdhY3RpdmF0aW9uJyk7XG5cblx0XHRcdFx0XHRpZiAoaW5pdGlhbGl6ZWQpXG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyArPSAnLCBpbmNvbWluZ193ZWlnaHQsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRpbmNvbWluZ19hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0ZWxzZSB7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9ICcsIGluY29taW5nX3dlaWdodCwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdGluY29taW5nX2FjdGl2YXRpb24sIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRcdGluaXRpYWxpemVkID0gdHJ1ZTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblxuXHRcdFx0XHRpbmZsdWVuY2VzLnB1c2gobmV1cm9uLklEKTtcblx0XHRcdFx0YnVpbGRTZW50ZW5jZShcImluZmx1ZW5jZXNbXCIgKyAoaW5mbHVlbmNlcy5sZW5ndGggLSAxKSArIFwiXSA9IFwiLCBpbmZsdWVuY2UsIHN0b3JlX3RyYWNlKTtcblx0XHRcdH1cblxuXHRcdFx0Zm9yICh2YXIgaSBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHR2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpXTtcblx0XHRcdFx0aWYgKGlucHV0LmdhdGVyKVxuXHRcdFx0XHRcdHZhciBpbnB1dF9nYWluID0gZ2V0VmFyKGlucHV0LCAnZ2FpbicpO1xuXHRcdFx0XHR2YXIgaW5wdXRfYWN0aXZhdGlvbiA9IGdldFZhcihpbnB1dC5mcm9tLCAnYWN0aXZhdGlvbicpO1xuXHRcdFx0XHR2YXIgdHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2VsZWdpYmlsaXR5JywgaW5wdXQuSUQsIHRoaXMudHJhY2Vcblx0XHRcdFx0XHQuZWxlZ2liaWxpdHlbaW5wdXQuSURdKTtcblx0XHRcdFx0aWYgKHRoaXMuc2VsZmNvbm5lY3RlZCgpKSB7XG5cdFx0XHRcdFx0aWYgKHRoaXMuc2VsZmNvbm5lY3Rpb24uZ2F0ZXIpIHtcblx0XHRcdFx0XHRcdGlmIChpbnB1dC5nYXRlcilcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZSh0cmFjZSwgJyA9ICcsIHNlbGZfZ2FpbiwgJyAqICcsIHNlbGZfd2VpZ2h0LFxuXHRcdFx0XHRcdFx0XHRcdCcgKiAnLCB0cmFjZSwgJyArICcsIGlucHV0X2dhaW4sICcgKiAnLCBpbnB1dF9hY3RpdmF0aW9uLFxuXHRcdFx0XHRcdFx0XHRcdHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZSh0cmFjZSwgJyA9ICcsIHNlbGZfZ2FpbiwgJyAqICcsIHNlbGZfd2VpZ2h0LFxuXHRcdFx0XHRcdFx0XHRcdCcgKiAnLCB0cmFjZSwgJyArICcsIGlucHV0X2FjdGl2YXRpb24sIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRcdFx0aWYgKGlucHV0LmdhdGVyKVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHRyYWNlLCAnID0gJywgc2VsZl93ZWlnaHQsICcgKiAnLCB0cmFjZSwgJyArICcsXG5cdFx0XHRcdFx0XHRcdFx0aW5wdXRfZ2FpbiwgJyAqICcsIGlucHV0X2FjdGl2YXRpb24sIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZSh0cmFjZSwgJyA9ICcsIHNlbGZfd2VpZ2h0LCAnICogJywgdHJhY2UsICcgKyAnLFxuXHRcdFx0XHRcdFx0XHRcdGlucHV0X2FjdGl2YXRpb24sIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdFx0aWYgKGlucHV0LmdhdGVyKVxuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZSh0cmFjZSwgJyA9ICcsIGlucHV0X2dhaW4sICcgKiAnLCBpbnB1dF9hY3RpdmF0aW9uLFxuXHRcdFx0XHRcdFx0XHRzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZSh0cmFjZSwgJyA9ICcsIGlucHV0X2FjdGl2YXRpb24sIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0fVxuXHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdFx0Ly8gZXh0ZW5kZWQgZWxlZ2liaWxpdHkgdHJhY2Vcblx0XHRcdFx0XHR2YXIgeHRyYWNlID0gdGhpcy50cmFjZS5leHRlbmRlZFtpZF07XG5cdFx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdFx0dmFyIGluZmx1ZW5jZSA9IGdldFZhcignYXV4Jyk7XG5cdFx0XHRcdFx0dmFyIG5ldXJvbl9vbGQgPSBnZXRWYXIobmV1cm9uLCAnb2xkJyk7XG5cblx0XHRcdFx0XHR2YXIgdHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2VsZWdpYmlsaXR5JywgaW5wdXQuSUQsIHRoaXMudHJhY2Vcblx0XHRcdFx0XHRcdC5lbGVnaWJpbGl0eVtpbnB1dC5JRF0pO1xuXHRcdFx0XHRcdHZhciB4dHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2V4dGVuZGVkJywgbmV1cm9uLklELCBpbnB1dC5JRCxcblx0XHRcdFx0XHRcdHRoaXMudHJhY2UuZXh0ZW5kZWRbbmV1cm9uLklEXVtpbnB1dC5JRF0pO1xuXHRcdFx0XHRcdGlmIChuZXVyb24uc2VsZmNvbm5lY3RlZCgpKVxuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9zZWxmX3dlaWdodCA9IGdldFZhcihuZXVyb24uc2VsZmNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyKVxuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9zZWxmX2dhaW4gPSBnZXRWYXIobmV1cm9uLnNlbGZjb25uZWN0aW9uLCAnZ2FpbicpO1xuXHRcdFx0XHRcdGlmIChuZXVyb24uc2VsZmNvbm5lY3RlZCgpKVxuXHRcdFx0XHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlcilcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZSh4dHJhY2UsICcgPSAnLCBuZXVyb25fc2VsZl9nYWluLCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHRuZXVyb25fc2VsZl93ZWlnaHQsICcgKiAnLCB4dHJhY2UsICcgKyAnLCBkZXJpdmF0aXZlLCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHR0cmFjZSwgJyAqICcsIFwiaW5mbHVlbmNlc1tcIiArIGluZmx1ZW5jZXMuaW5kZXhPZihuZXVyb24uSUQpICsgXCJdXCIsIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZSh4dHJhY2UsICcgPSAnLCBuZXVyb25fc2VsZl93ZWlnaHQsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdHh0cmFjZSwgJyArICcsIGRlcml2YXRpdmUsICcgKiAnLCB0cmFjZSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0XCJpbmZsdWVuY2VzW1wiICsgaW5mbHVlbmNlcy5pbmRleE9mKG5ldXJvbi5JRCkgKyBcIl1cIiwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoeHRyYWNlLCAnID0gJywgZGVyaXZhdGl2ZSwgJyAqICcsIHRyYWNlLCAnICogJyxcblx0XHRcdFx0XHRcdFx0XCJpbmZsdWVuY2VzW1wiICsgaW5mbHVlbmNlcy5pbmRleE9mKG5ldXJvbi5JRCkgKyBcIl1cIiwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0XHRmb3IgKHZhciBjb25uZWN0aW9uIGluIHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWQpIHtcblx0XHRcdFx0dmFyIGdhdGVkX2dhaW4gPSBnZXRWYXIodGhpcy5jb25uZWN0aW9ucy5nYXRlZFtjb25uZWN0aW9uXSwgJ2dhaW4nKTtcblx0XHRcdFx0YnVpbGRTZW50ZW5jZShnYXRlZF9nYWluLCAnID0gJywgYWN0aXZhdGlvbiwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHR9XG5cdFx0fVxuXHRcdGlmICghaXNJbnB1dCkge1xuXHRcdFx0dmFyIHJlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKHRoaXMsICdlcnJvcicsICdyZXNwb25zaWJpbGl0eScsIHRoaXMuZXJyb3Jcblx0XHRcdFx0LnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdGlmIChpc091dHB1dCkge1xuXHRcdFx0XHR2YXIgdGFyZ2V0ID0gZ2V0VmFyKCd0YXJnZXQnKTtcblx0XHRcdFx0YnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyA9ICcsIHRhcmdldCwgJyAtICcsIGFjdGl2YXRpb24sXG5cdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHRcdHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcblx0XHRcdFx0XHR2YXIgdHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2VsZWdpYmlsaXR5JywgaW5wdXQuSUQsIHRoaXMudHJhY2Vcblx0XHRcdFx0XHRcdC5lbGVnaWJpbGl0eVtpbnB1dC5JRF0pO1xuXHRcdFx0XHRcdHZhciBpbnB1dF93ZWlnaHQgPSBnZXRWYXIoaW5wdXQsICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGlucHV0X3dlaWdodCwgJyArPSAnLCByYXRlLCAnICogKCcsIHJlc3BvbnNpYmlsaXR5LFxuXHRcdFx0XHRcdFx0JyAqICcsIHRyYWNlLCAnKScsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0fVxuXHRcdFx0XHRvdXRwdXRzLnB1c2goYWN0aXZhdGlvbi5pZCk7XG5cdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRpZiAoIW5vUHJvamVjdGlvbnMgJiYgIW5vR2F0ZXMpIHtcblx0XHRcdFx0XHR2YXIgZXJyb3IgPSBnZXRWYXIoJ2F1eCcpO1xuXHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG5cdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkW2lkXTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb24gPSBjb25uZWN0aW9uLnRvO1xuXHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fd2VpZ2h0ID0gZ2V0VmFyKGNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuXHRcdFx0XHRcdFx0XHQncmVzcG9uc2liaWxpdHknLCBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkpO1xuXHRcdFx0XHRcdFx0aWYgKGNvbm5lY3Rpb24uZ2F0ZXIpIHtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fZ2FpbiA9IGdldFZhcihjb25uZWN0aW9uLCAnZ2FpbicpO1xuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVycm9yLCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0Y29ubmVjdGlvbl9nYWluLCAnICogJywgY29ubmVjdGlvbl93ZWlnaHQsXG5cdFx0XHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fSBlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZXJyb3IsICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHRjb25uZWN0aW9uX3dlaWdodCwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHR2YXIgcHJvamVjdGVkID0gZ2V0VmFyKHRoaXMsICdlcnJvcicsICdwcm9qZWN0ZWQnLCB0aGlzLmVycm9yLnByb2plY3RlZCk7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShwcm9qZWN0ZWQsICcgPSAnLCBkZXJpdmF0aXZlLCAnICogJywgZXJyb3IsXG5cdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShlcnJvciwgJyA9IDAnLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdFx0XHR2YXIgaW5mbHVlbmNlID0gZ2V0VmFyKCdhdXhfMicpO1xuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9vbGQgPSBnZXRWYXIobmV1cm9uLCAnb2xkJyk7XG5cdFx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyID09IHRoaXMpXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnID0gJywgbmV1cm9uX29sZCwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9IDAnLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHRmb3IgKHZhciBpbmZsdWVuY2VJbnB1dCBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXSkge1xuXHRcdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdW2luZmx1ZW5jZUlucHV0XTtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fd2VpZ2h0ID0gZ2V0VmFyKGNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9hY3RpdmF0aW9uID0gZ2V0VmFyKGNvbm5lY3Rpb24uZnJvbSwgJ2FjdGl2YXRpb24nKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgKz0gJywgY29ubmVjdGlvbl93ZWlnaHQsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdG5ldXJvbl9hY3RpdmF0aW9uLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcblx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZXJyb3IsICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LCAnICogJyxcblx0XHRcdFx0XHRcdFx0aW5mbHVlbmNlLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHRcdHZhciBnYXRlZCA9IGdldFZhcih0aGlzLCAnZXJyb3InLCAnZ2F0ZWQnLCB0aGlzLmVycm9yLmdhdGVkKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGdhdGVkLCAnID0gJywgZGVyaXZhdGl2ZSwgJyAqICcsIGVycm9yLFxuXHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgPSAnLCBwcm9qZWN0ZWQsICcgKyAnLCBnYXRlZCxcblx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaWRdO1xuXHRcdFx0XHRcdFx0dmFyIGdyYWRpZW50ID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0XHRcdHZhciB0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZWxlZ2liaWxpdHknLCBpbnB1dC5JRCwgdGhpc1xuXHRcdFx0XHRcdFx0XHQudHJhY2UuZWxlZ2liaWxpdHlbaW5wdXQuSURdKTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZ3JhZGllbnQsICcgPSAnLCBwcm9qZWN0ZWQsICcgKiAnLCB0cmFjZSxcblx0XHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG5cdFx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdFx0dmFyIHh0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZXh0ZW5kZWQnLCBuZXVyb24uSUQsXG5cdFx0XHRcdFx0XHRcdFx0aW5wdXQuSUQsIHRoaXMudHJhY2UuZXh0ZW5kZWRbbmV1cm9uLklEXVtpbnB1dC5JRF0pO1xuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGdyYWRpZW50LCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0eHRyYWNlLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHR2YXIgaW5wdXRfd2VpZ2h0ID0gZ2V0VmFyKGlucHV0LCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGlucHV0X3dlaWdodCwgJyArPSAnLCByYXRlLCAnICogJywgZ3JhZGllbnQsXG5cdFx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHR9XG5cblx0XHRcdFx0fSBlbHNlIGlmIChub0dhdGVzKSB7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyA9IDAnLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpIHtcblx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uID0gdGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbaWRdO1xuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24udG87XG5cdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbl93ZWlnaHQgPSBnZXRWYXIoY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG5cdFx0XHRcdFx0XHRcdCdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRcdFx0XHRpZiAoY29ubmVjdGlvbi5nYXRlcikge1xuXHRcdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbl9nYWluID0gZ2V0VmFyKGNvbm5lY3Rpb24sICdnYWluJyk7XG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LFxuXHRcdFx0XHRcdFx0XHRcdCcgKiAnLCBjb25uZWN0aW9uX2dhaW4sICcgKiAnLCBjb25uZWN0aW9uX3dlaWdodCxcblx0XHRcdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHR9IGVsc2Vcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksXG5cdFx0XHRcdFx0XHRcdFx0JyAqICcsIGNvbm5lY3Rpb25fd2VpZ2h0LCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgKj0gJywgZGVyaXZhdGl2ZSxcblx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaWRdO1xuXHRcdFx0XHRcdFx0dmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzXG5cdFx0XHRcdFx0XHRcdC50cmFjZS5lbGVnaWJpbGl0eVtpbnB1dC5JRF0pO1xuXHRcdFx0XHRcdFx0dmFyIGlucHV0X3dlaWdodCA9IGdldFZhcihpbnB1dCwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbnB1dF93ZWlnaHQsICcgKz0gJywgcmF0ZSwgJyAqICgnLFxuXHRcdFx0XHRcdFx0XHRyZXNwb25zaWJpbGl0eSwgJyAqICcsIHRyYWNlLCAnKScsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH0gZWxzZSBpZiAobm9Qcm9qZWN0aW9ucykge1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHRcdFx0dmFyIGluZmx1ZW5jZSA9IGdldFZhcignYXV4Jyk7XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX29sZCA9IGdldFZhcihuZXVyb24sICdvbGQnKTtcblx0XHRcdFx0XHRcdGlmIChuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIgPT0gdGhpcylcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAnLCBuZXVyb25fb2xkLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdGZvciAodmFyIGluZmx1ZW5jZUlucHV0IGluIHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdKSB7XG5cdFx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uID0gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF1baW5mbHVlbmNlSW5wdXRdO1xuXHRcdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbl93ZWlnaHQgPSBnZXRWYXIoY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdFx0XHR2YXIgbmV1cm9uX2FjdGl2YXRpb24gPSBnZXRWYXIoY29ubmVjdGlvbi5mcm9tLCAnYWN0aXZhdGlvbicpO1xuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyArPSAnLCBjb25uZWN0aW9uX3dlaWdodCwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0bmV1cm9uX2FjdGl2YXRpb24sIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdH1cblx0XHRcdFx0XHRcdHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuXHRcdFx0XHRcdFx0XHQncmVzcG9uc2liaWxpdHknLCBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkpO1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksXG5cdFx0XHRcdFx0XHRcdCcgKiAnLCBpbmZsdWVuY2UsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShyZXNwb25zaWJpbGl0eSwgJyAqPSAnLCBkZXJpdmF0aXZlLFxuXHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHRcdFx0XHR2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cdFx0XHRcdFx0XHR2YXIgZ3JhZGllbnQgPSBnZXRWYXIoJ2F1eCcpO1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShncmFkaWVudCwgJyA9IDAnLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcblx0XHRcdFx0XHRcdFx0XHQncmVzcG9uc2liaWxpdHknLCBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkpO1xuXHRcdFx0XHRcdFx0XHR2YXIgeHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdleHRlbmRlZCcsIG5ldXJvbi5JRCxcblx0XHRcdFx0XHRcdFx0XHRpbnB1dC5JRCwgdGhpcy50cmFjZS5leHRlbmRlZFtuZXVyb24uSURdW2lucHV0LklEXSk7XG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZ3JhZGllbnQsICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHR4dHJhY2UsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdH1cblx0XHRcdFx0XHRcdHZhciBpbnB1dF93ZWlnaHQgPSBnZXRWYXIoaW5wdXQsICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5wdXRfd2VpZ2h0LCAnICs9ICcsIHJhdGUsICcgKiAnLCBncmFkaWVudCxcblx0XHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdFx0YnVpbGRTZW50ZW5jZShiaWFzLCAnICs9ICcsIHJhdGUsICcgKiAnLCByZXNwb25zaWJpbGl0eSxcblx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdH1cblx0XHRyZXR1cm4ge1xuXHRcdFx0bWVtb3J5OiB2YXJJRCxcblx0XHRcdG5ldXJvbnM6IG5ldXJvbnMgKyAxLFxuXHRcdFx0aW5wdXRzOiBpbnB1dHMsXG5cdFx0XHRvdXRwdXRzOiBvdXRwdXRzLFxuXHRcdFx0dGFyZ2V0czogdGFyZ2V0cyxcblx0XHRcdHZhcmlhYmxlczogdmFyaWFibGVzLFxuXHRcdFx0YWN0aXZhdGlvbl9zZW50ZW5jZXM6IGFjdGl2YXRpb25fc2VudGVuY2VzLFxuXHRcdFx0dHJhY2Vfc2VudGVuY2VzOiB0cmFjZV9zZW50ZW5jZXMsXG5cdFx0XHRwcm9wYWdhdGlvbl9zZW50ZW5jZXM6IHByb3BhZ2F0aW9uX3NlbnRlbmNlcyxcblx0XHRcdGxheWVyczogbGF5ZXJzXG5cdFx0fVxuXHR9XG59XG5cbmV4cG9ydCBtb2R1bGUgTmV1cm9uIHtcblxuXHRleHBvcnQgaW50ZXJmYWNlIElOZXVyb25Db25uZWN0aW9ucyB7XG5cdFx0aW5wdXRzOiBTeW5hcHRpYy5EaWN0aW9uYXJ5PE5ldXJvbi5Db25uZWN0aW9uPjtcblx0XHRwcm9qZWN0ZWQ6IHt9O1xuXHRcdGdhdGVkOiB7fTtcblx0fVxuXG5cdGV4cG9ydCBjbGFzcyBDb25uZWN0aW9uIHtcblx0XHRJRCA9IENvbm5lY3Rpb24udWlkKCk7XG5cdFx0ZnJvbTtcblx0XHR0bztcblx0XHRnYWluOiBudW1iZXIgPSAxO1xuXHRcdHdlaWdodDogbnVtYmVyID0gMDtcblx0XHRnYXRlcjogYW55ID0gbnVsbDtcblx0XHRjb25zdHJ1Y3Rvcihmcm9tLCB0bywgd2VpZ2h0PzogbnVtYmVyKSB7XG5cdFx0XHRpZiAoIWZyb20gfHwgIXRvKVxuXHRcdFx0XHR0aHJvdyBcIkNvbm5lY3Rpb24gRXJyb3I6IEludmFsaWQgbmV1cm9uc1wiO1xuXHRcdFx0dGhpcy5mcm9tID0gZnJvbTtcblx0XHRcdHRoaXMudG8gPSB0bztcblx0XHRcdHRoaXMud2VpZ2h0ID0gdHlwZW9mIHdlaWdodCA9PSAndW5kZWZpbmVkJyB8fCBpc05hTih3ZWlnaHQpID8gTWF0aC5yYW5kb20oKSAqIC4yIC0gLjEgOlxuXHRcdFx0XHR3ZWlnaHQ7XG5cdFx0fVxuXHR9XG5cblx0ZXhwb3J0IHZhciBuZXVyb25RdHkgPSAwO1xuXHRleHBvcnQgZnVuY3Rpb24gdWlkKCk6IG51bWJlciB7XG5cdFx0cmV0dXJuIG5ldXJvblF0eSsrO1xuXHR9XG5cblx0ZXhwb3J0IGZ1bmN0aW9uIHF1YW50aXR5KCkge1xuXHRcdHJldHVybiB7XG5cdFx0XHRuZXVyb25zOiBuZXVyb25RdHksXG5cdFx0XHRjb25uZWN0aW9uczogQ29ubmVjdGlvbi5jb25uZWN0aW9uUXR5XG5cdFx0fVxuXHR9XG59XG5cbmV4cG9ydCBtb2R1bGUgTmV1cm9uLkNvbm5lY3Rpb24ge1xuXHRleHBvcnQgdmFyIGNvbm5lY3Rpb25RdHkgPSAwO1xuXHRleHBvcnQgZnVuY3Rpb24gdWlkKCk6IG51bWJlciB7XG5cdFx0cmV0dXJuIGNvbm5lY3Rpb25RdHkrKztcblx0fVxufSJdLCJzb3VyY2VSb290IjoiL3NvdXJjZS8ifQ==