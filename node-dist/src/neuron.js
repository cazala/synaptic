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
        this.state = this.selfconnection.gain * this.selfconnection.weight * this.state + this.bias;
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

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9uZXVyb24udHMiXSwibmFtZXMiOlsiTmV1cm9uIiwiTmV1cm9uLmNvbnN0cnVjdG9yIiwiTmV1cm9uLmFjdGl2YXRlIiwiTmV1cm9uLnByb3BhZ2F0ZSIsIk5ldXJvbi5wcm9qZWN0IiwiTmV1cm9uLmdhdGUiLCJOZXVyb24uc2VsZmNvbm5lY3RlZCIsIk5ldXJvbi5jb25uZWN0ZWQiLCJOZXVyb24uY2xlYXIiLCJOZXVyb24ucmVzZXQiLCJOZXVyb24ub3B0aW1pemUiLCJOZXVyb24uQ29ubmVjdGlvbiIsIk5ldXJvbi5Db25uZWN0aW9uLmNvbnN0cnVjdG9yIiwiTmV1cm9uLnVpZCIsIk5ldXJvbi5xdWFudGl0eSIsIk5ldXJvbi5Db25uZWN0aW9uLnVpZCJdLCJtYXBwaW5ncyI6IkFBQUEsb0NBQW9DO0FBR3BDLElBQU8sTUFBTSxXQUFXLFVBQVUsQ0FBQyxDQUFDO0FBRXBDLEFBVUE7OzRGQVI0RjtBQUU1Rjs7OztFQUlFO0lBRVcsTUFBTTtJQUFuQkEsU0FBYUEsTUFBTUE7UUFDbEJDLE9BQUVBLEdBQUdBLE1BQU1BLENBQUNBLEdBQUdBLEVBQUVBLENBQUNBO1FBQ2xCQSxVQUFLQSxHQUFHQSxJQUFJQSxDQUFDQTtRQUNiQSxnQkFBV0EsR0FBOEJBO1lBQ3hDQSxNQUFNQSxFQUFFQSxFQUFFQTtZQUNWQSxTQUFTQSxFQUFFQSxFQUFFQTtZQUNiQSxLQUFLQSxFQUFFQSxFQUFFQTtTQUNUQSxDQUFDQTtRQUNGQSxVQUFLQSxHQUFHQTtZQUNQQSxjQUFjQSxFQUFFQSxDQUFDQTtZQUNqQkEsU0FBU0EsRUFBRUEsQ0FBQ0E7WUFDWkEsS0FBS0EsRUFBRUEsQ0FBQ0E7U0FDUkEsQ0FBQ0E7UUFDRkEsVUFBS0EsR0FBR0E7WUFDUEEsV0FBV0EsRUFBRUEsRUFBRUE7WUFDZkEsUUFBUUEsRUFBRUEsRUFBRUE7WUFDWkEsVUFBVUEsRUFBRUEsRUFBRUE7U0FDZEEsQ0FBQ0E7UUFDRkEsVUFBS0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7UUFDVkEsUUFBR0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7UUFDUkEsZUFBVUEsR0FBR0EsQ0FBQ0EsQ0FBQ0E7UUFDZkEsbUJBQWNBLEdBQUdBLElBQUlBLE1BQU1BLENBQUNBLFVBQVVBLENBQUNBLElBQUlBLEVBQUVBLElBQUlBLEVBQUVBLENBQUNBLENBQUNBLENBQUNBLENBQUNBLDhCQUE4QkE7UUFDckZBLFdBQU1BLEdBQUdBLE1BQU1BLENBQUNBLFFBQVFBLENBQUNBO1FBQ3pCQSxlQUFVQSxHQUFHQSxFQUFFQSxDQUFDQTtRQUNoQkEsU0FBSUEsR0FBR0EsSUFBSUEsQ0FBQ0EsTUFBTUEsRUFBRUEsR0FBR0EsRUFBRUEsR0FBR0EsRUFBRUEsQ0FBQ0E7UUFDL0JBLGVBQVVBLEdBQUdBLENBQUNBLENBQUNBO0lBc3NCaEJBLENBQUNBO0lBcHNCQUQsc0JBQXNCQTtJQUN0QkEseUJBQVFBLEdBQVJBLFVBQVNBLEtBQWNBO1FBQ3RCRSxBQUNBQSxpREFEaURBO1FBQ2pEQSxFQUFFQSxDQUFDQSxDQUFDQSxPQUFPQSxLQUFLQSxJQUFJQSxXQUFXQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUNqQ0EsSUFBSUEsQ0FBQ0EsVUFBVUEsR0FBR0EsS0FBS0EsQ0FBQ0E7WUFDeEJBLElBQUlBLENBQUNBLFVBQVVBLEdBQUdBLENBQUNBLENBQUNBO1lBQ3BCQSxJQUFJQSxDQUFDQSxJQUFJQSxHQUFHQSxDQUFDQSxDQUFDQTtZQUNkQSxNQUFNQSxDQUFDQSxJQUFJQSxDQUFDQSxVQUFVQSxDQUFDQTtRQUN4QkEsQ0FBQ0E7UUFFREEsQUFDQUEsWUFEWUE7UUFDWkEsSUFBSUEsQ0FBQ0EsR0FBR0EsR0FBR0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0E7UUFFdEJBLEFBQ0FBLFNBRFNBO1FBQ1RBLElBQUlBLENBQUNBLEtBQUtBLEdBQUdBLElBQUlBLENBQUNBLGNBQWNBLENBQUNBLElBQUlBLEdBQUdBLElBQUlBLENBQUNBLGNBQWNBLENBQUNBLE1BQU1BLEdBQ2xFQSxJQUFJQSxDQUFDQSxLQUFLQSxHQUFHQSxJQUFJQSxDQUFDQSxJQUFJQSxDQUFDQTtRQUV2QkEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsSUFBSUEsSUFBSUEsQ0FBQ0EsV0FBV0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDdkNBLElBQUlBLFFBQVFBLEdBQUdBLElBQUlBLENBQUNBLFdBQVdBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBLENBQUNBO1lBQzFDQSxJQUFJQSxDQUFDQSxLQUFLQSxJQUFJQSxRQUFRQSxDQUFDQSxJQUFJQSxDQUFDQSxVQUFVQSxHQUFHQSxRQUFRQSxDQUFDQSxNQUFNQSxHQUFHQSxRQUFRQSxDQUFDQSxJQUFJQSxDQUFDQTtRQUMxRUEsQ0FBQ0E7UUFFREEsQUFDQUEsU0FEU0E7UUFDVEEsSUFBSUEsQ0FBQ0EsVUFBVUEsR0FBR0EsSUFBSUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsQ0FBQ0E7UUFFMUNBLEFBQ0FBLFFBRFFBO1FBQ1JBLElBQUlBLENBQUNBLFVBQVVBLEdBQUdBLElBQUlBLENBQUNBLE1BQU1BLENBQUNBLElBQUlBLENBQUNBLEtBQUtBLEVBQUVBLElBQUlBLENBQUNBLENBQUNBO1FBRWhEQSxBQUNBQSxnQkFEZ0JBO1lBQ1pBLFVBQVVBLEdBQUdBLEVBQUVBLENBQUNBO1FBQ3BCQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxFQUFFQSxJQUFJQSxJQUFJQSxDQUFDQSxLQUFLQSxDQUFDQSxRQUFRQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUNwQ0EsQUFDQUEsNkJBRDZCQTtnQkFDekJBLE1BQU1BLEdBQUdBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFFBQVFBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBQ3JDQSxJQUFJQSxNQUFNQSxHQUFHQSxJQUFJQSxDQUFDQSxVQUFVQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQTtZQUVqQ0EsQUFDQUEsOEdBRDhHQTtnQkFDMUdBLFNBQVNBLEdBQUdBLE1BQU1BLENBQUNBLGNBQWNBLENBQUNBLEtBQUtBLElBQUlBLElBQUlBLEdBQUdBLE1BQU1BLENBQUNBLEdBQUdBLEdBQUdBLENBQUNBLENBQUNBO1lBR3JFQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxRQUFRQSxJQUFJQSxJQUFJQSxDQUFDQSxLQUFLQSxDQUFDQSxVQUFVQSxDQUFDQSxNQUFNQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDdkRBLFNBQVNBLElBQUlBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFVBQVVBLENBQUNBLE1BQU1BLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLFFBQVFBLENBQUNBLENBQUNBLE1BQU1BLEdBQzlEQSxJQUFJQSxDQUFDQSxLQUFLQSxDQUFDQSxVQUFVQSxDQUFDQSxNQUFNQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxRQUFRQSxDQUFDQSxDQUFDQSxJQUFJQSxDQUFDQSxVQUFVQSxDQUFDQTtZQUM1REEsQ0FBQ0E7WUFDREEsVUFBVUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsU0FBU0EsQ0FBQ0E7UUFDbkNBLENBQUNBO1FBRURBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLElBQUlBLElBQUlBLENBQUNBLFdBQVdBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBO1lBQ3ZDQSxJQUFJQSxRQUFRQSxHQUFHQSxJQUFJQSxDQUFDQSxXQUFXQSxDQUFDQSxNQUFNQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUUxQ0EsQUFDQUEsNkJBRDZCQTtZQUM3QkEsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsV0FBV0EsQ0FBQ0EsUUFBUUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsSUFBSUEsQ0FBQ0EsY0FBY0EsQ0FBQ0EsSUFBSUEsR0FBR0EsSUFBSUEsQ0FBQ0EsY0FBY0EsQ0FDbkZBLE1BQU1BLEdBQUdBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFdBQVdBLENBQUNBLFFBQVFBLENBQUNBLEVBQUVBLENB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QUFTQSxHQUFHQSxJQUFJQSxDQUFDQSxVQUFVQSxHQUFHQSxLQUFLQSxDQUFDQTtZQUUvQ0EsS0FBS0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7WUFFVkEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsSUFBSUEsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsUUFBUUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ3BDQSxJQUFJQSxNQUFNQSxHQUFHQSxJQUFJQSxDQUFDQSxVQUFVQSxDQUFDQSxFQUFFQSxDQUFDQSxFQUFFQSxlQUFlQTtnQkFDakRBLElBQUlBLFNBQVNBLEdBQUdBLE1BQU1BLENBQUNBLGNBQWNBLENBQUNBLEtBQUtBLElBQUlBLElBQUlBLEdBQUdBLE1BQU1BLENBQUNBLEdBQUdBLEdBQUdBLENBQUNBLEVBQUVBLDJEQUEyREE7Z0JBR2pJQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxLQUFLQSxJQUFJQSxJQUFJQSxDQUFDQSxLQUFLQSxDQUFDQSxVQUFVQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtvQkFDN0NBLFNBQVNBLElBQUlBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFVBQVVBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLEtBQUtBLENBQUNBLENBQUNBLE1BQU1BLEdBQUdBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFVBQVVBLENBQzVFQSxNQUFNQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQSxJQUFJQSxDQUFDQSxVQUFVQSxDQUFDQTtnQkFDbkNBLENBQUNBO2dCQUNEQSxBQUNBQSxTQURTQTtnQkFDVEEsS0FBS0EsSUFBSUEsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsY0FBY0EsR0FBR0EsU0FBU0EsQ0FBQ0E7WUFDbERBLENBQUNBO1lBRURBLEFBQ0FBLDZCQUQ2QkE7WUFDN0JBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLEtBQUtBLEdBQUdBLElBQUlBLENBQUNBLFVBQVVBLEdBQUdBLEtBQUtB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QUFDQTtRQUM5REEsQ0FBQ0E7UUFFREEsQUFDQUEsMkNBRDJDQTtRQUMzQ0EsSUFBSUEsQ0FBQ0EsV0FBV0EsQ0FBQ0EsU0FBU0EsQ0FBQ0EsVUFBVUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsVUFBVUEsQ0FBQ0E7UUFDdkRBLElBQUlBLENBQUNBLFVBQVVBLENBQUNBLE1BQU1BLENBQUNBLEVBQUVBLENBQUNBLEdBQUdBLE1BQU1BLENBQUNBO1FBQ3BDQSxNQUFNQSxDQUFDQSxXQUFXQSxDQUFDQSxNQUFNQSxDQUFDQSxVQUFVQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxVQUFVQSxDQUFDQTtRQUN0REEsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsV0FBV0EsQ0FBQ0EsVUFBVUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7UUFFNUNBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLEVBQUVBLElBQUlBLE1BQU1BLENBQUNBLEtBQUtBLENBQUNBLFFBQVFBLENBQUNBLENBQUNBLENBQUNBO1lBQ3RDQSxJQUFJQSxLQUFLQSxHQUFHQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxRQUFRQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQTtZQUN0Q0EsS0FBS0EsQ0FBQ0EsVUFBVUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7UUFDMUJBLENBQUNBO1FBRURBLE1BQU1BLENBQUNBLFVBQVVBLENBQUNBO0lBQ25CQSxDQUFDQTtJQUVESixxQkFBSUEsR0FBSkEsVUFBS0EsVUFBVUE7UUFDZEssQUFDQUEsK0JBRCtCQTtRQUMvQkEsSUFBSUEsQ0FBQ0EsV0FBV0EsQ0FBQ0EsS0FBS0EsQ0FBQ0EsVUFBVUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsVUFBVUEsQ0FBQ0E7UUFFbkRBLElBQUlBLE1BQU1BLEdBQUdBLFVBQVVBLENBQUNBLEVBQUVBLENBQUNBO1FBQzNCQSxFQUFFQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxNQUFNQSxDQUFDQSxFQUFFQSxJ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SUEsQ0FBQ0EsVUFBVUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0E7d0JBQ2pDQSxJQUFJQSxTQUFTQSxHQUFHQSxNQUFNQSxDQUFDQSxPQUFPQSxDQUFDQSxDQUFDQTt3QkFDaENBLElBQUlBLFVBQVVBLEdBQUdBLE1BQU1BLENBQUNBLE1BQU1BLEVBQUVBLEtBQUtBLENBQUNBLENBQUNBO3dCQUN2Q0EsRUFBRUEsQ0FBQ0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsY0FBY0EsQ0FBQ0EsS0FBS0EsSUFBSUEsSUFBSUEsQ0FBQ0E7NEJBQ3ZDQSxhQUFhQSxDQUFDQSxTQUFTQSxFQUFFQSxLQUFLQSxFQUFFQSxVQUFVQSxFQUFFQSxpQkFBaUJBLENBQUNBLENBQUNBO3dCQUNoRUEsSUFBSUE7NEJBQ0hBLGFBQWFBLENBQUNBLFNBQVNBLEVBQUVBLE1BQU1BLEVBQUVBLGlCQUFpQkEsQ0FBQ0EsQ0FBQ0E7d0JBQ3JEQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxjQUFjQSxJQUFJQSxJQUFJQSxDQUFDQSxLQUFLQSxDQUFDQSxVQUFVQSxDQUFDQSxNQUFNQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTs0QkFDN0RBLElBQUlBLFVBQVVBLEdBQUdBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFVBQVVBLENBQUNBLE1BQU1BLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLGNBQWNBLENBQUNBLENBQUNBOzRCQUNsRUEsSUFBSUEsaUJBQWlCQSxHQUFHQSxNQUFNQSxDQUFDQSxVQUFVQSxFQUFFQSxRQUFRQSxDQUFDQSxDQUFDQTs0QkFDckRBLElBQUlBLGlCQUFpQkEsR0FBR0EsTUFBTUEsQ0FBQ0EsVUFBVUEsQ0FBQ0EsSUFBSUEsRUFBRUEsWUFBWUEsQ0FBQ0EsQ0FBQ0E7NEJBQzlEQSxhQUFhQSxDQUFDQSxTQUFTQSxFQUFFQSxNQUFNQSxFQUFFQSxpQkFBaUJBLEVBQUVBLEtBQUtBLEVBQ3hEQSxpQkFBaUJBLEVBQUVBLGlCQUFpQkEsQ0FBQ0EsQ0FBQ0E7d0JBQ3hDQSxDQUFDQTt3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QSxDQWZBWCxBQWVDVyxJQUFBWDtJQWZZQSxpQkFBVUEsR0FBVkEsVUFlWkEsQ0FBQUE7SUFFVUEsZ0JBQVNBLEdBQUdBLENBQUNBLENBQUNBO0lBQ3pCQSxTQUFnQkEsR0FBR0E7UUFDbEJhLE1BQU1BLENBQUNBLGdCQUFTQSxFQUFFQSxDQUFDQTtJQUNwQkEsQ0FBQ0E7SUFGZWIsVUFBR0EsR0FBSEEsR0FFZkEsQ0FBQUE7SUFFREEsU0FBZ0JBLFFBQVFBO1FBQ3ZCYyxNQUFNQSxDQUFDQTtZQUNOQSxPQUFPQSxFQUFFQSxnQkFBU0E7WUFDbEJBLFdBQVdBLEVBQUVBLFVBQVVBLENBQUNBLGFBQWFBO1NBQ3JDQSxDQUFBQTtJQUNGQSxDQUFDQTtJQUxlZCxlQUFRQSxHQUFSQSxRQUtmQSxDQUFBQTtBQUNGQSxDQUFDQSxFQXBDYSxNQUFNLEdBQU4sY0FBTSxLQUFOLGNBQU0sUUFvQ25CO0FBRUQsSUFBYyxNQUFNLENBS25CO0FBTEQsV0FBYyxNQUFNO0lBQUNBLElBQUFBLFVBQVVBLENBSzlCQTtJQUxvQkEsV0FBQUEsVUFBVUEsRUFBQ0EsQ0FBQ0E7UUFDckJXLHdCQUFhQSxHQUFHQSxDQUFDQSxDQUFDQTtRQUM3QkEsU0FBZ0JBLEdBQUdBO1lBQ2xCSSxNQUFNQSxDQUFDQSx3QkFBYUEsRUFBRUEsQ0FBQ0E7UUFDeEJBLENBQUNBO1FBRmVKLGNBQUdBLEdBQUhBLEdBRWZBLENBQUFBO0lBQ0ZBLENBQUNBLEVBTG9CWCxVQUFVQSxHQUFWQSxpQkFBVUEsS0FBVkEsaUJBQVVBLFFBSzlCQTtBQUFEQSxDQUFDQSxFQUxhLE1BQU0sR0FBTixjQUFNLEtBQU4sY0FBTSxRQUtuQiIsImZpbGUiOiJzcmMvbmV1cm9uLmpzIiwic291cmNlc0NvbnRlbnQiOlsiLy8vIDxyZWZlcmVuY2UgcGF0aD1cInN5bmFwdGljLnRzXCIgLz5cblxuaW1wb3J0IFN5bmFwdGljID0gcmVxdWlyZSgnLi9zeW5hcHRpYycpO1xuaW1wb3J0IFNxdWFzaCA9IHJlcXVpcmUoJy4vc3F1YXNoJyk7XG5cbi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgTkVVUk9OXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuXG4vKiBUUyBDSEFOR0VTOlxuXG5cdE5vdyBOZXVyb24uY29ubmVjdGVkKG5ldXJvbikgcmV0dXJucyBudWxsIGluc3RlYWQgb2YgZmFsc2VcblxuKi9cblxuZXhwb3J0IGNsYXNzIE5ldXJvbiB7XG5cdElEID0gTmV1cm9uLnVpZCgpO1xuXHRsYWJlbCA9IG51bGw7XG5cdGNvbm5lY3Rpb25zOiBOZXVyb24uSU5ldXJvbkNvbm5lY3Rpb25zID0ge1xuXHRcdGlucHV0czoge30sXG5cdFx0cHJvamVjdGVkOiB7fSxcblx0XHRnYXRlZDoge31cblx0fTtcblx0ZXJyb3IgPSB7XG5cdFx0cmVzcG9uc2liaWxpdHk6IDAsXG5cdFx0cHJvamVjdGVkOiAwLFxuXHRcdGdhdGVkOiAwXG5cdH07XG5cdHRyYWNlID0ge1xuXHRcdGVsZWdpYmlsaXR5OiB7fSxcblx0XHRleHRlbmRlZDoge30sXG5cdFx0aW5mbHVlbmNlczoge31cblx0fTtcblx0c3RhdGUgPSAwO1xuXHRvbGQgPSAwO1xuXHRhY3RpdmF0aW9uID0gMDtcblx0c2VsZmNvbm5lY3Rpb24gPSBuZXcgTmV1cm9uLkNvbm5lY3Rpb24odGhpcywgdGhpcywgMCk7IC8vIHdlaWdodCA9IDAgLT4gbm90IGNvbm5lY3RlZFxuXHRzcXVhc2ggPSBTcXVhc2guTE9HSVNUSUM7XG5cdG5laWdoYm9vcnMgPSB7fTtcblx0YmlhcyA9IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xO1xuXHRkZXJpdmF0aXZlID0gMDtcbiAgXG5cdC8vIGFjdGl2YXRlIHRoZSBuZXVyb25cblx0YWN0aXZhdGUoaW5wdXQ/OiBudW1iZXIpIHtcblx0XHQvLyBhY3RpdmF0aW9uIGZyb20gZW52aXJvbWVudCAoZm9yIGlucHV0IG5ldXJvbnMpXG5cdFx0aWYgKHR5cGVvZiBpbnB1dCAhPSAndW5kZWZpbmVkJykge1xuXHRcdFx0dGhpcy5hY3RpdmF0aW9uID0gaW5wdXQ7XG5cdFx0XHR0aGlzLmRlcml2YXRpdmUgPSAwO1xuXHRcdFx0dGhpcy5iaWFzID0gMDtcblx0XHRcdHJldHVybiB0aGlzLmFjdGl2YXRpb247XG5cdFx0fVxuXG5cdFx0Ly8gb2xkIHN0YXRlXG5cdFx0dGhpcy5vbGQgPSB0aGlzLnN0YXRlO1xuXG5cdFx0Ly8gZXEuIDE1XG5cdFx0dGhpcy5zdGF0ZSA9IHRoaXMuc2VsZmNvbm5lY3Rpb24uZ2FpbiAqIHRoaXMuc2VsZmNvbm5lY3Rpb24ud2VpZ2h0ICpcblx0XHR0aGlzLnN0YXRlICsgdGhpcy5iaWFzO1xuXG5cdFx0Zm9yICh2YXIgaSBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0dmFyIHRoZUlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cdFx0XHR0aGlzLnN0YXRlICs9IHRoZUlucHV0LmZyb20uYWN0aXZhdGlvbiAqIHRoZUlucHV0LndlaWdodCAqIHRoZUlucHV0LmdhaW47XG5cdFx0fVxuXG5cdFx0Ly8gZXEuIDE2XG5cdFx0dGhpcy5hY3RpdmF0aW9uID0gdGhpcy5zcXVhc2godGhpcy5zdGF0ZSk7XG5cblx0XHQvLyBmJyhzKVxuXHRcdHRoaXMuZGVyaXZhdGl2ZSA9IHRoaXMuc3F1YXNoKHRoaXMuc3RhdGUsIHRydWUpO1xuXG5cdFx0Ly8gdXBkYXRlIHRyYWNlc1xuXHRcdHZhciBpbmZsdWVuY2VzID0gW107XG5cdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0Ly8gZXh0ZW5kZWQgZWxlZ2liaWxpdHkgdHJhY2Vcblx0XHRcdHZhciB4dHJhY2UgPSB0aGlzLnRyYWNlLmV4dGVuZGVkW2lkXTtcblx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXG5cdFx0XHQvLyBpZiBnYXRlZCBuZXVyb24ncyBzZWxmY29ubmVjdGlvbiBpcyBnYXRlZCBieSB0aGlzIHVuaXQsIHRoZSBpbmZsdWVuY2Uga2VlcHMgdHJhY2sgb2YgdGhlIG5ldXJvbidzIG9sZCBzdGF0ZVxuXHRcdFx0dmFyIGluZmx1ZW5jZSA9IG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzID8gbmV1cm9uLm9sZCA6IDA7XG5cblx0XHRcdC8vIGluZGV4IHJ1bnMgb3ZlciBhbGwgdGhlIGluY29taW5nIGNvbm5lY3Rpb25zIHRvIHRoZSBnYXRlZCBuZXVyb24gdGhhdCBhcmUgZ2F0ZWQgYnkgdGhpcyB1bml0XG5cdFx0XHRmb3IgKHZhciBpbmNvbWluZyBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXSkgeyAvLyBjYXB0dXJlcyB0aGUgZWZmZWN0IHRoYXQgaGFzIGFuIGlucHV0IGNvbm5lY3Rpb24gdG8gdGhpcyB1bml0LCBvbiBhIG5ldXJvbiB0aGF0IGlzIGdhdGVkIGJ5IHRoaXMgdW5pdFxuXHRcdFx0XHRpbmZsdWVuY2UgKz0gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF1baW5jb21pbmddLndlaWdodCAqXG5cdFx0XHRcdHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdW2luY29taW5nXS5mcm9tLmFjdGl2YXRpb247XG5cdFx0XHR9XG5cdFx0XHRpbmZsdWVuY2VzW25ldXJvbi5JRF0gPSBpbmZsdWVuY2U7XG5cdFx0fVxuXG5cdFx0Zm9yICh2YXIgaSBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0dmFyIHRoZUlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cblx0XHRcdC8vIGVsZWdpYmlsaXR5IHRyYWNlIC0gRXEuIDE3XG5cdFx0XHR0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RoZUlucHV0LklEXSA9IHRoaXMuc2VsZmNvbm5lY3Rpb24uZ2FpbiAqIHRoaXMuc2VsZmNvbm5lY3Rpb25cblx0XHRcdC53ZWlnaHQgKiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RoZUlucHV0LklEXSArIHRoZUlucHV0LmdhaW4gKiB0aGVJbnB1dC5mcm9tXG5cdFx0XHQuYWN0aXZhdGlvbjtcblxuXHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHQvLyBleHRlbmRlZCBlbGVnaWJpbGl0eSB0cmFjZVxuXHRcdFx0XHR2YXIgeHRyYWNlID0gdGhpcy50cmFjZS5leHRlbmRlZFtpZF07XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHR2YXIgaW5mbHVlbmNlID0gaW5mbHVlbmNlc1tuZXVyb24uSURdO1xuXG5cdFx0XHRcdC8vIGVxLiAxOFxuXHRcdFx0XHR4dHJhY2VbdGhlSW5wdXQuSURdID0gbmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhaW4gKiBuZXVyb24uc2VsZmNvbm5lY3Rpb25cblx0XHRcdFx0LndlaWdodCAqIHh0cmFjZVt0aGVJbnB1dC5JRF0gKyB0aGlzLmRlcml2YXRpdmUgKiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W1xuXHRcdFx0XHR0aGVJbnB1dC5JRF0gKiBpbmZsdWVuY2U7XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFx0Ly8gIHVwZGF0ZSBnYXRlZCBjb25uZWN0aW9uJ3MgZ2FpbnNcblx0XHRmb3IgKHZhciBjb25uZWN0aW9uIGluIHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWQpIHtcblx0XHRcdHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWRbY29ubmVjdGlvbl0uZ2FpbiA9IHRoaXMuYWN0aXZhdGlvbjtcblx0XHR9XG5cblx0XHRyZXR1cm4gdGhpcy5hY3RpdmF0aW9uO1xuXHR9XG5cblx0Ly8gYmFjay1wcm9wYWdhdGUgdGhlIGVycm9yXG5cdHByb3BhZ2F0ZShyYXRlOiBudW1iZXIsIHRhcmdldD86IG51bWJlcikge1xuXHRcdC8vIGVycm9yIGFjY3VtdWxhdG9yXG5cdFx0dmFyIGVycm9yID0gMDtcblxuXHRcdC8vIHdoZXRoZXIgb3Igbm90IHRoaXMgbmV1cm9uIGlzIGluIHRoZSBvdXRwdXQgbGF5ZXJcblx0XHR2YXIgaXNPdXRwdXQgPSB0eXBlb2YgdGFyZ2V0ICE9ICd1bmRlZmluZWQnICYmIHRhcmdldCAhPSBudWxsO1xuXG5cdFx0Ly8gb3V0cHV0IG5ldXJvbnMgZ2V0IHRoZWlyIGVycm9yIGZyb20gdGhlIGVudmlyb21lbnRcblx0XHRpZiAoaXNPdXRwdXQpXG5cdFx0XHR0aGlzLmVycm9yLnJlc3BvbnNpYmlsaXR5ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgPSB0YXJnZXQgLSB0aGlzLmFjdGl2YXRpb247IC8vIEVxLiAxMFxuICAgIFxuXHRcdGVsc2UgLy8gdGhlIHJlc3Qgb2YgdGhlIG5ldXJvbiBjb21wdXRlIHRoZWlyIGVycm9yIHJlc3BvbnNpYmlsaXRpZXMgYnkgYmFja3Byb3BhZ2F0aW9uXG5cdFx0e1xuXHRcdFx0Ly8gZXJyb3IgcmVzcG9uc2liaWxpdGllcyBmcm9tIGFsbCB0aGUgY29ubmVjdGlvbnMgcHJvamVjdGVkIGZyb20gdGhpcyBuZXVyb25cblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG5cdFx0XHRcdHZhciBjb25uZWN0aW9uID0gdGhpcy5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbaWRdO1xuXHRcdFx0XHR2YXIgbmV1cm9uID0gY29ubmVjdGlvbi50bztcblx0XHRcdFx0Ly8gRXEuIDIxXG5cdFx0XHRcdGVycm9yICs9IG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSAqIGNvbm5lY3Rpb24uZ2FpbiAqIGNvbm5lY3Rpb24ud2VpZ2h0O1xuXHRcdFx0fVxuXG5cdFx0XHQvLyBwcm9qZWN0ZWQgZXJyb3IgcmVzcG9uc2liaWxpdHlcblx0XHRcdHRoaXMuZXJyb3IucHJvamVjdGVkID0gdGhpcy5kZXJpdmF0aXZlICogZXJyb3I7XG5cblx0XHRcdGVycm9yID0gMDtcblx0XHRcdC8vIGVycm9yIHJlc3BvbnNpYmlsaXRpZXMgZnJvbSBhbGwgdGhlIGNvbm5lY3Rpb25zIGdhdGVkIGJ5IHRoaXMgbmV1cm9uXG5cdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdOyAvLyBnYXRlZCBuZXVyb25cblx0XHRcdFx0dmFyIGluZmx1ZW5jZSA9IG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzID8gbmV1cm9uLm9sZCA6IDA7IC8vIGlmIGdhdGVkIG5ldXJvbidzIHNlbGZjb25uZWN0aW9uIGlzIGdhdGVkIGJ5IHRoaXMgbmV1cm9uXG5cblx0XHRcdFx0Ly8gaW5kZXggcnVucyBvdmVyIGFsbCB0aGUgY29ubmVjdGlvbnMgdG8gdGhlIGdhdGVkIG5ldXJvbiB0aGF0IGFyZSBnYXRlZCBieSB0aGlzIG5ldXJvblxuXHRcdFx0XHRmb3IgKHZhciBpbnB1dCBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbaWRdKSB7IC8vIGNhcHR1cmVzIHRoZSBlZmZlY3QgdGhhdCB0aGUgaW5wdXQgY29ubmVjdGlvbiBvZiB0aGlzIG5ldXJvbiBoYXZlLCBvbiBhIG5ldXJvbiB3aGljaCBpdHMgaW5wdXQvcyBpcy9hcmUgZ2F0ZWQgYnkgdGhpcyBuZXVyb25cblx0XHRcdFx0XHRpbmZsdWVuY2UgKz0gdGhpcy50cmFjZS5pbmZsdWVuY2VzW2lkXVtpbnB1dF0ud2VpZ2h0ICogdGhpcy50cmFjZS5pbmZsdWVuY2VzW1xuXHRcdFx0XHRcdG5ldXJvbi5JRF1baW5wdXRdLmZyb20uYWN0aXZhdGlvbjtcblx0XHRcdFx0fVxuXHRcdFx0XHQvLyBlcS4gMjJcblx0XHRcdFx0ZXJyb3IgKz0gbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5ICogaW5mbHVlbmNlO1xuXHRcdFx0fVxuXG5cdFx0XHQvLyBnYXRlZCBlcnJvciByZXNwb25zaWJpbGl0eVxuXHRcdFx0dGhpcy5lcnJvci5nYXRlZCA9IHRoaXMuZGVyaXZhdGl2ZSAqIGVycm9yO1xuXG5cdFx0XHQvLyBlcnJvciByZXNwb25zaWJpbGl0eSAtIEVxLiAyM1xuXHRcdFx0dGhpcy5lcnJvci5yZXNwb25zaWJpbGl0eSA9IHRoaXMuZXJyb3IucHJvamVjdGVkICsgdGhpcy5lcnJvci5nYXRlZDtcblx0XHR9XG5cblx0XHQvLyBsZWFybmluZyByYXRlXG5cdFx0cmF0ZSA9IHJhdGUgfHwgLjE7XG5cblx0XHQvLyBhZGp1c3QgYWxsIHRoZSBuZXVyb24ncyBpbmNvbWluZyBjb25uZWN0aW9uc1xuXHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKSB7XG5cdFx0XHR2YXIgdGhlSW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cblx0XHRcdC8vIEVxLiAyNFxuXHRcdFx0dmFyIGdyYWRpZW50ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgKiB0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RoZUlucHV0LklEXTtcblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdGdyYWRpZW50ICs9IG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSAqIHRoaXMudHJhY2UuZXh0ZW5kZWRbXG5cdFx0XHRcdG5ldXJvbi5JRF1bdGhlSW5wdXQuSURdO1xuXHRcdFx0fVxuXHRcdFx0dGhlSW5wdXQud2VpZ2h0ICs9IHJhdGUgKiBncmFkaWVudDsgLy8gYWRqdXN0IHdlaWdodHMgLSBha2EgbGVhcm5cblx0XHR9XG5cblx0XHQvLyBhZGp1c3QgYmlhc1xuXHRcdHRoaXMuYmlhcyArPSByYXRlICogdGhpcy5lcnJvci5yZXNwb25zaWJpbGl0eTtcblx0fVxuXG5cdHByb2plY3QobmV1cm9uLCB3ZWlnaHQ/OiBudW1iZXIpOiBOZXVyb24uQ29ubmVjdGlvbiB7XG5cdFx0Ly8gc2VsZi1jb25uZWN0aW9uXG5cdFx0aWYgKG5ldXJvbiA9PSB0aGlzKSB7XG5cdFx0XHR0aGlzLnNlbGZjb25uZWN0aW9uLndlaWdodCA9IDE7XG5cdFx0XHRyZXR1cm4gdGhpcy5zZWxmY29ubmVjdGlvbjtcblx0XHR9XG5cblx0XHQvLyBjaGVjayBpZiBjb25uZWN0aW9uIGFscmVhZHkgZXhpc3RzXG5cdFx0dmFyIGNvbm5lY3RlZCA9IHRoaXMuY29ubmVjdGVkKG5ldXJvbik7XG5cdFx0aWYgKGNvbm5lY3RlZCAmJiBjb25uZWN0ZWQudHlwZSA9PSBcInByb2plY3RlZFwiKSB7XG5cdFx0XHQvLyB1cGRhdGUgY29ubmVjdGlvblxuXHRcdFx0aWYgKHR5cGVvZiB3ZWlnaHQgIT0gJ3VuZGVmaW5lZCcpXG5cdFx0XHRcdGNvbm5lY3RlZC5jb25uZWN0aW9uLndlaWdodCA9IHdlaWdodDtcblx0XHRcdC8vIHJldHVybiBleGlzdGluZyBjb25uZWN0aW9uXG5cdFx0XHRyZXR1cm4gY29ubmVjdGVkLmNvbm5lY3Rpb247XG5cdFx0fSBlbHNlIHtcblx0XHRcdC8vIGNyZWF0ZSBhIG5ldyBjb25uZWN0aW9uXG5cdFx0XHR2YXIgY29ubmVjdGlvbiA9IG5ldyBOZXVyb24uQ29ubmVjdGlvbih0aGlzLCBuZXVyb24sIHdlaWdodCk7XG5cdFx0fVxuXG5cdFx0Ly8gcmVmZXJlbmNlIGFsbCB0aGUgY29ubmVjdGlvbnMgYW5kIHRyYWNlc1xuXHRcdHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcblx0XHR0aGlzLm5laWdoYm9vcnNbbmV1cm9uLklEXSA9IG5ldXJvbjtcblx0XHRuZXVyb24uY29ubmVjdGlvbnMuaW5wdXRzW2Nvbm5lY3Rpb24uSURdID0gY29ubmVjdGlvbjtcblx0XHRuZXVyb24udHJhY2UuZWxlZ2liaWxpdHlbY29ubmVjdGlvbi5JRF0gPSAwO1xuXG5cdFx0Zm9yICh2YXIgaWQgaW4gbmV1cm9uLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHR2YXIgdHJhY2UgPSBuZXVyb24udHJhY2UuZXh0ZW5kZWRbaWRdO1xuXHRcdFx0dHJhY2VbY29ubmVjdGlvbi5JRF0gPSAwO1xuXHRcdH1cblxuXHRcdHJldHVybiBjb25uZWN0aW9uO1xuXHR9XG5cblx0Z2F0ZShjb25uZWN0aW9uKSB7XG5cdFx0Ly8gYWRkIGNvbm5lY3Rpb24gdG8gZ2F0ZWQgbGlzdFxuXHRcdHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWRbY29ubmVjdGlvbi5JRF0gPSBjb25uZWN0aW9uO1xuXG5cdFx0dmFyIG5ldXJvbiA9IGNvbm5lY3Rpb24udG87XG5cdFx0aWYgKCEobmV1cm9uLklEIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpKSB7XG5cdFx0XHQvLyBleHRlbmRlZCB0cmFjZVxuXHRcdFx0dGhpcy5uZWlnaGJvb3JzW25ldXJvbi5JRF0gPSBuZXVyb247XG5cdFx0XHR2YXIgeHRyYWNlID0gdGhpcy50cmFjZS5leHRlbmRlZFtuZXVyb24uSURdID0ge307XG5cdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHR2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cdFx0XHRcdHh0cmFjZVtpbnB1dC5JRF0gPSAwO1xuXHRcdFx0fVxuXHRcdH1cblxuXHRcdC8vIGtlZXAgdHJhY2tcblx0XHRpZiAobmV1cm9uLklEIGluIHRoaXMudHJhY2UuaW5mbHVlbmNlcylcblx0XHRcdHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdLnB1c2goY29ubmVjdGlvbik7XG5cdFx0ZWxzZVxuXHRcdFx0dGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0gPSBbY29ubmVjdGlvbl07XG5cblx0XHQvLyBzZXQgZ2F0ZXJcblx0XHRjb25uZWN0aW9uLmdhdGVyID0gdGhpcztcblx0fVxuICBcblx0Ly8gcmV0dXJucyB0cnVlIG9yIGZhbHNlIHdoZXRoZXIgdGhlIG5ldXJvbiBpcyBzZWxmLWNvbm5lY3RlZCBvciBub3Rcblx0c2VsZmNvbm5lY3RlZCgpIHtcblx0XHRyZXR1cm4gdGhpcy5zZWxmY29ubmVjdGlvbi53ZWlnaHQgIT09IDA7XG5cdH1cblxuXHQvLyByZXR1cm5zIHRydWUgb3IgZmFsc2Ugd2hldGhlciB0aGUgbmV1cm9uIGlzIGNvbm5lY3RlZCB0byBhbm90aGVyIG5ldXJvbiAocGFyYW1ldGVyKVxuXHRjb25uZWN0ZWQobmV1cm9uKSB7XG5cdFx0dmFyIHJlc3VsdDoge1xuXHRcdFx0dHlwZTogc3RyaW5nO1xuXHRcdFx0Y29ubmVjdGlvbjogTmV1cm9uLkNvbm5lY3Rpb247XG5cdFx0fSA9IHtcblx0XHRcdFx0dHlwZTogbnVsbCxcblx0XHRcdFx0Y29ubmVjdGlvbjogbnVsbFxuXHRcdFx0fTtcblxuXHRcdGlmICh0aGlzID09IG5ldXJvbikge1xuXHRcdFx0aWYgKHRoaXMuc2VsZmNvbm5lY3RlZCgpKSB7XG5cdFx0XHRcdHJlc3VsdC50eXBlID0gJ3NlbGZjb25uZWN0aW9uJztcblx0XHRcdFx0cmVzdWx0LmNvbm5lY3Rpb24gPSB0aGlzLnNlbGZjb25uZWN0aW9uO1xuXHRcdFx0XHRyZXR1cm4gcmVzdWx0O1xuXHRcdFx0fSBlbHNlXG5cdFx0XHRcdHJldHVybiBudWxsO1xuXHRcdH1cblxuXHRcdGZvciAodmFyIHR5cGUgaW4gdGhpcy5jb25uZWN0aW9ucykge1xuXHRcdFx0Zm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zW3R5cGVdKSB7XG5cdFx0XHRcdHZhciBjb25uZWN0aW9uID0gdGhpcy5jb25uZWN0aW9uc1t0eXBlXVtjb25uZWN0aW9uXTtcblx0XHRcdFx0aWYgKGNvbm5lY3Rpb24udG8gPT0gbmV1cm9uKSB7XG5cdFx0XHRcdFx0cmVzdWx0LnR5cGUgPSB0eXBlO1xuXHRcdFx0XHRcdHJlc3VsdC5jb25uZWN0aW9uID0gY29ubmVjdGlvbjtcblx0XHRcdFx0XHRyZXR1cm4gcmVzdWx0O1xuXHRcdFx0XHR9IGVsc2UgaWYgKGNvbm5lY3Rpb24uZnJvbSA9PSBuZXVyb24pIHtcblx0XHRcdFx0XHRyZXN1bHQudHlwZSA9IHR5cGU7XG5cdFx0XHRcdFx0cmVzdWx0LmNvbm5lY3Rpb24gPSBjb25uZWN0aW9uO1xuXHRcdFx0XHRcdHJldHVybiByZXN1bHQ7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHR9XG5cblx0XHRyZXR1cm4gbnVsbDtcblx0fVxuXG5cdC8vIGNsZWFycyBhbGwgdGhlIHRyYWNlcyAodGhlIG5ldXJvbiBmb3JnZXRzIGl0J3MgY29udGV4dCwgYnV0IHRoZSBjb25uZWN0aW9ucyByZW1haW4gaW50YWN0KVxuXHRjbGVhcigpIHtcblxuXHRcdGZvciAodmFyIHRyYWNlIGluIHRoaXMudHJhY2UuZWxlZ2liaWxpdHkpXG5cdFx0XHR0aGlzLnRyYWNlLmVsZWdpYmlsaXR5W3RyYWNlXSA9IDA7XG5cblx0XHRmb3IgKHZhciB0cmFjZSBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKVxuXHRcdFx0Zm9yICh2YXIgZXh0ZW5kZWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZFt0cmFjZV0pXG5cdFx0XHRcdHRoaXMudHJhY2UuZXh0ZW5kZWRbdHJhY2VdW2V4dGVuZGVkXSA9IDA7XG5cblx0XHR0aGlzLmVycm9yLnJlc3BvbnNpYmlsaXR5ID0gdGhpcy5lcnJvci5wcm9qZWN0ZWQgPSB0aGlzLmVycm9yLmdhdGVkID0gMDtcblx0fVxuXG5cdC8vIGFsbCB0aGUgY29ubmVjdGlvbnMgYXJlIHJhbmRvbWl6ZWQgYW5kIHRoZSB0cmFjZXMgYXJlIGNsZWFyZWRcblx0cmVzZXQoKSB7XG5cdFx0dGhpcy5jbGVhcigpO1xuXG5cdFx0Zm9yICh2YXIgdHlwZSBpbiB0aGlzLmNvbm5lY3Rpb25zKVxuXHRcdFx0Zm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zW3R5cGVdKVxuXHRcdFx0XHR0aGlzLmNvbm5lY3Rpb25zW3R5cGVdW2Nvbm5lY3Rpb25dLndlaWdodCA9IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xO1xuXHRcdHRoaXMuYmlhcyA9IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xO1xuXG5cdFx0dGhpcy5vbGQgPSB0aGlzLnN0YXRlID0gdGhpcy5hY3RpdmF0aW9uID0gMDtcblx0fVxuXHRcblxuICBcblxuXHQvLyBoYXJkY29kZXMgdGhlIGJlaGF2aW91ciBvZiB0aGUgbmV1cm9uIGludG8gYW4gb3B0aW1pemVkIGZ1bmN0aW9uXG5cdG9wdGltaXplKG9wdGltaXplZCwgbGF5ZXIpOiBTeW5hcHRpYy5JQ29tcGlsZWRQYXJhbWV0ZXJzIHtcblxuXHRcdG9wdGltaXplZCA9IG9wdGltaXplZCB8fCB7fTtcblx0XHR2YXIgdGhhdCA9IHRoaXM7XG5cdFx0dmFyIHN0b3JlX2FjdGl2YXRpb24gPSBbXTtcblx0XHR2YXIgc3RvcmVfdHJhY2UgPSBbXTtcblx0XHR2YXIgc3RvcmVfcHJvcGFnYXRpb24gPSBbXTtcblx0XHR2YXIgdmFySUQgPSBvcHRpbWl6ZWQubWVtb3J5IHx8IDA7XG5cdFx0dmFyIG5ldXJvbnMgPSBvcHRpbWl6ZWQubmV1cm9ucyB8fCAxO1xuXHRcdHZhciBpbnB1dHMgPSBvcHRpbWl6ZWQuaW5wdXRzIHx8IFtdO1xuXHRcdHZhciB0YXJnZXRzID0gb3B0aW1pemVkLnRhcmdldHMgfHwgW107XG5cdFx0dmFyIG91dHB1dHMgPSBvcHRpbWl6ZWQub3V0cHV0cyB8fCBbXTtcblx0XHR2YXIgdmFyaWFibGVzID0gb3B0aW1pemVkLnZhcmlhYmxlcyB8fCB7fTtcblx0XHR2YXIgYWN0aXZhdGlvbl9zZW50ZW5jZXMgPSBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXMgfHwgW107XG5cdFx0dmFyIHRyYWNlX3NlbnRlbmNlcyA9IG9wdGltaXplZC50cmFjZV9zZW50ZW5jZXMgfHwgW107XG5cdFx0dmFyIHByb3BhZ2F0aW9uX3NlbnRlbmNlcyA9IG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMgfHwgW107XG5cdFx0dmFyIGxheWVycyA9IG9wdGltaXplZC5sYXllcnMgfHwgeyBfX2NvdW50OiAwLCBfX25ldXJvbjogMCB9O1xuXG5cdFx0Ly8gYWxsb2NhdGUgc2VudGVuY2VzXG5cdFx0dmFyIGFsbG9jYXRlID0gZnVuY3Rpb24oc3RvcmUpIHtcblx0XHRcdHZhciBhbGxvY2F0ZWQgPSBsYXllciBpbiBsYXllcnMgJiYgc3RvcmVbbGF5ZXJzLl9fY291bnRdO1xuXHRcdFx0aWYgKCFhbGxvY2F0ZWQpIHtcblx0XHRcdFx0bGF5ZXJzLl9fY291bnQgPSBzdG9yZS5wdXNoKFtdKSAtIDE7XG5cdFx0XHRcdGxheWVyc1tsYXllcl0gPSBsYXllcnMuX19jb3VudDtcblx0XHRcdH1cblx0XHR9XG5cdFx0YWxsb2NhdGUoYWN0aXZhdGlvbl9zZW50ZW5jZXMpO1xuXHRcdGFsbG9jYXRlKHRyYWNlX3NlbnRlbmNlcyk7XG5cdFx0YWxsb2NhdGUocHJvcGFnYXRpb25fc2VudGVuY2VzKTtcblx0XHR2YXIgY3VycmVudExheWVyID0gbGF5ZXJzLl9fY291bnQ7XG5cblx0XHQvLyBnZXQvcmVzZXJ2ZSBzcGFjZSBpbiBtZW1vcnkgYnkgY3JlYXRpbmcgYSB1bmlxdWUgSUQgZm9yIGEgdmFyaWFibGVsXG5cdFx0dmFyIGdldFZhciA9IGZ1bmN0aW9uKC4uLmFyZ3M6IGFueVtdKSB7XG5cdFx0XHR2YXIgaWQ7XG5cdFx0XHRpZiAoYXJncy5sZW5ndGggPT0gMSkge1xuXG5cdFx0XHRcdGlmIChhcmdzWzBdID09ICd0YXJnZXQnKSB7XG5cdFx0XHRcdFx0aWQgPSAndGFyZ2V0XycgKyB0YXJnZXRzLmxlbmd0aDtcblx0XHRcdFx0XHR0YXJnZXRzLnB1c2godmFySUQpO1xuXHRcdFx0XHR9IGVsc2Vcblx0XHRcdFx0XHRpZCA9IGFyZ3NbMF07XG5cdFx0XHRcdGlmIChpZCBpbiB2YXJpYWJsZXMpXG5cdFx0XHRcdFx0cmV0dXJuIHZhcmlhYmxlc1tpZF07XG5cdFx0XHRcdHJldHVybiB2YXJpYWJsZXNbaWRdID0ge1xuXHRcdFx0XHRcdHZhbHVlOiAwLFxuXHRcdFx0XHRcdGlkOiB2YXJJRCsrXG5cdFx0XHRcdH07XG5cdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHR2YXIgZXh0ZW5kZWQgPSBhcmdzLmxlbmd0aCA+IDI7XG5cdFx0XHRcdGlmIChleHRlbmRlZClcblx0XHRcdFx0XHR2YXIgdmFsdWUgPSBhcmdzLnBvcCgpO1xuXG5cdFx0XHRcdHZhciB1bml0ID0gYXJncy5zaGlmdCgpO1xuXHRcdFx0XHR2YXIgcHJvcCA9IGFyZ3MucG9wKCk7XG5cblx0XHRcdFx0aWYgKCFleHRlbmRlZClcblx0XHRcdFx0XHR2YXIgdmFsdWUgPSB1bml0W3Byb3BdO1xuXG5cdFx0XHRcdGlkID0gcHJvcCArICdfJztcblx0XHRcdFx0Zm9yICh2YXIgcHJvcGVydHkgaW4gYXJncylcblx0XHRcdFx0XHRpZCArPSBhcmdzW3Byb3BlcnR5XSArICdfJztcblx0XHRcdFx0aWQgKz0gdW5pdC5JRDtcblx0XHRcdFx0aWYgKGlkIGluIHZhcmlhYmxlcylcblx0XHRcdFx0XHRyZXR1cm4gdmFyaWFibGVzW2lkXTtcblxuXHRcdFx0XHRyZXR1cm4gdmFyaWFibGVzW2lkXSA9IHtcblx0XHRcdFx0XHR2YWx1ZTogdmFsdWUsXG5cdFx0XHRcdFx0aWQ6IHZhcklEKytcblx0XHRcdFx0fTtcblx0XHRcdH1cblx0XHR9O1xuXG5cdFx0Ly8gYnVpbGQgc2VudGVuY2Vcblx0XHR2YXIgYnVpbGRTZW50ZW5jZSA9IGZ1bmN0aW9uKC4uLmFyZ3M6IGFueVtdKSB7XG5cdFx0XHR2YXIgc3RvcmUgPSBhcmdzLnBvcCgpO1xuXHRcdFx0dmFyIHNlbnRlbmNlID0gXCJcIjtcblx0XHRcdGZvciAodmFyIGkgaW4gYXJncylcblx0XHRcdFx0aWYgKHR5cGVvZiBhcmdzW2ldID09ICdzdHJpbmcnKVxuXHRcdFx0XHRcdHNlbnRlbmNlICs9IGFyZ3NbaV07XG5cdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRzZW50ZW5jZSArPSAnRlsnICsgYXJnc1tpXS5pZCArICddJztcblxuXHRcdFx0c3RvcmUucHVzaChzZW50ZW5jZSArICc7Jyk7XG5cdFx0fVxuXG5cdFx0Ly8gaGVscGVyIHRvIGNoZWNrIGlmIGFuIG9iamVjdCBpcyBlbXB0eVxuXHRcdHZhciBpc0VtcHR5ID0gZnVuY3Rpb24ob2JqKSB7XG5cdFx0XHRmb3IgKHZhciBwcm9wIGluIG9iaikge1xuXHRcdFx0XHRpZiAob2JqLmhhc093blByb3BlcnR5KHByb3ApKVxuXHRcdFx0XHRcdHJldHVybiBmYWxzZTtcblx0XHRcdH1cblx0XHRcdHJldHVybiB0cnVlO1xuXHRcdH07XG5cblx0XHQvLyBjaGFyYWN0ZXJpc3RpY3Mgb2YgdGhlIG5ldXJvblxuXHRcdHZhciBub1Byb2plY3Rpb25zID0gaXNFbXB0eSh0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZCk7XG5cdFx0dmFyIG5vR2F0ZXMgPSBpc0VtcHR5KHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWQpO1xuXHRcdHZhciBpc0lucHV0ID0gbGF5ZXIgPT0gJ2lucHV0JyA/IHRydWUgOiBpc0VtcHR5KHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzKTtcblx0XHR2YXIgaXNPdXRwdXQgPSBsYXllciA9PSAnb3V0cHV0JyA/IHRydWUgOiBub1Byb2plY3Rpb25zICYmIG5vR2F0ZXM7XG5cblx0XHQvLyBvcHRpbWl6ZSBuZXVyb24ncyBiZWhhdmlvdXJcblx0XHR2YXIgcmF0ZSA9IGdldFZhcigncmF0ZScpO1xuXHRcdHZhciBhY3RpdmF0aW9uID0gZ2V0VmFyKHRoaXMsICdhY3RpdmF0aW9uJyk7XG5cdFx0aWYgKGlzSW5wdXQpXG5cdFx0XHRpbnB1dHMucHVzaChhY3RpdmF0aW9uLmlkKTtcblx0XHRlbHNlIHtcblx0XHRcdGFjdGl2YXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0ucHVzaChzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdHRyYWNlX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdLnB1c2goc3RvcmVfdHJhY2UpO1xuXHRcdFx0cHJvcGFnYXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0ucHVzaChzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHR2YXIgb2xkID0gZ2V0VmFyKHRoaXMsICdvbGQnKTtcblx0XHRcdHZhciBzdGF0ZSA9IGdldFZhcih0aGlzLCAnc3RhdGUnKTtcblx0XHRcdHZhciBiaWFzID0gZ2V0VmFyKHRoaXMsICdiaWFzJyk7XG5cdFx0XHRpZiAodGhpcy5zZWxmY29ubmVjdGlvbi5nYXRlcilcblx0XHRcdFx0dmFyIHNlbGZfZ2FpbiA9IGdldFZhcih0aGlzLnNlbGZjb25uZWN0aW9uLCAnZ2FpbicpO1xuXHRcdFx0aWYgKHRoaXMuc2VsZmNvbm5lY3RlZCgpKVxuXHRcdFx0XHR2YXIgc2VsZl93ZWlnaHQgPSBnZXRWYXIodGhpcy5zZWxmY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuXHRcdFx0YnVpbGRTZW50ZW5jZShvbGQsICcgPSAnLCBzdGF0ZSwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRpZiAodGhpcy5zZWxmY29ubmVjdGVkKCkpXG5cdFx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0aW9uLmdhdGVyKVxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2Uoc3RhdGUsICcgPSAnLCBzZWxmX2dhaW4sICcgKiAnLCBzZWxmX3dlaWdodCwgJyAqICcsXG5cdFx0XHRcdFx0XHRzdGF0ZSwgJyArICcsIGJpYXMsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShzdGF0ZSwgJyA9ICcsIHNlbGZfd2VpZ2h0LCAnICogJywgc3RhdGUsICcgKyAnLFxuXHRcdFx0XHRcdFx0Ymlhcywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRlbHNlXG5cdFx0XHRcdGJ1aWxkU2VudGVuY2Uoc3RhdGUsICcgPSAnLCBiaWFzLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdGZvciAodmFyIGkgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cdFx0XHRcdHZhciBpbnB1dF9hY3RpdmF0aW9uID0gZ2V0VmFyKGlucHV0LmZyb20sICdhY3RpdmF0aW9uJyk7XG5cdFx0XHRcdHZhciBpbnB1dF93ZWlnaHQgPSBnZXRWYXIoaW5wdXQsICd3ZWlnaHQnKTtcblx0XHRcdFx0aWYgKGlucHV0LmdhdGVyKVxuXHRcdFx0XHRcdHZhciBpbnB1dF9nYWluID0gZ2V0VmFyKGlucHV0LCAnZ2FpbicpO1xuXHRcdFx0XHRpZiAodGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV0uZ2F0ZXIpXG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShzdGF0ZSwgJyArPSAnLCBpbnB1dF9hY3RpdmF0aW9uLCAnICogJyxcblx0XHRcdFx0XHRcdGlucHV0X3dlaWdodCwgJyAqICcsIGlucHV0X2dhaW4sIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShzdGF0ZSwgJyArPSAnLCBpbnB1dF9hY3RpdmF0aW9uLCAnICogJyxcblx0XHRcdFx0XHRcdGlucHV0X3dlaWdodCwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHR9XG5cdFx0XHR2YXIgZGVyaXZhdGl2ZSA9IGdldFZhcih0aGlzLCAnZGVyaXZhdGl2ZScpO1xuXHRcdFx0c3dpdGNoICh0aGlzLnNxdWFzaCkge1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5MT0dJU1RJQzpcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSAoMSAvICgxICsgTWF0aC5leHAoLScsIHN0YXRlLCAnKSkpJyxcblx0XHRcdFx0XHRcdHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZGVyaXZhdGl2ZSwgJyA9ICcsIGFjdGl2YXRpb24sICcgKiAoMSAtICcsXG5cdFx0XHRcdFx0XHRhY3RpdmF0aW9uLCAnKScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJyZWFrO1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5UQU5IOlxuXHRcdFx0XHRcdHZhciBlUCA9IGdldFZhcignYXV4Jyk7XG5cdFx0XHRcdFx0dmFyIGVOID0gZ2V0VmFyKCdhdXhfMicpO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZVAsICcgPSBNYXRoLmV4cCgnLCBzdGF0ZSwgJyknLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVOLCAnID0gMSAvICcsIGVQLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSAoJywgZVAsICcgLSAnLCBlTiwgJykgLyAoJywgZVAsICcgKyAnLCBlTiwgJyknLCBzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGRlcml2YXRpdmUsICcgPSAxIC0gKCcsIGFjdGl2YXRpb24sICcgKiAnLCBhY3RpdmF0aW9uLCAnKScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJyZWFrO1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5JREVOVElUWTpcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGFjdGl2YXRpb24sICcgPSAnLCBzdGF0ZSwgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShkZXJpdmF0aXZlLCAnID0gMScsIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0XHRcdGJyZWFrO1xuXHRcdFx0XHRjYXNlIFNxdWFzaC5ITElNOlxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoYWN0aXZhdGlvbiwgJyA9ICsoJywgc3RhdGUsICcgPiAwKScsXG5cdFx0XHRcdFx0XHRzdG9yZV9hY3RpdmF0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGRlcml2YXRpdmUsICcgPSAxJywgc3RvcmVfYWN0aXZhdGlvbik7XG5cdFx0XHRcdFx0YnJlYWs7XG5cdFx0XHR9XG5cblxuXHRcdFx0dmFyIGluZmx1ZW5jZXMgPSBbXTtcblx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0Ly8gY2FsY3VsYXRlIGV4dGVuZGVkIGVsZWdpYmlsaXR5IHRyYWNlcyBpbiBhZHZhbmNlXG4gICAgICAgIFxuXHRcdFx0XHR2YXIgeHRyYWNlID0gdGhpcy50cmFjZS5leHRlbmRlZFtpZF07XG5cdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHR2YXIgaW5mbHVlbmNlID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0dmFyIG5ldXJvbl9vbGQgPSBnZXRWYXIobmV1cm9uLCAnb2xkJyk7XG5cdFx0XHRcdHZhciBpbml0aWFsaXplZCA9IGZhbHNlO1xuXHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyID09IHRoaXMpIHtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9ICcsIG5ldXJvbl9vbGQsIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRpbml0aWFsaXplZCA9IHRydWU7XG5cdFx0XHRcdH1cblx0XHRcdFx0Zm9yICh2YXIgaW5jb21pbmcgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0pIHtcblx0XHRcdFx0XHR2YXIgaW5jb21pbmdfd2VpZ2h0ID0gZ2V0VmFyKHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdXG5cdFx0XHRcdFx0W2luY29taW5nXSwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdHZhciBpbmNvbWluZ19hY3RpdmF0aW9uID0gZ2V0VmFyKHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdXG5cdFx0XHRcdFx0W2luY29taW5nXS5mcm9tLCAnYWN0aXZhdGlvbicpO1xuXG5cdFx0XHRcdFx0aWYgKGluaXRpYWxpemVkKVxuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgKz0gJywgaW5jb21pbmdfd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdFx0aW5jb21pbmdfYWN0aXZhdGlvbiwgc3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdGVsc2Uge1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAnLCBpbmNvbWluZ193ZWlnaHQsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRpbmNvbWluZ19hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRpbml0aWFsaXplZCA9IHRydWU7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9XG5cblx0XHRcdFx0aW5mbHVlbmNlcy5wdXNoKG5ldXJvbi5JRCk7XG5cdFx0XHRcdGJ1aWxkU2VudGVuY2UoXCJpbmZsdWVuY2VzW1wiICsgKGluZmx1ZW5jZXMubGVuZ3RoIC0gMSkgKyBcIl0gPSBcIiwgaW5mbHVlbmNlLCBzdG9yZV90cmFjZSk7XG5cdFx0XHR9XG5cblx0XHRcdGZvciAodmFyIGkgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaV07XG5cdFx0XHRcdGlmIChpbnB1dC5nYXRlcilcblx0XHRcdFx0XHR2YXIgaW5wdXRfZ2FpbiA9IGdldFZhcihpbnB1dCwgJ2dhaW4nKTtcblx0XHRcdFx0dmFyIGlucHV0X2FjdGl2YXRpb24gPSBnZXRWYXIoaW5wdXQuZnJvbSwgJ2FjdGl2YXRpb24nKTtcblx0XHRcdFx0dmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzLnRyYWNlXG5cdFx0XHRcdFx0LmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG5cdFx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0ZWQoKSkge1xuXHRcdFx0XHRcdGlmICh0aGlzLnNlbGZjb25uZWN0aW9uLmdhdGVyKSB7XG5cdFx0XHRcdFx0XHRpZiAoaW5wdXQuZ2F0ZXIpXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBzZWxmX2dhaW4sICcgKiAnLCBzZWxmX3dlaWdodCxcblx0XHRcdFx0XHRcdFx0XHQnICogJywgdHJhY2UsICcgKyAnLCBpbnB1dF9nYWluLCAnICogJywgaW5wdXRfYWN0aXZhdGlvbixcblx0XHRcdFx0XHRcdFx0XHRzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBzZWxmX2dhaW4sICcgKiAnLCBzZWxmX3dlaWdodCxcblx0XHRcdFx0XHRcdFx0XHQnICogJywgdHJhY2UsICcgKyAnLCBpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0XHRcdGlmIChpbnB1dC5nYXRlcilcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZSh0cmFjZSwgJyA9ICcsIHNlbGZfd2VpZ2h0LCAnICogJywgdHJhY2UsICcgKyAnLFxuXHRcdFx0XHRcdFx0XHRcdGlucHV0X2dhaW4sICcgKiAnLCBpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBzZWxmX3dlaWdodCwgJyAqICcsIHRyYWNlLCAnICsgJyxcblx0XHRcdFx0XHRcdFx0XHRpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRcdGlmIChpbnB1dC5nYXRlcilcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBpbnB1dF9nYWluLCAnICogJywgaW5wdXRfYWN0aXZhdGlvbixcblx0XHRcdFx0XHRcdFx0c3RvcmVfdHJhY2UpO1xuXHRcdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UodHJhY2UsICcgPSAnLCBpbnB1dF9hY3RpdmF0aW9uLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdH1cblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHRcdC8vIGV4dGVuZGVkIGVsZWdpYmlsaXR5IHRyYWNlXG5cdFx0XHRcdFx0dmFyIHh0cmFjZSA9IHRoaXMudHJhY2UuZXh0ZW5kZWRbaWRdO1xuXHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHRcdHZhciBpbmZsdWVuY2UgPSBnZXRWYXIoJ2F1eCcpO1xuXHRcdFx0XHRcdHZhciBuZXVyb25fb2xkID0gZ2V0VmFyKG5ldXJvbiwgJ29sZCcpO1xuXG5cdFx0XHRcdFx0dmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzLnRyYWNlXG5cdFx0XHRcdFx0XHQuZWxlZ2liaWxpdHlbaW5wdXQuSURdKTtcblx0XHRcdFx0XHR2YXIgeHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdleHRlbmRlZCcsIG5ldXJvbi5JRCwgaW5wdXQuSUQsXG5cdFx0XHRcdFx0XHR0aGlzLnRyYWNlLmV4dGVuZGVkW25ldXJvbi5JRF1baW5wdXQuSURdKTtcblx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0ZWQoKSlcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fc2VsZl93ZWlnaHQgPSBnZXRWYXIobmV1cm9uLnNlbGZjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlcilcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fc2VsZl9nYWluID0gZ2V0VmFyKG5ldXJvbi5zZWxmY29ubmVjdGlvbiwgJ2dhaW4nKTtcblx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0ZWQoKSlcblx0XHRcdFx0XHRcdGlmIChuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIpXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoeHRyYWNlLCAnID0gJywgbmV1cm9uX3NlbGZfZ2FpbiwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0bmV1cm9uX3NlbGZfd2VpZ2h0LCAnICogJywgeHRyYWNlLCAnICsgJywgZGVyaXZhdGl2ZSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0dHJhY2UsICcgKiAnLCBcImluZmx1ZW5jZXNbXCIgKyBpbmZsdWVuY2VzLmluZGV4T2YobmV1cm9uLklEKSArIFwiXVwiLCBzdG9yZV90cmFjZSk7XG5cdFx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoeHRyYWNlLCAnID0gJywgbmV1cm9uX3NlbGZfd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHR4dHJhY2UsICcgKyAnLCBkZXJpdmF0aXZlLCAnICogJywgdHJhY2UsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdFwiaW5mbHVlbmNlc1tcIiArIGluZmx1ZW5jZXMuaW5kZXhPZihuZXVyb24uSUQpICsgXCJdXCIsIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0XHRlbHNlXG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHh0cmFjZSwgJyA9ICcsIGRlcml2YXRpdmUsICcgKiAnLCB0cmFjZSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFwiaW5mbHVlbmNlc1tcIiArIGluZmx1ZW5jZXMuaW5kZXhPZihuZXVyb24uSUQpICsgXCJdXCIsIHN0b3JlX3RyYWNlKTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXHRcdFx0Zm9yICh2YXIgY29ubmVjdGlvbiBpbiB0aGlzLmNvbm5lY3Rpb25zLmdhdGVkKSB7XG5cdFx0XHRcdHZhciBnYXRlZF9nYWluID0gZ2V0VmFyKHRoaXMuY29ubmVjdGlvbnMuZ2F0ZWRbY29ubmVjdGlvbl0sICdnYWluJyk7XG5cdFx0XHRcdGJ1aWxkU2VudGVuY2UoZ2F0ZWRfZ2FpbiwgJyA9ICcsIGFjdGl2YXRpb24sIHN0b3JlX2FjdGl2YXRpb24pO1xuXHRcdFx0fVxuXHRcdH1cblx0XHRpZiAoIWlzSW5wdXQpIHtcblx0XHRcdHZhciByZXNwb25zaWJpbGl0eSA9IGdldFZhcih0aGlzLCAnZXJyb3InLCAncmVzcG9uc2liaWxpdHknLCB0aGlzLmVycm9yXG5cdFx0XHRcdC5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRpZiAoaXNPdXRwdXQpIHtcblx0XHRcdFx0dmFyIHRhcmdldCA9IGdldFZhcigndGFyZ2V0Jyk7XG5cdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgPSAnLCB0YXJnZXQsICcgLSAnLCBhY3RpdmF0aW9uLFxuXHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0XHR2YXIgaW5wdXQgPSB0aGlzLmNvbm5lY3Rpb25zLmlucHV0c1tpZF07XG5cdFx0XHRcdFx0dmFyIHRyYWNlID0gZ2V0VmFyKHRoaXMsICd0cmFjZScsICdlbGVnaWJpbGl0eScsIGlucHV0LklELCB0aGlzLnRyYWNlXG5cdFx0XHRcdFx0XHQuZWxlZ2liaWxpdHlbaW5wdXQuSURdKTtcblx0XHRcdFx0XHR2YXIgaW5wdXRfd2VpZ2h0ID0gZ2V0VmFyKGlucHV0LCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbnB1dF93ZWlnaHQsICcgKz0gJywgcmF0ZSwgJyAqICgnLCByZXNwb25zaWJpbGl0eSxcblx0XHRcdFx0XHRcdCcgKiAnLCB0cmFjZSwgJyknLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdH1cblx0XHRcdFx0b3V0cHV0cy5wdXNoKGFjdGl2YXRpb24uaWQpO1xuXHRcdFx0fSBlbHNlIHtcblx0XHRcdFx0aWYgKCFub1Byb2plY3Rpb25zICYmICFub0dhdGVzKSB7XG5cdFx0XHRcdFx0dmFyIGVycm9yID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZCkge1xuXHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSB0aGlzLmNvbm5lY3Rpb25zLnByb2plY3RlZFtpZF07XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gY29ubmVjdGlvbi50bztcblx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX3dlaWdodCA9IGdldFZhcihjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcblx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdGlmIChjb25uZWN0aW9uLmdhdGVyKSB7XG5cdFx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX2dhaW4gPSBnZXRWYXIoY29ubmVjdGlvbiwgJ2dhaW4nKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShlcnJvciwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdGNvbm5lY3Rpb25fZ2FpbiwgJyAqICcsIGNvbm5lY3Rpb25fd2VpZ2h0LFxuXHRcdFx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdH0gZWxzZVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVycm9yLCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0Y29ubmVjdGlvbl93ZWlnaHQsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0dmFyIHByb2plY3RlZCA9IGdldFZhcih0aGlzLCAnZXJyb3InLCAncHJvamVjdGVkJywgdGhpcy5lcnJvci5wcm9qZWN0ZWQpO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocHJvamVjdGVkLCAnID0gJywgZGVyaXZhdGl2ZSwgJyAqICcsIGVycm9yLFxuXHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZXJyb3IsICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0XHRcdHZhciBuZXVyb24gPSB0aGlzLm5laWdoYm9vcnNbaWRdO1xuXHRcdFx0XHRcdFx0dmFyIGluZmx1ZW5jZSA9IGdldFZhcignYXV4XzInKTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fb2xkID0gZ2V0VmFyKG5ldXJvbiwgJ29sZCcpO1xuXHRcdFx0XHRcdFx0aWYgKG5ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlciA9PSB0aGlzKVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9ICcsIG5ldXJvbl9vbGQsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdGVsc2Vcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0Zm9yICh2YXIgaW5mbHVlbmNlSW5wdXQgaW4gdGhpcy50cmFjZS5pbmZsdWVuY2VzW25ldXJvbi5JRF0pIHtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXVtpbmZsdWVuY2VJbnB1dF07XG5cdFx0XHRcdFx0XHRcdHZhciBjb25uZWN0aW9uX3dlaWdodCA9IGdldFZhcihjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHRcdHZhciBuZXVyb25fYWN0aXZhdGlvbiA9IGdldFZhcihjb25uZWN0aW9uLmZyb20sICdhY3RpdmF0aW9uJyk7XG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnICs9ICcsIGNvbm5lY3Rpb25fd2VpZ2h0LCAnICogJyxcblx0XHRcdFx0XHRcdFx0XHRuZXVyb25fYWN0aXZhdGlvbiwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG5cdFx0XHRcdFx0XHRcdCdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGVycm9yLCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdGluZmx1ZW5jZSwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHR2YXIgZ2F0ZWQgPSBnZXRWYXIodGhpcywgJ2Vycm9yJywgJ2dhdGVkJywgdGhpcy5lcnJvci5nYXRlZCk7XG5cdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShnYXRlZCwgJyA9ICcsIGRlcml2YXRpdmUsICcgKiAnLCBlcnJvcixcblx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnID0gJywgcHJvamVjdGVkLCAnICsgJywgZ2F0ZWQsXG5cdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0XHRcdHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcblx0XHRcdFx0XHRcdHZhciBncmFkaWVudCA9IGdldFZhcignYXV4Jyk7XG5cdFx0XHRcdFx0XHR2YXIgdHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2VsZWdpYmlsaXR5JywgaW5wdXQuSUQsIHRoaXNcblx0XHRcdFx0XHRcdFx0LnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0LklEXSk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGdyYWRpZW50LCAnID0gJywgcHJvamVjdGVkLCAnICogJywgdHJhY2UsXG5cdFx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMudHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbiA9IHRoaXMubmVpZ2hib29yc1tpZF07XG5cdFx0XHRcdFx0XHRcdHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuXHRcdFx0XHRcdFx0XHRcdCdyZXNwb25zaWJpbGl0eScsIG5ldXJvbi5lcnJvci5yZXNwb25zaWJpbGl0eSk7XG5cdFx0XHRcdFx0XHRcdHZhciB4dHJhY2UgPSBnZXRWYXIodGhpcywgJ3RyYWNlJywgJ2V4dGVuZGVkJywgbmV1cm9uLklELFxuXHRcdFx0XHRcdFx0XHRcdGlucHV0LklELCB0aGlzLnRyYWNlLmV4dGVuZGVkW25ldXJvbi5JRF1baW5wdXQuSURdKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShncmFkaWVudCwgJyArPSAnLCBuZXVyb25fcmVzcG9uc2liaWxpdHksICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdHh0cmFjZSwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fVxuXHRcdFx0XHRcdFx0dmFyIGlucHV0X3dlaWdodCA9IGdldFZhcihpbnB1dCwgJ3dlaWdodCcpO1xuXHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbnB1dF93ZWlnaHQsICcgKz0gJywgcmF0ZSwgJyAqICcsIGdyYWRpZW50LFxuXHRcdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXG5cdFx0XHRcdH0gZWxzZSBpZiAobm9HYXRlcykge1xuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdGZvciAodmFyIGlkIGluIHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG5cdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IHRoaXMuY29ubmVjdGlvbnMucHJvamVjdGVkW2lkXTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb24gPSBjb25uZWN0aW9uLnRvO1xuXHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fd2VpZ2h0ID0gZ2V0VmFyKGNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdHZhciBuZXVyb25fcmVzcG9uc2liaWxpdHkgPSBnZXRWYXIobmV1cm9uLCAnZXJyb3InLFxuXHRcdFx0XHRcdFx0XHQncmVzcG9uc2liaWxpdHknLCBuZXVyb24uZXJyb3IucmVzcG9uc2liaWxpdHkpO1xuXHRcdFx0XHRcdFx0aWYgKGNvbm5lY3Rpb24uZ2F0ZXIpIHtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fZ2FpbiA9IGdldFZhcihjb25uZWN0aW9uLCAnZ2FpbicpO1xuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSxcblx0XHRcdFx0XHRcdFx0XHQnICogJywgY29ubmVjdGlvbl9nYWluLCAnICogJywgY29ubmVjdGlvbl93ZWlnaHQsXG5cdFx0XHRcdFx0XHRcdFx0c3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0fSBlbHNlXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LFxuXHRcdFx0XHRcdFx0XHRcdCcgKiAnLCBjb25uZWN0aW9uX3dlaWdodCwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdH1cblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnICo9ICcsIGRlcml2YXRpdmUsXG5cdFx0XHRcdFx0XHRzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHMpIHtcblx0XHRcdFx0XHRcdHZhciBpbnB1dCA9IHRoaXMuY29ubmVjdGlvbnMuaW5wdXRzW2lkXTtcblx0XHRcdFx0XHRcdHZhciB0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZWxlZ2liaWxpdHknLCBpbnB1dC5JRCwgdGhpc1xuXHRcdFx0XHRcdFx0XHQudHJhY2UuZWxlZ2liaWxpdHlbaW5wdXQuSURdKTtcblx0XHRcdFx0XHRcdHZhciBpbnB1dF93ZWlnaHQgPSBnZXRWYXIoaW5wdXQsICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5wdXRfd2VpZ2h0LCAnICs9ICcsIHJhdGUsICcgKiAoJyxcblx0XHRcdFx0XHRcdFx0cmVzcG9uc2liaWxpdHksICcgKiAnLCB0cmFjZSwgJyknLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9IGVsc2UgaWYgKG5vUHJvamVjdGlvbnMpIHtcblx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKHJlc3BvbnNpYmlsaXR5LCAnID0gMCcsIHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0XHRcdHZhciBpbmZsdWVuY2UgPSBnZXRWYXIoJ2F1eCcpO1xuXHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9vbGQgPSBnZXRWYXIobmV1cm9uLCAnb2xkJyk7XG5cdFx0XHRcdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyID09IHRoaXMpXG5cdFx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoaW5mbHVlbmNlLCAnID0gJywgbmV1cm9uX29sZCwgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0ZWxzZVxuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGluZmx1ZW5jZSwgJyA9IDAnLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHRmb3IgKHZhciBpbmZsdWVuY2VJbnB1dCBpbiB0aGlzLnRyYWNlLmluZmx1ZW5jZXNbbmV1cm9uLklEXSkge1xuXHRcdFx0XHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IHRoaXMudHJhY2UuaW5mbHVlbmNlc1tuZXVyb24uSURdW2luZmx1ZW5jZUlucHV0XTtcblx0XHRcdFx0XHRcdFx0dmFyIGNvbm5lY3Rpb25fd2VpZ2h0ID0gZ2V0VmFyKGNvbm5lY3Rpb24sICd3ZWlnaHQnKTtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9hY3RpdmF0aW9uID0gZ2V0VmFyKGNvbm5lY3Rpb24uZnJvbSwgJ2FjdGl2YXRpb24nKTtcblx0XHRcdFx0XHRcdFx0YnVpbGRTZW50ZW5jZShpbmZsdWVuY2UsICcgKz0gJywgY29ubmVjdGlvbl93ZWlnaHQsICcgKiAnLFxuXHRcdFx0XHRcdFx0XHRcdG5ldXJvbl9hY3RpdmF0aW9uLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHR2YXIgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5ID0gZ2V0VmFyKG5ldXJvbiwgJ2Vycm9yJyxcblx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgKz0gJywgbmV1cm9uX3Jlc3BvbnNpYmlsaXR5LFxuXHRcdFx0XHRcdFx0XHQnICogJywgaW5mbHVlbmNlLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UocmVzcG9uc2liaWxpdHksICcgKj0gJywgZGVyaXZhdGl2ZSxcblx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHRmb3IgKHZhciBpZCBpbiB0aGlzLmNvbm5lY3Rpb25zLmlucHV0cykge1xuXHRcdFx0XHRcdFx0dmFyIGlucHV0ID0gdGhpcy5jb25uZWN0aW9ucy5pbnB1dHNbaWRdO1xuXHRcdFx0XHRcdFx0dmFyIGdyYWRpZW50ID0gZ2V0VmFyKCdhdXgnKTtcblx0XHRcdFx0XHRcdGJ1aWxkU2VudGVuY2UoZ3JhZGllbnQsICcgPSAwJywgc3RvcmVfcHJvcGFnYXRpb24pO1xuXHRcdFx0XHRcdFx0Zm9yICh2YXIgaWQgaW4gdGhpcy50cmFjZS5leHRlbmRlZCkge1xuXHRcdFx0XHRcdFx0XHR2YXIgbmV1cm9uID0gdGhpcy5uZWlnaGJvb3JzW2lkXTtcblx0XHRcdFx0XHRcdFx0dmFyIG5ldXJvbl9yZXNwb25zaWJpbGl0eSA9IGdldFZhcihuZXVyb24sICdlcnJvcicsXG5cdFx0XHRcdFx0XHRcdFx0J3Jlc3BvbnNpYmlsaXR5JywgbmV1cm9uLmVycm9yLnJlc3BvbnNpYmlsaXR5KTtcblx0XHRcdFx0XHRcdFx0dmFyIHh0cmFjZSA9IGdldFZhcih0aGlzLCAndHJhY2UnLCAnZXh0ZW5kZWQnLCBuZXVyb24uSUQsXG5cdFx0XHRcdFx0XHRcdFx0aW5wdXQuSUQsIHRoaXMudHJhY2UuZXh0ZW5kZWRbbmV1cm9uLklEXVtpbnB1dC5JRF0pO1xuXHRcdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGdyYWRpZW50LCAnICs9ICcsIG5ldXJvbl9yZXNwb25zaWJpbGl0eSwgJyAqICcsXG5cdFx0XHRcdFx0XHRcdFx0eHRyYWNlLCBzdG9yZV9wcm9wYWdhdGlvbik7XG5cdFx0XHRcdFx0XHR9XG5cdFx0XHRcdFx0XHR2YXIgaW5wdXRfd2VpZ2h0ID0gZ2V0VmFyKGlucHV0LCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdFx0XHRidWlsZFNlbnRlbmNlKGlucHV0X3dlaWdodCwgJyArPSAnLCByYXRlLCAnICogJywgZ3JhZGllbnQsXG5cdFx0XHRcdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHRcdGJ1aWxkU2VudGVuY2UoYmlhcywgJyArPSAnLCByYXRlLCAnICogJywgcmVzcG9uc2liaWxpdHksXG5cdFx0XHRcdHN0b3JlX3Byb3BhZ2F0aW9uKTtcblx0XHR9XG5cdFx0cmV0dXJuIHtcblx0XHRcdG1lbW9yeTogdmFySUQsXG5cdFx0XHRuZXVyb25zOiBuZXVyb25zICsgMSxcblx0XHRcdGlucHV0czogaW5wdXRzLFxuXHRcdFx0b3V0cHV0czogb3V0cHV0cyxcblx0XHRcdHRhcmdldHM6IHRhcmdldHMsXG5cdFx0XHR2YXJpYWJsZXM6IHZhcmlhYmxlcyxcblx0XHRcdGFjdGl2YXRpb25fc2VudGVuY2VzOiBhY3RpdmF0aW9uX3NlbnRlbmNlcyxcblx0XHRcdHRyYWNlX3NlbnRlbmNlczogdHJhY2Vfc2VudGVuY2VzLFxuXHRcdFx0cHJvcGFnYXRpb25fc2VudGVuY2VzOiBwcm9wYWdhdGlvbl9zZW50ZW5jZXMsXG5cdFx0XHRsYXllcnM6IGxheWVyc1xuXHRcdH1cblx0fVxufVxuXG5leHBvcnQgbW9kdWxlIE5ldXJvbiB7XG5cblx0ZXhwb3J0IGludGVyZmFjZSBJTmV1cm9uQ29ubmVjdGlvbnMge1xuXHRcdGlucHV0czogU3luYXB0aWMuRGljdGlvbmFyeTxOZXVyb24uQ29ubmVjdGlvbj47XG5cdFx0cHJvamVjdGVkOiB7fTtcblx0XHRnYXRlZDoge307XG5cdH1cblxuXHRleHBvcnQgY2xhc3MgQ29ubmVjdGlvbiB7XG5cdFx0SUQgPSBDb25uZWN0aW9uLnVpZCgpO1xuXHRcdGZyb207XG5cdFx0dG87XG5cdFx0Z2FpbjogbnVtYmVyID0gMTtcblx0XHR3ZWlnaHQ6IG51bWJlciA9IDA7XG5cdFx0Z2F0ZXI6IGFueSA9IG51bGw7XG5cdFx0Y29uc3RydWN0b3IoZnJvbSwgdG8sIHdlaWdodD86IG51bWJlcikge1xuXHRcdFx0aWYgKCFmcm9tIHx8ICF0bylcblx0XHRcdFx0dGhyb3cgXCJDb25uZWN0aW9uIEVycm9yOiBJbnZhbGlkIG5ldXJvbnNcIjtcblx0XHRcdHRoaXMuZnJvbSA9IGZyb207XG5cdFx0XHR0aGlzLnRvID0gdG87XG5cdFx0XHR0aGlzLndlaWdodCA9IHR5cGVvZiB3ZWlnaHQgPT0gJ3VuZGVmaW5lZCcgfHwgaXNOYU4od2VpZ2h0KSA/IE1hdGgucmFuZG9tKCkgKiAuMiAtIC4xIDpcblx0XHRcdFx0d2VpZ2h0O1xuXHRcdH1cblx0fVxuXG5cdGV4cG9ydCB2YXIgbmV1cm9uUXR5ID0gMDtcblx0ZXhwb3J0IGZ1bmN0aW9uIHVpZCgpOiBudW1iZXIge1xuXHRcdHJldHVybiBuZXVyb25RdHkrKztcblx0fVxuXG5cdGV4cG9ydCBmdW5jdGlvbiBxdWFudGl0eSgpIHtcblx0XHRyZXR1cm4ge1xuXHRcdFx0bmV1cm9uczogbmV1cm9uUXR5LFxuXHRcdFx0Y29ubmVjdGlvbnM6IENvbm5lY3Rpb24uY29ubmVjdGlvblF0eVxuXHRcdH1cblx0fVxufVxuXG5leHBvcnQgbW9kdWxlIE5ldXJvbi5Db25uZWN0aW9uIHtcblx0ZXhwb3J0IHZhciBjb25uZWN0aW9uUXR5ID0gMDtcblx0ZXhwb3J0IGZ1bmN0aW9uIHVpZCgpOiBudW1iZXIge1xuXHRcdHJldHVybiBjb25uZWN0aW9uUXR5Kys7XG5cdH1cbn0iXSwic291cmNlUm9vdCI6Ii9zb3VyY2UvIn0=