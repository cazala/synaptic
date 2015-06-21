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
        hardcode += "var F = Float64Array ? new Float64Array(" + optimized.memory + ") : []; ";
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
        hardcode += "var ownership = function(memoryBuffer){\nF = memoryBuffer;\nthis.memory = F;\n};\n";
        hardcode += "return {\nmemory: F,\nactivate: activate,\npropagate: propagate,\nownership: ownership\n};";
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
        for (var i in list) {
            var neuron = list[i].neuron;
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
            copy.squash = neuron.squash == Squash.LOGISTIC ? "LOGISTIC" : neuron.squash == Squash.TANH ? "TANH" : neuron.squash == Squash.IDENTITY ? "IDENTITY" : neuron.squash == Squash.HLIM ? "HLIM" : null;
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
        for (var i in list) {
            var neuron = list[i].neuron;
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
                    gater: neuron.selfconnection.gater ? ids[neuron.selfconnection.gater.ID] : null,
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
                        code += "    " + fakeNode + " [label = \"\", shape = point, width = 0.01, height = 0.01]\n";
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
        for (var i in data.inputs)
            activation += "F[" + data.inputs[i] + "] = input[" + i + "];\n";
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
        var hardcode = "var inputs = " + this.optimized.data.inputs.length + ";\n";
        hardcode += "var outputs = " + this.optimized.data.outputs.length + ";\n";
        hardcode += "var F = null;\n";
        hardcode += "var activate = " + this.optimized.activate.toString() + ";\n";
        hardcode += "var propagate = " + this.optimized.propagate.toString() + ";\n";
        hardcode += "onmessage = function(e){\n";
        hardcode += "F = e.data.memoryBuffer;\n";
        hardcode += "if (e.data.action == 'activate'){\n";
        hardcode += "if (e.data.input.length == inputs){\n";
        hardcode += "postMessage( { action: 'activate', output: activate(e.data.input), memoryBuffer: F }, [F.buffer]);\n";
        hardcode += "}\n}\nelse if (e.data.action == 'propagate'){\n";
        hardcode += "propagate(e.data.rate, e.data.target);\n";
        hardcode += "postMessage({ action: 'propagate', memoryBuffer: F }, [F.buffer]);\n";
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
            neuron.squash = config.squash in Squash ? Squash[config.squash] : Squash.LOGISTIC;
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

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9uZXR3b3JrLnRzIl0sIm5hbWVzIjpbIk5ldHdvcmsiLCJOZXR3b3JrLmNvbnN0cnVjdG9yIiwiTmV0d29yay5hY3RpdmF0ZSIsIk5ldHdvcmsucHJvcGFnYXRlIiwiTmV0d29yay5wcm9qZWN0IiwiTmV0d29yay5nYXRlIiwiTmV0d29yay5jbGVhciIsIk5ldHdvcmsucmVzZXQiLCJOZXR3b3JrLm9wdGltaXplIiwiTmV0d29yay5yZXN0b3JlIiwiTmV0d29yay5uZXVyb25zIiwiTmV0d29yay5pbnB1dHMiLCJOZXR3b3JrLm91dHB1dHMiLCJOZXR3b3JrLnNldCIsIk5ldHdvcmsuc2V0T3B0aW1pemUiLCJOZXR3b3JrLnRvSlNPTiIsIk5ldHdvcmsudG9Eb3QiLCJOZXR3b3JrLnN0YW5kYWxvbmUiLCJOZXR3b3JrLndvcmtlciIsIk5ldHdvcmsuY2xvbmUiLCJOZXR3b3JrLmZyb21KU09OIl0sIm1hcHBpbmdzIjoiQUFBQSxJQUFPLEtBQUssV0FBVyxTQUFTLENBQUMsQ0FBQztBQUNsQyxJQUFPLE1BQU0sV0FBVyxVQUFVLENBQUMsQ0FBQztBQUVwQyxJQUFPLE9BQU8sV0FBVyxVQUFVLENBQUMsQ0FBQztBQVNyQyxJQUFhLE9BQU87SUFPbkJBLFNBUFlBLE9BQU9BLENBT1BBLE1BQU1BO1FBTmxCQyxjQUFTQSxHQUFHQSxJQUFJQSxDQUFDQTtRQUNqQkEsV0FBTUEsR0FBR0E7WUFDUkEsS0FBS0EsRUFBRUEsSUFBSUE7WUFDWEEsTUFBTUEsRUFBRUEsRUFBRUE7WUFDVkEsTUFBTUEsRUFBRUEsSUFBSUE7U0FDWkEsQ0FBQ0E7UUFFREEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsT0FBT0EsTUFBTUEsSUFBSUEsV0FBV0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDbENBLElBQUlBLENBQUNBLE1BQU1BLEdBQUdBLE1BQU1BLElBQUlBO2dCQUN2QkEsS0FBS0EsRUFBRUEsSUFBSUE7Z0JBQ1hBLE1BQU1BLEVBQUVBLEVBQUVBO2dCQUNWQSxNQUFNQSxFQUFFQSxJQUFJQTthQUNaQSxDQUFDQTtZQUNGQSxJQUFJQSxDQUFDQSxTQUFTQSxHQUFHQSxJQUFJQSxDQUFDQTtRQUN2QkEsQ0FBQ0E7SUFDRkEsQ0FBQ0E7SUFFREQsZ0VBQWdFQTtJQUNoRUEsMEJBQVFBLEdBQVJBLFVBQVNBLEtBQUtBO1FBRWJFLEVBQUVBLENBQUNBLENBQUNBLElBQUlBLENBQUNBLFNBQVNBLEtBQUtBLEtBQUtBLENBQUNBLENBQUNBLENBQUNBO1lBQzlCQSxJQUFJQSxDQUFDQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxRQUFRQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQTtZQUNsQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsS0FBS0EsSUFBSUEsSUFBSUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsTUFBTUEsQ0FBQ0E7Z0JBQ3BDQSxJQUFJQSxDQUFDQSxNQUFNQSxDQUFDQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQSxRQUFRQSxFQUFFQSxDQUFDQTtZQUN0Q0EsTUFBTUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsUUFBUUEsRUFBRUEsQ0FBQ0E7UUFDdENBLENBQUNBO1FBQ0RBLElBQUlBLENBQUNBLENBQUNBO1lBQ0xBLEVBQUVBLENBQUNBLENBQUNBLElBQUlBLENBQUNBLFNBQVNBLElBQUlBLElBQUlBLENBQUNBO2dCQUMxQkEsSUFBSUEsQ0FBQ0EsUUFBUUEsRUFBRUEsQ0FBQ0E7WUFDakJBLE1BQU1BLENBQUNBLElBQUlBLENBQUNBLFNBQVNBLENBQUNBLFFBQVFBLENBQUNBLEtBQUtBLENBQUNBLENBQUNBO1FBQ3ZDQSxDQUFDQTtJQUNGQSxDQUFDQTtJQUVERiw0Q0FBNENBO0lBQzVDQSwyQkFBU0EsR0FBVEEsVUFBVUEsSUFBWUEsRUFBRUEsTUFBT0E7UUFFOUJHLEVBQUVBLENBQUNBLENBQUNBLElBQUlBLENBQUNBLFNBQVNBLEtBQUtBLEtBQUtBLENBQUNBLENBQUNBLENBQUNBO1lBQzlCQSxJQUFJQSxDQUFDQSxNQUFN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QUFHQSxFQUFFQSxDQUFDQTtRQUVqQkEsSUFBSUEsTUFBTUEsR0FBR0E7WUFDWkEsS0FBS0EsRUFBRUEsSUFBSUEsS0FBS0EsQ0FBQ0EsS0FBS0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDekJBLE1BQU1BLEVBQUVBLEVBQUVBO1lBQ1ZBLE1BQU1BLEVBQUVBLElBQUlBLEtBQUtBLENBQUNBLEtBQUtBLENBQUNBLENBQUNBLENBQUNBO1NBQzFCQSxDQUFBQTtRQUdEQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxJQUFJQSxJQUFJQSxDQUFDQSxPQUFPQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUM1QkEsSUFBSUEsTUFBTUEsR0FBR0EsSUFBSUEsQ0FBQ0EsT0FBT0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFFN0JBLElBQUlBLE1BQU1BLEdBQUdBLElBQUlBLE9BQU9BLENBQUNBLE1BQU1BLEVBQUVBLENBQUNBO1lBQ2xDQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxXQUFXQSxHQUFHQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxXQUFXQSxDQUFDQTtZQUNwREEsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsUUFBUUEsR0FBR0EsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsUUFBUUEsQ0FBQ0E7WUFDOUNBLE1BQU1BLENBQUNBLEtBQUtBLEdBQUdBLE1BQU1BLENBQUNBLEtBQUtBLENBQUNBO1lBQzVCQSxNQUFNQSxDQUFDQSxHQUFHQSxHQUFHQSxNQUFNQSxDQUFDQSxHQUFHQSxDQUFDQTtZQUN4QkEsTUFBTUEsQ0FBQ0EsVUFBVUEsR0FBR0EsTUFBTUEsQ0FBQ0EsVUFBVUEsQ0FBQ0E7WUFDdENBLE1BQU1BLENBQUNBLElBQUlBLEdBQUdBLE1BQU1BLENBQUNBLElBQUlBLENBQUNBO1lBQzFCQSxNQUFNQSxDQUFDQSxNQUFNQSxHQUFHQSxNQUFNQSxDQUFDQSxNQUFNQSxJQUFJQSxNQUFNQSxHQUFHQSxNQUFNQSxDQUFDQSxNQUFNQSxDQUFDQSxNQUFNQSxDQUFDQSxHQUM5REEsTUFBTUEsQ0FBQ0EsUUFBUUEsQ0FBQ0E7WUFDakJBLE9BQU9BLENBQUNBLElBQUlBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBO1lBRXJCQSxFQUFFQSxDQUFDQSxDQUFDQSxNQUFNQSxDQUFDQSxLQUFLQSxJQUFJQSxPQUFPQSxDQUFDQTtnQkFDM0JBLE1BQU1BLENBQUNBLEtBQUtBLENBQUNBLEdBQUdBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBO1lBQzFCQSxJQUFJQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxNQUFNQSxDQUFDQSxLQUFLQSxJQUFJQSxRQUFRQSxDQUFDQTtnQkFDakNBLE1BQU1BLENBQUNBLE1BQU1BLENBQUNBLEdBQUdBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBO1lBQzNCQSxJQUFJQSxDQUFDQSxDQUFDQTtnQkFDTEEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsT0FBT0EsTUFBTUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsSUFBSUEsV0FBV0EsQ0FBQ0E7b0JBQ3JEQSxNQUFNQSxDQUFDQSxNQUFNQSxDQUFDQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxHQUFHQSxJQUFJQSxLQUFLQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDbERBLE1BQU1BLENBQUNBLE1BQU1BLENBQUNBLE1BQU1BLENBQUNBLEtBQUtBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBO1lBQ3pDQSxDQUFDQTtRQUNGQSxDQUFDQTtRQUVEQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxJQUFJQSxJQUFJQSxDQUFDQSxXQUFXQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUNoQ0EsSUFBSUEsTUFBTUEsR0FBR0EsSUFBSUEsQ0FBQ0EsV0FBV0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDakNBLElBQUlBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLE1BQU1BLENBQUNBLElBQUlBLENBQUNBLENBQUNBO1lBQ2hDQSxJQUFJQSxFQUFFQSxHQUFHQSxPQUFPQSxDQUFDQSxNQUFNQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQTtZQUM1QkEsSUFBSUEsTUFBTUEsR0FBR0EsTUFBTUEsQ0FBQ0EsTUFBTUEsQ0FBQUE7WUFDMUJBLElBQUlBLEtBQUtBLEdBQUdBLE9BQU9BLENBQUNBLE1BQU1BLENBQUNBLEtBQUtBLENBQUNBLENBQUNBO1lBRWxDQSxJQUFJQSxVQUFVQSxHQUFHQSxJQUFJQSxDQUFDQSxPQUFPQSxDQUFDQSxFQUFFQSxFQUFFQSxNQUFNQSxDQUFDQSxDQUFDQTtZQUMxQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0EsS0FBS0EsQ0FBQ0E7Z0JBQ1RBLEtBQUtBLENBQUNBLElBQUlBLENBQUNBLFVBQVVBLENBQUNBLENBQUNBO1FBQ3pCQSxDQUFDQTtRQUVEQSxNQUFNQSxDQUFDQSxJQUFJQSxPQUFPQSxDQUFDQSxNQUFNQSxDQUFDQSxDQUFDQTtJQUM1QkEsQ0FBQ0E7SUFDRnBCLGNBQUNBO0FBQURBLENBbm1CQSxBQW1tQkNBLElBQUE7QUFubUJZLGVBQU8sR0FBUCxPQW1tQlosQ0FBQTtBQU9BIiwiZmlsZSI6InNyYy9uZXR3b3JrLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IGxheWVyID0gcmVxdWlyZSgnLi9sYXllcicpO1xuaW1wb3J0IFNxdWFzaCA9IHJlcXVpcmUoJy4vc3F1YXNoJyk7XG5pbXBvcnQgU3luYXB0aWMgPSByZXF1aXJlKCcuL3N5bmFwdGljJyk7XG5pbXBvcnQgX25ldXJvbiA9IHJlcXVpcmUoJy4vbmV1cm9uJyk7XG5cbi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBORVRXT1JLXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuXG5kZWNsYXJlIGZ1bmN0aW9uIGVzY2FwZShhOiBzdHJpbmcpOiBzdHJpbmc7XG5cblxuZXhwb3J0IGNsYXNzIE5ldHdvcmsge1xuXHRvcHRpbWl6ZWQgPSBudWxsO1xuXHRsYXllcnMgPSB7XG5cdFx0aW5wdXQ6IG51bGwsXG5cdFx0aGlkZGVuOiB7fSxcblx0XHRvdXRwdXQ6IG51bGxcblx0fTtcblx0Y29uc3RydWN0b3IobGF5ZXJzKSB7XG5cdFx0aWYgKHR5cGVvZiBsYXllcnMgIT0gJ3VuZGVmaW5lZCcpIHtcblx0XHRcdHRoaXMubGF5ZXJzID0gbGF5ZXJzIHx8IHtcblx0XHRcdFx0aW5wdXQ6IG51bGwsXG5cdFx0XHRcdGhpZGRlbjoge30sXG5cdFx0XHRcdG91dHB1dDogbnVsbFxuXHRcdFx0fTtcblx0XHRcdHRoaXMub3B0aW1pemVkID0gbnVsbDtcblx0XHR9XG5cdH1cblxuXHQvLyBmZWVkLWZvcndhcmQgYWN0aXZhdGlvbiBvZiBhbGwgdGhlIGxheWVycyB0byBwcm9kdWNlIGFuIG91cHV0XG5cdGFjdGl2YXRlKGlucHV0KSB7XG5cblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQgPT09IGZhbHNlKSB7XG5cdFx0XHR0aGlzLmxheWVycy5pbnB1dC5hY3RpdmF0ZShpbnB1dCk7XG5cdFx0XHRmb3IgKHZhciBsYXllciBpbiB0aGlzLmxheWVycy5oaWRkZW4pXG5cdFx0XHRcdHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl0uYWN0aXZhdGUoKTtcblx0XHRcdHJldHVybiB0aGlzLmxheWVycy5vdXRwdXQuYWN0aXZhdGUoKTtcblx0XHR9XG5cdFx0ZWxzZSB7XG5cdFx0XHRpZiAodGhpcy5vcHRpbWl6ZWQgPT0gbnVsbClcblx0XHRcdFx0dGhpcy5vcHRpbWl6ZSgpO1xuXHRcdFx0cmV0dXJuIHRoaXMub3B0aW1pemVkLmFjdGl2YXRlKGlucHV0KTtcblx0XHR9XG5cdH1cblxuXHQvLyBiYWNrLXByb3BhZ2F0ZSB0aGUgZXJyb3IgdGhydSB0aGUgbmV0d29ya1xuXHRwcm9wYWdhdGUocmF0ZTogbnVtYmVyLCB0YXJnZXQ/KSB7XG5cblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQgPT09IGZhbHNlKSB7XG5cdFx0XHR0aGlzLmxheWVycy5vdXRwdXQucHJvcGFnYXRlKHJhdGUsIHRhcmdldCk7XG5cdFx0XHR2YXIgcmV2ZXJzZSA9IFtdO1xuXHRcdFx0Zm9yICh2YXIgbGF5ZXIgaW4gdGhpcy5sYXllcnMuaGlkZGVuKVxuXHRcdFx0XHRyZXZlcnNlLnB1c2godGhpcy5sYXllcnMuaGlkZGVuW2xheWVyXSk7XG5cdFx0XHRyZXZlcnNlLnJldmVyc2UoKTtcblx0XHRcdGZvciAodmFyIGxheWVyIGluIHJldmVyc2UpXG5cdFx0XHRcdHJldmVyc2VbbGF5ZXJdLnByb3BhZ2F0ZShyYXRlKTtcblx0XHR9XG5cdFx0ZWxzZSB7XG5cdFx0XHRpZiAodGhpcy5vcHRpbWl6ZWQgPT0gbnVsbClcblx0XHRcdFx0dGhpcy5vcHRpbWl6ZSgpO1xuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucHJvcGFnYXRlKHJhdGUsIHRhcmdldCk7XG5cdFx0fVxuXHR9XG5cblx0Ly8gcHJvamVjdCBhIGNvbm5lY3Rpb24gdG8gYW5vdGhlciB1bml0IChlaXRoZXIgYSBuZXR3b3JrIG9yIGEgbGF5ZXIpXG5cdHByb2plY3QodW5pdCwgdHlwZSwgd2VpZ2h0cykge1xuXG5cdFx0aWYgKHRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcblxuXHRcdGlmICh1bml0IGluc3RhbmNlb2YgTmV0d29yaylcblx0XHRcdHJldHVybiB0aGlzLmxheWVycy5vdXRwdXQucHJvamVjdCh1bml0LmxheWVycy5pbnB1dCwgdHlwZSwgd2VpZ2h0cyk7XG5cblx0XHRpZiAodW5pdCBpbnN0YW5jZW9mIGxheWVyLkxheWVyKVxuXHRcdFx0cmV0dXJuIHRoaXMubGF5ZXJzLm91dHB1dC5wcm9qZWN0KHVuaXQsIHR5cGUsIHdlaWdodHMpO1xuXG5cdFx0dGhyb3cgXCJJbnZhbGlkIGFyZ3VtZW50LCB5b3UgY2FuIG9ubHkgcHJvamVjdCBjb25uZWN0aW9ucyB0byBMQVlFUlMgYW5kIE5FVFdPUktTIVwiO1xuXHR9XG5cblx0Ly8gbGV0IHRoaXMgbmV0d29yayBnYXRlIGEgY29ubmVjdGlvblxuXHRnYXRlKGNvbm5lY3Rpb24sIHR5cGUpIHtcblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplZC5yZXNldCgpO1xuXHRcdHRoaXMubGF5ZXJzLm91dHB1dC5nYXRlKGNvbm5lY3Rpb24sIHR5cGUpO1xuXHR9XG5cblx0Ly8gY2xlYXIgYWxsIGVsZWdpYmlsaXR5IHRyYWNlcyBhbmQgZXh0ZW5kZWQgZWxlZ2liaWxpdHkgdHJhY2VzICh0aGUgbmV0d29yayBmb3JnZXRzIGl0cyBjb250ZXh0LCBidXQgbm90IHdoYXQgd2FzIHRyYWluZWQpXG5cdGNsZWFyKCkge1xuXG5cdFx0dGhpcy5yZXN0b3JlKCk7XG5cblx0XHR2YXIgaW5wdXRMYXllciA9IHRoaXMubGF5ZXJzLmlucHV0LFxuXHRcdFx0b3V0cHV0TGF5ZXIgPSB0aGlzLmxheWVycy5vdXRwdXQ7XG5cblx0XHRpbnB1dExheWVyLmNsZWFyKCk7XG5cdFx0Zm9yICh2YXIgbGF5ZXIgaW4gdGhpcy5sYXllcnMuaGlkZGVuKSB7XG5cdFx0XHR2YXIgaGlkZGVuTGF5ZXIgPSB0aGlzLmxheWVycy5oaWRkZW5bbGF5ZXJdO1xuXHRcdFx0aGlkZGVuTGF5ZXIuY2xlYXIoKTtcblx0XHR9XG5cdFx0b3V0cHV0TGF5ZXIuY2xlYXIoKTtcblxuXHRcdGlmICh0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cdH1cblxuXHQvLyByZXNldCBhbGwgd2VpZ2h0cyBhbmQgY2xlYXIgYWxsIHRyYWNlcyAoZW5kcyB1cCBsaWtlIGEgbmV3IG5ldHdvcmspXG5cdHJlc2V0KCkge1xuXG5cdFx0dGhpcy5yZXN0b3JlKCk7XG5cblx0XHR2YXIgaW5wdXRMYXllciA9IHRoaXMubGF5ZXJzLmlucHV0LFxuXHRcdFx0b3V0cHV0TGF5ZXIgPSB0aGlzLmxheWVycy5vdXRwdXQ7XG5cblx0XHRpbnB1dExheWVyLnJlc2V0KCk7XG5cdFx0Zm9yICh2YXIgbGF5ZXIgaW4gdGhpcy5sYXllcnMuaGlkZGVuKSB7XG5cdFx0XHR2YXIgaGlkZGVuTGF5ZXIgPSB0aGlzLmxheWVycy5oaWRkZW5bbGF5ZXJdO1xuXHRcdFx0aGlkZGVuTGF5ZXIucmVzZXQoKTtcblx0XHR9XG5cdFx0b3V0cHV0TGF5ZXIucmVzZXQoKTtcblxuXHRcdGlmICh0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cdH1cblxuXHQvLyBoYXJkY29kZXMgdGhlIGJlaGF2aW91ciBvZiB0aGUgd2hvbGUgbmV0d29yayBpbnRvIGEgc2luZ2xlIG9wdGltaXplZCBmdW5jdGlvblxuXHRvcHRpbWl6ZSgpIHtcblxuXHRcdHZhciB0aGF0ID0gdGhpcztcblx0XHR2YXIgb3B0aW1pemVkOiBTeW5hcHRpYy5JQ29tcGlsZWRQYXJhbWV0ZXJzID0ge307XG5cdFx0dmFyIG5ldXJvbnMgPSB0aGlzLm5ldXJvbnMoKTtcblxuXHRcdGZvciAodmFyIGkgaW4gbmV1cm9ucykge1xuXHRcdFx0dmFyIG5ldXJvbiA9IG5ldXJvbnNbaV0ubmV1cm9uO1xuXHRcdFx0dmFyIGxheWVyID0gbmV1cm9uc1tpXS5sYXllcjtcblx0XHRcdC8qXG5cdFx0XHRGSVhNRTogZG9lcyB0aGlzIHdvcmtlZCBvbmNlP1xuXHRcdFx0XG5cdFx0XHR3aGlsZSAobmV1cm9uLm5ldXJvbilcblx0XHRcdFx0bmV1cm9uID0gbmV1cm9uLm5ldXJvbjtcblx0XHRcdCovXG5cblx0XHRcdG9wdGltaXplZCA9IG5ldXJvbi5vcHRpbWl6ZShvcHRpbWl6ZWQsIGxheWVyKTtcblx0XHR9XG5cblx0XHRmb3IgKHZhciBpIGluIG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMpXG5cdFx0XHRvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzW2ldLnJldmVyc2UoKTtcblx0XHRvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzLnJldmVyc2UoKTtcblxuXHRcdHZhciBoYXJkY29kZSA9IFwiXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgRiA9IEZsb2F0NjRBcnJheSA/IG5ldyBGbG9hdDY0QXJyYXkoXCIgKyBvcHRpbWl6ZWQubWVtb3J5ICtcblx0XHRcIikgOiBbXTsgXCI7XG5cdFx0Zm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQudmFyaWFibGVzKVxuXHRcdFx0aGFyZGNvZGUgKz0gXCJGW1wiICsgb3B0aW1pemVkLnZhcmlhYmxlc1tpXS5pZCArIFwiXSA9IFwiICsgKG9wdGltaXplZC52YXJpYWJsZXNbXG5cdFx0XHRcdGldLnZhbHVlIHx8IDApICsgXCI7IFwiO1xuXHRcdGhhcmRjb2RlICs9IFwidmFyIGFjdGl2YXRlID0gZnVuY3Rpb24oaW5wdXQpe1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwiaW5mbHVlbmNlcyA9IFtdO1wiO1xuXHRcdGZvciAodmFyIGkgaW4gb3B0aW1pemVkLmlucHV0cylcblx0XHRcdGhhcmRjb2RlICs9IFwiRltcIiArIG9wdGltaXplZC5pbnB1dHNbaV0gKyBcIl0gPSBpbnB1dFtcIiArIGkgKyBcIl07IFwiO1xuXHRcdGZvciAodmFyIGN1cnJlbnRMYXllciBpbiBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXMpIHtcblx0XHRcdGlmIChvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXS5sZW5ndGggPiAwKSB7XG5cdFx0XHRcdGZvciAodmFyIGN1cnJlbnROZXVyb24gaW4gb3B0aW1pemVkLmFjdGl2YXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0pIHtcblx0XHRcdFx0XHRoYXJkY29kZSArPSBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXVtjdXJyZW50TmV1cm9uXS5qb2luKFwiIFwiKTtcblx0XHRcdFx0XHRoYXJkY29kZSArPSBvcHRpbWl6ZWQudHJhY2Vfc2VudGVuY2VzW2N1cnJlbnRMYXllcl1bY3VycmVudE5ldXJvbl0uam9pbihcIiBcIik7XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHR9XG5cdFx0aGFyZGNvZGUgKz0gXCIgdmFyIG91dHB1dCA9IFtdOyBcIlxuXHRcdGZvciAodmFyIGkgaW4gb3B0aW1pemVkLm91dHB1dHMpXG5cdFx0XHRoYXJkY29kZSArPSBcIm91dHB1dFtcIiArIGkgKyBcIl0gPSBGW1wiICsgb3B0aW1pemVkLm91dHB1dHNbaV0gKyBcIl07IFwiO1xuXHRcdGhhcmRjb2RlICs9IFwicmV0dXJuIG91dHB1dDsgfTsgXCJcblx0XHRoYXJkY29kZSArPSBcInZhciBwcm9wYWdhdGUgPSBmdW5jdGlvbihyYXRlLCB0YXJnZXQpe1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwiRltcIiArIG9wdGltaXplZC52YXJpYWJsZXMucmF0ZS5pZCArIFwiXSA9IHJhdGU7IFwiO1xuXHRcdGZvciAodmFyIGkgaW4gb3B0aW1pemVkLnRhcmdldHMpXG5cdFx0XHRoYXJkY29kZSArPSBcIkZbXCIgKyBvcHRpbWl6ZWQudGFyZ2V0c1tpXSArIFwiXSA9IHRhcmdldFtcIiArIGkgKyBcIl07IFwiO1xuXHRcdGZvciAodmFyIGN1cnJlbnRMYXllciBpbiBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzKVxuXHRcdFx0Zm9yICh2YXIgY3VycmVudE5ldXJvbiBpbiBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0pXG5cdFx0XHRcdGhhcmRjb2RlICs9IG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXNbY3VycmVudExheWVyXVtjdXJyZW50TmV1cm9uXS5qb2luKFwiIFwiKSArIFwiIFwiO1xuXHRcdGhhcmRjb2RlICs9IFwiIH07XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz1cblx0XHRcInZhciBvd25lcnNoaXAgPSBmdW5jdGlvbihtZW1vcnlCdWZmZXIpe1xcbkYgPSBtZW1vcnlCdWZmZXI7XFxudGhpcy5tZW1vcnkgPSBGO1xcbn07XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz1cblx0XHRcInJldHVybiB7XFxubWVtb3J5OiBGLFxcbmFjdGl2YXRlOiBhY3RpdmF0ZSxcXG5wcm9wYWdhdGU6IHByb3BhZ2F0ZSxcXG5vd25lcnNoaXA6IG93bmVyc2hpcFxcbn07XCI7XG5cdFx0aGFyZGNvZGUgPSBoYXJkY29kZS5zcGxpdChcIjtcIikuam9pbihcIjtcXG5cIik7XG5cblx0XHR2YXIgY29uc3RydWN0b3IgPSBuZXcgRnVuY3Rpb24oaGFyZGNvZGUpO1xuXG5cdFx0dmFyIG5ldHdvcmsgPSBjb25zdHJ1Y3RvcigpO1xuXG5cdFx0bmV0d29yay5kYXRhID0ge1xuXHRcdFx0dmFyaWFibGVzOiBvcHRpbWl6ZWQudmFyaWFibGVzLFxuXHRcdFx0YWN0aXZhdGU6IG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlcyxcblx0XHRcdHByb3BhZ2F0ZTogb3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlcyxcblx0XHRcdHRyYWNlOiBvcHRpbWl6ZWQudHJhY2Vfc2VudGVuY2VzLFxuXHRcdFx0aW5wdXRzOiBvcHRpbWl6ZWQuaW5wdXRzLFxuXHRcdFx0b3V0cHV0czogb3B0aW1pemVkLm91dHB1dHMsXG5cdFx0XHRjaGVja19hY3RpdmF0aW9uOiB0aGlzLmFjdGl2YXRlLFxuXHRcdFx0Y2hlY2tfcHJvcGFnYXRpb246IHRoaXMucHJvcGFnYXRlXG5cdFx0fVxuXG5cdFx0bmV0d29yay5yZXNldCA9IGZ1bmN0aW9uKCkge1xuXHRcdFx0aWYgKHRoYXQub3B0aW1pemVkKSB7XG5cdFx0XHRcdHRoYXQub3B0aW1pemVkID0gbnVsbDtcblx0XHRcdFx0dGhhdC5hY3RpdmF0ZSA9IG5ldHdvcmsuZGF0YS5jaGVja19hY3RpdmF0aW9uO1xuXHRcdFx0XHR0aGF0LnByb3BhZ2F0ZSA9IG5ldHdvcmsuZGF0YS5jaGVja19wcm9wYWdhdGlvbjtcblx0XHRcdH1cblx0XHR9XG5cblx0XHR0aGlzLm9wdGltaXplZCA9IG5ldHdvcms7XG5cdFx0dGhpcy5hY3RpdmF0ZSA9IG5ldHdvcmsuYWN0aXZhdGU7XG5cdFx0dGhpcy5wcm9wYWdhdGUgPSBuZXR3b3JrLnByb3BhZ2F0ZTtcblx0fVxuXG5cdC8vIHJlc3RvcmVzIGFsbCB0aGUgdmFsdWVzIGZyb20gdGhlIG9wdGltaXplZCBuZXR3b3JrIHRoZSB0aGVpciByZXNwZWN0aXZlIG9iamVjdHMgaW4gb3JkZXIgdG8gbWFuaXB1bGF0ZSB0aGUgbmV0d29ya1xuXHRyZXN0b3JlKCkge1xuXHRcdGlmICghdGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHRyZXR1cm47XG5cblx0XHR2YXIgb3B0aW1pemVkID0gdGhpcy5vcHRpbWl6ZWQ7XG5cblx0XHR2YXIgZ2V0VmFsdWUgPSBmdW5jdGlvbiguLi5hcmdzOiBhbnlbXSkge1xuXHRcdFx0dmFyIHVuaXQgPSBhcmdzLnNoaWZ0KCk7XG5cdFx0XHR2YXIgcHJvcCA9IGFyZ3MucG9wKCk7XG5cblx0XHRcdHZhciBpZCA9IHByb3AgKyAnXyc7XG5cdFx0XHRmb3IgKHZhciBwcm9wZXJ0eSBpbiBhcmdzKVxuXHRcdFx0XHRpZCArPSBhcmdzW3Byb3BlcnR5XSArICdfJztcblx0XHRcdGlkICs9IHVuaXQuSUQ7XG5cblx0XHRcdHZhciBtZW1vcnkgPSBvcHRpbWl6ZWQubWVtb3J5O1xuXHRcdFx0dmFyIHZhcmlhYmxlcyA9IG9wdGltaXplZC5kYXRhLnZhcmlhYmxlcztcblxuXHRcdFx0aWYgKGlkIGluIHZhcmlhYmxlcylcblx0XHRcdFx0cmV0dXJuIG1lbW9yeVt2YXJpYWJsZXNbaWRdLmlkXTtcblx0XHRcdHJldHVybiAwO1xuXHRcdH1cblxuXHRcdHZhciBsaXN0ID0gdGhpcy5uZXVyb25zKCk7XG5cblx0XHQvLyBsaW5rIGlkJ3MgdG8gcG9zaXRpb25zIGluIHRoZSBhcnJheVxuXHRcdHZhciBpZHMgPSB7fTtcblx0XHRmb3IgKHZhciBpIGluIGxpc3QpIHtcblx0XHRcdHZhciBuZXVyb24gPSBsaXN0W2ldLm5ldXJvbjtcblx0XHRcdC8qXG5cdFx0XHRGSVhNRTogZG9lcyB0aGlzIHdvcmtlZCBvbmNlP1xuXHRcdFx0XG5cdFx0XHR3aGlsZSAobmV1cm9uLm5ldXJvbilcblx0XHRcdFx0bmV1cm9uID0gbmV1cm9uLm5ldXJvbjtcblx0XHRcdCovXG5cblx0XHRcdG5ldXJvbi5zdGF0ZSA9IGdldFZhbHVlKG5ldXJvbiwgJ3N0YXRlJyk7XG5cdFx0XHRuZXVyb24ub2xkID0gZ2V0VmFsdWUobmV1cm9uLCAnb2xkJyk7XG5cdFx0XHRuZXVyb24uYWN0aXZhdGlvbiA9IGdldFZhbHVlKG5ldXJvbiwgJ2FjdGl2YXRpb24nKTtcblx0XHRcdG5ldXJvbi5iaWFzID0gZ2V0VmFsdWUobmV1cm9uLCAnYmlhcycpO1xuXG5cdFx0XHRmb3IgKHZhciBpbnB1dCBpbiBuZXVyb24udHJhY2UuZWxlZ2liaWxpdHkpXG5cdFx0XHRcdG5ldXJvbi50cmFjZS5lbGVnaWJpbGl0eVtpbnB1dF0gPSBnZXRWYWx1ZShuZXVyb24sICd0cmFjZScsXG5cdFx0XHRcdFx0J2VsZWdpYmlsaXR5JywgaW5wdXQpO1xuXG5cdFx0XHRmb3IgKHZhciBnYXRlZCBpbiBuZXVyb24udHJhY2UuZXh0ZW5kZWQpXG5cdFx0XHRcdGZvciAodmFyIGlucHV0IGluIG5ldXJvbi50cmFjZS5leHRlbmRlZFtnYXRlZF0pXG5cdFx0XHRcdFx0bmV1cm9uLnRyYWNlLmV4dGVuZGVkW2dhdGVkXVtpbnB1dF0gPSBnZXRWYWx1ZShuZXVyb24sICd0cmFjZScsXG5cdFx0XHRcdFx0XHQnZXh0ZW5kZWQnLCBnYXRlZCwgaW5wdXQpO1xuXHRcdH1cblxuXHRcdC8vIGdldCBjb25uZWN0aW9uc1xuXHRcdGZvciAodmFyIGkgaW4gbGlzdCkge1xuXHRcdFx0dmFyIG5ldXJvbiA9IGxpc3RbaV0ubmV1cm9uO1xuXHRcdFx0Lypcblx0XHRcdEZJWE1FOiBkb2VzIHRoaXMgd29ya2VkIG9uY2U/XG5cdFx0XHRcblx0XHRcdHdoaWxlIChuZXVyb24ubmV1cm9uKVxuXHRcdFx0XHRuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuXHRcdFx0Ki9cblxuXHRcdFx0Zm9yICh2YXIgaiBpbiBuZXVyb24uY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG5cdFx0XHRcdHZhciBjb25uZWN0aW9uID0gbmV1cm9uLmNvbm5lY3Rpb25zLnByb2plY3RlZFtqXTtcblx0XHRcdFx0Y29ubmVjdGlvbi53ZWlnaHQgPSBnZXRWYWx1ZShjb25uZWN0aW9uLCAnd2VpZ2h0Jyk7XG5cdFx0XHRcdGNvbm5lY3Rpb24uZ2FpbiA9IGdldFZhbHVlKGNvbm5lY3Rpb24sICdnYWluJyk7XG5cdFx0XHR9XG5cdFx0fVxuXHR9XG5cblx0Ly8gcmV0dXJucyBhbGwgdGhlIG5ldXJvbnMgaW4gdGhlIG5ldHdvcmtcblx0bmV1cm9ucygpOiBOZXR3b3JrLklOZXR3b3JrTmV1cm9uW10ge1xuXHRcdHZhciBuZXVyb25zOiBOZXR3b3JrLklOZXR3b3JrTmV1cm9uW10gPSBbXTtcblxuXHRcdHZhciBpbnB1dExheWVyID0gdGhpcy5sYXllcnMuaW5wdXQubmV1cm9ucygpLFxuXHRcdFx0b3V0cHV0TGF5ZXIgPSB0aGlzLmxheWVycy5vdXRwdXQubmV1cm9ucygpO1xuXG5cdFx0Zm9yICh2YXIgbmV1cm9uIGluIGlucHV0TGF5ZXIpXG5cdFx0XHRuZXVyb25zLnB1c2goe1xuXHRcdFx0XHRuZXVyb246IGlucHV0TGF5ZXJbbmV1cm9uXSxcblx0XHRcdFx0bGF5ZXI6ICdpbnB1dCdcblx0XHRcdH0pO1xuXG5cdFx0Zm9yICh2YXIgbGF5ZXIgaW4gdGhpcy5sYXllcnMuaGlkZGVuKSB7XG5cdFx0XHR2YXIgaGlkZGVuTGF5ZXIgPSB0aGlzLmxheWVycy5oaWRkZW5bbGF5ZXJdLm5ldXJvbnMoKTtcblx0XHRcdGZvciAodmFyIG5ldXJvbiBpbiBoaWRkZW5MYXllcilcblx0XHRcdFx0bmV1cm9ucy5wdXNoKHtcblx0XHRcdFx0XHRuZXVyb246IGhpZGRlbkxheWVyW25ldXJvbl0sXG5cdFx0XHRcdFx0bGF5ZXI6IGxheWVyXG5cdFx0XHRcdH0pO1xuXHRcdH1cblx0XHRmb3IgKHZhciBuZXVyb24gaW4gb3V0cHV0TGF5ZXIpXG5cdFx0XHRuZXVyb25zLnB1c2goe1xuXHRcdFx0XHRuZXVyb246IG91dHB1dExheWVyW25ldXJvbl0sXG5cdFx0XHRcdGxheWVyOiAnb3V0cHV0J1xuXHRcdFx0fSk7XG5cblx0XHRyZXR1cm4gbmV1cm9ucztcblx0fVxuXG5cdC8vIHJldHVybnMgbnVtYmVyIG9mIGlucHV0cyBvZiB0aGUgbmV0d29ya1xuXHRpbnB1dHMoKTogbnVtYmVyIHtcblx0XHRyZXR1cm4gdGhpcy5sYXllcnMuaW5wdXQuc2l6ZTtcblx0fVxuXG5cdC8vIHJldHVybnMgbnVtYmVyIG9mIG91dHB1dHMgb2YgaHRlIG5ldHdvcmtcblx0b3V0cHV0cygpOiBudW1iZXIge1xuXHRcdHJldHVybiB0aGlzLmxheWVycy5vdXRwdXQuc2l6ZTtcblx0fVxuXG5cdC8vIHNldHMgdGhlIGxheWVycyBvZiB0aGUgbmV0d29ya1xuXHRzZXQobGF5ZXJzKSB7XG5cblx0XHR0aGlzLmxheWVycyA9IGxheWVycztcblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplZC5yZXNldCgpO1xuXHR9XG5cblx0c2V0T3B0aW1pemUoYm9vbCkge1xuXHRcdHRoaXMucmVzdG9yZSgpO1xuXHRcdGlmICh0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cdFx0dGhpcy5vcHRpbWl6ZWQgPSBib29sID8gbnVsbCA6IGZhbHNlO1xuXHR9XG5cblx0Ly8gcmV0dXJucyBhIGpzb24gdGhhdCByZXByZXNlbnRzIGFsbCB0aGUgbmV1cm9ucyBhbmQgY29ubmVjdGlvbnMgb2YgdGhlIG5ldHdvcmtcblx0dG9KU09OKGlnbm9yZVRyYWNlcykge1xuXG5cdFx0dGhpcy5yZXN0b3JlKCk7XG5cblx0XHR2YXIgbGlzdCA9IHRoaXMubmV1cm9ucygpO1xuXHRcdHZhciBuZXVyb25zID0gW107XG5cdFx0dmFyIGNvbm5lY3Rpb25zID0gW107XG5cblx0XHQvLyBsaW5rIGlkJ3MgdG8gcG9zaXRpb25zIGluIHRoZSBhcnJheVxuXHRcdHZhciBpZHMgPSB7fTtcblx0XHRmb3IgKHZhciBpIGluIGxpc3QpIHtcblx0XHRcdHZhciBuZXVyb24gPSBsaXN0W2ldLm5ldXJvbjtcblx0XHRcdC8qXG5cdFx0XHRGSVhNRTogZG9lcyB0aGlzIHdvcmtlZCBvbmNlP1xuXHRcdFx0XG5cdFx0XHR3aGlsZSAobmV1cm9uLm5ldXJvbilcblx0XHRcdFx0bmV1cm9uID0gbmV1cm9uLm5ldXJvbjtcblx0XHRcdCovXG5cblx0XHRcdGlkc1tuZXVyb24uSURdID0gaTtcblxuXHRcdFx0dmFyIGNvcHkgPSB7XG5cdFx0XHRcdHRyYWNlOiB7XG5cdFx0XHRcdFx0ZWxlZ2liaWxpdHk6IHt9LFxuXHRcdFx0XHRcdGV4dGVuZGVkOiB7fVxuXHRcdFx0XHR9LFxuXHRcdFx0XHRzdGF0ZTogbmV1cm9uLnN0YXRlLFxuXHRcdFx0XHRvbGQ6IG5ldXJvbi5vbGQsXG5cdFx0XHRcdGFjdGl2YXRpb246IG5ldXJvbi5hY3RpdmF0aW9uLFxuXHRcdFx0XHRiaWFzOiBuZXVyb24uYmlhcyxcblx0XHRcdFx0bGF5ZXI6IGxpc3RbaV0ubGF5ZXIsXG5cdFx0XHRcdHNxdWFzaDogbnVsbFxuXHRcdFx0fTtcblxuXHRcdFx0Y29weS5zcXVhc2ggPSBuZXVyb24uc3F1YXNoID09IFNxdWFzaC5MT0dJU1RJQyA/IFwiTE9HSVNUSUNcIiA6XG5cdFx0XHRcdG5ldXJvbi5zcXVhc2ggPT0gU3F1YXNoLlRBTkggPyBcIlRBTkhcIiA6XG5cdFx0XHRcdFx0bmV1cm9uLnNxdWFzaCA9PSBTcXVhc2guSURFTlRJVFkgPyBcIklERU5USVRZXCIgOlxuXHRcdFx0XHRcdFx0bmV1cm9uLnNxdWFzaCA9PSBTcXVhc2guSExJTSA/IFwiSExJTVwiIDpcblx0XHRcdFx0XHRcdFx0bnVsbDtcblxuXHRcdFx0bmV1cm9ucy5wdXNoKGNvcHkpO1xuXHRcdH1cblxuXHRcdGlmICghaWdub3JlVHJhY2VzKVxuXHRcdFx0Zm9yICh2YXIgaSBpbiBuZXVyb25zKSB7XG5cdFx0XHRcdHZhciBjb3BpZWROZXVyb24gPSBuZXVyb25zW2ldO1xuXG5cdFx0XHRcdGZvciAodmFyIGlucHV0IGluIG5ldXJvbi50cmFjZS5lbGVnaWJpbGl0eSlcblx0XHRcdFx0XHRjb3BpZWROZXVyb24udHJhY2UuZWxlZ2liaWxpdHlbaW5wdXRdID0gbmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0XTtcblxuXHRcdFx0XHRmb3IgKHZhciBnYXRlZCBpbiBuZXVyb24udHJhY2UuZXh0ZW5kZWQpIHtcblx0XHRcdFx0XHRjb3BpZWROZXVyb24udHJhY2UuZXh0ZW5kZWRbZ2F0ZWRdID0ge307XG5cdFx0XHRcdFx0Zm9yICh2YXIgaW5wdXQgaW4gbmV1cm9uLnRyYWNlLmV4dGVuZGVkW2dhdGVkXSlcblx0XHRcdFx0XHRcdGNvcGllZE5ldXJvbi50cmFjZS5leHRlbmRlZFtpZHNbZ2F0ZWRdXVtpbnB1dF0gPSBuZXVyb24udHJhY2UuZXh0ZW5kZWRbXG5cdFx0XHRcdFx0XHRnYXRlZF1baW5wdXRdO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cblx0XHQvLyBnZXQgY29ubmVjdGlvbnNcblx0XHRmb3IgKHZhciBpIGluIGxpc3QpIHtcblx0XHRcdHZhciBuZXVyb24gPSBsaXN0W2ldLm5ldXJvbjtcblx0XHRcdFx0XG5cdFx0XHQvKlxuXHRcdFx0RklYTUU6IGRvZXMgdGhpcyB3b3JrZWQgb25jZT9cblx0XHRcdFxuXHRcdFx0d2hpbGUgKG5ldXJvbi5uZXVyb24pXG5cdFx0XHRcdG5ldXJvbiA9IG5ldXJvbi5uZXVyb247XG5cdFx0XHQqL1xuXG5cdFx0XHRmb3IgKHZhciBqIGluIG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWQpIHtcblx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBuZXVyb24uY29ubmVjdGlvbnMucHJvamVjdGVkW2pdO1xuXHRcdFx0XHRjb25uZWN0aW9ucy5wdXNoKHtcblx0XHRcdFx0XHRmcm9tOiBpZHNbY29ubmVjdGlvbi5mcm9tLklEXSxcblx0XHRcdFx0XHR0bzogaWRzW2Nvbm5lY3Rpb24udG8uSURdLFxuXHRcdFx0XHRcdHdlaWdodDogY29ubmVjdGlvbi53ZWlnaHQsXG5cdFx0XHRcdFx0Z2F0ZXI6IGNvbm5lY3Rpb24uZ2F0ZXIgPyBpZHNbY29ubmVjdGlvbi5nYXRlci5JRF0gOiBudWxsLFxuXHRcdFx0XHR9KTtcblx0XHRcdH1cblx0XHRcdGlmIChuZXVyb24uc2VsZmNvbm5lY3RlZCgpKVxuXHRcdFx0XHRjb25uZWN0aW9ucy5wdXNoKHtcblx0XHRcdFx0XHRmcm9tOiBpZHNbbmV1cm9uLklEXSxcblx0XHRcdFx0XHR0bzogaWRzW25ldXJvbi5JRF0sXG5cdFx0XHRcdFx0d2VpZ2h0OiBuZXVyb24uc2VsZmNvbm5lY3Rpb24ud2VpZ2h0LFxuXHRcdFx0XHRcdGdhdGVyOiBuZXVyb24uc2VsZmNvbm5lY3Rpb24uZ2F0ZXIgPyBpZHNbbmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyXG5cdFx0XHRcdFx0XHQuSURdIDogbnVsbCxcblx0XHRcdFx0fSk7XG5cdFx0fVxuXG5cdFx0cmV0dXJuIHtcblx0XHRcdG5ldXJvbnM6IG5ldXJvbnMsXG5cdFx0XHRjb25uZWN0aW9uczogY29ubmVjdGlvbnNcblx0XHR9XG5cdH1cbiAgXG5cdC8vIGV4cG9ydCB0aGUgdG9wb2xvZ3kgaW50byBkb3QgbGFuZ3VhZ2Ugd2hpY2ggY2FuIGJlIHZpc3VhbGl6ZWQgYXMgZ3JhcGhzIHVzaW5nIGRvdFxuXHQvKiBleGFtcGxlOiAuLi4gY29uc29sZS5sb2cobmV0LnRvRG90TGFuZygpKTtcblx0XHRcdFx0JCBub2RlIGV4YW1wbGUuanMgPiBleGFtcGxlLmRvdFxuXHRcdFx0XHQkIGRvdCBleGFtcGxlLmRvdCAtVHBuZyA+IG91dC5wbmdcblx0Ki9cblx0dG9Eb3QoZWRnZWNvbm5lY3Rpb24pIHtcblx0XHRpZiAoISB0eXBlb2YgZWRnZWNvbm5lY3Rpb24pXG5cdFx0XHRlZGdlY29ubmVjdGlvbiA9IGZhbHNlO1xuXHRcdHZhciBjb2RlID0gXCJkaWdyYXBoIG5uIHtcXG4gICAgcmFua2RpciA9IEJUXFxuXCI7XG5cdFx0dmFyIGxheWVycyA9IFt0aGlzLmxheWVycy5pbnB1dF0uY29uY2F0KHRoaXMubGF5ZXJzLmhpZGRlbiwgdGhpcy5sYXllcnMub3V0cHV0KTtcblx0XHRmb3IgKHZhciBsYXllciBpbiBsYXllcnMpIHtcblx0XHRcdGZvciAodmFyIHRvIGluIGxheWVyc1tsYXllcl0uY29ubmVjdGVkdG8pIHsgLy8gcHJvamVjdGlvbnNcblx0XHRcdFx0dmFyIGNvbm5lY3Rpb24gPSBsYXllcnNbbGF5ZXJdLmNvbm5lY3RlZHRvW3RvXTtcblx0XHRcdFx0dmFyIGxheWVydG8gPSBjb25uZWN0aW9uLnRvO1xuXHRcdFx0XHR2YXIgc2l6ZSA9IGNvbm5lY3Rpb24uc2l6ZTtcblx0XHRcdFx0dmFyIGxheWVySUQgPSBsYXllcnMuaW5kZXhPZihsYXllcnNbbGF5ZXJdKTtcblx0XHRcdFx0dmFyIGxheWVydG9JRCA9IGxheWVycy5pbmRleE9mKGxheWVydG8pO1xuXHRcdFx0XHQvKiBodHRwOi8vc3RhY2tvdmVyZmxvdy5jb20vcXVlc3Rpb25zLzI2ODQ1NTQwL2Nvbm5lY3QtZWRnZXMtd2l0aC1ncmFwaC1kb3RcbiogRE9UIGRvZXMgbm90IHN1cHBvcnQgZWRnZS10by1lZGdlIGNvbm5lY3Rpb25zXG4qIFRoaXMgd29ya2Fyb3VuZCBwcm9kdWNlcyBzb21ld2hhdCB3ZWlyZCBncmFwaHMgLi4uXG5cdFx0XHRcdCovXG5cdFx0XHRcdGlmIChlZGdlY29ubmVjdGlvbikge1xuXHRcdFx0XHRcdGlmIChjb25uZWN0aW9uLmdhdGVkZnJvbS5sZW5ndGgpIHtcblx0XHRcdFx0XHRcdHZhciBmYWtlTm9kZSA9IFwiZmFrZVwiICsgbGF5ZXJJRCArIFwiX1wiICsgbGF5ZXJ0b0lEO1xuXHRcdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGZha2VOb2RlICtcblx0XHRcdFx0XHRcdFwiIFtsYWJlbCA9IFxcXCJcXFwiLCBzaGFwZSA9IHBvaW50LCB3aWR0aCA9IDAuMDEsIGhlaWdodCA9IDAuMDFdXFxuXCI7XG5cdFx0XHRcdFx0XHRjb2RlICs9IFwiICAgIFwiICsgbGF5ZXJJRCArIFwiIC0+IFwiICsgZmFrZU5vZGUgKyBcIiBbbGFiZWwgPSBcIiArIHNpemUgKyBcIiwgYXJyb3doZWFkID0gbm9uZV1cXG5cIjtcblx0XHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBmYWtlTm9kZSArIFwiIC0+IFwiICsgbGF5ZXJ0b0lEICsgXCJcXG5cIjtcblx0XHRcdFx0XHR9IGVsc2Vcblx0XHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBsYXllcklEICsgXCIgLT4gXCIgKyBsYXllcnRvSUQgKyBcIiBbbGFiZWwgPSBcIiArIHNpemUgKyBcIl1cXG5cIjtcblx0XHRcdFx0XHRmb3IgKHZhciBmcm9tIGluIGNvbm5lY3Rpb24uZ2F0ZWRmcm9tKSB7IC8vIGdhdGluZ3Ncblx0XHRcdFx0XHRcdHZhciBsYXllcmZyb20gPSBjb25uZWN0aW9uLmdhdGVkZnJvbVtmcm9tXS5sYXllcjtcblx0XHRcdFx0XHRcdHZhciB0eXBlID0gY29ubmVjdGlvbi5nYXRlZGZyb21bZnJvbV0udHlwZTtcblx0XHRcdFx0XHRcdHZhciBsYXllcmZyb21JRCA9IGxheWVycy5pbmRleE9mKGxheWVyZnJvbSk7XG5cdFx0XHRcdFx0XHRjb2RlICs9IFwiICAgIFwiICsgbGF5ZXJmcm9tSUQgKyBcIiAtPiBcIiArIGZha2VOb2RlICsgXCIgW2NvbG9yID0gYmx1ZV1cXG5cIjtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH0gZWxzZSB7XG5cdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGxheWVySUQgKyBcIiAtPiBcIiArIGxheWVydG9JRCArIFwiIFtsYWJlbCA9IFwiICsgc2l6ZSArIFwiXVxcblwiO1xuXHRcdFx0XHRcdGZvciAodmFyIGZyb20gaW4gY29ubmVjdGlvbi5nYXRlZGZyb20pIHsgLy8gZ2F0aW5nc1xuXHRcdFx0XHRcdFx0dmFyIGxheWVyZnJvbSA9IGNvbm5lY3Rpb24uZ2F0ZWRmcm9tW2Zyb21dLmxheWVyO1xuXHRcdFx0XHRcdFx0dmFyIHR5cGUgPSBjb25uZWN0aW9uLmdhdGVkZnJvbVtmcm9tXS50eXBlO1xuXHRcdFx0XHRcdFx0dmFyIGxheWVyZnJvbUlEID0gbGF5ZXJzLmluZGV4T2YobGF5ZXJmcm9tKTtcblx0XHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBsYXllcmZyb21JRCArIFwiIC0+IFwiICsgbGF5ZXJ0b0lEICsgXCIgW2NvbG9yID0gYmx1ZV1cXG5cIjtcblx0XHRcdFx0XHR9XG5cdFx0XHRcdH1cblx0XHRcdH1cblx0XHR9XG5cdFx0Y29kZSArPSBcIn1cXG5cIjtcblx0XHRyZXR1cm4ge1xuXHRcdFx0Y29kZTogY29kZSxcblx0XHRcdGxpbms6IFwiaHR0cHM6Ly9jaGFydC5nb29nbGVhcGlzLmNvbS9jaGFydD9jaGw9XCIgKyBlc2NhcGUoY29kZS5yZXBsYWNlKFwiLyAvZ1wiLCBcIitcIikpICsgXCImY2h0PWd2XCJcblx0XHR9XG5cdH1cblxuXHQvLyByZXR1cm5zIGEgZnVuY3Rpb24gdGhhdCB3b3JrcyBhcyB0aGUgYWN0aXZhdGlvbiBvZiB0aGUgbmV0d29yayBhbmQgY2FuIGJlIHVzZWQgd2l0aG91dCBkZXBlbmRpbmcgb24gdGhlIGxpYnJhcnlcblx0c3RhbmRhbG9uZSgpIHtcblx0XHRpZiAoIXRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZSgpO1xuXG5cdFx0dmFyIGRhdGEgPSB0aGlzLm9wdGltaXplZC5kYXRhO1xuXG5cdFx0Ly8gYnVpbGQgYWN0aXZhdGlvbiBmdW5jdGlvblxuXHRcdHZhciBhY3RpdmF0aW9uID0gXCJmdW5jdGlvbiAoaW5wdXQpIHtcXG5cIjtcblxuXHRcdC8vIGJ1aWxkIGlucHV0c1xuXHRcdGZvciAodmFyIGkgaW4gZGF0YS5pbnB1dHMpXG5cdFx0XHRhY3RpdmF0aW9uICs9IFwiRltcIiArIGRhdGEuaW5wdXRzW2ldICsgXCJdID0gaW5wdXRbXCIgKyBpICsgXCJdO1xcblwiO1xuXG5cdFx0Ly8gYnVpbGQgbmV0d29yayBhY3RpdmF0aW9uXG5cdFx0Zm9yICh2YXIgbmV1cm9uIGluIGRhdGEuYWN0aXZhdGUpIHsgLy8gc2hvdWxkbid0IHRoaXMgYmUgbGF5ZXI/XG5cdFx0XHRmb3IgKHZhciBzZW50ZW5jZSBpbiBkYXRhLmFjdGl2YXRlW25ldXJvbl0pXG5cdFx0XHRcdGFjdGl2YXRpb24gKz0gZGF0YS5hY3RpdmF0ZVtuZXVyb25dW3NlbnRlbmNlXSArIFwiXFxuXCI7XG5cdFx0fVxuXG5cdFx0Ly8gYnVpbGQgb3V0cHV0c1xuXHRcdGFjdGl2YXRpb24gKz0gXCJ2YXIgb3V0cHV0ID0gW107XFxuXCI7XG5cdFx0Zm9yICh2YXIgaSBpbiBkYXRhLm91dHB1dHMpXG5cdFx0XHRhY3RpdmF0aW9uICs9IFwib3V0cHV0W1wiICsgaSArIFwiXSA9IEZbXCIgKyBkYXRhLm91dHB1dHNbaV0gKyBcIl07XFxuXCI7XG5cdFx0YWN0aXZhdGlvbiArPSBcInJldHVybiBvdXRwdXQ7XFxufVwiO1xuXG5cdFx0Ly8gcmVmZXJlbmNlIGFsbCB0aGUgcG9zaXRpb25zIGluIG1lbW9yeVxuXHRcdHZhciBtZW1vcnkgPSBhY3RpdmF0aW9uLm1hdGNoKC9GXFxbKFxcZCspXFxdL2cpO1xuXHRcdHZhciBkaW1lbnNpb24gPSAwO1xuXHRcdHZhciBpZHMgPSB7fTtcblx0XHRmb3IgKHZhciBhZGRyZXNzIGluIG1lbW9yeSkge1xuXHRcdFx0dmFyIHRtcCA9IG1lbW9yeVthZGRyZXNzXS5tYXRjaCgvXFxkKy8pWzBdO1xuXHRcdFx0aWYgKCEodG1wIGluIGlkcykpIHtcblx0XHRcdFx0aWRzW3RtcF0gPSBkaW1lbnNpb24rKztcblx0XHRcdH1cblx0XHR9XG5cdFx0dmFyIGhhcmRjb2RlID0gXCJGID0ge1xcblwiO1xuXHRcdGZvciAodmFyIGkgaW4gaWRzKVxuXHRcdFx0aGFyZGNvZGUgKz0gaWRzW2ldICsgXCI6IFwiICsgdGhpcy5vcHRpbWl6ZWQubWVtb3J5W2ldICsgXCIsXFxuXCI7XG5cdFx0aGFyZGNvZGUgPSBoYXJkY29kZS5zdWJzdHJpbmcoMCwgaGFyZGNvZGUubGVuZ3RoIC0gMikgKyBcIlxcbn07XFxuXCI7XG5cdFx0aGFyZGNvZGUgPSBcInZhciBydW4gPSBcIiArIGFjdGl2YXRpb24ucmVwbGFjZSgvRlxcWyhcXGQrKV0vZywgZnVuY3Rpb24oXG5cdFx0XHRpbmRleCkge1xuXHRcdFx0cmV0dXJuICdGWycgKyBpZHNbaW5kZXgubWF0Y2goL1xcZCsvKVswXV0gKyAnXSdcblx0XHR9KS5yZXBsYWNlKFwie1xcblwiLCBcIntcXG5cIiArIGhhcmRjb2RlICsgXCJcIikgKyBcIjtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcInJldHVybiBydW5cIjtcblxuXHRcdC8vIHJldHVybiBzdGFuZGFsb25lIGZ1bmN0aW9uXG5cdFx0cmV0dXJuIG5ldyBGdW5jdGlvbihoYXJkY29kZSkoKTtcblx0fVxuXG5cdHdvcmtlcigpIHtcblx0XHRpZiAoIXRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZSgpO1xuXG5cdFx0dmFyIGhhcmRjb2RlID0gXCJ2YXIgaW5wdXRzID0gXCIgKyB0aGlzLm9wdGltaXplZC5kYXRhLmlucHV0cy5sZW5ndGggK1xuXHRcdFx0XCI7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgb3V0cHV0cyA9IFwiICsgdGhpcy5vcHRpbWl6ZWQuZGF0YS5vdXRwdXRzLmxlbmd0aCArXG5cdFx0XCI7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgRiA9IG51bGw7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgYWN0aXZhdGUgPSBcIiArIHRoaXMub3B0aW1pemVkLmFjdGl2YXRlLnRvU3RyaW5nKCkgK1xuXHRcdFwiO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwidmFyIHByb3BhZ2F0ZSA9IFwiICsgdGhpcy5vcHRpbWl6ZWQucHJvcGFnYXRlLnRvU3RyaW5nKCkgK1xuXHRcdFwiO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwib25tZXNzYWdlID0gZnVuY3Rpb24oZSl7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJGID0gZS5kYXRhLm1lbW9yeUJ1ZmZlcjtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcImlmIChlLmRhdGEuYWN0aW9uID09ICdhY3RpdmF0ZScpe1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwiaWYgKGUuZGF0YS5pbnB1dC5sZW5ndGggPT0gaW5wdXRzKXtcXG5cIjtcblx0XHRoYXJkY29kZSArPVxuXHRcdFwicG9zdE1lc3NhZ2UoIHsgYWN0aW9uOiAnYWN0aXZhdGUnLCBvdXRwdXQ6IGFjdGl2YXRlKGUuZGF0YS5pbnB1dCksIG1lbW9yeUJ1ZmZlcjogRiB9LCBbRi5idWZmZXJdKTtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcIn1cXG59XFxuZWxzZSBpZiAoZS5kYXRhLmFjdGlvbiA9PSAncHJvcGFnYXRlJyl7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJwcm9wYWdhdGUoZS5kYXRhLnJhdGUsIGUuZGF0YS50YXJnZXQpO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9XG5cdFx0XCJwb3N0TWVzc2FnZSh7IGFjdGlvbjogJ3Byb3BhZ2F0ZScsIG1lbW9yeUJ1ZmZlcjogRiB9LCBbRi5idWZmZXJdKTtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcIn1cXG59XFxuXCI7XG5cblx0XHR2YXIgYmxvYiA9IG5ldyBCbG9iKFtoYXJkY29kZV0pO1xuXHRcdHZhciBibG9iVVJMID0gKDxhbnk+d2luZG93KS5VUkwuY3JlYXRlT2JqZWN0VVJMKGJsb2IpO1xuXG5cdFx0cmV0dXJuIG5ldyBXb3JrZXIoYmxvYlVSTCk7XG5cdH1cblxuXHQvLyByZXR1cm5zIGEgY29weSBvZiB0aGUgbmV0d29ya1xuXHRjbG9uZShpZ25vcmVUcmFjZXMpIHtcblx0XHRyZXR1cm4gTmV0d29yay5mcm9tSlNPTih0aGlzLnRvSlNPTihpZ25vcmVUcmFjZXMpKTtcblx0fVxuXG5cdHN0YXRpYyBmcm9tSlNPTihqc29uKSB7XG5cblx0XHR2YXIgbmV1cm9ucyA9IFtdO1xuXG5cdFx0dmFyIGxheWVycyA9IHtcblx0XHRcdGlucHV0OiBuZXcgbGF5ZXIuTGF5ZXIoMCksXG5cdFx0XHRoaWRkZW46IFtdLFxuXHRcdFx0b3V0cHV0OiBuZXcgbGF5ZXIuTGF5ZXIoMClcblx0XHR9XG5cdFx0XG5cblx0XHRmb3IgKHZhciBpIGluIGpzb24ubmV1cm9ucykge1xuXHRcdFx0dmFyIGNvbmZpZyA9IGpzb24ubmV1cm9uc1tpXTtcblxuXHRcdFx0dmFyIG5ldXJvbiA9IG5ldyBfbmV1cm9uLk5ldXJvbigpO1xuXHRcdFx0bmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5ID0gY29uZmlnLnRyYWNlLmVsZWdpYmlsaXR5O1xuXHRcdFx0bmV1cm9uLnRyYWNlLmV4dGVuZGVkID0gY29uZmlnLnRyYWNlLmV4dGVuZGVkO1xuXHRcdFx0bmV1cm9uLnN0YXRlID0gY29uZmlnLnN0YXRlO1xuXHRcdFx0bmV1cm9uLm9sZCA9IGNvbmZpZy5vbGQ7XG5cdFx0XHRuZXVyb24uYWN0aXZhdGlvbiA9IGNvbmZpZy5hY3RpdmF0aW9uO1xuXHRcdFx0bmV1cm9uLmJpYXMgPSBjb25maWcuYmlhcztcblx0XHRcdG5ldXJvbi5zcXVhc2ggPSBjb25maWcuc3F1YXNoIGluIFNxdWFzaCA/IFNxdWFzaFtjb25maWcuc3F1YXNoXSA6XG5cdFx0XHRcdFNxdWFzaC5MT0dJU1RJQztcblx0XHRcdG5ldXJvbnMucHVzaChuZXVyb24pO1xuXG5cdFx0XHRpZiAoY29uZmlnLmxheWVyID09ICdpbnB1dCcpXG5cdFx0XHRcdGxheWVycy5pbnB1dC5hZGQobmV1cm9uKTtcblx0XHRcdGVsc2UgaWYgKGNvbmZpZy5sYXllciA9PSAnb3V0cHV0Jylcblx0XHRcdFx0bGF5ZXJzLm91dHB1dC5hZGQobmV1cm9uKTtcblx0XHRcdGVsc2Uge1xuXHRcdFx0XHRpZiAodHlwZW9mIGxheWVycy5oaWRkZW5bY29uZmlnLmxheWVyXSA9PSAndW5kZWZpbmVkJylcblx0XHRcdFx0XHRsYXllcnMuaGlkZGVuW2NvbmZpZy5sYXllcl0gPSBuZXcgbGF5ZXIuTGF5ZXIoMCk7XG5cdFx0XHRcdGxheWVycy5oaWRkZW5bY29uZmlnLmxheWVyXS5hZGQobmV1cm9uKTtcblx0XHRcdH1cblx0XHR9XG5cblx0XHRmb3IgKHZhciBpIGluIGpzb24uY29ubmVjdGlvbnMpIHtcblx0XHRcdHZhciBjb25maWcgPSBqc29uLmNvbm5lY3Rpb25zW2ldO1xuXHRcdFx0dmFyIGZyb20gPSBuZXVyb25zW2NvbmZpZy5mcm9tXTtcblx0XHRcdHZhciB0byA9IG5ldXJvbnNbY29uZmlnLnRvXTtcblx0XHRcdHZhciB3ZWlnaHQgPSBjb25maWcud2VpZ2h0XG5cdFx0XHR2YXIgZ2F0ZXIgPSBuZXVyb25zW2NvbmZpZy5nYXRlcl07XG5cblx0XHRcdHZhciBjb25uZWN0aW9uID0gZnJvbS5wcm9qZWN0KHRvLCB3ZWlnaHQpO1xuXHRcdFx0aWYgKGdhdGVyKVxuXHRcdFx0XHRnYXRlci5nYXRlKGNvbm5lY3Rpb24pO1xuXHRcdH1cblxuXHRcdHJldHVybiBuZXcgTmV0d29yayhsYXllcnMpO1xuXHR9XG59XG5cbmV4cG9ydCBtb2R1bGUgTmV0d29yayB7XG5cdGV4cG9ydCBpbnRlcmZhY2UgSU5ldHdvcmtOZXVyb24ge1xuXHRcdG5ldXJvbjogX25ldXJvbi5OZXVyb247XG5cdFx0bGF5ZXI6IHN0cmluZztcblx0fVxufSJdLCJzb3VyY2VSb290IjoiL3NvdXJjZS8ifQ==