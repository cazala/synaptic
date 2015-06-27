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

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9uZXR3b3JrLnRzIl0sIm5hbWVzIjpbIk5ldHdvcmsiLCJOZXR3b3JrLmNvbnN0cnVjdG9yIiwiTmV0d29yay5hY3RpdmF0ZSIsIk5ldHdvcmsucHJvcGFnYXRlIiwiTmV0d29yay5wcm9qZWN0IiwiTmV0d29yay5nYXRlIiwiTmV0d29yay5jbGVhciIsIk5ldHdvcmsucmVzZXQiLCJOZXR3b3JrLm9wdGltaXplIiwiTmV0d29yay5yZXN0b3JlIiwiTmV0d29yay5uZXVyb25zIiwiTmV0d29yay5pbnB1dHMiLCJOZXR3b3JrLm91dHB1dHMiLCJOZXR3b3JrLnNldCIsIk5ldHdvcmsuc2V0T3B0aW1pemUiLCJOZXR3b3JrLnRvSlNPTiIsIk5ldHdvcmsudG9Eb3QiLCJOZXR3b3JrLnN0YW5kYWxvbmUiLCJOZXR3b3JrLndvcmtlciIsIk5ldHdvcmsuY2xvbmUiLCJOZXR3b3JrLmZyb21KU09OIl0sIm1hcHBpbmdzIjoiQUFBQSxJQUFPLEtBQUssV0FBVyxTQUFTLENBQUMsQ0FBQztBQUNsQyxJQUFPLE1BQU0sV0FBVyxVQUFVLENBQUMsQ0FBQztBQUVwQyxJQUFPLE9BQU8sV0FBVyxVQUFVLENBQUMsQ0FBQztBQVNyQyxJQUFhLE9BQU87SUFPbkJBLFNBUFlBLE9BQU9BLENBT1BBLE1BQU9BO1FBTm5CQyxjQUFTQSxHQUFHQSxJQUFJQSxDQUFDQTtRQUNqQkEsV0FBTUEsR0FBR0E7WUFDUkEsS0FBS0EsRUFBRUEsSUFBSUE7WUFDWEEsTUFBTUEsRUFBRUEsRUFBRUE7WUFDVkEsTUFBTUEsRUFBRUEsSUFBSUE7U0FDWkEsQ0FBQ0E7UUFFREEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsT0FBT0EsTUFBTUEsSUFBSUEsV0FBV0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDbENBLElBQUlBLENBQUNBLE1BQU1BLEdBQUdBLE1BQU1BLElBQUlBO2dCQUN2QkEsS0FBS0EsRUFBRUEsSUFBSUE7Z0JBQ1hBLE1BQU1BLEVBQUVBLEVBQUVBO2dCQUNWQSxNQUFNQSxFQUFFQSxJ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LE1BQU1BLENBQUNBLEtBQUtBLENBQUNBLFFBQVFBLENBQUNBLEtBQUtBLENBQUNBLENBQUNBLEtBQUtBLENBQUNBLEdBQUdBLFFBQVFBLENBQUNBLE1BQU1BLEVBQUVBLE9BQU9BLEVBQzdEQSxVQUFVQSxFQUFFQSxLQUFLQSxFQUFFQSxLQUFLQSxDQUFDQSxDQUFDQTtRQUM5QkEsQ0FBQ0E7UUFHREEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsSUFBSUEsSUFBSUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDcEJBLElBQUlBLE1BQU1BLEdBQUdBLElBQUlBLENBQUNBLENBQUNBLENBQUNBLENBQUNBLE1BQU1BLENBQUNBO1lBUTVCQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxJQUFJQSxNQUFNQSxDQUFDQSxXQUFXQSxDQUFDQSxTQUFTQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDNUNBLElBQUlBLFVBQVVBLEdBQUdBLE1BQU1BLENBQUNBLFdBQVdBLENBQUNBLFNBQVNBLENBQUNBLENBQUNBLENBQUNBLENBQUNBO2dCQUNqREEsVUFBVUEsQ0FBQ0EsTUFBTUEsR0FBR0EsUUFBUUEsQ0FBQ0EsVUFBVUEsRUFBRUEsUUFBUUEsQ0FBQ0EsQ0FBQ0E7Z0JBQ25EQSxVQUFVQSxDQUFDQSxJQUFJQSxHQUFHQSxRQUFRQSxDQUFDQSxVQUFVQSxFQUFFQSxNQUFNQSxDQUFDQSxDQUFDQTtZQUNoREEsQ0FBQ0E7UUFDRkEsQ0FBQ0E7SUFDRkEsQ0FBQ0E7SUFFRFQseUNBQXlDQTtJQUN6Q0EseUJBQU9BLEdBQVBBO1FBQ0NVLElBQUlBLE9BQU9BLEdBQTZCQSxFQUFFQSxDQUFDQTtRQUUzQ0EsSUFBSUEsVUFBVUEsR0FBR0EsSUFBSUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsT0FBT0EsRUFBRUEsRUFDM0NBLFdBQVdBLEdBQUdBLElBQUlBLENBQUNBLE1BQU1BLENBQUNBLE1BQU1BLENBQUNBLE9BQU9BLEVBQUVBLENBQUNBO1FBRTVDQSxHQUFHQSxDQUFD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QUNBLENBQUNBO2dCQUM1Q0EsSUFBSUEsU0FBU0EsR0FBR0EsTUFBTUEsQ0FBQ0EsT0FBT0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ3hDQSxBQUlBQTs7O2tCQURFQTtnQkFDRkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsY0FBY0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7b0JBQ3BCQSxFQUFFQSxDQUFDQSxDQUFDQSxVQUFVQSxDQUFDQSxTQUFTQSxDQUFDQSxNQUFNQSxDQUFDQSxDQUFDQSxDQUFDQTt3QkFDakNBLElBQUlBLFFBQVFBLEdBQUdBLE1BQU1BLEdBQUdBLE9BQU9BLEdBQUdBLEdBQUdBLEdBQUdBLFNBQVNBLENBQUNBO3dCQUNsREEsSUFBSUEsSUFBSUEsTUFBTUEsR0FBR0EsUUFBUUEsR0FDekJBLCtEQUErREEsQ0FBQ0E7d0JBQ2hFQSxJQUFJQSxJQUFJQSxNQUFNQSxHQUFHQSxPQUFPQSxHQUFHQSxNQUFNQSxHQUFHQSxRQUFRQSxHQUFHQSxZQUFZQSxHQUFHQSxJQUFJQSxHQUFHQSx1QkFBdUJBLENBQUNBO3dCQUM3RkEsSUFBSUEsSUFBSUEsTUFBTUEsR0FBR0EsUUFBUUEsR0FBR0EsTUFBTUEsR0FBR0EsU0FBU0EsR0FBR0EsSUFBSUEsQ0FBQ0E7b0JBQ3ZEQSxDQUFDQTtvQkFBQ0EsSUFBSUE7d0JBQ0xBLElBQUlBLElBQUlBLE1BQU1BLEdBQUdBLE9BQU9BLEdBQUdBLE1BQU1BLEdBQUdBLFNBQVNBLEdBQUdBLFlBQVlBLEdBQUdBLElBQUlBLEdBQUdBLEtBQUtBLENBQUNBO29CQUM3RUEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsSUFBSUEsSUFBSUEsVUFBVUEsQ0FBQ0EsU0FBU0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7d0JBQ3ZDQSxJQUFJQSxTQUFTQSxHQUFHQSxVQUFVQSxDQUFDQSxTQUFTQSxDQUFDQSxJQUFJQSxDQUFDQSxDQUFDQSxLQUFLQSxDQUFDQTt3QkFDakRBLElBQUlBLElBQUlBLEdBQUdBLFVBQVVBLENBQUNBLFNBQVNBLENBQUNBLElBQUlBLENBQUNB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T0EsQ0FBQ0E7WUFDMUJBLFVBQVVBLElBQUlBLFNBQVNBLEdBQUdBLENBQUNBLEdBQUdBLFFBQVFBLEdBQUdBLElBQUlBLENBQUNBLE9BQU9BLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLE1BQU1BLENBQUNBO1FBQ25FQSxVQUFVQSxJQUFJQSxtQkFBbUJBLENBQUNBO1FBRWxDQSxBQUNBQSx3Q0FEd0NBO1lBQ3BDQSxNQUFNQSxHQUFHQSxVQUFVQSxDQUFDQSxLQUFLQSxDQUFDQSxhQUFhQSxDQUFDQSxDQUFDQTtRQUM3Q0EsSUFBSUEsU0FBU0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7UUFDbEJBLElBQUlBLEdBQUdBLEdBQUdBLEVBQUVBLENBQUNBO1FBQ2JBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLE9BQU9BLElBQUlBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBO1lBQzVCQSxJQUFJQSxHQUFHQSxHQUFHQSxNQUFNQSxDQUFDQSxPQUFPQSxDQUFDQSxDQUFDQSxLQUFLQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUMxQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsSUFBSUEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ25CQSxHQUFHQSxDQUFDQSxHQUFHQSxDQUFDQSxHQUFHQSxTQUFTQSxFQUFFQSxDQUFDQTtZQUN4QkEsQ0FBQ0E7UUFDRkEsQ0FBQ0E7UUFDREEsSUFBSUEsUUFBUUEsR0FBR0EsU0FBU0EsQ0FBQ0E7UUFDekJBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLElBQUlBLEdBQUdBLENBQUNBO1lBQ2pCQSxRQUFRQSxJQUFJQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxJQUFJQSxHQUFHQSxJQUFJQSxDQUFDQSxTQUFTQSxDQUFDQSxNQUFNQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxLQUFLQSxDQUFDQTtRQUM5REEsUUFBUUEsR0FBR0EsUUFBUUEsQ0FBQ0EsU0FB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QTtZQUNaQSxLQUFLQSxFQUFFQSxJQUFJQSxLQUFLQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUN6QkEsTUFBTUEsRUFBRUEsRUFBRUE7WUFDVkEsTUFBTUEsRUFBRUEsSUFBSUEsS0FBS0EsQ0FBQ0EsS0FBS0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7U0FDMUJBLENBQUFBO1FBR0RBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLElBQUlBLElBQUlBLENBQUNBLE9BQU9BLENBQUNBLENBQUNBLENBQUNBO1lBQzVCQSxJQUFJQSxNQUFNQSxHQUFHQSxJQUFJQSxDQUFDQSxPQUFPQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUU3QkEsSUFBSUEsTUFBTUEsR0FBR0EsSUFBSUEsT0FBT0EsQ0FBQ0EsTUFBTUEsRUFBRUEsQ0FBQ0E7WUFDbENBLE1BQU1BLENBQUNBLEtBQUtBLENBQUNBLFdBQVdBLEdBQUdBLE1BQU1BLENBQUNBLEtBQUtBLENBQUNBLFdBQVdBLENBQUNBO1lBQ3BEQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxRQUFRQSxHQUFHQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxRQUFRQSxDQUFDQTtZQUM5Q0EsTUFBTUEsQ0FBQ0EsS0FBS0EsR0FBR0EsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0E7WUFDNUJBLE1BQU1BLENBQUNBLEdBQUdBLEdBQUdBLE1BQU1BLENBQUNBLEdBQUdBLENBQUNBO1lBQ3hCQSxNQUFNQSxDQUFDQSxVQUFVQSxHQUFHQSxNQUFNQSxDQUFDQSxVQUFVQSxDQUFDQTtZQUN0Q0EsTUFBTUEsQ0FBQ0EsSUFBSUEsR0FBR0EsTUFBTUEsQ0FBQ0EsSUFBSUEsQ0FBQ0E7WUFDMUJBLE1BQU1BLENBQUNBLE1BQU1BLEdBQUdBLE1BQU1BLENBQUNBLE1BQU1BLElBQUlBLE1BQU1BLEdBQUdBLE1BQU1BLENBQUNBLE1BQU1BLENBQUNBLE1BQU1BLENBQUNBLEdBQzlEQSxNQUFNQSxDQUFDQSxRQUFRQSxDQUFDQTtZQUNqQkEsT0FBT0EsQ0FBQ0EsSUFBSUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0E7WUFFckJBLEVBQUVBLENBQUNBLENBQUNBLE1BQU1BLENBQUNBLEtBQUtBLElBQUlBLE9BQU9BLENBQUNBO2dCQUMzQkEsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0E7WUFDMUJBLElBQUlBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLE1BQU1BLENBQUNBLEtBQUtBLElBQUlBLFFBQVFBLENBQUNBO2dCQUNqQ0EsTUFBTUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsR0FBR0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0E7WUFDM0JBLElBQUlBLENBQUNBLENBQUNBO2dCQUNMQSxFQUFFQSxDQUFDQSxDQUFDQSxPQUFPQSxNQUFNQSxDQUFDQSxNQUFNQSxDQUFDQSxNQUFNQSxDQUFDQSxLQUFLQSxDQUFDQSxJQUFJQSxXQUFXQSxDQUFDQTtvQkFDckRBLE1BQU1BLENBQUNBLE1BQU1BLENBQUNBLE1BQU1BLENBQUNBLEtBQUtBLENBQUNBLEdBQUdBLElBQUlBLEtBQUtBLENBQUNBLEtBQUtBLENBQUNBLENBQUNBLENBQUNBLENBQUNBO2dCQUNsREEsTUFBTUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0E7WUFDekNBLENBQUNBO1FBQ0ZBLENBQUNBO1FBRURBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLElBQUlBLElBQUlBLENBQUNBLFdBQVdBLENBQUNBLENBQUNBLENBQUNBO1lBQ2hDQSxJQUFJQSxNQUFNQSxHQUFHQSxJQUFJQSxDQUFDQSxXQUFXQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUNqQ0EsSUFBSUEsSUFBSUEsR0FBR0EsT0FBT0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsQ0FBQ0E7WUFDaENBLElBQUlBLEVBQUVBLEdBQUdBLE9BQU9BLENBQUNBLE1BQU1BLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBQzVCQSxJQUFJQSxNQUFNQSxHQUFHQSxNQUFNQSxDQUFDQSxNQUFNQSxDQUFBQTtZQUMxQkEsSUFBSUEsS0FBS0EsR0FBR0EsT0FBT0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsQ0FBQ0E7WUFFbENBLElBQUlBLFVBQVVBLEdBQUdBLElBQUlBLENBQUNBLE9BQU9BLENBQUNBLEVBQUVBLEVBQUVBLE1BQU1BLENBQUNBLENBQUNBO1lBQzFDQSxFQUFFQSxDQUFDQSxDQUFDQSxLQUFLQSxDQUFDQTtnQkFDVEEsS0FBS0EsQ0FBQ0EsSUFBSUEsQ0FBQ0EsVUFBVUEsQ0FBQ0EsQ0FBQ0E7UUFDekJBLENBQUNBO1FBRURBLE1BQU1BLENBQUNBLElBQUlBLE9BQU9BLENBQUNBLE1BQU1BLENBQUNBLENBQUNBO0lBQzVCQSxDQUFDQTtJQUNGcEIsY0FBQ0E7QUFBREEsQ0FsbUJBLEFBa21CQ0EsSUFBQTtBQWxtQlksZUFBTyxHQUFQLE9Ba21CWixDQUFBO0FBT0EiLCJmaWxlIjoic3JjL25ldHdvcmsuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgbGF5ZXIgPSByZXF1aXJlKCcuL2xheWVyJyk7XG5pbXBvcnQgU3F1YXNoID0gcmVxdWlyZSgnLi9zcXVhc2gnKTtcbmltcG9ydCBTeW5hcHRpYyA9IHJlcXVpcmUoJy4vc3luYXB0aWMnKTtcbmltcG9ydCBfbmV1cm9uID0gcmVxdWlyZSgnLi9uZXVyb24nKTtcblxuLyoqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIE5FVFdPUktcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cbmRlY2xhcmUgZnVuY3Rpb24gZXNjYXBlKGE6IHN0cmluZyk6IHN0cmluZztcblxuXG5leHBvcnQgY2xhc3MgTmV0d29yayB7XG5cdG9wdGltaXplZCA9IG51bGw7XG5cdGxheWVycyA9IHtcblx0XHRpbnB1dDogbnVsbCxcblx0XHRoaWRkZW46IHt9LFxuXHRcdG91dHB1dDogbnVsbFxuXHR9O1xuXHRjb25zdHJ1Y3RvcihsYXllcnM/KSB7XG5cdFx0aWYgKHR5cGVvZiBsYXllcnMgIT0gJ3VuZGVmaW5lZCcpIHtcblx0XHRcdHRoaXMubGF5ZXJzID0gbGF5ZXJzIHx8IHtcblx0XHRcdFx0aW5wdXQ6IG51bGwsXG5cdFx0XHRcdGhpZGRlbjogW10sXG5cdFx0XHRcdG91dHB1dDogbnVsbFxuXHRcdFx0fTtcblx0XHR9XG5cdH1cblxuXHQvLyBmZWVkLWZvcndhcmQgYWN0aXZhdGlvbiBvZiBhbGwgdGhlIGxheWVycyB0byBwcm9kdWNlIGFuIG91cHV0XG5cdGFjdGl2YXRlKGlucHV0IDogU3luYXB0aWMuSU51bWVyaWNBcnJheSkge1xuXG5cdFx0aWYgKHRoaXMub3B0aW1pemVkID09PSBmYWxzZSkge1xuXHRcdFx0dGhpcy5sYXllcnMuaW5wdXQuYWN0aXZhdGUoaW5wdXQpO1xuXHRcdFx0Zm9yICh2YXIgbGF5ZXIgaW4gdGhpcy5sYXllcnMuaGlkZGVuKVxuXHRcdFx0XHR0aGlzLmxheWVycy5oaWRkZW5bbGF5ZXJdLmFjdGl2YXRlKCk7XG5cdFx0XHRyZXR1cm4gdGhpcy5sYXllcnMub3V0cHV0LmFjdGl2YXRlKCk7XG5cdFx0fVxuXHRcdGVsc2Uge1xuXHRcdFx0aWYgKHRoaXMub3B0aW1pemVkID09IG51bGwpXG5cdFx0XHRcdHRoaXMub3B0aW1pemUoKTtcblx0XHRcdHJldHVybiB0aGlzLm9wdGltaXplZC5hY3RpdmF0ZShpbnB1dCk7XG5cdFx0fVxuXHR9XG5cblx0Ly8gYmFjay1wcm9wYWdhdGUgdGhlIGVycm9yIHRocnUgdGhlIG5ldHdvcmtcblx0cHJvcGFnYXRlKHJhdGU6IG51bWJlciwgdGFyZ2V0PzogU3luYXB0aWMuSU51bWVyaWNBcnJheSkge1xuXG5cdFx0aWYgKHRoaXMub3B0aW1pemVkID09PSBmYWxzZSkge1xuXHRcdFx0dGhpcy5sYXllcnMub3V0cHV0LnByb3BhZ2F0ZShyYXRlLCB0YXJnZXQpO1xuXHRcdFx0dmFyIHJldmVyc2UgPSBbXTtcblx0XHRcdGZvciAodmFyIGxheWVyIGluIHRoaXMubGF5ZXJzLmhpZGRlbilcblx0XHRcdFx0cmV2ZXJzZS5wdXNoKHRoaXMubGF5ZXJzLmhpZGRlbltsYXllcl0pO1xuXHRcdFx0cmV2ZXJzZS5yZXZlcnNlKCk7XG5cdFx0XHRmb3IgKHZhciBsYXllciBpbiByZXZlcnNlKVxuXHRcdFx0XHRyZXZlcnNlW2xheWVyXS5wcm9wYWdhdGUocmF0ZSk7XG5cdFx0fVxuXHRcdGVsc2Uge1xuXHRcdFx0aWYgKHRoaXMub3B0aW1pemVkID09IG51bGwpXG5cdFx0XHRcdHRoaXMub3B0aW1pemUoKTtcblx0XHRcdHRoaXMub3B0aW1pemVkLnByb3BhZ2F0ZShyYXRlLCB0YXJnZXQpO1xuXHRcdH1cblx0fVxuXG5cdC8vIHByb2plY3QgYSBjb25uZWN0aW9uIHRvIGFub3RoZXIgdW5pdCAoZWl0aGVyIGEgbmV0d29yayBvciBhIGxheWVyKVxuXHRwcm9qZWN0KHVuaXQsIHR5cGUsIHdlaWdodHMpIHtcblxuXHRcdGlmICh0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemVkLnJlc2V0KCk7XG5cblx0XHRpZiAodW5pdCBpbnN0YW5jZW9mIE5ldHdvcmspXG5cdFx0XHRyZXR1cm4gdGhpcy5sYXllcnMub3V0cHV0LnByb2plY3QodW5pdC5sYXllcnMuaW5wdXQsIHR5cGUsIHdlaWdodHMpO1xuXG5cdFx0aWYgKHVuaXQgaW5zdGFuY2VvZiBsYXllci5MYXllcilcblx0XHRcdHJldHVybiB0aGlzLmxheWVycy5vdXRwdXQucHJvamVjdCh1bml0LCB0eXBlLCB3ZWlnaHRzKTtcblxuXHRcdHRocm93IFwiSW52YWxpZCBhcmd1bWVudCwgeW91IGNhbiBvbmx5IHByb2plY3QgY29ubmVjdGlvbnMgdG8gTEFZRVJTIGFuZCBORVRXT1JLUyFcIjtcblx0fVxuXG5cdC8vIGxldCB0aGlzIG5ldHdvcmsgZ2F0ZSBhIGNvbm5lY3Rpb25cblx0Z2F0ZShjb25uZWN0aW9uLCB0eXBlKSB7XG5cdFx0aWYgKHRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcblx0XHR0aGlzLmxheWVycy5vdXRwdXQuZ2F0ZShjb25uZWN0aW9uLCB0eXBlKTtcblx0fVxuXG5cdC8vIGNsZWFyIGFsbCBlbGVnaWJpbGl0eSB0cmFjZXMgYW5kIGV4dGVuZGVkIGVsZWdpYmlsaXR5IHRyYWNlcyAodGhlIG5ldHdvcmsgZm9yZ2V0cyBpdHMgY29udGV4dCwgYnV0IG5vdCB3aGF0IHdhcyB0cmFpbmVkKVxuXHRjbGVhcigpIHtcblxuXHRcdHRoaXMucmVzdG9yZSgpO1xuXG5cdFx0dmFyIGlucHV0TGF5ZXIgPSB0aGlzLmxheWVycy5pbnB1dCxcblx0XHRcdG91dHB1dExheWVyID0gdGhpcy5sYXllcnMub3V0cHV0O1xuXG5cdFx0aW5wdXRMYXllci5jbGVhcigpO1xuXHRcdGZvciAodmFyIGxheWVyIGluIHRoaXMubGF5ZXJzLmhpZGRlbikge1xuXHRcdFx0dmFyIGhpZGRlbkxheWVyID0gdGhpcy5sYXllcnMuaGlkZGVuW2xheWVyXTtcblx0XHRcdGhpZGRlbkxheWVyLmNsZWFyKCk7XG5cdFx0fVxuXHRcdG91dHB1dExheWVyLmNsZWFyKCk7XG5cblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplZC5yZXNldCgpO1xuXHR9XG5cblx0Ly8gcmVzZXQgYWxsIHdlaWdodHMgYW5kIGNsZWFyIGFsbCB0cmFjZXMgKGVuZHMgdXAgbGlrZSBhIG5ldyBuZXR3b3JrKVxuXHRyZXNldCgpIHtcblxuXHRcdHRoaXMucmVzdG9yZSgpO1xuXG5cdFx0dmFyIGlucHV0TGF5ZXIgPSB0aGlzLmxheWVycy5pbnB1dCxcblx0XHRcdG91dHB1dExheWVyID0gdGhpcy5sYXllcnMub3V0cHV0O1xuXG5cdFx0aW5wdXRMYXllci5yZXNldCgpO1xuXHRcdGZvciAodmFyIGxheWVyIGluIHRoaXMubGF5ZXJzLmhpZGRlbikge1xuXHRcdFx0dmFyIGhpZGRlbkxheWVyID0gdGhpcy5sYXllcnMuaGlkZGVuW2xheWVyXTtcblx0XHRcdGhpZGRlbkxheWVyLnJlc2V0KCk7XG5cdFx0fVxuXHRcdG91dHB1dExheWVyLnJlc2V0KCk7XG5cblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplZC5yZXNldCgpO1xuXHR9XG5cblx0Ly8gaGFyZGNvZGVzIHRoZSBiZWhhdmlvdXIgb2YgdGhlIHdob2xlIG5ldHdvcmsgaW50byBhIHNpbmdsZSBvcHRpbWl6ZWQgZnVuY3Rpb25cblx0b3B0aW1pemUoKSB7XG5cblx0XHR2YXIgdGhhdCA9IHRoaXM7XG5cdFx0dmFyIG9wdGltaXplZDogU3luYXB0aWMuSUNvbXBpbGVkUGFyYW1ldGVycyA9IHt9O1xuXHRcdHZhciBuZXVyb25zID0gdGhpcy5uZXVyb25zKCk7XG5cblx0XHRmb3IgKHZhciBpIGluIG5ldXJvbnMpIHtcblx0XHRcdHZhciBuZXVyb24gPSBuZXVyb25zW2ldLm5ldXJvbjtcblx0XHRcdHZhciBsYXllciA9IG5ldXJvbnNbaV0ubGF5ZXI7XG5cdFx0XHQvKlxuXHRcdFx0RklYTUU6IGRvZXMgdGhpcyB3b3JrZWQgb25jZT9cblx0XHRcdFxuXHRcdFx0d2hpbGUgKG5ldXJvbi5uZXVyb24pXG5cdFx0XHRcdG5ldXJvbiA9IG5ldXJvbi5uZXVyb247XG5cdFx0XHQqL1xuXG5cdFx0XHRvcHRpbWl6ZWQgPSBuZXVyb24ub3B0aW1pemUob3B0aW1pemVkLCBsYXllcik7XG5cdFx0fVxuXG5cdFx0Zm9yICh2YXIgaSBpbiBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzKVxuXHRcdFx0b3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlc1tpXS5yZXZlcnNlKCk7XG5cdFx0b3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlcy5yZXZlcnNlKCk7XG5cblx0XHR2YXIgaGFyZGNvZGUgPSBcIlwiO1xuXHRcdGhhcmRjb2RlICs9IFwidmFyIEYgPSBGbG9hdDY0QXJyYXkgPyBuZXcgRmxvYXQ2NEFycmF5KFwiICsgb3B0aW1pemVkLm1lbW9yeSArXG5cdFx0XCIpIDogW107IFwiO1xuXHRcdGZvciAodmFyIGkgaW4gb3B0aW1pemVkLnZhcmlhYmxlcylcblx0XHRcdGhhcmRjb2RlICs9IFwiRltcIiArIG9wdGltaXplZC52YXJpYWJsZXNbaV0uaWQgKyBcIl0gPSBcIiArIChvcHRpbWl6ZWQudmFyaWFibGVzW1xuXHRcdFx0XHRpXS52YWx1ZSB8fCAwKSArIFwiOyBcIjtcblx0XHRoYXJkY29kZSArPSBcInZhciBhY3RpdmF0ZSA9IGZ1bmN0aW9uKGlucHV0KXtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcImluZmx1ZW5jZXMgPSBbXTtcIjtcblx0XHRmb3IgKHZhciBpIGluIG9wdGltaXplZC5pbnB1dHMpXG5cdFx0XHRoYXJkY29kZSArPSBcIkZbXCIgKyBvcHRpbWl6ZWQuaW5wdXRzW2ldICsgXCJdID0gaW5wdXRbXCIgKyBpICsgXCJdOyBcIjtcblx0XHRmb3IgKHZhciBjdXJyZW50TGF5ZXIgaW4gb3B0aW1pemVkLmFjdGl2YXRpb25fc2VudGVuY2VzKSB7XG5cdFx0XHRpZiAob3B0aW1pemVkLmFjdGl2YXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl0ubGVuZ3RoID4gMCkge1xuXHRcdFx0XHRmb3IgKHZhciBjdXJyZW50TmV1cm9uIGluIG9wdGltaXplZC5hY3RpdmF0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdKSB7XG5cdFx0XHRcdFx0aGFyZGNvZGUgKz0gb3B0aW1pemVkLmFjdGl2YXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl1bY3VycmVudE5ldXJvbl0uam9pbihcIiBcIik7XG5cdFx0XHRcdFx0aGFyZGNvZGUgKz0gb3B0aW1pemVkLnRyYWNlX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdW2N1cnJlbnROZXVyb25dLmpvaW4oXCIgXCIpO1xuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0fVxuXHRcdGhhcmRjb2RlICs9IFwiIHZhciBvdXRwdXQgPSBbXTsgXCJcblx0XHRmb3IgKHZhciBpIGluIG9wdGltaXplZC5vdXRwdXRzKVxuXHRcdFx0aGFyZGNvZGUgKz0gXCJvdXRwdXRbXCIgKyBpICsgXCJdID0gRltcIiArIG9wdGltaXplZC5vdXRwdXRzW2ldICsgXCJdOyBcIjtcblx0XHRoYXJkY29kZSArPSBcInJldHVybiBvdXRwdXQ7IH07IFwiXG5cdFx0aGFyZGNvZGUgKz0gXCJ2YXIgcHJvcGFnYXRlID0gZnVuY3Rpb24ocmF0ZSwgdGFyZ2V0KXtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcIkZbXCIgKyBvcHRpbWl6ZWQudmFyaWFibGVzLnJhdGUuaWQgKyBcIl0gPSByYXRlOyBcIjtcblx0XHRmb3IgKHZhciBpIGluIG9wdGltaXplZC50YXJnZXRzKVxuXHRcdFx0aGFyZGNvZGUgKz0gXCJGW1wiICsgb3B0aW1pemVkLnRhcmdldHNbaV0gKyBcIl0gPSB0YXJnZXRbXCIgKyBpICsgXCJdOyBcIjtcblx0XHRmb3IgKHZhciBjdXJyZW50TGF5ZXIgaW4gb3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlcylcblx0XHRcdGZvciAodmFyIGN1cnJlbnROZXVyb24gaW4gb3B0aW1pemVkLnByb3BhZ2F0aW9uX3NlbnRlbmNlc1tjdXJyZW50TGF5ZXJdKVxuXHRcdFx0XHRoYXJkY29kZSArPSBvcHRpbWl6ZWQucHJvcGFnYXRpb25fc2VudGVuY2VzW2N1cnJlbnRMYXllcl1bY3VycmVudE5ldXJvbl0uam9pbihcIiBcIikgKyBcIiBcIjtcblx0XHRoYXJkY29kZSArPSBcIiB9O1xcblwiO1xuXHRcdGhhcmRjb2RlICs9XG5cdFx0XCJ2YXIgb3duZXJzaGlwID0gZnVuY3Rpb24obWVtb3J5QnVmZmVyKXtcXG5GID0gbWVtb3J5QnVmZmVyO1xcbnRoaXMubWVtb3J5ID0gRjtcXG59O1xcblwiO1xuXHRcdGhhcmRjb2RlICs9XG5cdFx0XCJyZXR1cm4ge1xcbm1lbW9yeTogRixcXG5hY3RpdmF0ZTogYWN0aXZhdGUsXFxucHJvcGFnYXRlOiBwcm9wYWdhdGUsXFxub3duZXJzaGlwOiBvd25lcnNoaXBcXG59O1wiO1xuXHRcdGhhcmRjb2RlID0gaGFyZGNvZGUuc3BsaXQoXCI7XCIpLmpvaW4oXCI7XFxuXCIpO1xuXG5cdFx0dmFyIGNvbnN0cnVjdG9yID0gbmV3IEZ1bmN0aW9uKGhhcmRjb2RlKTtcblxuXHRcdHZhciBuZXR3b3JrID0gY29uc3RydWN0b3IoKTtcblxuXHRcdG5ldHdvcmsuZGF0YSA9IHtcblx0XHRcdHZhcmlhYmxlczogb3B0aW1pemVkLnZhcmlhYmxlcyxcblx0XHRcdGFjdGl2YXRlOiBvcHRpbWl6ZWQuYWN0aXZhdGlvbl9zZW50ZW5jZXMsXG5cdFx0XHRwcm9wYWdhdGU6IG9wdGltaXplZC5wcm9wYWdhdGlvbl9zZW50ZW5jZXMsXG5cdFx0XHR0cmFjZTogb3B0aW1pemVkLnRyYWNlX3NlbnRlbmNlcyxcblx0XHRcdGlucHV0czogb3B0aW1pemVkLmlucHV0cyxcblx0XHRcdG91dHB1dHM6IG9wdGltaXplZC5vdXRwdXRzLFxuXHRcdFx0Y2hlY2tfYWN0aXZhdGlvbjogdGhpcy5hY3RpdmF0ZSxcblx0XHRcdGNoZWNrX3Byb3BhZ2F0aW9uOiB0aGlzLnByb3BhZ2F0ZVxuXHRcdH1cblxuXHRcdG5ldHdvcmsucmVzZXQgPSBmdW5jdGlvbigpIHtcblx0XHRcdGlmICh0aGF0Lm9wdGltaXplZCkge1xuXHRcdFx0XHR0aGF0Lm9wdGltaXplZCA9IG51bGw7XG5cdFx0XHRcdHRoYXQuYWN0aXZhdGUgPSBuZXR3b3JrLmRhdGEuY2hlY2tfYWN0aXZhdGlvbjtcblx0XHRcdFx0dGhhdC5wcm9wYWdhdGUgPSBuZXR3b3JrLmRhdGEuY2hlY2tfcHJvcGFnYXRpb247XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFx0dGhpcy5vcHRpbWl6ZWQgPSBuZXR3b3JrO1xuXHRcdHRoaXMuYWN0aXZhdGUgPSBuZXR3b3JrLmFjdGl2YXRlO1xuXHRcdHRoaXMucHJvcGFnYXRlID0gbmV0d29yay5wcm9wYWdhdGU7XG5cdH1cblxuXHQvLyByZXN0b3JlcyBhbGwgdGhlIHZhbHVlcyBmcm9tIHRoZSBvcHRpbWl6ZWQgbmV0d29yayB0aGUgdGhlaXIgcmVzcGVjdGl2ZSBvYmplY3RzIGluIG9yZGVyIHRvIG1hbmlwdWxhdGUgdGhlIG5ldHdvcmtcblx0cmVzdG9yZSgpIHtcblx0XHRpZiAoIXRoaXMub3B0aW1pemVkKVxuXHRcdFx0cmV0dXJuO1xuXG5cdFx0dmFyIG9wdGltaXplZCA9IHRoaXMub3B0aW1pemVkO1xuXG5cdFx0dmFyIGdldFZhbHVlID0gZnVuY3Rpb24oLi4uYXJnczogYW55W10pIHtcblx0XHRcdHZhciB1bml0ID0gYXJncy5zaGlmdCgpO1xuXHRcdFx0dmFyIHByb3AgPSBhcmdzLnBvcCgpO1xuXG5cdFx0XHR2YXIgaWQgPSBwcm9wICsgJ18nO1xuXHRcdFx0Zm9yICh2YXIgcHJvcGVydHkgaW4gYXJncylcblx0XHRcdFx0aWQgKz0gYXJnc1twcm9wZXJ0eV0gKyAnXyc7XG5cdFx0XHRpZCArPSB1bml0LklEO1xuXG5cdFx0XHR2YXIgbWVtb3J5ID0gb3B0aW1pemVkLm1lbW9yeTtcblx0XHRcdHZhciB2YXJpYWJsZXMgPSBvcHRpbWl6ZWQuZGF0YS52YXJpYWJsZXM7XG5cblx0XHRcdGlmIChpZCBpbiB2YXJpYWJsZXMpXG5cdFx0XHRcdHJldHVybiBtZW1vcnlbdmFyaWFibGVzW2lkXS5pZF07XG5cdFx0XHRyZXR1cm4gMDtcblx0XHR9XG5cblx0XHR2YXIgbGlzdCA9IHRoaXMubmV1cm9ucygpO1xuXG5cdFx0Ly8gbGluayBpZCdzIHRvIHBvc2l0aW9ucyBpbiB0aGUgYXJyYXlcblx0XHR2YXIgaWRzID0ge307XG5cdFx0Zm9yICh2YXIgaSBpbiBsaXN0KSB7XG5cdFx0XHR2YXIgbmV1cm9uID0gbGlzdFtpXS5uZXVyb247XG5cdFx0XHQvKlxuXHRcdFx0RklYTUU6IGRvZXMgdGhpcyB3b3JrZWQgb25jZT9cblx0XHRcdFxuXHRcdFx0d2hpbGUgKG5ldXJvbi5uZXVyb24pXG5cdFx0XHRcdG5ldXJvbiA9IG5ldXJvbi5uZXVyb247XG5cdFx0XHQqL1xuXG5cdFx0XHRuZXVyb24uc3RhdGUgPSBnZXRWYWx1ZShuZXVyb24sICdzdGF0ZScpO1xuXHRcdFx0bmV1cm9uLm9sZCA9IGdldFZhbHVlKG5ldXJvbiwgJ29sZCcpO1xuXHRcdFx0bmV1cm9uLmFjdGl2YXRpb24gPSBnZXRWYWx1ZShuZXVyb24sICdhY3RpdmF0aW9uJyk7XG5cdFx0XHRuZXVyb24uYmlhcyA9IGdldFZhbHVlKG5ldXJvbiwgJ2JpYXMnKTtcblxuXHRcdFx0Zm9yICh2YXIgaW5wdXQgaW4gbmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5KVxuXHRcdFx0XHRuZXVyb24udHJhY2UuZWxlZ2liaWxpdHlbaW5wdXRdID0gZ2V0VmFsdWUobmV1cm9uLCAndHJhY2UnLFxuXHRcdFx0XHRcdCdlbGVnaWJpbGl0eScsIGlucHV0KTtcblxuXHRcdFx0Zm9yICh2YXIgZ2F0ZWQgaW4gbmV1cm9uLnRyYWNlLmV4dGVuZGVkKVxuXHRcdFx0XHRmb3IgKHZhciBpbnB1dCBpbiBuZXVyb24udHJhY2UuZXh0ZW5kZWRbZ2F0ZWRdKVxuXHRcdFx0XHRcdG5ldXJvbi50cmFjZS5leHRlbmRlZFtnYXRlZF1baW5wdXRdID0gZ2V0VmFsdWUobmV1cm9uLCAndHJhY2UnLFxuXHRcdFx0XHRcdFx0J2V4dGVuZGVkJywgZ2F0ZWQsIGlucHV0KTtcblx0XHR9XG5cblx0XHQvLyBnZXQgY29ubmVjdGlvbnNcblx0XHRmb3IgKHZhciBpIGluIGxpc3QpIHtcblx0XHRcdHZhciBuZXVyb24gPSBsaXN0W2ldLm5ldXJvbjtcblx0XHRcdC8qXG5cdFx0XHRGSVhNRTogZG9lcyB0aGlzIHdvcmtlZCBvbmNlP1xuXHRcdFx0XG5cdFx0XHR3aGlsZSAobmV1cm9uLm5ldXJvbilcblx0XHRcdFx0bmV1cm9uID0gbmV1cm9uLm5ldXJvbjtcblx0XHRcdCovXG5cblx0XHRcdGZvciAodmFyIGogaW4gbmV1cm9uLmNvbm5lY3Rpb25zLnByb2plY3RlZCkge1xuXHRcdFx0XHR2YXIgY29ubmVjdGlvbiA9IG5ldXJvbi5jb25uZWN0aW9ucy5wcm9qZWN0ZWRbal07XG5cdFx0XHRcdGNvbm5lY3Rpb24ud2VpZ2h0ID0gZ2V0VmFsdWUoY29ubmVjdGlvbiwgJ3dlaWdodCcpO1xuXHRcdFx0XHRjb25uZWN0aW9uLmdhaW4gPSBnZXRWYWx1ZShjb25uZWN0aW9uLCAnZ2FpbicpO1xuXHRcdFx0fVxuXHRcdH1cblx0fVxuXG5cdC8vIHJldHVybnMgYWxsIHRoZSBuZXVyb25zIGluIHRoZSBuZXR3b3JrXG5cdG5ldXJvbnMoKTogTmV0d29yay5JTmV0d29ya05ldXJvbltdIHtcblx0XHR2YXIgbmV1cm9uczogTmV0d29yay5JTmV0d29ya05ldXJvbltdID0gW107XG5cblx0XHR2YXIgaW5wdXRMYXllciA9IHRoaXMubGF5ZXJzLmlucHV0Lm5ldXJvbnMoKSxcblx0XHRcdG91dHB1dExheWVyID0gdGhpcy5sYXllcnMub3V0cHV0Lm5ldXJvbnMoKTtcblxuXHRcdGZvciAodmFyIG5ldXJvbiBpbiBpbnB1dExheWVyKVxuXHRcdFx0bmV1cm9ucy5wdXNoKHtcblx0XHRcdFx0bmV1cm9uOiBpbnB1dExheWVyW25ldXJvbl0sXG5cdFx0XHRcdGxheWVyOiAnaW5wdXQnXG5cdFx0XHR9KTtcblxuXHRcdGZvciAodmFyIGxheWVyIGluIHRoaXMubGF5ZXJzLmhpZGRlbikge1xuXHRcdFx0dmFyIGhpZGRlbkxheWVyID0gdGhpcy5sYXllcnMuaGlkZGVuW2xheWVyXS5uZXVyb25zKCk7XG5cdFx0XHRmb3IgKHZhciBuZXVyb24gaW4gaGlkZGVuTGF5ZXIpXG5cdFx0XHRcdG5ldXJvbnMucHVzaCh7XG5cdFx0XHRcdFx0bmV1cm9uOiBoaWRkZW5MYXllcltuZXVyb25dLFxuXHRcdFx0XHRcdGxheWVyOiBsYXllclxuXHRcdFx0XHR9KTtcblx0XHR9XG5cdFx0Zm9yICh2YXIgbmV1cm9uIGluIG91dHB1dExheWVyKVxuXHRcdFx0bmV1cm9ucy5wdXNoKHtcblx0XHRcdFx0bmV1cm9uOiBvdXRwdXRMYXllcltuZXVyb25dLFxuXHRcdFx0XHRsYXllcjogJ291dHB1dCdcblx0XHRcdH0pO1xuXG5cdFx0cmV0dXJuIG5ldXJvbnM7XG5cdH1cblxuXHQvLyByZXR1cm5zIG51bWJlciBvZiBpbnB1dHMgb2YgdGhlIG5ldHdvcmtcblx0aW5wdXRzKCk6IG51bWJlciB7XG5cdFx0cmV0dXJuIHRoaXMubGF5ZXJzLmlucHV0LnNpemU7XG5cdH1cblxuXHQvLyByZXR1cm5zIG51bWJlciBvZiBvdXRwdXRzIG9mIGh0ZSBuZXR3b3JrXG5cdG91dHB1dHMoKTogbnVtYmVyIHtcblx0XHRyZXR1cm4gdGhpcy5sYXllcnMub3V0cHV0LnNpemU7XG5cdH1cblxuXHQvLyBzZXRzIHRoZSBsYXllcnMgb2YgdGhlIG5ldHdvcmtcblx0c2V0KGxheWVycykge1xuXG5cdFx0dGhpcy5sYXllcnMgPSBsYXllcnM7XG5cdFx0aWYgKHRoaXMub3B0aW1pemVkKVxuXHRcdFx0dGhpcy5vcHRpbWl6ZWQucmVzZXQoKTtcblx0fVxuXG5cdHNldE9wdGltaXplKGJvb2wpIHtcblx0XHR0aGlzLnJlc3RvcmUoKTtcblx0XHRpZiAodGhpcy5vcHRpbWl6ZWQpXG5cdFx0XHR0aGlzLm9wdGltaXplZC5yZXNldCgpO1xuXHRcdHRoaXMub3B0aW1pemVkID0gYm9vbCA/IG51bGwgOiBmYWxzZTtcblx0fVxuXG5cdC8vIHJldHVybnMgYSBqc29uIHRoYXQgcmVwcmVzZW50cyBhbGwgdGhlIG5ldXJvbnMgYW5kIGNvbm5lY3Rpb25zIG9mIHRoZSBuZXR3b3JrXG5cdHRvSlNPTihpZ25vcmVUcmFjZXMpIHtcblxuXHRcdHRoaXMucmVzdG9yZSgpO1xuXG5cdFx0dmFyIGxpc3QgPSB0aGlzLm5ldXJvbnMoKTtcblx0XHR2YXIgbmV1cm9ucyA9IFtdO1xuXHRcdHZhciBjb25uZWN0aW9ucyA9IFtdO1xuXG5cdFx0Ly8gbGluayBpZCdzIHRvIHBvc2l0aW9ucyBpbiB0aGUgYXJyYXlcblx0XHR2YXIgaWRzID0ge307XG5cdFx0Zm9yICh2YXIgaSBpbiBsaXN0KSB7XG5cdFx0XHR2YXIgbmV1cm9uID0gbGlzdFtpXS5uZXVyb247XG5cdFx0XHQvKlxuXHRcdFx0RklYTUU6IGRvZXMgdGhpcyB3b3JrZWQgb25jZT9cblx0XHRcdFxuXHRcdFx0d2hpbGUgKG5ldXJvbi5uZXVyb24pXG5cdFx0XHRcdG5ldXJvbiA9IG5ldXJvbi5uZXVyb247XG5cdFx0XHQqL1xuXG5cdFx0XHRpZHNbbmV1cm9uLklEXSA9IGk7XG5cblx0XHRcdHZhciBjb3B5ID0ge1xuXHRcdFx0XHR0cmFjZToge1xuXHRcdFx0XHRcdGVsZWdpYmlsaXR5OiB7fSxcblx0XHRcdFx0XHRleHRlbmRlZDoge31cblx0XHRcdFx0fSxcblx0XHRcdFx0c3RhdGU6IG5ldXJvbi5zdGF0ZSxcblx0XHRcdFx0b2xkOiBuZXVyb24ub2xkLFxuXHRcdFx0XHRhY3RpdmF0aW9uOiBuZXVyb24uYWN0aXZhdGlvbixcblx0XHRcdFx0YmlhczogbmV1cm9uLmJpYXMsXG5cdFx0XHRcdGxheWVyOiBsaXN0W2ldLmxheWVyLFxuXHRcdFx0XHRzcXVhc2g6IG51bGxcblx0XHRcdH07XG5cblx0XHRcdGNvcHkuc3F1YXNoID0gbmV1cm9uLnNxdWFzaCA9PSBTcXVhc2guTE9HSVNUSUMgPyBcIkxPR0lTVElDXCIgOlxuXHRcdFx0XHRuZXVyb24uc3F1YXNoID09IFNxdWFzaC5UQU5IID8gXCJUQU5IXCIgOlxuXHRcdFx0XHRcdG5ldXJvbi5zcXVhc2ggPT0gU3F1YXNoLklERU5USVRZID8gXCJJREVOVElUWVwiIDpcblx0XHRcdFx0XHRcdG5ldXJvbi5zcXVhc2ggPT0gU3F1YXNoLkhMSU0gPyBcIkhMSU1cIiA6XG5cdFx0XHRcdFx0XHRcdG51bGw7XG5cblx0XHRcdG5ldXJvbnMucHVzaChjb3B5KTtcblx0XHR9XG5cblx0XHRpZiAoIWlnbm9yZVRyYWNlcylcblx0XHRcdGZvciAodmFyIGkgaW4gbmV1cm9ucykge1xuXHRcdFx0XHR2YXIgY29waWVkTmV1cm9uID0gbmV1cm9uc1tpXTtcblxuXHRcdFx0XHRmb3IgKHZhciBpbnB1dCBpbiBuZXVyb24udHJhY2UuZWxlZ2liaWxpdHkpXG5cdFx0XHRcdFx0Y29waWVkTmV1cm9uLnRyYWNlLmVsZWdpYmlsaXR5W2lucHV0XSA9IG5ldXJvbi50cmFjZS5lbGVnaWJpbGl0eVtpbnB1dF07XG5cblx0XHRcdFx0Zm9yICh2YXIgZ2F0ZWQgaW4gbmV1cm9uLnRyYWNlLmV4dGVuZGVkKSB7XG5cdFx0XHRcdFx0Y29waWVkTmV1cm9uLnRyYWNlLmV4dGVuZGVkW2dhdGVkXSA9IHt9O1xuXHRcdFx0XHRcdGZvciAodmFyIGlucHV0IGluIG5ldXJvbi50cmFjZS5leHRlbmRlZFtnYXRlZF0pXG5cdFx0XHRcdFx0XHRjb3BpZWROZXVyb24udHJhY2UuZXh0ZW5kZWRbaWRzW2dhdGVkXV1baW5wdXRdID0gbmV1cm9uLnRyYWNlLmV4dGVuZGVkW1xuXHRcdFx0XHRcdFx0Z2F0ZWRdW2lucHV0XTtcblx0XHRcdFx0fVxuXHRcdFx0fVxuXG5cdFx0Ly8gZ2V0IGNvbm5lY3Rpb25zXG5cdFx0Zm9yICh2YXIgaSBpbiBsaXN0KSB7XG5cdFx0XHR2YXIgbmV1cm9uID0gbGlzdFtpXS5uZXVyb247XG5cdFx0XHRcdFxuXHRcdFx0Lypcblx0XHRcdEZJWE1FOiBkb2VzIHRoaXMgd29ya2VkIG9uY2U/XG5cdFx0XHRcblx0XHRcdHdoaWxlIChuZXVyb24ubmV1cm9uKVxuXHRcdFx0XHRuZXVyb24gPSBuZXVyb24ubmV1cm9uO1xuXHRcdFx0Ki9cblxuXHRcdFx0Zm9yICh2YXIgaiBpbiBuZXVyb24uY29ubmVjdGlvbnMucHJvamVjdGVkKSB7XG5cdFx0XHRcdHZhciBjb25uZWN0aW9uID0gbmV1cm9uLmNvbm5lY3Rpb25zLnByb2plY3RlZFtqXTtcblx0XHRcdFx0Y29ubmVjdGlvbnMucHVzaCh7XG5cdFx0XHRcdFx0ZnJvbTogaWRzW2Nvbm5lY3Rpb24uZnJvbS5JRF0sXG5cdFx0XHRcdFx0dG86IGlkc1tjb25uZWN0aW9uLnRvLklEXSxcblx0XHRcdFx0XHR3ZWlnaHQ6IGNvbm5lY3Rpb24ud2VpZ2h0LFxuXHRcdFx0XHRcdGdhdGVyOiBjb25uZWN0aW9uLmdhdGVyID8gaWRzW2Nvbm5lY3Rpb24uZ2F0ZXIuSURdIDogbnVsbCxcblx0XHRcdFx0fSk7XG5cdFx0XHR9XG5cdFx0XHRpZiAobmV1cm9uLnNlbGZjb25uZWN0ZWQoKSlcblx0XHRcdFx0Y29ubmVjdGlvbnMucHVzaCh7XG5cdFx0XHRcdFx0ZnJvbTogaWRzW25ldXJvbi5JRF0sXG5cdFx0XHRcdFx0dG86IGlkc1tuZXVyb24uSURdLFxuXHRcdFx0XHRcdHdlaWdodDogbmV1cm9uLnNlbGZjb25uZWN0aW9uLndlaWdodCxcblx0XHRcdFx0XHRnYXRlcjogbmV1cm9uLnNlbGZjb25uZWN0aW9uLmdhdGVyID8gaWRzW25ldXJvbi5zZWxmY29ubmVjdGlvbi5nYXRlclxuXHRcdFx0XHRcdFx0LklEXSA6IG51bGwsXG5cdFx0XHRcdH0pO1xuXHRcdH1cblxuXHRcdHJldHVybiB7XG5cdFx0XHRuZXVyb25zOiBuZXVyb25zLFxuXHRcdFx0Y29ubmVjdGlvbnM6IGNvbm5lY3Rpb25zXG5cdFx0fVxuXHR9XG4gIFxuXHQvLyBleHBvcnQgdGhlIHRvcG9sb2d5IGludG8gZG90IGxhbmd1YWdlIHdoaWNoIGNhbiBiZSB2aXN1YWxpemVkIGFzIGdyYXBocyB1c2luZyBkb3Rcblx0LyogZXhhbXBsZTogLi4uIGNvbnNvbGUubG9nKG5ldC50b0RvdExhbmcoKSk7XG5cdFx0XHRcdCQgbm9kZSBleGFtcGxlLmpzID4gZXhhbXBsZS5kb3Rcblx0XHRcdFx0JCBkb3QgZXhhbXBsZS5kb3QgLVRwbmcgPiBvdXQucG5nXG5cdCovXG5cdHRvRG90KGVkZ2Vjb25uZWN0aW9uKSB7XG5cdFx0aWYgKCEgdHlwZW9mIGVkZ2Vjb25uZWN0aW9uKVxuXHRcdFx0ZWRnZWNvbm5lY3Rpb24gPSBmYWxzZTtcblx0XHR2YXIgY29kZSA9IFwiZGlncmFwaCBubiB7XFxuICAgIHJhbmtkaXIgPSBCVFxcblwiO1xuXHRcdHZhciBsYXllcnMgPSBbdGhpcy5sYXllcnMuaW5wdXRdLmNvbmNhdCh0aGlzLmxheWVycy5oaWRkZW4sIHRoaXMubGF5ZXJzLm91dHB1dCk7XG5cdFx0Zm9yICh2YXIgbGF5ZXIgaW4gbGF5ZXJzKSB7XG5cdFx0XHRmb3IgKHZhciB0byBpbiBsYXllcnNbbGF5ZXJdLmNvbm5lY3RlZHRvKSB7IC8vIHByb2plY3Rpb25zXG5cdFx0XHRcdHZhciBjb25uZWN0aW9uID0gbGF5ZXJzW2xheWVyXS5jb25uZWN0ZWR0b1t0b107XG5cdFx0XHRcdHZhciBsYXllcnRvID0gY29ubmVjdGlvbi50bztcblx0XHRcdFx0dmFyIHNpemUgPSBjb25uZWN0aW9uLnNpemU7XG5cdFx0XHRcdHZhciBsYXllcklEID0gbGF5ZXJzLmluZGV4T2YobGF5ZXJzW2xheWVyXSk7XG5cdFx0XHRcdHZhciBsYXllcnRvSUQgPSBsYXllcnMuaW5kZXhPZihsYXllcnRvKTtcblx0XHRcdFx0LyogaHR0cDovL3N0YWNrb3ZlcmZsb3cuY29tL3F1ZXN0aW9ucy8yNjg0NTU0MC9jb25uZWN0LWVkZ2VzLXdpdGgtZ3JhcGgtZG90XG4qIERPVCBkb2VzIG5vdCBzdXBwb3J0IGVkZ2UtdG8tZWRnZSBjb25uZWN0aW9uc1xuKiBUaGlzIHdvcmthcm91bmQgcHJvZHVjZXMgc29tZXdoYXQgd2VpcmQgZ3JhcGhzIC4uLlxuXHRcdFx0XHQqL1xuXHRcdFx0XHRpZiAoZWRnZWNvbm5lY3Rpb24pIHtcblx0XHRcdFx0XHRpZiAoY29ubmVjdGlvbi5nYXRlZGZyb20ubGVuZ3RoKSB7XG5cdFx0XHRcdFx0XHR2YXIgZmFrZU5vZGUgPSBcImZha2VcIiArIGxheWVySUQgKyBcIl9cIiArIGxheWVydG9JRDtcblx0XHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBmYWtlTm9kZSArXG5cdFx0XHRcdFx0XHRcIiBbbGFiZWwgPSBcXFwiXFxcIiwgc2hhcGUgPSBwb2ludCwgd2lkdGggPSAwLjAxLCBoZWlnaHQgPSAwLjAxXVxcblwiO1xuXHRcdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGxheWVySUQgKyBcIiAtPiBcIiArIGZha2VOb2RlICsgXCIgW2xhYmVsID0gXCIgKyBzaXplICsgXCIsIGFycm93aGVhZCA9IG5vbmVdXFxuXCI7XG5cdFx0XHRcdFx0XHRjb2RlICs9IFwiICAgIFwiICsgZmFrZU5vZGUgKyBcIiAtPiBcIiArIGxheWVydG9JRCArIFwiXFxuXCI7XG5cdFx0XHRcdFx0fSBlbHNlXG5cdFx0XHRcdFx0XHRjb2RlICs9IFwiICAgIFwiICsgbGF5ZXJJRCArIFwiIC0+IFwiICsgbGF5ZXJ0b0lEICsgXCIgW2xhYmVsID0gXCIgKyBzaXplICsgXCJdXFxuXCI7XG5cdFx0XHRcdFx0Zm9yICh2YXIgZnJvbSBpbiBjb25uZWN0aW9uLmdhdGVkZnJvbSkgeyAvLyBnYXRpbmdzXG5cdFx0XHRcdFx0XHR2YXIgbGF5ZXJmcm9tID0gY29ubmVjdGlvbi5nYXRlZGZyb21bZnJvbV0ubGF5ZXI7XG5cdFx0XHRcdFx0XHR2YXIgdHlwZSA9IGNvbm5lY3Rpb24uZ2F0ZWRmcm9tW2Zyb21dLnR5cGU7XG5cdFx0XHRcdFx0XHR2YXIgbGF5ZXJmcm9tSUQgPSBsYXllcnMuaW5kZXhPZihsYXllcmZyb20pO1xuXHRcdFx0XHRcdFx0Y29kZSArPSBcIiAgICBcIiArIGxheWVyZnJvbUlEICsgXCIgLT4gXCIgKyBmYWtlTm9kZSArIFwiIFtjb2xvciA9IGJsdWVdXFxuXCI7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9IGVsc2Uge1xuXHRcdFx0XHRcdGNvZGUgKz0gXCIgICAgXCIgKyBsYXllcklEICsgXCIgLT4gXCIgKyBsYXllcnRvSUQgKyBcIiBbbGFiZWwgPSBcIiArIHNpemUgKyBcIl1cXG5cIjtcblx0XHRcdFx0XHRmb3IgKHZhciBmcm9tIGluIGNvbm5lY3Rpb24uZ2F0ZWRmcm9tKSB7IC8vIGdhdGluZ3Ncblx0XHRcdFx0XHRcdHZhciBsYXllcmZyb20gPSBjb25uZWN0aW9uLmdhdGVkZnJvbVtmcm9tXS5sYXllcjtcblx0XHRcdFx0XHRcdHZhciB0eXBlID0gY29ubmVjdGlvbi5nYXRlZGZyb21bZnJvbV0udHlwZTtcblx0XHRcdFx0XHRcdHZhciBsYXllcmZyb21JRCA9IGxheWVycy5pbmRleE9mKGxheWVyZnJvbSk7XG5cdFx0XHRcdFx0XHRjb2RlICs9IFwiICAgIFwiICsgbGF5ZXJmcm9tSUQgKyBcIiAtPiBcIiArIGxheWVydG9JRCArIFwiIFtjb2xvciA9IGJsdWVdXFxuXCI7XG5cdFx0XHRcdFx0fVxuXHRcdFx0XHR9XG5cdFx0XHR9XG5cdFx0fVxuXHRcdGNvZGUgKz0gXCJ9XFxuXCI7XG5cdFx0cmV0dXJuIHtcblx0XHRcdGNvZGU6IGNvZGUsXG5cdFx0XHRsaW5rOiBcImh0dHBzOi8vY2hhcnQuZ29vZ2xlYXBpcy5jb20vY2hhcnQ/Y2hsPVwiICsgZXNjYXBlKGNvZGUucmVwbGFjZShcIi8gL2dcIiwgXCIrXCIpKSArIFwiJmNodD1ndlwiXG5cdFx0fVxuXHR9XG5cblx0Ly8gcmV0dXJucyBhIGZ1bmN0aW9uIHRoYXQgd29ya3MgYXMgdGhlIGFjdGl2YXRpb24gb2YgdGhlIG5ldHdvcmsgYW5kIGNhbiBiZSB1c2VkIHdpdGhvdXQgZGVwZW5kaW5nIG9uIHRoZSBsaWJyYXJ5XG5cdHN0YW5kYWxvbmUoKSB7XG5cdFx0aWYgKCF0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemUoKTtcblxuXHRcdHZhciBkYXRhID0gdGhpcy5vcHRpbWl6ZWQuZGF0YTtcblxuXHRcdC8vIGJ1aWxkIGFjdGl2YXRpb24gZnVuY3Rpb25cblx0XHR2YXIgYWN0aXZhdGlvbiA9IFwiZnVuY3Rpb24gKGlucHV0KSB7XFxuXCI7XG5cblx0XHQvLyBidWlsZCBpbnB1dHNcblx0XHRmb3IgKHZhciBpIGluIGRhdGEuaW5wdXRzKVxuXHRcdFx0YWN0aXZhdGlvbiArPSBcIkZbXCIgKyBkYXRhLmlucHV0c1tpXSArIFwiXSA9IGlucHV0W1wiICsgaSArIFwiXTtcXG5cIjtcblxuXHRcdC8vIGJ1aWxkIG5ldHdvcmsgYWN0aXZhdGlvblxuXHRcdGZvciAodmFyIG5ldXJvbiBpbiBkYXRhLmFjdGl2YXRlKSB7IC8vIHNob3VsZG4ndCB0aGlzIGJlIGxheWVyP1xuXHRcdFx0Zm9yICh2YXIgc2VudGVuY2UgaW4gZGF0YS5hY3RpdmF0ZVtuZXVyb25dKVxuXHRcdFx0XHRhY3RpdmF0aW9uICs9IGRhdGEuYWN0aXZhdGVbbmV1cm9uXVtzZW50ZW5jZV0gKyBcIlxcblwiO1xuXHRcdH1cblxuXHRcdC8vIGJ1aWxkIG91dHB1dHNcblx0XHRhY3RpdmF0aW9uICs9IFwidmFyIG91dHB1dCA9IFtdO1xcblwiO1xuXHRcdGZvciAodmFyIGkgaW4gZGF0YS5vdXRwdXRzKVxuXHRcdFx0YWN0aXZhdGlvbiArPSBcIm91dHB1dFtcIiArIGkgKyBcIl0gPSBGW1wiICsgZGF0YS5vdXRwdXRzW2ldICsgXCJdO1xcblwiO1xuXHRcdGFjdGl2YXRpb24gKz0gXCJyZXR1cm4gb3V0cHV0O1xcbn1cIjtcblxuXHRcdC8vIHJlZmVyZW5jZSBhbGwgdGhlIHBvc2l0aW9ucyBpbiBtZW1vcnlcblx0XHR2YXIgbWVtb3J5ID0gYWN0aXZhdGlvbi5tYXRjaCgvRlxcWyhcXGQrKVxcXS9nKTtcblx0XHR2YXIgZGltZW5zaW9uID0gMDtcblx0XHR2YXIgaWRzID0ge307XG5cdFx0Zm9yICh2YXIgYWRkcmVzcyBpbiBtZW1vcnkpIHtcblx0XHRcdHZhciB0bXAgPSBtZW1vcnlbYWRkcmVzc10ubWF0Y2goL1xcZCsvKVswXTtcblx0XHRcdGlmICghKHRtcCBpbiBpZHMpKSB7XG5cdFx0XHRcdGlkc1t0bXBdID0gZGltZW5zaW9uKys7XG5cdFx0XHR9XG5cdFx0fVxuXHRcdHZhciBoYXJkY29kZSA9IFwiRiA9IHtcXG5cIjtcblx0XHRmb3IgKHZhciBpIGluIGlkcylcblx0XHRcdGhhcmRjb2RlICs9IGlkc1tpXSArIFwiOiBcIiArIHRoaXMub3B0aW1pemVkLm1lbW9yeVtpXSArIFwiLFxcblwiO1xuXHRcdGhhcmRjb2RlID0gaGFyZGNvZGUuc3Vic3RyaW5nKDAsIGhhcmRjb2RlLmxlbmd0aCAtIDIpICsgXCJcXG59O1xcblwiO1xuXHRcdGhhcmRjb2RlID0gXCJ2YXIgcnVuID0gXCIgKyBhY3RpdmF0aW9uLnJlcGxhY2UoL0ZcXFsoXFxkKyldL2csIGZ1bmN0aW9uKFxuXHRcdFx0aW5kZXgpIHtcblx0XHRcdHJldHVybiAnRlsnICsgaWRzW2luZGV4Lm1hdGNoKC9cXGQrLylbMF1dICsgJ10nXG5cdFx0fSkucmVwbGFjZShcIntcXG5cIiwgXCJ7XFxuXCIgKyBoYXJkY29kZSArIFwiXCIpICsgXCI7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJyZXR1cm4gcnVuXCI7XG5cblx0XHQvLyByZXR1cm4gc3RhbmRhbG9uZSBmdW5jdGlvblxuXHRcdHJldHVybiBuZXcgRnVuY3Rpb24oaGFyZGNvZGUpKCk7XG5cdH1cblxuXHR3b3JrZXIoKSB7XG5cdFx0aWYgKCF0aGlzLm9wdGltaXplZClcblx0XHRcdHRoaXMub3B0aW1pemUoKTtcblxuXHRcdHZhciBoYXJkY29kZSA9IFwidmFyIGlucHV0cyA9IFwiICsgdGhpcy5vcHRpbWl6ZWQuZGF0YS5pbnB1dHMubGVuZ3RoICtcblx0XHRcdFwiO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwidmFyIG91dHB1dHMgPSBcIiArIHRoaXMub3B0aW1pemVkLmRhdGEub3V0cHV0cy5sZW5ndGggK1xuXHRcdFwiO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwidmFyIEYgPSBudWxsO1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwidmFyIGFjdGl2YXRlID0gXCIgKyB0aGlzLm9wdGltaXplZC5hY3RpdmF0ZS50b1N0cmluZygpICtcblx0XHRcIjtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcInZhciBwcm9wYWdhdGUgPSBcIiArIHRoaXMub3B0aW1pemVkLnByb3BhZ2F0ZS50b1N0cmluZygpICtcblx0XHRcIjtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcIm9ubWVzc2FnZSA9IGZ1bmN0aW9uKGUpe1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwiRiA9IGUuZGF0YS5tZW1vcnlCdWZmZXI7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJpZiAoZS5kYXRhLmFjdGlvbiA9PSAnYWN0aXZhdGUnKXtcXG5cIjtcblx0XHRoYXJkY29kZSArPSBcImlmIChlLmRhdGEuaW5wdXQubGVuZ3RoID09IGlucHV0cyl7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz1cblx0XHRcInBvc3RNZXNzYWdlKCB7IGFjdGlvbjogJ2FjdGl2YXRlJywgb3V0cHV0OiBhY3RpdmF0ZShlLmRhdGEuaW5wdXQpLCBtZW1vcnlCdWZmZXI6IEYgfSwgW0YuYnVmZmVyXSk7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ9XFxufVxcbmVsc2UgaWYgKGUuZGF0YS5hY3Rpb24gPT0gJ3Byb3BhZ2F0ZScpe1xcblwiO1xuXHRcdGhhcmRjb2RlICs9IFwicHJvcGFnYXRlKGUuZGF0YS5yYXRlLCBlLmRhdGEudGFyZ2V0KTtcXG5cIjtcblx0XHRoYXJkY29kZSArPVxuXHRcdFwicG9zdE1lc3NhZ2UoeyBhY3Rpb246ICdwcm9wYWdhdGUnLCBtZW1vcnlCdWZmZXI6IEYgfSwgW0YuYnVmZmVyXSk7XFxuXCI7XG5cdFx0aGFyZGNvZGUgKz0gXCJ9XFxufVxcblwiO1xuXG5cdFx0dmFyIGJsb2IgPSBuZXcgQmxvYihbaGFyZGNvZGVdKTtcblx0XHR2YXIgYmxvYlVSTCA9ICg8YW55PndpbmRvdykuVVJMLmNyZWF0ZU9iamVjdFVSTChibG9iKTtcblxuXHRcdHJldHVybiBuZXcgV29ya2VyKGJsb2JVUkwpO1xuXHR9XG5cblx0Ly8gcmV0dXJucyBhIGNvcHkgb2YgdGhlIG5ldHdvcmtcblx0Y2xvbmUoaWdub3JlVHJhY2VzKSB7XG5cdFx0cmV0dXJuIE5ldHdvcmsuZnJvbUpTT04odGhpcy50b0pTT04oaWdub3JlVHJhY2VzKSk7XG5cdH1cblxuXHRzdGF0aWMgZnJvbUpTT04oanNvbikge1xuXG5cdFx0dmFyIG5ldXJvbnMgPSBbXTtcblxuXHRcdHZhciBsYXllcnMgPSB7XG5cdFx0XHRpbnB1dDogbmV3IGxheWVyLkxheWVyKDApLFxuXHRcdFx0aGlkZGVuOiBbXSxcblx0XHRcdG91dHB1dDogbmV3IGxheWVyLkxheWVyKDApXG5cdFx0fVxuXHRcdFxuXG5cdFx0Zm9yICh2YXIgaSBpbiBqc29uLm5ldXJvbnMpIHtcblx0XHRcdHZhciBjb25maWcgPSBqc29uLm5ldXJvbnNbaV07XG5cblx0XHRcdHZhciBuZXVyb24gPSBuZXcgX25ldXJvbi5OZXVyb24oKTtcblx0XHRcdG5ldXJvbi50cmFjZS5lbGVnaWJpbGl0eSA9IGNvbmZpZy50cmFjZS5lbGVnaWJpbGl0eTtcblx0XHRcdG5ldXJvbi50cmFjZS5leHRlbmRlZCA9IGNvbmZpZy50cmFjZS5leHRlbmRlZDtcblx0XHRcdG5ldXJvbi5zdGF0ZSA9IGNvbmZpZy5zdGF0ZTtcblx0XHRcdG5ldXJvbi5vbGQgPSBjb25maWcub2xkO1xuXHRcdFx0bmV1cm9uLmFjdGl2YXRpb24gPSBjb25maWcuYWN0aXZhdGlvbjtcblx0XHRcdG5ldXJvbi5iaWFzID0gY29uZmlnLmJpYXM7XG5cdFx0XHRuZXVyb24uc3F1YXNoID0gY29uZmlnLnNxdWFzaCBpbiBTcXVhc2ggPyBTcXVhc2hbY29uZmlnLnNxdWFzaF0gOlxuXHRcdFx0XHRTcXVhc2guTE9HSVNUSUM7XG5cdFx0XHRuZXVyb25zLnB1c2gobmV1cm9uKTtcblxuXHRcdFx0aWYgKGNvbmZpZy5sYXllciA9PSAnaW5wdXQnKVxuXHRcdFx0XHRsYXllcnMuaW5wdXQuYWRkKG5ldXJvbik7XG5cdFx0XHRlbHNlIGlmIChjb25maWcubGF5ZXIgPT0gJ291dHB1dCcpXG5cdFx0XHRcdGxheWVycy5vdXRwdXQuYWRkKG5ldXJvbik7XG5cdFx0XHRlbHNlIHtcblx0XHRcdFx0aWYgKHR5cGVvZiBsYXllcnMuaGlkZGVuW2NvbmZpZy5sYXllcl0gPT0gJ3VuZGVmaW5lZCcpXG5cdFx0XHRcdFx0bGF5ZXJzLmhpZGRlbltjb25maWcubGF5ZXJdID0gbmV3IGxheWVyLkxheWVyKDApO1xuXHRcdFx0XHRsYXllcnMuaGlkZGVuW2NvbmZpZy5sYXllcl0uYWRkKG5ldXJvbik7XG5cdFx0XHR9XG5cdFx0fVxuXG5cdFx0Zm9yICh2YXIgaSBpbiBqc29uLmNvbm5lY3Rpb25zKSB7XG5cdFx0XHR2YXIgY29uZmlnID0ganNvbi5jb25uZWN0aW9uc1tpXTtcblx0XHRcdHZhciBmcm9tID0gbmV1cm9uc1tjb25maWcuZnJvbV07XG5cdFx0XHR2YXIgdG8gPSBuZXVyb25zW2NvbmZpZy50b107XG5cdFx0XHR2YXIgd2VpZ2h0ID0gY29uZmlnLndlaWdodFxuXHRcdFx0dmFyIGdhdGVyID0gbmV1cm9uc1tjb25maWcuZ2F0ZXJdO1xuXG5cdFx0XHR2YXIgY29ubmVjdGlvbiA9IGZyb20ucHJvamVjdCh0bywgd2VpZ2h0KTtcblx0XHRcdGlmIChnYXRlcilcblx0XHRcdFx0Z2F0ZXIuZ2F0ZShjb25uZWN0aW9uKTtcblx0XHR9XG5cblx0XHRyZXR1cm4gbmV3IE5ldHdvcmsobGF5ZXJzKTtcblx0fVxufVxuXG5leHBvcnQgbW9kdWxlIE5ldHdvcmsge1xuXHRleHBvcnQgaW50ZXJmYWNlIElOZXR3b3JrTmV1cm9uIHtcblx0XHRuZXVyb246IF9uZXVyb24uTmV1cm9uO1xuXHRcdGxheWVyOiBzdHJpbmc7XG5cdH1cbn0iXSwic291cmNlUm9vdCI6Ii9zb3VyY2UvIn0=