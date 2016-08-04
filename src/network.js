const Neuron = require('./neuron');
const Layer = require('./layer');
const Trainer = require('./trainer');

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

      copy.squash = neuron.squash == Neuron.squash.LOGISTIC ? "LOGISTIC" :
        neuron.squash == Neuron.squash.TANH ? "TANH" :
          neuron.squash == Neuron.squash.IDENTITY ? "IDENTITY" :
            neuron.squash == Neuron.squash.HLIM ? "HLIM" :
              null;

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

  // Return a HTML5 WebWorker specialized on training the network stored in `memory`.
  // Train based on the given dataSet and options.
  // The worker returns the updated `memory` when done.
  worker(memory, set, options) {

    // Copy the options and set defaults (options might be different for each worker)
    let workerOptions = {};
    if (options) workerOptions = options;
    workerOptions.rate = options.rate || .2;
    workerOptions.iterations = options.iterations || 100000;
    workerOptions.error = options.error || .005;
    workerOptions.cost = options.cost || null;
    workerOptions.crossValidate = options.crossValidate || null;

    // Cost function might be different for each worker
    costFunction = `var cost = ${options && options.cost || this.cost || Trainer.cost.MSE};\n`;
    let workerFunction = Network.getWorkerSharedFunctions();
    workerFunction = workerFunction.replace(/var cost = options && options\.cost \|\| this\.cost \|\| Trainer\.cost\.MSE;/g, costFunction);

    // Set what we do when training is finished
    workerFunction = workerFunction.replace('return results;',
      'postMessage({action: "done", message: results, memoryBuffer: F}, [F.buffer]);');

    // Replace log with postmessage
    workerFunction = workerFunction.replace("console.log('iterations', iterations, 'error', error, 'rate', currentRate)",
      "postMessage({action: 'log', message: {\n" +
      "iterations: iterations,\n" +
      "error: error,\n" +
      "rate: currentRate\n" +
      "}\n" +
      "})");

    // Replace schedule with postmessage
    workerFunction = workerFunction.replace("abort = this.schedule.do({ error: error, iterations: iterations, rate: currentRate })",
      "postMessage({action: 'schedule', message: {\n" +
      "iterations: iterations,\n" +
      "error: error,\n" +
      "rate: currentRate\n" +
      "}\n" +
      "})");

    if (!this.optimized)
      this.optimize();

    let hardcode = `var inputs = ${this.optimized.data.inputs.length};\n`;
    hardcode += `var outputs = ${this.optimized.data.outputs.length};\n`;
    hardcode += `var F =  new Float64Array([${this.optimized.memory.toString()}]);\n`;
    hardcode += `var activate = ${this.optimized.activate.toString()};\n`;
    hardcode += `var propagate = ${this.optimized.propagate.toString()};\n`;
    hardcode +=
      `onmessage = function(e) {\nif (e.data.action == 'startTraining') {\ntrain(${JSON.stringify(set)},${JSON.stringify(workerOptions)});\n}\n}`;

    const workerSourceCode = `${workerFunction}\n${hardcode}`;
    const blob = new Blob([workerSourceCode]);
    const blobURL = window.URL.createObjectURL(blob);

    return new Worker(blobURL);
  }

  // returns a copy of the network
  clone() {
    return Network.fromJSON(this.toJSON());
  }
}

/**
 * Creates a static String to store the source code of the functions
 *  that are identical for all the workers (train, _trainSet, test)
 *
 * @return {String} Source code that can train a network inside a worker.
 * @static
 */
Network.getWorkerSharedFunctions = () => {
  // If we already computed the source code for the shared functions
  if (typeof Network._SHARED_WORKER_FUNCTIONS !== 'undefined')
    return Network._SHARED_WORKER_FUNCTIONS;

  // Otherwise compute and return the source code
  // We compute them by simply copying the source code of the train, _trainSet and test functions
  //  using the .toString() method

  // Load and name the train function
  let train_f = Trainer.prototype.train.toString();
  train_f = `${train_f.replace('function (set', 'function train(set')}\n`;

  // Load and name the _trainSet function
  let _trainSet_f = Trainer.prototype._trainSet.toString().replace(/this.network./g, '');
  _trainSet_f = `${_trainSet_f.replace('function (set', 'function _trainSet(set')}\n`;
  _trainSet_f = _trainSet_f.replace('this.crossValidate', 'crossValidate');
  _trainSet_f = _trainSet_f.replace('crossValidate = true', 'crossValidate = { }');

  // Load and name the test function
  let test_f = Trainer.prototype.test.toString().replace(/this.network./g, '');
  test_f = `${test_f.replace('function (set', 'function test(set')}\n`;

  return Network._SHARED_WORKER_FUNCTIONS = train_f + _trainSet_f + test_f;
};

// rebuild a network that has been stored in a json using the method toJSON()
Network.fromJSON = json => {

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
    neuron.squash = config.squash in Neuron.squash ? Neuron.squash[config.squash] : Neuron.squash.LOGISTIC;
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

  return new Network({
    input: new Layer(layers.input),
    hidden: layers.hidden.map(layer => new Layer(layer)),
    output: new Layer(layers.output)
  });
};

module.exports = Network;