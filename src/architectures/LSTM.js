import Network from '../network';
import Layer from '../layer';

export default class LSTM extends Network {
  constructor() {
    super();
    var args = Array.prototype.slice.call(arguments); // convert arguments to array
    if (args.length < 3)
      throw new Error("not enough layers (minimum 3) !!");

    var last = args.pop();
    var option = {
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
    } else {
      var outputs = last;
    }

    var inputs = args.shift();
    var layers = args;

    var inputLayer = new Layer(inputs);
    var hiddenLayers = [];
    var outputLayer = new Layer(outputs);

    var previous = null;

    // generate layers
    for (var i = 0; i < layers.length; i++) {
      // generate memory blocks (memory cell and respective gates)
      var size = layers[i];

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
    this.set({
      input: inputLayer,
      hidden: hiddenLayers,
      output: outputLayer
    });
  }
}