import network  = require('../network');
import trainer  = require('../trainer');
import Layer  = require('../layer');
import neuron = require('../neuron');

export class LSTM extends network.Network {
  trainer: trainer.Trainer;

  constructor(...args: any[]) {

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
    } else
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
    super({
      input: inputLayer,
      hidden: hiddenLayers,
      output: outputLayer
    });

    // trainer
    this.trainer = new trainer.Trainer(this);
  }
};
