import network  = require('../network');
import trainer  = require('../trainer');
import layer  = require('../layer');
import neuron = require('../neuron');
// Multilayer Perceptron
export class Perceptron extends network.Network {
  trainer: trainer.Trainer;

  constructor(...args: number[]) {

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
      
    super({
      input: input,
      hidden: hidden,
      output: output
    });
  
    // trainer for the network
    this.trainer = new trainer.Trainer(this);
  }
}; 