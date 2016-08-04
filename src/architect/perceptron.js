// Multilayer Perceptron
const Network = require('../network');
const Layer = require('../layer');
const Trainer = require('../trainer');

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