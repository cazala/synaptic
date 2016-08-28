// Multilayer Perceptron
const AbstractArchitecture = require('./abstract-architecture');
const Layer = require('../layer');

module.exports = class Perceptron extends AbstractArchitecture {
  constructor(...args) {
    if (args.length < 3)
      throw new Error(`Not enough layers. Minimum 3 expected, instead got ${args[0]}, ${args[1]}`);

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
  }
};