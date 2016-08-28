const AbstractArchitecture = require('./abstract-architecture');
const Layer = require('../layer');

module.exports = class Hopfield extends AbstractArchitecture {
  constructor(size) {
    const inputLayer = new Layer(size);
    const outputLayer = new Layer(size);

    inputLayer.project(outputLayer, Layer.connectionType.ALL_TO_ALL);

    super({
      input: inputLayer,
      hidden: [],
      output: outputLayer
    });
  }

  learn(patterns) {
    const set = [];
    for (let p in patterns)
      set.push({
        input: patterns[p],
        output: patterns[p]
      });

    return this.trainer.train(set, {
      iterations: 500000,
      error: .00005,
      rate: 1
    });
  }

  feed(pattern) {
    const output = this.activate(pattern);

    pattern = [];
    for (let i in output)
      pattern[i] = output[i] > .5 ? 1 : 0;

    return pattern;
  }
}