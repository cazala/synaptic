const Network = require('../network');
const Layer = require('../layer');
const Trainer = require('../trainer');

module.exports = class Hopfield extends Network {
  constructor(size) {
    const inputLayer = new Layer(size);
    const outputLayer = new Layer(size);

    inputLayer.project(outputLayer, Layer.connectionType.ALL_TO_ALL);

    super({
      input: inputLayer,
      hidden: [],
      output: outputLayer
    });

    this.trainer = new Trainer(this);
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