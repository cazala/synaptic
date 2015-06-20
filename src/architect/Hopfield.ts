import network  = require('../network');
import trainer  = require('../trainer');
import layer  = require('../layer');
import neuron = require('../neuron');

export class Hopfield extends network.Network {
  trainer: trainer.Trainer;

  constructor(size: number) {
    var inputLayer = new layer.Layer(size);
    var outputLayer = new layer.Layer(size);

    inputLayer.project(outputLayer, layer.Layer.connectionType.ALL_TO_ALL);

    super({
      input: inputLayer,
      hidden: [],
      output: outputLayer
    });

    this.trainer = new trainer.Trainer(this);
  }

  learn(patterns) {
    var set = [];
    for (var p in patterns)
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
    var output = this.activate(pattern);

    var patterns = [];
    for (var i in output)
      patterns[i] = output[i] > .5 ? 1 : 0;

    return patterns;
  }
}