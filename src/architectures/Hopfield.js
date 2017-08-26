import Network from '../network';
import Trainer from '../trainer';
import Layer from '../layer';

export default class Hopfield extends Network {
  constructor(size) {
    super();
    var inputLayer = new Layer(size);
    var outputLayer = new Layer(size);

    inputLayer.project(outputLayer, Layer.connectionType.ALL_TO_ALL);

    this.set({
      input: inputLayer,
      hidden: [],
      output: outputLayer
    });

    this.trainer = new Trainer(this);
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

    var pattern = [];
    for (var i in output)
      pattern[i] = output[i] > .5 ? 1 : 0;

    return pattern;
  }
}