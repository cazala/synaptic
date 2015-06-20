import network  = require('../network');
import trainer  = require('../trainer');
import layer  = require('../layer');
import neuron = require('../neuron');

export class Liquid extends network.Network {
  trainer: trainer.Trainer;

  constructor(inputs, hidden, outputs, connections, gates) {

    // create layers
    var inputLayer = new layer.Layer(inputs);
    var hiddenLayer = new layer.Layer(hidden);
    var outputLayer = new layer.Layer(outputs);

    // make connections and gates randomly among the neurons
    var neurons = hiddenLayer.neurons();
    var connectionList: neuron.Neuron.Connection[] = [];

    for (var i = 0; i < connections; i++) {
      // connect two random neurons
      var from = Math.random() * neurons.length | 0;
      var to = Math.random() * neurons.length | 0;
      var connection = neurons[from].project(neurons[to]);
      connectionList.push(connection);
    }

    for (var j = 0; j < gates; j++) {
      // pick a random gater neuron
      var gater = Math.random() * neurons.length | 0;
      // pick a random connection to gate
      var connectionNumber = Math.random() * connectionList.length | 0;
      // let the gater gate the connection
      neurons[gater].gate(connectionList[connectionNumber]);
    }

    // connect the layers
    inputLayer.project(hiddenLayer);
    hiddenLayer.project(outputLayer);

    // set the layers of the network
    super({
      input: inputLayer,
      hidden: [hiddenLayer],
      output: outputLayer
    });

    // trainer
    this.trainer = new trainer.Trainer(this);
  }
}
