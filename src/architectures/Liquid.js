import Network from '../network';
import Layer from '../layer';

export default class Liquid extends Network {
  constructor(inputs, hidden, outputs, connections, gates) {
    super();
    // create layers
    var inputLayer = new Layer(inputs);
    var hiddenLayer = new Layer(hidden);
    var outputLayer = new Layer(outputs);

    // make connections and gates randomly among the neurons
    var neurons = hiddenLayer.neurons();
    var connectionList = [];

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
      var connection = Math.random() * connectionList.length | 0;
      // let the gater gate the connection
      neurons[gater].gate(connectionList[connection]);
    }

    // connect the layers
    inputLayer.project(hiddenLayer);
    hiddenLayer.project(outputLayer);

    // set the layers of the network
    this.set({
      input: inputLayer,
      hidden: [hiddenLayer],
      output: outputLayer
    });
  }
}