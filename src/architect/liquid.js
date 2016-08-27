// Liquid State Machine
const Network = require('../network');
const Layer = require('../layer');
const Trainer = require('../trainer');

module.exports = class Liquid extends Network {
  constructor(inputs, hidden, outputs, connections, gates) {
    // create layers
    const inputLayer = new Layer(inputs);
    const hiddenLayer = new Layer(hidden);
    const outputLayer = new Layer(outputs);

    // make connections and gates randomly among the neurons
    const neurons = hiddenLayer.neurons();
    const connectionList = [];

    for (let i = 0; i < connections; i++) {
      // connect two random neurons
      const from = Math.random() * neurons.length | 0;
      const to = Math.random() * neurons.length | 0;
      var connection = neurons[from].project(neurons[to]);
      connectionList.push(connection);
    }

    for (let j = 0; j < gates; j++) {
      // pick a random gater neuron
      const gater = Math.random() * neurons.length | 0;
      // pick a random connection to gate
      var connection = Math.random() * connectionList.length | 0;
      // let the gater gate the connection
      neurons[gater].gate(connectionList[connection]);
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
    this.trainer = new Trainer(this);
  }
}