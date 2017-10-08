import LayerConnection from './LayerConnection';
import Neuron from './Neuron';
import Network from './Network';


// types of connections
const connectionType = {
  ALL_TO_ALL: "ALL TO ALL",
  ONE_TO_ONE: "ONE TO ONE",
  ALL_TO_ELSE: "ALL TO ELSE"
};

// types of gates
const gateType = {
  INPUT: "INPUT",
  OUTPUT: "OUTPUT",
  ONE_TO_ONE: "ONE TO ONE"
};


export default class Layer {
  static connectionType = connectionType;
  static gateType = gateType;

  constructor(size) {
    this.size = size | 0;
    this.list = [];

    this.connectedTo = [];

    while (size--) {
      var neuron = new Neuron();
      this.list.push(neuron);
    }
  }

  // activates all the neurons in the layer
  activate(input) {

    var activations = [];

    if (typeof input != 'undefined') {
      if (input.length != this.size)
        throw new Error('INPUT size and LAYER size must be the same to activate!');

      for (var id in this.list) {
        var neuron = this.list[id];
        var activation = neuron.activate(input[id]);
        activations.push(activation);
      }
    } else {
      for (var id in this.list) {
        var neuron = this.list[id];
        var activation = neuron.activate();
        activations.push(activation);
      }
    }
    return activations;
  }

// propagates the error on all the neurons of the layer
  propagate(rate, target) {

    if (typeof target != 'undefined') {
      if (target.length != this.size)
        throw new Error('TARGET size and LAYER size must be the same to propagate!');

      for (var id = this.list.length - 1; id >= 0; id--) {
        var neuron = this.list[id];
        neuron.propagate(rate, target[id]);
      }
    } else {
      for (var id = this.list.length - 1; id >= 0; id--) {
        var neuron = this.list[id];
        neuron.propagate(rate);
      }
    }
  }

// projects a connection from this layer to another one
  project(layer, type, weights) {

    if (layer instanceof Network)
      layer = layer.layers.input;

    if (layer instanceof Layer) {
      if (!this.connected(layer))
        return new LayerConnection(this, layer, type, weights);
    } else
      throw new Error('Invalid argument, you can only project connections to LAYERS and NETWORKS!');


  }

// gates a connection betwenn two layers
  gate(connection, type) {

    if (type == Layer.gateType.INPUT) {
      if (connection.to.size != this.size)
        throw new Error('GATER layer and CONNECTION.TO layer must be the same size in order to gate!');

      for (var id in connection.to.list) {
        var neuron = connection.to.list[id];
        var gater = this.list[id];
        for (var input in neuron.connections.inputs) {
          var gated = neuron.connections.inputs[input];
          if (gated.ID in connection.connections)
            gater.gate(gated);
        }
      }
    } else if (type == Layer.gateType.OUTPUT) {
      if (connection.from.size != this.size)
        throw new Error('GATER layer and CONNECTION.FROM layer must be the same size in order to gate!');

      for (var id in connection.from.list) {
        var neuron = connection.from.list[id];
        var gater = this.list[id];
        for (var projected in neuron.connections.projected) {
          var gated = neuron.connections.projected[projected];
          if (gated.ID in connection.connections)
            gater.gate(gated);
        }
      }
    } else if (type == Layer.gateType.ONE_TO_ONE) {
      if (connection.size != this.size)
        throw new Error('The number of GATER UNITS must be the same as the number of CONNECTIONS to gate!');

      for (var id in connection.list) {
        var gater = this.list[id];
        var gated = connection.list[id];
        gater.gate(gated);
      }
    }
    connection.gatedfrom.push({layer: this, type: type});
  }

// true or false whether the whole layer is self-connected or not
  selfconnected() {

    for (var id in this.list) {
      var neuron = this.list[id];
      if (!neuron.selfconnected())
        return false;
    }
    return true;
  }

// true of false whether the layer is connected to another layer (parameter) or not
  connected(layer) {
    // Check if ALL to ALL connection
    var connections = 0;
    for (var here in this.list) {
      for (var there in layer.list) {
        var from = this.list[here];
        var to = layer.list[there];
        var connected = from.connected(to);
        if (connected.type == 'projected')
          connections++;
      }
    }
    if (connections == this.size * layer.size)
      return Layer.connectionType.ALL_TO_ALL;

    // Check if ONE to ONE connection
    connections = 0;
    for (var neuron in this.list) {
      var from = this.list[neuron];
      var to = layer.list[neuron];
      var connected = from.connected(to);
      if (connected.type == 'projected')
        connections++;
    }
    if (connections == this.size)
      return Layer.connectionType.ONE_TO_ONE;
  }

// clears all the neuorns in the layer
  clear() {
    for (var id in this.list) {
      var neuron = this.list[id];
      neuron.clear();
    }
  }

// resets all the neurons in the layer
  reset() {
    for (var id in this.list) {
      var neuron = this.list[id];
      neuron.reset();
    }
  }

// returns all the neurons in the layer (array)
  neurons() {
    return this.list;
  }

// adds a neuron to the layer
  add(neuron) {
    neuron = neuron || new Neuron();
    this.list.push(neuron);
    this.size++;
  }

  set(options) {
    options = options || {};

    for (var i in this.list) {
      var neuron = this.list[i];
      if (options.label)
        neuron.label = options.label + '_' + neuron.ID;
      if (options.squash)
        neuron.squash = options.squash;
      if (options.bias)
        neuron.bias = options.bias;
    }
    return this;
  }
}