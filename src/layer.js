const Neuron = require('./neuron');

// avoiding cyclic dependency
const isNetwork = (object) => {
  if (!isNetwork.Network) {
    isNetwork.Network = require('./network');
  }

  return object instanceof isNetwork.Network;
};

/*******************************************************************************************
 LAYER
 *******************************************************************************************/

class Layer {
  constructor(size, label) {
    this.label = label || null;
    this.connectedTo = [];

    if (Array.isArray(size)) {
      const neurons = size;
      this.size = neurons.length;
      this.list = neurons.slice(); // cloning the array
    } else {
      this.size = size | 0;
      this.list = [];

      while (size--) {
        const neuron = new Neuron();
        this.list.push(neuron);
      }
    }
  }

  // activates all the neurons in the layer
  activate(input) {

    const activations = [];

    if (typeof input != 'undefined') {
      if (input.length != this.size)
        throw new Error("INPUT size and LAYER size must be the same to activate!");

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
        throw new Error("TARGET size and LAYER size must be the same to propagate!");

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

    if (isNetwork(layer))
      layer = layer.layers.input;

    if (layer instanceof Layer) {
      if (!this.connected(layer))
        return new Layer.connection(this, layer, type, weights);
    } else
      throw new Error("Invalid argument, you can only project connections to LAYERS and NETWORKS!");
  }

  // gates a connection betwenn two layers
  gate(connection, type) {

    if (type == Layer.gateType.INPUT) {
      if (connection.to.size != this.size)
        throw new Error("GATER layer and CONNECTION.TO layer must be the same size in order to gate!");

      for (var id in connection.to.list) {
        var neuron = connection.to.list[id];
        var gater = this.list[id];
        for (let input in neuron.connections.inputs) {
          var gated = neuron.connections.inputs[input];
          if (gated.ID in connection.connections)
            gater.gate(gated);
        }
      }
    } else if (type == Layer.gateType.OUTPUT) {
      if (connection.from.size != this.size)
        throw new Error("GATER layer and CONNECTION.FROM layer must be the same size in order to gate!");

      for (var id in connection.from.list) {
        var neuron = connection.from.list[id];
        var gater = this.list[id];
        for (let projected in neuron.connections.projected) {
          var gated = neuron.connections.projected[projected];
          if (gated.ID in connection.connections)
            gater.gate(gated);
        }
      }
    } else if (type == Layer.gateType.ONE_TO_ONE) {
      if (connection.size != this.size)
        throw new Error("The number of GATER UNITS must be the same as the number of CONNECTIONS to gate!");

      for (var id in connection.list) {
        var gater = this.list[id];
        var gated = connection.list[id];
        gater.gate(gated);
      }
    }
    connection.gatedfrom.push({layer: this, type});
  }

  // true or false whether the whole layer is self-connected or not
  selfconnected() {

    for (let id in this.list) {
      const neuron = this.list[id];
      if (!neuron.selfconnected())
        return false;
    }
    return true;
  }

  // true of false whether the layer is connected to another layer (parameter) or not
  connected(layer) {
    // Check if ALL to ALL connection
    let connections = 0;
    for (let here in this.list) {
      for (let there in layer.list) {
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
    for (let neuron in this.list) {
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
    for (let id in this.list) {
      const neuron = this.list[id];
      neuron.clear();
    }
  }

  // resets all the neurons in the layer
  reset() {
    for (let id in this.list) {
      const neuron = this.list[id];
      neuron.reset();
    }
  }

  // returns all the neurons in the layer (array)
  neurons() {
    return this.list;
  }


  set(options = {}) {
    for (let i in this.list) {
      const neuron = this.list[i];
      if (options.label)
        neuron.label = `${options.label}_${neuron.ID}`;
      if (options.squash)
        neuron.squash = options.squash;
      if (options.bias)
        neuron.bias = options.bias;
    }
    return this;
  }
}

// represents a connection from one layer to another, and keeps track of its weight and gain
Layer.connection = class LayerConnection {
  constructor(fromLayer, toLayer, type, weights) {
    this.ID = Layer.connection.uid();
    this.from = fromLayer;
    this.to = toLayer;
    this.selfconnection = toLayer == fromLayer;
    this.type = type;
    this.connections = {};
    this.list = [];
    this.size = 0;
    this.gatedfrom = [];

    if (typeof this.type == 'undefined') {
      if (fromLayer == toLayer)
        this.type = Layer.connectionType.ONE_TO_ONE;
      else
        this.type = Layer.connectionType.ALL_TO_ALL;
    }

    if (this.type == Layer.connectionType.ALL_TO_ALL ||
      this.type == Layer.connectionType.ALL_TO_ELSE) {
      for (let here in this.from.list) {
        for (let there in this.to.list) {
          var from = this.from.list[here];
          var to = this.to.list[there];
          if (this.type == Layer.connectionType.ALL_TO_ELSE && from == to)
            continue;
          var connection = from.project(to, weights);

          this.connections[connection.ID] = connection;
          this.size = this.list.push(connection);
        }
      }
    } else if (this.type == Layer.connectionType.ONE_TO_ONE) {

      for (let neuron in this.from.list) {
        var from = this.from.list[neuron];
        var to = this.to.list[neuron];
        var connection = from.project(to, weights);

        this.connections[connection.ID] = connection;
        this.size = this.list.push(connection);
      }
    }

    fromLayer.connectedTo.push(this);
  }
}

// types of connections
Layer.connectionType = {};
Layer.connectionType.ALL_TO_ALL = "ALL TO ALL";
Layer.connectionType.ONE_TO_ONE = "ONE TO ONE";
Layer.connectionType.ALL_TO_ELSE = "ALL TO ELSE";

// types of gates
Layer.gateType = {};
Layer.gateType.INPUT = "INPUT";
Layer.gateType.OUTPUT = "OUTPUT";
Layer.gateType.ONE_TO_ONE = "ONE TO ONE";

((() => {
  let connections = 0;
  Layer.connection.uid = () => connections++
}))();

module.exports = Layer;