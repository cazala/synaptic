const Synaptic = {
    Neuron: require('./neuron'),
    Layer: require('./layer'),
    Network: require('./network'),
    Trainer: require('./trainer'),
    Architect: require('./architect')
};

module.exports = Synaptic;
// for exports
module.exports.synaptic = Synaptic;

if (global.synaptic) {
  const oldSynaptic = global.synaptic;
  Synaptic.ninja = () => {
    global.synaptic = oldSynaptic;
    return Synaptic;
  };

  global.synaptic = Synaptic;
}
