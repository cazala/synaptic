var Synaptic = {
    Neuron: require('./neuron'),
    Layer: require('./layer'),
    Network: require('./network'),
    Trainer: require('./trainer'),
    Architect: require('./architect')
};

module.exports = Synaptic;

// Browser
if (typeof window == 'object') {
  (function(){
    var oldSynaptic = window['synaptic'];
    Synaptic.ninja = function(){
      window['synaptic'] = oldSynaptic;
      return Synaptic;
    };
  })();

  window['synaptic'] = Synaptic;
}
