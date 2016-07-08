var Synaptic = {
    Neuron: require('./neuron'),
    Layer: require('./layer'),
    Network: require('./network'),
    Trainer: require('./trainer'),
    Architect: require('./architect')
};

// CommonJS & AMD
if (typeof define !== 'undefined' && define.amd)
{
  define([], function(){ return Synaptic });
}

// Node.js
if (typeof module !== 'undefined' && module.exports)
{
  module.exports = Synaptic;
}

// Browser
if (typeof window == 'object')
{
  (function(){
    var oldSynaptic = window['synaptic'];
    Synaptic.ninja = function(){
      window['synaptic'] = oldSynaptic;
      return Synaptic;
    };
  })();

  window['synaptic'] = Synaptic;
}
