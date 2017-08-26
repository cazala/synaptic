import Neuron from './neuron';
import Layer from './layer';
import Trainer from './trainer'
import Network from './network';
import * as Architect from './architect';

var Synaptic = {
    Neuron,
    Layer,
    Network,
    Trainer,
    Architect
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
