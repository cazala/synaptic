var Synaptic = {};

Synaptic.Neuron = require('./neuron');
Synaptic.Layer = require('./layer');
Synaptic.Network = require('./network/index');
Synaptic.Trainer = require('./trainer');
Synaptic.Architect = require('./architect');

module.exports = Synaptic;

// Browser
if (typeof window == 'object') {
  //noinspection CommaExpressionJS
  Synaptic.ninja = ((oldSynaptic = window['synaptic']) =>
      () => (window['synaptic'] = oldSynaptic, Synaptic))();

  window['synaptic'] = Synaptic;
}
