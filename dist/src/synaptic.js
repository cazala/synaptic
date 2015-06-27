/*
********************************************************************************************
                                         SYNAPTIC
********************************************************************************************

Synaptic is a javascript neural network library for node.js and the browser, its generalized
algorithm is architecture-free, so you can build and train basically any type of first order
or even second order neural network architectures.

http://en.wikipedia.org/wiki/Recurrent_neural_network#Second_Order_Recurrent_Neural_Network

The library includes a few built-in architectures like multilayer perceptrons, multilayer
long-short term memory networks (LSTM) or liquid state machines, and a trainer capable of
training any given network, and includes built-in training tasks/tests like solving an XOR,
passing a Distracted Sequence Recall test or an Embeded Reber Grammar test.

The algorithm implemented by this library has been taken from Derek D. Monner's paper:

A generalized LSTM-like training algorithm for second-order recurrent neural networks
http://www.overcomplete.net/papers/nn2012.pdf

There are references to the equations in that paper commented through the source code.


********************************************************************************************/
var network = require('./network');
var layer = require('./layer');
var neuron = require('./neuron');
var trainer = require('./trainer');
var architect = require('./architect');
var squash = require('./squash');
var utils = require('./utils');
var Synaptic;
(function (Synaptic) {
    var oldSynaptic = typeof window != "undefined" && window && window['Synaptic'];
    function ninja() {
        window['synaptic'] = oldSynaptic;
        return Synaptic;
    }
    Synaptic.ninja = ninja;
    Synaptic.Neuron = neuron.Neuron;
    Synaptic.Layer = layer.Layer;
    Synaptic.Network = network.Network;
    Synaptic.Trainer = trainer.Trainer;
    Synaptic.Squash = squash;
    Synaptic.Architect = architect;
    Synaptic.Utils = utils.Utils;
})(Synaptic || (Synaptic = {}));
if (typeof window != "undefined")
    window['synaptic'] = Synaptic;
module.exports = Synaptic;

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9zeW5hcHRpYy50cyJdLCJuYW1lcyI6WyJTeW5hcHRpYyIsIlN5bmFwdGljLm5pbmphIl0sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OzZGQXdCNkY7QUFJN0YsSUFBTyxPQUFPLFdBQVcsV0FBVyxDQUFDLENBQUM7QUFDdEMsSUFBTyxLQUFLLFdBQVcsU0FBUyxDQUFDLENBQUM7QUFDbEMsSUFBTyxNQUFNLFdBQVcsVUFBVSxDQUFDLENBQUM7QUFDcEMsSUFBTyxPQUFPLFdBQVcsV0FBVyxDQUFDLENBQUM7QUFDdEMsSUFBTyxTQUFTLFdBQVcsYUFBYSxDQUFDLENBQUM7QUFDMUMsSUFBTyxNQUFNLFdBQVcsVUFBVSxDQUFDLENBQUM7QUFDcEMsSUFBTyxLQUFLLFdBQVcsU0FBUyxDQUFDLENBQUM7QUFJbEMsSUFBTyxRQUFRLENBcUNkO0FBckNELFdBQU8sUUFBUSxFQUFDLENBQUM7SUFLaEJBLElBQUlBLFdBQVdBLEdBQUdBLE9BQU9BLE1BQU1BLElBQUlBLFdBQVdBLElBQUlBLE1BQU1BLElBQUlBLE1BQU1BLENBQUNBLFVBQVVBLENBQUNBLENBQUNBO0lBRS9FQSxTQUFnQkEsS0FBS0E7UUFDaEJDLE1BQU1BLENBQUNBLFVBQVVBLENBQUNBLEdBQUdBLFdBQVdBLENBQUNBO1FBQ2pDQSxNQUFNQSxDQUFDQSxRQUFRQSxDQUFDQTtJQUNyQkEsQ0FBQ0E7SUFIZUQsY0FBS0EsR0FBTEEsS0FHZkEsQ0FBQUE7SUFvQlVBLGVBQU1BLEdBQUdBLE1BQU1BLENBQUNBLE1BQU1BLENBQUNBO0lBQ3ZCQSxjQUFLQSxHQUFHQSxLQUFLQSxDQUFDQSxLQUFLQSxDQUFDQTtJQUNwQkEsZ0JBQU9BLEdBQUdBLE9BQU9BLENBQUNBLE9BQU9BLENBQUNBO0lBQzFCQSxnQkFBT0EsR0FBR0EsT0FBT0EsQ0FBQ0EsT0FBT0EsQ0FBQ0E7SUFDMUJBLGVBQU1BLEdBQUdBLE1BQU1BLENBQUNBO0lBQ2hCQSxrQkFBU0EsR0FBR0EsU0FBU0EsQ0FBQ0E7SUFDdEJBLGNBQUtBLEdBQUdBLEtBQUtBLENBQUNBLEtBQUtBLENBQUNBO0FBQ2hDQSxDQUFDQSxFQXJDTSxRQUFRLEtBQVIsUUFBUSxRQXFDZDtBQUlELEVBQUUsQ0FBQSxDQUFDLE9BQU8sTUFBTSxJQUFJLFdBQVcsQ0FBQztJQUMvQixNQUFNLENBQUMsVUFBVSxDQUFDLEdBQUcsUUFBUSxDQUFDO0FBSC9CLGlCQUFTLFFBQVEsQ0FBQyIsImZpbGUiOiJzcmMvc3luYXB0aWMuanMiLCJzb3VyY2VzQ29udGVudCI6WyIvKlxuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgU1lOQVBUSUNcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG5cblN5bmFwdGljIGlzIGEgamF2YXNjcmlwdCBuZXVyYWwgbmV0d29yayBsaWJyYXJ5IGZvciBub2RlLmpzIGFuZCB0aGUgYnJvd3NlciwgaXRzIGdlbmVyYWxpemVkXG5hbGdvcml0aG0gaXMgYXJjaGl0ZWN0dXJlLWZyZWUsIHNvIHlvdSBjYW4gYnVpbGQgYW5kIHRyYWluIGJhc2ljYWxseSBhbnkgdHlwZSBvZiBmaXJzdCBvcmRlclxub3IgZXZlbiBzZWNvbmQgb3JkZXIgbmV1cmFsIG5ldHdvcmsgYXJjaGl0ZWN0dXJlcy5cblxuaHR0cDovL2VuLndpa2lwZWRpYS5vcmcvd2lraS9SZWN1cnJlbnRfbmV1cmFsX25ldHdvcmsjU2Vjb25kX09yZGVyX1JlY3VycmVudF9OZXVyYWxfTmV0d29ya1xuXG5UaGUgbGlicmFyeSBpbmNsdWRlcyBhIGZldyBidWlsdC1pbiBhcmNoaXRlY3R1cmVzIGxpa2UgbXVsdGlsYXllciBwZXJjZXB0cm9ucywgbXVsdGlsYXllclxubG9uZy1zaG9ydCB0ZXJtIG1lbW9yeSBuZXR3b3JrcyAoTFNUTSkgb3IgbGlxdWlkIHN0YXRlIG1hY2hpbmVzLCBhbmQgYSB0cmFpbmVyIGNhcGFibGUgb2ZcbnRyYWluaW5nIGFueSBnaXZlbiBuZXR3b3JrLCBhbmQgaW5jbHVkZXMgYnVpbHQtaW4gdHJhaW5pbmcgdGFza3MvdGVzdHMgbGlrZSBzb2x2aW5nIGFuIFhPUixcbnBhc3NpbmcgYSBEaXN0cmFjdGVkIFNlcXVlbmNlIFJlY2FsbCB0ZXN0IG9yIGFuIEVtYmVkZWQgUmViZXIgR3JhbW1hciB0ZXN0LlxuXG5UaGUgYWxnb3JpdGhtIGltcGxlbWVudGVkIGJ5IHRoaXMgbGlicmFyeSBoYXMgYmVlbiB0YWtlbiBmcm9tIERlcmVrIEQuIE1vbm5lcidzIHBhcGVyOlxuXG5BIGdlbmVyYWxpemVkIExTVE0tbGlrZSB0cmFpbmluZyBhbGdvcml0aG0gZm9yIHNlY29uZC1vcmRlciByZWN1cnJlbnQgbmV1cmFsIG5ldHdvcmtzXG5odHRwOi8vd3d3Lm92ZXJjb21wbGV0ZS5uZXQvcGFwZXJzL25uMjAxMi5wZGZcblxuVGhlcmUgYXJlIHJlZmVyZW5jZXMgdG8gdGhlIGVxdWF0aW9ucyBpbiB0aGF0IHBhcGVyIGNvbW1lbnRlZCB0aHJvdWdoIHRoZSBzb3VyY2UgY29kZS5cblxuXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKi9cblxuXG5cbmltcG9ydCBuZXR3b3JrID0gcmVxdWlyZSgnLi9uZXR3b3JrJyk7XG5pbXBvcnQgbGF5ZXIgPSByZXF1aXJlKCcuL2xheWVyJyk7XG5pbXBvcnQgbmV1cm9uID0gcmVxdWlyZSgnLi9uZXVyb24nKTtcbmltcG9ydCB0cmFpbmVyID0gcmVxdWlyZSgnLi90cmFpbmVyJyk7XG5pbXBvcnQgYXJjaGl0ZWN0ID0gcmVxdWlyZSgnLi9hcmNoaXRlY3QnKTtcbmltcG9ydCBzcXVhc2ggPSByZXF1aXJlKCcuL3NxdWFzaCcpO1xuaW1wb3J0IHV0aWxzID0gcmVxdWlyZSgnLi91dGlscycpO1xuXG5kZWNsYXJlIHZhciB3aW5kb3c7XG5cbm1vZHVsZSBTeW5hcHRpYyB7XG5cdGV4cG9ydCBpbnRlcmZhY2UgRGljdGlvbmFyeTxUPiB7XG5cdFx0W2lkOiBzdHJpbmddIDogVDtcblx0fVxuXHRcblx0dmFyIG9sZFN5bmFwdGljID0gdHlwZW9mIHdpbmRvdyAhPSBcInVuZGVmaW5lZFwiICYmIHdpbmRvdyAmJiB3aW5kb3dbJ1N5bmFwdGljJ107XG5cdFxuXHRleHBvcnQgZnVuY3Rpb24gbmluamEoKSB7XG4gICAgICB3aW5kb3dbJ3N5bmFwdGljJ10gPSBvbGRTeW5hcHRpYzsgXG4gICAgICByZXR1cm4gU3luYXB0aWM7XG5cdH1cblx0XG5cdGV4cG9ydCBpbnRlcmZhY2UgSUNvbXBpbGVkUGFyYW1ldGVycyB7XHRcblx0XHRtZW1vcnk/OiBhbnk7XG5cdFx0bmV1cm9ucz86IG51bWJlcjtcblx0XHRpbnB1dHM/OiBhbnlbXTtcblx0XHRvdXRwdXRzPzogYW55W107XG5cdFx0dGFyZ2V0cz86IGFueVtdO1xuXHRcdHZhcmlhYmxlcz86IGFueTtcblx0XHRhY3RpdmF0aW9uX3NlbnRlbmNlcz86IGFueVtdO1xuXHRcdHRyYWNlX3NlbnRlbmNlcz86IGFueVtdO1xuXHRcdHByb3BhZ2F0aW9uX3NlbnRlbmNlcz86IGFueVtdO1xuXHRcdGxheWVycz86IGFueTtcblx0fVxuXHRcblx0ZXhwb3J0IGludGVyZmFjZSBJTnVtZXJpY0FycmF5IHtcblx0ICBbaW5kZXg6IG51bWJlcl0gOiBudW1iZXI7XG5cdCAgbGVuZ3RoIDogbnVtYmVyO1xuXHR9XG5cdFxuXHRleHBvcnQgdmFyIE5ldXJvbiA9IG5ldXJvbi5OZXVyb247XG5cdGV4cG9ydCB2YXIgTGF5ZXIgPSBsYXllci5MYXllcjtcblx0ZXhwb3J0IHZhciBOZXR3b3JrID0gbmV0d29yay5OZXR3b3JrO1xuXHRleHBvcnQgdmFyIFRyYWluZXIgPSB0cmFpbmVyLlRyYWluZXI7XG5cdGV4cG9ydCB2YXIgU3F1YXNoID0gc3F1YXNoO1xuXHRleHBvcnQgdmFyIEFyY2hpdGVjdCA9IGFyY2hpdGVjdDtcblx0ZXhwb3J0IHZhciBVdGlscyA9IHV0aWxzLlV0aWxzO1xufVxuXG5leHBvcnQgPSBTeW5hcHRpYztcblxuaWYodHlwZW9mIHdpbmRvdyAhPSBcInVuZGVmaW5lZFwiKSBcblx0d2luZG93WydzeW5hcHRpYyddID0gU3luYXB0aWM7XG4iXSwic291cmNlUm9vdCI6Ii9zb3VyY2UvIn0=