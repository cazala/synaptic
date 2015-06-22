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
})(Synaptic || (Synaptic = {}));
if (typeof window != "undefined")
    window['synaptic'] = Synaptic;
module.exports = Synaptic;

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9zeW5hcHRpYy50cyJdLCJuYW1lcyI6WyJTeW5hcHRpYyIsIlN5bmFwdGljLm5pbmphIl0sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OzZGQXdCNkY7QUFJN0YsSUFBTyxPQUFPLFdBQVcsV0FBVyxDQUFDLENBQUM7QUFDdEMsSUFBTyxLQUFLLFdBQVcsU0FBUyxDQUFDLENBQUM7QUFDbEMsSUFBTyxNQUFNLFdBQVcsVUFBVSxDQUFDLENBQUM7QUFDcEMsSUFBTyxPQUFPLFdBQVcsV0FBVyxDQUFDLENBQUM7QUFDdEMsSUFBTyxTQUFTLFdBQVcsYUFBYSxDQUFDLENBQUM7QUFDMUMsSUFBTyxNQUFNLFdBQVcsVUFBVSxDQUFDLENBQUM7QUFJcEMsSUFBTyxRQUFRLENBb0NkO0FBcENELFdBQU8sUUFBUSxFQUFDLENBQUM7SUFLaEJBLElBQUlBLFdBQVdBLEdBQUdBLE9BQU9BLE1BQU1BLElBQUlBLFdBQVdBLElBQUlBLE1BQU1BLElBQUlBLE1BQU1BLENBQUNBLFVBQVVBLENBQUNBLENBQUNBO0lBRS9FQSxTQUFnQkEsS0FBS0E7UUFDaEJDLE1BQU1BLENBQUNBLFVBQVVBLENBQUNBLEdBQUdBLFdBQVdBLENBQUNBO1FBQ2pDQSxNQUFNQSxDQUFDQSxRQUFRQSxDQUFDQTtJQUNyQkEsQ0FBQ0E7SUFIZUQsY0FBS0EsR0FBTEEsS0FHZkEsQ0FBQUE7SUFvQlVBLGVBQU1BLEdBQUdBLE1BQU1BLENBQUNBLE1BQU1BLENBQUNBO0lBQ3ZCQSxjQUFLQSxHQUFHQSxLQUFLQSxDQUFDQSxLQUFLQSxDQUFDQTtJQUNwQkEsZ0JBQU9BLEdBQUdBLE9BQU9BLENBQUNBLE9BQU9BLENBQUNBO0lBQzFCQSxnQkFBT0EsR0FBR0EsT0FBT0EsQ0FBQ0EsT0FBT0EsQ0FBQ0E7SUFDMUJBLGVBQU1BLEdBQUdBLE1BQU1BLENBQUNBO0lBQ2hCQSxrQkFBU0EsR0FBR0EsU0FBU0EsQ0FBQ0E7QUFDbENBLENBQUNBLEVBcENNLFFBQVEsS0FBUixRQUFRLFFBb0NkO0FBSUQsRUFBRSxDQUFBLENBQUMsT0FBTyxNQUFNLElBQUksV0FBVyxDQUFDO0lBQy9CLE1BQU0sQ0FBQyxVQUFVLENBQUMsR0FBRyxRQUFRLENBQUM7QUFIL0IsaUJBQVMsUUFBUSxDQUFDIiwiZmlsZSI6InNyYy9zeW5hcHRpYy5qcyIsInNvdXJjZXNDb250ZW50IjpbIi8qXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBTWU5BUFRJQ1xuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcblxuU3luYXB0aWMgaXMgYSBqYXZhc2NyaXB0IG5ldXJhbCBuZXR3b3JrIGxpYnJhcnkgZm9yIG5vZGUuanMgYW5kIHRoZSBicm93c2VyLCBpdHMgZ2VuZXJhbGl6ZWRcbmFsZ29yaXRobSBpcyBhcmNoaXRlY3R1cmUtZnJlZSwgc28geW91IGNhbiBidWlsZCBhbmQgdHJhaW4gYmFzaWNhbGx5IGFueSB0eXBlIG9mIGZpcnN0IG9yZGVyXG5vciBldmVuIHNlY29uZCBvcmRlciBuZXVyYWwgbmV0d29yayBhcmNoaXRlY3R1cmVzLlxuXG5odHRwOi8vZW4ud2lraXBlZGlhLm9yZy93aWtpL1JlY3VycmVudF9uZXVyYWxfbmV0d29yayNTZWNvbmRfT3JkZXJfUmVjdXJyZW50X05ldXJhbF9OZXR3b3JrXG5cblRoZSBsaWJyYXJ5IGluY2x1ZGVzIGEgZmV3IGJ1aWx0LWluIGFyY2hpdGVjdHVyZXMgbGlrZSBtdWx0aWxheWVyIHBlcmNlcHRyb25zLCBtdWx0aWxheWVyXG5sb25nLXNob3J0IHRlcm0gbWVtb3J5IG5ldHdvcmtzIChMU1RNKSBvciBsaXF1aWQgc3RhdGUgbWFjaGluZXMsIGFuZCBhIHRyYWluZXIgY2FwYWJsZSBvZlxudHJhaW5pbmcgYW55IGdpdmVuIG5ldHdvcmssIGFuZCBpbmNsdWRlcyBidWlsdC1pbiB0cmFpbmluZyB0YXNrcy90ZXN0cyBsaWtlIHNvbHZpbmcgYW4gWE9SLFxucGFzc2luZyBhIERpc3RyYWN0ZWQgU2VxdWVuY2UgUmVjYWxsIHRlc3Qgb3IgYW4gRW1iZWRlZCBSZWJlciBHcmFtbWFyIHRlc3QuXG5cblRoZSBhbGdvcml0aG0gaW1wbGVtZW50ZWQgYnkgdGhpcyBsaWJyYXJ5IGhhcyBiZWVuIHRha2VuIGZyb20gRGVyZWsgRC4gTW9ubmVyJ3MgcGFwZXI6XG5cbkEgZ2VuZXJhbGl6ZWQgTFNUTS1saWtlIHRyYWluaW5nIGFsZ29yaXRobSBmb3Igc2Vjb25kLW9yZGVyIHJlY3VycmVudCBuZXVyYWwgbmV0d29ya3Ncbmh0dHA6Ly93d3cub3ZlcmNvbXBsZXRlLm5ldC9wYXBlcnMvbm4yMDEyLnBkZlxuXG5UaGVyZSBhcmUgcmVmZXJlbmNlcyB0byB0aGUgZXF1YXRpb25zIGluIHRoYXQgcGFwZXIgY29tbWVudGVkIHRocm91Z2ggdGhlIHNvdXJjZSBjb2RlLlxuXG5cbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuXG5cblxuaW1wb3J0IG5ldHdvcmsgPSByZXF1aXJlKCcuL25ldHdvcmsnKTtcbmltcG9ydCBsYXllciA9IHJlcXVpcmUoJy4vbGF5ZXInKTtcbmltcG9ydCBuZXVyb24gPSByZXF1aXJlKCcuL25ldXJvbicpO1xuaW1wb3J0IHRyYWluZXIgPSByZXF1aXJlKCcuL3RyYWluZXInKTtcbmltcG9ydCBhcmNoaXRlY3QgPSByZXF1aXJlKCcuL2FyY2hpdGVjdCcpO1xuaW1wb3J0IHNxdWFzaCA9IHJlcXVpcmUoJy4vc3F1YXNoJyk7XG5cbmRlY2xhcmUgdmFyIHdpbmRvdztcblxubW9kdWxlIFN5bmFwdGljIHtcblx0ZXhwb3J0IGludGVyZmFjZSBEaWN0aW9uYXJ5PFQ+IHtcblx0XHRbaWQ6IHN0cmluZ10gOiBUO1xuXHR9XG5cdFxuXHR2YXIgb2xkU3luYXB0aWMgPSB0eXBlb2Ygd2luZG93ICE9IFwidW5kZWZpbmVkXCIgJiYgd2luZG93ICYmIHdpbmRvd1snU3luYXB0aWMnXTtcblx0XG5cdGV4cG9ydCBmdW5jdGlvbiBuaW5qYSgpIHtcbiAgICAgIHdpbmRvd1snc3luYXB0aWMnXSA9IG9sZFN5bmFwdGljOyBcbiAgICAgIHJldHVybiBTeW5hcHRpYztcblx0fVxuXHRcblx0ZXhwb3J0IGludGVyZmFjZSBJQ29tcGlsZWRQYXJhbWV0ZXJzIHtcdFxuXHRcdG1lbW9yeT86IGFueTtcblx0XHRuZXVyb25zPzogbnVtYmVyO1xuXHRcdGlucHV0cz86IGFueVtdO1xuXHRcdG91dHB1dHM/OiBhbnlbXTtcblx0XHR0YXJnZXRzPzogYW55W107XG5cdFx0dmFyaWFibGVzPzogYW55O1xuXHRcdGFjdGl2YXRpb25fc2VudGVuY2VzPzogYW55W107XG5cdFx0dHJhY2Vfc2VudGVuY2VzPzogYW55W107XG5cdFx0cHJvcGFnYXRpb25fc2VudGVuY2VzPzogYW55W107XG5cdFx0bGF5ZXJzPzogYW55O1xuXHR9XG5cdFxuXHRleHBvcnQgaW50ZXJmYWNlIElOdW1lcmljQXJyYXkge1xuXHQgIFtpbmRleDogbnVtYmVyXSA6IG51bWJlcjtcblx0ICBsZW5ndGggOiBudW1iZXI7XG5cdH1cblx0XG5cdGV4cG9ydCB2YXIgTmV1cm9uID0gbmV1cm9uLk5ldXJvbjtcblx0ZXhwb3J0IHZhciBMYXllciA9IGxheWVyLkxheWVyO1xuXHRleHBvcnQgdmFyIE5ldHdvcmsgPSBuZXR3b3JrLk5ldHdvcms7XG5cdGV4cG9ydCB2YXIgVHJhaW5lciA9IHRyYWluZXIuVHJhaW5lcjtcblx0ZXhwb3J0IHZhciBTcXVhc2ggPSBzcXVhc2g7XG5cdGV4cG9ydCB2YXIgQXJjaGl0ZWN0ID0gYXJjaGl0ZWN0O1xufVxuXG5leHBvcnQgPSBTeW5hcHRpYztcblxuaWYodHlwZW9mIHdpbmRvdyAhPSBcInVuZGVmaW5lZFwiKSBcblx0d2luZG93WydzeW5hcHRpYyddID0gU3luYXB0aWM7XG4iXSwic291cmNlUm9vdCI6Ii9zb3VyY2UvIn0=