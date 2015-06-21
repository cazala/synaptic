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

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy9zeW5hcHRpYy50cyJdLCJuYW1lcyI6WyJTeW5hcHRpYyIsIlN5bmFwdGljLm5pbmphIl0sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OzZGQXdCNkY7QUFJN0YsSUFBTyxPQUFPLFdBQVcsV0FBVyxDQUFDLENBQUM7QUFDdEMsSUFBTyxLQUFLLFdBQVcsU0FBUyxDQUFDLENBQUM7QUFDbEMsSUFBTyxNQUFNLFdBQVcsVUFBVSxDQUFDLENBQUM7QUFDcEMsSUFBTyxPQUFPLFdBQVcsV0FBVyxDQUFDLENBQUM7QUFDdEMsSUFBTyxTQUFTLFdBQVcsYUFBYSxDQUFDLENBQUM7QUFDMUMsSUFBTyxNQUFNLFdBQVcsVUFBVSxDQUFDLENBQUM7QUFJcEMsSUFBTyxRQUFRLENBK0JkO0FBL0JELFdBQU8sUUFBUSxFQUFDLENBQUM7SUFLaEJBLElBQUlBLFdBQVdBLEdBQUdBLE9BQU9BLE1BQU1BLElBQUlBLFdBQVdBLElBQUlBLE1BQU1BLElBQUlBLE1BQU1BLENBQUNBLFVBQVVBLENBQUNBLENBQUNBO0lBRS9FQSxTQUFnQkEsS0FBS0E7UUFDaEJDLE1BQU1BLENBQUNBLFVBQVVBLENBQUNBLEdBQUdBLFdBQVdBLENBQUNBO1FBQ2pDQSxNQUFNQSxDQUFDQSxRQUFRQSxDQUFDQTtJQUNyQkEsQ0FBQ0E7SUFIZUQsY0FBS0EsR0FBTEEsS0FHZkEsQ0FBQUE7SUFlVUEsZUFBTUEsR0FBR0EsTUFBTUEsQ0FBQ0EsTUFBTUEsQ0FBQ0E7SUFDdkJBLGNBQUtBLEdBQUdBLEtBQUtBLENBQUNBLEtBQUtBLENBQUNBO0lBQ3BCQSxnQkFBT0EsR0FBR0EsT0FBT0EsQ0FBQ0EsT0FBT0EsQ0FBQ0E7SUFDMUJBLGdCQUFPQSxHQUFHQSxPQUFPQSxDQUFDQSxPQUFPQSxDQUFDQTtJQUMxQkEsZUFBTUEsR0FBR0EsTUFBTUEsQ0FBQ0E7SUFDaEJBLGtCQUFTQSxHQUFHQSxTQUFTQSxDQUFDQTtBQUNsQ0EsQ0FBQ0EsRUEvQk0sUUFBUSxLQUFSLFFBQVEsUUErQmQ7QUFJRCxFQUFFLENBQUEsQ0FBQyxPQUFPLE1BQU0sSUFBSSxXQUFXLENBQUM7SUFDL0IsTUFBTSxDQUFDLFVBQVUsQ0FBQyxHQUFHLFFBQVEsQ0FBQztBQUgvQixpQkFBUyxRQUFRLENBQUMiLCJmaWxlIjoic3JjL3N5bmFwdGljLmpzIiwic291cmNlc0NvbnRlbnQiOlsiLypcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFNZTkFQVElDXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxuXG5TeW5hcHRpYyBpcyBhIGphdmFzY3JpcHQgbmV1cmFsIG5ldHdvcmsgbGlicmFyeSBmb3Igbm9kZS5qcyBhbmQgdGhlIGJyb3dzZXIsIGl0cyBnZW5lcmFsaXplZFxuYWxnb3JpdGhtIGlzIGFyY2hpdGVjdHVyZS1mcmVlLCBzbyB5b3UgY2FuIGJ1aWxkIGFuZCB0cmFpbiBiYXNpY2FsbHkgYW55IHR5cGUgb2YgZmlyc3Qgb3JkZXJcbm9yIGV2ZW4gc2Vjb25kIG9yZGVyIG5ldXJhbCBuZXR3b3JrIGFyY2hpdGVjdHVyZXMuXG5cbmh0dHA6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvUmVjdXJyZW50X25ldXJhbF9uZXR3b3JrI1NlY29uZF9PcmRlcl9SZWN1cnJlbnRfTmV1cmFsX05ldHdvcmtcblxuVGhlIGxpYnJhcnkgaW5jbHVkZXMgYSBmZXcgYnVpbHQtaW4gYXJjaGl0ZWN0dXJlcyBsaWtlIG11bHRpbGF5ZXIgcGVyY2VwdHJvbnMsIG11bHRpbGF5ZXJcbmxvbmctc2hvcnQgdGVybSBtZW1vcnkgbmV0d29ya3MgKExTVE0pIG9yIGxpcXVpZCBzdGF0ZSBtYWNoaW5lcywgYW5kIGEgdHJhaW5lciBjYXBhYmxlIG9mXG50cmFpbmluZyBhbnkgZ2l2ZW4gbmV0d29yaywgYW5kIGluY2x1ZGVzIGJ1aWx0LWluIHRyYWluaW5nIHRhc2tzL3Rlc3RzIGxpa2Ugc29sdmluZyBhbiBYT1IsXG5wYXNzaW5nIGEgRGlzdHJhY3RlZCBTZXF1ZW5jZSBSZWNhbGwgdGVzdCBvciBhbiBFbWJlZGVkIFJlYmVyIEdyYW1tYXIgdGVzdC5cblxuVGhlIGFsZ29yaXRobSBpbXBsZW1lbnRlZCBieSB0aGlzIGxpYnJhcnkgaGFzIGJlZW4gdGFrZW4gZnJvbSBEZXJlayBELiBNb25uZXIncyBwYXBlcjpcblxuQSBnZW5lcmFsaXplZCBMU1RNLWxpa2UgdHJhaW5pbmcgYWxnb3JpdGhtIGZvciBzZWNvbmQtb3JkZXIgcmVjdXJyZW50IG5ldXJhbCBuZXR3b3Jrc1xuaHR0cDovL3d3dy5vdmVyY29tcGxldGUubmV0L3BhcGVycy9ubjIwMTIucGRmXG5cblRoZXJlIGFyZSByZWZlcmVuY2VzIHRvIHRoZSBlcXVhdGlvbnMgaW4gdGhhdCBwYXBlciBjb21tZW50ZWQgdGhyb3VnaCB0aGUgc291cmNlIGNvZGUuXG5cblxuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cblxuXG5pbXBvcnQgbmV0d29yayA9IHJlcXVpcmUoJy4vbmV0d29yaycpO1xuaW1wb3J0IGxheWVyID0gcmVxdWlyZSgnLi9sYXllcicpO1xuaW1wb3J0IG5ldXJvbiA9IHJlcXVpcmUoJy4vbmV1cm9uJyk7XG5pbXBvcnQgdHJhaW5lciA9IHJlcXVpcmUoJy4vdHJhaW5lcicpO1xuaW1wb3J0IGFyY2hpdGVjdCA9IHJlcXVpcmUoJy4vYXJjaGl0ZWN0Jyk7XG5pbXBvcnQgc3F1YXNoID0gcmVxdWlyZSgnLi9zcXVhc2gnKTtcblxuZGVjbGFyZSB2YXIgd2luZG93O1xuXG5tb2R1bGUgU3luYXB0aWMge1xuXHRleHBvcnQgaW50ZXJmYWNlIERpY3Rpb25hcnk8VD4ge1xuXHRcdFtpZDogc3RyaW5nXSA6IFQ7XG5cdH1cblx0XG5cdHZhciBvbGRTeW5hcHRpYyA9IHR5cGVvZiB3aW5kb3cgIT0gXCJ1bmRlZmluZWRcIiAmJiB3aW5kb3cgJiYgd2luZG93WydTeW5hcHRpYyddO1xuXHRcblx0ZXhwb3J0IGZ1bmN0aW9uIG5pbmphKCkge1xuICAgICAgd2luZG93WydzeW5hcHRpYyddID0gb2xkU3luYXB0aWM7IFxuICAgICAgcmV0dXJuIFN5bmFwdGljO1xuXHR9XG5cdFxuXHRleHBvcnQgaW50ZXJmYWNlIElDb21waWxlZFBhcmFtZXRlcnMge1x0XG5cdFx0bWVtb3J5PzogYW55O1xuXHRcdG5ldXJvbnM/OiBudW1iZXI7XG5cdFx0aW5wdXRzPzogYW55W107XG5cdFx0b3V0cHV0cz86IGFueVtdO1xuXHRcdHRhcmdldHM/OiBhbnlbXTtcblx0XHR2YXJpYWJsZXM/OiBhbnk7XG5cdFx0YWN0aXZhdGlvbl9zZW50ZW5jZXM/OiBhbnlbXTtcblx0XHR0cmFjZV9zZW50ZW5jZXM/OiBhbnlbXTtcblx0XHRwcm9wYWdhdGlvbl9zZW50ZW5jZXM/OiBhbnlbXTtcblx0XHRsYXllcnM/OiBhbnk7XG5cdH1cblx0XG5cdGV4cG9ydCB2YXIgTmV1cm9uID0gbmV1cm9uLk5ldXJvbjtcblx0ZXhwb3J0IHZhciBMYXllciA9IGxheWVyLkxheWVyO1xuXHRleHBvcnQgdmFyIE5ldHdvcmsgPSBuZXR3b3JrLk5ldHdvcms7XG5cdGV4cG9ydCB2YXIgVHJhaW5lciA9IHRyYWluZXIuVHJhaW5lcjtcblx0ZXhwb3J0IHZhciBTcXVhc2ggPSBzcXVhc2g7XG5cdGV4cG9ydCB2YXIgQXJjaGl0ZWN0ID0gYXJjaGl0ZWN0O1xufVxuXG5leHBvcnQgPSBTeW5hcHRpYztcblxuaWYodHlwZW9mIHdpbmRvdyAhPSBcInVuZGVmaW5lZFwiKSBcblx0d2luZG93WydzeW5hcHRpYyddID0gU3luYXB0aWM7XG4iXSwic291cmNlUm9vdCI6Ii9zb3VyY2UvIn0=