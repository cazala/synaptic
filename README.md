Synaptic [![Build Status](https://travis-ci.org/cazala/synaptic.svg?branch=master)](https://travis-ci.org/cazala/synaptic) [![Join the chat at https://synapticjs.slack.com](https://synaptic-slack-ugiqacqvmd.now.sh/badge.svg)](https://synaptic-slack-ugiqacqvmd.now.sh/)
========

## Important: [Synaptic 2.x](https://github.com/cazala/synaptic/issues/140) is in stage of discussion now! Feel free to participate

Synaptic is a javascript neural network library for **node.js** and the **browser**, its generalized algorithm is architecture-free, so you can build and train basically any type of first order or even [second order neural network](http://en.wikipedia.org/wiki/Recurrent_neural_network#Second_Order_Recurrent_Neural_Network) architectures.

This library includes a few built-in architectures like [multilayer perceptrons](http://en.wikipedia.org/wiki/Multilayer_perceptron), [multilayer long-short term memory](http://en.wikipedia.org/wiki/Long_short_term_memory) networks (LSTM), [liquid state machines](http://en.wikipedia.org/wiki/Liquid_state_machine) or [Hopfield](http://en.wikipedia.org/wiki/Hopfield_network) networks, and a trainer capable of training any given network, which includes built-in training tasks/tests like solving an XOR, completing a Distracted Sequence Recall task or an [Embedded Reber Grammar](http://www.willamette.edu/~gorr/classes/cs449/reber.html) test, so you can easily test and compare the performance of different architectures.


The algorithm implemented by this library has been taken from Derek D. Monner's paper:

[A generalized LSTM-like training algorithm for second-order recurrent neural networks](http://www.overcomplete.net/papers/nn2012.pdf)


There are references to the equations in that paper commented through the source code.

#### Introduction

If you have no prior knowledge about Neural Networks, you should start by [reading this guide](https://github.com/cazala/synaptic/wiki/Neural-Networks-101).


If you want a practical example on how to feed data to a neural network, then take a look at [this article](https://github.com/cazala/synaptic/wiki/Normalization-101).

You may also want to take a look at [this article](http://blog.webkid.io/neural-networks-in-javascript/).

#### Demos

- [Solve an XOR](http://caza.la/synaptic/#/xor)
- [Discrete Sequence Recall Task](http://caza.la/synaptic/#/dsr)
- [Learn Image Filters](http://caza.la/synaptic/#/image-filters)
- [Paint an Image](http://caza.la/synaptic/#/paint-an-image)
- [Self Organizing Map](http://caza.la/synaptic/#/self-organizing-map)
- [Read from Wikipedia](http://caza.la/synaptic/#/wikipedia)
- [Creating a Simple Neural Network (Video)](https://scrimba.com/casts/cast-1980)

The source code of these demos can be found in [this branch](https://github.com/cazala/synaptic/tree/gh-pages/scripts).

#### Getting started

- [Neurons](https://github.com/cazala/synaptic/wiki/Neurons/)
- [Layers](https://github.com/cazala/synaptic/wiki/Layers/)
- [Networks](https://github.com/cazala/synaptic/wiki/Networks/)
- [Trainer](https://github.com/cazala/synaptic/wiki/Trainer/)
- [Architect](https://github.com/cazala/synaptic/wiki/Architect/)

To try out the examples, checkout the [gh-pages](https://github.com/cazala/synaptic/tree/gh-pages) branch.

`git checkout gh-pages`


## Overview

### Installation

##### In node

You can install synaptic with [npm](http://npmjs.org):

```cmd
npm install synaptic --save
```

##### In the browser

You can install synaptic with [bower](http://bower.io):

```cmd
bower install synaptic
```

Or you can simply use the CDN link, kindly provided by [CDNjs](https://cdnjs.com/)

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/synaptic/1.0.12/synaptic.js"></script>
```

### Usage

```javascript
var synaptic = require('synaptic'); // this line is not needed in the browser
var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;
```

Now you can start to create networks, train them, or use built-in networks from the [Architect](http://github.com/cazala/synaptic#architect).

### Examples

##### Perceptron

This is how you can create a simple **perceptron**:

![perceptron](http://www.codeproject.com/KB/dotnet/predictor/network.jpg).

```javascript
function Perceptron(input, hidden, output)
{
	// create the layers
	var inputLayer = new Layer(input);
	var hiddenLayer = new Layer(hidden);
	var outputLayer = new Layer(output);

	// connect the layers
	inputLayer.project(hiddenLayer);
	hiddenLayer.project(outputLayer);

	// set the layers
	this.set({
		input: inputLayer,
		hidden: [hiddenLayer],
		output: outputLayer
	});
}

// extend the prototype chain
Perceptron.prototype = new Network();
Perceptron.prototype.constructor = Perceptron;
```

Now you can test your new network by creating a trainer and teaching the perceptron to learn an XOR

```javascript
var myPerceptron = new Perceptron(2,3,1);
var myTrainer = new Trainer(myPerceptron);

myTrainer.XOR(); // { error: 0.004998819355993572, iterations: 21871, time: 356 }

myPerceptron.activate([0,0]); // 0.0268581547421616
myPerceptron.activate([1,0]); // 0.9829673642853368
myPerceptron.activate([0,1]); // 0.9831714267395621
myPerceptron.activate([1,1]); // 0.02128894618097928
```

##### Long Short-Term Memory

This is how you can create a simple **long short-term memory** network with input gate, forget gate, output gate, and peephole connections:

![long short-term memory](http://people.idsia.ch/~juergen/lstmcell4.jpg)

```javascript
function LSTM(input, blocks, output)
{
	// create the layers
	var inputLayer = new Layer(input);
	var inputGate = new Layer(blocks);
	var forgetGate = new Layer(blocks);
	var memoryCell = new Layer(blocks);
	var outputGate = new Layer(blocks);
	var outputLayer = new Layer(output);

	// connections from input layer
	var input = inputLayer.project(memoryCell);
	inputLayer.project(inputGate);
	inputLayer.project(forgetGate);
	inputLayer.project(outputGate);

	// connections from memory cell
	var output = memoryCell.project(outputLayer);

	// self-connection
	var self = memoryCell.project(memoryCell);

	// peepholes
	memoryCell.project(inputGate);
	memoryCell.project(forgetGate);
	memoryCell.project(outputGate);

	// gates
	inputGate.gate(input, Layer.gateType.INPUT);
	forgetGate.gate(self, Layer.gateType.ONE_TO_ONE);
	outputGate.gate(output, Layer.gateType.OUTPUT);

	// input to output direct connection
	inputLayer.project(outputLayer);

	// set the layers of the neural network
	this.set({
		input: inputLayer,
		hidden: [inputGate, forgetGate, memoryCell, outputGate],
		output: outputLayer
	});
}

// extend the prototype chain
LSTM.prototype = new Network();
LSTM.prototype.constructor = LSTM;
```

These are examples for explanatory purposes, the [Architect](https://github.com/cazala/synaptic/wiki/Architect/) already includes Multilayer Perceptrons and
Multilayer LSTM network architectures.

## Contribute

**Synaptic** is an Open Source project that started in Buenos Aires, Argentina. Anybody in the world is welcome to contribute to the development of the project.

If you want to contribute feel free to send PR's, just make sure to run **npm run test** and **npm run build** before submitting it. This way you'll run all the test specs and build the web distribution files.

<3
