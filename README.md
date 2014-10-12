Synapse
=======

Synapse.js is a javascript/node.js neural network library, its generalized algorythm is architecture-free, so you can build and train basically any type of first order or even [second order neural network](http://en.wikipedia.org/wiki/Recurrent_neural_network#Second_Order_Recurrent_Neural_Network) architectures.

This library includes a few built-in architectures like [multilayer perceptrons](http://en.wikipedia.org/wiki/Multilayer_perceptron), [multilayer long-short term memory](http://en.wikipedia.org/wiki/Long_short_term_memory) networks (LSTM) or [liquid state machines](http://en.wikipedia.org/wiki/Liquid_state_machine), and a trainer capable of training any given network, which includes built-in training tasks/tests like solving an XOR, completing a Distracted Sequence Recall task or an [Embeded Reber Grammar](http://www.willamette.edu/~gorr/classes/cs449/reber.html) test, so you can easily test and compare the performance of different architectures.


The algorythm implemented by this library has been taken from Derek D. Monner's paper:

[A generalized LSTM-like training algorithm for second-order recurrent neural networks](http://www.overcomplete.net/papers/nn2012.pdf)


There are references to the equations in that paper commented through the source code.

##Overview

###Usage

######In node
You can download synapse with [npm](http://npmjs.org):
`npm install synapse`

######In the browser
Just include the file synapse.js (you can find it in the /lib directory) with a script tag in your HTML:
`<script src="synapse.js"></script>`
