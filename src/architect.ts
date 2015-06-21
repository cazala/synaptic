import hopfield = require('./architect/Hopfield');
import lstm = require('./architect/LSTM');
import lsm = require('./architect/Liquid');
import perceptron = require('./architect/Perceptron');

export var LSTM = lstm.LSTM;
export var Liquid = lsm.Liquid;
export var Hopfield = hopfield.Hopfield;
export var Perceptron = perceptron.Perceptron;
