import hopfield = require('./architect/Hopfield');
import lstm = require('./architect/LSTM');
import lsm = require('./architect/Liquid');
import perceptron = require('./architect/Perceptron');
import mb = require('./architect/NTM');

export var LSTM = lstm.LSTM;
export var Liquid = lsm.Liquid;
export var Hopfield = hopfield.Hopfield;
export var Perceptron = perceptron.Perceptron;
export var NTM = mb.NTM;