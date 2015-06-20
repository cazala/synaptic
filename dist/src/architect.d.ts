import hopfield = require('./architect/hopfield');
import lstm = require('./architect/LSTM');
import lsm = require('./architect/Liquid');
import perceptron = require('./architect/Perceptron');
export declare var LSTM: typeof lstm.LSTM;
export declare var Liquid: typeof lsm.Liquid;
export declare var Hopfield: typeof hopfield.Hopfield;
export declare var Perceptron: typeof perceptron.Perceptron;
