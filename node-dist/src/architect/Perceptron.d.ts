import network = require('../network');
import trainer = require('../trainer');
export declare class Perceptron extends network.Network {
    trainer: trainer.Trainer;
    constructor(...args: number[]);
}
