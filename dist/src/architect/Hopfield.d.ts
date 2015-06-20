import network = require('../network');
import trainer = require('../trainer');
export declare class Hopfield extends network.Network {
    trainer: trainer.Trainer;
    constructor(size: number);
    learn(patterns: any): {
        error: number;
        iterations: number;
        time: number;
    };
    feed(pattern: any): any[];
}
