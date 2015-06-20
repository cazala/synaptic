import network = require('../network');
import trainer = require('../trainer');
export declare class Liquid extends network.Network {
    trainer: trainer.Trainer;
    constructor(inputs: any, hidden: any, outputs: any, connections: any, gates: any);
}
