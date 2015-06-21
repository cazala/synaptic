import network = require('../network');
import trainer = require('../trainer');
export declare class LSTM extends network.Network {
    trainer: trainer.Trainer;
    constructor(...args: any[]);
}
