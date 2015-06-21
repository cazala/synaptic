import net = require('./network');
/*******************************************************************************************
                                        TRAINER
*******************************************************************************************/
export declare class Trainer {
    network: net.Network;
    rate: any;
    iterations: number;
    error: number;
    cost: Trainer.ITrainerCostFn;
    schedule: any;
    constructor(network: net.Network, options?: any);
    train(set: any, options: any): {
        error: number;
        iterations: number;
        time: number;
    };
    workerTrain(set: any, callback: any, options: any): void;
    XOR(options: any): {
        error: number;
        iterations: number;
        time: number;
    };
    DSR(options: any): {
        iterations: number;
        success: number;
        error: number;
        time: number;
    };
    ERG(options: any): {
        iterations: number;
        error: number;
        time: number;
        test: (str: any) => boolean;
        generate: () => string;
    };
}
export declare module Trainer {
    interface ITrainerCostFn {
        (target: any, output: any): number;
    }
    var cost: {
        CROSS_ENTROPY: (target: any, output: any) => number;
        MSE: (target: any, output: any) => number;
    };
}
