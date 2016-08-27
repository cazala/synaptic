var WorkerNetworkWorker = require('./network.worker.js');
var Network = require('./network');

class WorkerNetworkAPI {
  constructor() {
    var worker = this.worker = new WorkerNetwork();
    this.callbacks = {};
    this.nextCallbackId = 0;

    worker.addEventListener("message", function ({data}) {
      if (data && data.callbackId) {
        if (worker.callbacks[data.callbackId] instanceof Function) {
          worker.callbacks[data.callbackId](data);
        }
        if (!worker.callbacks[data.callbackId].isMultiCallback) {
          delete worker.callbacks[data.callbackId];
        }
      }
    });
  }

  call(method, args, callback, isMultiCallback = false) {
    var callbackId = worker.nextCallbackId++;
    message.callbackId = callbackId;
    worker.callbacks[callbackId] = callback;
    worker.callbacks[callbackId].isMultiCallback = isMultiCallback;

    worker.postMessage({method, args});
  };
}

module.exports = class WorkerNetwork extends Network {
  constructor(...args) {
    super(...args);
    this._worker = new WorkerNetworkWorker();

  }
};


var inherits = require('../util').inherits;
var BaseNetwork = require('../base-network');

module.exports = WorkerNetwork;
inherits(WorkerNetwork, BaseNetwork);

function WorkerNetwork() {

}

WorkerNetwork.prototype;