var WorkerNetworkWorker = require('worker?inline!./worker-index.worker.js');

module.exports = class WorkerNetworkProxyAPI {
  constructor() {
    var worker = this.worker = new WorkerNetworkWorker();
    this.callbacks = {};
    this.nextCallbackId = 1;

    worker.onmessage = ({data: {response: [error, data] = [], callbackId} = {}}) =>
      this.callbacks[callbackId](error, data);
  }

  subscribe(method, args, callback) {
    const callbackId = this.nextCallbackId++;
    this.callbacks[callbackId] = callback;
    this.worker.postMessage({method, args, callbackId});
    return callbackId;
  }

  unsubscribe(callbackId) {
    return delete this.callbacks[callbackId];
  }

  callMethod(method, args) {
    return new Promise((resolve, reject) => {
      const callbackId = this.subscribe(method, args, (err, data) => {
        this.unsubscribe(callbackId);
        err ? reject(err) : resolve(data);
      });
    })
  };
};
