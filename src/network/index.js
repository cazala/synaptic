const Network = require('./network');

class SynapticNetwork extends Network {
}

module.exports = SynapticNetwork;

// features enabling
if (global.Worker) {
  // this code should be kept on
  const WorkerProxyNetwork = require('./worker-proxy-network');
  // Return a HTML5 WebWorker specialized on training the network stored in `memory`.
  // Train based on the given dataSet and options.
  // The worker returns the updated `memory` when done.
  SynapticNetwork.prototype.worker = function () {
    return WorkerProxyNetwork.fromJSON(this.toJSON());
  }
}
