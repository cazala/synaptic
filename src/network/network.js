const Network = require('./network');
const WorkerProxyNetwork = require('./worker-proxy-network');

class BrowserNetwork extends Network {

  // Return a HTML5 WebWorker specialized on training the network stored in `memory`.
  // Train based on the given dataSet and options.
  // The worker returns the updated `memory` when done.
  worker() {
    return WorkerProxyNetwork.fromJSON(this.toJSON());
  }
}

module.exports = BrowserNetwork;