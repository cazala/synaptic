var WorkerNetwork = require('./network');
var constants = require('./worker-communication-constants');

onmessage = function (e) {
  const {method, args, callbackId} = e.data;
  try {
    var response = callMethod(method, args);
    postMessage({response: [null, response], callbackId})
  } catch (e) {
    postMessage({response: [e.message], callbackId})
  }
};

const callMethod = (function() {
  var instance;

  return function callMethod(method, args) {
    var result;
    switch (method) {
      case constants.GET:
        result = instance[args[0]];
        break;
      case constants.SET:
        instance[method][args[0]] = args[1];
        break;
      case constants.FROM_JSON:
        instance = WorkerNetwork.fromJSON(args[0]);
        break;
      default:
        result = instance[method](...args);
        break;
    }


    return result;
  }
})();

