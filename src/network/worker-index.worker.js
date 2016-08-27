const WorkerNetwork = require('./worker-network');

const Class = WorkerNetwork;
let singleton;

onmessage = function(e) {
  const {method, args, callbackId} = e.data;
  var response = callMethod(method, args);
  postMessage({response, callbackId})
};

function callMethod(method, args) {
  var target = singleton ? singleton : Class;
  try {
    var result = Class[method](...args);
    if (result instanceof Class) {
      singleton = result;
      return undefined;
    }
    return [null, result];
  } catch (e) {
    console.error(e);
    return [e.message]
  }
}

