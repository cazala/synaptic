var Network = require('./network');
var constants = require('./worker-communication-constants');
var WorkerNetworkProxyAPI = require('./worker-network-proxy-api');

const workerNetworkConstructorInternalKey = {};

const workerNetworkProxyKey = '_workerNetworkProxyAPI';
const getGetterName = key => `get${key[0].toUpperCase()}${key.slice(1)}`;
const getSetterName = key => `set${key[0].toUpperCase()}${key.slice(1)}`;

class WorkerProxyNetwork extends Network {
  constructor(key, workerNetworkProxyAPI) {
    if (key !== workerNetworkConstructorInternalKey)
      throw new Error(
        `cannot initiate ${this.name} directly as it is constructed in an async manner. ` +
        `Instead of that use ${this.name}.fromJSON() or ${this.name}.construct`);

    super();
    this[workerNetworkProxyKey] = workerNetworkProxyAPI;
  }
}

module.exports = WorkerProxyNetwork;

WorkerProxyNetwork.construct = function (proxy) { return new this(workerNetworkConstructorInternalKey, proxy) };

WorkerProxyNetwork.fromJSON = function (jsonAlikeObject, proxyAPI = new WorkerNetworkProxyAPI()) {
  return proxyAPI.callMethod(constants.FROM_JSON, [jsonAlikeObject])
    .then(() => this.construct(proxyAPI))
};


const prototypeKeys = Object.getOwnPropertyNames(Network.prototype);

const methodKeys = prototypeKeys.filter(key => (Network.prototype[key] instanceof Function));
const propertyKeys = prototypeKeys.filter(key => !(Network.prototype[key] instanceof Function));

const createDef = (key, value, length) => {
  Object.defineProperty(value, 'name', {value: key});
  if (length !== undefined)
    Object.defineProperty(value, 'length', {value: length});
  return {key, descriptor: {value}};
};

const definitions = [
  ...methodKeys
    .map(key =>
      createDef(key,
        function (...args) { return this[workerNetworkProxyKey].callMethod(key, args) }, Network.prototype[key].length)),
  ...propertyKeys
    .map(key => ({
      key,
      descriptor: {
        get() {
          throw new Error(`cannot get property ${this.constructor.name}#${key} directly as actual data reflected is stored in a worker. `
            + `To get it use method ${this.constructor.name}#${getGetterName(key)} with a signature () => Promise<value>`)
        },
        set() {
          throw new Error(`cannot set property ${this.constructor.name}#${key} directly as actual data reflected is stored in a worker. `
            + `To set it use method ${this.constructor.name}#${getSetterName(key)} with a signature (value) => Promise<>`)
        }
      }
    })),
  ...propertyKeys
    .map(key => createDef(getGetterName(key),
      function () { return this[workerNetworkProxyKey].callMethod(constants.GET, [key]) })),
  ...propertyKeys
    .map(key => createDef(getSetterName(key),
      function (value) { return this[workerNetworkProxyKey].callMethod(constants.SET, [key, value]) }))
];

for (const definition of definitions)
  Object.defineProperty(WorkerProxyNetwork.prototype, definition.key, definition.descriptor);