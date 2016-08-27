const nodeMajorVersion = parseInt(process.versions.node.split('.')[0]);

if (!nodeMajorVersion || nodeMajorVersion < 6) {
  module.exports = require('./dist/synaptic')
} else {
  try {
    module.exports = require('./src/synaptic')
  } catch (e) {
    module.exports = require('./dist/synaptic')
  }
}