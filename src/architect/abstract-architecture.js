const Network = require('../network');
const Trainer = require('../trainer');

class AbstractArchitecture extends Network {
  constructor(layers) {
    super(layers);
    this.trainer = new this.constructor.Trainer(this);
  }
}

module.exports = AbstractArchitecture;

AbstractArchitecture.NetworkClass = Network;
AbstractArchitecture.Trainer = Trainer;
