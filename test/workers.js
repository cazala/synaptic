var synaptic = require('../src/synaptic.js');

var chai = require('chai');
chai.use(require('chai-stats/lib/stats'));
chai.use(require('chai-as-promised'));
var {assert, expect} = chai;

var {Perceptron} = synaptic.Architect;

describe('Browser: Web Worker', function () {
  if (!global.Worker) {
    return
  }

  var perceptron;
  var perceptronWorker;
  beforeEach(() => {
    perceptron = new Perceptron(2, 3, 1);
    perceptron.trainer.XOR();
    return perceptron.worker()
      .then(network => {
        perceptronWorker = network
      })
  });

  it('Should at least work', () =>
    assert.isFulfilled(perceptronWorker.activate([1, 0])));

  it("should return near-0 value on [0,0]", function () {
    expect(perceptronWorker.activate([0, 0])).to.eventually.be.at.most(.49, "[0,0] did not output 0");
  });

  it("should return near-1 value on [0,1]", function () {
    expect(perceptronWorker.activate([0, 1])).to.eventually.be.at.most(.51, "[0,1] did not output 1");
  });

  it("should return near-1 value on [1,0]", function () {
    expect(perceptronWorker.activate([1, 0])).to.eventually.be.at.most(.51, "[1,0] did not output 1");
  });

  it("should return near-0 value on [1,1]", function () {
    expect(perceptronWorker.activate([1, 1])).to.eventually.be.at.most(.49, "[1,1] did not output 0");
  });
});