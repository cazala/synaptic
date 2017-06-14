// import
var chai = require('chai');
chai.use(require('chai-stats'));
var assert = chai.assert;

var Perceptron = synaptic.Architect.Perceptron;
var LSTM = synaptic.Architect.LSTM;
var Layer = synaptic.Layer;
var Network = synaptic.Network;
var Trainer = synaptic.Trainer;

var learningRate = .5;

// utils
function noRepeat(range, avoid) {
  var number = Math.random() * range | 0;
  for (var i in avoid) {
    if (number == avoid[i]) {
      return noRepeat(range, avoid);
    }
  }
  return number;
}

function equal(prediction, output) {
  for (var i in prediction)
    if (Math.round(prediction[i]) != output[i])
      return false;
  return true;
}

function generateRandomArray(size) {
  var array = [];
  for (var j = 0; j < size; j++)
    array.push(Math.random() + .5 | 0);
  return array;
}

function calculateMse(a, b) {
  var mse = 0;
  for (var k in a)
    mse += Math.pow(a[k] - b[k], 2);
  mse /= a.length;

  return mse;
}

function equalWithError(output, expected, error) {
  return Math.abs(output - expected) <= error;
}

// specs

describe('Basic Neural Network', function () {

  it("trains an AND gate", function () {

    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    inputLayer.project(outputLayer);

    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    var trainer = new Trainer(network);

    var trainingSet = [{
      input: [0, 0],
      output: [0]
    }, {
      input: [0, 1],
      output: [0]
    }, {
      input: [1, 0],
      output: [0]
    }, {
      input: [1, 1],
      output: [1]
    }];

    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 0, "[0,1] did not output 0");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 0, "[1,0] did not output 0");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  it("trains an OR gate", function () {

    var inputLayer = new Layer(2),
      outputLayer = new Layer(1);

    inputLayer.project(outputLayer);

    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    var trainer = new Trainer(network);

    var trainingSet = [{
      input: [0, 0],
      output: [0]
    }, {
      input: [0, 1],
      output: [1]
    }, {
      input: [1, 0],
      output: [1]
    }, {
      input: [1, 1],
      output: [1]
    }];

    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    var test00 = Math.round(network.activate([0, 0]));
    assert.equal(test00, 0, "[0,0] did not output 0");

    var test01 = Math.round(network.activate([0, 1]));
    assert.equal(test01, 1, "[0,1] did not output 1");

    var test10 = Math.round(network.activate([1, 0]));
    assert.equal(test10, 1, "[1,0] did not output 1");

    var test11 = Math.round(network.activate([1, 1]));
    assert.equal(test11, 1, "[1,1] did not output 1");
  });

  it("trains a NOT gate", function () {

    var inputLayer = new Layer(1),
      outputLayer = new Layer(1);

    inputLayer.project(outputLayer);

    var network = new Network({
      input: inputLayer,
      output: outputLayer
    });

    var trainer = new Trainer(network);
    var trainingSet = [{
      input: [0],
      output: [1]
    }, {
      input: [1],
      output: [0]
    }];

    trainer.train(trainingSet, {
      iterations: 1000,
      error: .001
    });

    var test0 = Math.round(network.activate([0]));
    assert.equal(test0, 1, "0 did not output 1");

    var test1 = Math.round(network.activate([1]));
    assert.equal(test1, 0, "1 did not output 0");
  });
});

describe("Perceptron - XOR", function () {

  var perceptron = new Perceptron(2, 3, 1);
  var trainer = new Trainer(perceptron);
  trainer.XOR();

  it("should return near-0 value on [0,0]", function () {
    assert.isAtMost(perceptron.activate([0, 0]), .49, "[0,0] did not output 0");
  });

  it("should return near-1 value on [0,1]", function () {
    assert.isAtLeast(perceptron.activate([0, 1]), .51, "[0,1] did not output 1");
  });

  it("should return near-1 value on [1,0]", function () {
    assert.isAtLeast(perceptron.activate([1, 0]), .51, "[1,0] did not output 1");
  });

  it("should return near-0 value on [1,1]", function () {
    assert.isAtMost(perceptron.activate([1, 1]), .49, "[1,1] did not output 0");
  });
});

describe("Perceptron - SIN", function () {
  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);
  var trainingSet = [];

  while (trainingSet.length < 800) {
    var inputValue = Math.random() * Math.PI * 2;
    trainingSet.push({
      input: [inputValue],
      output: [mySin(inputValue)]
    });
  }

  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
  });

  [0, .5 * Math.PI, 2]
    .forEach(function (x) {
      var y = mySin(x);
      it("should return value around " + y + " when [" + x + "] is on input", function () {
        // near scalability: abs(expected-actual) < 0.5 * 10**(-decimal)
        // 0.5 * Math.pow(10, -.15) => 0.35397289219206896
        assert.almostEqual(sinNetwork.activate([x])[0], y, .15);
      });
    });

  var errorResult = results.error;
  it("Sin error: " + errorResult, function () {
    assert.isAtMost(errorResult, .001, "Sin error not less than or equal to desired error.");
  });
});

describe("Perceptron - SIN - CrossValidate", function () {

  var mySin = function (x) {
    return (Math.sin(x) + 1) / 2;
  };

  var sinNetwork = new Perceptron(1, 12, 1);
  var trainer = new Trainer(sinNetwork);

  var trainingSet = Array.apply(null, Array(800)).map(function () {
    var inputValue = Math.random() * Math.PI * 2;
    return {
      input: [inputValue],
      output: [mySin(inputValue)]
    };
  });

  var results = trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
    crossValidate: {
      testSize: .3,
      testError: 1e-6
    }
  });

  var test0 = sinNetwork.activate([0])[0];
  var expected0 = mySin(0);
  it("input: [0] output: " + test0 + ", expected: " + expected0, function () {
    assert.isAtMost(Math.abs(test0 - expected0), .035, "[0] did not output " + expected0);
  });

  var test05PI = sinNetwork.activate([.5 * Math.PI])[0];
  var expected05PI = mySin(.5 * Math.PI);
  it("input: [0.5*Math.PI] output: " + test05PI + ", expected: " + expected05PI, function () {
    assert.isAtMost(Math.abs(test05PI - expected05PI), .035, "[0.5*Math.PI] did not output " + expected05PI);
  });

  var test2 = sinNetwork.activate([2])[0];
  var expected2 = mySin(2);
  it("input: [2] output: " + test2 + ", expected: " + expected2, function () {
    var eq = equalWithError(test2, expected2, .035);
    assert.equal(eq, true, "[2] did not output " + expected2);
  });

  var errorResult = results.error;
  it("CrossValidation error: " + errorResult, function () {
    var lessThanOrEqualError = errorResult <= .001;
    assert.equal(lessThanOrEqualError, true, "CrossValidation error not less than or equal to desired error.");
  });
});

describe("LSTM - Discrete Sequence Recall", function () {
  var targets = [2, 4];
  var distractors = [3, 5];
  var prompts = [0, 1];
  var length = 9;

  var lstm = new LSTM(5, 3, 2);
  var trainer = new Trainer(lstm);

  trainer.DSR({
    targets: targets,
    distractors: distractors,
    prompts: prompts,
    length: length,
    rate: .17,
    iterations: 250000
  });

  var symbols = targets.length + distractors.length + prompts.length;
  var sequence = [],
    indexes = [],
    positions = [];
  var sequenceLength = length - prompts.length;

  for (i = 0; i < sequenceLength; i++) {
    var any = Math.random() * distractors.length | 0;
    sequence.push(distractors[any]);
  }
  indexes = [], positions = [];
  for (i = 0; i < prompts.length; i++) {
    indexes.push(Math.random() * targets.length | 0);
    positions.push(noRepeat(sequenceLength, positions));
  }
  positions = positions.sort();
  for (i = 0; i < prompts.length; i++) {
    sequence[positions[i]] = targets[indexes[i]];
    sequence.push(prompts[i]);
  }

  var check = function (which) {
    // generate input from sequence
    var input = [];
    for (j = 0; j < symbols; j++)
      input[j] = 0;
    input[sequence[which]] = 1;

    // generate target output
    var output = [];
    for (j = 0; j < targets.length; j++)
      output[j] = 0;

    if (which >= sequenceLength) {
      var index = which - sequenceLength;
      output[indexes[index]] = 1;
    }

    // check result
    var prediction = lstm.activate(input);
    return {
      prediction: prediction,
      output: output
    };
  };

  var value = function (array) {
    var max = .5;
    var res = -1;
    for (var i in array)
      if (array[i] > max) {
        max = array[i];
        res = i;
      }
    return res == -1 ? '-' : targets[res];
  };

  it("targets: " + targets, function () {
    assert(true);
  });
  it("distractors: " + distractors, function () {
    assert(true);
  });
  it("prompts: " + prompts, function () {
    assert(true);
  });
  it("length: " + length + "\n", function () {
    assert(true);
  });

  for (var i = 0; i < length; i++) {
    var test = check(i);
    it((i + 1) + ") input: " + sequence[i] + " output: " + value(test.prediction),
      function () {
        var ok = equal(test.prediction, test.output);
        assert(ok);
      });
  }
});

describe("LSTM - Timing Task", function () {
  var network = new LSTM(2, 7, 1);
  var trainer = new Trainer(network);
  var result = trainer.timingTask({
    log: false,
    trainSamples: 4000,
    testSamples: 500
  });

  it("should complete the training in less than 200 iterations", function () {
    assert(result.train.iterations <= 200);
  });

  it("should pass the test with an error smaller than 0.05", function () {
    assert(result.test.error < .05);
  });
});

describe("Optimized and Unoptimized Networks Equivalency", function () {

  var optimized;
  var unoptimized;
  beforeEach(function () {
    optimized = new LSTM(2, 1, 1);
    unoptimized = optimized.clone();
    unoptimized.setOptimize(false);
  });


  it('should produce the same output for both networks', function () {
    this.timeout(30000);
    for (var i = 0; i < 1000; i++) {
      var input = generateRandomArray(2);
      var target = generateRandomArray(1);
      optimized.activate(input);
      unoptimized.activate(input);
      optimized.propagate(learningRate, target);
      unoptimized.propagate(learningRate, target);
    }
    var mse = calculateMse(optimized.activate(input), unoptimized.activate(input));
    assert.isAtMost(mse, 1e-9, 'output should be same for both networks after ' + i + ' iterations');
  });
});

describe("toJSON/fromJSON Networks Equivalency", function () {
  var original;
  var imported;
  beforeEach(function () {
    original = new LSTM(10, 5, 5);
    imported = Network.fromJSON(original.toJSON());
  });

  it('should produce the same output for both networks', function () {
    this.timeout(30000);
    for (var i = 0; i < 1000; i++) {
      var input = generateRandomArray(10);
      var output1 = original.activate(input);
      var output2 = imported.activate(input);

      var target = generateRandomArray(5);

      // propagate networks
      original.propagate(learningRate, target);
      imported.propagate(learningRate, target);

      assert.isAtMost(calculateMse(output1, output2), 1e-10,
        'output should be same for both networks after ' + i + ' iterations');
    }
  });
});

describe("Cloned Networks Equivalency", function () {
  var original;
  var cloned;
  beforeEach(function () {
    original = new LSTM(10, 5, 5);
    cloned = Network.fromJSON(original.toJSON());
  });

  it('should produce the same output for both networks', function () {
    this.timeout(30000);
    for (var i = 0; i < 1000; i++) {
      var input = generateRandomArray(10);
      var output1 = original.activate(input);
      var output2 = cloned.activate(input);

      var target = generateRandomArray(5);

      // propagate networks
      original.propagate(learningRate, target);
      cloned.propagate(learningRate, target);

      assert.isAtMost(calculateMse(output1, output2), 1e-10,
        'output should be same for both networks after ' + i + ' iterations');
    }
  });
});

describe("Scheduled Tasks", function () {
  var perceptron = new Perceptron(2, 3, 1);
  var trainer = new Trainer(perceptron);

  it('should stop training at 3000 iterations', function () {
    var final_stats = trainer.XOR({
      iterations: 3000,
      rate: 0.000001,
      error: 0.000001,
      schedule: {
        every: 1000,
        do: function (data) {
          return data.iterations == 20000;
        }
      }
    });
    assert.equal(final_stats.iterations, 3000)
  });

  it('should abort the training at 2000 iterations', function () {
    var final_stats = trainer.XOR({
      iterations: 3000,
      rate: 0.000001,
      error: 0.000001,
      schedule: {
        every: 1000,
        do: function (data) {
          return data.iterations == 2000;
        }
      }
    });
    assert.equal(final_stats.iterations, 2000)
  });

  it('should work even if schedule.do() returns no value', function () {
    var final_stats = trainer.XOR({
      iterations: 3000,
      rate: 0.000001,
      error: 0.000001,
      schedule: {
        every: 1000,
        do: function (data) {}
      }
    });
    assert.equal(final_stats.iterations, 3000)
  });

});

describe("Rate Callback Check", function () {
  var perceptron = new Perceptron(2, 3, 1);
  var trainer = new Trainer(perceptron);

  it('should switch rate from 0.01 to 0.005 after 1000 iterations', function () {
    var final_stats = trainer.XOR({
      iterations: 2000,
      rate: function (iterations, error) {
        return iterations < 1000 ? 0.01 : 0.005
      },
      error: 0.000001,
      schedule: {
        every: 1,
        do: function (data) {
          switch (data.iterations) {
            case 1:
            case 500:
            case 999:
              assert.equal(data.rate, 0.01);
              break;

            case 1000:
            case 1500:
            case 2000:
              assert.equal(data.rate, 0.005);
              break;
          }
        }
      }
    });
  });
});

describe("Rate Array Check", function () {
  var perceptron = new Perceptron(2, 3, 1);
  var trainer = new Trainer(perceptron);

  it('should switch rate from 0.01 to 0.005 after 1000 iterations', function () {
    var final_stats = trainer.XOR({
      iterations: 2000,
      rate: [0.01, 0.005],
      error: 0.000001,
      schedule: {
        every: 1,
        do: function (data) {
          switch (data.iterations) {
            case 1:
            case 500:
            case 999:
              assert.equal(data.rate, 0.01);
              break;

            case 1000:
            case 1500:
            case 2000:
              assert.equal(data.rate, 0.005);
              break;
          }
        }
      }
    });
  });
});
