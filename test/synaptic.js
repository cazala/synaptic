// import

var assert = require('assert'),
  synaptic = require('../src/synaptic');

var Perceptron = synaptic.Architect.Perceptron,
  LSTM = synaptic.Architect.LSTM,
  Layer = synaptic.Layer,
  Network = synaptic.Network,
  Trainer = synaptic.Trainer;


// utils

function noRepeat (range, avoid) {
  var number = Math.random() * range | 0;
  for (var i in avoid){
    if (number == avoid[i]){
      return noRepeat(range,avoid);
    }
  }
  return number;
};

function equal (prediction, output) {
  for (var i in prediction)
    if (Math.round(prediction[i]) != output[i])
      return false;
  return true;
};

function generateRandomArray (size){
    var array = [];
    for (var j = 0; j < size; j++)
        array.push(Math.random() + .5 | 0);
    return array;
}

function compare (a, b) {
  var mse = 0;
  for (var k in a)
    mse += Math.pow(a[k] - b[k], 2);
  mse /= a.length;

  return mse < 1e-10;
}

// specs

describe('Basic Neural Network', function() {

  it("trains an AND gate", function() {

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
  
  it("trains an OR gate", function() {

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

  it("trains a NOT gate", function() {

    var inputLayer = new Layer(1),
      outputLayer = new Layer(1),
      network;

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

describe("Perceptron - XOR", function() {

  var perceptron = new Perceptron(2, 3, 1);
  perceptron.trainer.XOR();

  var test00 = Math.round(perceptron.activate([0, 0]));
  it("input: [0,0] output: " + test00, function() {

    assert.equal(test00, 0, "[0,0] did not output 0");
  });

  var test01 = Math.round(perceptron.activate([0, 1]));
  it("input: [0,1] output: " + test01, function() {

    assert.equal(test01, 1, "[0,1] did not output 1");
  });

  var test10 = Math.round(perceptron.activate([1, 0]));
  it("input: [1,0] output: " + test10, function() {

    assert.equal(test10, 1, "[1,0] did not output 1");
  });

  var test11 = Math.round(perceptron.activate([1, 1]));
  it("input: [1,1] output: " + test11, function() {

    assert.equal(test11, 0, "[1,1] did not output 0");
  });
});

describe("Perceptron - SIN", function() {

  var mySin = function(x) {
    return (Math.sin(x)+1)/2;
  };

  var equalError = function(output, expected, error) {
    return Math.abs(output - expected) <= error;
  };

  var sinNetwork = new Perceptron(1, 12, 1);

  var trainingSet = Array.apply(null, Array(1000)).map(function () {
    var inputValue = Math.random() * Math.PI * 2;
    return {
      input: [inputValue],
      output: [mySin(inputValue)]
    }
  });

  var results = sinNetwork.trainer.train(trainingSet, {
    iterations: 2000,
    log: false,
    error: 1e-6,
    cost: Trainer.cost.MSE,
  });
 
  var test0 = sinNetwork.activate([0])[0];
  it("input: [0] output: " + test0, function() {
    var expected = mySin(0);
    var eq = equalError(test0, expected, .03);
    assert.equal(eq, true, "[0] did not output " + expected);
  });

  var test05PI = sinNetwork.activate([.5*Math.PI])[0];
  it("input: [0.5*Math.PI] output: " + test05PI, function() {
    var expected = mySin(.5*Math.PI);
    var eq = equalError(test05PI, expected, .03);
    assert.equal(eq, true, "[0.5*Math.PI] did not output " + expected);
  });

  var test2 = sinNetwork.activate([2])[0];
  it("input: [2] output: " + test2, function() {
    var expected = mySin(2);
    var eq = equalError(test2, expected, .03);
    assert.equal(eq, true, "[2] did not output " + expected);
  });

  var errorResult = results.error;
  it("Sin error: " + errorResult, function() {
    var lessThanOrEqualError = errorResult <= .03;
    assert.equal(lessThanOrEqualError, true, "Sin error not less than or equal to desired error.");
  });
});

describe("Perceptron - SIN - CrossValidate", function() {

  var mySin = function(x) {
    return (Math.sin(x)+1)/2;
  };

  var equalError = function(output, expected, error) {
    return Math.abs(output - expected) <= error;
  };

  var sinNetwork = new Perceptron(1, 12, 1);

  var trainingSet = Array.apply(null, Array(1000)).map(function () {
    var inputValue = Math.random() * Math.PI * 2;
    return {
      input: [inputValue],
      output: [mySin(inputValue)]
    }
  });

  var results = sinNetwork.trainer.train(trainingSet, {
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
  it("input: [0] output: " + test0, function() {
    var expected = mySin(0);
    var eq = equalError(test0, expected, .03);
    assert.equal(eq, true, "[0] did not output " + expected);
  });

  var test05PI = sinNetwork.activate([.5*Math.PI])[0];
  it("input: [0.5*Math.PI] output: " + test05PI, function() {
    var expected = mySin(.5*Math.PI);
    var eq = equalError(test05PI, expected, .03);
    assert.equal(eq, true, "[0.5*Math.PI] did not output " + expected);
  });

  var test2 = sinNetwork.activate([2])[0];
  it("input: [2] output: " + test2, function() {
    var expected = mySin(2);
    var eq = equalError(test2, expected, .03);
    assert.equal(eq, true, "[2] did not output " + expected);
  });

  var errorResult = results.error;
  it("CrossValidation error: " + errorResult, function() {
    var lessThanOrEqualError = errorResult <= .03;
    assert.equal(lessThanOrEqualError, true, "CrossValidation error not less than or equal to desired error.");
  });
});

describe("LSTM - Discrete Sequence Recall", function() {

  var targets = [2, 4];
  var distractors = [3, 5];
  var prompts = [0, 1];
  var length = 9;

  var lstm = new LSTM(5, 3, 2);
  lstm.trainer.DSR({
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

  var check = function(which) {
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

  var value = function(array) {
    var max = .5;
    var res = -1;
    for (var i in array)
      if (array[i] > max) {
        max = array[i];
        res = i;
      }
    return res == -1 ? '-' : targets[res];
  };

  it("targets: " + targets, function() {
    assert(true);
  });
  it("distractors: " + distractors, function() {
    assert(true);
  });
  it("prompts: " + prompts, function() {
    assert(true);
  });
  it("length: " + length + "\n", function() {
    assert(true);
  });

  for (var i = 0; i < length; i++) {
    var test = check(i);
    it((i + 1) + ") input: " + sequence[i] + " output: " + value(test.prediction),
      function() {
        var ok = equal(test.prediction, test.output);
        assert(ok);
      });
  }
});

describe("LSTM - Timing Task", function() {
  var network = new synaptic.Architect.LSTM(2,7,1);
  var result = network.trainer.timingTask({ 
    log: false,
    trainSamples: 4000,
    testSamples: 500
  });
  
  it("should complete the training in less than 200 iterations", function() {
    assert(result.train.iterations <= 200);
  });

  it("should pass the test with an error smaller than 0.05", function() {
    assert(result.test.error < .05);
  });
});

describe("Optimized and Unoptimized Networks Equivalency", function() {
  var optimized = new LSTM(2,1,1)

  var unoptimized = optimized.clone();
  unoptimized.setOptimize(false);

  var learningRate = .5;
  var iterations = 1000;

  for (var i = 1; i <= iterations; i++)
  {
      //random input
      var input = generateRandomArray(2);

      // activate networks
      var output1 = optimized.activate(input);
      var output2 = unoptimized.activate(input);

      if (i % 100 == 0)
        it('should produce the same output for both networks after ' + i + ' iterations', function(){
          assert(compare(output1, output2));
        });

      // random target
      var target = generateRandomArray(1);

      // propagate networks
      optimized.propagate(learningRate, target);
      unoptimized.propagate(learningRate, target);
  }
});

describe("toJSON/fromJSON Networks Equivalency", function() {
  var original = new LSTM(10,5,5);

  var exported = original.toJSON();
  var imported = Network.fromJSON(exported);

  var learningRate = .5;
  var iterations = 1000;

  for (var i = 1; i <= iterations; i++)
  {
      //random input
      var input = generateRandomArray(10);

      // activate networks
      var output1 = original.activate(input);
      var output2 = imported.activate(input);

      if (i % 100 == 0)
        it('should produce the same output for both networks after ' + i + ' iterations', function(){
          assert(compare(output1, output2));
        });

      // random target
      var target = generateRandomArray(5);

      // propagate networks
      original.propagate(learningRate, target);
      imported.propagate(learningRate, target);
  }
});

describe("Cloned Networks Equivalency", function() {
  
  var original = new LSTM(10,5,5);

  var cloned = original.clone();

  var learningRate = .5;
  var iterations = 1000;

  for (var i = 1; i <= iterations; i++)
  {
      //random input
      var input = generateRandomArray(10);

      // activate networks
      var output1 = original.activate(input);
      var output2 = cloned.activate(input);

      if (i % 100 == 0)
        it('should produce the same output for both networks after ' + i + ' iterations', function(){
          assert(compare(output1, output2));
        });

      // random target
      var target = generateRandomArray(5);

      // propagate networks
      original.propagate(learningRate, target);
      cloned.propagate(learningRate, target);
  }
});

describe("Scheduled Tasks", function() {
  var perceptron = new Perceptron(2, 3, 1);

  it('should stop training at 3000 iterations', function(){
    var final_stats = perceptron.trainer.XOR({
      iterations: 3000,
      rate: 0.000001,
      error: 0.000001,
      schedule: {
          every: 1000,
          do: function(data) {
            if( data.iterations == 20000){
              return true
            }
          }
        }
    });
    assert.equal( final_stats.iterations, 3000 )
  });

  it('should abort the training at 2000 iterations', function(){
    var final_stats = perceptron.trainer.XOR({
      iterations: 3000,
      rate: 0.000001,
      error: 0.000001,
      schedule: {
          every: 1000,
          do: function(data) {
            if( data.iterations == 2000){
              return true
            }
          }
        }
    });
    assert.equal( final_stats.iterations, 2000 )
  });

  it('should work even if shedule.do() returns no value', function(){
    var final_stats = perceptron.trainer.XOR({
      iterations: 3000,
      rate: 0.000001,
      error: 0.000001,
      schedule: {
          every: 1000,
          do: function(data) {}
        }
    });
    assert.equal( final_stats.iterations, 3000 )
  });

});
