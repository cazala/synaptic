// import

var assert = require('assert'),
  synaptic = require('../src/synaptic');

var Perceptron = synaptic.Architect.Perceptron,
  LSTM = synaptic.Architect.LSTM,
  Layer = synaptic.Layer,
  Network = synaptic.Network,
  Trainer = synaptic.Trainer;


// utils

var noRepeat = function(range, avoid) {
  var number = Math.random() * range | 0;
  var used = false;
  for (var i in avoid)
    if (number == avoid[i])
      used = true;
  return used ? noRepeat(range, avoid) : number;
};

var equal = function(prediction, output) {
  for (var i in prediction)
    if (Math.round(prediction[i]) != output[i])
      return false;
  return true;
};

var generateRandomArray = function(size){
    var array = [];
    for (var j = 0; j < size; j++)
        array.push(Math.random() + .5 | 0);
    return array;
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

    var test10 = Math.round(network.activate([0, 1]));
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

    var test10 = Math.round(network.activate([0, 1]));
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

describe("Optimized and Unoptimized Networks Equivalency", function() {
  var optimized = new Perceptron(10,15,5);

  var unoptimized = optimized.clone();
  unoptimized.setOptimize(false);

  var learningRate = .5;
  var iterations = 1000;

  for (var i = 1; i <= iterations; i++)
  {
      //random input
      var input = generateRandomArray(10);

      // activate networks
      var output1 = optimized.activate(input);
      var output2 = unoptimized.activate(input);

      if (i % 100 == 0)
        it(' same output for both networks after ' + i + ' iterations', function(){
          var diff = false;
          for (var k in output1)
            if (output1[k] - output2[k] != 0)
              diff = true;
          assert(!diff);
        });

      // random target
      var target = generateRandomArray(5);

      // propagate networks
      optimized.propagate(learningRate, target);
      unoptimized.propagate(learningRate, target);
  }
});

describe("toJSON/fromJSON Networks Equivalency", function() {
  var original = new Perceptron(10,15,5);

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
        it(' same output for both networks after ' + i + ' iterations', function(){
          var diff = false;
          for (var k in output1)
            if (output1[k] - output2[k] != 0)
              diff = true;
          assert(!diff);
        });

      // random target
      var target = generateRandomArray(5);

      // propagate networks
      original.propagate(learningRate, target);
      imported.propagate(learningRate, target);
  }
});

describe("Cloned Networks Equivalency", function() {
  var original = new Perceptron(10,15,5);
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
        it(' same output for both networks after ' + i + ' iterations', function(){
          var diff = false;
          for (var k in output1)
            if (output1[k] - output2[k] != 0)
              diff = true;
          assert(!diff);
        });

      // random target
      var target = generateRandomArray(5);

      // propagate networks
      original.propagate(learningRate, target);
      cloned.propagate(learningRate, target);
  }
});

describe("Manual Override", function() {
  var perceptron = new Perceptron(2, 3, 1);

  it('iterations ended at 2000, not full 3000', function(){
    var final_stats = perceptron.trainer.XOR({
      iterations: 3000,
      rate: 0.000001,
      error: 0.000001,
      customLog: {
          every: 1000,
          do: function(data) {
            // console.log( data.iterations, 'e', data.error )
            if( data.iterations == 2000){
              return false
            }else{
              return true
            }
          }
        }
    });
    assert.equal( final_stats.iterations, 2000 )
  });
});