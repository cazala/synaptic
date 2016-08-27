/*******************************************************************************************
                                        TRAINER
*******************************************************************************************/

class Trainer {
  constructor(network, options={}) {
    this.network = network;
    this.rate = options.rate || .2;
    this.iterations = options.iterations || 100000;
    this.error = options.error || .005;
    this.cost = options.cost || null;
    this.crossValidate = options.crossValidate || null;
  }

  // trains any given set to a network
  train(set, options) {

    let error = 1;
    let iterations = bucketSize = 0;
    let abort = false;
    let currentRate;
    const cost = options && options.cost || this.cost || Trainer.cost.MSE;
    let crossValidate = false, testSet, trainSet;

    const start = Date.now();

    if (options) {
      if (options.iterations)
        this.iterations = options.iterations;
      if (options.error)
        this.error = options.error;
      if (options.rate)
        this.rate = options.rate;
      if (options.cost)
        this.cost = options.cost;
      if (options.schedule)
        this.schedule = options.schedule;
      if (options.customLog){
        // for backward compatibility with code that used customLog
        console.log('Deprecated: use schedule instead of customLog')
        this.schedule = options.customLog;
      }
      if (this.crossValidate || options.crossValidate) {
        if(!this.crossValidate) this.crossValidate = {};
        crossValidate = true;
        if (options.crossValidate.testSize)
          this.crossValidate.testSize = options.crossValidate.testSize;
        if (options.crossValidate.testError)
          this.crossValidate.testError = options.crossValidate.testError;
      }
    }

    currentRate = this.rate;
    if(Array.isArray(this.rate)) {
      var bucketSize = Math.floor(this.iterations / this.rate.length);
    }

    if(crossValidate) {
      const numTrain = Math.ceil((1 - this.crossValidate.testSize) * set.length);
      trainSet = set.slice(0, numTrain);
      testSet = set.slice(numTrain);
    }

    var lastError = 0;
    while ((!abort && iterations < this.iterations && error > this.error)) {
      if (crossValidate && error <= this.crossValidate.testError) {
        break;
      }

      let currentSetSize = set.length;
      error = 0;
      iterations++;

      if(bucketSize > 0) {
        const currentBucket = Math.floor(iterations / bucketSize);
        currentRate = this.rate[currentBucket] || currentRate;
      }
      
      if(typeof this.rate === 'function') {
        currentRate = this.rate(iterations, lastError);
      }

      if (crossValidate) {
        this._trainSet(trainSet, currentRate, cost);
        error += this.test(testSet).error;
        currentSetSize = 1;
      } else {
        error += this._trainSet(set, currentRate, cost);
        currentSetSize = set.length;
      }

      // check error
      error /= currentSetSize;
      lastError = error;

      if (options) {
        if (this.schedule && this.schedule.every && iterations %
          this.schedule.every == 0)
          abort = this.schedule.do({ error, iterations, rate: currentRate });
        else if (options.log && iterations % options.log == 0) {
          console.log('iterations', iterations, 'error', error, 'rate', currentRate);
        };
        if (options.shuffle)
          shuffle(set);
      }
    }

    const results = {
      error,
      iterations,
      time: Date.now() - start
    };

    return results;
  }

  // trains any given set to a network, using a WebWorker (only for the browser). Returns a Promise of the results.
  trainAsync(set, options) {
    const train = this.workerTrain.bind(this);
    return new Promise((resolve, reject) => {
      try {
        train(set, resolve, options, true)
      } catch(e) {
        reject(e)
      }
    })
  }

  // preforms one training epoch and returns the error (private function used in this.train)
  _trainSet(set, currentRate, costFunction) {
    let errorSum = 0;
    for (let train in set) {
      const input = set[train].input;
      const target = set[train].output;

      const output = this.network.activate(input);
      this.network.propagate(currentRate, target);

      errorSum += costFunction(target, output);
    }
    return errorSum;
  }

  // tests a set and returns the error and elapsed time
  test(set, options) {

    let error = 0;
    let input, output, target;
    const cost = options && options.cost || this.cost || Trainer.cost.MSE;

    const start = Date.now();

    for (let test in set) {
      input = set[test].input;
      target = set[test].output;
      output = this.network.activate(input);
      error += cost(target, output);
    }

    error /= set.length;

    const results = {
      error,
      time: Date.now() - start
    };

    return results;
  }

  // trains any given set to a network using a WebWorker [deprecated: use trainAsync instead]
  workerTrain(set, callback, options, suppressWarning) {

    if (!suppressWarning) {
      console.warn('Deprecated: do not use `workerTrain`, use `trainAsync` instead.')
    }
    const that = this;

    if (!this.network.optimized)
      this.network.optimize();

    // Create a new worker
    const worker = this.network.worker(this.network.optimized.memory, set, options);

    // train the worker
    worker.onmessage = e => {
      switch(e.data.action) {
          case 'done':
            const iterations = e.data.message.iterations;
            const error = e.data.message.error;
            const time = e.data.message.time;

            that.network.optimized.ownership(e.data.memoryBuffer);

            // Done callback
            callback({
              error,
              iterations,
              time
            });

            // Delete the worker and all its associated memory
            worker.terminate();
          break;

          case 'log':
            console.log(e.data.message);

          case 'schedule':
            if (options && options.schedule && typeof options.schedule.do === 'function') {
              const scheduled = options.schedule.do;
              scheduled(e.data.message)
            }
          break;
      }
    };

    // Start the worker
    worker.postMessage({action: 'startTraining'});
  }

  // trains an XOR to the network
  XOR(options) {

    if (this.network.inputs() != 2 || this.network.outputs() != 1)
      throw new Error("Incompatible network (2 inputs, 1 output)");

    const defaults = {
      iterations: 100000,
      log: false,
      shuffle: true,
      cost: Trainer.cost.MSE
    };

    if (options)
      for (let i in options)
        defaults[i] = options[i];

    return this.train([{
      input: [0, 0],
      output: [0]
    }, {
      input: [1, 0],
      output: [1]
    }, {
      input: [0, 1],
      output: [1]
    }, {
      input: [1, 1],
      output: [0]
    }], defaults);
  }

  // trains the network to pass a Distracted Sequence Recall test
  DSR(options={}) {
    const targets = options.targets || [2, 4, 7, 8];
    const distractors = options.distractors || [3, 5, 6, 9];
    const prompts = options.prompts || [0, 1];
    const length = options.length || 24;
    const criterion = options.success || 0.95;
    const iterations = options.iterations || 100000;
    const rate = options.rate || .1;
    const log = options.log || 0;
    const schedule = options.schedule || {};
    const cost = options.cost || this.cost || Trainer.cost.CROSS_ENTROPY;

    let trial, correct, i, j, success;
    trial = correct = i = j = success = 0;
    let error = 1;
    const symbols = targets.length + distractors.length + prompts.length;

    const noRepeat = (range, avoid) => {
      const number = Math.random() * range | 0;
      let used = false;
      for (let i in avoid)
        if (number == avoid[i])
          used = true;
      return used ? noRepeat(range, avoid) : number;
    };

    const equal = (prediction, output) => {
      for (let i in prediction)
        if (Math.round(prediction[i]) != output[i])
          return false;
      return true;
    };

    const start = Date.now();

    while (trial < iterations && (success < criterion || trial % 1000 != 0)) {
      // generate sequence
      const sequence = [], sequenceLength = length - prompts.length;
      for (i = 0; i < sequenceLength; i++) {
        const any = Math.random() * distractors.length | 0;
        sequence.push(distractors[any]);
      }
      const indexes = [];
      let positions = [];
      for (i = 0; i < prompts.length; i++) {
        indexes.push(Math.random() * targets.length | 0);
        positions.push(noRepeat(sequenceLength, positions));
      }
      positions = positions.sort();
      for (i = 0; i < prompts.length; i++) {
        sequence[positions[i]] = targets[indexes[i]];
        sequence.push(prompts[i]);
      }

      //train sequence
      let distractorsCorrect;
      let targetsCorrect = distractorsCorrect = 0;
      error = 0;
      for (i = 0; i < length; i++) {
        // generate input from sequence
        const input = [];
        for (j = 0; j < symbols; j++)
          input[j] = 0;
        input[sequence[i]] = 1;

        // generate target output
        const output = [];
        for (j = 0; j < targets.length; j++)
          output[j] = 0;

        if (i >= sequenceLength) {
          const index = i - sequenceLength;
          output[indexes[index]] = 1;
        }

        // check result
        const prediction = this.network.activate(input);

        if (equal(prediction, output))
          if (i < sequenceLength)
            distractorsCorrect++;
          else
            targetsCorrect++;
        else {
          this.network.propagate(rate, output);
        }

        error += cost(output, prediction);

        if (distractorsCorrect + targetsCorrect == length)
          correct++;
      }

      // calculate error
      if (trial % 1000 == 0)
        correct = 0;
      trial++;
      let divideError = trial % 1000;
      divideError = divideError == 0 ? 1000 : divideError;
      success = correct / divideError;
      error /= length;

      // log
      if (log && trial % log == 0)
        console.log("iterations:", trial, " success:", success, " correct:",
          correct, " time:", Date.now() - start, " error:", error);
      if (schedule.do && schedule.every && trial % schedule.every == 0)
        schedule.do({
          iterations: trial,
          success,
          error,
          time: Date.now() - start,
          correct
        });
    }

    return {
      iterations: trial,
      success,
      error,
      time: Date.now() - start
    }
  }

  // train the network to learn an Embeded Reber Grammar
  ERG(options={}) {
    const iterations = options.iterations || 150000;
    const criterion = options.error || .05;
    const rate = options.rate || .1;
    const log = options.log || 500;
    const cost = options.cost || this.cost || Trainer.cost.CROSS_ENTROPY;

    // gramar node
    class Node {
      constructor() {
        this.paths = [];
      }

      connect(node, value) {
        this.paths.push({
          node,
          value
        });
        return this;
      }

      any() {
        if (this.paths.length == 0)
          return false;
        const index = Math.random() * this.paths.length | 0;
        return this.paths[index];
      }

      test(value) {
        for (let i in this.paths)
          if (this.paths[i].value == value)
            return this.paths[i];
        return false;
      }
    }

    const reberGrammar = () => {

      // build a reber grammar
      const output = new Node();
      const n1 = (new Node()).connect(output, "E");
      const n2 = (new Node()).connect(n1, "S");
      const n3 = (new Node()).connect(n1, "V").connect(n2, "P");
      const n4 = (new Node()).connect(n2, "X");
      n4.connect(n4, "S");
      const n5 = (new Node()).connect(n3, "V");
      n5.connect(n5, "T");
      n2.connect(n5, "X");
      const n6 = (new Node()).connect(n4, "T").connect(n5, "P");
      const input = (new Node()).connect(n6, "B");

      return {
        input,
        output
      }
    };

    // build an embeded reber grammar
    const embededReberGrammar = () => {
      const reber1 = reberGrammar();
      const reber2 = reberGrammar();

      const output = new Node();
      const n1 = (new Node).connect(output, "E");
      reber1.output.connect(n1, "T");
      reber2.output.connect(n1, "P");
      const n2 = (new Node).connect(reber1.input, "P").connect(reber2.input,
        "T");
      const input = (new Node).connect(n2, "B");

      return {
        input,
        output
      }

    };

    // generate an ERG sequence
    const generate = () => {
      const node = embededReberGrammar().input;
      let next = node.any();
      let str = "";
      while (next) {
        str += next.value;
        next = next.node.any();
      }
      return str;
    };

    // test if a string matches an embeded reber grammar
    const test = str => {
      let node = embededReberGrammar().input;
      let i = 0;
      let ch = str.charAt(i);
      while (i < str.length) {
        const next = node.test(ch);
        if (!next)
          return false;
        node = next.node;
        ch = str.charAt(++i);
      }
      return true;
    };

    // helper to check if the output and the target vectors match
    const different = (array1, array2) => {
      let max1 = 0;
      let i1 = -1;
      let max2 = 0;
      let i2 = -1;
      for (let i in array1) {
        if (array1[i] > max1) {
          max1 = array1[i];
          i1 = i;
        }
        if (array2[i] > max2) {
          max2 = array2[i];
          i2 = i;
        }
      }

      return i1 != i2;
    };

    let iteration = 0;
    let error = 1;
    const table = {
      "B": 0,
      "P": 1,
      "T": 2,
      "X": 3,
      "S": 4,
      "E": 5
    };

    const start = Date.now();
    while (iteration < iterations && error > criterion) {
      let i = 0;
      error = 0;

      // ERG sequence to learn
      const sequence = generate();

      // input
      let read = sequence.charAt(i);
      // target
      let predict = sequence.charAt(i + 1);

      // train
      while (i < sequence.length - 1) {
        const input = [];
        const target = [];
        for (let j = 0; j < 6; j++) {
          input[j] = 0;
          target[j] = 0;
        }
        input[table[read]] = 1;
        target[table[predict]] = 1;

        const output = this.network.activate(input);

        if (different(output, target))
          this.network.propagate(rate, target);

        read = sequence.charAt(++i);
        predict = sequence.charAt(i + 1);

        error += cost(target, output);
      }
      error /= sequence.length;
      iteration++;
      if (iteration % log == 0) {
        console.log("iterations:", iteration, " time:", Date.now() - start,
          " error:", error);
      }
    }

    return {
      iterations: iteration,
      error,
      time: Date.now() - start,
      test,
      generate
    }
  }

  timingTask(options) {

    if (this.network.inputs() != 2 || this.network.outputs() != 1)
      throw new Error("Invalid Network: must have 2 inputs and one output");

    if (typeof options == 'undefined')
      options = {};

    // helper
    function getSamples (trainingSize, testSize){

      // sample size
      const size = trainingSize + testSize;

      // generate samples
      let t = 0;
      const set = [];
      for (let i = 0; i < size; i++) {
        set.push({ input: [0,0], output: [0] });
      }
      while(t < size - 20) {
          let n = Math.round(Math.random() * 20);
          set[t].input[0] = 1;
          for (let j = t; j <= t + n; j++){
              set[j].input[1] = n / 20;
              set[j].output[0] = 0.5;
          }
          t += n;
          n = Math.round(Math.random() * 20);
          for (let k = t+1; k <= (t + n) &&  k < size; k++)
              set[k].input[1] = set[t].input[1];
          t += n;
      }

      // separate samples between train and test sets
      const trainingSet = []; const testSet = [];
      for (let l = 0; l < size; l++)
          (l < trainingSize ? trainingSet : testSet).push(set[l]);

      // return samples
      return {
          train: trainingSet,
          test: testSet
      }
    }

    const iterations = options.iterations || 200;
    const error = options.error || .005;
    const rate = options.rate || [.03, .02];
    const log = options.log === false ? false : options.log || 10;
    const cost = options.cost || this.cost || Trainer.cost.MSE;
    const trainingSamples = options.trainSamples || 7000;
    const testSamples = options.trainSamples || 1000;

    // samples for training and testing
    const samples = getSamples(trainingSamples, testSamples);

    // train
    const result = this.train(samples.train, {
      rate,
      log,
      iterations,
      error,
      cost
    });

    return {
      train: result,
      test: this.test(samples.test)
    }
  }
}

// Built-in cost functions
Trainer.cost = {
  // Eq. 9
  CROSS_ENTROPY(target, output) {
    let crossentropy = 0;
    for (let i in output)
      crossentropy -= (target[i] * Math.log(output[i]+1e-15)) + ((1-target[i]) * Math.log((1+1e-15)-output[i])); // +1e-15 is a tiny push away to avoid Math.log(0)
    return crossentropy;
  },
  MSE(target, output) {
    let mse = 0;
    for (let i in output)
      mse += Math.pow(target[i] - output[i], 2);
    return mse / output.length;
  },
  BINARY(target, output) {
    let misses = 0;
    for (let i in output)
      misses += Math.round(target[i] * 2) != Math.round(output[i] * 2);
    return misses;
  }
};

//+ Jonas Raoni Soares Silva
//@ http://jsfromhell.com/array/shuffle [v1.0]
function shuffle(o) { //v1.0
  for (let j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x);
  return o;
};

module.exports = Trainer;