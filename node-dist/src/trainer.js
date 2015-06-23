/*******************************************************************************************
                                        TRAINER
*******************************************************************************************/
var Trainer = (function () {
    function Trainer(network, options) {
        this.rate = .2;
        this.iterations = 100000;
        this.error = .005;
        options = options || {};
        this.network = network;
        this.rate = options.rate || .2;
        this.iterations = options.iterations || 100000;
        this.error = options.error || .005;
        this.cost = options.cost || Trainer.cost.CROSS_ENTROPY;
    }
    // trains any given set to a network
    Trainer.prototype.train = function (set, options) {
        var error = 1;
        var iterations = 0, bucketSize = 0;
        var abort_training = false;
        var input, output, target, currentRate;
        var start = Date.now();
        if (options) {
            if (options.shuffle) {
                //+ Jonas Raoni Soares Silva
                //@ http://jsfromhell.com/array/shuffle [v1.0]
                function shuffle(o) {
                    for (var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x)
                        ;
                    return o;
                }
                ;
            }
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
            if (options.customLog) {
                // for backward compatibility with code that used customLog
                console.log('Deprecated: use schedule instead of customLog');
                this.schedule = options.customLog;
            }
        }
        currentRate = this.rate;
        if (Array.isArray(this.rate)) {
            bucketSize = Math.floor(this.iterations / this.rate.length);
        }
        while (!abort_training && iterations < this.iterations && error > this.error) {
            error = 0;
            if (bucketSize > 0) {
                var currentBucket = Math.floor(iterations / bucketSize);
                currentRate = this.rate[currentBucket];
            }
            for (var train in set) {
                input = set[train].input;
                target = set[train].output;
                output = this.network.activate(input);
                this.network.propagate(currentRate, target);
                error += this.cost(target, output);
            }
            // check error
            iterations++;
            error /= set.length;
            if (options) {
                if (this.schedule && this.schedule.every && iterations % this.schedule.every == 0) {
                    abort_training = this.schedule.do({
                        error: error,
                        iterations: iterations,
                        rate: currentRate
                    });
                }
                else if (options.log && iterations % options.log == 0) {
                    console.log('iterations', iterations, 'error', error, 'rate', currentRate);
                }
                ;
                if (options.shuffle)
                    shuffle(set);
            }
        }
        var results = {
            error: error,
            iterations: iterations,
            time: Date.now() - start
        };
        return results;
    };
    // trains any given set to a network using a WebWorker
    Trainer.prototype.workerTrain = function (set, callback, options) {
        var that = this;
        var error = 1;
        var iterations = 0, bucketSize = 0;
        var input, output, target, currentRate;
        var length = set.length;
        var abort_training = false;
        var start = Date.now();
        if (options) {
            if (options.shuffle) {
                //+ Jonas Raoni Soares Silva
                //@ http://jsfromhell.com/array/shuffle [v1.0]
                function shuffle(o) {
                    for (var j, x, i = o.length; i; j = Math.floor(Math.random() * i), x = o[--i], o[i] = o[j], o[j] = x)
                        ;
                    return o;
                }
                ;
            }
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
            if (options.customLog)
                // for backward compatibility with code that used customLog
                console.log('Deprecated: use schedule instead of customLog');
            this.schedule = options.customLog;
        }
        // dynamic learning rate
        currentRate = this.rate;
        if (Array.isArray(this.rate)) {
            bucketSize = Math.floor(this.iterations / this.rate.length);
        }
        // create a worker
        var worker = this.network.worker();
        // activate the network
        function activateWorker(input) {
            worker.postMessage({
                action: "activate",
                input: input,
                memoryBuffer: that.network.optimized.memory
            }, [that.network.optimized.memory.buffer]);
        }
        // backpropagate the network
        function propagateWorker(target) {
            if (bucketSize > 0) {
                var currentBucket = Math.floor(iterations / bucketSize);
                currentRate = this.rate[currentBucket];
            }
            worker.postMessage({
                action: "propagate",
                target: target,
                rate: currentRate,
                memoryBuffer: that.network.optimized.memory
            }, [that.network.optimized.memory.buffer]);
        }
        // train the worker
        worker.onmessage = function (e) {
            // give control of the memory back to the network
            that.network.optimized.ownership(e.data.memoryBuffer);
            if (e.data.action == "propagate") {
                if (index >= length) {
                    index = 0;
                    iterations++;
                    error /= set.length;
                    // log
                    if (options) {
                        if (this.schedule && this.schedule.every && iterations % this.schedule.every == 0)
                            abort_training = this.schedule.do({
                                error: error,
                                iterations: iterations
                            });
                        else if (options.log && iterations % options.log == 0) {
                            console.log('iterations', iterations, 'error', error);
                        }
                        ;
                        if (options.shuffle)
                            shuffle(set);
                    }
                    if (!abort_training && iterations < that.iterations && error > that.error) {
                        activateWorker(set[index].input);
                    }
                    else {
                        // callback
                        callback({
                            error: error,
                            iterations: iterations,
                            time: Date.now() - start
                        });
                    }
                    error = 0;
                }
                else {
                    activateWorker(set[index].input);
                }
            }
            if (e.data.action == "activate") {
                error += that.cost(set[index].output, e.data.output);
                propagateWorker(set[index].output);
                index++;
            }
        };
        // kick it
        var index = 0;
        var iterations = 0;
        activateWorker(set[index].input);
    };
    // trains an XOR to the network
    Trainer.prototype.XOR = function (options) {
        if (this.network.inputs() != 2 || this.network.outputs() != 1)
            throw "Error: Incompatible network (2 inputs, 1 output)";
        var defaults = {
            iterations: 100000,
            log: false,
            shuffle: true,
            cost: Trainer.cost.MSE
        };
        if (options)
            for (var i in options)
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
    };
    // trains the network to pass a Distracted Sequence Recall test
    Trainer.prototype.DSR = function (options) {
        options = options || {};
        var targets = options.targets || [2, 4, 7, 8];
        var distractors = options.distractors || [3, 5, 6, 9];
        var prompts = options.prompts || [0, 1];
        var length = options.length || 24;
        var criterion = options.success || 0.95;
        var iterations = options.iterations || 100000;
        var rate = options.rate || .1;
        var log = options.log || 0;
        var schedule = options.schedule || {};
        var correct = 0;
        var i = 0;
        var success = 0;
        var trial = i = correct = j = success = 0, error = 1, symbols = targets.length + distractors.length + prompts.length;
        var noRepeat = function (range, avoid) {
            var number = Math.random() * range | 0;
            var used = false;
            for (var i in avoid)
                if (number == avoid[i])
                    used = true;
            return used ? noRepeat(range, avoid) : number;
        };
        var equal = function (prediction, output) {
            for (var i in prediction)
                if (Math.round(prediction[i]) != output[i])
                    return false;
            return true;
        };
        var start = Date.now();
        while (trial < iterations && (success < criterion || trial % 1000 != 0)) {
            // generate sequence
            var sequence = [], sequenceLength = length - prompts.length;
            for (i = 0; i < sequenceLength; i++) {
                var any = Math.random() * distractors.length | 0;
                sequence.push(distractors[any]);
            }
            var indexes = [], positions = [];
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
            var distractorsCorrect;
            var targetsCorrect = distractorsCorrect = 0;
            error = 0;
            for (i = 0; i < length; i++) {
                // generate input from sequence
                var input = [];
                for (j = 0; j < symbols; j++)
                    input[j] = 0;
                input[sequence[i]] = 1;
                // generate target output
                var output = [];
                for (j = 0; j < targets.length; j++)
                    output[j] = 0;
                if (i >= sequenceLength) {
                    var index = i - sequenceLength;
                    output[indexes[index]] = 1;
                }
                // check result
                var prediction = this.network.activate(input);
                if (equal(prediction, output))
                    if (i < sequenceLength)
                        distractorsCorrect++;
                    else
                        targetsCorrect++;
                else {
                    this.network.propagate(rate, output);
                }
                var delta = 0;
                for (var j in prediction)
                    delta += Math.pow(output[j] - prediction[j], 2);
                error += delta / this.network.outputs();
                if (distractorsCorrect + targetsCorrect == length)
                    correct++;
            }
            // calculate error
            if (trial % 1000 == 0)
                correct = 0;
            trial++;
            var divideError = trial % 1000;
            divideError = divideError == 0 ? 1000 : divideError;
            success = correct / divideError;
            error /= length;
            // log
            if (log && trial % log == 0)
                console.log("iterations:", trial, " success:", success, " correct:", correct, " time:", Date.now() - start, " error:", error);
            if (schedule.do && schedule.every && trial % schedule.every == 0) {
                schedule.do({
                    iterations: trial,
                    success: success,
                    error: error,
                    time: Date.now() - start,
                    correct: correct
                });
            }
        }
        return {
            iterations: trial,
            success: success,
            error: error,
            time: Date.now() - start
        };
    };
    // train the network to learn an Embeded Reber Grammar
    Trainer.prototype.ERG = function (options) {
        options = options || {};
        var iterations = options.iterations || 150000;
        var criterion = options.error || .05;
        var rate = options.rate || .1;
        var log = options.log || 500;
        // gramar node
        var Node = function () {
            this.paths = [];
        };
        Node.prototype = {
            connect: function (node, value) {
                this.paths.push({
                    node: node,
                    value: value
                });
                return this;
            },
            any: function () {
                if (this.paths.length == 0)
                    return false;
                var index = Math.random() * this.paths.length | 0;
                return this.paths[index];
            },
            test: function (value) {
                for (var i in this.paths)
                    if (this.paths[i].value == value)
                        return this.paths[i];
                return false;
            }
        };
        var reberGrammar = function () {
            // build a reber grammar
            var output = new Node();
            var n1 = (new Node()).connect(output, "E");
            var n2 = (new Node()).connect(n1, "S");
            var n3 = (new Node()).connect(n1, "V").connect(n2, "P");
            var n4 = (new Node()).connect(n2, "X");
            n4.connect(n4, "S");
            var n5 = (new Node()).connect(n3, "V");
            n5.connect(n5, "T");
            n2.connect(n5, "X");
            var n6 = (new Node()).connect(n4, "T").connect(n5, "P");
            var input = (new Node()).connect(n6, "B");
            return {
                input: input,
                output: output
            };
        };
        // build an embeded reber grammar
        var embededReberGrammar = function () {
            var reber1 = reberGrammar();
            var reber2 = reberGrammar();
            var output = new Node();
            var n1 = (new Node).connect(output, "E");
            reber1.output.connect(n1, "T");
            reber2.output.connect(n1, "P");
            var n2 = (new Node).connect(reber1.input, "P").connect(reber2.input, "T");
            var input = (new Node).connect(n2, "B");
            return {
                input: input,
                output: output
            };
        };
        // generate an ERG sequence
        var generate = function () {
            var node = embededReberGrammar().input;
            var next = node.any();
            var str = "";
            while (next) {
                str += next.value;
                next = next.node.any();
            }
            return str;
        };
        // test if a string matches an embeded reber grammar
        var test = function (str) {
            var node = embededReberGrammar().input;
            var i = 0;
            var ch = str.charAt(i);
            while (i < str.length) {
                var next = node.test(ch);
                if (!next)
                    return false;
                node = next.node;
                ch = str.charAt(++i);
            }
            return true;
        };
        // helper to check if the output and the target vectors match
        var different = function (array1, array2) {
            var max1 = 0;
            var i1 = -1;
            var max2 = 0;
            var i2 = -1;
            for (var i in array1) {
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
        var iteration = 0;
        var error = 1;
        var table = {
            "B": 0,
            "P": 1,
            "T": 2,
            "X": 3,
            "S": 4,
            "E": 5
        };
        var start = Date.now();
        while (iteration < iterations && error > criterion) {
            var i = 0;
            error = 0;
            // ERG sequence to learn
            var sequence = generate();
            // input
            var read = sequence.charAt(i);
            // target
            var predict = sequence.charAt(i + 1);
            while (i < sequence.length - 1) {
                var input = [];
                var target = [];
                for (var j = 0; j < 6; j++) {
                    input[j] = 0;
                    target[j] = 0;
                }
                input[table[read]] = 1;
                target[table[predict]] = 1;
                var output = this.network.activate(input);
                if (different(output, target))
                    this.network.propagate(rate, target);
                read = sequence.charAt(++i);
                predict = sequence.charAt(i + 1);
                var delta = 0;
                for (var k in output)
                    delta += Math.pow(target[k] - output[k], 2);
                delta /= output.length;
                error += delta;
            }
            error /= sequence.length;
            iteration++;
            if (iteration % log == 0) {
                console.log("iterations:", iteration, " time:", Date.now() - start, " error:", error);
            }
        }
        return {
            iterations: iteration,
            error: error,
            time: Date.now() - start,
            test: test,
            generate: generate
        };
    };
    return Trainer;
})();
exports.Trainer = Trainer;
var Trainer;
(function (Trainer) {
    Trainer.cost = {
        // Eq. 9
        CROSS_ENTROPY: function (target, output) {
            var crossentropy = 0;
            for (var i in output)
                crossentropy -= (target[i] * Math.log(output[i] + 1e-15)) + ((1 - target[i]) * Math.log((1 + 1e-15) - output[i]));
            return crossentropy;
        },
        MSE: function (target, output) {
            var mse = 0;
            for (var i in output)
                mse += Math.pow(target[i] - output[i], 2);
            return mse / output.length;
        }
    };
})(Trainer = exports.Trainer || (exports.Trainer = {}));

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy90cmFpbmVyLnRzIl0sIm5hbWVzIjpbIlRyYWluZXIiLCJUcmFpbmVyLmNvbnN0cnVjdG9yIiwiVHJhaW5lci50cmFpbiIsIlRyYWluZXIudHJhaW4uc2h1ZmZsZSIsIlRyYWluZXIud29ya2VyVHJhaW4iLCJUcmFpbmVyLndvcmtlclRyYWluLnNodWZmbGUiLCJUcmFpbmVyLndvcmtlclRyYWluLmFjdGl2YXRlV29ya2VyIiwiVHJhaW5lci53b3JrZXJUcmFpbi5wcm9wYWdhdGVXb3JrZXIiLCJUcmFpbmVyLlhPUiIsIlRyYWluZXIuRFNSIiwiVHJhaW5lci5FUkciXSwibWFwcGluZ3MiOiJBQUVBLEFBSUE7OzRGQUY0RjtJQUUvRSxPQUFPO0lBUWxCQSxTQVJXQSxPQUFPQSxDQVFOQSxPQUFvQkEsRUFBRUEsT0FBYUE7UUFOL0NDLFNBQUlBLEdBQVFBLEVBQUVBLENBQUNBO1FBQ2ZBLGVBQVVBLEdBQUdBLE1BQU1BLENBQUNBO1FBQ3BCQSxVQUFLQSxHQUFHQSxJQUFJQSxDQUFDQTtRQUtYQSxPQUFPQSxHQUFHQSxPQUFPQSxJQUFJQSxFQUFFQSxDQUFDQTtRQUN4QkEsSUFBSUEsQ0FBQ0EsT0FBT0EsR0FBR0EsT0FBT0EsQ0FBQ0E7UUFDdkJBLElBQUlBLENBQUNBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLElBQUlBLEVBQUVBLENBQUNBO1FBQy9CQSxJQUFJQSxDQUFDQSxVQUFVQSxHQUFHQSxPQUFPQSxDQUFDQSxVQUFVQSxJQUFJQSxNQUFNQSxDQUFDQTtRQUMvQ0EsSUFBSUEsQ0FBQ0EsS0FBS0EsR0FBR0EsT0FBT0EsQ0FBQ0EsS0FBS0EsSUFBSUEsSUFBSUEsQ0FBQUE7UUFDbENBLElBQUlBLENBQUNBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLElBQUlBLE9BQU9BLENBQUNBLElBQUlBLENBQUNBLGFBQWFBLENBQUNBO0lBQ3pEQSxDQUFDQTtJQUVERCxvQ0FBb0NBO0lBQ3BDQSx1QkFBS0EsR0FBTEEsVUFBTUEsR0FBR0EsRUFBRUEsT0FBT0E7UUFFaEJFLElBQUlBLEtBQUtBLEdBQUdBLENBQUNBLENBQUNBO1FBQ2RBLElBQUlBLFVBQVVBLEdBQUdBLENBQUNBLEVBQUVBLFVBQVVBLEdBQUdBLENBQUNBLENBQUNBO1FBQ25DQSxJQUFJQSxjQUFjQSxHQUFHQSxLQUFLQSxDQUFDQTtRQUMzQkEsSUFBSUEsS0FBS0EsRUFBRUEsTUFBTUEsRUFBRUEsTUFBTUEsRUFBRUEsV0FBV0EsQ0FBQ0E7UUFFdkNBLElBQUlBLEtBQUtBLEdBQUdBLElBQUlBLENBQUNBLEdBQUdBLEVBQUVBLENBQUNBO1FBRXZCQSxFQUFFQSxDQUFDQSxDQUFDQSxPQUFPQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUNaQSxFQUFFQSxDQUFDQSxDQUFDQSxPQUFPQSxDQUFDQSxPQUFPQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDcEJBLEFBRUFBLDRCQUY0QkE7Z0JBQzVCQSw4Q0FBOENBO3lCQUNyQ0EsT0FBT0EsQ0FBQ0EsQ0FBQ0E7b0JBQ2hCQyxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxFQUFFQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxNQUFNQSxFQUFFQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxJQUFJQSxDQUFDQSxLQUFLQSxDQUFDQSxJQUFJQSxDQUFDQSxNQUFNQSxFQUFFQSxHQUFHQSxDQUFDQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQTt3QkFBQ0EsQ0FBQ0E7b0JBQ3RHQSxNQUFNQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDWEEsQ0FBQ0E7Z0JBQUFELENBQUNBO1lBQ0pBLENBQUNBO1lBQ0RBLEVBQUVBLENBQUNBLENBQUNBLE9BQU9BLENBQUNBLFVBQVVBLENBQUNBO2dCQUNyQkEsSUFBSUEsQ0FBQ0EsVUFBVUEsR0FBR0EsT0FBT0EsQ0FBQ0EsVUFBVUEsQ0FBQ0E7WUFDdkNBLEVBQUVBLENBQUNBLENBQUNBLE9BQU9BLENBQUNBLEtBQUtBLENBQUNBO2dCQUNoQkEsSUFBSUEsQ0FBQ0EsS0FBS0EsR0FBR0EsT0FBT0EsQ0FBQ0EsS0FBS0EsQ0FBQ0E7WUFDN0JBLEVBQUVBLENBQUNBLENBQUNBLE9BQU9BLENBQUNBLElBQUlBLENBQUNBO2dCQUNmQSxJQUFJQSxDQUFDQSxJQUFJQSxHQUFHQSxPQUFPQSxDQUFDQSxJQUFJQSxDQUFDQTtZQUMzQkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsSUFBSUEsQ0FBQ0E7Z0JBQ2ZBLElBQUlBLENBQUNBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLENBQUNBO1lBQzNCQSxFQUFFQSxDQUFDQSxDQUFDQSxPQUFPQSxDQUFDQSxRQUFRQSxDQUFDQTtnQkFDbkJBLElBQUlBLENBQUNBLFFBQVFBLEdBQUdBLE9BQU9BLENBQUNBLFFBQVFBLENBQUNBO1lBQ25DQSxFQUFFQSxDQUFDQSxDQUFDQSxPQUFPQSxDQUFDQSxTQUFTQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDdEJBLEFBQ0FBLDJEQUQyREE7Z0JBQzNEQSxPQUFPQSxDQUFDQSxHQUFHQSxDQUFDQSwrQ0FBK0NBLENBQUNBLENBQUFBO2dCQUM1REEsSUFBSUEsQ0FBQ0EsUUFBUUEsR0FBR0EsT0FBT0EsQ0FBQ0EsU0FBU0EsQ0FBQ0E7WUFDcENBLENBQUNBO1FBQ0hBLENBQUNBO1FBRURBLFdBQVdBLEdBQUdBLElBQUlBLENBQUNBLElBQUlBLENBQUNBO1FBQ3hCQSxFQUFFQSxDQUFDQSxDQUFDQSxLQUFLQSxDQUFDQSxPQUFPQSxDQUFDQSxJQUFJQSxDQUFDQSxJQUFJQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUM3QkEsVUFBVUEsR0FBR0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsSUFBSUEsQ0FBQ0EsVUFBVUEsR0FBR0EsSUFBSUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0E7UUFDOURBLENBQUNBO1FBR0RBLE9BQU9BLENBQUNBLGNBQWNBLElBQUlBLFVBQVVBLEdBQUdBLElBQUlBLENBQUNBLFVBQVVBLElBQUlBLEtBQUtBLEdBQUdBLElBQUlBLENBQUNBLEtBQUtBLEVBQUVBLENBQUNBO1lBQzdFQSxLQUFLQSxHQUFHQSxDQUFDQSxDQUFDQTtZQUVWQSxFQUFFQSxDQUFDQSxDQUFDQSxVQUFVQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDbkJBLElBQUlBLGFBQWFBLEdBQUdBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLFVBQVVBLEdBQUdBLFVBQVVBLENBQUNBLENBQUNBO2dCQUN4REEsV0FBV0EsR0FBR0EsSUFBSUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsYUFBYUEsQ0FBQ0EsQ0FBQ0E7WUFDekNBLENBQUNBO1lBRURBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLEtBQUtBLElBQUlBLEdBQUdBLENBQUNBLENBQUNBLENBQUNBO2dCQUN0QkEsS0FBS0EsR0FBR0EsR0FBR0EsQ0FBQ0EsS0FBS0EsQ0FBQ0EsQ0FBQ0EsS0FBS0EsQ0FBQ0E7Z0JBQ3pCQSxNQUFNQSxHQUFHQSxHQUFHQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQSxNQUFNQSxDQUFDQTtnQkFFM0JBLE1BQU1BLEdBQUdBLElBQUlBLENBQUNBLE9BQU9BLENBQUNBLFFBQVFBLENBQUNBLEtBQUtBLENBQUNBLENBQUNBO2dCQUN0Q0EsSUFBSUEsQ0FBQ0EsT0FBT0EsQ0FBQ0EsU0FBU0EsQ0FBQ0EsV0FBV0EsRUFBRUEsTUFBTUEsQ0FBQ0EsQ0FBQ0E7Z0JBRTVDQSxLQUFLQSxJQUFJQSxJQUFJQSxDQUFDQSxJQUFJQSxDQUFDQSxNQUFNQSxFQUFFQSxNQUFNQSxDQUFDQSxDQUFDQTtZQUNyQ0EsQ0FBQ0E7WUFFREEsQUFDQUEsY0FEY0E7WUFDZEEsVUFBVUEsRUFBRUEsQ0FBQ0E7WUFDYkEsS0FBS0EsSUFBSUEsR0FBR0EsQ0FBQ0EsTUFBTUEsQ0FBQ0E7WUFFcEJBLEVBQUVBLENBQUNBLENBQUNBLE9BQU9BLENBQUNBLENBQUNBLENBQUNBO2dCQUNaQSxFQUFFQSxDQUFDQSxDQUFDQSxJQUFJQSxDQUFDQSxRQUFRQSxJQUFJQSxJQUFJQSxDQUFDQSxRQUFRQSxDQUFDQSxLQUFLQSxJQUFJQSxVQUFVQSxHQUFHQSxJQUFJQSxDQUFDQSxRQUFRQSxDQUFDQSxLQUFLQSxJQUFJQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtvQkFFbEZBLGNBQWNBLEdBQUdBLElBQUlBLENBQUNBLFFBQVFBLENBQUNBLEVBQUVBLENBQUNBO3dCQUNoQ0EsS0FBS0EsRUFBRUEsS0FBS0E7d0JBQ1pBLFVBQVVBLEVBQUVBLFVBQVVBO3dCQUN0QkEsSUFBSUEsRUFBRUEsV0FBV0E7cUJBQ2xCQSxDQUFDQSxDQUFDQTtnQkFFTEEsQ0FBQ0E7Z0JBQUNBLElBQUlBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLE9BQU9BLENBQUNBLEdBQUdBLElBQUlBLFVBQVVBLEdBQUdBLE9BQU9BLENBQUNBLEdBQUdBLElBQUlBLENBQUNBLENBQUNBLENBQUNBLENBQUNBO29CQUN4REEsT0FBT0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsWUFBWUEsRUFBRUEsVUFBVUEsRUFBRUEsT0FBT0EsRUFBRUEsS0FBS0EsRUFBRUEsTUFBTUEsRUFBRUEsV0FBV0EsQ0FBQ0EsQ0FBQ0E7Z0JBQzdFQSxDQUFDQTtnQkFBQUEsQ0FBQ0E7Z0JBQ0ZBLEVBQUVBLENBQUNBLENBQUNBLE9BQU9BLENBQUNBLE9BQU9BLENBQUNBO29CQUNsQkEsT0FBT0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7WUFDakJBLENBQUNBO1FBQ0hBLENBQUNBO1FBRURBLElBQUlBLE9BQU9BLEdBQUdBO1lBQ1pBLEtBQUtBLEVBQUVBLEtBQUtBO1lBQ1pBLFVBQVVBLEVBQUVBLFVBQVVBO1lBQ3RCQSxJQUFJQSxFQUFFQSxJQUFJQSxDQUFDQSxHQUFHQSxFQUFFQSxHQUFHQSxLQUFLQTtTQUN6QkEsQ0FBQUE7UUFFREEsTUFBTUEsQ0FBQ0EsT0FBT0EsQ0FBQ0E7SUFDakJBLENBQUNBO0lBRURGLHNEQUFzREE7SUFDdERBLDZCQUFXQSxHQUFYQSxVQUFZQSxHQUFHQSxFQUFFQSxRQUFRQSxFQUFFQSxPQUFPQTtRQUVoQ0ksSUFBSUEsSUFBSUEsR0FBR0EsSUFBSUEsQ0FBQ0E7UUFDaEJBLElBQUlBLEtBQUtBLEdBQUdBLENBQUNBLENBQUNBO1FBQ2RBLElBQUlBLFVBQVVBLEdBQUdBLENBQUNBLEVBQUVBLFVBQVVBLEdBQUdBLENBQUNBLENBQUNBO1FBQ25DQSxJQUFJQSxLQUFLQSxFQUFFQSxNQUFNQSxFQUFFQSxNQUFNQSxFQUFFQSxXQUFXQSxDQUFDQTtRQUN2Q0EsSUFBSUEsTUFBTUEsR0FBR0EsR0FBR0EsQ0FBQ0EsTUFBTUEsQ0FBQ0E7UUFDeEJBLElBQUlBLGNBQWNBLEdBQUdBLEtBQUtBLENBQUNBO1FBRTNCQSxJQUFJQSxLQUFLQSxHQUFHQSxJQUFJQSxDQUFDQSxHQUFHQSxFQUFFQSxDQUFDQTtRQUV2QkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDWkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ3BCQSxBQUVBQSw0QkFGNEJBO2dCQUM1QkEsOENBQThDQTt5QkFDckNBLE9BQU9BLENBQUNBLENBQUNBO29CQUNoQkMsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsTUFBTUEsRUFBRUEsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsSUFBSUEsQ0FBQ0EsTUFBTUEsRUFBRUEsR0FDMURBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBO3dCQUFDQSxDQUFDQTtvQkFDekNBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBO2dCQUNYQSxDQUFDQTtnQkFBQUQsQ0FBQ0E7WUFDSkEsQ0FBQ0E7WUFDREEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsVUFBVUEsQ0FBQ0E7Z0JBQ3JCQSxJQUFJQSxDQUFDQSxVQUFVQSxHQUFHQSxPQUFPQSxDQUFDQSxVQUFVQSxDQUFDQTtZQUN2Q0EsRUFBRUEsQ0FBQ0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsS0FBS0EsQ0FBQ0E7Z0JBQ2hCQSxJQUFJQSxDQUFDQSxLQUFLQSxHQUFHQSxPQUFPQSxDQUFDQSxLQUFLQSxDQUFDQTtZQUM3QkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsSUFBSUEsQ0FBQ0E7Z0JBQ2ZBLElBQUlBLENBQUNBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLENBQUNBO1lBQzNCQSxFQUFFQSxDQUFDQSxDQUFDQSxPQUFPQSxDQUFDQSxJQUFJQSxDQUFDQTtnQkFDZkEsSUFBSUEsQ0FBQ0EsSUFBSUEsR0FBR0EsT0FBT0EsQ0FBQ0EsSUFBSUEsQ0FBQ0E7WUFDM0JBLEVBQUVBLENBQUNBLENBQUNBLE9BQU9BLENBQUNBLFFBQVFBLENBQUNBO2dCQUNuQkEsSUFBSUEsQ0FBQ0EsUUFBUUEsR0FBR0EsT0FBT0EsQ0FBQ0EsUUFBUUEsQ0FBQ0E7WUFDbkNBLEVBQUVBLENBQUNBLENBQUNBLE9BQU9BLENBQUNBLFNBQVNBLENBQUNBO2dCQUNwQkEsQUFDQUEsMkRBRDJEQTtnQkFDM0RBLE9BQU9BLENBQUNBLEdBQUdBLENBQUNBLCtDQUErQ0EsQ0FBQ0EsQ0FBQUE7WUFDOURBLElBQUlBLENBQUNBLFFBQVFBLEdBQUdBLE9BQU9BLENBQUNBLFNBQVNBLENBQUNBO1FBQ3BDQSxDQUFDQTtRQUVEQSxBQUNBQSx3QkFEd0JBO1FBQ3hCQSxXQUFXQSxHQUFHQSxJQUFJQSxDQUFDQSxJQUFJQSxDQUFDQTtRQUN4QkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsS0FBS0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsSUFBSUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFDN0JBLFVBQVVBLEdBQUdBLElBQUlBLENBQUNBLEtBQUtBLENBQUNBLElBQUlBLENBQUNBLFVBQVVBLEdBQUdBLElBQUlBLENBQUNBLElBQUlBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBO1FBQzlEQSxDQUFDQTtRQUVEQSxBQUNBQSxrQkFEa0JBO1lBQ2RBLE1BQU1BLEdBQUdBLElBQUlBLENBQUNBLE9BQU9BLENBQUNBLE1BQU1BLEVBQUVBLENBQUNBO1FBRW5DQSxBQUNBQSx1QkFEdUJBO2lCQUNkQSxjQUFjQSxDQUFDQSxLQUFLQTtZQUMzQkUsTUFBTUEsQ0FBQ0EsV0FBV0EsQ0FBQ0E7Z0JBQ2pCQSxNQUFNQSxFQUFFQSxVQUFVQTtnQkFDbEJBLEtBQUtBLEVBQUVBLEtBQUtBO2dCQUNaQSxZQUFZQSxFQUFFQSxJQUFJQSxDQUFDQSxPQUFPQSxDQUFDQSxTQUFTQSxDQUFDQSxNQUFNQTthQUM1Q0EsRUFBRUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsT0FBT0EsQ0FBQ0EsU0FBU0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7UUFDN0NBLENBQUNBO1FBRURGLEFBQ0FBLDRCQUQ0QkE7aUJBQ25CQSxlQUFlQSxDQUFDQSxNQUFNQTtZQUM3QkcsRUFBRUEsQ0FBQ0EsQ0FBQ0EsVUFBVUEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ25CQSxJQUFJQSxhQUFhQSxHQUFHQSxJQUFJQSxDQUFDQSxLQUFLQSxDQUFDQSxVQUFVQSxHQUFHQSxVQUFVQSxDQUFDQSxDQUFDQTtnQkFDeERBLFdBQVdBLEdBQUdBLElBQUlBLENBQUNBLElBQUlBLENBQUNBLGFBQWFBLENBQUNBLENBQUNBO1lBQ3pDQSxDQUFDQTtZQUNEQSxNQUFNQSxDQUFDQSxXQUFXQSxDQUFDQTtnQkFDakJBLE1BQU1BLEVBQUVBLFdBQVdBO2dCQUNuQkEsTUFBTUEsRUFBRUEsTUFBTUE7Z0JBQ2RBLElBQUlBLEVBQUVBLFdBQVdBO2dCQUNqQkEsWUFBWUEsRUFBRUEsSUFBSUEsQ0FBQ0EsT0FBT0EsQ0FBQ0EsU0FBU0EsQ0FBQ0EsTUFBTUE7YUFDNUNBLEVBQUVBLENBQUNBLElBQUlBLENBQUNBLE9BQU9BLENBQUNBLFNBQVNBLENBQUNBLE1BQU1BLENBQUNBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBO1FBQzdDQSxDQUFDQTtRQUVESCxBQUNBQSxtQkFEbUJBO1FBQ25CQSxNQUFNQSxDQUFDQSxTQUFTQSxHQUFHQSxVQUFTQSxDQUFDQTtZQUMzQixBQUNBLGlEQURpRDtZQUNqRCxJQUFJLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUMsQ0FBQztZQUV0RCxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLE1BQU0sSUFBSSxXQUFXLENBQUMsQ0FBQyxDQUFDO2dCQUNqQyxFQUFFLENBQUMsQ0FBQyxLQUFLLElBQUksTUFBTSxDQUFDLENBQUMsQ0FBQztvQkFDcEIsS0FBSyxHQUFHLENBQUMsQ0FBQztvQkFDVixVQUFVLEVBQUUsQ0FBQztvQkFDYixLQUFLLElBQUksR0FBRyxDQUFDLE1BQU0sQ0FBQztvQkFFcEIsQUFDQSxNQURNO29CQUNOLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7d0JBQ1osRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFFBQVEsSUFBSSxJQUFJLENBQUMsUUFBUSxDQUFDLEtBQUssSUFBSSxVQUFVLEdBQUcsSUFBSSxDQUFDLFFBQVEsQ0FBQyxLQUFLLElBQUksQ0FBQyxDQUFDOzRCQUNoRixjQUFjLEdBQUcsSUFBSSxDQUFDLFFBQVEsQ0FBQyxFQUFFLENBQUM7Z0NBQ2hDLEtBQUssRUFBRSxLQUFLO2dDQUNaLFVBQVUsRUFBRSxVQUFVOzZCQUN2QixDQUFDLENBQUM7d0JBQ0wsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxHQUFHLElBQUksVUFBVSxHQUFHLE9BQU8sQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQzs0QkFDdEQsT0FBTyxDQUFDLEdBQUcsQ0FBQyxZQUFZLEVBQUUsVUFBVSxFQUFFLE9BQU8sRUFBRSxLQUFLLENBQUMsQ0FBQzt3QkFDeEQsQ0FBQzt3QkFBQSxDQUFDO3dCQUNGLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUM7NEJBQ2xCLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQztvQkFDakIsQ0FBQztvQkFFRCxFQUFFLENBQUMsQ0FBQyxDQUFDLGNBQWMsSUFBSSxVQUFVLEdBQUcsSUFBSSxDQUFDLFVBQVUsSUFBSSxLQUFLLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7d0JBQzFFLGNBQWMsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7b0JBQ25DLENBQUM7b0JBQUMsSUFBSSxDQUFDLENBQUM7d0JBQ04sQUFDQSxXQURXO3dCQUNYLFFBQVEsQ0FBQzs0QkFDUCxLQUFLLEVBQUUsS0FBSzs0QkFDWixVQUFVLEVBQUUsVUFBVTs0QkFDdEIsSUFBSSxFQUFFLElBQUksQ0FBQyxHQUFHLEVBQUUsR0FBRyxLQUFLO3lCQUN6QixDQUFDLENBQUE7b0JBQ0osQ0FBQztvQkFDRCxLQUFLLEdBQUcsQ0FBQyxDQUFDO2dCQUNaLENBQUM7Z0JBQUMsSUFBSSxDQUFDLENBQUM7b0JBQ04sY0FBYyxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQztnQkFDbkMsQ0FBQztZQUNILENBQUM7WUFFRCxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLE1BQU0sSUFBSSxVQUFVLENBQUMsQ0FBQyxDQUFDO2dCQUNoQyxLQUFLLElBQUksSUFBSSxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7Z0JBQ3JELGVBQWUsQ0FBQyxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUM7Z0JBQ25DLEtBQUssRUFBRSxDQUFDO1lBQ1YsQ0FBQztRQUNILENBQUMsQ0FBQUE7UUFFREEsQUFDQUEsVUFEVUE7WUFDTkEsS0FBS0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7UUFDZEEsSUFBSUEsVUFBVUEsR0FBR0EsQ0FBQ0EsQ0FBQ0E7UUFDbkJBLGNBQWNBLENBQUNBLEdBQUdBLENBQUNBLEtBQUtBLENBQUNBLENBQUNBLEtBQUtBLENBQUNBLENBQUNBO0lBQ25DQSxDQUFDQTtJQUVESiwrQkFBK0JBO0lBQy9CQSxxQkFBR0EsR0FBSEEsVUFBSUEsT0FBT0E7UUFFVFEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsSUFBSUEsQ0FBQ0EsT0FBT0EsQ0FBQ0EsTUFBTUEsRUFBRUEsSUFBSUEsQ0FBQ0EsSUFBSUEsSUFBSUEsQ0FBQ0EsT0FBT0EsQ0FBQ0EsT0FBT0EsRUFBRUEsSUFBSUEsQ0FBQ0EsQ0FBQ0E7WUFDNURBLE1BQU1BLGtEQUFrREEsQ0FBQ0E7UUFFM0RBLElBQUlBLFFBQVFBLEdBQUdBO1lBQ2JBLFVBQVVBLEVBQUVBLE1BQU1BO1lBQ2xCQSxHQUFHQSxFQUFFQSxLQUFLQTtZQUNWQSxPQUFPQSxFQUFFQSxJQUFJQTtZQUNiQSxJQUFJQSxFQUFFQSxPQUFPQSxDQUFDQSxJQUFJQSxDQUFDQSxHQUFHQTtTQUN2QkEsQ0FBQUE7UUFFREEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsT0FBT0EsQ0FBQ0E7WUFDVkEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsSUFBSUEsT0FBT0EsQ0FBQ0E7Z0JBQ3BCQSxRQUFRQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxPQUFPQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtRQUU3QkEsTUFBTUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsQ0FBQ0E7WUFDakJBLEtBQUtBLEVBQUVBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBQ2JBLE1BQU1BLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO1NBQ1pBLEVBQUVBO1lBQ0NBLEtBQUtBLEVBQUVBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBQ2JBLE1BQU1BLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO1NBQ1pBLEVBQUVBO1lBQ0RBLEtBQUtBLEVBQUVBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBQ2JBLE1BQU1BLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO1NBQ1pBLEVBQUVBO1lBQ0RBLEtBQUtBLEVBQUVBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBQ2JBLE1BQU1BLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO1NBQ1pBLENBQUNBLEVBQUVBLFFBQVFBLENBQUNBLENBQUNBO0lBQ2xCQSxDQUFDQTtJQUVEUiwrREFBK0RBO0lBQy9EQSxxQkFBR0EsR0FBSEEsVUFBSUEsT0FBT0E7UUFDVFMsT0FBT0EsR0FBR0EsT0FBT0EsSUFBSUEsRUFBRUEsQ0FBQ0E7UUFFeEJBLElBQUlBLE9BQU9BLEdBQUdBLE9BQU9BLENBQUNBLE9BQU9BLElBQUlBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLEVBQUVBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO1FBQzlDQSxJQUFJQSxXQUFXQSxHQUFHQSxPQUFPQSxDQUFDQSxXQUFXQSxJQUFJQSxDQUFDQSxDQUFDQSxFQUFFQSxDQUFDQSxFQUFFQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxDQUFDQTtRQUN0REEsSUFBSUEsT0FBT0EsR0FBR0EsT0FBT0EsQ0FBQ0EsT0FBT0EsSUFBSUEsQ0FBQ0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7UUFDeENBLElBQUlBLE1BQU1BLEdBQUdBLE9BQU9BLENBQUNBLE1BQU1BLElBQUlBLEVBQUVBLENBQUNBO1FBQ2xDQSxJQUFJQSxTQUFTQSxHQUFHQSxPQUFPQSxDQUFDQSxP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QyxDQUFDLENBQUM7WUFDNUMsTUFBTSxDQUFDLEdBQUcsR0FBRyxNQUFNLENBQUMsTUFBTSxDQUFDO1FBQzdCLENBQUM7S0FDRkEsQ0FBQUE7QUFDSEEsQ0FBQ0EsRUF0QmEsT0FBTyxHQUFQLGVBQU8sS0FBUCxlQUFPLFFBc0JwQiIsImZpbGUiOiJzcmMvdHJhaW5lci5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBuZXQgPSByZXF1aXJlKCcuL25ldHdvcmsnKTtcblxuLyoqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKipcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBUUkFJTkVSXG4qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqL1xuXG5leHBvcnQgY2xhc3MgVHJhaW5lciB7XG4gIG5ldHdvcms6IG5ldC5OZXR3b3JrO1xuICByYXRlOiBhbnkgPSAuMjtcbiAgaXRlcmF0aW9ucyA9IDEwMDAwMDtcbiAgZXJyb3IgPSAuMDA1O1xuICBjb3N0OiBUcmFpbmVyLklUcmFpbmVyQ29zdEZuO1xuICBzY2hlZHVsZTogYW55O1xuXG4gIGNvbnN0cnVjdG9yKG5ldHdvcms6IG5ldC5OZXR3b3JrLCBvcHRpb25zPzogYW55KSB7XG4gICAgb3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG4gICAgdGhpcy5uZXR3b3JrID0gbmV0d29yaztcbiAgICB0aGlzLnJhdGUgPSBvcHRpb25zLnJhdGUgfHwgLjI7XG4gICAgdGhpcy5pdGVyYXRpb25zID0gb3B0aW9ucy5pdGVyYXRpb25zIHx8IDEwMDAwMDtcbiAgICB0aGlzLmVycm9yID0gb3B0aW9ucy5lcnJvciB8fCAuMDA1XG4gICAgdGhpcy5jb3N0ID0gb3B0aW9ucy5jb3N0IHx8IFRyYWluZXIuY29zdC5DUk9TU19FTlRST1BZO1xuICB9XG5cbiAgLy8gdHJhaW5zIGFueSBnaXZlbiBzZXQgdG8gYSBuZXR3b3JrXG4gIHRyYWluKHNldCwgb3B0aW9ucykge1xuXG4gICAgdmFyIGVycm9yID0gMTtcbiAgICB2YXIgaXRlcmF0aW9ucyA9IDAsIGJ1Y2tldFNpemUgPSAwO1xuICAgIHZhciBhYm9ydF90cmFpbmluZyA9IGZhbHNlO1xuICAgIHZhciBpbnB1dCwgb3V0cHV0LCB0YXJnZXQsIGN1cnJlbnRSYXRlO1xuXG4gICAgdmFyIHN0YXJ0ID0gRGF0ZS5ub3coKTtcblxuICAgIGlmIChvcHRpb25zKSB7XG4gICAgICBpZiAob3B0aW9ucy5zaHVmZmxlKSB7XG4gICAgICAgIC8vKyBKb25hcyBSYW9uaSBTb2FyZXMgU2lsdmFcbiAgICAgICAgLy9AIGh0dHA6Ly9qc2Zyb21oZWxsLmNvbS9hcnJheS9zaHVmZmxlIFt2MS4wXVxuICAgICAgICBmdW5jdGlvbiBzaHVmZmxlKG8pIHsgLy92MS4wXG4gICAgICAgICAgZm9yICh2YXIgaiwgeCwgaSA9IG8ubGVuZ3RoOyBpOyBqID0gTWF0aC5mbG9vcihNYXRoLnJhbmRvbSgpICogaSksIHggPSBvWy0taV0sIG9baV0gPSBvW2pdLCBvW2pdID0geCk7XG4gICAgICAgICAgcmV0dXJuIG87XG4gICAgICAgIH07XG4gICAgICB9XG4gICAgICBpZiAob3B0aW9ucy5pdGVyYXRpb25zKVxuICAgICAgICB0aGlzLml0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnM7XG4gICAgICBpZiAob3B0aW9ucy5lcnJvcilcbiAgICAgICAgdGhpcy5lcnJvciA9IG9wdGlvbnMuZXJyb3I7XG4gICAgICBpZiAob3B0aW9ucy5yYXRlKVxuICAgICAgICB0aGlzLnJhdGUgPSBvcHRpb25zLnJhdGU7XG4gICAgICBpZiAob3B0aW9ucy5jb3N0KVxuICAgICAgICB0aGlzLmNvc3QgPSBvcHRpb25zLmNvc3Q7XG4gICAgICBpZiAob3B0aW9ucy5zY2hlZHVsZSlcbiAgICAgICAgdGhpcy5zY2hlZHVsZSA9IG9wdGlvbnMuc2NoZWR1bGU7XG4gICAgICBpZiAob3B0aW9ucy5jdXN0b21Mb2cpIHtcbiAgICAgICAgLy8gZm9yIGJhY2t3YXJkIGNvbXBhdGliaWxpdHkgd2l0aCBjb2RlIHRoYXQgdXNlZCBjdXN0b21Mb2dcbiAgICAgICAgY29uc29sZS5sb2coJ0RlcHJlY2F0ZWQ6IHVzZSBzY2hlZHVsZSBpbnN0ZWFkIG9mIGN1c3RvbUxvZycpXG4gICAgICAgIHRoaXMuc2NoZWR1bGUgPSBvcHRpb25zLmN1c3RvbUxvZztcbiAgICAgIH1cbiAgICB9XG5cbiAgICBjdXJyZW50UmF0ZSA9IHRoaXMucmF0ZTtcbiAgICBpZiAoQXJyYXkuaXNBcnJheSh0aGlzLnJhdGUpKSB7XG4gICAgICBidWNrZXRTaXplID0gTWF0aC5mbG9vcih0aGlzLml0ZXJhdGlvbnMgLyB0aGlzLnJhdGUubGVuZ3RoKTtcbiAgICB9XG5cblxuICAgIHdoaWxlICghYWJvcnRfdHJhaW5pbmcgJiYgaXRlcmF0aW9ucyA8IHRoaXMuaXRlcmF0aW9ucyAmJiBlcnJvciA+IHRoaXMuZXJyb3IpIHtcbiAgICAgIGVycm9yID0gMDtcblxuICAgICAgaWYgKGJ1Y2tldFNpemUgPiAwKSB7XG4gICAgICAgIHZhciBjdXJyZW50QnVja2V0ID0gTWF0aC5mbG9vcihpdGVyYXRpb25zIC8gYnVja2V0U2l6ZSk7XG4gICAgICAgIGN1cnJlbnRSYXRlID0gdGhpcy5yYXRlW2N1cnJlbnRCdWNrZXRdO1xuICAgICAgfVxuXG4gICAgICBmb3IgKHZhciB0cmFpbiBpbiBzZXQpIHtcbiAgICAgICAgaW5wdXQgPSBzZXRbdHJhaW5dLmlucHV0O1xuICAgICAgICB0YXJnZXQgPSBzZXRbdHJhaW5dLm91dHB1dDtcblxuICAgICAgICBvdXRwdXQgPSB0aGlzLm5ldHdvcmsuYWN0aXZhdGUoaW5wdXQpO1xuICAgICAgICB0aGlzLm5ldHdvcmsucHJvcGFnYXRlKGN1cnJlbnRSYXRlLCB0YXJnZXQpO1xuXG4gICAgICAgIGVycm9yICs9IHRoaXMuY29zdCh0YXJnZXQsIG91dHB1dCk7XG4gICAgICB9XG5cbiAgICAgIC8vIGNoZWNrIGVycm9yXG4gICAgICBpdGVyYXRpb25zKys7XG4gICAgICBlcnJvciAvPSBzZXQubGVuZ3RoO1xuXG4gICAgICBpZiAob3B0aW9ucykge1xuICAgICAgICBpZiAodGhpcy5zY2hlZHVsZSAmJiB0aGlzLnNjaGVkdWxlLmV2ZXJ5ICYmIGl0ZXJhdGlvbnMgJSB0aGlzLnNjaGVkdWxlLmV2ZXJ5ID09IDApIHtcblxuICAgICAgICAgIGFib3J0X3RyYWluaW5nID0gdGhpcy5zY2hlZHVsZS5kbyh7XG4gICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgICAgICAgcmF0ZTogY3VycmVudFJhdGVcbiAgICAgICAgICB9KTtcbiAgICAgXG4gICAgICAgIH0gZWxzZSBpZiAob3B0aW9ucy5sb2cgJiYgaXRlcmF0aW9ucyAlIG9wdGlvbnMubG9nID09IDApIHtcbiAgICAgICAgICBjb25zb2xlLmxvZygnaXRlcmF0aW9ucycsIGl0ZXJhdGlvbnMsICdlcnJvcicsIGVycm9yLCAncmF0ZScsIGN1cnJlbnRSYXRlKTtcbiAgICAgICAgfTtcbiAgICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSlcbiAgICAgICAgICBzaHVmZmxlKHNldCk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdmFyIHJlc3VsdHMgPSB7XG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0XG4gICAgfVxuXG4gICAgcmV0dXJuIHJlc3VsdHM7XG4gIH1cblxuICAvLyB0cmFpbnMgYW55IGdpdmVuIHNldCB0byBhIG5ldHdvcmsgdXNpbmcgYSBXZWJXb3JrZXJcbiAgd29ya2VyVHJhaW4oc2V0LCBjYWxsYmFjaywgb3B0aW9ucykge1xuXG4gICAgdmFyIHRoYXQgPSB0aGlzO1xuICAgIHZhciBlcnJvciA9IDE7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSAwLCBidWNrZXRTaXplID0gMDtcbiAgICB2YXIgaW5wdXQsIG91dHB1dCwgdGFyZ2V0LCBjdXJyZW50UmF0ZTtcbiAgICB2YXIgbGVuZ3RoID0gc2V0Lmxlbmd0aDtcbiAgICB2YXIgYWJvcnRfdHJhaW5pbmcgPSBmYWxzZTtcblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICBpZiAob3B0aW9ucykge1xuICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSkge1xuICAgICAgICAvLysgSm9uYXMgUmFvbmkgU29hcmVzIFNpbHZhXG4gICAgICAgIC8vQCBodHRwOi8vanNmcm9taGVsbC5jb20vYXJyYXkvc2h1ZmZsZSBbdjEuMF1cbiAgICAgICAgZnVuY3Rpb24gc2h1ZmZsZShvKSB7IC8vdjEuMFxuICAgICAgICAgIGZvciAodmFyIGosIHgsIGkgPSBvLmxlbmd0aDsgaTsgaiA9IE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqXG4gICAgICAgICAgICBpKSwgeCA9IG9bLS1pXSwgb1tpXSA9IG9bal0sIG9bal0gPSB4KTtcbiAgICAgICAgICByZXR1cm4gbztcbiAgICAgICAgfTtcbiAgICAgIH1cbiAgICAgIGlmIChvcHRpb25zLml0ZXJhdGlvbnMpXG4gICAgICAgIHRoaXMuaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucztcbiAgICAgIGlmIChvcHRpb25zLmVycm9yKVxuICAgICAgICB0aGlzLmVycm9yID0gb3B0aW9ucy5lcnJvcjtcbiAgICAgIGlmIChvcHRpb25zLnJhdGUpXG4gICAgICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZTtcbiAgICAgIGlmIChvcHRpb25zLmNvc3QpXG4gICAgICAgIHRoaXMuY29zdCA9IG9wdGlvbnMuY29zdDtcbiAgICAgIGlmIChvcHRpb25zLnNjaGVkdWxlKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5zY2hlZHVsZTtcbiAgICAgIGlmIChvcHRpb25zLmN1c3RvbUxvZylcbiAgICAgICAgLy8gZm9yIGJhY2t3YXJkIGNvbXBhdGliaWxpdHkgd2l0aCBjb2RlIHRoYXQgdXNlZCBjdXN0b21Mb2dcbiAgICAgICAgY29uc29sZS5sb2coJ0RlcHJlY2F0ZWQ6IHVzZSBzY2hlZHVsZSBpbnN0ZWFkIG9mIGN1c3RvbUxvZycpXG4gICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5jdXN0b21Mb2c7XG4gICAgfVxuXG4gICAgLy8gZHluYW1pYyBsZWFybmluZyByYXRlXG4gICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGU7XG4gICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5yYXRlKSkge1xuICAgICAgYnVja2V0U2l6ZSA9IE1hdGguZmxvb3IodGhpcy5pdGVyYXRpb25zIC8gdGhpcy5yYXRlLmxlbmd0aCk7XG4gICAgfVxuXG4gICAgLy8gY3JlYXRlIGEgd29ya2VyXG4gICAgdmFyIHdvcmtlciA9IHRoaXMubmV0d29yay53b3JrZXIoKTtcblxuICAgIC8vIGFjdGl2YXRlIHRoZSBuZXR3b3JrXG4gICAgZnVuY3Rpb24gYWN0aXZhdGVXb3JrZXIoaW5wdXQpIHtcbiAgICAgIHdvcmtlci5wb3N0TWVzc2FnZSh7XG4gICAgICAgIGFjdGlvbjogXCJhY3RpdmF0ZVwiLFxuICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgIG1lbW9yeUJ1ZmZlcjogdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnlcbiAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyBiYWNrcHJvcGFnYXRlIHRoZSBuZXR3b3JrXG4gICAgZnVuY3Rpb24gcHJvcGFnYXRlV29ya2VyKHRhcmdldCkge1xuICAgICAgaWYgKGJ1Y2tldFNpemUgPiAwKSB7XG4gICAgICAgIHZhciBjdXJyZW50QnVja2V0ID0gTWF0aC5mbG9vcihpdGVyYXRpb25zIC8gYnVja2V0U2l6ZSk7XG4gICAgICAgIGN1cnJlbnRSYXRlID0gdGhpcy5yYXRlW2N1cnJlbnRCdWNrZXRdO1xuICAgICAgfVxuICAgICAgd29ya2VyLnBvc3RNZXNzYWdlKHtcbiAgICAgICAgYWN0aW9uOiBcInByb3BhZ2F0ZVwiLFxuICAgICAgICB0YXJnZXQ6IHRhcmdldCxcbiAgICAgICAgcmF0ZTogY3VycmVudFJhdGUsXG4gICAgICAgIG1lbW9yeUJ1ZmZlcjogdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnlcbiAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyB0cmFpbiB0aGUgd29ya2VyXG4gICAgd29ya2VyLm9ubWVzc2FnZSA9IGZ1bmN0aW9uKGUpIHtcbiAgICAgIC8vIGdpdmUgY29udHJvbCBvZiB0aGUgbWVtb3J5IGJhY2sgdG8gdGhlIG5ldHdvcmtcbiAgICAgIHRoYXQubmV0d29yay5vcHRpbWl6ZWQub3duZXJzaGlwKGUuZGF0YS5tZW1vcnlCdWZmZXIpO1xuXG4gICAgICBpZiAoZS5kYXRhLmFjdGlvbiA9PSBcInByb3BhZ2F0ZVwiKSB7XG4gICAgICAgIGlmIChpbmRleCA+PSBsZW5ndGgpIHtcbiAgICAgICAgICBpbmRleCA9IDA7XG4gICAgICAgICAgaXRlcmF0aW9ucysrO1xuICAgICAgICAgIGVycm9yIC89IHNldC5sZW5ndGg7XG5cbiAgICAgICAgICAvLyBsb2dcbiAgICAgICAgICBpZiAob3B0aW9ucykge1xuICAgICAgICAgICAgaWYgKHRoaXMuc2NoZWR1bGUgJiYgdGhpcy5zY2hlZHVsZS5ldmVyeSAmJiBpdGVyYXRpb25zICUgdGhpcy5zY2hlZHVsZS5ldmVyeSA9PSAwKVxuICAgICAgICAgICAgICBhYm9ydF90cmFpbmluZyA9IHRoaXMuc2NoZWR1bGUuZG8oe1xuICAgICAgICAgICAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zXG4gICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgZWxzZSBpZiAob3B0aW9ucy5sb2cgJiYgaXRlcmF0aW9ucyAlIG9wdGlvbnMubG9nID09IDApIHtcbiAgICAgICAgICAgICAgY29uc29sZS5sb2coJ2l0ZXJhdGlvbnMnLCBpdGVyYXRpb25zLCAnZXJyb3InLCBlcnJvcik7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSlcbiAgICAgICAgICAgICAgc2h1ZmZsZShzZXQpO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIGlmICghYWJvcnRfdHJhaW5pbmcgJiYgaXRlcmF0aW9ucyA8IHRoYXQuaXRlcmF0aW9ucyAmJiBlcnJvciA+IHRoYXQuZXJyb3IpIHtcbiAgICAgICAgICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAvLyBjYWxsYmFja1xuICAgICAgICAgICAgY2FsbGJhY2soe1xuICAgICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICAgIGl0ZXJhdGlvbnM6IGl0ZXJhdGlvbnMsXG4gICAgICAgICAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydFxuICAgICAgICAgICAgfSlcbiAgICAgICAgICB9XG4gICAgICAgICAgZXJyb3IgPSAwO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChlLmRhdGEuYWN0aW9uID09IFwiYWN0aXZhdGVcIikge1xuICAgICAgICBlcnJvciArPSB0aGF0LmNvc3Qoc2V0W2luZGV4XS5vdXRwdXQsIGUuZGF0YS5vdXRwdXQpO1xuICAgICAgICBwcm9wYWdhdGVXb3JrZXIoc2V0W2luZGV4XS5vdXRwdXQpO1xuICAgICAgICBpbmRleCsrO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIGtpY2sgaXRcbiAgICB2YXIgaW5kZXggPSAwO1xuICAgIHZhciBpdGVyYXRpb25zID0gMDtcbiAgICBhY3RpdmF0ZVdvcmtlcihzZXRbaW5kZXhdLmlucHV0KTtcbiAgfVxuXG4gIC8vIHRyYWlucyBhbiBYT1IgdG8gdGhlIG5ldHdvcmtcbiAgWE9SKG9wdGlvbnMpIHtcblxuICAgIGlmICh0aGlzLm5ldHdvcmsuaW5wdXRzKCkgIT0gMiB8fCB0aGlzLm5ldHdvcmsub3V0cHV0cygpICE9IDEpXG4gICAgICB0aHJvdyBcIkVycm9yOiBJbmNvbXBhdGlibGUgbmV0d29yayAoMiBpbnB1dHMsIDEgb3V0cHV0KVwiO1xuXG4gICAgdmFyIGRlZmF1bHRzID0ge1xuICAgICAgaXRlcmF0aW9uczogMTAwMDAwLFxuICAgICAgbG9nOiBmYWxzZSxcbiAgICAgIHNodWZmbGU6IHRydWUsXG4gICAgICBjb3N0OiBUcmFpbmVyLmNvc3QuTVNFXG4gICAgfVxuXG4gICAgaWYgKG9wdGlvbnMpXG4gICAgICBmb3IgKHZhciBpIGluIG9wdGlvbnMpXG4gICAgICAgIGRlZmF1bHRzW2ldID0gb3B0aW9uc1tpXTtcblxuICAgIHJldHVybiB0aGlzLnRyYWluKFt7XG4gICAgICBpbnB1dDogWzAsIDBdLFxuICAgICAgb3V0cHV0OiBbMF1cbiAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMSwgMF0sXG4gICAgICAgIG91dHB1dDogWzFdXG4gICAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMCwgMV0sXG4gICAgICAgIG91dHB1dDogWzFdXG4gICAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMSwgMV0sXG4gICAgICAgIG91dHB1dDogWzBdXG4gICAgICB9XSwgZGVmYXVsdHMpO1xuICB9XG5cbiAgLy8gdHJhaW5zIHRoZSBuZXR3b3JrIHRvIHBhc3MgYSBEaXN0cmFjdGVkIFNlcXVlbmNlIFJlY2FsbCB0ZXN0XG4gIERTUihvcHRpb25zKSB7XG4gICAgb3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG5cbiAgICB2YXIgdGFyZ2V0cyA9IG9wdGlvbnMudGFyZ2V0cyB8fCBbMiwgNCwgNywgOF07XG4gICAgdmFyIGRpc3RyYWN0b3JzID0gb3B0aW9ucy5kaXN0cmFjdG9ycyB8fCBbMywgNSwgNiwgOV07XG4gICAgdmFyIHByb21wdHMgPSBvcHRpb25zLnByb21wdHMgfHwgWzAsIDFdO1xuICAgIHZhciBsZW5ndGggPSBvcHRpb25zLmxlbmd0aCB8fCAyNDtcbiAgICB2YXIgY3JpdGVyaW9uID0gb3B0aW9ucy5zdWNjZXNzIHx8IDAuOTU7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnMgfHwgMTAwMDAwO1xuICAgIHZhciByYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4xO1xuICAgIHZhciBsb2cgPSBvcHRpb25zLmxvZyB8fCAwO1xuICAgIHZhciBzY2hlZHVsZSA9IG9wdGlvbnMuc2NoZWR1bGUgfHwge307XG4gICAgdmFyIGNvcnJlY3QgPSAwO1xuICAgIHZhciBpID0gMDtcbiAgICB2YXIgc3VjY2VzcyA9IDA7XG4gICAgdmFyIHRyaWFsID0gaSA9IGNvcnJlY3QgPSBqID0gc3VjY2VzcyA9IDAsXG4gICAgICBlcnJvciA9IDEsXG4gICAgICBzeW1ib2xzID0gdGFyZ2V0cy5sZW5ndGggKyBkaXN0cmFjdG9ycy5sZW5ndGggKyBwcm9tcHRzLmxlbmd0aDtcblxuICAgIHZhciBub1JlcGVhdCA9IGZ1bmN0aW9uKHJhbmdlLCBhdm9pZCkge1xuICAgICAgdmFyIG51bWJlciA9IE1hdGgucmFuZG9tKCkgKiByYW5nZSB8IDA7XG4gICAgICB2YXIgdXNlZCA9IGZhbHNlO1xuICAgICAgZm9yICh2YXIgaSBpbiBhdm9pZClcbiAgICAgICAgaWYgKG51bWJlciA9PSBhdm9pZFtpXSlcbiAgICAgICAgICB1c2VkID0gdHJ1ZTtcbiAgICAgIHJldHVybiB1c2VkID8gbm9SZXBlYXQocmFuZ2UsIGF2b2lkKSA6IG51bWJlcjtcbiAgICB9XG5cbiAgICB2YXIgZXF1YWwgPSBmdW5jdGlvbihwcmVkaWN0aW9uLCBvdXRwdXQpIHtcbiAgICAgIGZvciAodmFyIGkgaW4gcHJlZGljdGlvbilcbiAgICAgICAgaWYgKE1hdGgucm91bmQocHJlZGljdGlvbltpXSkgIT0gb3V0cHV0W2ldKVxuICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICB3aGlsZSAodHJpYWwgPCBpdGVyYXRpb25zICYmIChzdWNjZXNzIDwgY3JpdGVyaW9uIHx8IHRyaWFsICUgMTAwMCAhPSAwKSkge1xuICAgICAgLy8gZ2VuZXJhdGUgc2VxdWVuY2VcbiAgICAgIHZhciBzZXF1ZW5jZSA9IFtdLFxuICAgICAgICBzZXF1ZW5jZUxlbmd0aCA9IGxlbmd0aCAtIHByb21wdHMubGVuZ3RoO1xuICAgICAgZm9yIChpID0gMDsgaSA8IHNlcXVlbmNlTGVuZ3RoOyBpKyspIHtcbiAgICAgICAgdmFyIGFueSA9IE1hdGgucmFuZG9tKCkgKiBkaXN0cmFjdG9ycy5sZW5ndGggfCAwO1xuICAgICAgICBzZXF1ZW5jZS5wdXNoKGRpc3RyYWN0b3JzW2FueV0pO1xuICAgICAgfVxuICAgICAgdmFyIGluZGV4ZXMgPSBbXSxcbiAgICAgICAgcG9zaXRpb25zID0gW107XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgcHJvbXB0cy5sZW5ndGg7IGkrKykge1xuICAgICAgICBpbmRleGVzLnB1c2goTWF0aC5yYW5kb20oKSAqIHRhcmdldHMubGVuZ3RoIHwgMCk7XG4gICAgICAgIHBvc2l0aW9ucy5wdXNoKG5vUmVwZWF0KHNlcXVlbmNlTGVuZ3RoLCBwb3NpdGlvbnMpKTtcbiAgICAgIH1cbiAgICAgIHBvc2l0aW9ucyA9IHBvc2l0aW9ucy5zb3J0KCk7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgcHJvbXB0cy5sZW5ndGg7IGkrKykge1xuICAgICAgICBzZXF1ZW5jZVtwb3NpdGlvbnNbaV1dID0gdGFyZ2V0c1tpbmRleGVzW2ldXTtcbiAgICAgICAgc2VxdWVuY2UucHVzaChwcm9tcHRzW2ldKTtcbiAgICAgIH1cblxuICAgICAgLy90cmFpbiBzZXF1ZW5jZVxuICAgICAgdmFyIGRpc3RyYWN0b3JzQ29ycmVjdDtcbiAgICAgIHZhciB0YXJnZXRzQ29ycmVjdCA9IGRpc3RyYWN0b3JzQ29ycmVjdCA9IDA7XG4gICAgICBlcnJvciA9IDA7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgbGVuZ3RoOyBpKyspIHtcbiAgICAgICAgLy8gZ2VuZXJhdGUgaW5wdXQgZnJvbSBzZXF1ZW5jZVxuICAgICAgICB2YXIgaW5wdXQgPSBbXTtcbiAgICAgICAgZm9yIChqID0gMDsgaiA8IHN5bWJvbHM7IGorKylcbiAgICAgICAgICBpbnB1dFtqXSA9IDA7XG4gICAgICAgIGlucHV0W3NlcXVlbmNlW2ldXSA9IDE7XG5cbiAgICAgICAgLy8gZ2VuZXJhdGUgdGFyZ2V0IG91dHB1dFxuICAgICAgICB2YXIgb3V0cHV0ID0gW107XG4gICAgICAgIGZvciAoaiA9IDA7IGogPCB0YXJnZXRzLmxlbmd0aDsgaisrKVxuICAgICAgICAgIG91dHB1dFtqXSA9IDA7XG5cbiAgICAgICAgaWYgKGkgPj0gc2VxdWVuY2VMZW5ndGgpIHtcbiAgICAgICAgICB2YXIgaW5kZXggPSBpIC0gc2VxdWVuY2VMZW5ndGg7XG4gICAgICAgICAgb3V0cHV0W2luZGV4ZXNbaW5kZXhdXSA9IDE7XG4gICAgICAgIH1cblxuICAgICAgICAvLyBjaGVjayByZXN1bHRcbiAgICAgICAgdmFyIHByZWRpY3Rpb24gPSB0aGlzLm5ldHdvcmsuYWN0aXZhdGUoaW5wdXQpO1xuXG4gICAgICAgIGlmIChlcXVhbChwcmVkaWN0aW9uLCBvdXRwdXQpKVxuICAgICAgICAgIGlmIChpIDwgc2VxdWVuY2VMZW5ndGgpXG4gICAgICAgICAgICBkaXN0cmFjdG9yc0NvcnJlY3QrKztcbiAgICAgICAgICBlbHNlXG4gICAgICAgICAgICB0YXJnZXRzQ29ycmVjdCsrO1xuICAgICAgICBlbHNlIHtcbiAgICAgICAgICB0aGlzLm5ldHdvcmsucHJvcGFnYXRlKHJhdGUsIG91dHB1dCk7XG4gICAgICAgIH1cblxuICAgICAgICB2YXIgZGVsdGEgPSAwO1xuICAgICAgICBmb3IgKHZhciBqIGluIHByZWRpY3Rpb24pXG4gICAgICAgICAgZGVsdGEgKz0gTWF0aC5wb3cob3V0cHV0W2pdIC0gcHJlZGljdGlvbltqXSwgMik7XG4gICAgICAgIGVycm9yICs9IGRlbHRhIC8gdGhpcy5uZXR3b3JrLm91dHB1dHMoKTtcblxuICAgICAgICBpZiAoZGlzdHJhY3RvcnNDb3JyZWN0ICsgdGFyZ2V0c0NvcnJlY3QgPT0gbGVuZ3RoKVxuICAgICAgICAgIGNvcnJlY3QrKztcbiAgICAgIH1cblxuICAgICAgLy8gY2FsY3VsYXRlIGVycm9yXG4gICAgICBpZiAodHJpYWwgJSAxMDAwID09IDApXG4gICAgICAgIGNvcnJlY3QgPSAwO1xuICAgICAgdHJpYWwrKztcbiAgICAgIHZhciBkaXZpZGVFcnJvciA9IHRyaWFsICUgMTAwMDtcbiAgICAgIGRpdmlkZUVycm9yID0gZGl2aWRlRXJyb3IgPT0gMCA/IDEwMDAgOiBkaXZpZGVFcnJvcjtcbiAgICAgIHN1Y2Nlc3MgPSBjb3JyZWN0IC8gZGl2aWRlRXJyb3I7XG4gICAgICBlcnJvciAvPSBsZW5ndGg7XG5cbiAgICAgIC8vIGxvZ1xuICAgICAgaWYgKGxvZyAmJiB0cmlhbCAlIGxvZyA9PSAwKVxuICAgICAgICBjb25zb2xlLmxvZyhcIml0ZXJhdGlvbnM6XCIsIHRyaWFsLCBcIiBzdWNjZXNzOlwiLCBzdWNjZXNzLCBcIiBjb3JyZWN0OlwiLFxuICAgICAgICAgIGNvcnJlY3QsIFwiIHRpbWU6XCIsIERhdGUubm93KCkgLSBzdGFydCwgXCIgZXJyb3I6XCIsIGVycm9yKTtcbiAgICAgIGlmIChzY2hlZHVsZS5kbyAmJiBzY2hlZHVsZS5ldmVyeSAmJiB0cmlhbCAlIHNjaGVkdWxlLmV2ZXJ5ID09IDApIHtcbiAgICAgICAgc2NoZWR1bGUuZG8oe1xuICAgICAgICAgIGl0ZXJhdGlvbnM6IHRyaWFsLFxuICAgICAgICAgIHN1Y2Nlc3M6IHN1Y2Nlc3MsXG4gICAgICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydCxcbiAgICAgICAgICBjb3JyZWN0OiBjb3JyZWN0XG4gICAgICAgIH0pO1xuXG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgIGl0ZXJhdGlvbnM6IHRyaWFsLFxuICAgICAgc3VjY2Vzczogc3VjY2VzcyxcbiAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydFxuICAgIH1cbiAgfVxuXG4gIC8vIHRyYWluIHRoZSBuZXR3b3JrIHRvIGxlYXJuIGFuIEVtYmVkZWQgUmViZXIgR3JhbW1hclxuICBFUkcob3B0aW9ucykge1xuXG4gICAgb3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnMgfHwgMTUwMDAwO1xuICAgIHZhciBjcml0ZXJpb24gPSBvcHRpb25zLmVycm9yIHx8IC4wNTtcbiAgICB2YXIgcmF0ZSA9IG9wdGlvbnMucmF0ZSB8fCAuMTtcbiAgICB2YXIgbG9nID0gb3B0aW9ucy5sb2cgfHwgNTAwO1xuXG4gICAgLy8gZ3JhbWFyIG5vZGVcbiAgICB2YXIgTm9kZSA9IGZ1bmN0aW9uKCkge1xuICAgICAgdGhpcy5wYXRocyA9IFtdO1xuICAgIH1cbiAgICBOb2RlLnByb3RvdHlwZSA9IHtcbiAgICAgIGNvbm5lY3Q6IGZ1bmN0aW9uKG5vZGUsIHZhbHVlKSB7XG4gICAgICAgIHRoaXMucGF0aHMucHVzaCh7XG4gICAgICAgICAgbm9kZTogbm9kZSxcbiAgICAgICAgICB2YWx1ZTogdmFsdWVcbiAgICAgICAgfSk7XG4gICAgICAgIHJldHVybiB0aGlzO1xuICAgICAgfSxcbiAgICAgIGFueTogZnVuY3Rpb24oKSB7XG4gICAgICAgIGlmICh0aGlzLnBhdGhzLmxlbmd0aCA9PSAwKVxuICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgdmFyIGluZGV4ID0gTWF0aC5yYW5kb20oKSAqIHRoaXMucGF0aHMubGVuZ3RoIHwgMDtcbiAgICAgICAgcmV0dXJuIHRoaXMucGF0aHNbaW5kZXhdO1xuICAgICAgfSxcbiAgICAgIHRlc3Q6IGZ1bmN0aW9uKHZhbHVlKSB7XG4gICAgICAgIGZvciAodmFyIGkgaW4gdGhpcy5wYXRocylcbiAgICAgICAgICBpZiAodGhpcy5wYXRoc1tpXS52YWx1ZSA9PSB2YWx1ZSlcbiAgICAgICAgICAgIHJldHVybiB0aGlzLnBhdGhzW2ldO1xuICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdmFyIHJlYmVyR3JhbW1hciA9IGZ1bmN0aW9uKCkge1xuXG4gICAgICAvLyBidWlsZCBhIHJlYmVyIGdyYW1tYXJcbiAgICAgIHZhciBvdXRwdXQgPSBuZXcgTm9kZSgpO1xuICAgICAgdmFyIG4xID0gKG5ldyBOb2RlKCkpLmNvbm5lY3Qob3V0cHV0LCBcIkVcIik7XG4gICAgICB2YXIgbjIgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuMSwgXCJTXCIpO1xuICAgICAgdmFyIG4zID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjEsIFwiVlwiKS5jb25uZWN0KG4yLCBcIlBcIik7XG4gICAgICB2YXIgbjQgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuMiwgXCJYXCIpXG4gICAgICBuNC5jb25uZWN0KG40LCBcIlNcIik7XG4gICAgICB2YXIgbjUgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuMywgXCJWXCIpXG4gICAgICBuNS5jb25uZWN0KG41LCBcIlRcIik7XG4gICAgICBuMi5jb25uZWN0KG41LCBcIlhcIilcbiAgICAgIHZhciBuNiA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG40LCBcIlRcIikuY29ubmVjdChuNSwgXCJQXCIpO1xuICAgICAgdmFyIGlucHV0ID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjYsIFwiQlwiKVxuXG4gICAgICByZXR1cm4ge1xuICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgIG91dHB1dDogb3V0cHV0XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8gYnVpbGQgYW4gZW1iZWRlZCByZWJlciBncmFtbWFyXG4gICAgdmFyIGVtYmVkZWRSZWJlckdyYW1tYXIgPSBmdW5jdGlvbigpIHtcbiAgICAgIHZhciByZWJlcjEgPSByZWJlckdyYW1tYXIoKTtcbiAgICAgIHZhciByZWJlcjIgPSByZWJlckdyYW1tYXIoKTtcblxuICAgICAgdmFyIG91dHB1dCA9IG5ldyBOb2RlKCk7XG4gICAgICB2YXIgbjEgPSAobmV3IE5vZGUpLmNvbm5lY3Qob3V0cHV0LCBcIkVcIik7XG4gICAgICByZWJlcjEub3V0cHV0LmNvbm5lY3QobjEsIFwiVFwiKTtcbiAgICAgIHJlYmVyMi5vdXRwdXQuY29ubmVjdChuMSwgXCJQXCIpO1xuICAgICAgdmFyIG4yID0gKG5ldyBOb2RlKS5jb25uZWN0KHJlYmVyMS5pbnB1dCwgXCJQXCIpLmNvbm5lY3QocmViZXIyLmlucHV0LFxuICAgICAgICBcIlRcIik7XG4gICAgICB2YXIgaW5wdXQgPSAobmV3IE5vZGUpLmNvbm5lY3QobjIsIFwiQlwiKTtcblxuICAgICAgcmV0dXJuIHtcbiAgICAgICAgaW5wdXQ6IGlucHV0LFxuICAgICAgICBvdXRwdXQ6IG91dHB1dFxuICAgICAgfVxuXG4gICAgfVxuXG4gICAgLy8gZ2VuZXJhdGUgYW4gRVJHIHNlcXVlbmNlXG4gICAgdmFyIGdlbmVyYXRlID0gZnVuY3Rpb24oKSB7XG4gICAgICB2YXIgbm9kZSA9IGVtYmVkZWRSZWJlckdyYW1tYXIoKS5pbnB1dDtcbiAgICAgIHZhciBuZXh0ID0gbm9kZS5hbnkoKTtcbiAgICAgIHZhciBzdHIgPSBcIlwiO1xuICAgICAgd2hpbGUgKG5leHQpIHtcbiAgICAgICAgc3RyICs9IG5leHQudmFsdWU7XG4gICAgICAgIG5leHQgPSBuZXh0Lm5vZGUuYW55KCk7XG4gICAgICB9XG4gICAgICByZXR1cm4gc3RyO1xuICAgIH1cblxuICAgIC8vIHRlc3QgaWYgYSBzdHJpbmcgbWF0Y2hlcyBhbiBlbWJlZGVkIHJlYmVyIGdyYW1tYXJcbiAgICB2YXIgdGVzdCA9IGZ1bmN0aW9uKHN0cikge1xuICAgICAgdmFyIG5vZGUgPSBlbWJlZGVkUmViZXJHcmFtbWFyKCkuaW5wdXQ7XG4gICAgICB2YXIgaSA9IDA7XG4gICAgICB2YXIgY2ggPSBzdHIuY2hhckF0KGkpO1xuICAgICAgd2hpbGUgKGkgPCBzdHIubGVuZ3RoKSB7XG4gICAgICAgIHZhciBuZXh0ID0gbm9kZS50ZXN0KGNoKTtcbiAgICAgICAgaWYgKCFuZXh0KVxuICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgbm9kZSA9IG5leHQubm9kZTtcbiAgICAgICAgY2ggPSBzdHIuY2hhckF0KCsraSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG5cbiAgICAvLyBoZWxwZXIgdG8gY2hlY2sgaWYgdGhlIG91dHB1dCBhbmQgdGhlIHRhcmdldCB2ZWN0b3JzIG1hdGNoXG4gICAgdmFyIGRpZmZlcmVudCA9IGZ1bmN0aW9uKGFycmF5MSwgYXJyYXkyKSB7XG4gICAgICB2YXIgbWF4MSA9IDA7XG4gICAgICB2YXIgaTEgPSAtMTtcbiAgICAgIHZhciBtYXgyID0gMDtcbiAgICAgIHZhciBpMiA9IC0xO1xuICAgICAgZm9yICh2YXIgaSBpbiBhcnJheTEpIHtcbiAgICAgICAgaWYgKGFycmF5MVtpXSA+IG1heDEpIHtcbiAgICAgICAgICBtYXgxID0gYXJyYXkxW2ldO1xuICAgICAgICAgIGkxID0gaTtcbiAgICAgICAgfVxuICAgICAgICBpZiAoYXJyYXkyW2ldID4gbWF4Mikge1xuICAgICAgICAgIG1heDIgPSBhcnJheTJbaV07XG4gICAgICAgICAgaTIgPSBpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIHJldHVybiBpMSAhPSBpMjtcbiAgICB9XG5cbiAgICB2YXIgaXRlcmF0aW9uID0gMDtcbiAgICB2YXIgZXJyb3IgPSAxO1xuICAgIHZhciB0YWJsZSA9IHtcbiAgICAgIFwiQlwiOiAwLFxuICAgICAgXCJQXCI6IDEsXG4gICAgICBcIlRcIjogMixcbiAgICAgIFwiWFwiOiAzLFxuICAgICAgXCJTXCI6IDQsXG4gICAgICBcIkVcIjogNVxuICAgIH1cblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG4gICAgd2hpbGUgKGl0ZXJhdGlvbiA8IGl0ZXJhdGlvbnMgJiYgZXJyb3IgPiBjcml0ZXJpb24pIHtcbiAgICAgIHZhciBpID0gMDtcbiAgICAgIGVycm9yID0gMDtcblxuICAgICAgLy8gRVJHIHNlcXVlbmNlIHRvIGxlYXJuXG4gICAgICB2YXIgc2VxdWVuY2UgPSBnZW5lcmF0ZSgpO1xuXG4gICAgICAvLyBpbnB1dFxuICAgICAgdmFyIHJlYWQgPSBzZXF1ZW5jZS5jaGFyQXQoaSk7XG4gICAgICAvLyB0YXJnZXRcbiAgICAgIHZhciBwcmVkaWN0ID0gc2VxdWVuY2UuY2hhckF0KGkgKyAxKTtcblxuICAgICAgLy8gdHJhaW5cbiAgICAgIHdoaWxlIChpIDwgc2VxdWVuY2UubGVuZ3RoIC0gMSkge1xuICAgICAgICB2YXIgaW5wdXQgPSBbXTtcbiAgICAgICAgdmFyIHRhcmdldCA9IFtdO1xuICAgICAgICBmb3IgKHZhciBqID0gMDsgaiA8IDY7IGorKykge1xuICAgICAgICAgIGlucHV0W2pdID0gMDtcbiAgICAgICAgICB0YXJnZXRbal0gPSAwO1xuICAgICAgICB9XG4gICAgICAgIGlucHV0W3RhYmxlW3JlYWRdXSA9IDE7XG4gICAgICAgIHRhcmdldFt0YWJsZVtwcmVkaWN0XV0gPSAxO1xuXG4gICAgICAgIHZhciBvdXRwdXQgPSB0aGlzLm5ldHdvcmsuYWN0aXZhdGUoaW5wdXQpO1xuXG4gICAgICAgIGlmIChkaWZmZXJlbnQob3V0cHV0LCB0YXJnZXQpKVxuICAgICAgICAgIHRoaXMubmV0d29yay5wcm9wYWdhdGUocmF0ZSwgdGFyZ2V0KTtcblxuICAgICAgICByZWFkID0gc2VxdWVuY2UuY2hhckF0KCsraSk7XG4gICAgICAgIHByZWRpY3QgPSBzZXF1ZW5jZS5jaGFyQXQoaSArIDEpO1xuXG4gICAgICAgIHZhciBkZWx0YSA9IDA7XG4gICAgICAgIGZvciAodmFyIGsgaW4gb3V0cHV0KVxuICAgICAgICAgIGRlbHRhICs9IE1hdGgucG93KHRhcmdldFtrXSAtIG91dHB1dFtrXSwgMilcbiAgICAgICAgZGVsdGEgLz0gb3V0cHV0Lmxlbmd0aDtcblxuICAgICAgICBlcnJvciArPSBkZWx0YTtcbiAgICAgIH1cbiAgICAgIGVycm9yIC89IHNlcXVlbmNlLmxlbmd0aDtcbiAgICAgIGl0ZXJhdGlvbisrO1xuICAgICAgaWYgKGl0ZXJhdGlvbiAlIGxvZyA9PSAwKSB7XG4gICAgICAgIGNvbnNvbGUubG9nKFwiaXRlcmF0aW9uczpcIiwgaXRlcmF0aW9uLCBcIiB0aW1lOlwiLCBEYXRlLm5vdygpIC0gc3RhcnQsXG4gICAgICAgICAgXCIgZXJyb3I6XCIsIGVycm9yKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4ge1xuICAgICAgaXRlcmF0aW9uczogaXRlcmF0aW9uLFxuICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0LFxuICAgICAgdGVzdDogdGVzdCxcbiAgICAgIGdlbmVyYXRlOiBnZW5lcmF0ZVxuICAgIH1cbiAgfVxuXG59XG5cbmV4cG9ydCBtb2R1bGUgVHJhaW5lciB7XG4gIC8vIEJ1aWx0LWluIGNvc3QgZnVuY3Rpb25zXG4gIFxuICBleHBvcnQgaW50ZXJmYWNlIElUcmFpbmVyQ29zdEZuIHtcbiAgICAodGFyZ2V0LCBvdXRwdXQpOiBudW1iZXI7XG4gIH1cblxuICBleHBvcnQgdmFyIGNvc3QgPSB7XG4gICAgLy8gRXEuIDlcbiAgICBDUk9TU19FTlRST1BZOiBmdW5jdGlvbih0YXJnZXQsIG91dHB1dCkge1xuICAgICAgdmFyIGNyb3NzZW50cm9weSA9IDA7XG4gICAgICBmb3IgKHZhciBpIGluIG91dHB1dClcbiAgICAgICAgY3Jvc3NlbnRyb3B5IC09ICh0YXJnZXRbaV0gKiBNYXRoLmxvZyhvdXRwdXRbaV0gKyAxZS0xNSkpICsgKCgxIC0gdGFyZ2V0W2ldKSAqIE1hdGgubG9nKCgxICsgMWUtMTUpIC0gb3V0cHV0W2ldKSk7IC8vICsxZS0xNSBpcyBhIHRpbnkgcHVzaCBhd2F5IHRvIGF2b2lkIE1hdGgubG9nKDApXG4gICAgICByZXR1cm4gY3Jvc3NlbnRyb3B5O1xuICAgIH0sXG4gICAgTVNFOiBmdW5jdGlvbih0YXJnZXQsIG91dHB1dCkge1xuICAgICAgdmFyIG1zZSA9IDA7XG4gICAgICBmb3IgKHZhciBpIGluIG91dHB1dClcbiAgICAgICAgbXNlICs9IE1hdGgucG93KHRhcmdldFtpXSAtIG91dHB1dFtpXSwgMik7XG4gICAgICByZXR1cm4gbXNlIC8gb3V0cHV0Lmxlbmd0aDtcbiAgICB9XG4gIH1cbn0iXSwic291cmNlUm9vdCI6Ii9zb3VyY2UvIn0=