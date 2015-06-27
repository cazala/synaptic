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
                    console.log('iterations', iterations, 'error', error, 'rate', currentRate, 'T:', target, 'O:', output);
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
        CROSS_ENTROPY_SOFTMAX: function (target, output) {
            var crossentropy = 0;
            for (var i in output)
                crossentropy -= target[i] * Math.log(output[i] + 1e-15);
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

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy90cmFpbmVyLnRzIl0sIm5hbWVzIjpbIlRyYWluZXIiLCJUcmFpbmVyLmNvbnN0cnVjdG9yIiwiVHJhaW5lci50cmFpbiIsIlRyYWluZXIudHJhaW4uc2h1ZmZsZSIsIlRyYWluZXIud29ya2VyVHJhaW4iLCJUcmFpbmVyLndvcmtlclRyYWluLnNodWZmbGUiLCJUcmFpbmVyLndvcmtlclRyYWluLmFjdGl2YXRlV29ya2VyIiwiVHJhaW5lci53b3JrZXJUcmFpbi5wcm9wYWdhdGVXb3JrZXIiLCJUcmFpbmVyLlhPUiIsIlRyYWluZXIuRFNSIiwiVHJhaW5lci5FUkciXSwibWFwcGluZ3MiOiJBQUVBLEFBSUE7OzRGQUY0RjtJQUUvRSxPQUFPO0lBUWxCQSxTQVJXQSxPQUFPQSxDQVFOQSxPQUFvQkEsRUFBRUEsT0FBYUE7UUFOL0NDLFNBQUlBLEdBQVFBLEVBQUVBLENBQUNBO1FBQ2ZBLGVBQVVBLEdBQUdBLE1BQU1BLENBQUNBO1FBQ3BCQSxVQUFLQSxHQUFHQSxJQUFJQSxDQUFDQTtRQUtYQSxPQUFPQSxHQUFHQSxPQUFPQSxJQUFJQSxFQUFFQSxDQUFDQTtRQUN4QkEsSUFBSUEsQ0FBQ0EsT0FBT0EsR0FBR0EsT0FBT0EsQ0FBQ0E7UUFDdkJBLElBQUlBLENBQUNBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLElBQUlBLEVBQUVBLENBQUNBO1FBQy9CQSxJQUFJQSxDQUFDQSxVQUFVQSxHQUFHQSxPQUFPQSxDQUFDQSxVQUFVQSxJQUFJQSxNQUFNQSxDQUFDQTtRQUMvQ0EsSUFBSUEsQ0FBQ0EsS0FBS0EsR0FBR0EsT0FBT0EsQ0FBQ0EsS0FBS0EsSUFBSUEsSUFBSUEsQ0FBQUE7UUFDbENBLElBQUlBLENBQUNBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLElBQUlBLE9BQU9BLENBQUNBLElBQUlBLENBQUNBLGFBQWFBLENBQUNBO0lBQ3pEQSxDQUFDQTtJQUVERCxvQ0FBb0NBO0lBQ3BDQSx1QkFBS0EsR0FBTEEsVUFBTUEs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QUFJQSxPQUFPQSxHQUFHQSxFQUFFQSxFQUNkQSxTQUFTQSxHQUFHQSxFQUFFQSxDQUFDQTtZQUNqQkEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsT0FBT0EsQ0FBQ0EsTUFBTUEsRUFBRUEsQ0FBQ0EsRUFBRUEsRUFBRUEsQ0FBQ0E7Z0JBQ3BDQSxPQUFPQSxDQUFDQSxJQUFJQSxDQUFDQSxJQUFJQSxDQUFDQSxNQUFNQSxFQUFFQSxHQUFHQSxPQUFPQSxDQUFDQSxNQUFNQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDakRBLFNBQVNBLENBQUNBLElBQUlBLENBQUNBLFFBQVFBLENBQUNBLGNBQWNBLEVBQUVBLFNBQVNBLENBQUNBLENBQUNBLENBQUNBO1lBQ3REQSxDQUFDQTtZQUNEQSxTQUFTQSxHQUFHQSxTQUFTQSxDQUFDQSxJQUFJQSxFQUFFQSxDQUFDQTtZQUM3QkEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsT0FBT0EsQ0FBQ0EsTUFBTUEsRUFBRUEsQ0FBQ0EsRUFBRUEsRUFBRUEsQ0FBQ0E7Z0JBQ3BDQSxRQUFRQSxDQUFDQSxTQUFTQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxPQUFPQSxDQUFDQSxPQUFPQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDN0NBLFFBQVFBLENBQUNBLElBQUlBLENBQUNBLE9BQU9BLENBQUNBLENBQUNBLENBQUNBLENBQUNBLENBQUNBO1lBQzVCQSxDQUFDQTtZQUVEQSxBQUNBQSxnQkFEZ0JBO2dCQUNaQSxrQkFBa0JBLENBQUNBO1lBQ3ZCQSxJQUFJQSxjQUFjQSxHQUFHQSxrQkFBa0JBLEdBQUdBLENBQUNBLENBQUNBO1lBQzVDQSxLQUFLQSxHQUFHQSxDQUFDQSxDQUFDQTtZQUNWQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxNQUFNQSxFQUFFQSxDQUFDQSxFQUFFQSxFQUFFQSxDQUFDQTtnQkFDNUJBLEFBQ0FBLCtCQUQrQkE7b0JBQzNCQSxLQUFLQSxHQUFHQSxFQUFFQSxDQUFDQTtnQkFDZkEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsT0FBT0EsRUFBRUEsQ0FBQ0EsRUFBRUE7b0JBQzFCQSxLQUFLQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQTtnQkFDZkEsS0FBS0EsQ0FBQ0EsUUFBUUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7Z0JBRXZCQSxBQUNBQSx5QkFEeUJBO29CQUNyQkEsTUFBTUEsR0FBR0EsRUFBRUEsQ0FBQ0E7Z0JBQ2hCQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxPQUFPQSxDQUFDQSxNQUFNQSxFQUFFQSxDQUFDQSxFQUFFQTtvQkFDakNBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBO2dCQUVoQkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsSUFBSUEsY0FBY0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7b0JBQ3hCQSxJQUFJQSxLQUFLQSxHQUFHQSxDQUFDQSxHQUFHQSxjQUFjQSxDQUFDQTtvQkFDL0JBLE1BQU1BLENBQUNBLE9BQU9BLENBQUNBLEtBQUtBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBO2dCQUM3QkEsQ0FBQ0E7Z0JBRURBLEFBQ0FBLGVBRGVBO29CQUNYQSxVQUFVQSxHQUFHQSxJQUFJQSxDQUFDQSxPQUFPQSxDQUFDQSxRQUFRQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQTtnQkFFOUNBLEVBQUVBLENBQUNBLENBQUNBLEtBQUtBLENBQUNBLFVBQVVBLEVBQUVBLE1BQU1BLENBQUNBLENBQUNBO29CQUM1QkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsY0FBY0EsQ0FBQ0E7d0JBQ3JCQSxrQkFBa0JBLEVBQUVBLENBQUNBO29CQUN2QkEsSUFBSUE7d0JBQ0ZBLGNBQWNBLEVBQUVBLENBQUNBO2dCQUNyQkEsSUFBSUEsQ0FBQ0EsQ0FBQ0E7b0JBQ0pBLElBQUlBLENBQUNBLE9BQU9BLENBQUNBLFNBQVNBLENBQUNBLElBQUlBLEVBQUVBLE1BQU1BLENBQUNBLENBQUNBO2dCQUN2Q0EsQ0FBQ0E7Z0JBRURBLElBQUlBLEtBQUtBLEdBQUdBLENBQUNBLENBQUNBO2dCQUNkQSxHQUFHQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxJQUFJQSxVQUFVQSxDQUFDQTtvQkFDdkJBLEtBQUtBLElBQUlBLElBQUlBLENBQUNBLEdBQUdBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLFVBQVVBLENBQUNBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO2dCQUNsREEsS0FBS0EsSUFBSUEsS0FBS0EsR0FBR0EsSUFBSUEsQ0FBQ0EsT0FBT0EsQ0FBQ0EsT0FBT0EsRUFBRUEsQ0FBQ0E7Z0JBRXhDQSxFQUFFQSxDQUFDQSxDQUFDQSxrQkFBa0JBLEdBQUdBLGNBQWNBLElBQUlBLE1BQU1BLENBQUNBO29CQUNoREEsT0FBT0EsRUFBRUEsQ0FBQ0E7WUFDZEEsQ0FBQ0E7WUFFREEsQUFDQUEsa0JBRGtCQTtZQUNsQkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsS0FBS0EsR0FBR0EsSUFBSUEsSUFBSUEsQ0FBQ0EsQ0FBQ0E7Z0JBQ3BCQSxPQUFPQSxHQUFHQSxDQUFDQSxDQUFDQTtZQUNkQSxLQUFLQSxFQUFFQSxDQUFDQTtZQUNSQSxJQUFJQSxXQUFXQSxHQUFHQSxLQUFLQSxHQUFHQSxJQUFJQSxDQUFDQTtZQUMvQkEsV0FBV0EsR0FBR0EsV0FBV0EsSUFBSUEsQ0FBQ0EsR0FBR0EsSUFBSUEsR0FBR0EsV0FBV0EsQ0FBQ0E7WUFDcERBLE9BQU9BLEdBQUdBLE9BQU9BLEdBQUdBLFdBQVdBLENBQUNBO1lBQ2hDQSxLQUFLQSxJQUFJQSxNQUFNQSxDQUFDQTtZQUVoQkEsQUFDQUEsTUFETUE7WUFDTkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsR0FBR0EsSUFBSUEsS0FBS0EsR0FBR0EsR0FBR0EsSUFBSUEsQ0FBQ0EsQ0FBQ0E7Z0JBQzFCQSxPQUFPQSxDQUFDQSxHQUFHQSxDQUFDQSxhQUFhQSxFQUFFQSxLQUFLQSxFQUFFQSxXQUFXQSxFQUFFQSxPQUFPQSxFQUFFQSxXQUFXQSxFQUNqRUEsT0FBT0EsRUFBRUEsUUFBUUEsRUFBRUEsSUFBSUEsQ0FBQ0EsR0FBR0EsRUFBRUEsR0FBR0EsS0FBS0EsRUFBRUEsU0FBU0EsRUFBRUEsS0FBS0EsQ0FBQ0EsQ0FBQ0E7WUFDN0RBLEVBQUVBLENBQUNBLENBQUNBLFFBQVFBLENBQUNBLEVBQUVBLElBQUlBLFFBQVFBLENBQUNBLEtBQUtBLElBQUlBLEtBQUtBLEdBQUdBLFFBQVFBLENBQUNBLEtBQUtBLElBQUlBLENBQUNBLENBQUNBLENBQUNBLENBQUNBO2dCQUNqRUEsUUFBUUEsQ0FBQ0EsRUFBRUEsQ0FBQ0E7b0JBQ1ZBLFVBQVVBLEVBQUVBLEtBQUtBO29CQUNqQkEsT0FBT0EsRUFBRUEsT0FBT0E7b0JBQ2hCQSxLQUFLQSxFQUFFQSxLQUFLQTtvQkFDWkEsSUFBSUEsRUFBRUEsSUFBSUEsQ0FBQ0EsR0FBR0EsRUFBRUEsR0FBR0EsS0FBS0E7b0JBQ3hCQSxPQUFPQSxFQUFFQSxPQUFPQTtpQkFDakJBLENBQUNBLENBQUNBO1lBRUxBLENBQUNBO1FBQ0hBLENBQUNBO1FBRURBLE1BQU1BLENBQUNBO1lBQ0xBLFVBQVVBLEVBQUVBLEtBQUtBO1lBQ2pCQSxPQUFPQSxFQUFFQSxPQUFPQTtZQUNoQkEsS0FBS0EsRUFBRUEsS0FBS0E7WUFDWkEsSUFBSUEsRUFBRUEsSUFBSUEsQ0FBQ0EsR0FBR0EsRUFBRUEsR0FBR0EsS0FBS0E7U0FDekJBLENBQUFBO0lBQ0hBLENBQUNBO0lBRURULHNEQUFzREE7SUFDdERBLHFCQUFHQSxHQUFIQSxVQUFJQSxPQUFPQTtRQUVUVSxPQUFPQSxHQUFHQSxPQUFPQSxJQUFJQSxFQUFFQSxDQUFDQTtRQUN4QkEsSUFBSUEsVUFBVUEsR0FBR0EsT0FBT0EsQ0FBQ0EsVUFBVUEsSUFBSUEsTUFBTUEsQ0FBQ0E7UUFDOUNBLElBQUlBLFNBQVNBLEdBQUdBLE9BQU9BLENBQUNBLEtBQUtBLElBQUlBLEdBQUdBLENBQUNBO1FBQ3JDQSxJQUFJQSxJQUFJQSxHQUFHQSxPQUFPQSxDQUFDQSxJQUFJQSxJQUFJQSxFQUFFQSxDQUFDQTtRQUM5QkEsSUFBSUEsR0FBR0EsR0FBR0EsT0FBT0EsQ0FBQ0EsR0FBR0EsSUFBSUEsR0FBR0EsQ0FBQ0E7UUFFN0JBLEFBQ0FBLGNBRGNBO1lBQ1ZBLElBQUlBLEdBQUdBO1lBQ1QsSUFBSSxDQUFDLEtBQUssR0FBRyxFQUFFLENBQUM7UUFDbEIsQ0FBQyxDQUFBQTtRQUNEQSxJQUFJQSxDQUFDQSxTQUFTQSxHQUFHQTtZQUNmQSxPQUFPQSxFQUFFQSxVQUFTQSxJQUFJQSxFQUFFQSxLQUFLQTtnQkFDM0IsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUM7b0JBQ2QsSUFBSSxFQUFFLElBQUk7b0JBQ1YsS0FBSyxFQUFFLEtBQUs7aUJBQ2IsQ0FBQyxDQUFDO2dCQUNILE1BQU0sQ0FBQyxJQUFJLENBQUM7WUFDZCxDQUFDO1lBQ0RBLEdBQUdBLEVBQUVBO2dCQUNILEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsTUFBTSxJQUFJLENBQUMsQ0FBQztvQkFDekIsTUFBTSxDQUFDLEtBQUssQ0FBQztnQkFDZixJQUFJLEtBQUssR0FBRyxJQUFJLENBQUMsTUFBTSxFQUFFLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDO2dCQUNsRCxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQztZQUMzQixDQUFDO1lBQ0RBLElBQUlBLEVBQUVBLFVBQVNBLEtBQUtBO2dCQUNsQixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxLQUFLLENBQUM7b0JBQ3ZCLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsS0FBSyxJQUFJLEtBQUssQ0FBQzt3QkFDL0IsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ3pCLE1BQU0sQ0FBQyxLQUFLLENBQUM7WUFDZixDQUFDO1NBQ0ZBLENBQUFBO1FBRURBLElBQUlBLFlBQVlBLEdBQUdBO1lBRWpCLEFBQ0Esd0JBRHdCO2dCQUNwQixNQUFNLEdBQUcsSUFBSSxJQUFJLEVBQUUsQ0FBQztZQUN4QixJQUFJLEVBQUUsR0FBRyxDQUFDLElBQUksSUFBSSxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1lBQzNDLElBQUksRUFBRSxHQUFHLENBQUMsSUFBSSxJQUFJLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDdkMsSUFBSSxFQUFFLEdBQUcsQ0FBQyxJQUFJLElBQUksRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1lBQ3hELElBQUksRUFBRSxHQUFHLENBQUMsSUFBSSxJQUFJLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUE7WUFDdEMsRUFBRSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDcEIsSUFBSSxFQUFFLEdBQUcsQ0FBQyxJQUFJLElBQUksRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQTtZQUN0QyxFQUFFLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQztZQUNwQixFQUFFLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQTtZQUNuQixJQUFJLEVBQUUsR0FBRyxDQUFDLElBQUksSUFBSSxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDeEQsSUFBSSxLQUFLLEdBQUcsQ0FBQyxJQUFJLElBQUksRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQTtZQUV6QyxNQUFNLENBQUM7Z0JBQ0wsS0FBSyxFQUFFLEtBQUs7Z0JBQ1osTUFBTSxFQUFFLE1BQU07YUFDZixDQUFBO1FBQ0gsQ0FBQyxDQUFBQTtRQUVEQSxBQUNBQSxpQ0FEaUNBO1lBQzdCQSxtQkFBbUJBLEdBQUdBO1lBQ3hCLElBQUksTUFBTSxHQUFHLFlBQVksRUFBRSxDQUFDO1lBQzVCLElBQUksTUFBTSxHQUFHLFlBQVksRUFBRSxDQUFDO1lBRTVCLElBQUksTUFBTSxHQUFHLElBQUksSUFBSSxFQUFFLENBQUM7WUFDeEIsSUFBSSxFQUFFLEdBQUcsQ0FBQyxJQUFJLElBQUksQ0FBQyxDQUFDLE9BQU8sQ0FBQyxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDekMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1lBQy9CLE1BQU0sQ0FBQyxNQUFNLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQztZQUMvQixJQUFJLEVBQUUsR0FBRyxDQUFDLElBQUksSUFBSSxDQUFDLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxLQUFLLEVBQUUsR0FBRyxDQUFDLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxLQUFLLEVBQ2pFLEdBQUcsQ0FBQyxDQUFDO1lBQ1AsSUFBSSxLQUFLLEdBQUcsQ0FBQyxJQUFJLElBQUksQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFFeEMsTUFBTSxDQUFDO2dCQUNMLEtBQUssRUFBRSxLQUFLO2dCQUNaLE1BQU0sRUFBRSxNQUFNO2FBQ2YsQ0FBQTtRQUVILENBQUMsQ0FBQUE7UUFFREEsQUFDQUEsMkJBRDJCQTtZQUN2QkEsUUFBUUEsR0FBR0E7WUFDYixJQUFJLElBQUksR0FBRyxtQkFBbUIsRUFBRSxDQUFDLEtBQUssQ0FBQztZQUN2QyxJQUFJLElBQUksR0FBRyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUM7WUFDdEIsSUFBSSxHQUFHLEdBQUcsRUFBRSxDQUFDO1lBQ2IsT0FBTyxJQUFJLEVBQUUsQ0FBQztnQkFDWixHQUFHLElBQUksSUFBSSxDQUFDLEtBQUssQ0FBQztnQkFDbEIsSUFBSSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUM7WUFDekIsQ0FBQztZQUNELE1BQU0sQ0FBQyxHQUFHLENBQUM7UUFDYixDQUFDLENBQUFBO1FBRURBLEFBQ0FBLG9EQURvREE7WUFDaERBLElBQUlBLEdBQUdBLFVBQVNBLEdBQUdBO1lBQ3JCLElBQUksSUFBSSxHQUFHLG1CQUFtQixFQUFFLENBQUMsS0FBSyxDQUFDO1lBQ3ZDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQztZQUNWLElBQUksRUFBRSxHQUFHLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDdkIsT0FBTyxDQUFDLEdBQUcsR0FBRyxDQUFDLE1BQU0sRUFBRSxDQUFDO2dCQUN0QixJQUFJLElBQUksR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDO2dCQUN6QixFQUFFLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQztvQkFDUixNQUFNLENBQUMsS0FBSyxDQUFDO2dCQUNmLElBQUksR0FBRyxJQUFJLENBQUMsSUFBSSxDQUFDO2dCQUNqQixFQUFFLEdBQUcsR0FBRyxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1lBQ3ZCLENBQUM7WUFDRCxNQUFNLENBQUMsSUFBSSxDQUFDO1FBQ2QsQ0FBQyxDQUFBQTtRQUVEQSxBQUNBQSw2REFENkRBO1lBQ3pEQSxTQUFTQSxHQUFHQSxVQUFTQSxNQUFNQSxFQUFFQSxNQUFNQTtZQUNyQyxJQUFJLElBQUksR0FBRyxDQUFDLENBQUM7WUFDYixJQUFJLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQztZQUNaLElBQUksSUFBSSxHQUFHLENBQUMsQ0FBQztZQUNiLElBQUksRUFBRSxHQUFHLENBQUMsQ0FBQyxDQUFDO1lBQ1osR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDO2dCQUNyQixFQUFFLENBQUMsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQztvQkFDckIsSUFBSSxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztvQkFDakIsRUFBRSxHQUFHLENBQUMsQ0FBQztnQkFDVCxDQUFDO2dCQUNELEVBQUUsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDO29CQUNyQixJQUFJLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO29CQUNqQixFQUFFLEdBQUcsQ0FBQyxDQUFDO2dCQUNULENBQUM7WUFDSCxDQUFDO1lBRUQsTUFBTSxDQUFDLEVBQUUsSUFBSSxFQUFFLENBQUM7UUFDbEIsQ0FBQyxDQUFBQTtRQUVEQSxJQUFJQSxTQUFTQSxHQUFHQSxDQUFDQSxDQUFDQTtRQUNsQkEsSUFBSUEsS0FBS0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7UUFDZEEsSUFBSUEsS0FBS0EsR0FBR0E7WUFDVkEsR0FBR0EsRUFBRUEsQ0FBQ0E7WUFDTkEsR0FBR0EsRUFBRUEsQ0FBQ0E7WUFDTkEsR0FBR0EsRUFBRUEsQ0FBQ0E7WUFDTkEsR0FBR0EsRUFBRUEsQ0FBQ0E7WUFDTkEsR0FBR0EsRUFBRUEsQ0FBQ0E7WUFDTkEsR0FBR0EsRUFBRUEsQ0FBQ0E7U0FDUEEsQ0FBQUE7UUFFREEsSUFBSUEsS0FBS0EsR0FBR0EsSUFBSUEsQ0FBQ0EsR0FBR0EsRUFBRUEsQ0FBQ0E7UUFDdkJBLE9BQU9BLFNBQVNBLEdBQUdBLFVBQVVBLElBQUlBLEtBQUtBLEdBQUdBLFNBQVNBLEVBQUVBLENBQUNBO1lBQ25EQSxJQUFJQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQTtZQUNWQSxLQUFLQSxHQUFHQSxDQUFDQSxDQUFDQTtZQUVWQSxBQUNBQSx3QkFEd0JBO2dCQUNwQkEsUUFBUUEsR0FBR0EsUUFBUUEsRUFBRUEsQ0FBQ0E7WUFFMUJBLEFBQ0FBLFFBRFFBO2dCQUNKQSxJQUFJQSxHQUFHQSxRQUFRQSxDQUFDQSxNQUFNQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUM5QkEsQUFDQUEsU0FEU0E7Z0JBQ0xBLE9BQU9BLEdBQUdBLFFBQVFBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLENBQUNBO1lBR3JDQSxPQUFPQSxDQUFDQSxHQUFHQSxRQUFRQSxDQUFDQSxNQUFNQSxHQUFHQSxDQUFDQSxFQUFFQSxDQUFDQTtnQkFDL0JBLElBQUlBLEtBQUtBLEdBQUdBLEVBQUVBLENBQUNBO2dCQUNmQSxJQUFJQSxNQUFNQSxHQUFHQSxFQUFFQSxDQUFDQTtnQkFDaEJBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLEVBQUVBLENBQUNBLEdBQUdBLENBQUNBLEVBQUVBLENBQUNBLEVBQUVBLEVBQUVBLENBQUNBO29CQUMzQkEsS0FBS0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7b0JBQ2JBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBO2dCQUNoQkEsQ0FBQ0E7Z0JBQ0RBLEtBQUtBLENBQUNBLEtBQUtBLENBQUNBLElBQUlBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBO2dCQUN2QkEsTUFBTUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7Z0JBRTNCQSxJQUFJQSxNQUFNQSxHQUFHQSxJQUFJQSxDQUFDQSxPQUFPQSxDQUFDQSxRQUFRQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQTtnQkFFMUNBLEVBQUVBLENBQUNBLENBQUNBLFNBQVNBLENBQUNBLE1BQU1BLEVBQUVBLE1BQU1BLENBQUNBLENBQUNBO29CQUM1QkEsSUFBSUEsQ0FBQ0EsT0FBT0EsQ0FBQ0EsU0FBU0EsQ0FBQ0EsSUFBSUEsRUFBRUEsTUFBTUEsQ0FBQ0EsQ0FBQ0E7Z0JBRXZDQSxJQUFJQSxHQUFHQSxRQUFRQSxDQUFDQSxNQUFNQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDNUJBLE9BQU9BLEdBQUdBLFFBQVFBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLENBQUNBO2dCQUVqQ0EsSUFBSUEsS0FBS0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ2RBLEdBQUdBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBLElBQUlBLE1BQU1BLENBQUNBO29CQUNuQkEsS0FBS0EsSUFBSUEsSUFBSUEsQ0FBQ0EsR0FBR0EsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsTUFBTUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0EsQ0FBQUE7Z0JBQzdDQSxLQUFLQSxJQUFJQSxNQUFNQSxDQUFDQSxNQUFNQSxDQUFDQTtnQkFFdkJBLEtBQUtBLElBQUlBLEtBQUtBLENBQUNBO1lBQ2pCQSxDQUFDQTtZQUNEQSxLQUFLQSxJQUFJQSxRQUFRQSxDQUFDQSxNQUFNQSxDQUFDQTtZQUN6QkEsU0FBU0EsRUFBRUEsQ0FBQ0E7WUFDWkEsRUFBRUEsQ0FBQ0EsQ0FBQ0EsU0FBU0EsR0FBR0EsR0FBR0EsSUFBSUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ3pCQSxPQUFPQSxDQUFDQSxHQUFHQSxDQUFDQSxhQUFhQSxFQUFFQSxTQUFTQSxFQUFFQSxRQUFRQSxFQUFFQSxJQUFJQSxDQUFDQSxHQUFHQSxFQUFFQSxHQUFHQSxLQUFLQSxFQUNoRUEsU0FBU0EsRUFBRUEsS0FBS0EsQ0FBQ0EsQ0FBQ0E7WUFDdEJBLENBQUNBO1FBQ0hBLENBQUNBO1FBRURBLE1BQU1BLENBQUNBO1lBQ0xBLFVBQVVBLEVBQUVBLFNBQVNBO1lBQ3JCQSxLQUFLQSxFQUFFQSxLQUFLQTtZQUNaQSxJQUFJQSxFQUFFQSxJQUFJQSxDQUFDQSxHQUFHQSxFQUFFQSxHQUFHQSxLQUFLQTtZQUN4QkEsSUFBSUEsRUFBRUEsSUFBSUE7WUFDVkEsUUFBUUEsRUFBRUEsUUFBUUE7U0FDbkJBLENBQUFBO0lBQ0hBLENBQUNBO0lBRUhWLGNBQUNBO0FBQURBLENBMWtCQSxBQTBrQkNBLElBQUE7QUExa0JZLGVBQU8sR0FBUCxPQTBrQlosQ0FBQTtBQUVELElBQWMsT0FBTyxDQTRCcEI7QUE1QkQsV0FBYyxPQUFPLEVBQUMsQ0FBQztJQU9WQSxZQUFJQSxHQUFHQTtRQUNoQkEsQUFDQUEsUUFEUUE7UUFDUkEsYUFBYUEsRUFBRUEsVUFBU0EsTUFBTUEsRUFBRUEsTUFBTUE7WUFDcEMsSUFBSSxZQUFZLEdBQUcsQ0FBQyxDQUFDO1lBQ3JCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksTUFBTSxDQUFDO2dCQUNuQixZQUFZLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsS0FBSyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEdBQUcsS0FBSyxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUNwSCxNQUFNLENBQUMsWUFBWSxDQUFDO1FBQ3RCLENBQUM7UUFDREEscUJBQXFCQSxFQUFFQSxVQUFTQSxNQUFNQSxFQUFFQSxNQUFNQTtZQUM1QyxJQUFJLFlBQVksR0FBRyxDQUFDLENBQUM7WUFDckIsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxNQUFNLENBQUM7Z0JBQ25CLFlBQVksSUFBSSxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsS0FBSyxDQUFDLENBQUM7WUFDMUQsTUFBTSxDQUFDLFlBQVksQ0FBQztRQUN0QixDQUFDO1FBQ0RBLEdBQUdBLEVBQUVBLFVBQVNBLE1BQU1BLEVBQUVBLE1BQU1BO1lBQzFCLElBQUksR0FBRyxHQUFHLENBQUMsQ0FBQztZQUNaLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksTUFBTSxDQUFDO2dCQUNuQixHQUFHLElBQUksSUFBSSxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1lBQzVDLE1BQU0sQ0FBQyxHQUFHLEdBQUcsTUFBTSxDQUFDLE1BQU0sQ0FBQztRQUM3QixDQUFDO0tBQ0ZBLENBQUFBO0FBQ0hBLENBQUNBLEVBNUJhLE9BQU8sR0FBUCxlQUFPLEtBQVAsZUFBTyxRQTRCcEIiLCJmaWxlIjoic3JjL3RyYWluZXIuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgbmV0ID0gcmVxdWlyZSgnLi9uZXR3b3JrJyk7XG5cbi8qKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgVFJBSU5FUlxuKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKi9cblxuZXhwb3J0IGNsYXNzIFRyYWluZXIge1xuICBuZXR3b3JrOiBuZXQuTmV0d29yaztcbiAgcmF0ZTogYW55ID0gLjI7XG4gIGl0ZXJhdGlvbnMgPSAxMDAwMDA7XG4gIGVycm9yID0gLjAwNTtcbiAgY29zdDogVHJhaW5lci5JVHJhaW5lckNvc3RGbjtcbiAgc2NoZWR1bGU6IGFueTtcblxuICBjb25zdHJ1Y3RvcihuZXR3b3JrOiBuZXQuTmV0d29yaywgb3B0aW9ucz86IGFueSkge1xuICAgIG9wdGlvbnMgPSBvcHRpb25zIHx8IHt9O1xuICAgIHRoaXMubmV0d29yayA9IG5ldHdvcms7XG4gICAgdGhpcy5yYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4yO1xuICAgIHRoaXMuaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucyB8fCAxMDAwMDA7XG4gICAgdGhpcy5lcnJvciA9IG9wdGlvbnMuZXJyb3IgfHwgLjAwNVxuICAgIHRoaXMuY29zdCA9IG9wdGlvbnMuY29zdCB8fCBUcmFpbmVyLmNvc3QuQ1JPU1NfRU5UUk9QWTtcbiAgfVxuXG4gIC8vIHRyYWlucyBhbnkgZ2l2ZW4gc2V0IHRvIGEgbmV0d29ya1xuICB0cmFpbihzZXQsIG9wdGlvbnMpIHtcblxuICAgIHZhciBlcnJvciA9IDE7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSAwLCBidWNrZXRTaXplID0gMDtcbiAgICB2YXIgYWJvcnRfdHJhaW5pbmcgPSBmYWxzZTtcbiAgICB2YXIgaW5wdXQsIG91dHB1dCwgdGFyZ2V0LCBjdXJyZW50UmF0ZTtcblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICBpZiAob3B0aW9ucykge1xuICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSkge1xuICAgICAgICAvLysgSm9uYXMgUmFvbmkgU29hcmVzIFNpbHZhXG4gICAgICAgIC8vQCBodHRwOi8vanNmcm9taGVsbC5jb20vYXJyYXkvc2h1ZmZsZSBbdjEuMF1cbiAgICAgICAgZnVuY3Rpb24gc2h1ZmZsZShvKSB7IC8vdjEuMFxuICAgICAgICAgIGZvciAodmFyIGosIHgsIGkgPSBvLmxlbmd0aDsgaTsgaiA9IE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqIGkpLCB4ID0gb1stLWldLCBvW2ldID0gb1tqXSwgb1tqXSA9IHgpO1xuICAgICAgICAgIHJldHVybiBvO1xuICAgICAgICB9O1xuICAgICAgfVxuICAgICAgaWYgKG9wdGlvbnMuaXRlcmF0aW9ucylcbiAgICAgICAgdGhpcy5pdGVyYXRpb25zID0gb3B0aW9ucy5pdGVyYXRpb25zO1xuICAgICAgaWYgKG9wdGlvbnMuZXJyb3IpXG4gICAgICAgIHRoaXMuZXJyb3IgPSBvcHRpb25zLmVycm9yO1xuICAgICAgaWYgKG9wdGlvbnMucmF0ZSlcbiAgICAgICAgdGhpcy5yYXRlID0gb3B0aW9ucy5yYXRlO1xuICAgICAgaWYgKG9wdGlvbnMuY29zdClcbiAgICAgICAgdGhpcy5jb3N0ID0gb3B0aW9ucy5jb3N0O1xuICAgICAgaWYgKG9wdGlvbnMuc2NoZWR1bGUpXG4gICAgICAgIHRoaXMuc2NoZWR1bGUgPSBvcHRpb25zLnNjaGVkdWxlO1xuICAgICAgaWYgKG9wdGlvbnMuY3VzdG9tTG9nKSB7XG4gICAgICAgIC8vIGZvciBiYWNrd2FyZCBjb21wYXRpYmlsaXR5IHdpdGggY29kZSB0aGF0IHVzZWQgY3VzdG9tTG9nXG4gICAgICAgIGNvbnNvbGUubG9nKCdEZXByZWNhdGVkOiB1c2Ugc2NoZWR1bGUgaW5zdGVhZCBvZiBjdXN0b21Mb2cnKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5jdXN0b21Mb2c7XG4gICAgICB9XG4gICAgfVxuXG4gICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGU7XG4gICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5yYXRlKSkge1xuICAgICAgYnVja2V0U2l6ZSA9IE1hdGguZmxvb3IodGhpcy5pdGVyYXRpb25zIC8gdGhpcy5yYXRlLmxlbmd0aCk7XG4gICAgfVxuXG5cbiAgICB3aGlsZSAoIWFib3J0X3RyYWluaW5nICYmIGl0ZXJhdGlvbnMgPCB0aGlzLml0ZXJhdGlvbnMgJiYgZXJyb3IgPiB0aGlzLmVycm9yKSB7XG4gICAgICBlcnJvciA9IDA7XG5cbiAgICAgIGlmIChidWNrZXRTaXplID4gMCkge1xuICAgICAgICB2YXIgY3VycmVudEJ1Y2tldCA9IE1hdGguZmxvb3IoaXRlcmF0aW9ucyAvIGJ1Y2tldFNpemUpO1xuICAgICAgICBjdXJyZW50UmF0ZSA9IHRoaXMucmF0ZVtjdXJyZW50QnVja2V0XTtcbiAgICAgIH1cblxuICAgICAgZm9yICh2YXIgdHJhaW4gaW4gc2V0KSB7XG4gICAgICAgIGlucHV0ID0gc2V0W3RyYWluXS5pbnB1dDtcbiAgICAgICAgdGFyZ2V0ID0gc2V0W3RyYWluXS5vdXRwdXQ7XG5cbiAgICAgICAgb3V0cHV0ID0gdGhpcy5uZXR3b3JrLmFjdGl2YXRlKGlucHV0KTtcbiAgICAgICAgdGhpcy5uZXR3b3JrLnByb3BhZ2F0ZShjdXJyZW50UmF0ZSwgdGFyZ2V0KTtcblxuICAgICAgICBlcnJvciArPSB0aGlzLmNvc3QodGFyZ2V0LCBvdXRwdXQpO1xuICAgICAgfVxuXG4gICAgICAvLyBjaGVjayBlcnJvclxuICAgICAgaXRlcmF0aW9ucysrO1xuICAgICAgZXJyb3IgLz0gc2V0Lmxlbmd0aDtcblxuICAgICAgaWYgKG9wdGlvbnMpIHtcbiAgICAgICAgaWYgKHRoaXMuc2NoZWR1bGUgJiYgdGhpcy5zY2hlZHVsZS5ldmVyeSAmJiBpdGVyYXRpb25zICUgdGhpcy5zY2hlZHVsZS5ldmVyeSA9PSAwKSB7XG5cbiAgICAgICAgICBhYm9ydF90cmFpbmluZyA9IHRoaXMuc2NoZWR1bGUuZG8oe1xuICAgICAgICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgICAgICAgaXRlcmF0aW9uczogaXRlcmF0aW9ucyxcbiAgICAgICAgICAgIHJhdGU6IGN1cnJlbnRSYXRlXG4gICAgICAgICAgfSk7XG4gICAgIFxuICAgICAgICB9IGVsc2UgaWYgKG9wdGlvbnMubG9nICYmIGl0ZXJhdGlvbnMgJSBvcHRpb25zLmxvZyA9PSAwKSB7XG4gICAgICAgICAgY29uc29sZS5sb2coJ2l0ZXJhdGlvbnMnLCBpdGVyYXRpb25zLCAnZXJyb3InLCBlcnJvciwgJ3JhdGUnLCBjdXJyZW50UmF0ZSwgJ1Q6JywgdGFyZ2V0LCAnTzonLCBvdXRwdXQpO1xuICAgICAgICB9O1xuICAgICAgICBpZiAob3B0aW9ucy5zaHVmZmxlKVxuICAgICAgICAgIHNodWZmbGUoc2V0KTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB2YXIgcmVzdWx0cyA9IHtcbiAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgIGl0ZXJhdGlvbnM6IGl0ZXJhdGlvbnMsXG4gICAgICB0aW1lOiBEYXRlLm5vdygpIC0gc3RhcnRcbiAgICB9XG5cbiAgICByZXR1cm4gcmVzdWx0cztcbiAgfVxuXG4gIC8vIHRyYWlucyBhbnkgZ2l2ZW4gc2V0IHRvIGEgbmV0d29yayB1c2luZyBhIFdlYldvcmtlclxuICB3b3JrZXJUcmFpbihzZXQsIGNhbGxiYWNrLCBvcHRpb25zKSB7XG5cbiAgICB2YXIgdGhhdCA9IHRoaXM7XG4gICAgdmFyIGVycm9yID0gMTtcbiAgICB2YXIgaXRlcmF0aW9ucyA9IDAsIGJ1Y2tldFNpemUgPSAwO1xuICAgIHZhciBpbnB1dCwgb3V0cHV0LCB0YXJnZXQsIGN1cnJlbnRSYXRlO1xuICAgIHZhciBsZW5ndGggPSBzZXQubGVuZ3RoO1xuICAgIHZhciBhYm9ydF90cmFpbmluZyA9IGZhbHNlO1xuXG4gICAgdmFyIHN0YXJ0ID0gRGF0ZS5ub3coKTtcblxuICAgIGlmIChvcHRpb25zKSB7XG4gICAgICBpZiAob3B0aW9ucy5zaHVmZmxlKSB7XG4gICAgICAgIC8vKyBKb25hcyBSYW9uaSBTb2FyZXMgU2lsdmFcbiAgICAgICAgLy9AIGh0dHA6Ly9qc2Zyb21oZWxsLmNvbS9hcnJheS9zaHVmZmxlIFt2MS4wXVxuICAgICAgICBmdW5jdGlvbiBzaHVmZmxlKG8pIHsgLy92MS4wXG4gICAgICAgICAgZm9yICh2YXIgaiwgeCwgaSA9IG8ubGVuZ3RoOyBpOyBqID0gTWF0aC5mbG9vcihNYXRoLnJhbmRvbSgpICpcbiAgICAgICAgICAgIGkpLCB4ID0gb1stLWldLCBvW2ldID0gb1tqXSwgb1tqXSA9IHgpO1xuICAgICAgICAgIHJldHVybiBvO1xuICAgICAgICB9O1xuICAgICAgfVxuICAgICAgaWYgKG9wdGlvbnMuaXRlcmF0aW9ucylcbiAgICAgICAgdGhpcy5pdGVyYXRpb25zID0gb3B0aW9ucy5pdGVyYXRpb25zO1xuICAgICAgaWYgKG9wdGlvbnMuZXJyb3IpXG4gICAgICAgIHRoaXMuZXJyb3IgPSBvcHRpb25zLmVycm9yO1xuICAgICAgaWYgKG9wdGlvbnMucmF0ZSlcbiAgICAgICAgdGhpcy5yYXRlID0gb3B0aW9ucy5yYXRlO1xuICAgICAgaWYgKG9wdGlvbnMuY29zdClcbiAgICAgICAgdGhpcy5jb3N0ID0gb3B0aW9ucy5jb3N0O1xuICAgICAgaWYgKG9wdGlvbnMuc2NoZWR1bGUpXG4gICAgICAgIHRoaXMuc2NoZWR1bGUgPSBvcHRpb25zLnNjaGVkdWxlO1xuICAgICAgaWYgKG9wdGlvbnMuY3VzdG9tTG9nKVxuICAgICAgICAvLyBmb3IgYmFja3dhcmQgY29tcGF0aWJpbGl0eSB3aXRoIGNvZGUgdGhhdCB1c2VkIGN1c3RvbUxvZ1xuICAgICAgICBjb25zb2xlLmxvZygnRGVwcmVjYXRlZDogdXNlIHNjaGVkdWxlIGluc3RlYWQgb2YgY3VzdG9tTG9nJylcbiAgICAgIHRoaXMuc2NoZWR1bGUgPSBvcHRpb25zLmN1c3RvbUxvZztcbiAgICB9XG5cbiAgICAvLyBkeW5hbWljIGxlYXJuaW5nIHJhdGVcbiAgICBjdXJyZW50UmF0ZSA9IHRoaXMucmF0ZTtcbiAgICBpZiAoQXJyYXkuaXNBcnJheSh0aGlzLnJhdGUpKSB7XG4gICAgICBidWNrZXRTaXplID0gTWF0aC5mbG9vcih0aGlzLml0ZXJhdGlvbnMgLyB0aGlzLnJhdGUubGVuZ3RoKTtcbiAgICB9XG5cbiAgICAvLyBjcmVhdGUgYSB3b3JrZXJcbiAgICB2YXIgd29ya2VyID0gdGhpcy5uZXR3b3JrLndvcmtlcigpO1xuXG4gICAgLy8gYWN0aXZhdGUgdGhlIG5ldHdvcmtcbiAgICBmdW5jdGlvbiBhY3RpdmF0ZVdvcmtlcihpbnB1dCkge1xuICAgICAgd29ya2VyLnBvc3RNZXNzYWdlKHtcbiAgICAgICAgYWN0aW9uOiBcImFjdGl2YXRlXCIsXG4gICAgICAgIGlucHV0OiBpbnB1dCxcbiAgICAgICAgbWVtb3J5QnVmZmVyOiB0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeVxuICAgICAgfSwgW3RoYXQubmV0d29yay5vcHRpbWl6ZWQubWVtb3J5LmJ1ZmZlcl0pO1xuICAgIH1cblxuICAgIC8vIGJhY2twcm9wYWdhdGUgdGhlIG5ldHdvcmtcbiAgICBmdW5jdGlvbiBwcm9wYWdhdGVXb3JrZXIodGFyZ2V0KSB7XG4gICAgICBpZiAoYnVja2V0U2l6ZSA+IDApIHtcbiAgICAgICAgdmFyIGN1cnJlbnRCdWNrZXQgPSBNYXRoLmZsb29yKGl0ZXJhdGlvbnMgLyBidWNrZXRTaXplKTtcbiAgICAgICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGVbY3VycmVudEJ1Y2tldF07XG4gICAgICB9XG4gICAgICB3b3JrZXIucG9zdE1lc3NhZ2Uoe1xuICAgICAgICBhY3Rpb246IFwicHJvcGFnYXRlXCIsXG4gICAgICAgIHRhcmdldDogdGFyZ2V0LFxuICAgICAgICByYXRlOiBjdXJyZW50UmF0ZSxcbiAgICAgICAgbWVtb3J5QnVmZmVyOiB0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeVxuICAgICAgfSwgW3RoYXQubmV0d29yay5vcHRpbWl6ZWQubWVtb3J5LmJ1ZmZlcl0pO1xuICAgIH1cblxuICAgIC8vIHRyYWluIHRoZSB3b3JrZXJcbiAgICB3b3JrZXIub25tZXNzYWdlID0gZnVuY3Rpb24oZSkge1xuICAgICAgLy8gZ2l2ZSBjb250cm9sIG9mIHRoZSBtZW1vcnkgYmFjayB0byB0aGUgbmV0d29ya1xuICAgICAgdGhhdC5uZXR3b3JrLm9wdGltaXplZC5vd25lcnNoaXAoZS5kYXRhLm1lbW9yeUJ1ZmZlcik7XG5cbiAgICAgIGlmIChlLmRhdGEuYWN0aW9uID09IFwicHJvcGFnYXRlXCIpIHtcbiAgICAgICAgaWYgKGluZGV4ID49IGxlbmd0aCkge1xuICAgICAgICAgIGluZGV4ID0gMDtcbiAgICAgICAgICBpdGVyYXRpb25zKys7XG4gICAgICAgICAgZXJyb3IgLz0gc2V0Lmxlbmd0aDtcblxuICAgICAgICAgIC8vIGxvZ1xuICAgICAgICAgIGlmIChvcHRpb25zKSB7XG4gICAgICAgICAgICBpZiAodGhpcy5zY2hlZHVsZSAmJiB0aGlzLnNjaGVkdWxlLmV2ZXJ5ICYmIGl0ZXJhdGlvbnMgJSB0aGlzLnNjaGVkdWxlLmV2ZXJ5ID09IDApXG4gICAgICAgICAgICAgIGFib3J0X3RyYWluaW5nID0gdGhpcy5zY2hlZHVsZS5kbyh7XG4gICAgICAgICAgICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgICAgICAgICAgIGl0ZXJhdGlvbnM6IGl0ZXJhdGlvbnNcbiAgICAgICAgICAgICAgfSk7XG4gICAgICAgICAgICBlbHNlIGlmIChvcHRpb25zLmxvZyAmJiBpdGVyYXRpb25zICUgb3B0aW9ucy5sb2cgPT0gMCkge1xuICAgICAgICAgICAgICBjb25zb2xlLmxvZygnaXRlcmF0aW9ucycsIGl0ZXJhdGlvbnMsICdlcnJvcicsIGVycm9yKTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICBpZiAob3B0aW9ucy5zaHVmZmxlKVxuICAgICAgICAgICAgICBzaHVmZmxlKHNldCk7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgaWYgKCFhYm9ydF90cmFpbmluZyAmJiBpdGVyYXRpb25zIDwgdGhhdC5pdGVyYXRpb25zICYmIGVycm9yID4gdGhhdC5lcnJvcikge1xuICAgICAgICAgICAgYWN0aXZhdGVXb3JrZXIoc2V0W2luZGV4XS5pbnB1dCk7XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIC8vIGNhbGxiYWNrXG4gICAgICAgICAgICBjYWxsYmFjayh7XG4gICAgICAgICAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgICAgICAgICAgaXRlcmF0aW9uczogaXRlcmF0aW9ucyxcbiAgICAgICAgICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0XG4gICAgICAgICAgICB9KVxuICAgICAgICAgIH1cbiAgICAgICAgICBlcnJvciA9IDA7XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgYWN0aXZhdGVXb3JrZXIoc2V0W2luZGV4XS5pbnB1dCk7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgaWYgKGUuZGF0YS5hY3Rpb24gPT0gXCJhY3RpdmF0ZVwiKSB7XG4gICAgICAgIGVycm9yICs9IHRoYXQuY29zdChzZXRbaW5kZXhdLm91dHB1dCwgZS5kYXRhLm91dHB1dCk7XG4gICAgICAgIHByb3BhZ2F0ZVdvcmtlcihzZXRbaW5kZXhdLm91dHB1dCk7XG4gICAgICAgIGluZGV4Kys7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8ga2ljayBpdFxuICAgIHZhciBpbmRleCA9IDA7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSAwO1xuICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICB9XG5cbiAgLy8gdHJhaW5zIGFuIFhPUiB0byB0aGUgbmV0d29ya1xuICBYT1Iob3B0aW9ucykge1xuXG4gICAgaWYgKHRoaXMubmV0d29yay5pbnB1dHMoKSAhPSAyIHx8IHRoaXMubmV0d29yay5vdXRwdXRzKCkgIT0gMSlcbiAgICAgIHRocm93IFwiRXJyb3I6IEluY29tcGF0aWJsZSBuZXR3b3JrICgyIGlucHV0cywgMSBvdXRwdXQpXCI7XG5cbiAgICB2YXIgZGVmYXVsdHMgPSB7XG4gICAgICBpdGVyYXRpb25zOiAxMDAwMDAsXG4gICAgICBsb2c6IGZhbHNlLFxuICAgICAgc2h1ZmZsZTogdHJ1ZSxcbiAgICAgIGNvc3Q6IFRyYWluZXIuY29zdC5NU0VcbiAgICB9XG5cbiAgICBpZiAob3B0aW9ucylcbiAgICAgIGZvciAodmFyIGkgaW4gb3B0aW9ucylcbiAgICAgICAgZGVmYXVsdHNbaV0gPSBvcHRpb25zW2ldO1xuXG4gICAgcmV0dXJuIHRoaXMudHJhaW4oW3tcbiAgICAgIGlucHV0OiBbMCwgMF0sXG4gICAgICBvdXRwdXQ6IFswXVxuICAgIH0sIHtcbiAgICAgICAgaW5wdXQ6IFsxLCAwXSxcbiAgICAgICAgb3V0cHV0OiBbMV1cbiAgICAgIH0sIHtcbiAgICAgICAgaW5wdXQ6IFswLCAxXSxcbiAgICAgICAgb3V0cHV0OiBbMV1cbiAgICAgIH0sIHtcbiAgICAgICAgaW5wdXQ6IFsxLCAxXSxcbiAgICAgICAgb3V0cHV0OiBbMF1cbiAgICAgIH1dLCBkZWZhdWx0cyk7XG4gIH1cblxuICAvLyB0cmFpbnMgdGhlIG5ldHdvcmsgdG8gcGFzcyBhIERpc3RyYWN0ZWQgU2VxdWVuY2UgUmVjYWxsIHRlc3RcbiAgRFNSKG9wdGlvbnMpIHtcbiAgICBvcHRpb25zID0gb3B0aW9ucyB8fCB7fTtcblxuICAgIHZhciB0YXJnZXRzID0gb3B0aW9ucy50YXJnZXRzIHx8IFsyLCA0LCA3LCA4XTtcbiAgICB2YXIgZGlzdHJhY3RvcnMgPSBvcHRpb25zLmRpc3RyYWN0b3JzIHx8IFszLCA1LCA2LCA5XTtcbiAgICB2YXIgcHJvbXB0cyA9IG9wdGlvbnMucHJvbXB0cyB8fCBbMCwgMV07XG4gICAgdmFyIGxlbmd0aCA9IG9wdGlvbnMubGVuZ3RoIHx8IDI0O1xuICAgIHZhciBjcml0ZXJpb24gPSBvcHRpb25zLnN1Y2Nlc3MgfHwgMC45NTtcbiAgICB2YXIgaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucyB8fCAxMDAwMDA7XG4gICAgdmFyIHJhdGUgPSBvcHRpb25zLnJhdGUgfHwgLjE7XG4gICAgdmFyIGxvZyA9IG9wdGlvbnMubG9nIHx8IDA7XG4gICAgdmFyIHNjaGVkdWxlID0gb3B0aW9ucy5zY2hlZHVsZSB8fCB7fTtcbiAgICB2YXIgY29ycmVjdCA9IDA7XG4gICAgdmFyIGkgPSAwO1xuICAgIHZhciBzdWNjZXNzID0gMDtcbiAgICB2YXIgdHJpYWwgPSBpID0gY29ycmVjdCA9IGogPSBzdWNjZXNzID0gMCxcbiAgICAgIGVycm9yID0gMSxcbiAgICAgIHN5bWJvbHMgPSB0YXJnZXRzLmxlbmd0aCArIGRpc3RyYWN0b3JzLmxlbmd0aCArIHByb21wdHMubGVuZ3RoO1xuXG4gICAgdmFyIG5vUmVwZWF0ID0gZnVuY3Rpb24ocmFuZ2UsIGF2b2lkKSB7XG4gICAgICB2YXIgbnVtYmVyID0gTWF0aC5yYW5kb20oKSAqIHJhbmdlIHwgMDtcbiAgICAgIHZhciB1c2VkID0gZmFsc2U7XG4gICAgICBmb3IgKHZhciBpIGluIGF2b2lkKVxuICAgICAgICBpZiAobnVtYmVyID09IGF2b2lkW2ldKVxuICAgICAgICAgIHVzZWQgPSB0cnVlO1xuICAgICAgcmV0dXJuIHVzZWQgPyBub1JlcGVhdChyYW5nZSwgYXZvaWQpIDogbnVtYmVyO1xuICAgIH1cblxuICAgIHZhciBlcXVhbCA9IGZ1bmN0aW9uKHByZWRpY3Rpb24sIG91dHB1dCkge1xuICAgICAgZm9yICh2YXIgaSBpbiBwcmVkaWN0aW9uKVxuICAgICAgICBpZiAoTWF0aC5yb3VuZChwcmVkaWN0aW9uW2ldKSAhPSBvdXRwdXRbaV0pXG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuXG4gICAgdmFyIHN0YXJ0ID0gRGF0ZS5ub3coKTtcblxuICAgIHdoaWxlICh0cmlhbCA8IGl0ZXJhdGlvbnMgJiYgKHN1Y2Nlc3MgPCBjcml0ZXJpb24gfHwgdHJpYWwgJSAxMDAwICE9IDApKSB7XG4gICAgICAvLyBnZW5lcmF0ZSBzZXF1ZW5jZVxuICAgICAgdmFyIHNlcXVlbmNlID0gW10sXG4gICAgICAgIHNlcXVlbmNlTGVuZ3RoID0gbGVuZ3RoIC0gcHJvbXB0cy5sZW5ndGg7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgc2VxdWVuY2VMZW5ndGg7IGkrKykge1xuICAgICAgICB2YXIgYW55ID0gTWF0aC5yYW5kb20oKSAqIGRpc3RyYWN0b3JzLmxlbmd0aCB8IDA7XG4gICAgICAgIHNlcXVlbmNlLnB1c2goZGlzdHJhY3RvcnNbYW55XSk7XG4gICAgICB9XG4gICAgICB2YXIgaW5kZXhlcyA9IFtdLFxuICAgICAgICBwb3NpdGlvbnMgPSBbXTtcbiAgICAgIGZvciAoaSA9IDA7IGkgPCBwcm9tcHRzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIGluZGV4ZXMucHVzaChNYXRoLnJhbmRvbSgpICogdGFyZ2V0cy5sZW5ndGggfCAwKTtcbiAgICAgICAgcG9zaXRpb25zLnB1c2gobm9SZXBlYXQoc2VxdWVuY2VMZW5ndGgsIHBvc2l0aW9ucykpO1xuICAgICAgfVxuICAgICAgcG9zaXRpb25zID0gcG9zaXRpb25zLnNvcnQoKTtcbiAgICAgIGZvciAoaSA9IDA7IGkgPCBwcm9tcHRzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIHNlcXVlbmNlW3Bvc2l0aW9uc1tpXV0gPSB0YXJnZXRzW2luZGV4ZXNbaV1dO1xuICAgICAgICBzZXF1ZW5jZS5wdXNoKHByb21wdHNbaV0pO1xuICAgICAgfVxuXG4gICAgICAvL3RyYWluIHNlcXVlbmNlXG4gICAgICB2YXIgZGlzdHJhY3RvcnNDb3JyZWN0O1xuICAgICAgdmFyIHRhcmdldHNDb3JyZWN0ID0gZGlzdHJhY3RvcnNDb3JyZWN0ID0gMDtcbiAgICAgIGVycm9yID0gMDtcbiAgICAgIGZvciAoaSA9IDA7IGkgPCBsZW5ndGg7IGkrKykge1xuICAgICAgICAvLyBnZW5lcmF0ZSBpbnB1dCBmcm9tIHNlcXVlbmNlXG4gICAgICAgIHZhciBpbnB1dCA9IFtdO1xuICAgICAgICBmb3IgKGogPSAwOyBqIDwgc3ltYm9sczsgaisrKVxuICAgICAgICAgIGlucHV0W2pdID0gMDtcbiAgICAgICAgaW5wdXRbc2VxdWVuY2VbaV1dID0gMTtcblxuICAgICAgICAvLyBnZW5lcmF0ZSB0YXJnZXQgb3V0cHV0XG4gICAgICAgIHZhciBvdXRwdXQgPSBbXTtcbiAgICAgICAgZm9yIChqID0gMDsgaiA8IHRhcmdldHMubGVuZ3RoOyBqKyspXG4gICAgICAgICAgb3V0cHV0W2pdID0gMDtcblxuICAgICAgICBpZiAoaSA+PSBzZXF1ZW5jZUxlbmd0aCkge1xuICAgICAgICAgIHZhciBpbmRleCA9IGkgLSBzZXF1ZW5jZUxlbmd0aDtcbiAgICAgICAgICBvdXRwdXRbaW5kZXhlc1tpbmRleF1dID0gMTtcbiAgICAgICAgfVxuXG4gICAgICAgIC8vIGNoZWNrIHJlc3VsdFxuICAgICAgICB2YXIgcHJlZGljdGlvbiA9IHRoaXMubmV0d29yay5hY3RpdmF0ZShpbnB1dCk7XG5cbiAgICAgICAgaWYgKGVxdWFsKHByZWRpY3Rpb24sIG91dHB1dCkpXG4gICAgICAgICAgaWYgKGkgPCBzZXF1ZW5jZUxlbmd0aClcbiAgICAgICAgICAgIGRpc3RyYWN0b3JzQ29ycmVjdCsrO1xuICAgICAgICAgIGVsc2VcbiAgICAgICAgICAgIHRhcmdldHNDb3JyZWN0Kys7XG4gICAgICAgIGVsc2Uge1xuICAgICAgICAgIHRoaXMubmV0d29yay5wcm9wYWdhdGUocmF0ZSwgb3V0cHV0KTtcbiAgICAgICAgfVxuXG4gICAgICAgIHZhciBkZWx0YSA9IDA7XG4gICAgICAgIGZvciAodmFyIGogaW4gcHJlZGljdGlvbilcbiAgICAgICAgICBkZWx0YSArPSBNYXRoLnBvdyhvdXRwdXRbal0gLSBwcmVkaWN0aW9uW2pdLCAyKTtcbiAgICAgICAgZXJyb3IgKz0gZGVsdGEgLyB0aGlzLm5ldHdvcmsub3V0cHV0cygpO1xuXG4gICAgICAgIGlmIChkaXN0cmFjdG9yc0NvcnJlY3QgKyB0YXJnZXRzQ29ycmVjdCA9PSBsZW5ndGgpXG4gICAgICAgICAgY29ycmVjdCsrO1xuICAgICAgfVxuXG4gICAgICAvLyBjYWxjdWxhdGUgZXJyb3JcbiAgICAgIGlmICh0cmlhbCAlIDEwMDAgPT0gMClcbiAgICAgICAgY29ycmVjdCA9IDA7XG4gICAgICB0cmlhbCsrO1xuICAgICAgdmFyIGRpdmlkZUVycm9yID0gdHJpYWwgJSAxMDAwO1xuICAgICAgZGl2aWRlRXJyb3IgPSBkaXZpZGVFcnJvciA9PSAwID8gMTAwMCA6IGRpdmlkZUVycm9yO1xuICAgICAgc3VjY2VzcyA9IGNvcnJlY3QgLyBkaXZpZGVFcnJvcjtcbiAgICAgIGVycm9yIC89IGxlbmd0aDtcblxuICAgICAgLy8gbG9nXG4gICAgICBpZiAobG9nICYmIHRyaWFsICUgbG9nID09IDApXG4gICAgICAgIGNvbnNvbGUubG9nKFwiaXRlcmF0aW9uczpcIiwgdHJpYWwsIFwiIHN1Y2Nlc3M6XCIsIHN1Y2Nlc3MsIFwiIGNvcnJlY3Q6XCIsXG4gICAgICAgICAgY29ycmVjdCwgXCIgdGltZTpcIiwgRGF0ZS5ub3coKSAtIHN0YXJ0LCBcIiBlcnJvcjpcIiwgZXJyb3IpO1xuICAgICAgaWYgKHNjaGVkdWxlLmRvICYmIHNjaGVkdWxlLmV2ZXJ5ICYmIHRyaWFsICUgc2NoZWR1bGUuZXZlcnkgPT0gMCkge1xuICAgICAgICBzY2hlZHVsZS5kbyh7XG4gICAgICAgICAgaXRlcmF0aW9uczogdHJpYWwsXG4gICAgICAgICAgc3VjY2Vzczogc3VjY2VzcyxcbiAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0LFxuICAgICAgICAgIGNvcnJlY3Q6IGNvcnJlY3RcbiAgICAgICAgfSk7XG5cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4ge1xuICAgICAgaXRlcmF0aW9uczogdHJpYWwsXG4gICAgICBzdWNjZXNzOiBzdWNjZXNzLFxuICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0XG4gICAgfVxuICB9XG5cbiAgLy8gdHJhaW4gdGhlIG5ldHdvcmsgdG8gbGVhcm4gYW4gRW1iZWRlZCBSZWJlciBHcmFtbWFyXG4gIEVSRyhvcHRpb25zKSB7XG5cbiAgICBvcHRpb25zID0gb3B0aW9ucyB8fCB7fTtcbiAgICB2YXIgaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucyB8fCAxNTAwMDA7XG4gICAgdmFyIGNyaXRlcmlvbiA9IG9wdGlvbnMuZXJyb3IgfHwgLjA1O1xuICAgIHZhciByYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4xO1xuICAgIHZhciBsb2cgPSBvcHRpb25zLmxvZyB8fCA1MDA7XG5cbiAgICAvLyBncmFtYXIgbm9kZVxuICAgIHZhciBOb2RlID0gZnVuY3Rpb24oKSB7XG4gICAgICB0aGlzLnBhdGhzID0gW107XG4gICAgfVxuICAgIE5vZGUucHJvdG90eXBlID0ge1xuICAgICAgY29ubmVjdDogZnVuY3Rpb24obm9kZSwgdmFsdWUpIHtcbiAgICAgICAgdGhpcy5wYXRocy5wdXNoKHtcbiAgICAgICAgICBub2RlOiBub2RlLFxuICAgICAgICAgIHZhbHVlOiB2YWx1ZVxuICAgICAgICB9KTtcbiAgICAgICAgcmV0dXJuIHRoaXM7XG4gICAgICB9LFxuICAgICAgYW55OiBmdW5jdGlvbigpIHtcbiAgICAgICAgaWYgKHRoaXMucGF0aHMubGVuZ3RoID09IDApXG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICB2YXIgaW5kZXggPSBNYXRoLnJhbmRvbSgpICogdGhpcy5wYXRocy5sZW5ndGggfCAwO1xuICAgICAgICByZXR1cm4gdGhpcy5wYXRoc1tpbmRleF07XG4gICAgICB9LFxuICAgICAgdGVzdDogZnVuY3Rpb24odmFsdWUpIHtcbiAgICAgICAgZm9yICh2YXIgaSBpbiB0aGlzLnBhdGhzKVxuICAgICAgICAgIGlmICh0aGlzLnBhdGhzW2ldLnZhbHVlID09IHZhbHVlKVxuICAgICAgICAgICAgcmV0dXJuIHRoaXMucGF0aHNbaV07XG4gICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB2YXIgcmViZXJHcmFtbWFyID0gZnVuY3Rpb24oKSB7XG5cbiAgICAgIC8vIGJ1aWxkIGEgcmViZXIgZ3JhbW1hclxuICAgICAgdmFyIG91dHB1dCA9IG5ldyBOb2RlKCk7XG4gICAgICB2YXIgbjEgPSAobmV3IE5vZGUoKSkuY29ubmVjdChvdXRwdXQsIFwiRVwiKTtcbiAgICAgIHZhciBuMiA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4xLCBcIlNcIik7XG4gICAgICB2YXIgbjMgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuMSwgXCJWXCIpLmNvbm5lY3QobjIsIFwiUFwiKTtcbiAgICAgIHZhciBuNCA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4yLCBcIlhcIilcbiAgICAgIG40LmNvbm5lY3QobjQsIFwiU1wiKTtcbiAgICAgIHZhciBuNSA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4zLCBcIlZcIilcbiAgICAgIG41LmNvbm5lY3QobjUsIFwiVFwiKTtcbiAgICAgIG4yLmNvbm5lY3QobjUsIFwiWFwiKVxuICAgICAgdmFyIG42ID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjQsIFwiVFwiKS5jb25uZWN0KG41LCBcIlBcIik7XG4gICAgICB2YXIgaW5wdXQgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuNiwgXCJCXCIpXG5cbiAgICAgIHJldHVybiB7XG4gICAgICAgIGlucHV0OiBpbnB1dCxcbiAgICAgICAgb3V0cHV0OiBvdXRwdXRcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBidWlsZCBhbiBlbWJlZGVkIHJlYmVyIGdyYW1tYXJcbiAgICB2YXIgZW1iZWRlZFJlYmVyR3JhbW1hciA9IGZ1bmN0aW9uKCkge1xuICAgICAgdmFyIHJlYmVyMSA9IHJlYmVyR3JhbW1hcigpO1xuICAgICAgdmFyIHJlYmVyMiA9IHJlYmVyR3JhbW1hcigpO1xuXG4gICAgICB2YXIgb3V0cHV0ID0gbmV3IE5vZGUoKTtcbiAgICAgIHZhciBuMSA9IChuZXcgTm9kZSkuY29ubmVjdChvdXRwdXQsIFwiRVwiKTtcbiAgICAgIHJlYmVyMS5vdXRwdXQuY29ubmVjdChuMSwgXCJUXCIpO1xuICAgICAgcmViZXIyLm91dHB1dC5jb25uZWN0KG4xLCBcIlBcIik7XG4gICAgICB2YXIgbjIgPSAobmV3IE5vZGUpLmNvbm5lY3QocmViZXIxLmlucHV0LCBcIlBcIikuY29ubmVjdChyZWJlcjIuaW5wdXQsXG4gICAgICAgIFwiVFwiKTtcbiAgICAgIHZhciBpbnB1dCA9IChuZXcgTm9kZSkuY29ubmVjdChuMiwgXCJCXCIpO1xuXG4gICAgICByZXR1cm4ge1xuICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgIG91dHB1dDogb3V0cHV0XG4gICAgICB9XG5cbiAgICB9XG5cbiAgICAvLyBnZW5lcmF0ZSBhbiBFUkcgc2VxdWVuY2VcbiAgICB2YXIgZ2VuZXJhdGUgPSBmdW5jdGlvbigpIHtcbiAgICAgIHZhciBub2RlID0gZW1iZWRlZFJlYmVyR3JhbW1hcigpLmlucHV0O1xuICAgICAgdmFyIG5leHQgPSBub2RlLmFueSgpO1xuICAgICAgdmFyIHN0ciA9IFwiXCI7XG4gICAgICB3aGlsZSAobmV4dCkge1xuICAgICAgICBzdHIgKz0gbmV4dC52YWx1ZTtcbiAgICAgICAgbmV4dCA9IG5leHQubm9kZS5hbnkoKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBzdHI7XG4gICAgfVxuXG4gICAgLy8gdGVzdCBpZiBhIHN0cmluZyBtYXRjaGVzIGFuIGVtYmVkZWQgcmViZXIgZ3JhbW1hclxuICAgIHZhciB0ZXN0ID0gZnVuY3Rpb24oc3RyKSB7XG4gICAgICB2YXIgbm9kZSA9IGVtYmVkZWRSZWJlckdyYW1tYXIoKS5pbnB1dDtcbiAgICAgIHZhciBpID0gMDtcbiAgICAgIHZhciBjaCA9IHN0ci5jaGFyQXQoaSk7XG4gICAgICB3aGlsZSAoaSA8IHN0ci5sZW5ndGgpIHtcbiAgICAgICAgdmFyIG5leHQgPSBub2RlLnRlc3QoY2gpO1xuICAgICAgICBpZiAoIW5leHQpXG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICBub2RlID0gbmV4dC5ub2RlO1xuICAgICAgICBjaCA9IHN0ci5jaGFyQXQoKytpKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIC8vIGhlbHBlciB0byBjaGVjayBpZiB0aGUgb3V0cHV0IGFuZCB0aGUgdGFyZ2V0IHZlY3RvcnMgbWF0Y2hcbiAgICB2YXIgZGlmZmVyZW50ID0gZnVuY3Rpb24oYXJyYXkxLCBhcnJheTIpIHtcbiAgICAgIHZhciBtYXgxID0gMDtcbiAgICAgIHZhciBpMSA9IC0xO1xuICAgICAgdmFyIG1heDIgPSAwO1xuICAgICAgdmFyIGkyID0gLTE7XG4gICAgICBmb3IgKHZhciBpIGluIGFycmF5MSkge1xuICAgICAgICBpZiAoYXJyYXkxW2ldID4gbWF4MSkge1xuICAgICAgICAgIG1heDEgPSBhcnJheTFbaV07XG4gICAgICAgICAgaTEgPSBpO1xuICAgICAgICB9XG4gICAgICAgIGlmIChhcnJheTJbaV0gPiBtYXgyKSB7XG4gICAgICAgICAgbWF4MiA9IGFycmF5MltpXTtcbiAgICAgICAgICBpMiA9IGk7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgcmV0dXJuIGkxICE9IGkyO1xuICAgIH1cblxuICAgIHZhciBpdGVyYXRpb24gPSAwO1xuICAgIHZhciBlcnJvciA9IDE7XG4gICAgdmFyIHRhYmxlID0ge1xuICAgICAgXCJCXCI6IDAsXG4gICAgICBcIlBcIjogMSxcbiAgICAgIFwiVFwiOiAyLFxuICAgICAgXCJYXCI6IDMsXG4gICAgICBcIlNcIjogNCxcbiAgICAgIFwiRVwiOiA1XG4gICAgfVxuXG4gICAgdmFyIHN0YXJ0ID0gRGF0ZS5ub3coKTtcbiAgICB3aGlsZSAoaXRlcmF0aW9uIDwgaXRlcmF0aW9ucyAmJiBlcnJvciA+IGNyaXRlcmlvbikge1xuICAgICAgdmFyIGkgPSAwO1xuICAgICAgZXJyb3IgPSAwO1xuXG4gICAgICAvLyBFUkcgc2VxdWVuY2UgdG8gbGVhcm5cbiAgICAgIHZhciBzZXF1ZW5jZSA9IGdlbmVyYXRlKCk7XG5cbiAgICAgIC8vIGlucHV0XG4gICAgICB2YXIgcmVhZCA9IHNlcXVlbmNlLmNoYXJBdChpKTtcbiAgICAgIC8vIHRhcmdldFxuICAgICAgdmFyIHByZWRpY3QgPSBzZXF1ZW5jZS5jaGFyQXQoaSArIDEpO1xuXG4gICAgICAvLyB0cmFpblxuICAgICAgd2hpbGUgKGkgPCBzZXF1ZW5jZS5sZW5ndGggLSAxKSB7XG4gICAgICAgIHZhciBpbnB1dCA9IFtdO1xuICAgICAgICB2YXIgdGFyZ2V0ID0gW107XG4gICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgNjsgaisrKSB7XG4gICAgICAgICAgaW5wdXRbal0gPSAwO1xuICAgICAgICAgIHRhcmdldFtqXSA9IDA7XG4gICAgICAgIH1cbiAgICAgICAgaW5wdXRbdGFibGVbcmVhZF1dID0gMTtcbiAgICAgICAgdGFyZ2V0W3RhYmxlW3ByZWRpY3RdXSA9IDE7XG5cbiAgICAgICAgdmFyIG91dHB1dCA9IHRoaXMubmV0d29yay5hY3RpdmF0ZShpbnB1dCk7XG5cbiAgICAgICAgaWYgKGRpZmZlcmVudChvdXRwdXQsIHRhcmdldCkpXG4gICAgICAgICAgdGhpcy5uZXR3b3JrLnByb3BhZ2F0ZShyYXRlLCB0YXJnZXQpO1xuXG4gICAgICAgIHJlYWQgPSBzZXF1ZW5jZS5jaGFyQXQoKytpKTtcbiAgICAgICAgcHJlZGljdCA9IHNlcXVlbmNlLmNoYXJBdChpICsgMSk7XG5cbiAgICAgICAgdmFyIGRlbHRhID0gMDtcbiAgICAgICAgZm9yICh2YXIgayBpbiBvdXRwdXQpXG4gICAgICAgICAgZGVsdGEgKz0gTWF0aC5wb3codGFyZ2V0W2tdIC0gb3V0cHV0W2tdLCAyKVxuICAgICAgICBkZWx0YSAvPSBvdXRwdXQubGVuZ3RoO1xuXG4gICAgICAgIGVycm9yICs9IGRlbHRhO1xuICAgICAgfVxuICAgICAgZXJyb3IgLz0gc2VxdWVuY2UubGVuZ3RoO1xuICAgICAgaXRlcmF0aW9uKys7XG4gICAgICBpZiAoaXRlcmF0aW9uICUgbG9nID09IDApIHtcbiAgICAgICAgY29uc29sZS5sb2coXCJpdGVyYXRpb25zOlwiLCBpdGVyYXRpb24sIFwiIHRpbWU6XCIsIERhdGUubm93KCkgLSBzdGFydCxcbiAgICAgICAgICBcIiBlcnJvcjpcIiwgZXJyb3IpO1xuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb24sXG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICB0aW1lOiBEYXRlLm5vdygpIC0gc3RhcnQsXG4gICAgICB0ZXN0OiB0ZXN0LFxuICAgICAgZ2VuZXJhdGU6IGdlbmVyYXRlXG4gICAgfVxuICB9XG5cbn1cblxuZXhwb3J0IG1vZHVsZSBUcmFpbmVyIHtcbiAgLy8gQnVpbHQtaW4gY29zdCBmdW5jdGlvbnNcbiAgXG4gIGV4cG9ydCBpbnRlcmZhY2UgSVRyYWluZXJDb3N0Rm4ge1xuICAgICh0YXJnZXQsIG91dHB1dCk6IG51bWJlcjtcbiAgfVxuXG4gIGV4cG9ydCB2YXIgY29zdCA9IHtcbiAgICAvLyBFcS4gOVxuICAgIENST1NTX0VOVFJPUFk6IGZ1bmN0aW9uKHRhcmdldCwgb3V0cHV0KSB7XG4gICAgICB2YXIgY3Jvc3NlbnRyb3B5ID0gMDtcbiAgICAgIGZvciAodmFyIGkgaW4gb3V0cHV0KVxuICAgICAgICBjcm9zc2VudHJvcHkgLT0gKHRhcmdldFtpXSAqIE1hdGgubG9nKG91dHB1dFtpXSArIDFlLTE1KSkgKyAoKDEgLSB0YXJnZXRbaV0pICogTWF0aC5sb2coKDEgKyAxZS0xNSkgLSBvdXRwdXRbaV0pKTsgLy8gKzFlLTE1IGlzIGEgdGlueSBwdXNoIGF3YXkgdG8gYXZvaWQgTWF0aC5sb2coMClcbiAgICAgIHJldHVybiBjcm9zc2VudHJvcHk7XG4gICAgfSxcbiAgICBDUk9TU19FTlRST1BZX1NPRlRNQVg6IGZ1bmN0aW9uKHRhcmdldCwgb3V0cHV0KSB7XG4gICAgICB2YXIgY3Jvc3NlbnRyb3B5ID0gMDtcbiAgICAgIGZvciAodmFyIGkgaW4gb3V0cHV0KVxuICAgICAgICBjcm9zc2VudHJvcHkgLT0gdGFyZ2V0W2ldICogTWF0aC5sb2cob3V0cHV0W2ldICsgMWUtMTUpO1xuICAgICAgcmV0dXJuIGNyb3NzZW50cm9weTtcbiAgICB9LFxuICAgIE1TRTogZnVuY3Rpb24odGFyZ2V0LCBvdXRwdXQpIHtcbiAgICAgIHZhciBtc2UgPSAwO1xuICAgICAgZm9yICh2YXIgaSBpbiBvdXRwdXQpXG4gICAgICAgIG1zZSArPSBNYXRoLnBvdyh0YXJnZXRbaV0gLSBvdXRwdXRbaV0sIDIpO1xuICAgICAgcmV0dXJuIG1zZSAvIG91dHB1dC5sZW5ndGg7XG4gICAgfVxuICB9XG59Il0sInNvdXJjZVJvb3QiOiIvc291cmNlLyJ9