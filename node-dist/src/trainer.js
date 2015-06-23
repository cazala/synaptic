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
        var _this = this;
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
                    setTimeout(function () {
                        _this.iterations -= iterations;
                        _this.train(set, options);
                    });
                    return;
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
        var _this = this;
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
                setTimeout(function () {
                    _this.iterations -= trial;
                    _this.DSR(options);
                });
                return;
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

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy90cmFpbmVyLnRzIl0sIm5hbWVzIjpbIlRyYWluZXIiLCJUcmFpbmVyLmNvbnN0cnVjdG9yIiwiVHJhaW5lci50cmFpbiIsIlRyYWluZXIudHJhaW4uc2h1ZmZsZSIsIlRyYWluZXIud29ya2VyVHJhaW4iLCJUcmFpbmVyLndvcmtlclRyYWluLnNodWZmbGUiLCJUcmFpbmVyLndvcmtlclRyYWluLmFjdGl2YXRlV29ya2VyIiwiVHJhaW5lci53b3JrZXJUcmFpbi5wcm9wYWdhdGVXb3JrZXIiLCJUcmFpbmVyLlhPUiIsIlRyYWluZXIuRFNSIiwiVHJhaW5lci5FUkciXSwibWFwcGluZ3MiOiJBQUVBLEFBSUE7OzRGQUY0RjtJQUUvRSxPQUFPO0lBUWxCQSxTQVJXQSxPQUFPQSxDQVFOQSxPQUFvQkEsRUFBRUEsT0FBYUE7UUFOL0NDLFNBQUlBLEdBQVFBLEVBQUVBLENBQUNBO1FBQ2ZBLGVBQVVBLEdBQUdBLE1BQU1BLENBQUNBO1FBQ3BCQSxVQUFLQSxHQUFHQSxJQUFJQSxDQUFDQTtRQUtYQSxPQUFPQSxHQUFHQSxPQUFPQSxJQUFJQSxFQUFFQSxDQUFDQTtRQUN4QkEsSUFBSUEsQ0FBQ0EsT0FBT0EsR0FBR0EsT0FBT0EsQ0FBQ0E7UUFDdkJBLElBQUlBLENBQUNBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLElBQUlBLEVBQUVBLENBQUNBO1FBQy9CQSxJQUFJQSxDQUFDQSxVQUFVQSxHQUFHQSxPQUFPQSxDQUFDQSxVQUFVQSxJQUFJQSxNQUFNQSxDQUFDQTtRQUMvQ0EsSUFBSUEsQ0FBQ0EsS0FBS0EsR0FBR0EsT0FBT0EsQ0FBQ0EsS0FBS0EsSUFBSUEsSUFBSUEsQ0FBQUE7UUFDbENBLElBQUlBLENBQUNBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLElBQUlBLE9BQU9BLENBQUNBLElBQUlBLENBQUNBLGFBQWFBLENBQUNBO0lBQ3pEQSxDQUFDQTtJQUVERCxvQ0FBb0NBO0lBQ3BDQSx1QkFBS0EsR0FBTEEsVUFBTUEs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SUEsQ0FBQ0EsTUFBTUEsRUFBRUEsR0FBR0EsV0FBV0EsQ0FBQ0EsTUFBTUEsR0FBR0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ2pEQSxRQUFRQSxDQUFDQSxJQUFJQSxDQUFDQSxXQUFXQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUNsQ0EsQ0FBQ0E7WUFDREEsSUFBSUEsT0FBT0EsR0FBR0EsRUFBRUEsRUFDZEEsU0FBU0EsR0FBR0EsRUFBRUEsQ0FBQ0E7WUFDakJBLEdBQUdBLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLEVBQUVBLENBQUNBLEdBQUdBLE9BQU9BLENBQUNBLE1BQU1BLEVBQUVBLENBQUNBLEVBQUVBLEVBQUVBLENBQUNBO2dCQUNwQ0EsT0FBT0EsQ0FBQ0EsSUFBSUEsQ0FBQ0EsSUFBSUEsQ0FBQ0EsTUFBTUEsRUFBRUEsR0FBR0EsT0FBT0EsQ0FBQ0EsTUFBTUEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ2pEQSxTQUFTQSxDQUFDQSxJQUFJQSxDQUFDQSxRQUFRQSxDQUFDQSxjQUFjQSxFQUFFQSxTQUFTQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUN0REEsQ0FBQ0E7WUFDREEsU0FBU0EsR0FBR0EsU0FBU0EsQ0FBQ0EsSUFBSUEsRUFBRUEsQ0FBQ0E7WUFDN0JBLEdBQUdBLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLEVBQUVBLENBQUNBLEdBQUdBLE9BQU9BLENBQUNBLE1BQU1BLEVBQUVBLENBQUNBLEVBQUVBLEVBQUVBLENBQUNBO2dCQUNwQ0EsUUFBUUEsQ0FBQ0EsU0FBU0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsT0FBT0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7Z0JBQzdDQSxRQUFRQSxDQUFDQSxJQUFJQSxDQUFDQSxPQUFPQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtZQUM1QkEsQ0FBQ0E7WUFFREEsQUFDQUEsZ0JBRGdCQTtnQkFDWkEsa0JBQWtCQSxDQUFDQTtZQUN2QkEsSUFBSUEsY0FBY0EsR0FBR0Esa0JBQWtCQSxHQUFHQSxDQUFDQSxDQUFDQTtZQUM1Q0EsS0FBS0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7WUFDVkEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsTUFBTUEsRUFBRUEsQ0FBQ0EsRUFBRUEsRUFBRUEsQ0FBQ0E7Z0JBQzVCQSxBQUNBQSwrQkFEK0JBO29CQUMzQkEsS0FBS0EsR0FBR0EsRUFBRUEsQ0FBQ0E7Z0JBQ2ZBLEdBQUdBLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLEVBQUVBLENBQUNBLEdBQUdBLE9BQU9BLEVBQUVBLENBQUNBLEVBQUVBO29CQUMxQkEsS0FBS0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ2ZBLEtBQUtBLENBQUNBLFFBQVFBLENBQUNBLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBO2dCQUV2QkEsQUFDQUEseUJBRHlCQTtvQkFDckJBLE1BQU1BLEdBQUdBLEVBQUVBLENBQUNBO2dCQUNoQkEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsT0FBT0EsQ0FBQ0EsTUFBTUEsRUFBRUEsQ0FBQ0EsRUFBRUE7b0JBQ2pDQSxNQUFNQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQTtnQkFFaEJBLEVBQUVBLENBQUNBLENBQUNBLENBQUNBLElBQUlBLGNBQWNBLENBQUNBLENBQUNBLENBQUNBO29CQUN4QkEsSUFBSUEsS0FBS0EsR0FBR0EsQ0FBQ0EsR0FBR0EsY0FBY0EsQ0FBQ0E7b0JBQy9CQSxNQUFNQSxDQUFDQSxPQUFPQSxDQUFDQSxLQUFLQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxDQUFDQTtnQkFDN0JBLENBQUNBO2dCQUVEQSxBQUNBQSxlQURlQTtvQkFDWEEsVUFBVUEsR0FBR0EsSUFBSUEsQ0FBQ0EsT0FBT0EsQ0FBQ0EsUUFBUUEsQ0FBQ0EsS0FBS0EsQ0FBQ0EsQ0FBQ0E7Z0JBRTlDQSxFQUFFQSxDQUFDQSxDQUFDQSxLQUFLQSxDQUFDQSxVQUFVQSxFQUFFQSxNQUFNQSxDQUFDQSxDQUFDQTtvQkFDNUJBLEVBQUVBLENBQUNBLENBQUNBLENBQUNBLEdBQUdBLGNBQWNBLENBQUNBO3dCQUNyQkEsa0JBQWtCQSxFQUFFQSxDQUFDQTtvQkFDdkJBLElBQUlBO3dCQUNGQSxjQUFjQSxFQUFFQSxDQUFDQTtnQkFDckJBLElBQUlBLENBQUNBLENBQUNBO29CQUNKQSxJQUFJQSxDQUFDQSxPQUFPQSxDQUFDQSxTQUFTQSxDQUFDQSxJQUFJQSxFQUFFQSxNQUFNQSxDQUFDQSxDQUFDQTtnQkFDdkNBLENBQUNBO2dCQUVEQSxJQUFJQSxLQUFLQSxHQUFHQSxDQUFDQSxDQUFDQTtnQkFDZEEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsSUFBSUEsVUFBVUEsQ0FBQ0E7b0JBQ3ZCQSxLQUFLQSxJQUFJQSxJQUFJQSxDQUFDQSxHQUFHQSxDQUFDQSxNQUFNQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxVQUFVQSxDQUFDQSxDQUFDQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDbERBLEtBQUtBLElBQUlBLEtBQUtBLEdBQUdBLElBQUlBLENBQUNBLE9BQU9BLENBQUNBLE9BQU9BLEVBQUVBLENBQUNBO2dCQUV4Q0EsRUFBRUEsQ0FBQ0EsQ0FBQ0Esa0JBQWtCQSxHQUFHQSxjQUFjQSxJQUFJQSxNQUFNQSxDQUFDQTtvQkFDaERBLE9BQU9BLEVBQUVBLENBQUNBO1lBQ2RBLENBQUNBO1lBRURBLEFBQ0FBLGtCQURrQkE7WUFDbEJBLEVBQUVBLENBQUNBLENBQUNBLEtBQUtBLEdBQUdBLElBQUlBLElBQUlBLENBQUNBLENBQUNBO2dCQUNwQkEsT0FBT0EsR0FBR0EsQ0FBQ0EsQ0FBQ0E7WUFDZEEsS0FBS0EsRUFBRUEsQ0FBQ0E7WUFDUkEsSUFBSUEsV0FBV0EsR0FBR0EsS0FBS0EsR0FBR0EsSUFBSUEsQ0FBQ0E7WUFDL0JBLFdBQVdBLEdBQUdBLFdBQVdBLElBQUlBLENBQUNBLEdBQUdBLElBQUlBLEdBQUdBLFdBQVdBLENBQUNBO1lBQ3BEQSxPQUFPQSxHQUFHQSxPQUFPQSxHQUFHQSxXQUFXQSxDQUFDQTtZQUNoQ0EsS0FBS0EsSUFBSUEsTUFBTUEsQ0FBQ0E7WUFFaEJBLEFBQ0FBLE1BRE1BO1lBQ05BLEVBQUVBLENBQUNBLENBQUNBLEdBQUdBLElBQUlBLEtBQUtBLEdBQUdBLEdBQUdBLElBQUlBLENBQUNBLENBQUNBO2dCQUMxQkEsT0FBT0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsYUFBYUEsRUFBRUEsS0FBS0EsRUFBRUEsV0FBV0EsRUFBRUEsT0FBT0EsRUFBRUEsV0FBV0EsRUFDakVBLE9BQU9BLEVBQUVBLFFBQVFBLEVBQUVBLElBQUlBLENBQUNBLEdBQUdBLEVBQUVBLEdBQUdBLEtBQUtBLEVBQUVBLFNBQVNBLEVBQUVBLEtBQUtBLENBQUNBLENBQUNBO1lBQzdEQSxFQUFFQSxDQUFDQSxDQUFDQSxRQUFRQSxDQUFDQSxFQUFFQSxJQUFJQSxRQUFRQSxDQUFDQSxLQUFLQSxJQUFJQSxLQUFLQSxHQUFHQSxRQUFRQSxDQUFDQSxLQUFLQSxJQUFJQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDakVBLFFBQVFBLENBQUNBLEVBQUVBLENBQUNBO29CQUNWQSxVQUFVQSxFQUFFQSxLQUFLQTtvQkFDakJBLE9BQU9BLEVBQUVBLE9BQU9BO29CQUNoQkEsS0FBS0EsRUFBRUEsS0FBS0E7b0JBQ1pBLElBQUlBLEVBQUVBLElBQUlBLENBQUNBLEdBQUdBLEVBQUVBLEdBQUdBLEtBQUtBO29CQUN4QkEsT0FBT0EsRUFBRUEsT0FBT0E7aUJBQ2pCQSxDQUFDQSxDQUFDQTtnQkFFSEEsVUFBVUEsQ0FBQ0E7b0JBQ1RBLEtBQUlBLENBQUNBLFVBQVVBLElBQUlBLEtBQUtBLENBQUNBO29CQUN6QkEsS0FBSUEsQ0FBQ0EsR0FBR0EsQ0FBQ0EsT0FBT0EsQ0FBQ0EsQ0FBQ0E7Z0JBQ3BCQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFFSEEsTUFBTUEsQ0FBQ0E7WUFDVEEsQ0FBQ0E7UUFDSEEsQ0FBQ0E7UUFFREEsTUFBTUEsQ0FBQ0E7WUFDTEEsVUFBVUEsRUFBRUEsS0FBS0E7WUFDakJBLE9BQU9BLEVBQUVBLE9BQU9BO1lBQ2hCQSxLQUFLQSxFQUFFQSxLQUFLQTtZQUNaQSxJQUFJQSxFQUFFQSxJQUFJQSxDQUFDQSxHQUFHQSxFQUFFQSxHQUFHQSxLQUFLQTtTQUN6QkEsQ0FBQUE7SUFDSEEsQ0FBQ0E7SUFFRFQsc0RBQXNEQTtJQUN0REEscUJBQUdBLEdBQUhBLFVBQUlBLE9BQU9BO1FBRVRVLE9BQU9BLEdBQUdBLE9BQU9BLElBQUlBLEVBQUVBLENBQUNBO1FBQ3hCQSxJQUFJQSxVQUFVQSxHQUFHQSxPQUFPQSxDQUFDQSxVQUFVQSxJQUFJQSxNQUFNQSxDQUFDQTtRQUM5Q0EsSUFBSUEsU0FBU0EsR0FBR0EsT0FBT0EsQ0FBQ0EsS0FBS0EsSUFBSUEsR0FBR0EsQ0FBQ0E7UUFDckNBLElBQUlBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLElBQUlBLEVBQUVBLENBQUNBO1FBQzlCQSxJQUFJQSxHQUFHQSxHQUFHQSxPQUFPQSxDQUFDQSxHQUFHQSxJQUFJQSxHQUFHQSxDQUFDQTtRQUU3QkEsQUFDQUEsY0FEY0E7WUFDVkEsSUFBSUEsR0FBR0E7WUFDVCxJQUFJLENBQUMsS0FBSyxHQUFHLEVBQUUsQ0FBQztRQUNsQixDQUFDLENBQUFBO1FBQ0RBLElBQUlBLENBQUNBLFNBQVNBLEdBQUdBO1lBQ2ZBLE9BQU9BLEVBQUVBLFVBQVNBLElBQUlBLEVBQUVBLEtBQUtBO2dCQUMzQixJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQztvQkFDZCxJQUFJLEVBQUUsSUFBSTtvQkFDVixLQUFLLEVBQUUsS0FBSztpQkFDYixDQUFDLENBQUM7Z0JBQ0gsTUFBTSxDQUFDLElBQUksQ0FBQztZQUNkLENBQUM7WUFDREEsR0FBR0EsRUFBRUE7Z0JBQ0gsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxNQUFNLElBQUksQ0FBQyxDQUFDO29CQUN6QixNQUFNLENBQUMsS0FBSyxDQUFDO2dCQUNmLElBQUksS0FBSyxHQUFHLElBQUksQ0FBQyxNQUFNLEVBQUUsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUM7Z0JBQ2xELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDO1lBQzNCLENBQUM7WUFDREEsSUFBSUEsRUFBRUEsVUFBU0EsS0FBS0E7Z0JBQ2xCLEdBQUcsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLEtBQUssQ0FBQztvQkFDdkIsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLElBQUksS0FBSyxDQUFDO3dCQUMvQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDekIsTUFBTSxDQUFDLEtBQUssQ0FBQztZQUNmLENBQUM7U0FDRkEsQ0FBQUE7UUFFREEsSUFBSUEsWUFBWUEsR0FBR0E7WUFFakIsQUFDQSx3QkFEd0I7Z0JBQ3BCLE1BQU0sR0FBRyxJQUFJLElBQUksRUFBRSxDQUFDO1lBQ3hCLElBQUksRUFBRSxHQUFHLENBQUMsSUFBSSxJQUFJLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDM0MsSUFBSSxFQUFFLEdBQUcsQ0FBQyxJQUFJLElBQUksRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQztZQUN2QyxJQUFJLEVBQUUsR0FBRyxDQUFDLElBQUksSUFBSSxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDeEQsSUFBSSxFQUFFLEdBQUcsQ0FBQyxJQUFJLElBQUksRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQTtZQUN0QyxFQUFFLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQztZQUNwQixJQUFJLEVBQUUsR0FBRyxDQUFDLElBQUksSUFBSSxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFBO1lBQ3RDLEVBQUUsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1lBQ3BCLEVBQUUsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFBO1lBQ25CLElBQUksRUFBRSxHQUFHLENBQUMsSUFBSSxJQUFJLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQztZQUN4RCxJQUFJLEtBQUssR0FBRyxDQUFDLElBQUksSUFBSSxFQUFFLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFBO1lBRXpDLE1BQU0sQ0FBQztnQkFDTCxLQUFLLEVBQUUsS0FBSztnQkFDWixNQUFNLEVBQUUsTUFBTTthQUNmLENBQUE7UUFDSCxDQUFDLENBQUFBO1FBRURBLEFBQ0FBLGlDQURpQ0E7WUFDN0JBLG1CQUFtQkEsR0FBR0E7WUFDeEIsSUFBSSxNQUFNLEdBQUcsWUFBWSxFQUFFLENBQUM7WUFDNUIsSUFBSSxNQUFNLEdBQUcsWUFBWSxFQUFFLENBQUM7WUFFNUIsSUFBSSxNQUFNLEdBQUcsSUFBSSxJQUFJLEVBQUUsQ0FBQztZQUN4QixJQUFJLEVBQUUsR0FBRyxDQUFDLElBQUksSUFBSSxDQUFDLENBQUMsT0FBTyxDQUFDLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQztZQUN6QyxNQUFNLENBQUMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsR0FBRyxDQUFDLENBQUM7WUFDL0IsTUFBTSxDQUFDLE1BQU0sQ0FBQyxPQUFPLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDO1lBQy9CLElBQUksRUFBRSxHQUFHLENBQUMsSUFBSSxJQUFJLENBQUMsQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLEtBQUssRUFBRSxHQUFHLENBQUMsQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLEtBQUssRUFDakUsR0FBRyxDQUFDLENBQUM7WUFDUCxJQUFJLEtBQUssR0FBRyxDQUFDLElBQUksSUFBSSxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQztZQUV4QyxNQUFNLENBQUM7Z0JBQ0wsS0FBSyxFQUFFLEtBQUs7Z0JBQ1osTUFBTSxFQUFFLE1BQU07YUFDZixDQUFBO1FBRUgsQ0FBQyxDQUFBQTtRQUVEQSxBQUNBQSwyQkFEMkJBO1lBQ3ZCQSxRQUFRQSxHQUFHQTtZQUNiLElBQUksSUFBSSxHQUFHLG1CQUFtQixFQUFFLENBQUMsS0FBSyxDQUFDO1lBQ3ZDLElBQUksSUFBSSxHQUFHLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQztZQUN0QixJQUFJLEdBQUcsR0FBRyxFQUFFLENBQUM7WUFDYixPQUFPLElBQUksRUFBRSxDQUFDO2dCQUNaLEdBQUcsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDO2dCQUNsQixJQUFJLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxHQUFHLEVBQUUsQ0FBQztZQUN6QixDQUFDO1lBQ0QsTUFBTSxDQUFDLEdBQUcsQ0FBQztRQUNiLENBQUMsQ0FBQUE7UUFFREEsQUFDQUEsb0RBRG9EQTtZQUNoREEsSUFBSUEsR0FBR0EsVUFBU0EsR0FBR0E7WUFDckIsSUFBSSxJQUFJLEdBQUcsbUJBQW1CLEVBQUUsQ0FBQyxLQUFLLENBQUM7WUFDdkMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDO1lBQ1YsSUFBSSxFQUFFLEdBQUcsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN2QixPQUFPLENBQUMsR0FBRyxHQUFHLENBQUMsTUFBTSxFQUFFLENBQUM7Z0JBQ3RCLElBQUksSUFBSSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUM7Z0JBQ3pCLEVBQUUsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDO29CQUNSLE1BQU0sQ0FBQyxLQUFLLENBQUM7Z0JBQ2YsSUFBSSxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUM7Z0JBQ2pCLEVBQUUsR0FBRyxHQUFHLENBQUMsTUFBTSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7WUFDdkIsQ0FBQztZQUNELE1BQU0sQ0FBQyxJQUFJLENBQUM7UUFDZCxDQUFDLENBQUFBO1FBRURBLEFBQ0FBLDZEQUQ2REE7WUFDekRBLFNBQVNBLEdBQUdBLFVBQVNBLE1BQU1BLEVBQUVBLE1BQU1BO1lBQ3JDLElBQUksSUFBSSxHQUFHLENBQUMsQ0FBQztZQUNiLElBQUksRUFBRSxHQUFHLENBQUMsQ0FBQyxDQUFDO1lBQ1osSUFBSSxJQUFJLEdBQUcsQ0FBQyxDQUFDO1lBQ2IsSUFBSSxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUM7WUFDWixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLE1BQU0sQ0FBQyxDQUFDLENBQUM7Z0JBQ3JCLEVBQUUsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDO29CQUNyQixJQUFJLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO29CQUNqQixFQUFFLEdBQUcsQ0FBQyxDQUFDO2dCQUNULENBQUM7Z0JBQ0QsRUFBRSxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUM7b0JBQ3JCLElBQUksR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQ2pCLEVBQUUsR0FBRyxDQUFDLENBQUM7Z0JBQ1QsQ0FBQztZQUNILENBQUM7WUFFRCxNQUFNLENBQUMsRUFBRSxJQUFJLEVBQUUsQ0FBQztRQUNsQixDQUFDLENBQUFBO1FBRURBLElBQUlBLFNBQVNBLEdBQUdBLENBQUNBLENBQUNBO1FBQ2xCQSxJQUFJQSxLQUFLQSxHQUFHQSxDQUFDQSxDQUFDQTtRQUNkQSxJQUFJQSxLQUFLQSxHQUFHQTtZQUNWQSxHQUFHQSxFQUFFQSxDQUFDQTtZQUNOQSxHQUFHQSxFQUFFQSxDQUFDQTtZQUNOQSxHQUFHQSxFQUFFQSxDQUFDQTtZQUNOQSxHQUFHQSxFQUFFQSxDQUFDQTtZQUNOQSxHQUFHQSxFQUFFQSxDQUFDQTtZQUNOQSxHQUFHQSxFQUFFQSxDQUFDQTtTQUNQQSxDQUFBQTtRQUVEQSxJQUFJQSxLQUFLQSxHQUFHQSxJQUFJQSxDQUFDQSxHQUFHQSxFQUFFQSxDQUFDQTtRQUN2QkEsT0FBT0EsU0FBU0EsR0FBR0EsVUFBVUEsSUFBSUEsS0FBS0EsR0FBR0EsU0FBU0EsRUFBRUEsQ0FBQ0E7WUFDbkRBLElBQUlBLENBQUNBLEdBQUdBLENBQUNBLENBQUNBO1lBQ1ZBLEtBQUtBLEdBQUdBLENBQUNBLENBQUNBO1lBRVZBLEFBQ0FBLHdCQUR3QkE7Z0JBQ3BCQSxRQUFRQSxHQUFHQSxRQUFRQSxFQUFFQSxDQUFDQTtZQUUxQkEsQUFDQUEsUUFEUUE7Z0JBQ0pBLElBQUlBLEdBQUdBLFFBQVFBLENBQUNBLE1BQU1BLENBQUNBLENBQUNBLENBQUNBLENBQUNBO1lBQzlCQSxBQUNBQSxTQURTQTtnQkFDTEEsT0FBT0EsR0FBR0EsUUFBUUEsQ0FBQ0EsTUFBTUEsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7WUFHckNBLE9BQU9BLENBQUNBLEdBQUdBLFFBQVFBLENBQUNBLE1BQU1BLEdBQUdBLENBQUNBLEVBQUVBLENBQUNBO2dCQUMvQkEsSUFBSUEsS0FBS0EsR0FBR0EsRUFBRUEsQ0FBQ0E7Z0JBQ2ZBLElBQUlBLE1BQU1BLEdBQUdBLEVBQUVBLENBQUNBO2dCQUNoQkEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsRUFBRUEsRUFBRUEsQ0FBQ0E7b0JBQzNCQSxLQUFLQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxD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LENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsR0FBRyxLQUFLLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3BILE1BQU0sQ0FBQyxZQUFZLENBQUM7UUFDdEIsQ0FBQztRQUNEQSxHQUFHQSxFQUFFQSxVQUFTQSxNQUFNQSxFQUFFQSxNQUFNQTtZQUMxQixJQUFJLEdBQUcsR0FBRyxDQUFDLENBQUM7WUFDWixHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLE1BQU0sQ0FBQztnQkFDbkIsR0FBRyxJQUFJLElBQUksQ0FBQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztZQUM1QyxNQUFNLENBQUMsR0FBRyxHQUFHLE1BQU0sQ0FBQyxNQUFNLENBQUM7UUFDN0IsQ0FBQztLQUNGQSxDQUFBQTtBQUNIQSxDQUFDQSxFQXRCYSxPQUFPLEdBQVAsZUFBTyxLQUFQLGVBQU8sUUFzQnBCIiwiZmlsZSI6InNyYy90cmFpbmVyLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IG5ldCA9IHJlcXVpcmUoJy4vbmV0d29yaycpO1xuXG4vKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFRSQUlORVJcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cbmV4cG9ydCBjbGFzcyBUcmFpbmVyIHtcbiAgbmV0d29yazogbmV0Lk5ldHdvcms7XG4gIHJhdGU6IGFueSA9IC4yO1xuICBpdGVyYXRpb25zID0gMTAwMDAwO1xuICBlcnJvciA9IC4wMDU7XG4gIGNvc3Q6IFRyYWluZXIuSVRyYWluZXJDb3N0Rm47XG4gIHNjaGVkdWxlOiBhbnk7XG5cbiAgY29uc3RydWN0b3IobmV0d29yazogbmV0Lk5ldHdvcmssIG9wdGlvbnM/OiBhbnkpIHtcbiAgICBvcHRpb25zID0gb3B0aW9ucyB8fCB7fTtcbiAgICB0aGlzLm5ldHdvcmsgPSBuZXR3b3JrO1xuICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZSB8fCAuMjtcbiAgICB0aGlzLml0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnMgfHwgMTAwMDAwO1xuICAgIHRoaXMuZXJyb3IgPSBvcHRpb25zLmVycm9yIHx8IC4wMDVcbiAgICB0aGlzLmNvc3QgPSBvcHRpb25zLmNvc3QgfHwgVHJhaW5lci5jb3N0LkNST1NTX0VOVFJPUFk7XG4gIH1cblxuICAvLyB0cmFpbnMgYW55IGdpdmVuIHNldCB0byBhIG5ldHdvcmtcbiAgdHJhaW4oc2V0LCBvcHRpb25zKSB7XG5cbiAgICB2YXIgZXJyb3IgPSAxO1xuICAgIHZhciBpdGVyYXRpb25zID0gMCwgYnVja2V0U2l6ZSA9IDA7XG4gICAgdmFyIGFib3J0X3RyYWluaW5nID0gZmFsc2U7XG4gICAgdmFyIGlucHV0LCBvdXRwdXQsIHRhcmdldCwgY3VycmVudFJhdGU7XG5cbiAgICB2YXIgc3RhcnQgPSBEYXRlLm5vdygpO1xuXG4gICAgaWYgKG9wdGlvbnMpIHtcbiAgICAgIGlmIChvcHRpb25zLnNodWZmbGUpIHtcbiAgICAgICAgLy8rIEpvbmFzIFJhb25pIFNvYXJlcyBTaWx2YVxuICAgICAgICAvL0AgaHR0cDovL2pzZnJvbWhlbGwuY29tL2FycmF5L3NodWZmbGUgW3YxLjBdXG4gICAgICAgIGZ1bmN0aW9uIHNodWZmbGUobykgeyAvL3YxLjBcbiAgICAgICAgICBmb3IgKHZhciBqLCB4LCBpID0gby5sZW5ndGg7IGk7IGogPSBNYXRoLmZsb29yKE1hdGgucmFuZG9tKCkgKiBpKSwgeCA9IG9bLS1pXSwgb1tpXSA9IG9bal0sIG9bal0gPSB4KTtcbiAgICAgICAgICByZXR1cm4gbztcbiAgICAgICAgfTtcbiAgICAgIH1cbiAgICAgIGlmIChvcHRpb25zLml0ZXJhdGlvbnMpXG4gICAgICAgIHRoaXMuaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucztcbiAgICAgIGlmIChvcHRpb25zLmVycm9yKVxuICAgICAgICB0aGlzLmVycm9yID0gb3B0aW9ucy5lcnJvcjtcbiAgICAgIGlmIChvcHRpb25zLnJhdGUpXG4gICAgICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZTtcbiAgICAgIGlmIChvcHRpb25zLmNvc3QpXG4gICAgICAgIHRoaXMuY29zdCA9IG9wdGlvbnMuY29zdDtcbiAgICAgIGlmIChvcHRpb25zLnNjaGVkdWxlKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5zY2hlZHVsZTtcbiAgICAgIGlmIChvcHRpb25zLmN1c3RvbUxvZykge1xuICAgICAgICAvLyBmb3IgYmFja3dhcmQgY29tcGF0aWJpbGl0eSB3aXRoIGNvZGUgdGhhdCB1c2VkIGN1c3RvbUxvZ1xuICAgICAgICBjb25zb2xlLmxvZygnRGVwcmVjYXRlZDogdXNlIHNjaGVkdWxlIGluc3RlYWQgb2YgY3VzdG9tTG9nJylcbiAgICAgICAgdGhpcy5zY2hlZHVsZSA9IG9wdGlvbnMuY3VzdG9tTG9nO1xuICAgICAgfVxuICAgIH1cblxuICAgIGN1cnJlbnRSYXRlID0gdGhpcy5yYXRlO1xuICAgIGlmIChBcnJheS5pc0FycmF5KHRoaXMucmF0ZSkpIHtcbiAgICAgIGJ1Y2tldFNpemUgPSBNYXRoLmZsb29yKHRoaXMuaXRlcmF0aW9ucyAvIHRoaXMucmF0ZS5sZW5ndGgpO1xuICAgIH1cblxuXG4gICAgd2hpbGUgKCFhYm9ydF90cmFpbmluZyAmJiBpdGVyYXRpb25zIDwgdGhpcy5pdGVyYXRpb25zICYmIGVycm9yID4gdGhpcy5lcnJvcikge1xuICAgICAgZXJyb3IgPSAwO1xuXG4gICAgICBpZiAoYnVja2V0U2l6ZSA+IDApIHtcbiAgICAgICAgdmFyIGN1cnJlbnRCdWNrZXQgPSBNYXRoLmZsb29yKGl0ZXJhdGlvbnMgLyBidWNrZXRTaXplKTtcbiAgICAgICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGVbY3VycmVudEJ1Y2tldF07XG4gICAgICB9XG5cbiAgICAgIGZvciAodmFyIHRyYWluIGluIHNldCkge1xuICAgICAgICBpbnB1dCA9IHNldFt0cmFpbl0uaW5wdXQ7XG4gICAgICAgIHRhcmdldCA9IHNldFt0cmFpbl0ub3V0cHV0O1xuXG4gICAgICAgIG91dHB1dCA9IHRoaXMubmV0d29yay5hY3RpdmF0ZShpbnB1dCk7XG4gICAgICAgIHRoaXMubmV0d29yay5wcm9wYWdhdGUoY3VycmVudFJhdGUsIHRhcmdldCk7XG5cbiAgICAgICAgZXJyb3IgKz0gdGhpcy5jb3N0KHRhcmdldCwgb3V0cHV0KTtcbiAgICAgIH1cblxuICAgICAgLy8gY2hlY2sgZXJyb3JcbiAgICAgIGl0ZXJhdGlvbnMrKztcbiAgICAgIGVycm9yIC89IHNldC5sZW5ndGg7XG5cbiAgICAgIGlmIChvcHRpb25zKSB7XG4gICAgICAgIGlmICh0aGlzLnNjaGVkdWxlICYmIHRoaXMuc2NoZWR1bGUuZXZlcnkgJiYgaXRlcmF0aW9ucyAlIHRoaXMuc2NoZWR1bGUuZXZlcnkgPT0gMCkge1xuXG4gICAgICAgICAgYWJvcnRfdHJhaW5pbmcgPSB0aGlzLnNjaGVkdWxlLmRvKHtcbiAgICAgICAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgICAgICAgIGl0ZXJhdGlvbnM6IGl0ZXJhdGlvbnMsXG4gICAgICAgICAgICByYXRlOiBjdXJyZW50UmF0ZVxuICAgICAgICAgIH0pO1xuXG4gICAgICAgICAgc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICAgICAgICB0aGlzLml0ZXJhdGlvbnMgLT0gaXRlcmF0aW9ucztcbiAgICAgICAgICAgIHRoaXMudHJhaW4oc2V0LCBvcHRpb25zKTtcbiAgICAgICAgICB9KTtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH0gZWxzZSBpZiAob3B0aW9ucy5sb2cgJiYgaXRlcmF0aW9ucyAlIG9wdGlvbnMubG9nID09IDApIHtcbiAgICAgICAgICBjb25zb2xlLmxvZygnaXRlcmF0aW9ucycsIGl0ZXJhdGlvbnMsICdlcnJvcicsIGVycm9yLCAncmF0ZScsIGN1cnJlbnRSYXRlKTtcbiAgICAgICAgfTtcbiAgICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSlcbiAgICAgICAgICBzaHVmZmxlKHNldCk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdmFyIHJlc3VsdHMgPSB7XG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0XG4gICAgfVxuXG4gICAgcmV0dXJuIHJlc3VsdHM7XG4gIH1cblxuICAvLyB0cmFpbnMgYW55IGdpdmVuIHNldCB0byBhIG5ldHdvcmsgdXNpbmcgYSBXZWJXb3JrZXJcbiAgd29ya2VyVHJhaW4oc2V0LCBjYWxsYmFjaywgb3B0aW9ucykge1xuXG4gICAgdmFyIHRoYXQgPSB0aGlzO1xuICAgIHZhciBlcnJvciA9IDE7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSAwLCBidWNrZXRTaXplID0gMDtcbiAgICB2YXIgaW5wdXQsIG91dHB1dCwgdGFyZ2V0LCBjdXJyZW50UmF0ZTtcbiAgICB2YXIgbGVuZ3RoID0gc2V0Lmxlbmd0aDtcbiAgICB2YXIgYWJvcnRfdHJhaW5pbmcgPSBmYWxzZTtcblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICBpZiAob3B0aW9ucykge1xuICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSkge1xuICAgICAgICAvLysgSm9uYXMgUmFvbmkgU29hcmVzIFNpbHZhXG4gICAgICAgIC8vQCBodHRwOi8vanNmcm9taGVsbC5jb20vYXJyYXkvc2h1ZmZsZSBbdjEuMF1cbiAgICAgICAgZnVuY3Rpb24gc2h1ZmZsZShvKSB7IC8vdjEuMFxuICAgICAgICAgIGZvciAodmFyIGosIHgsIGkgPSBvLmxlbmd0aDsgaTsgaiA9IE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqXG4gICAgICAgICAgICBpKSwgeCA9IG9bLS1pXSwgb1tpXSA9IG9bal0sIG9bal0gPSB4KTtcbiAgICAgICAgICByZXR1cm4gbztcbiAgICAgICAgfTtcbiAgICAgIH1cbiAgICAgIGlmIChvcHRpb25zLml0ZXJhdGlvbnMpXG4gICAgICAgIHRoaXMuaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucztcbiAgICAgIGlmIChvcHRpb25zLmVycm9yKVxuICAgICAgICB0aGlzLmVycm9yID0gb3B0aW9ucy5lcnJvcjtcbiAgICAgIGlmIChvcHRpb25zLnJhdGUpXG4gICAgICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZTtcbiAgICAgIGlmIChvcHRpb25zLmNvc3QpXG4gICAgICAgIHRoaXMuY29zdCA9IG9wdGlvbnMuY29zdDtcbiAgICAgIGlmIChvcHRpb25zLnNjaGVkdWxlKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5zY2hlZHVsZTtcbiAgICAgIGlmIChvcHRpb25zLmN1c3RvbUxvZylcbiAgICAgICAgLy8gZm9yIGJhY2t3YXJkIGNvbXBhdGliaWxpdHkgd2l0aCBjb2RlIHRoYXQgdXNlZCBjdXN0b21Mb2dcbiAgICAgICAgY29uc29sZS5sb2coJ0RlcHJlY2F0ZWQ6IHVzZSBzY2hlZHVsZSBpbnN0ZWFkIG9mIGN1c3RvbUxvZycpXG4gICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5jdXN0b21Mb2c7XG4gICAgfVxuXG4gICAgLy8gZHluYW1pYyBsZWFybmluZyByYXRlXG4gICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGU7XG4gICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5yYXRlKSkge1xuICAgICAgYnVja2V0U2l6ZSA9IE1hdGguZmxvb3IodGhpcy5pdGVyYXRpb25zIC8gdGhpcy5yYXRlLmxlbmd0aCk7XG4gICAgfVxuXG4gICAgLy8gY3JlYXRlIGEgd29ya2VyXG4gICAgdmFyIHdvcmtlciA9IHRoaXMubmV0d29yay53b3JrZXIoKTtcblxuICAgIC8vIGFjdGl2YXRlIHRoZSBuZXR3b3JrXG4gICAgZnVuY3Rpb24gYWN0aXZhdGVXb3JrZXIoaW5wdXQpIHtcbiAgICAgIHdvcmtlci5wb3N0TWVzc2FnZSh7XG4gICAgICAgIGFjdGlvbjogXCJhY3RpdmF0ZVwiLFxuICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgIG1lbW9yeUJ1ZmZlcjogdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnlcbiAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyBiYWNrcHJvcGFnYXRlIHRoZSBuZXR3b3JrXG4gICAgZnVuY3Rpb24gcHJvcGFnYXRlV29ya2VyKHRhcmdldCkge1xuICAgICAgaWYgKGJ1Y2tldFNpemUgPiAwKSB7XG4gICAgICAgIHZhciBjdXJyZW50QnVja2V0ID0gTWF0aC5mbG9vcihpdGVyYXRpb25zIC8gYnVja2V0U2l6ZSk7XG4gICAgICAgIGN1cnJlbnRSYXRlID0gdGhpcy5yYXRlW2N1cnJlbnRCdWNrZXRdO1xuICAgICAgfVxuICAgICAgd29ya2VyLnBvc3RNZXNzYWdlKHtcbiAgICAgICAgYWN0aW9uOiBcInByb3BhZ2F0ZVwiLFxuICAgICAgICB0YXJnZXQ6IHRhcmdldCxcbiAgICAgICAgcmF0ZTogY3VycmVudFJhdGUsXG4gICAgICAgIG1lbW9yeUJ1ZmZlcjogdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnlcbiAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyB0cmFpbiB0aGUgd29ya2VyXG4gICAgd29ya2VyLm9ubWVzc2FnZSA9IGZ1bmN0aW9uKGUpIHtcbiAgICAgIC8vIGdpdmUgY29udHJvbCBvZiB0aGUgbWVtb3J5IGJhY2sgdG8gdGhlIG5ldHdvcmtcbiAgICAgIHRoYXQubmV0d29yay5vcHRpbWl6ZWQub3duZXJzaGlwKGUuZGF0YS5tZW1vcnlCdWZmZXIpO1xuXG4gICAgICBpZiAoZS5kYXRhLmFjdGlvbiA9PSBcInByb3BhZ2F0ZVwiKSB7XG4gICAgICAgIGlmIChpbmRleCA+PSBsZW5ndGgpIHtcbiAgICAgICAgICBpbmRleCA9IDA7XG4gICAgICAgICAgaXRlcmF0aW9ucysrO1xuICAgICAgICAgIGVycm9yIC89IHNldC5sZW5ndGg7XG5cbiAgICAgICAgICAvLyBsb2dcbiAgICAgICAgICBpZiAob3B0aW9ucykge1xuICAgICAgICAgICAgaWYgKHRoaXMuc2NoZWR1bGUgJiYgdGhpcy5zY2hlZHVsZS5ldmVyeSAmJiBpdGVyYXRpb25zICUgdGhpcy5zY2hlZHVsZS5ldmVyeSA9PSAwKVxuICAgICAgICAgICAgICBhYm9ydF90cmFpbmluZyA9IHRoaXMuc2NoZWR1bGUuZG8oe1xuICAgICAgICAgICAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zXG4gICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgZWxzZSBpZiAob3B0aW9ucy5sb2cgJiYgaXRlcmF0aW9ucyAlIG9wdGlvbnMubG9nID09IDApIHtcbiAgICAgICAgICAgICAgY29uc29sZS5sb2coJ2l0ZXJhdGlvbnMnLCBpdGVyYXRpb25zLCAnZXJyb3InLCBlcnJvcik7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSlcbiAgICAgICAgICAgICAgc2h1ZmZsZShzZXQpO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIGlmICghYWJvcnRfdHJhaW5pbmcgJiYgaXRlcmF0aW9ucyA8IHRoYXQuaXRlcmF0aW9ucyAmJiBlcnJvciA+IHRoYXQuZXJyb3IpIHtcbiAgICAgICAgICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAvLyBjYWxsYmFja1xuICAgICAgICAgICAgY2FsbGJhY2soe1xuICAgICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICAgIGl0ZXJhdGlvbnM6IGl0ZXJhdGlvbnMsXG4gICAgICAgICAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydFxuICAgICAgICAgICAgfSlcbiAgICAgICAgICB9XG4gICAgICAgICAgZXJyb3IgPSAwO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChlLmRhdGEuYWN0aW9uID09IFwiYWN0aXZhdGVcIikge1xuICAgICAgICBlcnJvciArPSB0aGF0LmNvc3Qoc2V0W2luZGV4XS5vdXRwdXQsIGUuZGF0YS5vdXRwdXQpO1xuICAgICAgICBwcm9wYWdhdGVXb3JrZXIoc2V0W2luZGV4XS5vdXRwdXQpO1xuICAgICAgICBpbmRleCsrO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIGtpY2sgaXRcbiAgICB2YXIgaW5kZXggPSAwO1xuICAgIHZhciBpdGVyYXRpb25zID0gMDtcbiAgICBhY3RpdmF0ZVdvcmtlcihzZXRbaW5kZXhdLmlucHV0KTtcbiAgfVxuXG4gIC8vIHRyYWlucyBhbiBYT1IgdG8gdGhlIG5ldHdvcmtcbiAgWE9SKG9wdGlvbnMpIHtcblxuICAgIGlmICh0aGlzLm5ldHdvcmsuaW5wdXRzKCkgIT0gMiB8fCB0aGlzLm5ldHdvcmsub3V0cHV0cygpICE9IDEpXG4gICAgICB0aHJvdyBcIkVycm9yOiBJbmNvbXBhdGlibGUgbmV0d29yayAoMiBpbnB1dHMsIDEgb3V0cHV0KVwiO1xuXG4gICAgdmFyIGRlZmF1bHRzID0ge1xuICAgICAgaXRlcmF0aW9uczogMTAwMDAwLFxuICAgICAgbG9nOiBmYWxzZSxcbiAgICAgIHNodWZmbGU6IHRydWUsXG4gICAgICBjb3N0OiBUcmFpbmVyLmNvc3QuTVNFXG4gICAgfVxuXG4gICAgaWYgKG9wdGlvbnMpXG4gICAgICBmb3IgKHZhciBpIGluIG9wdGlvbnMpXG4gICAgICAgIGRlZmF1bHRzW2ldID0gb3B0aW9uc1tpXTtcblxuICAgIHJldHVybiB0aGlzLnRyYWluKFt7XG4gICAgICBpbnB1dDogWzAsIDBdLFxuICAgICAgb3V0cHV0OiBbMF1cbiAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMSwgMF0sXG4gICAgICAgIG91dHB1dDogWzFdXG4gICAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMCwgMV0sXG4gICAgICAgIG91dHB1dDogWzFdXG4gICAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMSwgMV0sXG4gICAgICAgIG91dHB1dDogWzBdXG4gICAgICB9XSwgZGVmYXVsdHMpO1xuICB9XG5cbiAgLy8gdHJhaW5zIHRoZSBuZXR3b3JrIHRvIHBhc3MgYSBEaXN0cmFjdGVkIFNlcXVlbmNlIFJlY2FsbCB0ZXN0XG4gIERTUihvcHRpb25zKSB7XG4gICAgb3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG5cbiAgICB2YXIgdGFyZ2V0cyA9IG9wdGlvbnMudGFyZ2V0cyB8fCBbMiwgNCwgNywgOF07XG4gICAgdmFyIGRpc3RyYWN0b3JzID0gb3B0aW9ucy5kaXN0cmFjdG9ycyB8fCBbMywgNSwgNiwgOV07XG4gICAgdmFyIHByb21wdHMgPSBvcHRpb25zLnByb21wdHMgfHwgWzAsIDFdO1xuICAgIHZhciBsZW5ndGggPSBvcHRpb25zLmxlbmd0aCB8fCAyNDtcbiAgICB2YXIgY3JpdGVyaW9uID0gb3B0aW9ucy5zdWNjZXNzIHx8IDAuOTU7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnMgfHwgMTAwMDAwO1xuICAgIHZhciByYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4xO1xuICAgIHZhciBsb2cgPSBvcHRpb25zLmxvZyB8fCAwO1xuICAgIHZhciBzY2hlZHVsZSA9IG9wdGlvbnMuc2NoZWR1bGUgfHwge307XG4gICAgdmFyIGNvcnJlY3QgPSAwO1xuICAgIHZhciBpID0gMDtcbiAgICB2YXIgc3VjY2VzcyA9IDA7XG4gICAgdmFyIHRyaWFsID0gaSA9IGNvcnJlY3QgPSBqID0gc3VjY2VzcyA9IDAsXG4gICAgICBlcnJvciA9IDEsXG4gICAgICBzeW1ib2xzID0gdGFyZ2V0cy5sZW5ndGggKyBkaXN0cmFjdG9ycy5sZW5ndGggKyBwcm9tcHRzLmxlbmd0aDtcblxuICAgIHZhciBub1JlcGVhdCA9IGZ1bmN0aW9uKHJhbmdlLCBhdm9pZCkge1xuICAgICAgdmFyIG51bWJlciA9IE1hdGgucmFuZG9tKCkgKiByYW5nZSB8IDA7XG4gICAgICB2YXIgdXNlZCA9IGZhbHNlO1xuICAgICAgZm9yICh2YXIgaSBpbiBhdm9pZClcbiAgICAgICAgaWYgKG51bWJlciA9PSBhdm9pZFtpXSlcbiAgICAgICAgICB1c2VkID0gdHJ1ZTtcbiAgICAgIHJldHVybiB1c2VkID8gbm9SZXBlYXQocmFuZ2UsIGF2b2lkKSA6IG51bWJlcjtcbiAgICB9XG5cbiAgICB2YXIgZXF1YWwgPSBmdW5jdGlvbihwcmVkaWN0aW9uLCBvdXRwdXQpIHtcbiAgICAgIGZvciAodmFyIGkgaW4gcHJlZGljdGlvbilcbiAgICAgICAgaWYgKE1hdGgucm91bmQocHJlZGljdGlvbltpXSkgIT0gb3V0cHV0W2ldKVxuICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICB3aGlsZSAodHJpYWwgPCBpdGVyYXRpb25zICYmIChzdWNjZXNzIDwgY3JpdGVyaW9uIHx8IHRyaWFsICUgMTAwMCAhPSAwKSkge1xuICAgICAgLy8gZ2VuZXJhdGUgc2VxdWVuY2VcbiAgICAgIHZhciBzZXF1ZW5jZSA9IFtdLFxuICAgICAgICBzZXF1ZW5jZUxlbmd0aCA9IGxlbmd0aCAtIHByb21wdHMubGVuZ3RoO1xuICAgICAgZm9yIChpID0gMDsgaSA8IHNlcXVlbmNlTGVuZ3RoOyBpKyspIHtcbiAgICAgICAgdmFyIGFueSA9IE1hdGgucmFuZG9tKCkgKiBkaXN0cmFjdG9ycy5sZW5ndGggfCAwO1xuICAgICAgICBzZXF1ZW5jZS5wdXNoKGRpc3RyYWN0b3JzW2FueV0pO1xuICAgICAgfVxuICAgICAgdmFyIGluZGV4ZXMgPSBbXSxcbiAgICAgICAgcG9zaXRpb25zID0gW107XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgcHJvbXB0cy5sZW5ndGg7IGkrKykge1xuICAgICAgICBpbmRleGVzLnB1c2goTWF0aC5yYW5kb20oKSAqIHRhcmdldHMubGVuZ3RoIHwgMCk7XG4gICAgICAgIHBvc2l0aW9ucy5wdXNoKG5vUmVwZWF0KHNlcXVlbmNlTGVuZ3RoLCBwb3NpdGlvbnMpKTtcbiAgICAgIH1cbiAgICAgIHBvc2l0aW9ucyA9IHBvc2l0aW9ucy5zb3J0KCk7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgcHJvbXB0cy5sZW5ndGg7IGkrKykge1xuICAgICAgICBzZXF1ZW5jZVtwb3NpdGlvbnNbaV1dID0gdGFyZ2V0c1tpbmRleGVzW2ldXTtcbiAgICAgICAgc2VxdWVuY2UucHVzaChwcm9tcHRzW2ldKTtcbiAgICAgIH1cblxuICAgICAgLy90cmFpbiBzZXF1ZW5jZVxuICAgICAgdmFyIGRpc3RyYWN0b3JzQ29ycmVjdDtcbiAgICAgIHZhciB0YXJnZXRzQ29ycmVjdCA9IGRpc3RyYWN0b3JzQ29ycmVjdCA9IDA7XG4gICAgICBlcnJvciA9IDA7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgbGVuZ3RoOyBpKyspIHtcbiAgICAgICAgLy8gZ2VuZXJhdGUgaW5wdXQgZnJvbSBzZXF1ZW5jZVxuICAgICAgICB2YXIgaW5wdXQgPSBbXTtcbiAgICAgICAgZm9yIChqID0gMDsgaiA8IHN5bWJvbHM7IGorKylcbiAgICAgICAgICBpbnB1dFtqXSA9IDA7XG4gICAgICAgIGlucHV0W3NlcXVlbmNlW2ldXSA9IDE7XG5cbiAgICAgICAgLy8gZ2VuZXJhdGUgdGFyZ2V0IG91dHB1dFxuICAgICAgICB2YXIgb3V0cHV0ID0gW107XG4gICAgICAgIGZvciAoaiA9IDA7IGogPCB0YXJnZXRzLmxlbmd0aDsgaisrKVxuICAgICAgICAgIG91dHB1dFtqXSA9IDA7XG5cbiAgICAgICAgaWYgKGkgPj0gc2VxdWVuY2VMZW5ndGgpIHtcbiAgICAgICAgICB2YXIgaW5kZXggPSBpIC0gc2VxdWVuY2VMZW5ndGg7XG4gICAgICAgICAgb3V0cHV0W2luZGV4ZXNbaW5kZXhdXSA9IDE7XG4gICAgICAgIH1cblxuICAgICAgICAvLyBjaGVjayByZXN1bHRcbiAgICAgICAgdmFyIHByZWRpY3Rpb24gPSB0aGlzLm5ldHdvcmsuYWN0aXZhdGUoaW5wdXQpO1xuXG4gICAgICAgIGlmIChlcXVhbChwcmVkaWN0aW9uLCBvdXRwdXQpKVxuICAgICAgICAgIGlmIChpIDwgc2VxdWVuY2VMZW5ndGgpXG4gICAgICAgICAgICBkaXN0cmFjdG9yc0NvcnJlY3QrKztcbiAgICAgICAgICBlbHNlXG4gICAgICAgICAgICB0YXJnZXRzQ29ycmVjdCsrO1xuICAgICAgICBlbHNlIHtcbiAgICAgICAgICB0aGlzLm5ldHdvcmsucHJvcGFnYXRlKHJhdGUsIG91dHB1dCk7XG4gICAgICAgIH1cblxuICAgICAgICB2YXIgZGVsdGEgPSAwO1xuICAgICAgICBmb3IgKHZhciBqIGluIHByZWRpY3Rpb24pXG4gICAgICAgICAgZGVsdGEgKz0gTWF0aC5wb3cob3V0cHV0W2pdIC0gcHJlZGljdGlvbltqXSwgMik7XG4gICAgICAgIGVycm9yICs9IGRlbHRhIC8gdGhpcy5uZXR3b3JrLm91dHB1dHMoKTtcblxuICAgICAgICBpZiAoZGlzdHJhY3RvcnNDb3JyZWN0ICsgdGFyZ2V0c0NvcnJlY3QgPT0gbGVuZ3RoKVxuICAgICAgICAgIGNvcnJlY3QrKztcbiAgICAgIH1cblxuICAgICAgLy8gY2FsY3VsYXRlIGVycm9yXG4gICAgICBpZiAodHJpYWwgJSAxMDAwID09IDApXG4gICAgICAgIGNvcnJlY3QgPSAwO1xuICAgICAgdHJpYWwrKztcbiAgICAgIHZhciBkaXZpZGVFcnJvciA9IHRyaWFsICUgMTAwMDtcbiAgICAgIGRpdmlkZUVycm9yID0gZGl2aWRlRXJyb3IgPT0gMCA/IDEwMDAgOiBkaXZpZGVFcnJvcjtcbiAgICAgIHN1Y2Nlc3MgPSBjb3JyZWN0IC8gZGl2aWRlRXJyb3I7XG4gICAgICBlcnJvciAvPSBsZW5ndGg7XG5cbiAgICAgIC8vIGxvZ1xuICAgICAgaWYgKGxvZyAmJiB0cmlhbCAlIGxvZyA9PSAwKVxuICAgICAgICBjb25zb2xlLmxvZyhcIml0ZXJhdGlvbnM6XCIsIHRyaWFsLCBcIiBzdWNjZXNzOlwiLCBzdWNjZXNzLCBcIiBjb3JyZWN0OlwiLFxuICAgICAgICAgIGNvcnJlY3QsIFwiIHRpbWU6XCIsIERhdGUubm93KCkgLSBzdGFydCwgXCIgZXJyb3I6XCIsIGVycm9yKTtcbiAgICAgIGlmIChzY2hlZHVsZS5kbyAmJiBzY2hlZHVsZS5ldmVyeSAmJiB0cmlhbCAlIHNjaGVkdWxlLmV2ZXJ5ID09IDApIHtcbiAgICAgICAgc2NoZWR1bGUuZG8oe1xuICAgICAgICAgIGl0ZXJhdGlvbnM6IHRyaWFsLFxuICAgICAgICAgIHN1Y2Nlc3M6IHN1Y2Nlc3MsXG4gICAgICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydCxcbiAgICAgICAgICBjb3JyZWN0OiBjb3JyZWN0XG4gICAgICAgIH0pO1xuXG4gICAgICAgIHNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgICAgIHRoaXMuaXRlcmF0aW9ucyAtPSB0cmlhbDtcbiAgICAgICAgICB0aGlzLkRTUihvcHRpb25zKTtcbiAgICAgICAgfSk7XG5cbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICBpdGVyYXRpb25zOiB0cmlhbCxcbiAgICAgIHN1Y2Nlc3M6IHN1Y2Nlc3MsXG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICB0aW1lOiBEYXRlLm5vdygpIC0gc3RhcnRcbiAgICB9XG4gIH1cblxuICAvLyB0cmFpbiB0aGUgbmV0d29yayB0byBsZWFybiBhbiBFbWJlZGVkIFJlYmVyIEdyYW1tYXJcbiAgRVJHKG9wdGlvbnMpIHtcblxuICAgIG9wdGlvbnMgPSBvcHRpb25zIHx8IHt9O1xuICAgIHZhciBpdGVyYXRpb25zID0gb3B0aW9ucy5pdGVyYXRpb25zIHx8IDE1MDAwMDtcbiAgICB2YXIgY3JpdGVyaW9uID0gb3B0aW9ucy5lcnJvciB8fCAuMDU7XG4gICAgdmFyIHJhdGUgPSBvcHRpb25zLnJhdGUgfHwgLjE7XG4gICAgdmFyIGxvZyA9IG9wdGlvbnMubG9nIHx8IDUwMDtcblxuICAgIC8vIGdyYW1hciBub2RlXG4gICAgdmFyIE5vZGUgPSBmdW5jdGlvbigpIHtcbiAgICAgIHRoaXMucGF0aHMgPSBbXTtcbiAgICB9XG4gICAgTm9kZS5wcm90b3R5cGUgPSB7XG4gICAgICBjb25uZWN0OiBmdW5jdGlvbihub2RlLCB2YWx1ZSkge1xuICAgICAgICB0aGlzLnBhdGhzLnB1c2goe1xuICAgICAgICAgIG5vZGU6IG5vZGUsXG4gICAgICAgICAgdmFsdWU6IHZhbHVlXG4gICAgICAgIH0pO1xuICAgICAgICByZXR1cm4gdGhpcztcbiAgICAgIH0sXG4gICAgICBhbnk6IGZ1bmN0aW9uKCkge1xuICAgICAgICBpZiAodGhpcy5wYXRocy5sZW5ndGggPT0gMClcbiAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgIHZhciBpbmRleCA9IE1hdGgucmFuZG9tKCkgKiB0aGlzLnBhdGhzLmxlbmd0aCB8IDA7XG4gICAgICAgIHJldHVybiB0aGlzLnBhdGhzW2luZGV4XTtcbiAgICAgIH0sXG4gICAgICB0ZXN0OiBmdW5jdGlvbih2YWx1ZSkge1xuICAgICAgICBmb3IgKHZhciBpIGluIHRoaXMucGF0aHMpXG4gICAgICAgICAgaWYgKHRoaXMucGF0aHNbaV0udmFsdWUgPT0gdmFsdWUpXG4gICAgICAgICAgICByZXR1cm4gdGhpcy5wYXRoc1tpXTtcbiAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgfVxuICAgIH1cblxuICAgIHZhciByZWJlckdyYW1tYXIgPSBmdW5jdGlvbigpIHtcblxuICAgICAgLy8gYnVpbGQgYSByZWJlciBncmFtbWFyXG4gICAgICB2YXIgb3V0cHV0ID0gbmV3IE5vZGUoKTtcbiAgICAgIHZhciBuMSA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG91dHB1dCwgXCJFXCIpO1xuICAgICAgdmFyIG4yID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjEsIFwiU1wiKTtcbiAgICAgIHZhciBuMyA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4xLCBcIlZcIikuY29ubmVjdChuMiwgXCJQXCIpO1xuICAgICAgdmFyIG40ID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjIsIFwiWFwiKVxuICAgICAgbjQuY29ubmVjdChuNCwgXCJTXCIpO1xuICAgICAgdmFyIG41ID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjMsIFwiVlwiKVxuICAgICAgbjUuY29ubmVjdChuNSwgXCJUXCIpO1xuICAgICAgbjIuY29ubmVjdChuNSwgXCJYXCIpXG4gICAgICB2YXIgbjYgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuNCwgXCJUXCIpLmNvbm5lY3QobjUsIFwiUFwiKTtcbiAgICAgIHZhciBpbnB1dCA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG42LCBcIkJcIilcblxuICAgICAgcmV0dXJuIHtcbiAgICAgICAgaW5wdXQ6IGlucHV0LFxuICAgICAgICBvdXRwdXQ6IG91dHB1dFxuICAgICAgfVxuICAgIH1cblxuICAgIC8vIGJ1aWxkIGFuIGVtYmVkZWQgcmViZXIgZ3JhbW1hclxuICAgIHZhciBlbWJlZGVkUmViZXJHcmFtbWFyID0gZnVuY3Rpb24oKSB7XG4gICAgICB2YXIgcmViZXIxID0gcmViZXJHcmFtbWFyKCk7XG4gICAgICB2YXIgcmViZXIyID0gcmViZXJHcmFtbWFyKCk7XG5cbiAgICAgIHZhciBvdXRwdXQgPSBuZXcgTm9kZSgpO1xuICAgICAgdmFyIG4xID0gKG5ldyBOb2RlKS5jb25uZWN0KG91dHB1dCwgXCJFXCIpO1xuICAgICAgcmViZXIxLm91dHB1dC5jb25uZWN0KG4xLCBcIlRcIik7XG4gICAgICByZWJlcjIub3V0cHV0LmNvbm5lY3QobjEsIFwiUFwiKTtcbiAgICAgIHZhciBuMiA9IChuZXcgTm9kZSkuY29ubmVjdChyZWJlcjEuaW5wdXQsIFwiUFwiKS5jb25uZWN0KHJlYmVyMi5pbnB1dCxcbiAgICAgICAgXCJUXCIpO1xuICAgICAgdmFyIGlucHV0ID0gKG5ldyBOb2RlKS5jb25uZWN0KG4yLCBcIkJcIik7XG5cbiAgICAgIHJldHVybiB7XG4gICAgICAgIGlucHV0OiBpbnB1dCxcbiAgICAgICAgb3V0cHV0OiBvdXRwdXRcbiAgICAgIH1cblxuICAgIH1cblxuICAgIC8vIGdlbmVyYXRlIGFuIEVSRyBzZXF1ZW5jZVxuICAgIHZhciBnZW5lcmF0ZSA9IGZ1bmN0aW9uKCkge1xuICAgICAgdmFyIG5vZGUgPSBlbWJlZGVkUmViZXJHcmFtbWFyKCkuaW5wdXQ7XG4gICAgICB2YXIgbmV4dCA9IG5vZGUuYW55KCk7XG4gICAgICB2YXIgc3RyID0gXCJcIjtcbiAgICAgIHdoaWxlIChuZXh0KSB7XG4gICAgICAgIHN0ciArPSBuZXh0LnZhbHVlO1xuICAgICAgICBuZXh0ID0gbmV4dC5ub2RlLmFueSgpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHN0cjtcbiAgICB9XG5cbiAgICAvLyB0ZXN0IGlmIGEgc3RyaW5nIG1hdGNoZXMgYW4gZW1iZWRlZCByZWJlciBncmFtbWFyXG4gICAgdmFyIHRlc3QgPSBmdW5jdGlvbihzdHIpIHtcbiAgICAgIHZhciBub2RlID0gZW1iZWRlZFJlYmVyR3JhbW1hcigpLmlucHV0O1xuICAgICAgdmFyIGkgPSAwO1xuICAgICAgdmFyIGNoID0gc3RyLmNoYXJBdChpKTtcbiAgICAgIHdoaWxlIChpIDwgc3RyLmxlbmd0aCkge1xuICAgICAgICB2YXIgbmV4dCA9IG5vZGUudGVzdChjaCk7XG4gICAgICAgIGlmICghbmV4dClcbiAgICAgICAgICByZXR1cm4gZmFsc2U7XG4gICAgICAgIG5vZGUgPSBuZXh0Lm5vZGU7XG4gICAgICAgIGNoID0gc3RyLmNoYXJBdCgrK2kpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuXG4gICAgLy8gaGVscGVyIHRvIGNoZWNrIGlmIHRoZSBvdXRwdXQgYW5kIHRoZSB0YXJnZXQgdmVjdG9ycyBtYXRjaFxuICAgIHZhciBkaWZmZXJlbnQgPSBmdW5jdGlvbihhcnJheTEsIGFycmF5Mikge1xuICAgICAgdmFyIG1heDEgPSAwO1xuICAgICAgdmFyIGkxID0gLTE7XG4gICAgICB2YXIgbWF4MiA9IDA7XG4gICAgICB2YXIgaTIgPSAtMTtcbiAgICAgIGZvciAodmFyIGkgaW4gYXJyYXkxKSB7XG4gICAgICAgIGlmIChhcnJheTFbaV0gPiBtYXgxKSB7XG4gICAgICAgICAgbWF4MSA9IGFycmF5MVtpXTtcbiAgICAgICAgICBpMSA9IGk7XG4gICAgICAgIH1cbiAgICAgICAgaWYgKGFycmF5MltpXSA+IG1heDIpIHtcbiAgICAgICAgICBtYXgyID0gYXJyYXkyW2ldO1xuICAgICAgICAgIGkyID0gaTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICByZXR1cm4gaTEgIT0gaTI7XG4gICAgfVxuXG4gICAgdmFyIGl0ZXJhdGlvbiA9IDA7XG4gICAgdmFyIGVycm9yID0gMTtcbiAgICB2YXIgdGFibGUgPSB7XG4gICAgICBcIkJcIjogMCxcbiAgICAgIFwiUFwiOiAxLFxuICAgICAgXCJUXCI6IDIsXG4gICAgICBcIlhcIjogMyxcbiAgICAgIFwiU1wiOiA0LFxuICAgICAgXCJFXCI6IDVcbiAgICB9XG5cbiAgICB2YXIgc3RhcnQgPSBEYXRlLm5vdygpO1xuICAgIHdoaWxlIChpdGVyYXRpb24gPCBpdGVyYXRpb25zICYmIGVycm9yID4gY3JpdGVyaW9uKSB7XG4gICAgICB2YXIgaSA9IDA7XG4gICAgICBlcnJvciA9IDA7XG5cbiAgICAgIC8vIEVSRyBzZXF1ZW5jZSB0byBsZWFyblxuICAgICAgdmFyIHNlcXVlbmNlID0gZ2VuZXJhdGUoKTtcblxuICAgICAgLy8gaW5wdXRcbiAgICAgIHZhciByZWFkID0gc2VxdWVuY2UuY2hhckF0KGkpO1xuICAgICAgLy8gdGFyZ2V0XG4gICAgICB2YXIgcHJlZGljdCA9IHNlcXVlbmNlLmNoYXJBdChpICsgMSk7XG5cbiAgICAgIC8vIHRyYWluXG4gICAgICB3aGlsZSAoaSA8IHNlcXVlbmNlLmxlbmd0aCAtIDEpIHtcbiAgICAgICAgdmFyIGlucHV0ID0gW107XG4gICAgICAgIHZhciB0YXJnZXQgPSBbXTtcbiAgICAgICAgZm9yICh2YXIgaiA9IDA7IGogPCA2OyBqKyspIHtcbiAgICAgICAgICBpbnB1dFtqXSA9IDA7XG4gICAgICAgICAgdGFyZ2V0W2pdID0gMDtcbiAgICAgICAgfVxuICAgICAgICBpbnB1dFt0YWJsZVtyZWFkXV0gPSAxO1xuICAgICAgICB0YXJnZXRbdGFibGVbcHJlZGljdF1dID0gMTtcblxuICAgICAgICB2YXIgb3V0cHV0ID0gdGhpcy5uZXR3b3JrLmFjdGl2YXRlKGlucHV0KTtcblxuICAgICAgICBpZiAoZGlmZmVyZW50KG91dHB1dCwgdGFyZ2V0KSlcbiAgICAgICAgICB0aGlzLm5ldHdvcmsucHJvcGFnYXRlKHJhdGUsIHRhcmdldCk7XG5cbiAgICAgICAgcmVhZCA9IHNlcXVlbmNlLmNoYXJBdCgrK2kpO1xuICAgICAgICBwcmVkaWN0ID0gc2VxdWVuY2UuY2hhckF0KGkgKyAxKTtcblxuICAgICAgICB2YXIgZGVsdGEgPSAwO1xuICAgICAgICBmb3IgKHZhciBrIGluIG91dHB1dClcbiAgICAgICAgICBkZWx0YSArPSBNYXRoLnBvdyh0YXJnZXRba10gLSBvdXRwdXRba10sIDIpXG4gICAgICAgIGRlbHRhIC89IG91dHB1dC5sZW5ndGg7XG5cbiAgICAgICAgZXJyb3IgKz0gZGVsdGE7XG4gICAgICB9XG4gICAgICBlcnJvciAvPSBzZXF1ZW5jZS5sZW5ndGg7XG4gICAgICBpdGVyYXRpb24rKztcbiAgICAgIGlmIChpdGVyYXRpb24gJSBsb2cgPT0gMCkge1xuICAgICAgICBjb25zb2xlLmxvZyhcIml0ZXJhdGlvbnM6XCIsIGl0ZXJhdGlvbiwgXCIgdGltZTpcIiwgRGF0ZS5ub3coKSAtIHN0YXJ0LFxuICAgICAgICAgIFwiIGVycm9yOlwiLCBlcnJvcik7XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgIGl0ZXJhdGlvbnM6IGl0ZXJhdGlvbixcbiAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydCxcbiAgICAgIHRlc3Q6IHRlc3QsXG4gICAgICBnZW5lcmF0ZTogZ2VuZXJhdGVcbiAgICB9XG4gIH1cblxufVxuXG5leHBvcnQgbW9kdWxlIFRyYWluZXIge1xuICAvLyBCdWlsdC1pbiBjb3N0IGZ1bmN0aW9uc1xuICBcbiAgZXhwb3J0IGludGVyZmFjZSBJVHJhaW5lckNvc3RGbiB7XG4gICAgKHRhcmdldCwgb3V0cHV0KTogbnVtYmVyO1xuICB9XG5cbiAgZXhwb3J0IHZhciBjb3N0ID0ge1xuICAgIC8vIEVxLiA5XG4gICAgQ1JPU1NfRU5UUk9QWTogZnVuY3Rpb24odGFyZ2V0LCBvdXRwdXQpIHtcbiAgICAgIHZhciBjcm9zc2VudHJvcHkgPSAwO1xuICAgICAgZm9yICh2YXIgaSBpbiBvdXRwdXQpXG4gICAgICAgIGNyb3NzZW50cm9weSAtPSAodGFyZ2V0W2ldICogTWF0aC5sb2cob3V0cHV0W2ldICsgMWUtMTUpKSArICgoMSAtIHRhcmdldFtpXSkgKiBNYXRoLmxvZygoMSArIDFlLTE1KSAtIG91dHB1dFtpXSkpOyAvLyArMWUtMTUgaXMgYSB0aW55IHB1c2ggYXdheSB0byBhdm9pZCBNYXRoLmxvZygwKVxuICAgICAgcmV0dXJuIGNyb3NzZW50cm9weTtcbiAgICB9LFxuICAgIE1TRTogZnVuY3Rpb24odGFyZ2V0LCBvdXRwdXQpIHtcbiAgICAgIHZhciBtc2UgPSAwO1xuICAgICAgZm9yICh2YXIgaSBpbiBvdXRwdXQpXG4gICAgICAgIG1zZSArPSBNYXRoLnBvdyh0YXJnZXRbaV0gLSBvdXRwdXRbaV0sIDIpO1xuICAgICAgcmV0dXJuIG1zZSAvIG91dHB1dC5sZW5ndGg7XG4gICAgfVxuICB9XG59Il0sInNvdXJjZVJvb3QiOiIvc291cmNlLyJ9