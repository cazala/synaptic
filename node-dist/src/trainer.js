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
                if (this.schedule && this.schedule.every && iterations % this.schedule.every == 0)
                    abort_training = this.schedule.do({
                        error: error,
                        iterations: iterations,
                        rate: currentRate
                    });
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
            if (schedule.do && schedule.every && trial % schedule.every == 0)
                schedule.do({
                    iterations: trial,
                    success: success,
                    error: error,
                    time: Date.now() - start,
                    correct: correct
                });
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

//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbInNyYy90cmFpbmVyLnRzIl0sIm5hbWVzIjpbIlRyYWluZXIiLCJUcmFpbmVyLmNvbnN0cnVjdG9yIiwiVHJhaW5lci50cmFpbiIsIlRyYWluZXIudHJhaW4uc2h1ZmZsZSIsIlRyYWluZXIud29ya2VyVHJhaW4iLCJUcmFpbmVyLndvcmtlclRyYWluLnNodWZmbGUiLCJUcmFpbmVyLndvcmtlclRyYWluLmFjdGl2YXRlV29ya2VyIiwiVHJhaW5lci53b3JrZXJUcmFpbi5wcm9wYWdhdGVXb3JrZXIiLCJUcmFpbmVyLlhPUiIsIlRyYWluZXIuRFNSIiwiVHJhaW5lci5FUkciXSwibWFwcGluZ3MiOiJBQUVBLEFBSUE7OzRGQUY0RjtJQUUvRSxPQUFPO0lBUWxCQSxTQVJXQSxPQUFPQSxDQVFOQSxPQUFvQkEsRUFBRUEsT0FBYUE7UUFOL0NDLFNBQUlBLEdBQVFBLEVBQUVBLENBQUNBO1FBQ2ZBLGVBQVVBLEdBQUdBLE1BQU1BLENBQUNBO1FBQ3BCQSxVQUFLQSxHQUFHQSxJQUFJQSxDQUFDQTtRQUtYQSxPQUFPQSxHQUFHQSxPQUFPQSxJQUFJQSxFQUFFQSxDQUFDQTtRQUN4QkEsSUFBSUEsQ0FBQ0EsT0FBT0EsR0FBR0EsT0FBT0EsQ0FBQ0E7UUFDdkJBLElBQUlBLENBQUNBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLElBQUlBLEVBQUVBLENBQUNBO1FBQy9CQSxJQUFJQSxDQUFDQSxVQUFVQSxHQUFHQSxPQUFPQSxDQUFDQSxVQUFVQSxJQUFJQSxNQUFNQSxDQUFDQTtRQUMvQ0EsSUFBSUEsQ0FBQ0EsS0FBS0EsR0FBR0EsT0FBT0EsQ0FBQ0EsS0FBS0EsSUFBSUEsSUFBSUEsQ0FBQUE7UUFDbENBLElBQUlBLENBQUNBLElBQUlBLEdBQUdBLE9BQU9BLENBQUNBLElBQUlBLElBQUlBLE9BQU9BLENBQUNBLElBQUlBLENBQUNBLGFBQWFBLENBQUNBO0lBQ3pEQSxDQUFDQTtJQUVERCxvQ0FBb0NBO0lBQ3BDQSx1QkFBS0EsR0FBTEEsVUFBTUEs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LENBQUNBLENBQUNBO1lBQ2JBLE1BQU1BLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO1NBQ1pBLEVBQUVBO1lBQ0NBLEtBQUtBLEVBQUVBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBQ2JBLE1BQU1BLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO1NBQ1pBLEVBQUVBO1lBQ0RBLEtBQUtBLEVBQUVBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBQ2JBLE1BQU1BLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO1NBQ1pBLEVBQUVBO1lBQ0RBLEtBQUtBLEVBQUVBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBO1lBQ2JBLE1BQU1BLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO1NBQ1pBLENBQUNBLEVBQUVBLFFBQVFBLENBQUNBLENBQUNBO0lBQ2xCQSxDQUFDQTtJQUVEUiwrREFBK0RBO0lBQy9EQSxxQkFBR0EsR0FBSEEsVUFBSUEsT0FBT0E7UUFDVFMsT0FBT0EsR0FBR0EsT0FBT0EsSUFBSUEsRUFBRUEsQ0FBQ0E7UUFFeEJBLElBQUlBLE9BQU9BLEdBQUdBLE9BQU9BLENBQUNBLE9BQU9BLElBQUlBLENBQUNBLENBQUNBLEVBQUVBLENBQUNBLEVBQUVBLENBQUNBLEVBQUVBLENBQUNBLENBQUNBLENBQUNBO1FBQzlDQSxJQUFJQSxXQUFXQSxHQUFHQSxPQUFPQSxDQUFDQSxXQUFXQSxJQUFJQSxDQUFDQSxDQUFDQSxFQUFFQSxDQUFDQSxFQUFFQSxDQUFDQSxFQUFFQSxDQUFDQSxDQUFDQSxDQUFDQTtRQUN0REEsSUFBSUEsT0FBT0EsR0FBR0EsT0FBT0EsQ0FBQ0EsT0FBT0EsSUFBSUEsQ0FBQ0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsQ0FBQ0EsQ0FBQ0E7UUFDeENBLElBQUlBLE1BQU1BLEdBQUdBLE9BQU9BLENBQUNBLE1BQU1BLElBQUlBLEVBQUVBLENBQUNBO1FBQ2xDQSxJQUFJQSxTQUFTQSxHQUFHQSxPQUFPQSxDQUFDQSxPQUFPQSxJQUFJQSxJQUFJQSxDQUFDQTtR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Q0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsT0FBT0EsQ0FBQ0EsTUFBTUEsRUFBRUEsQ0FBQ0EsRUFBRUEsRUFBRUEsQ0FBQ0E7Z0JBQ3BDQSxPQUFPQSxDQUFDQSxJQUFJQSxDQUFDQSxJQUFJQSxDQUFDQSxNQUFNQSxFQUFFQSxHQUFHQSxPQUFPQSxDQUFDQSxNQUFNQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDakRBLFNBQVNBLENBQUNBLElBQUlBLENBQUNBLFFBQVFBLENBQUNBLGNBQWNBLEVBQUVBLFNBQVNBLENBQUNBLENBQUNBLENBQUNBO1lBQ3REQSxDQUFDQTtZQUNEQSxTQUFTQSxHQUFHQSxTQUFTQSxDQUFDQSxJQUFJQSxFQUFFQSxDQUFDQTtZQUM3QkEsR0FBR0EsQ0FBQ0EsQ0FBQ0EsQ0FBQ0EsR0FBR0EsQ0FBQ0EsRUFBRUEsQ0FBQ0EsR0FBR0EsT0FBT0EsQ0FBQ0EsTUFBTUEsRUFBRUEsQ0FBQ0EsRUFBRUEsRUFBRUEsQ0FBQ0E7Z0JBQ3BDQSxRQUFRQSxDQUFDQSxTQUFTQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxPQUFPQSxDQUFDQSxPQUFPQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQSxDQUFDQTtnQkFDN0NBLFFBQVFBLENBQUNBLElBQUlBLENBQUNBLE9BQU9BLENBQUNBLENBQUNBLENBQUNBLENBQUNBLENBQUNBO1lBQzVCQSxDQUFDQTtZQUVEQSxBQUNBQSxnQkFEZ0JBO2dCQUNaQSxrQkFBa0JBLENBQUNBO1lBQ3ZCQSxJQUFJQSxjQUFjQSxHQUFHQSxrQkFBa0JBLEdBQUdBLENBQUNBLENBQUNBO1lBQzVDQSxLQUFLQSxHQUFHQSxDQUFDQSxDQUFDQTtZQUNWQSxHQUFHQSxDQUFDQSxDQUFDQSxDQUFDQSxHQUFHQSxDQUFDQSxFQUFFQSxDQUFDQSxHQUFHQSxNQUFNQSxFQUFFQSxDQUFDQSxFQUFFQSxFQUFFQSxDQUFDQTtnQkFDNUJBLEFBQ0FBLCtCQUQrQkE7b0JBQzNCQSxLQUFLQSxHQUFHQSxFQUFFQSxDQUFD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TUFBTSxDQUFDO1FBQzdCLENBQUM7S0FDRkEsQ0FBQUE7QUFDSEEsQ0FBQ0EsRUF0QmEsT0FBTyxHQUFQLGVBQU8sS0FBUCxlQUFPLFFBc0JwQiIsImZpbGUiOiJzcmMvdHJhaW5lci5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBuZXQgPXJlcXVpcmUoJy4vbmV0d29yaycpO1xuXG4vKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFRSQUlORVJcbioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKioqKiovXG5cbmV4cG9ydCBjbGFzcyBUcmFpbmVyIHtcbiAgbmV0d29yazogbmV0Lk5ldHdvcms7XG4gIHJhdGU6IGFueSA9IC4yO1xuICBpdGVyYXRpb25zID0gMTAwMDAwO1xuICBlcnJvciA9IC4wMDU7XG4gIGNvc3Q6IFRyYWluZXIuSVRyYWluZXJDb3N0Rm47XG4gIHNjaGVkdWxlOiBhbnk7XG5cbiAgY29uc3RydWN0b3IobmV0d29yazogbmV0Lk5ldHdvcmssIG9wdGlvbnM/OiBhbnkpIHtcbiAgICBvcHRpb25zID0gb3B0aW9ucyB8fCB7fTtcbiAgICB0aGlzLm5ldHdvcmsgPSBuZXR3b3JrO1xuICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZSB8fCAuMjtcbiAgICB0aGlzLml0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnMgfHwgMTAwMDAwO1xuICAgIHRoaXMuZXJyb3IgPSBvcHRpb25zLmVycm9yIHx8IC4wMDVcbiAgICB0aGlzLmNvc3QgPSBvcHRpb25zLmNvc3QgfHwgVHJhaW5lci5jb3N0LkNST1NTX0VOVFJPUFk7XG4gIH1cblxuICAvLyB0cmFpbnMgYW55IGdpdmVuIHNldCB0byBhIG5ldHdvcmtcbiAgdHJhaW4oc2V0LCBvcHRpb25zKSB7XG5cbiAgICB2YXIgZXJyb3IgPSAxO1xuICAgIHZhciBpdGVyYXRpb25zID0gMCwgYnVja2V0U2l6ZSA9IDA7XG4gICAgdmFyIGFib3J0X3RyYWluaW5nID0gZmFsc2U7XG4gICAgdmFyIGlucHV0LCBvdXRwdXQsIHRhcmdldCwgY3VycmVudFJhdGU7XG5cbiAgICB2YXIgc3RhcnQgPSBEYXRlLm5vdygpO1xuXG4gICAgaWYgKG9wdGlvbnMpIHtcbiAgICAgIGlmIChvcHRpb25zLnNodWZmbGUpIHtcbiAgICAgICAgLy8rIEpvbmFzIFJhb25pIFNvYXJlcyBTaWx2YVxuICAgICAgICAvL0AgaHR0cDovL2pzZnJvbWhlbGwuY29tL2FycmF5L3NodWZmbGUgW3YxLjBdXG4gICAgICAgIGZ1bmN0aW9uIHNodWZmbGUobykgeyAvL3YxLjBcbiAgICAgICAgICBmb3IgKHZhciBqLCB4LCBpID0gby5sZW5ndGg7IGk7IGogPSBNYXRoLmZsb29yKE1hdGgucmFuZG9tKCkgKiBpKSwgeCA9IG9bLS1pXSwgb1tpXSA9IG9bal0sIG9bal0gPSB4KTtcbiAgICAgICAgICByZXR1cm4gbztcbiAgICAgICAgfTtcbiAgICAgIH1cbiAgICAgIGlmIChvcHRpb25zLml0ZXJhdGlvbnMpXG4gICAgICAgIHRoaXMuaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucztcbiAgICAgIGlmIChvcHRpb25zLmVycm9yKVxuICAgICAgICB0aGlzLmVycm9yID0gb3B0aW9ucy5lcnJvcjtcbiAgICAgIGlmIChvcHRpb25zLnJhdGUpXG4gICAgICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZTtcbiAgICAgIGlmIChvcHRpb25zLmNvc3QpXG4gICAgICAgIHRoaXMuY29zdCA9IG9wdGlvbnMuY29zdDtcbiAgICAgIGlmIChvcHRpb25zLnNjaGVkdWxlKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5zY2hlZHVsZTtcbiAgICAgIGlmIChvcHRpb25zLmN1c3RvbUxvZykge1xuICAgICAgICAvLyBmb3IgYmFja3dhcmQgY29tcGF0aWJpbGl0eSB3aXRoIGNvZGUgdGhhdCB1c2VkIGN1c3RvbUxvZ1xuICAgICAgICBjb25zb2xlLmxvZygnRGVwcmVjYXRlZDogdXNlIHNjaGVkdWxlIGluc3RlYWQgb2YgY3VzdG9tTG9nJylcbiAgICAgICAgdGhpcy5zY2hlZHVsZSA9IG9wdGlvbnMuY3VzdG9tTG9nO1xuICAgICAgfVxuICAgIH1cblxuICAgIGN1cnJlbnRSYXRlID0gdGhpcy5yYXRlO1xuICAgIGlmIChBcnJheS5pc0FycmF5KHRoaXMucmF0ZSkpIHtcbiAgICAgIGJ1Y2tldFNpemUgPSBNYXRoLmZsb29yKHRoaXMuaXRlcmF0aW9ucyAvIHRoaXMucmF0ZS5sZW5ndGgpO1xuICAgIH1cblxuXG4gICAgd2hpbGUgKCFhYm9ydF90cmFpbmluZyAmJiBpdGVyYXRpb25zIDwgdGhpcy5pdGVyYXRpb25zICYmIGVycm9yID4gdGhpcy5lcnJvcikge1xuICAgICAgZXJyb3IgPSAwO1xuXG4gICAgICBpZiAoYnVja2V0U2l6ZSA+IDApIHtcbiAgICAgICAgdmFyIGN1cnJlbnRCdWNrZXQgPSBNYXRoLmZsb29yKGl0ZXJhdGlvbnMgLyBidWNrZXRTaXplKTtcbiAgICAgICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGVbY3VycmVudEJ1Y2tldF07XG4gICAgICB9XG5cbiAgICAgIGZvciAodmFyIHRyYWluIGluIHNldCkge1xuICAgICAgICBpbnB1dCA9IHNldFt0cmFpbl0uaW5wdXQ7XG4gICAgICAgIHRhcmdldCA9IHNldFt0cmFpbl0ub3V0cHV0O1xuXG4gICAgICAgIG91dHB1dCA9IHRoaXMubmV0d29yay5hY3RpdmF0ZShpbnB1dCk7XG4gICAgICAgIHRoaXMubmV0d29yay5wcm9wYWdhdGUoY3VycmVudFJhdGUsIHRhcmdldCk7XG5cbiAgICAgICAgZXJyb3IgKz0gdGhpcy5jb3N0KHRhcmdldCwgb3V0cHV0KTtcbiAgICAgIH1cblxuICAgICAgLy8gY2hlY2sgZXJyb3JcbiAgICAgIGl0ZXJhdGlvbnMrKztcbiAgICAgIGVycm9yIC89IHNldC5sZW5ndGg7XG5cbiAgICAgIGlmIChvcHRpb25zKSB7XG4gICAgICAgIGlmICh0aGlzLnNjaGVkdWxlICYmIHRoaXMuc2NoZWR1bGUuZXZlcnkgJiYgaXRlcmF0aW9ucyAlXG4gICAgICAgICAgdGhpcy5zY2hlZHVsZS5ldmVyeSA9PSAwKVxuICAgICAgICAgIGFib3J0X3RyYWluaW5nID0gdGhpcy5zY2hlZHVsZS5kbyh7XG4gICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgICAgICAgcmF0ZTogY3VycmVudFJhdGVcbiAgICAgICAgICB9KTtcbiAgICAgICAgZWxzZSBpZiAob3B0aW9ucy5sb2cgJiYgaXRlcmF0aW9ucyAlIG9wdGlvbnMubG9nID09IDApIHtcbiAgICAgICAgICBjb25zb2xlLmxvZygnaXRlcmF0aW9ucycsIGl0ZXJhdGlvbnMsICdlcnJvcicsIGVycm9yLCAncmF0ZScsIGN1cnJlbnRSYXRlKTtcbiAgICAgICAgfTtcbiAgICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSlcbiAgICAgICAgICBzaHVmZmxlKHNldCk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdmFyIHJlc3VsdHMgPSB7XG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zLFxuICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0XG4gICAgfVxuXG4gICAgcmV0dXJuIHJlc3VsdHM7XG4gIH1cblxuICAvLyB0cmFpbnMgYW55IGdpdmVuIHNldCB0byBhIG5ldHdvcmsgdXNpbmcgYSBXZWJXb3JrZXJcbiAgd29ya2VyVHJhaW4oc2V0LCBjYWxsYmFjaywgb3B0aW9ucykge1xuXG4gICAgdmFyIHRoYXQgPSB0aGlzO1xuICAgIHZhciBlcnJvciA9IDE7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSAwLCBidWNrZXRTaXplID0gMDtcbiAgICB2YXIgaW5wdXQsIG91dHB1dCwgdGFyZ2V0LCBjdXJyZW50UmF0ZTtcbiAgICB2YXIgbGVuZ3RoID0gc2V0Lmxlbmd0aDtcbiAgICB2YXIgYWJvcnRfdHJhaW5pbmcgPSBmYWxzZTtcblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICBpZiAob3B0aW9ucykge1xuICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSkge1xuICAgICAgICAvLysgSm9uYXMgUmFvbmkgU29hcmVzIFNpbHZhXG4gICAgICAgIC8vQCBodHRwOi8vanNmcm9taGVsbC5jb20vYXJyYXkvc2h1ZmZsZSBbdjEuMF1cbiAgICAgICAgZnVuY3Rpb24gc2h1ZmZsZShvKSB7IC8vdjEuMFxuICAgICAgICAgIGZvciAodmFyIGosIHgsIGkgPSBvLmxlbmd0aDsgaTsgaiA9IE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqXG4gICAgICAgICAgICBpKSwgeCA9IG9bLS1pXSwgb1tpXSA9IG9bal0sIG9bal0gPSB4KTtcbiAgICAgICAgICByZXR1cm4gbztcbiAgICAgICAgfTtcbiAgICAgIH1cbiAgICAgIGlmIChvcHRpb25zLml0ZXJhdGlvbnMpXG4gICAgICAgIHRoaXMuaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucztcbiAgICAgIGlmIChvcHRpb25zLmVycm9yKVxuICAgICAgICB0aGlzLmVycm9yID0gb3B0aW9ucy5lcnJvcjtcbiAgICAgIGlmIChvcHRpb25zLnJhdGUpXG4gICAgICAgIHRoaXMucmF0ZSA9IG9wdGlvbnMucmF0ZTtcbiAgICAgIGlmIChvcHRpb25zLmNvc3QpXG4gICAgICAgIHRoaXMuY29zdCA9IG9wdGlvbnMuY29zdDtcbiAgICAgIGlmIChvcHRpb25zLnNjaGVkdWxlKVxuICAgICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5zY2hlZHVsZTtcbiAgICAgIGlmIChvcHRpb25zLmN1c3RvbUxvZylcbiAgICAgICAgLy8gZm9yIGJhY2t3YXJkIGNvbXBhdGliaWxpdHkgd2l0aCBjb2RlIHRoYXQgdXNlZCBjdXN0b21Mb2dcbiAgICAgICAgY29uc29sZS5sb2coJ0RlcHJlY2F0ZWQ6IHVzZSBzY2hlZHVsZSBpbnN0ZWFkIG9mIGN1c3RvbUxvZycpXG4gICAgICB0aGlzLnNjaGVkdWxlID0gb3B0aW9ucy5jdXN0b21Mb2c7XG4gICAgfVxuXG4gICAgLy8gZHluYW1pYyBsZWFybmluZyByYXRlXG4gICAgY3VycmVudFJhdGUgPSB0aGlzLnJhdGU7XG4gICAgaWYgKEFycmF5LmlzQXJyYXkodGhpcy5yYXRlKSkge1xuICAgICAgYnVja2V0U2l6ZSA9IE1hdGguZmxvb3IodGhpcy5pdGVyYXRpb25zIC8gdGhpcy5yYXRlLmxlbmd0aCk7XG4gICAgfVxuXG4gICAgLy8gY3JlYXRlIGEgd29ya2VyXG4gICAgdmFyIHdvcmtlciA9IHRoaXMubmV0d29yay53b3JrZXIoKTtcblxuICAgIC8vIGFjdGl2YXRlIHRoZSBuZXR3b3JrXG4gICAgZnVuY3Rpb24gYWN0aXZhdGVXb3JrZXIoaW5wdXQpIHtcbiAgICAgIHdvcmtlci5wb3N0TWVzc2FnZSh7XG4gICAgICAgIGFjdGlvbjogXCJhY3RpdmF0ZVwiLFxuICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgIG1lbW9yeUJ1ZmZlcjogdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnlcbiAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyBiYWNrcHJvcGFnYXRlIHRoZSBuZXR3b3JrXG4gICAgZnVuY3Rpb24gcHJvcGFnYXRlV29ya2VyKHRhcmdldCkge1xuICAgICAgaWYgKGJ1Y2tldFNpemUgPiAwKSB7XG4gICAgICAgIHZhciBjdXJyZW50QnVja2V0ID0gTWF0aC5mbG9vcihpdGVyYXRpb25zIC8gYnVja2V0U2l6ZSk7XG4gICAgICAgIGN1cnJlbnRSYXRlID0gdGhpcy5yYXRlW2N1cnJlbnRCdWNrZXRdO1xuICAgICAgfVxuICAgICAgd29ya2VyLnBvc3RNZXNzYWdlKHtcbiAgICAgICAgYWN0aW9uOiBcInByb3BhZ2F0ZVwiLFxuICAgICAgICB0YXJnZXQ6IHRhcmdldCxcbiAgICAgICAgcmF0ZTogY3VycmVudFJhdGUsXG4gICAgICAgIG1lbW9yeUJ1ZmZlcjogdGhhdC5uZXR3b3JrLm9wdGltaXplZC5tZW1vcnlcbiAgICAgIH0sIFt0aGF0Lm5ldHdvcmsub3B0aW1pemVkLm1lbW9yeS5idWZmZXJdKTtcbiAgICB9XG5cbiAgICAvLyB0cmFpbiB0aGUgd29ya2VyXG4gICAgd29ya2VyLm9ubWVzc2FnZSA9IGZ1bmN0aW9uKGUpIHtcbiAgICAgIC8vIGdpdmUgY29udHJvbCBvZiB0aGUgbWVtb3J5IGJhY2sgdG8gdGhlIG5ldHdvcmtcbiAgICAgIHRoYXQubmV0d29yay5vcHRpbWl6ZWQub3duZXJzaGlwKGUuZGF0YS5tZW1vcnlCdWZmZXIpO1xuXG4gICAgICBpZiAoZS5kYXRhLmFjdGlvbiA9PSBcInByb3BhZ2F0ZVwiKSB7XG4gICAgICAgIGlmIChpbmRleCA+PSBsZW5ndGgpIHtcbiAgICAgICAgICBpbmRleCA9IDA7XG4gICAgICAgICAgaXRlcmF0aW9ucysrO1xuICAgICAgICAgIGVycm9yIC89IHNldC5sZW5ndGg7XG5cbiAgICAgICAgICAvLyBsb2dcbiAgICAgICAgICBpZiAob3B0aW9ucykge1xuICAgICAgICAgICAgaWYgKHRoaXMuc2NoZWR1bGUgJiYgdGhpcy5zY2hlZHVsZS5ldmVyeSAmJiBpdGVyYXRpb25zICUgdGhpcy5zY2hlZHVsZS5ldmVyeSA9PSAwKVxuICAgICAgICAgICAgICBhYm9ydF90cmFpbmluZyA9IHRoaXMuc2NoZWR1bGUuZG8oe1xuICAgICAgICAgICAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgICAgICAgICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb25zXG4gICAgICAgICAgICAgIH0pO1xuICAgICAgICAgICAgZWxzZSBpZiAob3B0aW9ucy5sb2cgJiYgaXRlcmF0aW9ucyAlIG9wdGlvbnMubG9nID09IDApIHtcbiAgICAgICAgICAgICAgY29uc29sZS5sb2coJ2l0ZXJhdGlvbnMnLCBpdGVyYXRpb25zLCAnZXJyb3InLCBlcnJvcik7XG4gICAgICAgICAgICB9O1xuICAgICAgICAgICAgaWYgKG9wdGlvbnMuc2h1ZmZsZSlcbiAgICAgICAgICAgICAgc2h1ZmZsZShzZXQpO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIGlmICghYWJvcnRfdHJhaW5pbmcgJiYgaXRlcmF0aW9ucyA8IHRoYXQuaXRlcmF0aW9ucyAmJiBlcnJvciA+IHRoYXQuZXJyb3IpIHtcbiAgICAgICAgICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAvLyBjYWxsYmFja1xuICAgICAgICAgICAgY2FsbGJhY2soe1xuICAgICAgICAgICAgICBlcnJvcjogZXJyb3IsXG4gICAgICAgICAgICAgIGl0ZXJhdGlvbnM6IGl0ZXJhdGlvbnMsXG4gICAgICAgICAgICAgIHRpbWU6IERhdGUubm93KCkgLSBzdGFydFxuICAgICAgICAgICAgfSlcbiAgICAgICAgICB9XG4gICAgICAgICAgZXJyb3IgPSAwO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIGFjdGl2YXRlV29ya2VyKHNldFtpbmRleF0uaW5wdXQpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChlLmRhdGEuYWN0aW9uID09IFwiYWN0aXZhdGVcIikge1xuICAgICAgICBlcnJvciArPSB0aGF0LmNvc3Qoc2V0W2luZGV4XS5vdXRwdXQsIGUuZGF0YS5vdXRwdXQpO1xuICAgICAgICBwcm9wYWdhdGVXb3JrZXIoc2V0W2luZGV4XS5vdXRwdXQpO1xuICAgICAgICBpbmRleCsrO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIGtpY2sgaXRcbiAgICB2YXIgaW5kZXggPSAwO1xuICAgIHZhciBpdGVyYXRpb25zID0gMDtcbiAgICBhY3RpdmF0ZVdvcmtlcihzZXRbaW5kZXhdLmlucHV0KTtcbiAgfVxuXG4gIC8vIHRyYWlucyBhbiBYT1IgdG8gdGhlIG5ldHdvcmtcbiAgWE9SKG9wdGlvbnMpIHtcblxuICAgIGlmICh0aGlzLm5ldHdvcmsuaW5wdXRzKCkgIT0gMiB8fCB0aGlzLm5ldHdvcmsub3V0cHV0cygpICE9IDEpXG4gICAgICB0aHJvdyBcIkVycm9yOiBJbmNvbXBhdGlibGUgbmV0d29yayAoMiBpbnB1dHMsIDEgb3V0cHV0KVwiO1xuXG4gICAgdmFyIGRlZmF1bHRzID0ge1xuICAgICAgaXRlcmF0aW9uczogMTAwMDAwLFxuICAgICAgbG9nOiBmYWxzZSxcbiAgICAgIHNodWZmbGU6IHRydWUsXG4gICAgICBjb3N0OiBUcmFpbmVyLmNvc3QuTVNFXG4gICAgfVxuXG4gICAgaWYgKG9wdGlvbnMpXG4gICAgICBmb3IgKHZhciBpIGluIG9wdGlvbnMpXG4gICAgICAgIGRlZmF1bHRzW2ldID0gb3B0aW9uc1tpXTtcblxuICAgIHJldHVybiB0aGlzLnRyYWluKFt7XG4gICAgICBpbnB1dDogWzAsIDBdLFxuICAgICAgb3V0cHV0OiBbMF1cbiAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMSwgMF0sXG4gICAgICAgIG91dHB1dDogWzFdXG4gICAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMCwgMV0sXG4gICAgICAgIG91dHB1dDogWzFdXG4gICAgICB9LCB7XG4gICAgICAgIGlucHV0OiBbMSwgMV0sXG4gICAgICAgIG91dHB1dDogWzBdXG4gICAgICB9XSwgZGVmYXVsdHMpO1xuICB9XG5cbiAgLy8gdHJhaW5zIHRoZSBuZXR3b3JrIHRvIHBhc3MgYSBEaXN0cmFjdGVkIFNlcXVlbmNlIFJlY2FsbCB0ZXN0XG4gIERTUihvcHRpb25zKSB7XG4gICAgb3B0aW9ucyA9IG9wdGlvbnMgfHwge307XG5cbiAgICB2YXIgdGFyZ2V0cyA9IG9wdGlvbnMudGFyZ2V0cyB8fCBbMiwgNCwgNywgOF07XG4gICAgdmFyIGRpc3RyYWN0b3JzID0gb3B0aW9ucy5kaXN0cmFjdG9ycyB8fCBbMywgNSwgNiwgOV07XG4gICAgdmFyIHByb21wdHMgPSBvcHRpb25zLnByb21wdHMgfHwgWzAsIDFdO1xuICAgIHZhciBsZW5ndGggPSBvcHRpb25zLmxlbmd0aCB8fCAyNDtcbiAgICB2YXIgY3JpdGVyaW9uID0gb3B0aW9ucy5zdWNjZXNzIHx8IDAuOTU7XG4gICAgdmFyIGl0ZXJhdGlvbnMgPSBvcHRpb25zLml0ZXJhdGlvbnMgfHwgMTAwMDAwO1xuICAgIHZhciByYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4xO1xuICAgIHZhciBsb2cgPSBvcHRpb25zLmxvZyB8fCAwO1xuICAgIHZhciBzY2hlZHVsZSA9IG9wdGlvbnMuc2NoZWR1bGUgfHwge307XG4gICAgdmFyIGNvcnJlY3QgPSAwO1xuICAgIHZhciBpID0gMDtcbiAgICB2YXIgc3VjY2VzcyA9IDA7XG4gICAgdmFyIHRyaWFsID0gaSA9IGNvcnJlY3QgPSBqID0gc3VjY2VzcyA9IDAsXG4gICAgICBlcnJvciA9IDEsXG4gICAgICBzeW1ib2xzID0gdGFyZ2V0cy5sZW5ndGggKyBkaXN0cmFjdG9ycy5sZW5ndGggKyBwcm9tcHRzLmxlbmd0aDtcblxuICAgIHZhciBub1JlcGVhdCA9IGZ1bmN0aW9uKHJhbmdlLCBhdm9pZCkge1xuICAgICAgdmFyIG51bWJlciA9IE1hdGgucmFuZG9tKCkgKiByYW5nZSB8IDA7XG4gICAgICB2YXIgdXNlZCA9IGZhbHNlO1xuICAgICAgZm9yICh2YXIgaSBpbiBhdm9pZClcbiAgICAgICAgaWYgKG51bWJlciA9PSBhdm9pZFtpXSlcbiAgICAgICAgICB1c2VkID0gdHJ1ZTtcbiAgICAgIHJldHVybiB1c2VkID8gbm9SZXBlYXQocmFuZ2UsIGF2b2lkKSA6IG51bWJlcjtcbiAgICB9XG5cbiAgICB2YXIgZXF1YWwgPSBmdW5jdGlvbihwcmVkaWN0aW9uLCBvdXRwdXQpIHtcbiAgICAgIGZvciAodmFyIGkgaW4gcHJlZGljdGlvbilcbiAgICAgICAgaWYgKE1hdGgucm91bmQocHJlZGljdGlvbltpXSkgIT0gb3V0cHV0W2ldKVxuICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIHZhciBzdGFydCA9IERhdGUubm93KCk7XG5cbiAgICB3aGlsZSAodHJpYWwgPCBpdGVyYXRpb25zICYmIChzdWNjZXNzIDwgY3JpdGVyaW9uIHx8IHRyaWFsICUgMTAwMCAhPSAwKSkge1xuICAgICAgLy8gZ2VuZXJhdGUgc2VxdWVuY2VcbiAgICAgIHZhciBzZXF1ZW5jZSA9IFtdLFxuICAgICAgICBzZXF1ZW5jZUxlbmd0aCA9IGxlbmd0aCAtIHByb21wdHMubGVuZ3RoO1xuICAgICAgZm9yIChpID0gMDsgaSA8IHNlcXVlbmNlTGVuZ3RoOyBpKyspIHtcbiAgICAgICAgdmFyIGFueSA9IE1hdGgucmFuZG9tKCkgKiBkaXN0cmFjdG9ycy5sZW5ndGggfCAwO1xuICAgICAgICBzZXF1ZW5jZS5wdXNoKGRpc3RyYWN0b3JzW2FueV0pO1xuICAgICAgfVxuICAgICAgdmFyIGluZGV4ZXMgPSBbXSxcbiAgICAgICAgcG9zaXRpb25zID0gW107XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgcHJvbXB0cy5sZW5ndGg7IGkrKykge1xuICAgICAgICBpbmRleGVzLnB1c2goTWF0aC5yYW5kb20oKSAqIHRhcmdldHMubGVuZ3RoIHwgMCk7XG4gICAgICAgIHBvc2l0aW9ucy5wdXNoKG5vUmVwZWF0KHNlcXVlbmNlTGVuZ3RoLCBwb3NpdGlvbnMpKTtcbiAgICAgIH1cbiAgICAgIHBvc2l0aW9ucyA9IHBvc2l0aW9ucy5zb3J0KCk7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgcHJvbXB0cy5sZW5ndGg7IGkrKykge1xuICAgICAgICBzZXF1ZW5jZVtwb3NpdGlvbnNbaV1dID0gdGFyZ2V0c1tpbmRleGVzW2ldXTtcbiAgICAgICAgc2VxdWVuY2UucHVzaChwcm9tcHRzW2ldKTtcbiAgICAgIH1cblxuICAgICAgLy90cmFpbiBzZXF1ZW5jZVxuICAgICAgdmFyIGRpc3RyYWN0b3JzQ29ycmVjdDtcbiAgICAgIHZhciB0YXJnZXRzQ29ycmVjdCA9IGRpc3RyYWN0b3JzQ29ycmVjdCA9IDA7XG4gICAgICBlcnJvciA9IDA7XG4gICAgICBmb3IgKGkgPSAwOyBpIDwgbGVuZ3RoOyBpKyspIHtcbiAgICAgICAgLy8gZ2VuZXJhdGUgaW5wdXQgZnJvbSBzZXF1ZW5jZVxuICAgICAgICB2YXIgaW5wdXQgPSBbXTtcbiAgICAgICAgZm9yIChqID0gMDsgaiA8IHN5bWJvbHM7IGorKylcbiAgICAgICAgICBpbnB1dFtqXSA9IDA7XG4gICAgICAgIGlucHV0W3NlcXVlbmNlW2ldXSA9IDE7XG5cbiAgICAgICAgLy8gZ2VuZXJhdGUgdGFyZ2V0IG91dHB1dFxuICAgICAgICB2YXIgb3V0cHV0ID0gW107XG4gICAgICAgIGZvciAoaiA9IDA7IGogPCB0YXJnZXRzLmxlbmd0aDsgaisrKVxuICAgICAgICAgIG91dHB1dFtqXSA9IDA7XG5cbiAgICAgICAgaWYgKGkgPj0gc2VxdWVuY2VMZW5ndGgpIHtcbiAgICAgICAgICB2YXIgaW5kZXggPSBpIC0gc2VxdWVuY2VMZW5ndGg7XG4gICAgICAgICAgb3V0cHV0W2luZGV4ZXNbaW5kZXhdXSA9IDE7XG4gICAgICAgIH1cblxuICAgICAgICAvLyBjaGVjayByZXN1bHRcbiAgICAgICAgdmFyIHByZWRpY3Rpb24gPSB0aGlzLm5ldHdvcmsuYWN0aXZhdGUoaW5wdXQpO1xuXG4gICAgICAgIGlmIChlcXVhbChwcmVkaWN0aW9uLCBvdXRwdXQpKVxuICAgICAgICAgIGlmIChpIDwgc2VxdWVuY2VMZW5ndGgpXG4gICAgICAgICAgICBkaXN0cmFjdG9yc0NvcnJlY3QrKztcbiAgICAgICAgICBlbHNlXG4gICAgICAgICAgICB0YXJnZXRzQ29ycmVjdCsrO1xuICAgICAgICBlbHNlIHtcbiAgICAgICAgICB0aGlzLm5ldHdvcmsucHJvcGFnYXRlKHJhdGUsIG91dHB1dCk7XG4gICAgICAgIH1cblxuICAgICAgICB2YXIgZGVsdGEgPSAwO1xuICAgICAgICBmb3IgKHZhciBqIGluIHByZWRpY3Rpb24pXG4gICAgICAgICAgZGVsdGEgKz0gTWF0aC5wb3cob3V0cHV0W2pdIC0gcHJlZGljdGlvbltqXSwgMik7XG4gICAgICAgIGVycm9yICs9IGRlbHRhIC8gdGhpcy5uZXR3b3JrLm91dHB1dHMoKTtcblxuICAgICAgICBpZiAoZGlzdHJhY3RvcnNDb3JyZWN0ICsgdGFyZ2V0c0NvcnJlY3QgPT0gbGVuZ3RoKVxuICAgICAgICAgIGNvcnJlY3QrKztcbiAgICAgIH1cblxuICAgICAgLy8gY2FsY3VsYXRlIGVycm9yXG4gICAgICBpZiAodHJpYWwgJSAxMDAwID09IDApXG4gICAgICAgIGNvcnJlY3QgPSAwO1xuICAgICAgdHJpYWwrKztcbiAgICAgIHZhciBkaXZpZGVFcnJvciA9IHRyaWFsICUgMTAwMDtcbiAgICAgIGRpdmlkZUVycm9yID0gZGl2aWRlRXJyb3IgPT0gMCA/IDEwMDAgOiBkaXZpZGVFcnJvcjtcbiAgICAgIHN1Y2Nlc3MgPSBjb3JyZWN0IC8gZGl2aWRlRXJyb3I7XG4gICAgICBlcnJvciAvPSBsZW5ndGg7XG5cbiAgICAgIC8vIGxvZ1xuICAgICAgaWYgKGxvZyAmJiB0cmlhbCAlIGxvZyA9PSAwKVxuICAgICAgICBjb25zb2xlLmxvZyhcIml0ZXJhdGlvbnM6XCIsIHRyaWFsLCBcIiBzdWNjZXNzOlwiLCBzdWNjZXNzLCBcIiBjb3JyZWN0OlwiLFxuICAgICAgICAgIGNvcnJlY3QsIFwiIHRpbWU6XCIsIERhdGUubm93KCkgLSBzdGFydCwgXCIgZXJyb3I6XCIsIGVycm9yKTtcbiAgICAgIGlmIChzY2hlZHVsZS5kbyAmJiBzY2hlZHVsZS5ldmVyeSAmJiB0cmlhbCAlIHNjaGVkdWxlLmV2ZXJ5ID09IDApXG4gICAgICAgIHNjaGVkdWxlLmRvKHtcbiAgICAgICAgICBpdGVyYXRpb25zOiB0cmlhbCxcbiAgICAgICAgICBzdWNjZXNzOiBzdWNjZXNzLFxuICAgICAgICAgIGVycm9yOiBlcnJvcixcbiAgICAgICAgICB0aW1lOiBEYXRlLm5vdygpIC0gc3RhcnQsXG4gICAgICAgICAgY29ycmVjdDogY29ycmVjdFxuICAgICAgICB9KTtcbiAgICB9XG5cbiAgICByZXR1cm4ge1xuICAgICAgaXRlcmF0aW9uczogdHJpYWwsXG4gICAgICBzdWNjZXNzOiBzdWNjZXNzLFxuICAgICAgZXJyb3I6IGVycm9yLFxuICAgICAgdGltZTogRGF0ZS5ub3coKSAtIHN0YXJ0XG4gICAgfVxuICB9XG5cbiAgLy8gdHJhaW4gdGhlIG5ldHdvcmsgdG8gbGVhcm4gYW4gRW1iZWRlZCBSZWJlciBHcmFtbWFyXG4gIEVSRyhvcHRpb25zKSB7XG5cbiAgICBvcHRpb25zID0gb3B0aW9ucyB8fCB7fTtcbiAgICB2YXIgaXRlcmF0aW9ucyA9IG9wdGlvbnMuaXRlcmF0aW9ucyB8fCAxNTAwMDA7XG4gICAgdmFyIGNyaXRlcmlvbiA9IG9wdGlvbnMuZXJyb3IgfHwgLjA1O1xuICAgIHZhciByYXRlID0gb3B0aW9ucy5yYXRlIHx8IC4xO1xuICAgIHZhciBsb2cgPSBvcHRpb25zLmxvZyB8fCA1MDA7XG5cbiAgICAvLyBncmFtYXIgbm9kZVxuICAgIHZhciBOb2RlID0gZnVuY3Rpb24oKSB7XG4gICAgICB0aGlzLnBhdGhzID0gW107XG4gICAgfVxuICAgIE5vZGUucHJvdG90eXBlID0ge1xuICAgICAgY29ubmVjdDogZnVuY3Rpb24obm9kZSwgdmFsdWUpIHtcbiAgICAgICAgdGhpcy5wYXRocy5wdXNoKHtcbiAgICAgICAgICBub2RlOiBub2RlLFxuICAgICAgICAgIHZhbHVlOiB2YWx1ZVxuICAgICAgICB9KTtcbiAgICAgICAgcmV0dXJuIHRoaXM7XG4gICAgICB9LFxuICAgICAgYW55OiBmdW5jdGlvbigpIHtcbiAgICAgICAgaWYgKHRoaXMucGF0aHMubGVuZ3RoID09IDApXG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICB2YXIgaW5kZXggPSBNYXRoLnJhbmRvbSgpICogdGhpcy5wYXRocy5sZW5ndGggfCAwO1xuICAgICAgICByZXR1cm4gdGhpcy5wYXRoc1tpbmRleF07XG4gICAgICB9LFxuICAgICAgdGVzdDogZnVuY3Rpb24odmFsdWUpIHtcbiAgICAgICAgZm9yICh2YXIgaSBpbiB0aGlzLnBhdGhzKVxuICAgICAgICAgIGlmICh0aGlzLnBhdGhzW2ldLnZhbHVlID09IHZhbHVlKVxuICAgICAgICAgICAgcmV0dXJuIHRoaXMucGF0aHNbaV07XG4gICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB2YXIgcmViZXJHcmFtbWFyID0gZnVuY3Rpb24oKSB7XG5cbiAgICAgIC8vIGJ1aWxkIGEgcmViZXIgZ3JhbW1hclxuICAgICAgdmFyIG91dHB1dCA9IG5ldyBOb2RlKCk7XG4gICAgICB2YXIgbjEgPSAobmV3IE5vZGUoKSkuY29ubmVjdChvdXRwdXQsIFwiRVwiKTtcbiAgICAgIHZhciBuMiA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4xLCBcIlNcIik7XG4gICAgICB2YXIgbjMgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuMSwgXCJWXCIpLmNvbm5lY3QobjIsIFwiUFwiKTtcbiAgICAgIHZhciBuNCA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4yLCBcIlhcIilcbiAgICAgIG40LmNvbm5lY3QobjQsIFwiU1wiKTtcbiAgICAgIHZhciBuNSA9IChuZXcgTm9kZSgpKS5jb25uZWN0KG4zLCBcIlZcIilcbiAgICAgIG41LmNvbm5lY3QobjUsIFwiVFwiKTtcbiAgICAgIG4yLmNvbm5lY3QobjUsIFwiWFwiKVxuICAgICAgdmFyIG42ID0gKG5ldyBOb2RlKCkpLmNvbm5lY3QobjQsIFwiVFwiKS5jb25uZWN0KG41LCBcIlBcIik7XG4gICAgICB2YXIgaW5wdXQgPSAobmV3IE5vZGUoKSkuY29ubmVjdChuNiwgXCJCXCIpXG5cbiAgICAgIHJldHVybiB7XG4gICAgICAgIGlucHV0OiBpbnB1dCxcbiAgICAgICAgb3V0cHV0OiBvdXRwdXRcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBidWlsZCBhbiBlbWJlZGVkIHJlYmVyIGdyYW1tYXJcbiAgICB2YXIgZW1iZWRlZFJlYmVyR3JhbW1hciA9IGZ1bmN0aW9uKCkge1xuICAgICAgdmFyIHJlYmVyMSA9IHJlYmVyR3JhbW1hcigpO1xuICAgICAgdmFyIHJlYmVyMiA9IHJlYmVyR3JhbW1hcigpO1xuXG4gICAgICB2YXIgb3V0cHV0ID0gbmV3IE5vZGUoKTtcbiAgICAgIHZhciBuMSA9IChuZXcgTm9kZSkuY29ubmVjdChvdXRwdXQsIFwiRVwiKTtcbiAgICAgIHJlYmVyMS5vdXRwdXQuY29ubmVjdChuMSwgXCJUXCIpO1xuICAgICAgcmViZXIyLm91dHB1dC5jb25uZWN0KG4xLCBcIlBcIik7XG4gICAgICB2YXIgbjIgPSAobmV3IE5vZGUpLmNvbm5lY3QocmViZXIxLmlucHV0LCBcIlBcIikuY29ubmVjdChyZWJlcjIuaW5wdXQsXG4gICAgICAgIFwiVFwiKTtcbiAgICAgIHZhciBpbnB1dCA9IChuZXcgTm9kZSkuY29ubmVjdChuMiwgXCJCXCIpO1xuXG4gICAgICByZXR1cm4ge1xuICAgICAgICBpbnB1dDogaW5wdXQsXG4gICAgICAgIG91dHB1dDogb3V0cHV0XG4gICAgICB9XG5cbiAgICB9XG5cbiAgICAvLyBnZW5lcmF0ZSBhbiBFUkcgc2VxdWVuY2VcbiAgICB2YXIgZ2VuZXJhdGUgPSBmdW5jdGlvbigpIHtcbiAgICAgIHZhciBub2RlID0gZW1iZWRlZFJlYmVyR3JhbW1hcigpLmlucHV0O1xuICAgICAgdmFyIG5leHQgPSBub2RlLmFueSgpO1xuICAgICAgdmFyIHN0ciA9IFwiXCI7XG4gICAgICB3aGlsZSAobmV4dCkge1xuICAgICAgICBzdHIgKz0gbmV4dC52YWx1ZTtcbiAgICAgICAgbmV4dCA9IG5leHQubm9kZS5hbnkoKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBzdHI7XG4gICAgfVxuXG4gICAgLy8gdGVzdCBpZiBhIHN0cmluZyBtYXRjaGVzIGFuIGVtYmVkZWQgcmViZXIgZ3JhbW1hclxuICAgIHZhciB0ZXN0ID0gZnVuY3Rpb24oc3RyKSB7XG4gICAgICB2YXIgbm9kZSA9IGVtYmVkZWRSZWJlckdyYW1tYXIoKS5pbnB1dDtcbiAgICAgIHZhciBpID0gMDtcbiAgICAgIHZhciBjaCA9IHN0ci5jaGFyQXQoaSk7XG4gICAgICB3aGlsZSAoaSA8IHN0ci5sZW5ndGgpIHtcbiAgICAgICAgdmFyIG5leHQgPSBub2RlLnRlc3QoY2gpO1xuICAgICAgICBpZiAoIW5leHQpXG4gICAgICAgICAgcmV0dXJuIGZhbHNlO1xuICAgICAgICBub2RlID0gbmV4dC5ub2RlO1xuICAgICAgICBjaCA9IHN0ci5jaGFyQXQoKytpKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIC8vIGhlbHBlciB0byBjaGVjayBpZiB0aGUgb3V0cHV0IGFuZCB0aGUgdGFyZ2V0IHZlY3RvcnMgbWF0Y2hcbiAgICB2YXIgZGlmZmVyZW50ID0gZnVuY3Rpb24oYXJyYXkxLCBhcnJheTIpIHtcbiAgICAgIHZhciBtYXgxID0gMDtcbiAgICAgIHZhciBpMSA9IC0xO1xuICAgICAgdmFyIG1heDIgPSAwO1xuICAgICAgdmFyIGkyID0gLTE7XG4gICAgICBmb3IgKHZhciBpIGluIGFycmF5MSkge1xuICAgICAgICBpZiAoYXJyYXkxW2ldID4gbWF4MSkge1xuICAgICAgICAgIG1heDEgPSBhcnJheTFbaV07XG4gICAgICAgICAgaTEgPSBpO1xuICAgICAgICB9XG4gICAgICAgIGlmIChhcnJheTJbaV0gPiBtYXgyKSB7XG4gICAgICAgICAgbWF4MiA9IGFycmF5MltpXTtcbiAgICAgICAgICBpMiA9IGk7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgcmV0dXJuIGkxICE9IGkyO1xuICAgIH1cblxuICAgIHZhciBpdGVyYXRpb24gPSAwO1xuICAgIHZhciBlcnJvciA9IDE7XG4gICAgdmFyIHRhYmxlID0ge1xuICAgICAgXCJCXCI6IDAsXG4gICAgICBcIlBcIjogMSxcbiAgICAgIFwiVFwiOiAyLFxuICAgICAgXCJYXCI6IDMsXG4gICAgICBcIlNcIjogNCxcbiAgICAgIFwiRVwiOiA1XG4gICAgfVxuXG4gICAgdmFyIHN0YXJ0ID0gRGF0ZS5ub3coKTtcbiAgICB3aGlsZSAoaXRlcmF0aW9uIDwgaXRlcmF0aW9ucyAmJiBlcnJvciA+IGNyaXRlcmlvbikge1xuICAgICAgdmFyIGkgPSAwO1xuICAgICAgZXJyb3IgPSAwO1xuXG4gICAgICAvLyBFUkcgc2VxdWVuY2UgdG8gbGVhcm5cbiAgICAgIHZhciBzZXF1ZW5jZSA9IGdlbmVyYXRlKCk7XG5cbiAgICAgIC8vIGlucHV0XG4gICAgICB2YXIgcmVhZCA9IHNlcXVlbmNlLmNoYXJBdChpKTtcbiAgICAgIC8vIHRhcmdldFxuICAgICAgdmFyIHByZWRpY3QgPSBzZXF1ZW5jZS5jaGFyQXQoaSArIDEpO1xuXG4gICAgICAvLyB0cmFpblxuICAgICAgd2hpbGUgKGkgPCBzZXF1ZW5jZS5sZW5ndGggLSAxKSB7XG4gICAgICAgIHZhciBpbnB1dCA9IFtdO1xuICAgICAgICB2YXIgdGFyZ2V0ID0gW107XG4gICAgICAgIGZvciAodmFyIGogPSAwOyBqIDwgNjsgaisrKSB7XG4gICAgICAgICAgaW5wdXRbal0gPSAwO1xuICAgICAgICAgIHRhcmdldFtqXSA9IDA7XG4gICAgICAgIH1cbiAgICAgICAgaW5wdXRbdGFibGVbcmVhZF1dID0gMTtcbiAgICAgICAgdGFyZ2V0W3RhYmxlW3ByZWRpY3RdXSA9IDE7XG5cbiAgICAgICAgdmFyIG91dHB1dCA9IHRoaXMubmV0d29yay5hY3RpdmF0ZShpbnB1dCk7XG5cbiAgICAgICAgaWYgKGRpZmZlcmVudChvdXRwdXQsIHRhcmdldCkpXG4gICAgICAgICAgdGhpcy5uZXR3b3JrLnByb3BhZ2F0ZShyYXRlLCB0YXJnZXQpO1xuXG4gICAgICAgIHJlYWQgPSBzZXF1ZW5jZS5jaGFyQXQoKytpKTtcbiAgICAgICAgcHJlZGljdCA9IHNlcXVlbmNlLmNoYXJBdChpICsgMSk7XG5cbiAgICAgICAgdmFyIGRlbHRhID0gMDtcbiAgICAgICAgZm9yICh2YXIgayBpbiBvdXRwdXQpXG4gICAgICAgICAgZGVsdGEgKz0gTWF0aC5wb3codGFyZ2V0W2tdIC0gb3V0cHV0W2tdLCAyKVxuICAgICAgICBkZWx0YSAvPSBvdXRwdXQubGVuZ3RoO1xuXG4gICAgICAgIGVycm9yICs9IGRlbHRhO1xuICAgICAgfVxuICAgICAgZXJyb3IgLz0gc2VxdWVuY2UubGVuZ3RoO1xuICAgICAgaXRlcmF0aW9uKys7XG4gICAgICBpZiAoaXRlcmF0aW9uICUgbG9nID09IDApIHtcbiAgICAgICAgY29uc29sZS5sb2coXCJpdGVyYXRpb25zOlwiLCBpdGVyYXRpb24sIFwiIHRpbWU6XCIsIERhdGUubm93KCkgLSBzdGFydCxcbiAgICAgICAgICBcIiBlcnJvcjpcIiwgZXJyb3IpO1xuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICBpdGVyYXRpb25zOiBpdGVyYXRpb24sXG4gICAgICBlcnJvcjogZXJyb3IsXG4gICAgICB0aW1lOiBEYXRlLm5vdygpIC0gc3RhcnQsXG4gICAgICB0ZXN0OiB0ZXN0LFxuICAgICAgZ2VuZXJhdGU6IGdlbmVyYXRlXG4gICAgfVxuICB9XG5cbn1cblxuZXhwb3J0IG1vZHVsZSBUcmFpbmVyIHtcbiAgLy8gQnVpbHQtaW4gY29zdCBmdW5jdGlvbnNcbiAgXG4gIGV4cG9ydCBpbnRlcmZhY2UgSVRyYWluZXJDb3N0Rm4ge1xuICAgICh0YXJnZXQsIG91dHB1dCk6IG51bWJlcjtcbiAgfVxuXG4gIGV4cG9ydCB2YXIgY29zdCA9IHtcbiAgICAvLyBFcS4gOVxuICAgIENST1NTX0VOVFJPUFk6IGZ1bmN0aW9uKHRhcmdldCwgb3V0cHV0KSB7XG4gICAgICB2YXIgY3Jvc3NlbnRyb3B5ID0gMDtcbiAgICAgIGZvciAodmFyIGkgaW4gb3V0cHV0KVxuICAgICAgICBjcm9zc2VudHJvcHkgLT0gKHRhcmdldFtpXSAqIE1hdGgubG9nKG91dHB1dFtpXSArIDFlLTE1KSkgKyAoKDEgLSB0YXJnZXRbaV0pICogTWF0aC5sb2coKDEgKyAxZS0xNSkgLSBvdXRwdXRbaV0pKTsgLy8gKzFlLTE1IGlzIGEgdGlueSBwdXNoIGF3YXkgdG8gYXZvaWQgTWF0aC5sb2coMClcbiAgICAgIHJldHVybiBjcm9zc2VudHJvcHk7XG4gICAgfSxcbiAgICBNU0U6IGZ1bmN0aW9uKHRhcmdldCwgb3V0cHV0KSB7XG4gICAgICB2YXIgbXNlID0gMDtcbiAgICAgIGZvciAodmFyIGkgaW4gb3V0cHV0KVxuICAgICAgICBtc2UgKz0gTWF0aC5wb3codGFyZ2V0W2ldIC0gb3V0cHV0W2ldLCAyKTtcbiAgICAgIHJldHVybiBtc2UgLyBvdXRwdXQubGVuZ3RoO1xuICAgIH1cbiAgfVxufSJdLCJzb3VyY2VSb290IjoiL3NvdXJjZS8ifQ==