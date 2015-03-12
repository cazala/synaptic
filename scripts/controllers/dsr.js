'use strict';

/**
 * @ngdoc function
 * @name gitHubApp.controller:DsrCtrl
 * @description
 * # DsrCtrl
 * Controller of the gitHubApp
 */
angular.module('gitHubApp')
  .controller('DsrCtrl', function ($scope, $rootScope, $timeout) {
    $rootScope.navbarActive = "demos";

    var LSTM = null;

	var targets = [2,4];
	var distractors = [3,5];
	var prompts = [0, 1];
	var length = 10;

	var sequence = null;
	var sequenceLength = 0;
	var targetsCorrect = 0;
	var distractorsCorrect = 0;
	var indexes = [];
	var positions = [];
	var symbol = 0;
	var symbols = targets.length + distractors.length + prompts.length;
	var json = null;

	$scope.sequence = [];

	$scope.train = function(){

		$scope.trainingStarted = true;
		$scope.trainingNetwork = true;

		var inputs = symbols;
		var outputs = targets.length;
		var hidden = inputs * .8 | 0;

		// create long short term memory network
		LSTM = new Architect.LSTM(inputs,hidden,outputs);

		$("body").animate({ scrollTop: $(document).height() }, function()
			{
			$timeout(function(){

				var results = LSTM.trainer.DSR({
					targets: targets,
					distractors: distractors,
					prompts: prompts,
					length: length,
					iterations: 250000,
					rate: .17
				});

				if (results.iterations < 250000)
				{
					$scope.results = results;
					validate();
				} else{
					train();
				}
			}, 32);
		});
	}

	var validate = $scope.validate = function(){
		
		$scope.trainingNetwork = false;

		sequence = [], sequenceLength = length - prompts.length;
		for (i  = 0; i < sequenceLength; i ++)
		{
			var any = Math.random() * distractors.length | 0;
			sequence.push(distractors[any]);
		}
		indexes = [], positions = [];
		for (i = 0; i < prompts.length; i ++)
		{
			indexes.push(Math.random() * targets.length | 0);
			positions.push(noRepeat(sequenceLength, positions));
		}
		positions = positions.sort();
		for (i = 0; i < prompts.length; i ++)
		{
			sequence[positions[i]] = targets[indexes[i]];
			sequence.push(prompts[i]);
		}
		targetsCorrect = distractorsCorrect = symbol = 0;

		$scope.sequence = [];
		next();
	}

	var next = function(){
		// generate input from sequence
		var input = [];
		for (var j = 0; j < symbols; j++)
			input[j] = 0;
		input[sequence[symbol]] = 1;

		// generate target output
		var output = [];
		for (j = 0; j < targets.length; j++)
			output[j] = 0;

		if (symbol >= sequenceLength)
		{
			var index = symbol - sequenceLength;
			output[indexes[index]] = 1;
		}

		// check result
		var prediction = LSTM.activate(input);
		var ok = equal(prediction, output);
		if (ok)
			if (i < sequenceLength)
				distractorsCorrect++;
			else
				targetsCorrect++;
		var val = value(prediction);

		$scope.sequence.push({
			input: sequence[symbol],
			output: targets[val],
			ok: ok
		});

		symbol++;
		
		if (symbol < sequence.length)
		{
			$("body").animate({ scrollTop: $(document).height() }, function(){
				$timeout(next, 100);
			});
		} else {
			if (distractorsCorrect + targetsCorrect == length)
			{
				$scope.trainingSuccess = true;
				$timeout(function(){
					$("body").animate({ scrollTop: $(document).height() });
				}, 32);
			} else {
				validate();
			}
		}
	}

	var noRepeat = function(range, avoid)
	{
		var number = Math.random() * range | 0;
		var used = false;
		for (var i in avoid)
			if (number == avoid[i])
				used = true;
		return used ? noRepeat(range, avoid) : number;
	}
	var equal = function(prediction, output){
		for (var i in prediction)
			if (Math.round(prediction[i]) != output[i])
				return false;
		return true;
	}

	var value = function(array){
		var max = .5;
		var res = -1;
		for (var i in array)
			if (array[i] > max)
			{
				max = array[i];
				res = i;
			}
		return res == -1 ? 'none' : res;
	}

	$scope.getColor = function(id){
		if (typeof id == 'undefined')
			return "fa fa-circle-o text-muted";
		return "fa fa-circle text-" + (id == 0 ? 'muted' : id == 1 ? 'black' : id == 2 ? 'info' : id == 3 ? 'warning' : id == 4 ? 'success' : 'danger');
	}

	$scope.getTitle = function(id){
		if (typeof id == 'undefined')
			return "none";
		return id;
	}

  });
