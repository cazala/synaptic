'use strict';

/**
 * @ngdoc function
 * @name gitHubApp.controller:ImageFiltersCtrl
 * @description
 * # ImageFiltersCtrl
 * Controller of the gitHubApp
 */
angular.module('gitHubApp')
  .controller('SelfOrganizingMapCtrl', function ($scope, $rootScope, $timeout, $location) {
    $rootScope.navbarActive = "demos";

    $scope.word = "";
    $scope.map = {};

    

    var first = true;
    var hopfield = new Architect.Hopfield(80);
    var trainer = new Trainer(hopfield, {cost: Trainer.cost.CROSS_ENTROPY});

    $scope.valid = function(){
    	var valid = typeof $scope.word == 'string';
    	for (var i in $scope.draw)
    		for (var j in $scope.draw[i])
    			if ($scope.draw[i][j] == $scope.word)
    				return false;
    	return valid;
    }
    $scope.keyup= function(evt){
    	if (evt.which == 32 || evt.which == 13)
    	{
    		if ($scope.valid() && $scope.word.length > 0)
    		{
    			feed($scope.word.trim());
    			$scope.word = "";
    		}
    	}
    }



    var feed = function(word){
    	console.log(word);
		var input = ascii2bin(word);
		console.log(input.join(''));
		var output = hopfield.feed(input);
		var key = output.join('');

		if (key in $scope.map)
		{
			$scope.map[key].push(word);
		} else {
			var learn = [];
			$scope.map[key] = [word];

			for (var i in $scope.map)
			{
				learn.push(i.split(''));
			}

			var set = [];
			for (var p in learn)
				set.push({
				  input: learn[p],
				  output: learn[p]
				});

			doTrain(set);
		}
		console.log(key);

		preview();
    }

    var ascii2bin = function(ascii)
    {
    	var bin = "00000000000000000000000000000000000000000000000000000000000000000000000000000000";
    	for (var i = 0; i < ascii.length; i++)
    	{
    		var code = ascii.charCodeAt(i);
    		bin += ('00000000' + code.toString(2)).slice(-8);
    	}
    	return bin.slice(-10 * 8).split('').reverse();
    }

    var bin2dec = function(bin){
    	return parseInt(bin.join(''), 2);
    }

    var doTrain = function(set){
    	trainer.train(set, {
			iterations: 100,
			error: .5,
			rate: .05
		});
    }

    $scope.map[ascii2bin("cat").join('')] = ["cat"];
    $scope.map[ascii2bin("dog").join('')] = ["dog"];
    $scope.map[ascii2bin("john").join('')] = ["john"];
    var learn = [ascii2bin("john"), ascii2bin("dog"), ascii2bin("cat")];
    var set = [];
	for (var p in learn)
		set.push({
		  input: learn[p],
		  output: learn[p]
		});

	doTrain(set);

	var preview = function(){
		var draw = []
		for (var i in $scope.map)
		{
			var row = draw.push([]) - 1;
			for (var j in $scope.map[i])
				draw[row].push($scope.map[i][j]);
		}
		$scope.draw = draw;
	}

	preview();

  });
