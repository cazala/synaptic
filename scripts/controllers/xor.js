'use strict';

/**
 * @ngdoc function
 * @name gitHubApp.controller:XorCtrl
 * @description
 * # XorCtrl
 * Controller of the gitHubApp
 */
angular.module('gitHubApp')
  .controller('XorCtrl', function ($scope, $rootScope) {
    $rootScope.navbarActive = "demos";

    var perceptron = null;

    $scope.outputs = [];

	$scope.train = function(){
		perceptron = new synaptic.Architect.Perceptron(2,3,1);
		$scope.results = perceptron.trainer.XOR({ 
			iterations: 100000,
			error: .0001,
			rate: 1
		});
		validate();
	}

	var validate = function(){
		$scope.outputs = [];
		$scope.outputs.push({
			input: '0 0',
			output: perceptron.activate([0,0])[0].toFixed(3),
			target: 0
		});
		$scope.outputs.push({
			input: '0 1',
			output: perceptron.activate([0,1])[0].toFixed(3),
			target: 1
		});
		$scope.outputs.push({
			input: '1 0',
			output: perceptron.activate([1,0])[0].toFixed(3),
			target: 1
		});
		$scope.outputs.push({
			input: '1 1',
			output: perceptron.activate([1,1])[0].toFixed(3),
			target: 0
		});
	}
  });
