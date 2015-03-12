'use strict';

/**
 * @ngdoc function
 * @name gitHubApp.controller:ImageFiltersCtrl
 * @description
 * # ImageFiltersCtrl
 * Controller of the gitHubApp
 */
angular.module('gitHubApp')
  .controller('ImageFiltersCtrl', function ($scope, $rootScope, $timeout, $location) {
    $rootScope.navbarActive = "demos";

    $scope.filter = "Grayscale";
    $scope.url = "images/cat_grayscale.png";
    $scope.disableTrain = false;
    $scope.setFilter = function(filter){
    	$scope.filter = filter;
    	$scope.url = "images/cat_" + filter.toLowerCase().split(' ').join('_') + ".png";

    	$scope.disableTrain = false;
    }

    var perceptron = null;
	var index = 0;
	var color_data = null;
	var filtered_data = null;
	var original_data = null;
	var canvas = null;
	var context = null;
	var size = 125 * 125;
	var trial = 0;
	var px = null;

	canvas = canvas || document.getElementById('canvas-demo3');
    context = context || canvas.getContext('2d');

	var getData = function(imageObj){

		canvas = canvas || document.getElementById('canvas-demo3');
        context = context || canvas.getContext('2d');

        context.drawImage(imageObj, 0, 0);

        var imageData = context.getImageData(0, 0, 125, 125);
        return imageData.data;
	}

	var train = $scope.train = function(){

		$scope.disableTrain = true;

		trial = 0;

		perceptron = new Architect.Perceptron(27,8,3);
        color_data = getData(document.getElementById('input'));
        filtered_data = getData(document.getElementById('output'));
		original_data = getData(document.getElementById('original'));

		if (!$scope.trainingStarted)
		{
			$scope.trainingStarted = true;
			iteration();
		}
	}

	
	var iteration = function(){
		trial++;
		$scope.iteration = trial;

		for (index = 0; index < size; index+=2)
		{
			px = pixel(color_data, 0, 0);
			px = px.concat(pixel(color_data, -1, -1));
			px = px.concat(pixel(color_data, 0, -1));
			px = px.concat(pixel(color_data, 1, -1));
			px = px.concat(pixel(color_data, -1, 0));
			px = px.concat(pixel(color_data, 1, 0));
			px = px.concat(pixel(color_data, -1, 1));
			px = px.concat(pixel(color_data, 0, 1));
			px = px.concat(pixel(color_data, 1, 1));
			perceptron.activate(px);
			perceptron.propagate(.12, pixel(filtered_data,0,0));
		}
		preview();
	}

	var pixel = function(data, ox, oy){
		var y = index / 125 | 0;
		var x = index % 125;

		if (ox && (x + ox) > 0 && (x + ox) < 125)
			x += ox;
		if (oy && (y + oy) > 0 && (y + oy) < 125)
			y += oy;

		var red = data[((125 * y) + x) * 4];
        var green = data[((125 * y) + x) * 4 + 1];
        var blue = data[((125 * y) + x) * 4 + 2];

        return [red / 255, green / 255, blue / 255];
	}

	var preview = function(){
		var imageData = context.getImageData(0, 0, 125, 125);
		for (index = 0; index < size; index++)
		{
			var px = pixel(original_data, 0, 0);
			px = px.concat(pixel(original_data, -1, -1));
			px = px.concat(pixel(original_data, 0, -1));
			px = px.concat(pixel(original_data, 1, -1));
			px = px.concat(pixel(original_data, -1, 0));
			px = px.concat(pixel(original_data, 1, 0));
			px = px.concat(pixel(original_data, -1, 1));
			px = px.concat(pixel(original_data, 0, 1));
			px = px.concat(pixel(original_data, 1, 1));
			var rgb = perceptron.activate(px);
			imageData.data[index * 4] = (rgb[0] )* 255;
			imageData.data[index * 4 + 1] = (rgb[1] ) * 255;
			imageData.data[index * 4 + 2] = (rgb[2] ) * 255;
		}
		context.putImageData(imageData,0,0);
		
		if ($location.$$path == '/image-filters')
			$timeout(iteration, 16);
	}
  });
