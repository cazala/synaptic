'use strict';

/**
 * @ngdoc function
 * @name gitHubApp.controller:PaintAnImageCtrl
 * @description
 * # PaintAnImageCtrl
 * Controller of the gitHubApp
 */
angular.module('gitHubApp')
  .controller('PaintAnImageCtrl', function ($scope, $rootScope, $timeout, $location) {
  	$rootScope.navbarActive = "demos";

    var perceptron = null;
	var worker = null;
	var index = 0;
	var twitter_data = null;
	var canvas = null;
	var context = null;
	var size = 125 * 125;
	var iteration = 0;
	var to = null;
	var px = null;

	var getData = function(imageObj){

		canvas = canvas || document.getElementById('canvas-demo4');
        context = context || canvas.getContext('2d');

        context.drawImage(imageObj, 0, 0);

        var imageData = context.getImageData(0, 0, 125, 125);
        return imageData.data;
	}

	var train = $scope.train = function(){

		$scope.trainingStarted = true;

		iteration = 0;
		to && clearTimeout(to);
		perceptron = new Architect.Perceptron(2,15,3);
        twitter_data = getData(document.getElementById('twitter'));
        $(".train").show();
		preview();
	}

	var iterate = function(){
		
		for (var x = 0; x < 125; x+=1)
		{
			for(var y = 0; y < 125; y+=1)
			{
				var dynamicRate =  .01/(1+.0005*iteration);
				px = pixel(twitter,x,y)
				perceptron.activate([x/125,y/125]);
				perceptron.propagate(dynamicRate, pixel(twitter_data,x,y));
			}
		}
		preview();
	}

	var pixel = function(data,x,y){

		var red = data[((125 * y) + x) * 4];
        var green = data[((125 * y) + x) * 4 + 1];
        var blue = data[((125 * y) + x) * 4 + 2];

        return [red / 255, green / 255, blue / 255];
	}

	var preview = function(){
		$scope.iteration = ++iteration;
		var imageData = context.getImageData(0, 0, 125, 125);
		for (var x = 0; x < 125; x++)
		{
			for(var y = 0; y < 125; y++)
			{
				var rgb = perceptron.activate([x/125,y/125]);
				imageData.data[((125 * y) + x) * 4] = (rgb[0] )* 255;
				imageData.data[((125 * y) + x) * 4 + 1] = (rgb[1] ) * 255;
				imageData.data[((125 * y) + x) * 4 + 2] = (rgb[2] ) * 255;
			}
		}
		context.putImageData(imageData,0,0);

		if ($location.$$path == '/paint-an-image')
			$timeout(iterate, 16);
	}
  });
