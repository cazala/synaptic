'use strict';

/**
 * @ngdoc function
 * @name gitHubApp.controller:AboutCtrl
 * @description
 * # AboutCtrl
 * Controller of the gitHubApp
 */
angular.module('gitHubApp')
  .controller('AboutCtrl', function ($scope) {
    $scope.awesomeThings = [
      'HTML5 Boilerplate',
      'AngularJS',
      'Karma'
    ];
  });
