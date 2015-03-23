'use strict';

/**
 * @ngdoc overview
 * @name gitHubApp
 * @description
 * # gitHubApp
 *
 * Main module of the application.
 */
angular
  .module('gitHubApp', [
    'ngAnimate',
    'ngCookies',
    'ngResource',
    'ngRoute',
    'ngSanitize',
    'ngTouch',
    'ui.bootstrap'
  ])
  .config(function ($routeProvider) {
    $routeProvider
      .when('/', {
        templateUrl: 'views/main.html',
        controller: 'MainCtrl'
      })
      .when('/about', {
        templateUrl: 'views/about.html',
        controller: 'AboutCtrl'
      })
      .when('/wikipedia', {
        templateUrl: 'views/wikipedia.html',
        controller: 'WikipediaCtrl'
      })
      .when('/xor', {
        templateUrl: 'views/xor.html',
        controller: 'XorCtrl'
      })
      .when('/dsr', {
        templateUrl: 'views/dsr.html',
        controller: 'DsrCtrl'
      })
      .when('/image-filters', {
        templateUrl: 'views/image-filters.html',
        controller: 'ImageFiltersCtrl'
      })
      .when('/paint-an-image', {
        templateUrl: 'views/paint-an-image.html',
        controller: 'PaintAnImageCtrl'
      })
      .when('/self-organizing-map', {
        templateUrl: 'views/self-organizing-map.html',
        controller: 'SelfOrganizingMapCtrl'
      })
      .otherwise({
        redirectTo: '/'
      });
  });
