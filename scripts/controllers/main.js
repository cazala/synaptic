'use strict';

/**
 * @ngdoc function
 * @name gitHubApp.controller:MainCtrl
 * @description
 * # MainCtrl
 * Controller of the gitHubApp
 */
angular.module('gitHubApp')
  .controller('MainCtrl', function ($rootScope, $window) {
    $rootScope.navbarActive = "home";

    var elements = $('.reveal');
    var win = $(window);

    elements.css('opacity', 0);

    var isVisible = function(elem)
	{
	    var docViewTop = win.scrollTop();
	    var docViewBottom = docViewTop + win.height();

	    var elemTop = elem.offset().top;
	    var elemBottom = elemTop + elem.height();

	    return ((elemBottom <= docViewBottom) && (elemTop >= docViewTop));
	}

    win.scroll(function(event) {
	  elements.each(function(i, el) {
	    var el = $(el);
	    if (isVisible(el)) {
	      el.addClass("animated fadeInUp"); 
	    } 
	  });
	});
	
    blastoff(); // kick it
  });
