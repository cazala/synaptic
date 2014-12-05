var license = '/*\n\nThe MIT License (MIT)\n\nCopyright (c) 2014 Juan Cazala - juancazala.com\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in\nall copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\nTHE SOFTWARE\n\n\n\n********************************************************************************************\n                                         SYNAPTIC\n********************************************************************************************\n\nSynaptic is a javascript neural network library for node.js and the browser, its generalized\nalgorithm is architecture-free, so you can build and train basically any type of first order\nor even second order neural network architectures.\n\nhttp://en.wikipedia.org/wiki/Recurrent_neural_network#Second_Order_Recurrent_Neural_Network\n\nThe library includes a few built-in architectures like multilayer perceptrons, multilayer\nlong-short term memory networks (LSTM) or liquid state machines, and a trainer capable of\ntraining any given network, and includes built-in training tasks/tests like solving an XOR,\npassing a Distracted Sequence Recall test or an Embeded Reber Grammar test.\n\nThe algorithm implemented by this library has been taken from Derek D. Monner\'s paper:\n\n\nA generalized LSTM-like training algorithm for second-order recurrent neural networks\nhttp://www.overcomplete.net/papers/nn2012.pdf\n\nThere are references to the equations in that paper commented through the source code.\n\n\n********************************************************************************************/\n'

// import
var gulp = require('gulp');
var browserify = require('browserify');
var transform = require('vinyl-transform');
var uglify = require('gulp-uglify');
var mocha = require('gulp-mocha');
var prepend = require('gulp-insert').prepend;

// build 
gulp.task('default', function () {
  var browserified = transform(function(filename) {
    var b = browserify(filename);
    return b.bundle();
  });
  
  return gulp.src(['./src/synaptic.js'])
    .pipe(browserified)
    .pipe(uglify())
    .pipe(prepend(license))
    .pipe(gulp.dest('./dist'));
});

// run tests with mocha
gulp.task('test', function () {
    return gulp.src('test/synaptic.js', {read: false})
        .pipe(mocha());
});

// uncompressed build, with sourcemaps
gulp.task('debug', function () {
  var browserified = transform(function(filename) {
    var b = browserify({ entries: [filename], debug: true });
    return b.bundle();
  });
  
  return gulp.src(['./src/synaptic.js'])
    .pipe(browserified)
    .pipe(gulp.dest('./dist'));
});

// watch for changed and re-build (debug)
gulp.task('dev', function () {
   gulp.watch('./src/*.js', ['debug']);
});
