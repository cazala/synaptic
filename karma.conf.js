// Karma configuration
const webpack = require('webpack');
const webpackConfig = require('./webpack.config');

const ignoreLibCovStatsModule = new webpack.IgnorePlugin(/lib-cov\/stats/);
webpackConfig.plugins = [ignoreLibCovStatsModule];

module.exports = function(config) {
  config.set({
    basePath: '',
    frameworks: ['mocha'],
    files: [
      'test/[^_]*.js'
    ],
    exclude: [
    ],
    preprocessors: {
      'test/*.js': ['webpack'],
    },
    reporters: ['progress'],
    port: 9876,
    colors: true,
    logLevel: config.LOG_INFO,
    autoWatch: true,
    singleRun: false,
    concurrency: Infinity,
    browserNoActivityTimeout: 60000,
    webpack: webpackConfig
  })
}
