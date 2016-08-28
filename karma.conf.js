var webpack = require('webpack');
var webpackConfig = require('./webpack.config');
// Karma configuration

module.exports = function(config) {
  config.set({
    basePath: '',
    frameworks: ['mocha'],
    files: [
      'test/*.js',
    ],
    exclude: [
    ],
    preprocessors: {
      'test/*.js': ['webpack'],
    },
    client: {
        grep: /^[^(Node)]/
    },
    reporters: ['progress'],
    port: 9876,
    colors: true,
    logLevel: config.LOG_INFO,
    autoWatch: true,
    singleRun: false,
    concurrency: Infinity,
    browserNoActivityTimeout: 60000,
    webpack: {
      loaders: webpackConfig.loaders,
      plugins: webpackConfig.plugins
        .concat([
        // does not affect production build, only used for tests
        new webpack.DefinePlugin({
          'process.env.SYNAPTIC_PREFER_SRC': JSON.stringify(process.env.SYNAPTIC_PREFER_SRC),
          'process.env.KARMA': 'true',
        }),
        new webpack.NoErrorsPlugin(),
      ])
    },
  })
}
