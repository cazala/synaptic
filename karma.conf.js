// Karma configuration

module.exports = function(config) {
  config.set({
    basePath: '',
    frameworks: ['mocha'],
    files: [
      'dist/synaptic.js',
      'test/[^_]*.js'
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
  })
}
