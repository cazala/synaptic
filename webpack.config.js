var webpack = require('webpack');
var license = require('./prebuild.js');

module.exports = {
  context: __dirname,
  entry: [
    './src/synaptic.js'
  ],
  output: {
    path: 'dist',
    filename: 'synaptic.js',
    libraryTarget: 'umd'
  },
  module: {
    loaders: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'babel',
        query: {
          presets: ['es2015']
        }
      }
    ]
  },
  plugins: [
    new webpack.NoErrorsPlugin(),
    new webpack.BannerPlugin(license())
  ]
};
