const webpack = require('webpack');
const license = require('./prebuild.js');
const path = require('path');

module.exports = {
  context: __dirname,
  entry: {
    synaptic: path.resolve(__dirname, './src/synaptic.js'),
    'synaptic.min': path.resolve(__dirname, './src/synaptic.js')
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /(node_modules|bower_components)/,
        use: {
          loader: 'babel-loader'
        }
      }
    ]
  },
  output: {
    path: path.resolve(__dirname, './dist'),
    filename: '[name].js',
    library: 'synaptic',
    libraryTarget: 'umd'
  },
  plugins: [
    new webpack.NoErrorsPlugin(),
    new webpack.BannerPlugin(license())
  ]
}
