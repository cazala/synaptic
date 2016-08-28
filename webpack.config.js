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
    libraryTarget: 'umd',
    umdNamedDefine: 'synaptic',
  },
  loaders: [
    {
      test: /\.js$/,
      exclude: /\/node_modules\//,
      loaders: ['babel'],
    },
  ],
  plugins: [
    new webpack.BannerPlugin(license()),
    new webpack.DefinePlugin({
      'process.env.WEBPACK': 'true',
    }),

  ]
};
