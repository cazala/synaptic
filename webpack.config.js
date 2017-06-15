var webpack = require('webpack')
var license = require('./prebuild.js')

module.exports = {
  context: __dirname,
  entry: {
    synaptic: './src/synaptic.js',
    'synaptic.min': './src/synaptic.js'
  },
  output: {
    path: 'dist',
    filename: '[name].js',
    library: 'synaptic',
    libraryTarget: 'umd'
  },
  plugins: [
    new webpack.NoErrorsPlugin(),
    new webpack.BannerPlugin(license())
  ]
}
