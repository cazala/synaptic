var webpack = require('webpack')
var license = require('./prebuild.js')
module.exports = {
  context: __dirname,
  entry: [
    './src/synaptic.js'
  ],
  output: {
    path: 'dist',
    filename: 'synaptic.js',
  },
  plugins: [
    new webpack.NoErrorsPlugin(),
    new webpack.BannerPlugin(license())
  ]
}
