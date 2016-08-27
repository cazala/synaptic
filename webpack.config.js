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
      exclude: /node_modules/,
      loader: 'babel',
      query: {
        presets: ['es2015']
      }
    },
    {
      test: /\.worker\.js/,
      loader: 'worker'
    }
  ],
  plugins: [
    // does not affect production build, only used for tests
    new webpack.DefinePlugin({
      SYNAPTIC_PREFER_SRC: JSON.stringify(process.env.SYNAPTIC_PREFER_SRC)
    }),
    new webpack.NoErrorsPlugin(),
    new webpack.BannerPlugin(license())
  ]
};
