// update license year and version
var fs = require('fs')
module.exports = function() {
  var version = require('./package.json').version
  var license = fs.readFileSync('LICENSE', 'utf-8')
  .replace(/\(c\) ([0-9]+)/, '(c) ' + (new Date).getFullYear())
  .replace(/SYNAPTIC \(v(.*)\)/, 'SYNAPTIC (v' + version + ')')
  fs.writeFileSync('LICENSE', license)
  return license
}
