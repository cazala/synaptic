var code = 'process.on("message", function(msg){ onmessage({data:msg}) });\n' +
           'function postMessage(msg){ process.send(msg) }\n' +
           process.argv[2];

try {
  eval(code);
} catch(err) {
  process.send({
    action: 'log',
    message: 'Synaptic Worker Fail: '+err.message+'\n'+err.stack
  });
}
