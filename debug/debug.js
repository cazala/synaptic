var dataSets = {
  'XOR' : [
    { input: [0,0], output: [0] },
    { input: [0,1], output: [1] },
    { input: [1,0], output: [1] },
    { input: [1,1], output: [0] }
  ],
  'XNOR' : [
    { input: [0,0], output: [1] },
    { input: [0,1], output: [0] },
    { input: [1,0], output: [0] },
    { input: [1,1], output: [1] }
  ],
  'AND' : [
    { input: [0,0], output: [0] },
    { input: [0,1], output: [0] },
    { input: [1,0], output: [0] },
    { input: [1,1], output: [1] }
  ],
  'OR' : [
    { input: [0,0], output: [0] },
    { input: [0,1], output: [1] },
    { input: [1,0], output: [1] },
    { input: [1,1], output: [1] }
  ]
};

/** TEST SYNAPTIC **/
var iterations = 0;
var time = 0;
var error = 0;
for(var dataSet in dataSets){
  dataSet = dataSets[dataSet];

  var network = new synaptic.Architect.Perceptron(2, 4, 1);
  var trainer = new synaptic.Trainer(network);
  var results = trainer.train(dataSet, {
    iterations: 1000,
    error: 0.03,
    rate: 0.3
  });

  iterations += results.iterations;
  time += results.time;
  error += results.error;
}

$('.synaptic').append("Synaptic iterations: " + iterations);
$('.synaptic').append(", time: " + time + 'ms');
$('.synaptic').append(", error: " + error);
