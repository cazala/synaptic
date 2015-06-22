import network = require('../network');
import trainer = require('../trainer');
import Layer = require('../layer');
import neuron = require('../neuron');
import Synaptic = require('../synaptic');
import Squash = require('../squash');

export class MemoryTape {
  data: Float64Array[];

  blockWidth: number;
  blocks: number;

  layer: Layer.Layer;

  prevLayerActivate: any;
  prevLayerPropagate: any;

  memoryAttentionLocation: number = 0;
  memoryAttentionWeight: number = 0;

  outputLayer: Layer.Layer;

  constructor(
    memoryBlocks: number,
    layer: Layer.Layer,
    inputGate: Layer.Layer,
    forgetGate: Layer.Layer
    ) {

    this.blocks = memoryBlocks;

    this.data = new Array(memoryBlocks);

    this.blockWidth = layer.list.length;

    this.layer = layer;
    
    this.outputLayer = new Layer.Layer(layer.list.length);
    
    this.outputLayer.project(layer, Layer.Layer.connectionType.ONE_TO_ONE);

    for (var location = 0; location < this.blocks; location++) {
      var array = this.data[location] = new Float64Array(this.blockWidth);
      for (var i = 0; i < array.length; i++) {
        array[i] = Math.random();
      }
    }
    // Hack the layer!
    
    var similarAddresses : Float64Array;
    // elegible memblocks for read/write operations
    var elegibleIndexes = [0, 1, 2];
    var elegibleWeights = [0.01, 1, 0.01]; // shifting, softmax
    var focus = 1;
    
    this.prevLayerActivate = this.layer.activate.bind(this.layer);
    this.prevLayerPropagate = this.layer.propagate.bind(this.layer);
    
    var key: Float64Array;
    
    this.layer.propagate = (rate: number, target?: Synaptic.INumericArray) => {
      this.prevLayerPropagate(rate, target);
      
      var addGate = inputGate.currentActivation;
      var eraseGate = forgetGate.currentActivation;
      
      for (var n = 0; n < elegibleIndexes.length; n++) {
        var M = this.data[elegibleIndexes[n]];
       
        for (var i = 0; i < M.length; i++) {
          // do erase operations on the memory tape. NTM: 3.2 (3)
          M[i] *= 1 - eraseGate[i] * key[i] * elegibleWeights[n];
          // do add operations on the memory tape. NTM: 3.2 (4)
          M[i] += addGate[i] * key[i] * elegibleWeights[n] * rate;
        }
      }
    }

    this.layer.activate = (input?: Synaptic.INumericArray): Float64Array => {
      var result = this.prevLayerActivate(input); 
      key = MemoryTape.softMaxArray( new Float64Array(result));
      
      similarAddresses = this.getSimilarAdresses(key);
      
      this.memoryAttentionWeight = 0;
      
      for (var address = 0; address < similarAddresses.length; address++) {
        var ß = similarAddresses[address];
        if(ß > this.memoryAttentionWeight){
          this.memoryAttentionWeight = ß;
          this.memoryAttentionLocation = address;
        }
      }
      
      elegibleIndexes = [this.memoryAttentionLocation - 1, this.memoryAttentionLocation, this.memoryAttentionLocation + 1];
      
      focus = this.memoryAttentionWeight;

      elegibleWeights = [0.1, 0.8, 0.1]; // shifting, softmax

      for (var n = 0; n < elegibleIndexes.length; n++) {
        var index = elegibleIndexes[n];
        
        if(index < 0)
          index += similarAddresses.length;
        else if(index >= similarAddresses.length)
          index -= similarAddresses.length;
          
        elegibleIndexes[n] = index;
        
        elegibleWeights[n] = elegibleWeights[n] / focus * similarAddresses[index];
      }

      this.outputLayer.list.forEach((neuron, i) => {
        // modify the current key (readVector)
        var tmpKey = 0;
        
        for (var n = 0; n < elegibleIndexes.length; n++) {
          tmpKey += this.data[elegibleIndexes[n]][i] * elegibleWeights[n];
        }

        neuron.activate(tmpKey);
      });

      return result;
    }
  }

  static getSimilarity(arrayA: Synaptic.INumericArray, arrayB: Synaptic.INumericArray): number {
    // http://en.wikipedia.org/wiki/Cosine_similarity
    // NTM: 3.3.1 (6)
    var dotPr = 0;

    var acumA = 0, acumB = 0;

    for (var i = 0; i < arrayA.length; i++) {
      dotPr += arrayA[i] * arrayB[i];
      acumA += arrayA[i] * arrayA[i];
      acumB += arrayB[i] * arrayB[i];
    }

    return dotPr / (Math.sqrt(acumA) * Math.sqrt(acumB) + .00005);
  }

  static softMaxArray<T extends Synaptic.INumericArray>(array: T, sharpen = 1): T {
    // for all i ∈ array
    // sum = ∑ array[n]^e
    // i = î^e / sum
    // where the result ∑ array[0..n] = 1

    if (!array.length) return array;

    sharpen = sharpen || 1;

    var sum = 0;

    // sum = ∑ array[n]^e
    for (var i = 0; i < array.length; i++) {
      array[i] = Math.exp(sharpen * array[i]);
      sum += array[i];
    }

    if (sum != 0) {
      for (var i = 0; i < array.length; i++) array[i] /= sum;
    } else {
      var div = 1 / array.length;
      for (var i = 0; i < array.length; i++) array[i] = div;
    }

    return array;
  }
  
  // obtains an array of similarity indexes for each memoryBlock
  getSimilarAdresses(weights: Synaptic.INumericArray): Float64Array {
    //checkpoint: 10th cigarret
    var addresses = new Float64Array(this.data.length);

    for (var i = 0; i < this.data.length; i++)
      addresses[i] = MemoryTape.getSimilarity(this.data[i], weights);

    return addresses;
  }
}


export class MemoryBlock extends network.Network {
  trainer: trainer.Trainer;
  memoryTape: MemoryTape;

  constructor(inputSize: number, memoryBlocks: number, memoryWidth: number, outputSize: number) {


    var option = {
      peepholes: Layer.Layer.connectionType.ALL_TO_ALL,
      hiddentohidden: false,
      outtohidden: false,
      outtogates: false,
      intoout: true,
    };

    var inputLayer = new Layer.Layer(inputSize);
    var hiddenLayers = [];
    var outputLayer = new Layer.Layer(outputSize);


    //#region generate layers

    // generate memory blocks (memory cell and respective gates)

    var inputGate = new Layer.Layer(memoryWidth).set({
      bias: 1
    });

    var forgetGate = new Layer.Layer(memoryWidth).set({
      bias: 1
    });

    var memoryCell = new Layer.Layer(memoryWidth);



    var outputGate = new Layer.Layer(memoryWidth).set({
      bias: 1
    });

    hiddenLayers.push(inputGate);
    hiddenLayers.push(forgetGate);
    hiddenLayers.push(memoryCell);
    hiddenLayers.push(outputGate);

    // connections from input layer
    var input = inputLayer.project(memoryCell);
    inputLayer.project(inputGate);
    inputLayer.project(forgetGate);
    inputLayer.project(outputGate);

    // connections from memory cell
    var output = memoryCell.project(outputLayer);

    // self-connection
    var self = memoryCell.project(memoryCell);
      
    // peepholes
    memoryCell.project(inputGate, option.peepholes);
    memoryCell.project(forgetGate, option.peepholes);
    memoryCell.project(outputGate, option.peepholes);

    // gates
    inputGate.gate(input, Layer.Layer.gateType.INPUT);
    forgetGate.gate(self, Layer.Layer.gateType.ONE_TO_ONE);
    outputGate.gate(output, Layer.Layer.gateType.OUTPUT);


    this.memoryTape = new MemoryTape(memoryBlocks, memoryCell, inputGate, forgetGate);
  
    hiddenLayers.push(this.memoryTape.outputLayer);    
    
    //#endregion

    // input to output direct connection
    if (option.intoout)
      inputLayer.project(outputLayer);

    // set the layers of the neural network
    super({
      input: inputLayer,
      hidden: hiddenLayers,
      output: outputLayer
    });
    
    
    // DO NOT OPTIMIZE THIS NETWORK
    this.optimized = false;

    // trainer
    this.trainer = new trainer.Trainer(this);
  }
};
