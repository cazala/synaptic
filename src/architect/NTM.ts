import network = require('../network');
import trainer = require('../trainer');
import Layer = require('../layer');
import neuron = require('../neuron');
import Synaptic = require('../synaptic');
import Squash = require('../squash');
import _utils = require('../utils');

var Utils = _utils.Utils;

export class NTM extends network.Network {
  trainer: trainer.Trainer;

  data: Float64Array[];

  blockWidth: number;
  blocks: number;

  heads: Head[] = new Array();

  inputValues: Float64Array;

  inputLayer: Layer.Layer;
  hiddenLayer: Layer.Layer;
  outputLayer: Layer.Layer;

  dirty = false;

  constructor(inputs: number, outputs: number, memBlocks: number, blockWidth: number, heads: number, hiddenSize: number) {
    // build the memory
    
    super();

    this.trainer = new trainer.Trainer(this);

    this.blocks = memBlocks;
    this.blockWidth = blockWidth;

    this.data = new Array(this.blocks);
    for (var index = 0; index < this.data.length; index++) {
      this.data[index] = new Float64Array(blockWidth);

    }

    this.clean();
    
    // build the network
    
    var inputLength = inputs + heads * memBlocks;

    this.inputValues = new Float64Array(inputLength);

    this.layers.input = this.inputLayer = new Layer.Layer(inputLength);
    this.hiddenLayer = new Layer.Layer(hiddenSize);
    this.layers.output = this.outputLayer = new Layer.Layer(outputs);



    this.inputLayer.project(this.hiddenLayer, Layer.Layer.connectionType.ALL_TO_ALL);
    this.hiddenLayer.project(this.outputLayer, Layer.Layer.connectionType.ALL_TO_ALL);

    var inputCounter = inputs - 1;

    for (var headIndex = 0; headIndex < heads; headIndex++) {
      this.addHead(this.inputValues.subarray(inputCounter, inputCounter + memBlocks));
      inputCounter += memBlocks;
    }

    this.optimized = false;
  }

  clean() {
    for (var location = 0; location < this.blocks; location++) {
      Utils.initRandomSoftmaxArray(this.data[location]);
    }
    this.dirty = false;
  }

  activate(input: Synaptic.INumericArray) {
    this.inputValues.set(<any>input);

    this.inputLayer.activate(this.inputValues);
    this.hiddenLayer.activate();

    this.doTimeStep();

    return this.outputLayer.activate();
  }

  propagate(rate: number, target: Synaptic.INumericArray) {
    this.outputLayer.propagate(rate, target);
    for (var i = this.heads.length - 1; i >= 0; i--) {
      this.heads[i].layer.propagate(rate);
    }
    this.hiddenLayer.propagate(rate);
    this.dirty = true;
  }

  addHead(subArray: Float64Array): Head {
    var head = new Head(this, subArray);
    this.heads.push(head);
    return head;
  }

  doTimeStep() {
    this.heads.forEach((head, headIndex) => {
      head.doTimeStep();
    });
    
    // parallelizable
    this.heads.forEach((head, headIndex) => {
      this.doErase(head.w_weightings, head.eraseGate);
    });
    
    // parallelizable
    this.heads.forEach((head, headIndex) => {
      this.doAdd(head.w_weightings, head.addGate);
    });
    
    //this.data.forEach((e) => e = Utils.softMax(e))
  }

  doAdd(w: Synaptic.INumericArray, addGate: Synaptic.INumericArray) {
    for (var n = 0; n < this.blocks; n++) {
      var M = this.data[n];
      for (var i = 0; i < this.blockWidth; i++) {
        M[i] += addGate[n] * w[i];
      }
    }
  }

  doErase(w: Synaptic.INumericArray, eraseGate: Synaptic.INumericArray) {
    for (var n = 0; n < this.blocks; n++) {
      var M = this.data[n];
      for (var i = 0; i < this.blockWidth; i++) {
        M[i] *= 1 - eraseGate[n] * w[i];
      }
    }
  }
}



export class Head {
  static ADDITIONAL_INPUT_VALUES = 3;

  memory: NTM;

  w_weightings: Float64Array;
  eraseGate: Float64Array;
  addGate: Float64Array;
  k_keys: Float64Array;
  g_interpolation: number;
  Y_focus: number;
  s_shiftingValue: number = null;
  s_shiftingVector: Float64Array;
  wc_focusedWeights: Float64Array;
  readVector: Float64Array;
  ß_keyStrength: number;
  prevFocus: number = 1;

  shiftLength: number;

  layer: Layer.Layer;

  circulantMatrix: Float64Array[];

  constructor(memory: NTM, destinationArray?: Float64Array) {
    this.memory = memory;
    this.wc_focusedWeights = new Float64Array(this.memory.blocks);
    this.w_weightings = new Float64Array(this.memory.blocks);

    Utils.initRandomSoftmaxArray(this.w_weightings);

    this.shiftLength = 3; //this.memory.blocks;

    this.s_shiftingVector = new Float64Array(this.shiftLength);
    this.k_keys = new Float64Array(this.memory.blockWidth);
    this.ß_keyStrength = 0;
    this.eraseGate = new Float64Array(this.memory.blocks);
    this.addGate = new Float64Array(this.memory.blocks);
    this.readVector = destinationArray || new Float64Array(this.memory.blocks);

    this.layer = new Layer.Layer(this.memory.blockWidth + this.memory.blocks * 3 + Head.ADDITIONAL_INPUT_VALUES + this.shiftLength);

    this.memory.hiddenLayer.project(this.layer, Layer.Layer.connectionType.ALL_TO_ALL);
    this.layer.project(this.memory.outputLayer, Layer.Layer.connectionType.ALL_TO_ALL);

    this.circulantMatrix = Utils.buildCirculantMatrix(this.memory.blocks);
  }

  private readParams(activation: Synaptic.INumericArray) {

    this.ß_keyStrength = Squash.SOFTPLUS(activation[0]);
    this.g_interpolation = Squash.LOGISTIC(activation[1]);
    this.Y_focus = Math.log(Math.exp(activation[2] + 1)) + 1;//Squash.SOFTPLUS(activation[2]) + 1;

    var startAt = 3;
    for (var k = 0; k < this.k_keys.length; k++) {
      this.k_keys[k] = this.layer.list[k + startAt].activation;
    }

    startAt += this.k_keys.length;
    for (var k = 0; k < this.addGate.length; k++) {
      this.addGate[k] = this.layer.list[k + startAt].derivative;
    }

    startAt += this.addGate.length;
    for (var k = 0; k < this.eraseGate.length; k++) {
      this.eraseGate[k] = Squash.LOGISTIC(this.layer.list[k + startAt].activation);
    }

    startAt += this.eraseGate.length;
    for (var k = 0; k < this.shiftLength; k++) {
      this.s_shiftingVector[k] = this.layer.list[k + startAt].activation;
    }

    var M = this.memory.data;
    
    // focus by content, obtains an array of similarity indexes for each memoryBlock
    for (var i = 0; i < M.length; i++)
      this.wc_focusedWeights[i] = Utils.getCosineSimilarity(M[i], this.k_keys);
    
    // focus by location (interpolation)
    Utils.interpolateArray(this.wc_focusedWeights, this.w_weightings, this.g_interpolation);
    
    // convolutional shift
    this.doShiftings();
     
    // sharpening
    Utils.sharpArray(this.w_weightings, this.wc_focusedWeights, this.Y_focus);
    
    // since ∑ w = 1, we have to softmax the array
    Utils.softMax(this.w_weightings);
    
    /// we got wt!
  }

  doShiftings() {
    Utils.softMax(this.s_shiftingVector);

    Utils.vectorInvertedShifting(this.wc_focusedWeights, this.s_shiftingVector);
  }

  doTimeStep() {
    var activation = this.layer.activate();

    this.readParams(activation);
    
    // reading
    for (var index = 0; index < this.memory.blocks; index++) {
      this.readVector[index] = 0;
      for (var cell = 0; cell < this.memory.blockWidth; cell++) {
        this.readVector[index] += this.memory.data[index][cell] * this.w_weightings[index];
      }
    }
  }
}