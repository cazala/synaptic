import network = require('../network');
import trainer = require('../trainer');
import Layer = require('../layer');
import neuron = require('../neuron');
import Synaptic = require('../synaptic');
import Squash = require('../squash');

export class Utils {
  static softMax<T extends Synaptic.INumericArray>(array: T): T {
    // for all i ∈ array
    // sum = ∑ array[n]^e
    // i = î^e / sum
    // where the result ∑ array[0..n] = 1

    if (!array.length) return array;

    var sum = 0;

    // sum = ∑ array[n]^e
    for (var i = 0; i < array.length; i++) {
      array[i] = Math.exp(array[i]);
      sum += array[i];
    }

    for (var i = 0; i < array.length; i++) array[i] /= sum;

    return array;
  }
  static softMaxSharpen<T extends Synaptic.INumericArray>(array: T, sharpen = 1): T {
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


  static getCosineSimilarity(arrayA: Synaptic.INumericArray, arrayB: Synaptic.INumericArray): number {
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

  static interpolateArray(output_inputA: Synaptic.INumericArray, inputB: Synaptic.INumericArray, g) {
    // 3.3.2 focus by location (7)
    var gInverted = 1 - g;
    for (var i = 0; i < output_inputA.length; i++)
      output_inputA[i] = output_inputA[i] * g + gInverted * inputB[i];
    return output_inputA;
  }
  
  // w_sharpWn
  static sharpArray(output: Synaptic.INumericArray, wn: Synaptic.INumericArray, Y: number) {
    // 3.3.2 (9)
    var sum = 0;

    // ∀ a ∈ wn → a = a^Y
    // sum = ∑ a^Y 

    for (var i = 0; i < wn.length; i++) {
      wn[i] = Math.pow(wn[i], Y);
      sum += wn[i];
    }

    // ∀ a ∈ wn → a = a^Y / sum
    if (sum != 0) {
      for (var i = 0; i < wn.length; i++) output[i] = wn[i] / sum;
    } else {
      var div = 1 / wn.length;
      for (var i = 0; i < wn.length; i++) output[i] = div;
    }
  }
  
  //wn_shift
  static scalarShifting(wg: Synaptic.INumericArray, shiftScalar: number) {
    // w~ 3.3.2 (8)
    var shiftings = new Float64Array(wg.length);
    var wn = new Float64Array(wg.length);

    var intPart = shiftScalar | 0;
    var decimalPart = shiftScalar - intPart;

    shiftings[intPart % shiftings.length] = 1 - decimalPart;
    shiftings[(intPart + 1) % shiftings.length] = decimalPart;


    for (var i = 0; i < wn.length; i++) {
      var acum = 0;
      for (var j = 0; j < wn.length; j++) {
        if ((i - j) < 0)
          acum += wg[j] * shiftings[shiftings.length - Math.abs(i - j)];
        else
          acum += wg[j] * shiftings[(i - j) % shiftings.length];
      }
      wn[i] = acum;
    }

    return wn;
  }

  static normalizeShift(shift: Float64Array) {
    var sum = 0;
    for (var i = 0; i < shift.length; i++) {
      sum += shift[i];
    }
    for (var j = 0; j < shift.length; j++) {
      shift[j] /= sum;
    }
  }

  static vectorInvertedShifting(wg: Float64Array, shiftings: Synaptic.INumericArray) {
    // w~ 3.3.2 (8)

    
    var ret = new Float64Array(wg.length);

    var corrimientoIndex = -((shiftings.length - 1) / 2) | 0;

    var circulantMatrix = Utils.buildCirculantMatrix(wg.length);

    for (var i = 0; i < wg.length; i++) {
      for (var x = 0; x < wg.length; x++) {
        var tmp = 0;

        for (var shift = 0; shift < shiftings.length; shift++) {

          var matRow = i - x + corrimientoIndex + shift;

          while (matRow < 0)
            matRow += wg.length;

          matRow %= wg.length;

          tmp += wg[circulantMatrix[x][matRow]] * shiftings[shift];
        }

        ret[i] = tmp;
      }

    }

    wg.set(ret);
  }

  static initRandomSoftmaxArray(array: Float64Array): void {
    for (var i = 0; i < array.length; i++) {
      array[i] = Math.random();
    }

    Utils.softMax(array);
  }

  static buildCirculantMatrix(length: number, offset: number = 0): Float64Array[] {
    var ret = [];

    for (var i = 0; i < length; i++) {
      var arr = new Float64Array(length);
      ret.push(arr);
      for (var n = 0; n < length; n++) {
        arr[n] = ((i + n) % length);
      }
    }

    return ret;
  }
}



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

  private readParams(activation: Float64Array) {

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