import Synaptic = require('./synaptic');

export class Utils {
  
  static transformationMatrixCache: { [index: number] : Float64Array[] } = {};
  
  static softMax<T extends Synaptic.INumericArray>(outputArray: T): T {
    // for all i ∈ array
    // sum = ∑ array[n]^e
    // i = î^e / sum
    // where the result ∑ array[0..n] = 1

    if (!outputArray.length) return outputArray;

    var sum = 0;

    // sum = ∑ array[n]^e
    for (var i = 0; i < outputArray.length; i++) {
      outputArray[i] = Math.exp(outputArray[i]);
      sum += outputArray[i];
    }

    for (var i = 0; i < outputArray.length; i++) outputArray[i] /= sum;

    return outputArray;
  }

  static softMaxDerivative<T extends Synaptic.INumericArray>(outputArray: T): T {
    // http://sysmagazine.com/posts/155235/
    if (!outputArray.length) return outputArray;

    var sum = 0;

    // sum = ∑ array[n]^e
    for (var i = 0; i < outputArray.length; i++) {
      outputArray[i] = Math.exp(outputArray[i]);
      sum += outputArray[i];
    }

    for (var i = 0; i < outputArray.length; i++) {
      var t = outputArray[i] /= sum;
      
      outputArray[i] = t * (1 - t);
    }

    return outputArray;
  }

  static softMaxReinforcement<T extends Synaptic.INumericArray>(array: T, temperature = 1): T {
    // Reinforcement learning

    if (!array.length) return array;

    temperature = temperature || 1;

    var sum = 0;

    // sum = ∑ array[n]^e
    for (var i = 0; i < array.length; i++) {
      array[i] = Math.exp(array[i] / temperature);
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

    var circulantMatrix = Utils.transformationMatrixCache[wg.length] || (Utils.transformationMatrixCache[wg.length] = Utils.buildCirculantMatrix(wg.length));

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
