// import

var assert = require('assert'),
  Utils = require('../node-dist/src/architect/NTM').Utils;


// utils

var noRepeat = function(range, avoid) {
  var number = Math.random() * range | 0;
  var used = false;
  for (var i in avoid)
    if (number == avoid[i])
      used = true;
  return used ? noRepeat(range, avoid) : number;
};

var equal = function(prediction, output) {
  for (var i in prediction)
    if (Math.round(prediction[i]) != output[i])
      return false;
  return true;
};

var generateRandomArray = function(size){
    var array = [];
    for (var j = 0; j < size; j++)
        array.push(Math.random() + .5 | 0);
    return array;
}

var sumAll = function(arr){
  var tmp = 0;
  for (var index = 0; index < arr.length; index++) {
    tmp += arr[index];
  }
  return parseFloat(tmp.toFixed(3));
}

// specs


describe("NTM Utils", function() {
  
  it('Test equality arrays', function(){
    var fixedArray = new Float64Array(5);
    
    fixedArray.set([0,0,0,0,1]);
    
    assert.deepEqual(fixedArray, [0,0,0,0,1], 'Float64Array vs array');
  });
  
  it('Softmax array', function(){
    var fixedArray = new Float64Array(5);
    
    fixedArray.set([0,0,0,0,1]);
    
    assert.equal(sumAll(Utils.softMax(fixedArray)), 1, 'Fixed array equals 1');
    
    
    fixedArray.set([0,0,0,0,0]);
    
    assert.equal(sumAll(Utils.softMax(fixedArray)), 1, 'Fixed zero array equals 1');
    
    fixedArray.set([Math.random()* 3,Math.random() * 6,Math.random() * 4,Math.random() * 10,Math.random()]);
    
    assert.equal(sumAll(Utils.softMax(fixedArray)), 1, 'Random positive array equals 1');
    
    fixedArray.set([Math.random()* -9,Math.random() * 6,Math.random() * -4,Math.random() * 10,Math.random()]);
    
    assert.equal(sumAll(Utils.softMax(fixedArray)), 1, 'Random signed array equals 1');
  });
  /*
  it('Generate circulant matrix', function(){
    var fixedArray = new Float64Array(5);
    
    fixedArray.set([0,0,0,0,1]);
    
    assert.equal(Utils.vectorShifting(fixedArray, [0, 1, 0]), [0,0,0,0,1], 'Unchanged shiftings');
  });
  */
  it('Shifting', function(){
    var fixedArray = new Float64Array(5);
    
    fixedArray.set([0,0,0,0,1]);
    Utils.vectorInvertedShifting(fixedArray, [0, 1, 0]);
    console.log('0', fixedArray);
    assert.deepEqual(fixedArray, [0,0,0,0,1], 'Unchanged shiftings');
    
    fixedArray.set([0,0,0,0,1]);
    Utils.vectorInvertedShifting(fixedArray, [0, 0, 0]);
    assert.deepEqual(fixedArray, [0,0,0,0,0], 'All zeros in shift');
    
    fixedArray.set([0,0,0,0,1]);
    Utils.vectorInvertedShifting(fixedArray, [1, 0, 0]);
    console.log('+1', fixedArray);
    assert.deepEqual(fixedArray, [1,0,0,0,0], 'Plus one shift');
    
    fixedArray.set([0,0,0,0,1]);
    Utils.vectorInvertedShifting(fixedArray, [0, 0, 1]);
    console.log('-1', fixedArray);
    assert.deepEqual(fixedArray, [0,0,0,1,0], 'Minus one shift');

    fixedArray.set([1,0,0,0,2]);
    Utils.vectorInvertedShifting(fixedArray, [1, 0, 0]);
    console.log('Valued', fixedArray);
    assert.deepEqual(fixedArray, [2,1,0,0,0], 'Plus one shift, two values');
    
    fixedArray.set([1,1,1,1,1]);
    Utils.vectorInvertedShifting(fixedArray, [1, 1, 1]);
    console.log('Mixing values', fixedArray);
    assert.deepEqual(fixedArray, [3,3,3,3,3], 'Mixing three values');

    fixedArray.set([1,1,1,1,1]);
    Utils.vectorInvertedShifting(fixedArray, [1, 1, 0]);
    console.log('Mixing values', fixedArray);
    assert.deepEqual(fixedArray, [2,2,2,2,2], 'Mixing two values');
    
    fixedArray.set([1,1,1,1,1]);
    Utils.vectorInvertedShifting(fixedArray, [.5, 1, 0.5]);
    console.log('Mixing values', fixedArray);
    assert.deepEqual(fixedArray, [2,2,2,2,2], 'Mixing threeº values');
    
    
    fixedArray.set([1,1,1,1,1]);
    Utils.vectorInvertedShifting(fixedArray, [.5, .5, 1, .5, .5]);
    console.log('Mixing values', fixedArray);
    assert.deepEqual(fixedArray, [3,3,3,3,3], 'Mixing five values');
    
    
    
    
    
    
  });
});
