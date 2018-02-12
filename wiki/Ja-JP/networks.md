# Networks

基本的にネットワークはは[Layer(層)](https://github.com/cazala/synaptic/blob/master/wiki/Ja-JP/layers.md)の配列です。中には入力層、幾つかの隠れた層と出力層があります。ネットワークは、層のように接続を投影してゲートすることができます。ネットワークは最適化、拡張、JSONにエクスポート、`worker`またはスタンドアロンの関数に変換することができます。クローンすることもできます。

```js
var inputLayer = new Layer(4);
var hiddenLayer = new Layer(6);
var outputLayer = new Layer(2);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var myNetwork = new Network({
	input: inputLayer,
	hidden: [hiddenLayer],
	output: outputLayer
});
```
project
=======


ネットワークは層と同じよう接続を投影したり、2つの他のネットワーク間の接続をゲートすることができます。 接続するネットワークと`connectionType`を指定する必要があります。

```js
myNetwork.project(otherNetwork, Layer.connectionType.ALL_TO_ALL); 
/* 	
	All the neurons in myNetwork's output layer now project a connection
	to all the neurons in otherNetwork's input layer.
*/
```

`connectionType`には2種類があります。

- `Layer.connectionType.ALL_TO_ALL`: 層Aのすべてのニューロンと、層Bのすべてのニューロンを接続します。
- `Layer.connectionType.ONE_TO_ONE`: 層Aからのすべてのニューロンは層Bの1つのニューロンに接続します。両方の層は**同じサイズ**でなければなりません。
- `Layer.connectionType.ALL_TO_ELSE`: 自己接続でのみ有効です。 層からのすべてのニューロンを同じ層内の他のすべてのニューロンに接続します。異なる層間の接続で使用される場合は、`ALL_TO_ALL`と同じ結果が得られます。

`connectionType`を指定しないと`Layer.connectionType.ALL_TO_ALL`になります。

**project**は別のネットワークまたは層にゲートできる`LayerConnection`オブジェクトを返します。

gate
====

ネットワークは、2つの他のネットワーク、層、または層の自己接続間の接続をゲートできます。

```js
var connection = A.project(B);
C.gate(connection, Layer.gateType.INPUT_GATE); // now C's output layer gates the connection between A's output layer and B's input layer (input gate)
```

3種類の`gateType`があります。例えば、ネットワークCがネットワークAとネットワークBとの間の接続をゲートしている場合、

- `Layer.gateType.INPUT_GATE`: Cからの全てのニューロンはBへのすべての入力接続をゲートする。
- `Layer.gateType.OUTPUT_GATE`: CからのすべてのニューロンはAからのすべての出力接続をゲートする。
- `Layer.gateType.ONE_TO_ONE`: Cからの各ニューロンは、AからBへの1つの接続をゲートする。自己接続された層をゲートするのに有用な`gateType`です。A、B、Cの角層が同じサイズでないと使えません。

activate
========

ネットワークがアクティブ化されると入力層をアクティブにするための入力を提供します。隠れた層が順番に有効化されます。 最後に出力層がアクティブ化され、そのアクティブ化が戻されます。

```js
var inputLayer = new Layer(4);
var hiddenLayer = new Layer(6);
var outputLayer = new Layer(2);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var myNetwork = new Network({
	input: inputLayer,
	hidden: [hiddenLayer],
	output: outputLayer
});

myNetwork.activate([1,0,1,0]); // [0.5200553602396137, 0.4792707231811006]
```

propagate
=========

目標値と学習率をネットワークに提供すると、入力層に到着するまで出力層からすべての隠れ層に逆の順序でエラーをバックプロパゲージします。 たとえば、ネットワークを使ってXORを解決する場合：

```js
// create the network
var inputLayer = new Layer(2);
var hiddenLayer = new Layer(3);
var outputLayer = new Layer(1);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var myNetwork = new Network({
	input: inputLayer,
	hidden: [hiddenLayer],
	output: outputLayer
});

// train the network
var learningRate = .3;
for (var i = 0; i < 20000; i++)
{
	// 0,0 => 0
	myNetwork.activate([0,0]);
	myNetwork.propagate(learningRate, [0]);

	// 0,1 => 1
	myNetwork.activate([0,1]);
	myNetwork.propagate(learningRate, [1]);

	// 1,0 => 1
	myNetwork.activate([1,0]);
	myNetwork.propagate(learningRate, [1]);

	// 1,1 => 0
	myNetwork.activate([1,1]);
	myNetwork.propagate(learningRate, [0]);
}


// test the network
myNetwork.activate([0,0]); // [0.015020775950893527]
myNetwork.activate([0,1]); // [0.9815816381088985]
myNetwork.activate([1,0]); // [0.9871822457132193]
myNetwork.activate([1,1]); // [0.012950087641929467]
```

optimize
========

ネットワークがアクティブになると自動的に最適化されます。 アアクティブ化された後にコンソールに`activate`または`propagate`を出力すると次のようになります。

```js
function (input){
F[1] = input[0];
 F[2] = input[1];
 F[3] = input[2];
 F[4] = input[3];
 F[6] = F[7];
 F[7] = F[8];
 F[7] += F[1] * F[9];
 F[7] += F[2] * F[10];
 F[7] += F[3] * F[11];
 F[7] += F[4] * F[12];
 F[5] = (1 / (1 + Math.exp(-F[7])));
 F[13] = F[5] * (1 - F[5]);
 ...
 ```

この最適化により、ネットワークのパフォーマンスが大幅に向上します。

extend
======

ネットワークの拡張方法については[Examples](http://github.com/cazala/synaptic#examples) セクションを参照してください。

toJSON/fromJSON
===============

ネットワークはJSONとして保存してから元に戻すことができます。

```js
var exported = myNetwork.toJSON();
var imported = Network.fromJSON(exported);
```

`imported` will be a new instance of `Network` that is an exact clone of `myNetwork`

worker
======

ネットワークはWebWorkerに変換できます。 この機能はnode.jsでは動作せず、すべてのブラウザでサポートされていません。 Blobをサポートしている必要があります。

```js
// training set
var learningRate = .3;
var trainingSet = [
	{
		input: [0,0],
		output: [0]
	},
	{
		input: [0,1],
		output: [1]
	},
	{
		input: [1,0],
		output: [1]
	},
	{
		input: [1,1],
		output: [0]
	},
];

// create a network
var inputLayer = new Layer(2);
var hiddenLayer = new Layer(3);
var outputLayer = new Layer(1);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var myNetwork = new Network({
	input: inputLayer,
	hidden: [hiddenLayer],
	output: outputLayer
});

// create a worker
var myWorker = myNetwork.worker();

// activate the network
function activateWorker(input)
{
	myWorker.postMessage({ 
		action: "activate",
		input: input,
		memoryBuffer: myNetwork.optimized.memory
	}, [myNetwork.optimized.memory.buffer]);
}

// backpropagate the network
function propagateWorker(target){
	myWorker.postMessage({ 
		action: "propagate",
		target: target,
		rate: learningRate,
		memoryBuffer: myNetwork.optimized.memory
	}, [myNetwork.optimized.memory.buffer]);
}

// train the worker
myWorker.onmessage = function(e){
	// give control of the memory back to the network - this is mandatory!
	myNetwork.optimized.ownership(e.data.memoryBuffer);

	if (e.data.action == "propagate")
	{
		if (index >= 4)
		{
			index = 0;
			iterations++;
			if (iterations % 100 == 0)
			{
				var output00 = myNetwork.activate([0,0]);
				var output01 = myNetwork.activate([0,1]);
				var output10 = myNetwork.activate([1,0]);
				var output11 = myNetwork.activate([1,1]);

				console.log("0,0 => ", output00);
				console.log("0,1 => ", output01);
				console.log("1,0 => ", output10);
				console.log("1,1 => ", output11, "\n");
			}
		}

		activateWorker(trainingSet[index].input);
	}

	if (e.data.action == "activate")
	{
		propagateWorker(trainingSet[index].output);	
		index++;
	}
}

// kick it
var index = 0;
var iterations = 0;
activateWorker(trainingSet[index].input);
```

standalone
==========

ネットワークは単一のjavascript関数にエクスポートすることができます。 ネットワークがすでに訓練されている場合に便利な機能です。関数の中には必要な操作がすべて含まれています。 シナプスやその他のライブラリには依存しません。

```js
var inputLayer = new Layer(4);
var hiddenLayer = new Layer(6);
var outputLayer = new Layer(2);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var myNetwork = new Network({
	input: inputLayer,
	hidden: [hiddenLayer],
	output: outputLayer
});

var standalone = myNetwork.standalone();

myNetwork.activate([1,0,1,0]); 	// [0.5466397925108878, 0.5121246668637663]
standalone([1,0,1,0]);	 // [0.5466397925108878, 0.5121246668637663]
```

clone
=====

完全に新しいインスタンスにネットワークをクローンすることができます。 クローンされたネットワークと同じ接続とトレースを持っています。

```js
var inputLayer = new Layer(4);
var hiddenLayer = new Layer(6);
var outputLayer = new Layer(2);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var myNetwork = new Network({
	input: inputLayer,
	hidden: [hiddenLayer],
	output: outputLayer
});

var clone = myNetwork.clone();

myNetwork.activate([1,0,1,0]); 	// [0.5466397925108878, 0.5121246668637663]
clone.activate([1,0,1,0]);	 // [0.5466397925108878, 0.5121246668637663]
```

neurons
=======

`neurons()`はネットワーク内のすべてのニューロンをアクティブ化順に並べた配列を返します。

set
===

`set(layers)`は`Network`コンストラクタと同じ形式の層を持つオブジェクトを受け取ります。 ネットワークの層として設定します。 独自のアーキテクチャを作成するために`Network`クラスを拡張するときに便利です。[Examples](http://github.com/cazala/synaptic#examples)のセクションを参照してください。

```js
var inputLayer = new Layer(4);
var hiddenLayer = new Layer(6);
var outputLayer = new Layer(2);

inputLayer.project(hiddenLayer);
hiddenLayer.project(outputLayer);

var myNetwork = new Network();

myNetwork.set({
	input: inputLayer,
	hidden: [hiddenLayer],
	output: outputLayer
});
```

clear
=====

`clear()`はネットワークのコンテキストメモリをクリアします。ウェイトは変更しません。

前回のアクティベーションのコンテキストを使用しない、新しいデータシーケンスでLSTMネットワークをアクティブ化するときに便利です。
