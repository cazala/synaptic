## Networks (神經網路)

網路基本上是一系列的層。它們有一個輸入層、若干個隱藏層和一個輸出層。網路可以像層一樣投射和連接連接，啟動和傳播。還可以對網路進行最佳化、擴展、導出到JSON、轉換為工作單元或獨立函數，並複製。

```javascript
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

### project

網路可以將連接投射到另一個網路，或者以類似 [Layer](http://github.com/cazala/synaptic#layer) 的方式在兩個其他網路之間建立連接。您必須提供您要連接的網路和 `connectionType`：

```javascript
myNetwork.project(otherNetwork, Layer.connectionType.ALL_TO_ALL);
/*
	myNetwork輸出層中的所有神經元現在都會投影一個連接到其他網路輸入層中的所有神經元。
*/
```

有三個連接類型（`connectionTypes`）:

- `Layer.connectionType.ALL_TO_ALL`: 它將 A 層的每個神經元連接到 B 層的每個神經元
- `Layer.connectionType.ONE_TO_ONE`: 它將來自 A 層的每個神經元連接到 B 層的一個神經元。兩個層必須是相同的大小才能工作。
- `Layer.connectionType.ALL_TO_ELSE`: 僅用於自我連接。
它將一個層的每個神經元連接到同一層的所有其他神經元，除了它本身。
如果此連接類型用於不同層之間的連接，則會產生與 `ALL_TO_ALL` 相同的結果。

如果未指定，則連接類型始終為 `Layer.connectionType.ALL_TO_ALL`。

該方法**投射**返回一個 `LayerConnection` 對象，可以由另一個網路或門接受

### gate

網路可以關閉兩個其他網路或層之間的連接，或者層的自連接。

```javascript
var connection = A.project(B);
C.gate(connection, Layer.gateType.INPUT_GATE); // now C's output layer gates the connection between A's output layer and B's input layer (input gate)
```

有三種門類型（gateType）：
- `Layer.gateType.INPUT_GATE`: 如果網路 C 是網路 A 和 B 之間的門連接，那麼 C 輸出層的所有神經元都將輸入連接到 B 的輸入層。
- `Layer.gateType.OUTPUT_GATE`：如果網路 C 是網路 A 和 B 之間的門控連接，那麼 C 輸出層的所有神經元都來自輸出層的輸出連接。
- `Layer.gateType.ONE_TO_ONE`: 如果網路 C 是網路 A 和 B 之間的門控連接，那麼 C 輸出層的每個神經元都從輸出層到 B 輸入層連接一個連接。使用這種 gateType、一個輸出層，B 的輸入層和輸出層 C 的大小必須相同。

### activate

當一個網路被啟動時，必須提供一個輸入來啟動輸入層，然後所有的隱藏層才被啟動，最後啟動輸出層並返回啟動。

```javascript
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

### propagate
您可以向網路提供目標值和學習速率，並將輸出層中的錯誤反向傳播到所有隱藏層，直到到達輸入層為止。例如，這是你如何訓練一個網路如何解決 XOR：

```javascript
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

### optimize
網路在第一次啟動後會自動進行最佳化，如果您在 console 中啟動網路實例的啟動或傳播方法後，它將如下所示：

```javascript
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

這極大地提高了網路的性能。

### extend
您可以在 [範例](https://github.com/NoobTW/synaptic/blob/master/README_Zh-TW.md#%E7%A4%BA%E7%AF%84%E7%94%A8%E4%BE%8B) 部分看到如何擴展網路。

### toJSON/fromJSON

網路可以儲存為 JSON，然後重新載入：

```javascript
var exported = myNetwork.toJSON();
var imported = Network.fromJSON(exported);
```

`imported` 將是一個新的 `Network` 實例，是一個確切的 `myNetwork` 的複製體

### worker

網路可以轉換成 WebWorker。此功能在 node.js 中不起作用，並且在不是在每個瀏覽器上都支援（它必須支援 Blob）。

```javascript
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

### standalone

網路可以導出到一個單一的 JavaScript 函式。這在你的網路已經被訓練並且你只需要使用它的時候很有用，因為獨立函數只是一個帶有數組和操作的 JavaScript 函數，並不依賴於 Synaptic 或者任何其他的庫。

```javascript
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

### clone

網路可以複製到一個全新的實例，具有相同的連接和跟蹤。

```javascript
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

### neurons

`neurons()` 方法按照啟動順序返回一個包含網路中所有神經元的數組。

### set

方法集（layers）接收一個與 Network 的構造函數具有相同格式的對象，並將它們設置為 Network 的各個層，當擴展 Network 類來創建自己的體系結構時，這是非常有用的。
請參閱 [範例](https://github.com/NoobTW/synaptic/blob/master/README_Zh-TW.md#%E7%A4%BA%E7%AF%84%E7%94%A8%E4%BE%8B) 部分。

```javascript
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

### clear

`clear()` 方法清除網路上下文記憶體，同時保持網路權重不變。

當網路需要使用不應該使用先前啟動的上下文的新數據序列來啟動時，這在 LSTM 中是有用的。