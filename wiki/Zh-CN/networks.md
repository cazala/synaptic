## Networks (神经网络)

网络基本上是一系列的层。它们有一个输入层、若干个隐藏层和一个输出层。网络可以像层一样投射和连接连接，激活和传播。还可以对网络进行优化、扩展、导出到JSON、转换为工作单元或独立函数，并克隆。

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

网络可以将连接投影到另一个网络，或者以类似 [Layer](http://github.com/cazala/synaptic#layer) 的方式在两个其他网络之间建立连接。您必须提供您要连接的网络和 `connectionType`：

```javascript
myNetwork.project(otherNetwork, Layer.connectionType.ALL_TO_ALL); 
/* 	
	myNetwork输出层中的所有神经元现在都会投影一个连接到其他网络输入层中的所有神经元。
*/
```

有三个连接类型（`connectionTypes`）:

- `Layer.connectionType.ALL_TO_ALL`: 它将A层的每个神经元连接到B层的每个神经元
- `Layer.connectionType.ONE_TO_ONE`: 它将来自A层的每个神经元连接到B层的一个神经元。两个层必须是相同的大小才能工作。
- `Layer.connectionType.ALL_TO_ELSE`: 仅用于自我连接。
它将一个层的每个神经元连接到同一层的所有其他神经元，除了它本身。
如果此连接类型用于不同层之间的连接，则会产生与ALL_TO_ALL相同的结果。

如果未指定，则连接类型始终为Layer.connectionType.ALL_TO_ALL。

该方法项目返回一个 LayerConnection 对象，可以由另一个网络或门接受

### gate

网络可以关闭两个其他网络或层之间的连接，或者层的自连接。

```javascript
var connection = A.project(B);
C.gate(connection, Layer.gateType.INPUT_GATE); // now C's output layer gates the connection between A's output layer and B's input layer (input gate)
```

有三种门类型（gateType）：
- `Layer.gateType.INPUT_GATE`: 如果网络C是网络A和B之间的门连接，那么C输出层的所有神经元都将输入连接到B的输入层。
- `Layer.gateType.OUTPUT_GATE`：如果网络C是网络A和B之间的门控连接，那么C输出层的所有神经元都来自输出层的输出连接。
- `Layer.gateType.ONE_TO_ONE`: 如果网络C是网络A和B之间的门控连接，那么C输出层的每个神经元都从输出层到B输入层连接一个连接。使用这种gateYype、一个输出层，B的输入层和输出层C的大小必须相同。

### activate

当一个网络被激活时，必须提供一个输入来激活输入层，然后所有的隐藏层才被激活，最后激活输出层并返回激活。

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
您可以向网络提供目标值和学习速率，并将输出层中的错误反向传播到所有隐藏层，直到到达输入层为止。例如，这是你如何训练一个网络如何解决异或：

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
网络在第一次激活后会自动进行优化，如果您在控制台中激活网络实例的激活或传播方法后，它将如下所示：

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

这极大地提高了网络的性能。

### extend
您可以在“[示例](http://github.com/cazala/synaptic#examples)”部分看到如何扩展网络。

### toJSON/fromJSON

网络可以存储为JSON，然后恢复：

```javascript
var exported = myNetwork.toJSON();
var imported = Network.fromJSON(exported);
```

`imported` 将是一个新的 `Network`实例，是一个确切的 `myNetwork` 的克隆

### worker

网络可以转换成WebWorker。此功能在node.js中不起作用，并且在每个浏览器上都不支持（它必须支持Blob）。

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

网络可以导出到一个单一的 JavaScript 函数。这在你的网络已经被训练并且你只需要使用它的时候很有用，因为独立函数只是一个带有数组和操作的 Javascript 函数，并不依赖于Synaptic或者任何其他的库。

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

网络可以克隆到一个全新的实例，具有相同的连接和跟踪。

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

`neurons()` 方法按照激活顺序返回一个包含网络中所有神经元的数组。

### set

方法集（layers）接收一个与Network的构造函数具有相同格式的对象，并将它们设置为Network的各个层，当扩展Network类来创建自己的体系结构时，这是非常有用的。
请参阅 [示例](http://github.com/cazala/synaptic#examples) 部分。

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

`clear()` 方法清除网络上下文内存，同时保持网络权重不变。

当网络需要使用不应该使用先前激活的上下文的新数据序列来激活时，这在LSTM中是有用的。