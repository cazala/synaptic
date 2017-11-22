## Layers (层)

通常你不会使用单个神经元，而是使用层代替。一层基本上是一组神经元，它们可以做几乎和神经元一样的工作，但它使编程过程更快。

要创建一个层，只需指定它的大小（该层中的神经元数量）：

```javascript
var myLayer = new Layer(5);
```
### project

一个层可以投射到另一层的连接。你必须提供层要连接和连接类型：

```javascript
var A = new Layer(5);
var B = new Layer(3);
A.project(B, Layer.connectionType.ALL_TO_ALL); // A层中的所有神经元现在都在与B层中的所有神经元建立连接。
```

层也可以自我连接：

```javascript
A.project(A, Layer.connectionType.ONE_TO_ONE);
```

这里有三种连接类型(`connectionType`)：

- `Layer.connectionType.ALL_TO_ALL`: 它把每一个神经元从A层连接到B层的每个神经元。
- `Layer.connectionType.ONE_TO_ONE`: 它把每一个神经元从A层连接到B层的一个神经元中。每层的大小必须相同
- `Layer.connectionType.ALL_TO_ELSE`: 仅在自我连接中有用。它把每个神经元从一层神经元连接到同一层的所有其他神经元，除了自身。如果客户端是用于不同层之间的连接，它产生相同的结果 `ALL_TO_ALL`。

**NOTE**: 当连接两个不同层时，连接类型默认是 `Layer.connectionType.ALL_TO_ALL`，当一个层连接到它自己时(如：`myLayer.project(myLayer`)，连接类型默认是`Layer.connectionType.ONE_TO_ONE` 

该方法返回一个 `layerconnection` 项目对象，可由另一层门接受

### gate

一个层可以屏蔽两个其他层之间的连接，或者层的自连接。

```javascript
var A = new Layer(5);
var B = new Layer(3);
var connection = A.project(B);

var C = new Layer(4);
C.gate(connection, Layer.gateType.INPUT_GATE); // now C gates the connection between A and B (input gate)
```

这里有三种门类型（`gateType's`）:

- `Layer.gateType.INPUT_GATE`: 如果C层是A层和B层之间的门控连接，那么C门的所有神经元都是B的输入连接。
- `Layer.gateType.OUTPUT_GATE`: 如果C层是A层和B层之间的门控连接，那么C门的所有神经元都来自A的输出连接。
- `Layer.gateType.ONE_TO_ONE`: 如果C层是A层和B层之间的门控连接，那么C神经元的每个神经元都有一个从A到B的连接，这对于门控自连接层是有用的。使用这种gateYype，A，B和C的大小必须相同

### activate

当一个层激活时，它会依次激活该层中的所有神经元，并返回一个具有输出的数组。它接受一系列的激活作为参数（用于输入层）：

```javascript
var A = new Layer(5);
var B = new Layer(3);
A.project(B);

A.activate([1,0,1,0,1]); // [1,0,1,0,1]
B.activate(); // [0.3280457, 0.83243247, 0.5320423]
```

### propagate(传播)

在激活后，你可以教给这个层正确的输出应该是什么（又名：训练），这是由反向传播误差完成的。要使用传播方法，必须提供学习率和目标值（浮动数组在0到1之间）。

例如，如果我想在A层激活[1,0,1,0,1]时训练B层输出[0,0]：

```javascript
var A = new Layer(5);
var B = new Layer(2);
A.project(B);

var learningRate = .3;

for (var i = 0; i < 20000; i++)
{
	// when A activates [1, 0, 1, 0, 1]
	A.activate([1,0,1,0,1]);

	// train B to activate [0,0]
	B.activate();
	B.propagate(learningRate, [0,0]);
}

// test it
A.activate([1,0,1,0,1]);
B.activate(); // [0.004606949693864496, 0.004606763721459169]
```

压缩函数和偏置

您可以使用方法集来设置图层中所有神经元的压缩函数和偏差

```javascript
myLayer.set({
	squash: Neuron.squash.TANH,
	bias: 0
})
```

### neurons （神经元）

neurons() 方法以激活顺序返回层中所有神经元的数组。