# Neurons

神经元是神经网络的基本单元。它们可以连接在一起，也可以用来连接其他神经元之间的连接。一个神经元可以执行基本上4个操作：项目连接，门连接，激活和传播。

## project

神经元可以将连接投射到另一个神经元（即将神经元A连接到神经元B）。这是如何完成的：

```javascript
var A = new Neuron();
var B = new Neuron();
A.project(B); // A now projects a connection to B
```

神经元也可以自我连接：

`A.project(A);`

该方法项目返回一个连接对象，可以由另一个神经元门接收。

## gate

神经元可以关闭两个神经元之间的连接，或者神经元的自连接。这使您可以创建[二阶神经网络](https://en.wikipedia.org/wiki/Recurrent_neural_network#Second_order_RNN)体系结构。

```javascript
var A = new Neuron();
var B = new Neuron();
var connection = A.project(B);

var C = new Neuron();
C.gate(connection); // now C gates the connection between A and B
```

## activate

当一个神经元激活时，它从它的所有输入连接计算出它的状态，并用它的激活函数压缩它，并返回输出（激活）。您可以将激活作为参数提供（对于输入图层中的神经元很有用，它必须是介于0和1之间的浮点数）。例如：

```javascript
var A = new Neuron();
var B = new Neuron();
A.project(B);

A.activate(0.5); // 0.5
B.activate(); // 0.3244554645
```

## propagate
激活之后，你可以教导神经元本来应该是正确的输出（又名列车）。这是通过反向传播错误来完成的。要使用传播方法，你必须提供一个学习率和一个目标值（在0和1之间浮动）。

例如，当神经元A激活1时，您可以如何训练神经元B激活0：

```javascript
var A = new Neuron();
var B = new Neuron();
A.project(B);

var learningRate = .3;

for(var i = 0; i < 20000; i++)
{
	// when A activates 1
	A.activate(1);
	
	// train B to activate 0
	B.activate();
	B.propagate(learningRate, 0); 
}

// test it
A.activate(1);
B.activate(); // 0.006540565760853365
```

### 压缩功能和偏置

默认情况下，神经元使用 [Logistic Sigmoid](http://en.wikipedia.org/wiki/Logistic_function) 作为挤压/激活功能，以及随机偏差。您可以通过以下方式更改这些属性：

```javascript
var A = new Neuron();
A.squash = Neuron.squash.TANH;
A.bias = 1;
```

有5个内置的压缩功能，但你也可以创建自己的：

- [Neuron.squash.LOGISTIC](http://commons.wikimedia.org/wiki/File:SigmoidFunction.png)
- [Neuron.squash.TANH](http://commons.wikimedia.org/wiki/File:TanhFunction.jpg)
- [Neuron.squash.IDENTITY](http://en.wikipedia.org/wiki/File:Function-x.svg)
- [Neuron.squash.HLIM](http://commons.wikimedia.org/wiki/File:HardLimitFunction.png)
- [Neuron.squash.ReLU](http://i.stack.imgur.com/8CGlM.png)

在[这里](https://wagenaartje.github.io/neataptic/docs/methods/activation/)查阅更多压缩相关的方法