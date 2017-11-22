## Architect

### Perceptron(感知器)

---

这种架构允许你创建多层感知器，也被称为前馈神经网络。它们由一系列层组成，每个层完全连接到下一层。

![multilayer perceptron](http://www.codeproject.com/KB/dotnet/predictor/network.jpg "Multilayer Perceptron Architecture")

你必须提供至少3层（输入、隐藏和输出），但你可以使用尽可能多的隐藏层。这是一个感知器，在输入层有2个神经元，隐藏层有3个神经元，输出层有1个神经元：

```javascript
var myPerceptron = new Architect.Perceptron(2,3,1);
```

这是一个深层多层感知器，在输入层有2个神经元，4个隐藏层，每层有10个神经元，输出层有1个神经元。

```javascript
var myPerceptron = new Architect.Perceptron(2, 10, 10, 10, 10, 1);
```

### LSTM(长短期记忆网络)

---

[长短期记忆](http://en.wikipedia.org/wiki/Long_short_term_memory)是一种非常适合于从经验中学习来区分、处理和预测时间序列的方法。

![long short-term memory](http://people.idsia.ch/~juergen/lstmcell4.jpg "Long Short-Term Memory Architecture")

要使用这种体系结构，你必须设置至少一个输入层，一个内存块程序集（由四层组成：输入门、存储单元、遗忘门和输出门）和输出层。

```javascript
var myLSTM = new Architect.LSTM(2,6,1);
```

你也可以设置很多层的内存块：

```javascript
var myLSTM = new Architect.LSTM(2,4,4,4,1);
```

这个LSTM网络有3个内存块组件，每个内存块组件都有4个内存单元组成（输入门、存储单元、遗忘门和输出门），请注意，每个块仍然从前一层和第一层获得输入。如果你想要一个更有序的结构，看看 [Neataptic](https://github.com/wagenaartje/neataptic), Neataptic还包括一个在训练期间清除网络的选项，使其更好地用于序列预测。

### Liquid(暂时不知道怎么翻译合适)

---

Liquid 结构允许你创建 Liquid 机器。在这些网络中，神经元是随机连接的。连接的递归性质使时变输入变为网络节点激活的时空模式。

要使用这种体系结构，必须设置输入层的大小、池的大小、输出层的大小、池中随机连接的数量以及连接中的随机门的数量。

```javascript
var input = 2;
var pool = 20;
var output = 1;
var connections = 30;
var gates = 10;

var myLiquidStateMachine = new Architect.Liquid(input, pool, output, connections, gates);
```

### Hopfield

---

[Hopfield](http://en.wikipedia.org/wiki/Hopfield_network) 架构用作内容可寻址存储器。他们被训练来记忆模式，然后当向网络提供新的模式时，它将返回与被训练记住的模式中最相似的模式。

```javascript
var hopfield = new Architect.Hopfield(10) // 为10位模式创建一个网络

// 教网络两种不同的模式
hopfield.learn([
	[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
])

// 将新的模式输入到网络中，它将返回与它被训练来记住的最相似的模式。
hopfield.feed([0,1,0,1,0,1,0,1,1,1]) // [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
hopfield.feed([1,1,1,1,1,0,0,1,0,0]) // [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
```

你可以通过扩展网络类来创建自己的体系结构。可以查看 [示例](http://github.com/cazala/synaptic#examples) 以获得有关这部分的更多信息。