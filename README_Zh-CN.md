Synaptic [![Build Status](https://travis-ci.org/cazala/synaptic.svg?branch=master)](https://travis-ci.org/cazala/synaptic) [![Join the chat at https://synapticjs.slack.com](https://synaptic-slack-ugiqacqvmd.now.sh/badge.svg)](https://synaptic-slack-ugiqacqvmd.now.sh/)
========

## 重要信息: [Synaptic 2.x](https://github.com/cazala/synaptic/issues/140) 现在正在讨论阶段！欢迎参加

Synaptics 是一个为 **node.js** 和 **浏览器** 环境打造的 javascript 神经网络库, 它的通用算法是无架构的，所以基本上可以构建和训练任何类型的一阶甚至 [二阶神经网络体系结构](http://en.wikipedia.org/wiki/Recurrent_neural_network#Second_Order_Recurrent_Neural_Network).

这个库包含一些内置的架构，如[多层感知器](http://en.wikipedia.org/wiki/Multilayer_perceptron), [多层长短期记忆网络](http://en.wikipedia.org/wiki/Long_short_term_memory) (LSTM), [液体状态机](http://en.wikipedia.org/wiki/Liquid_state_machine) 和 [Hopfield](http://en.wikipedia.org/wiki/Hopfield_network) 网络, 和一个能够训练任何给定的网络的训练器, 其中包括
解决异或的内置训练/测试，完成离散序列记忆任务或[嵌入式Reber语法测试](http://www.willamette.edu/~gorr/classes/cs449/reber.html)，因此你可以轻松测试和比较不同体系结构的性能。


该库实现的算法来自Derek D. Monner的论文：

[二阶循环神经网络的广义LSTM训练算法](http://www.overcomplete.net/papers/nn2012.pdf)


参考文献中的方程已在源代码中进行了注释

#### 介绍

> 这里是 WIKI 的 [中文文档](https://github.com/cazala/synaptic/blob/master/wiki/Zh-CN/home.md)

如果你对神经网络没有任何了解，你应该首先 [阅读本指南](https://github.com/cazala/synaptic/blob/master/wiki/Zh-CN/neural-networks-101.md)。


如果你想要一个关于如何将数据提供给神经网络的实例，那么看看 [这篇文章](https://github.com/cazala/synaptic/blob/master/wiki/Zh-CN/normalization-101.md).

你可能还想看看 [这篇文章](http://blog.webkid.io/neural-networks-in-javascript/).

#### 演示用例

- [解决异或(Solve an XOR)](http://caza.la/synaptic/#/xor)
- [离散序列记忆任务(Discrete Sequence Recall Task)](http://caza.la/synaptic/#/dsr)
- [学习图像滤波器(Learn Image Filters)](http://caza.la/synaptic/#/image-filters)
- [画一幅画(Paint an Image)](http://caza.la/synaptic/#/paint-an-image)
- [自组织映射(Self Organizing Map)](http://caza.la/synaptic/#/self-organizing-map)
- [读维基百科(Read from Wikipedia)](http://caza.la/synaptic/#/wikipedia)
- [创建简单的神经网络（视频）Creating a Simple Neural Network (Video)
](https://scrimba.com/casts/cast-1980)

这些演示的源代码可以在[这个分支](https://github.com/cazala/synaptic/tree/gh-pages/scripts)中找到.

#### 起步

- [神经元(Neurons)](https://github.com/cazala/synaptic/blob/master/wiki/Zh-CN/neurons.md)
- [层(Layers)](https://github.com/cazala/synaptic/blob/master/wiki/Zh-CN/layers.md)
- [神经网络(Networks)](https://github.com/cazala/synaptic/blob/master/wiki/Zh-CN/networks.md)
- [训练器(Trainer)](https://github.com/cazala/synaptic/blob/master/wiki/Zh-CN/trainer.md)
- [构造器(Architect)](https://github.com/cazala/synaptic/blob/master/wiki/Zh-CN/architect.md)

要尝试这些示例, 请切换到 [gh-pages](https://github.com/cazala/synaptic/tree/gh-pages) 分支.

`git checkout gh-pages`


## 概述

### 如何安装

##### 在 node 环境中

你可以使用 [npm](http://npmjs.org) 安装 synaptic:

```cmd
npm install synaptic --save
```

##### 在浏览器环境中

你可以使用 [bower](http://bower.io) 安装 synaptic:

```cmd
bower install synaptic
```

或者你也可以简单的使用 [CDNjs](https://cdnjs.com/) 提供的 CDN 链接

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/synaptic/1.1.4/synaptic.js"></script>
```

### 如何使用

```javascript
var synaptic = require('synaptic'); // 在浏览器环境中不需要本行代码
var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;
```

现在你可以开始创建神经网络，并训练它们，或者使用 [构造器](https://github.com/cazala/synaptic/blob/master/wiki/Zh-CN/architect.md) 中内置的神经网络.

### 例子

##### 感知器

如何创建一个简单的**感知器**:

![perceptron](http://www.codeproject.com/KB/dotnet/predictor/network.jpg).

```javascript
function Perceptron(input, hidden, output)
{
	// create the layers
	var inputLayer = new Layer(input);
	var hiddenLayer = new Layer(hidden);
	var outputLayer = new Layer(output);

	// connect the layers
	inputLayer.project(hiddenLayer);
	hiddenLayer.project(outputLayer);

	// set the layers
	this.set({
		input: inputLayer,
		hidden: [hiddenLayer],
		output: outputLayer
	});
}

// extend the prototype chain
Perceptron.prototype = new Network();
Perceptron.prototype.constructor = Perceptron;
```

现在你可以通过创建一个训练器并教它如何解决异或问题来训练你的神经网络

```javascript
var myPerceptron = new Perceptron(2,3,1);
var myTrainer = new Trainer(myPerceptron);

myTrainer.XOR(); // { error: 0.004998819355993572, iterations: 21871, time: 356 }

myPerceptron.activate([0,0]); // 0.0268581547421616
myPerceptron.activate([1,0]); // 0.9829673642853368
myPerceptron.activate([0,1]); // 0.9831714267395621
myPerceptron.activate([1,1]); // 0.02128894618097928
```

##### 时间递归神经网络

下面是如何创建一个由输入层、隐藏层、输出层和窥视孔连接组成的 **时间递归神经网络**

![时间递归神经网络](http://people.idsia.ch/~juergen/lstmcell4.jpg)

```javascript
function LSTM(input, blocks, output)
{
	// create the layers
	var inputLayer = new Layer(input);
	var inputGate = new Layer(blocks);
	var forgetGate = new Layer(blocks);
	var memoryCell = new Layer(blocks);
	var outputGate = new Layer(blocks);
	var outputLayer = new Layer(output);

	// connections from input layer
	var input = inputLayer.project(memoryCell);
	inputLayer.project(inputGate);
	inputLayer.project(forgetGate);
	inputLayer.project(outputGate);

	// connections from memory cell
	var output = memoryCell.project(outputLayer);

	// self-connection
	var self = memoryCell.project(memoryCell);

	// peepholes
	memoryCell.project(inputGate);
	memoryCell.project(forgetGate);
	memoryCell.project(outputGate);

	// gates
	inputGate.gate(input, Layer.gateType.INPUT);
	forgetGate.gate(self, Layer.gateType.ONE_TO_ONE);
	outputGate.gate(output, Layer.gateType.OUTPUT);

	// input to output direct connection
	inputLayer.project(outputLayer);

	// set the layers of the neural network
	this.set({
		input: inputLayer,
		hidden: [inputGate, forgetGate, memoryCell, outputGate],
		output: outputLayer
	});
}

// extend the prototype chain
LSTM.prototype = new Network();
LSTM.prototype.constructor = LSTM;
```

这些是用于说明目的的示例，[构造器](https://github.com/cazala/synaptic/blob/master/wiki/Zh-CN/architect.md) 中已经内置了多层感知器和多层LSTM网络架构.

## 贡献

**Synaptic** 是一个开源项目，始于阿根廷，布宜诺斯艾利斯。欢迎世界上任何人为项目的发展作出贡献.

如果你想推送合并请求到本项目，请务必在提交前运行 **npm run test** 和 **npm run build**. 这样将会运行所有测试用例并构建Web发布文件.

## 支持

如果你喜欢这个项目，并且想对本项目提供支持，可以给我买一杯[神奇的互联网啤酒](https://i.imgur.com/mScSiOo.jpg):

```
BTC: 16ePagGBbHfm2d6esjMXcUBTNgqpnLWNeK
ETH: 0xa423bfe9db2dc125dd3b56f215e09658491cc556
XMR: 46WNbmwXpYxiBpkbHjAgjC65cyzAxtaaBQjcGpAZquhBKw2r8NtPQniEgMJcwFMCZzSBrEJtmPsTR54MoGBDbjTi2W1XmgM
```

<3
