Synaptic [![Build Status](https://travis-ci.org/cazala/synaptic.svg?branch=master)](https://travis-ci.org/cazala/synaptic) [![Join the chat at https://synapticjs.slack.com](https://synaptic-slack.now.sh/badge.svg)](https://synaptic-slack.now.sh/)
========

## 重要訊息: [Synaptic 2.x](https://github.com/cazala/synaptic/issues/140) 現在正在討論階段！歡迎參加

Synaptics 是一個為 **node.js** 和 **瀏覽器** 環境打造的 javascript 類神經網路庫，它的通用算法是無架構的，所以基本上可以構建和訓練任何類型的一階甚至 [二階神經網路體系結構](http://en.wikipedia.org/wiki/Recurrent_neural_network#Second_Order_Recurrent_Neural_Network).

這個庫包含一些內建的架構，如[多層感知器](http://en.wikipedia.org/wiki/Multilayer_perceptron)、[多層長短期記憶網路](http://en.wikipedia.org/wiki/Long_short_term_memory)（LSTM）、[液體狀態機](http://en.wikipedia.org/wiki/Liquid_state_machine) 和 [Hopfield](http://en.wikipedia.org/wiki/Hopfield_network) 網路, 和一個能夠訓練任何給定的網路的訓練器, 其中包括
解決異或的內建訓練/測試，完成離散序列記憶任務或[嵌入式　Reber　語法測試](http://www.willamette.edu/~gorr/classes/cs449/reber.html)，因此你可以輕鬆測試和比較不同體系結構的性能。


該庫實現的算法來自 Derek D. Monner 的論文：

[二階循環神經網路的廣義LSTM訓練算法](http://www.overcomplete.net/papers/nn2012.pdf)

參考文獻中的方程已在原始碼中進行了註釋

#### 介紹

> 這裡是 WIKI 的 [繁體中文文件](https://github.com/cazala/synaptic/blob/master/wiki/Zh-TW/home.md)

如果你對神經網路沒有任何了解，你應該先 [閱讀本指南](https://github.com/cazala/synaptic/blob/master/wiki/Zh-TW/neural-networks-101.md)。


如果你想要一個關於如何將數據提供給神經網路的範例，那麼看看 [這篇文章](https://github.com/cazala/synaptic/blob/master/wiki/Zh-TW/normalization-101.md).

你可能還想看看 [這篇文章](http://blog.webkid.io/neural-networks-in-javascript/).

#### 示範用例

- [解決 XOR 運算(Solve an XOR)](http://caza.la/synaptic/#/xor)
- [離散序列記憶任務(Discrete Sequence Recall Task)](http://caza.la/synaptic/#/dsr)
- [學習圖像濾波器(Learn Image Filters)](http://caza.la/synaptic/#/image-filters)
- [畫一幅畫(Paint an Image)](http://caza.la/synaptic/#/paint-an-image)
- [自我組織映射圖(Self Organizing Map)](http://caza.la/synaptic/#/self-organizing-map)
- [閱讀維基百科(Read from Wikipedia)](http://caza.la/synaptic/#/wikipedia)
- [建立簡單的神經網路（影片）Creating a Simple Neural Network (Video)](https://scrimba.com/casts/cast-1980)

這些示範的原始碼可以在 [這個分支](https://github.com/cazala/synaptic/tree/gh-pages/scripts) 中找到.

#### 開始

- [神經元(Neurons)](https://github.com/cazala/synaptic/blob/master/wiki/Zh-TW/neurons.md)
- [層(Layers)](https://github.com/cazala/synaptic/blob/master/wiki/Zh-TW/layers.md)
- [神經網路(Networks)](https://github.com/cazala/synaptic/blob/master/wiki/Zh-TW/networks.md)
- [訓練器(Trainer)](https://github.com/cazala/synaptic/blob/master/wiki/Zh-TW/trainer.md)
- [構造器(Architect)](https://github.com/cazala/synaptic/blob/master/wiki/Zh-TW/architect.md)

要嘗試這些範例，請切換到 [gh-pages](https://github.com/cazala/synaptic/tree/gh-pages) 分支.

`git checkout gh-pages`


## 概述

### 如何安裝

##### 在 node 環境中

你可以使用 [npm](http://npmjs.org) 安裝 synaptic:

```cmd
npm install synaptic --save
```

##### 在瀏覽器環境中

你可以使用 [bower](http://bower.io) 安裝 synaptic:

```cmd
bower install synaptic
```

或者你也可以直接使用 [CDNjs](https://cdnjs.com/) 提供的 CDN 連結

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/synaptic/1.1.4/synaptic.js"></script>
```

### 如何使用

```javascript
var synaptic = require('synaptic'); // 在瀏覽器環境中不需要這行程式碼
var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;
```

現在你可以開始創建神經網路，並訓練它們，或者使用 [構造器](https://github.com/cazala/synaptic/blob/master/wiki/Zh-TW/architect.md) 中內建的神經網路。

### 例子

##### 感知器

如何創建一個簡單的**感知器**:

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

現在你可以透過建立一個訓練器並教它如何解決 XOR 問題來訓練你的神經網路

```javascript
var myPerceptron = new Perceptron(2,3,1);
var myTrainer = new Trainer(myPerceptron);

myTrainer.XOR(); // { error: 0.004998819355993572, iterations: 21871, time: 356 }

myPerceptron.activate([0,0]); // 0.0268581547421616
myPerceptron.activate([1,0]); // 0.9829673642853368
myPerceptron.activate([0,1]); // 0.9831714267395621
myPerceptron.activate([1,1]); // 0.02128894618097928
```

##### 時間遞迴神經網路

下面是如何建立一個由輸入層、隱藏層、輸出層和窺視孔連接組成的**時間遞迴神經網路**（LSTM）

![時間遞迴神經網路](http://people.idsia.ch/~juergen/lstmcell4.jpg)

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

這些是用於說明目的的範例，[構造器](https://github.com/cazala/synaptic/blob/master/wiki/Zh-TW/architect.md) 中已經內建了多層感知器和多層 LSTM 網路架構.

## 貢獻

**Synaptic** 是一個開放原始碼專案，始於阿根廷的布宜諾斯艾利斯。歡迎世界上任何人為此項目的發展作出貢獻.

如果你想發 PR 到本專案，請務必在提交前執行 **npm run test** 和 **npm run build**。這樣將會執行所有測試範例並構建 Web 發佈文件.

## 支持

如果你喜歡這個專案，並且想對本專案提供支持，可以用[神奇的網路貨幣](https://i.imgur.com/mScSiOo.jpg)幫我買一杯啤酒：

```
BTC: 16ePagGBbHfm2d6esjMXcUBTNgqpnLWNeK
ETH: 0xa423bfe9db2dc125dd3b56f215e09658491cc556
XMR: 46WNbmwXpYxiBpkbHjAgjC65cyzAxtaaBQjcGpAZquhBKw2r8NtPQniEgMJcwFMCZzSBrEJtmPsTR54MoGBDbjTi2W1XmgM
```

<3
