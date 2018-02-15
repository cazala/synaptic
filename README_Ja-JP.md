Synaptic [![Build Status](https://travis-ci.org/cazala/synaptic.svg?branch=master)](https://travis-ci.org/cazala/synaptic) [![チャットにご参加下さい!](https://synaptic-slack.now.sh/badge.svg)](https://synaptic-slack.now.sh/)
========

Synapticとは、node.jsとブラウザのためのjavascriptニューラルネットワークライブラリです。一般化されたアーキテクチャフリーなアルゴリズムを利用しているので、あらゆるタイプのfirst order(1次)または[second order(2次)](http://en.wikipedia.org/wiki/Recurrent_neural_network#Second_Order_Recurrent_Neural_Network)ニューラルネットワークアーキテクチャをビルドしてトレーニングすることができます。

このライブラリには、[multilayer perceptron(多層パーセプトロン)](https://ja.wikipedia.org/wiki/%E5%A4%9A%E5%B1%A4%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88%E3%83%AD%E3%83%B3)、[multilayer long-short term memory(多層長短期記憶)](http://en.wikipedia.org/wiki/Long_short_term_memory)ネットワーク（LSTM）、[liquid state machines(液体状態機械)](http://en.wikipedia.org/wiki/Liquid_state_machine)、および[Hopfield(ホップフィールド)](https://ja.wikipedia.org/wiki/%E3%83%9B%E3%83%83%E3%83%97%E3%83%95%E3%82%A3%E3%83%BC%E3%83%AB%E3%83%89%E3%83%BB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF)ネットワークのようないくつかのアーキテクチャが含まれています。 Synapticには、任意のネットワークを訓練できるトレーナーもいます。 XORの解析、Distracted Sequence Recall(ディストラクション・シーケンス・リコール）タスクの実行、[Embedded Reber Grammar(組み込みReber文法)](http://www.willamette.edu/~gorr/classes/cs449/reber.html)テストの実行などのトレーニングテストが含まれています。 さまざまなアーキテクチャのパフォーマンスを簡単にテストして比較することができます。

SynapticはDerek D. Monnerの論文のアルゴリズムを使用しています。

[A generalized LSTM-like training algorithm for second-order recurrent neural networks](http://www.overcomplete.net/papers/nn2012.pdf)

方程式の参照はソースコードでコメントされています。

#### 紹介

ニューラルネットワークを初めてお使いの方は、まず[このガイド](https://github.com/cazala/synaptic/wiki/Neural-Networks-101)を読むことからお勧めします。


ニューラルネットワークにデータを与える方法については、[この記事](https://github.com/cazala/synaptic/wiki/Normalization-101)をご覧ください。

[この記事](http://blog.webkid.io/neural-networks-in-javascript/)もお勧めします。

#### デモ

- [Solve an XOR(XORを解く)](http://caza.la/synaptic/#/xor)
- [Discrete Sequence Recall Task(離散シーケンスリコールタスク)](http://caza.la/synaptic/#/dsr)
- [Learn Image Filters(画像フィルタ)](http://caza.la/synaptic/#/image-filters)
- [Paint an Image(画像をペイント)](http://caza.la/synaptic/#/paint-an-image)
- [Self Organizing Map(自己組織化マップ)](http://caza.la/synaptic/#/self-organizing-map)
- [Read from Wikipedia(Wikipediaから読む)](http://caza.la/synaptic/#/wikipedia)
- [Creating a Simple Neural Network (簡単なニューラルネットワーク作成)(動画)](https://scrimba.com/casts/cast-1980)

デモのソースコードは[こちらのブランチ](https://github.com/cazala/synaptic/tree/gh-pages/scripts)からアクセスしてください。

#### 初めに

- [Neurons(ニューロン)](https://github.com/cazala/synaptic/blob/master/wiki/Ja-JP/neurons.md)
- [Layers(層)](https://github.com/cazala/synaptic/blob/master/wiki/Ja-JP/layers.md)
- [Networks(ネットワーク)](https://github.com/cazala/synaptic/blob/master/wiki/Ja-JP/networks.md)
- [Trainer(トレイナー)](https://github.com/cazala/synaptic/blob/master/wiki/Ja-JP/trainer.md)
- [Architect(アーキテクト)](https://github.com/cazala/synaptic/blob/master/wiki/Ja-JP/architect.md)

[こちらのブランチ](https://github.com/cazala/synaptic/tree/gh-pages)をチェックアウトして試してみてください。

`git checkout gh-pages`

## 概要

### インストール

##### Node

[npm](http://npmjs.org)を利用してインストールしてください。

```cmd
npm install synaptic --save
```

##### ブラウザ

[bower](http://bower.io)を利用してインストールしてください。

```cmd
bower install synaptic
```

または[CDNjs](https://cdnjs.com/)のリンクを貼り付けてください。

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/synaptic/1.1.4/synaptic.js"></script>
```

### 使い方

```javascript
var synaptic = require('synaptic'); // this line is not needed in the browser
var Neuron = synaptic.Neuron,
	Layer = synaptic.Layer,
	Network = synaptic.Network,
	Trainer = synaptic.Trainer,
	Architect = synaptic.Architect;
```

これでネットワークの作成とトレーニングが可能になります。[Architect](https://github.com/cazala/synaptic/blob/master/wiki/Ja-JP/architect.md)を利用して組み込みネットワークを使うのも可能です。

### 例

##### パーセプトロン

簡単な**パーセプトロン**作成方法

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

今度はトレーナーをつくり、パーセプトロンにXORを教えます。

```javascript
var myPerceptron = new Perceptron(2,3,1);
var myTrainer = new Trainer(myPerceptron);

myTrainer.XOR(); // { error: 0.004998819355993572, iterations: 21871, time: 356 }

myPerceptron.activate([0,0]); // 0.0268581547421616
myPerceptron.activate([1,0]); // 0.9829673642853368
myPerceptron.activate([0,1]); // 0.9831714267395621
myPerceptron.activate([1,1]); // 0.02128894618097928
```

##### Long Short-Term Memory(長短期記憶)

簡単な**long short-term memory**(長期短期記憶)ネットワークを作成する方法です。 入力ゲート、忘れゲート、出力ゲート、およびピープホール接続があります。

![long short-term memory](http://people.idsia.ch/~juergen/lstmcell4.jpg)

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

あくまで説明のための例です。 [Architect](https://github.com/cazala/synaptic/wiki/Architect/)には、すでに多層パーセプトロンとマルチ層LSTMネットワークアーキテクチャが含まれています。

## Contribute

**Synaptic**はアルゼンチンのブエノスアイレスで始まったオープンソースプロジェクトです。 開発参加したい方は大歓迎です。
貢献したい場合は、プルリクエストを送ってください。 ただその前にnpm run testとnpm run buildを実行してください。 するとすべてのテスト仕様を実行し、Web配布ファイルが作成されます。

## Support

支持希望の方は是非[magic internet money](https://i.imgur.com/mScSiOo.jpg)でビールを奢ってください！

```
BTC: 16ePagGBbHfm2d6esjMXcUBTNgqpnLWNeK
ETH: 0xa423bfe9db2dc125dd3b56f215e09658491cc556
LTC: LeeemeZj6YL6pkTTtEGHFD6idDxHBF2HXa
XMR: 46WNbmwXpYxiBpkbHjAgjC65cyzAxtaaBQjcGpAZquhBKw2r8NtPQniEgMJcwFMCZzSBrEJtmPsTR54MoGBDbjTi2W1XmgM
```

<3
