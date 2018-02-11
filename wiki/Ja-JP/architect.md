Perceptron
==========

フィードフォワードニューラルネットワークとも呼ばれる、多層パーセプトロンを作成できます。 一連の層で構成され、それぞれが完全に次の層に接続されています。

![multilayer perceptron](http://www.codeproject.com/KB/dotnet/predictor/network.jpg "Multilayer Perceptron Architecture")

最低でも3つの層（入力、非表示、出力）を提供する必要がありますが、必要な数だけ隠し層を使用することができます。 例えば、入力層に2つのニューロン、隠れ層に3つのニューロン、出力層に1つのニューロンを持つ`Perceptron`（パーセプトロン）を設定する場合：

```javascript
var myPerceptron = new Architect.Perceptron(2,3,1);
```

入力層に2つのニューロン、それぞれ10のニューロンを持つ4つの隠れ層、出力層に1つのニューロンを持つ深層マルチレイヤパーセプトロンの設定ですとこうなります。

```javascript
var myPerceptron = new Architect.Perceptron(2, 10, 10, 10, 10, 1);
```
LSTM
====

[long short-term memory(長い短期記憶)](http://en.wikipedia.org/wiki/Long_short_term_memory)は、イベント間で非常に長いタイムラグがある時系列を分類、処理、予測するのに適したアーキテクチャです。

![long short-term memory](http://people.idsia.ch/~juergen/lstmcell4.jpg "Long Short-Term Memory Architecture")

少なくとも1つの入力層、1つのメモリブロックアセンブリ、および出力層を設定する必要があります。 ブロックアセンブリは入力ゲート、メモリセル、忘却ゲートおよび出力ゲートの4つの層があります。

```javascript
var myLSTM = new Architect.LSTM(2,6,1);
```

メモリブロックの多くの層を設定することもできます。

```javascript
var myLSTM = new Architect.LSTM(2,4,4,4,1);
```

この例のLSTMネットワークには3つのメモリブロックアセンブリがあります。 各アセンブリには4つのメモリセル、入力ゲート、メモリセル、忘却ゲートおよび出力ゲートがあります。 各ブロックは依然として前の層と最初の層から入力を受け取ります。 Neatapticは、よりシーケンシャルなアーキテクチャとトレーニング中にネットワークをクリアするオプションを備えており、シーケンス予測の方がより優れています。

Liquid
======


`Liquid`アーキテクチャでは[Liquid State Machine(液体状態機械)](http://en.wikipedia.org/wiki/Liquid_state_machine)を作成できます。ニューロンは互いにランダムに接続されています。 時変入力は`Activation`の時空間パターンに変わります。

入力層のサイズ、プールのサイズ、出力レイヤのサイズ、プール内のランダムな接続の数、および接続の中のランダムなゲートの数を入力する必要があります。

```javascript
var input = 2;
var pool = 20;
var output = 1;
var connections = 30;
var gates = 10;

var myLiquidStateMachine = new Architect.Liquid(input, pool, output, connections, gates);
```

Hopfield
========

[Hopfield(ホップフィールド)](https://ja.wikipedia.org/wiki/%E3%83%9B%E3%83%83%E3%83%97%E3%83%95%E3%82%A3%E3%83%BC%E3%83%AB%E3%83%89%E3%83%BB%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF)アーキテクチャはコンテンツアドレス可能なメモリに使用されます。 最初にパターンを記憶するように訓練されています。 新しいパターンを取得すると覚えて訓練されたパターンから最も類似したパターンを返します。

```javascript
var hopfield = new Architect.Hopfield(10) // create a network for 10-bit patterns

// teach the network two different patterns
hopfield.learn([
	[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
])

// feed new patterns to the network and it will return the most similar to the ones it was trained to remember
hopfield.feed([0,1,0,1,0,1,0,1,1,1]) // [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
hopfield.feed([1,1,1,1,1,0,0,1,0,0]) // [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
```

`Network`クラスを拡張することで独自のアーキテクチャを作成できます。[Examples](http://github.com/cazala/synaptic#examples)のセクションを参照してください。
