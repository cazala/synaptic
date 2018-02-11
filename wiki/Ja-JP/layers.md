# Layers

通常は単一のニューロンではなく層を使用します。 層は基本的にニューロンの配列であり、できることもニューロンと似ていますが、プログラミングの処理が高速になります。

ニューロンの数を指定するだけで層を作成することができます。

```javascript 
var myLayer = new Layer(5);
```

project
=======

接続する層と`connectionType`を指定すると、層を別の層に投影することができます。

```javascript
var A = new Layer(5);
var B = new Layer(3);
A.project(B, Layer.connectionType.ALL_TO_ALL); // All the neurons in layer A now project a connection to all the neurons in layer B
```

層は自己接続することもできます。

```javascript
A.project(A, Layer.connectionType.ONE_TO_ONE);
```

2種類の`connectionType`があります。

- `Layer.connectionType.ALL_TO_ALL`: 層Aのすべてのニューロンを層Bのすべてのニューロンに接続します。
- `Layer.connectionType.ONE_TO_ONE`: 層Aの各ニューロンと層Bの1つのニューロンを接続します。両方の層が**同じサイズ**でないとダメです。
- `Layer.connectionType.ALL_TO_ELSE`: 自己接続の時のみ有効です。層からのすべてのニューロンを同じ層内の他のすべてのニューロンに接続します。 他の層と接続として使用されると`ALL_TO_ALL`と同じ結果が生成されます。

**注記**：2つの異なる層を接続する場合、connectionTypeのデフォルトは`Layer.connectionType.ALL_TO_ALL`になります。 層を自身に接続するときのデフォルトは`Layer.connectionType.ONE_TO_ONE`です。（たとえば、`myLayer.project(myLayer)`）

**project**は他の層がゲートできる`LayerConnection`オブジェクトを返します。

gate
====

層は2つの他の層間の接続、または層の自己接続をゲートすることができます。

```javascript
var A = new Layer(5);
var B = new Layer(3);
var connection = A.project(B);

var C = new Layer(4);
C.gate(connection, Layer.gateType.INPUT_GATE); // now C gates the connection between A and B (input gate)
```

3種類の`gateType`があります。例えば、ゲートCがゲートAとゲートBとの間の接続をゲートしている場合、

- `Layer.gateType.INPUT_GATE`: Cからの全てのニューロンはBへのすべての入力接続をゲートする。
- `Layer.gateType.OUTPUT_GATE`: CからのすべてのニューロンはAからのすべての出力接続をゲートする。
- `Layer.gateType.ONE_TO_ONE`: Cからの各ニューロンは、AからBへの1つの接続をゲートする。自己接続された層をゲートするのに有用な`gateType`です。A、B、Cの角層が同じサイズでないと使えません。

activate
========

層がアクティブになると、層内のすべてのニューロンが順番にアクティブ化され、出力 配列が返されます。 入力層の場合、アクティブになるニューロンを渡すことも可能です。

```javascript
var A = new Layer(5);
var B = new Layer(3);
A.project(B);

A.activate([1,0,1,0,1]); // [1,0,1,0,1]
B.activate(); // [0.3280457, 0.83243247, 0.5320423]
```

propagate
=========

`activate`の後、エラーをバックプロパゲーティングすると正しいはずだった出力を教えることが可能になります。**propagate**メソッドを使用するには、学習率と目標値（0と1の間のfloat 配列）を指定する必要があります。

例えば、層Aが`[1,0,1,0,1]`を活性化するときに層Bに`[0,0]`を返すように指示する場合：

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

#### スカッシュ関数とバイアス

**set**を使うと層内のすべてのニューロンのsquashing function(スカッシュ関数)とバイアスが設定されます。

```javascript
myLayer.set({
	squash: Neuron.squash.TANH,
	bias: 0
})
```

neurons
=======

`neurons()`は、層内のすべてのニューロンがアクティブ化された順の配列を返します。
