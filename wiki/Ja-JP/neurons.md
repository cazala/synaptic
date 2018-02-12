# Neurons

ニューロンはニューラルネットワークの基本単位です。ニューロンは一緒に接続することや他のニューロン間の接続をゲートすることができます。

ニューロンはプロジェクト接続、ゲート接続、アクティブ化と伝播の4つの操作をします。

project
=======

ニューロンは別のニューロンへの接続を投影することができます。
例えば、

```javascript
var A = new Neuron();
var B = new Neuron();
A.project(B); // A now projects a connection to B
```

ニューロンは自己接続することもできます。

`A.project(A);`

**project**というメソッドはConnectionオブジェクトを返します。Connectionは別のニューロンにゲートされることができます。

gate
====

ニューロンは、2つのニューロン間の接続、または、自己接続をゲートすることができます。これにより、[second order neural network(2次ニューラルネットワーク)](https://en.wikipedia.org/wiki/Recurrent_neural_network#Second_order_RNN)アーキテクチャを作成することができます。

```javascript
var A = new Neuron();
var B = new Neuron();
var connection = A.project(B);

var C = new Neuron();
C.gate(connection); // now C gates the connection between A and B
```


activate
========

ニューロンが活動化するとすべての入力接続から状態を計算します。 状態は活性化関数を使用して押しつ部して出力(activation)を返します。
activationは0と1の間のfloatなので、パラメータとして指定できます。
例えば、

```javascript
var A = new Neuron();
var B = new Neuron();
A.project(B);

A.activate(0.5); // 0.5
B.activate(); // 0.3244554645
```



