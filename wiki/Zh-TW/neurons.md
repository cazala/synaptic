# Neurons

神經元是神經網路的基本單元。它們可以連接在一起，也可以用來連接其他神經元之間的連接。一個神經元可以執行基本上 4 個操作：項目連接、門連接、啟動和傳播。

## project

神經元可以將連接投射到另一個神經元（即將神經元 A 連接到神經元 B）。這是完成它的方法：

```javascript
var A = new Neuron();
var B = new Neuron();
A.project(B); // A now projects a connection to B
```

神經元也可以自我連接：

`A.project(A);`

該方法項目返回一個連接對象，可以由另一個神經元門接收。

## gate

神經元可以關閉兩個神經元之間的連接，或者神經元的自我連接。這使您可以建立 [二階神經網路](https://en.wikipedia.org/wiki/Recurrent_neural_network#Second_order_RNN)體系結構。

```javascript
var A = new Neuron();
var B = new Neuron();
var connection = A.project(B);

var C = new Neuron();
C.gate(connection); // now C gates the connection between A and B
```

## activate

當一個神經元啟動時，它從它的所有輸入連接計算出它的狀態，並用它的啟動函數壓縮它，並返回輸出（啟動）。您可以將啟動作為參數提供（對於輸入圖層中的神經元很有用，它必須是介於 0 和 1 之間的浮點數）。例如：

```javascript
var A = new Neuron();
var B = new Neuron();
A.project(B);

A.activate(0.5); // 0.5
B.activate(); // 0.3244554645
```

## propagate
啟動之後，你可以教導神經元本來應該是正確的輸出（又名訓練）。這是通過反向傳播錯誤來完成的。要使用傳播方法，你必須提供一個學習率和一個目標值（在 0 和 1 之間浮動）。

例如，當神經元 A 啟動 1 時，您可以如何訓練神經元 B 啟動 0：

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

### 壓縮功能和偏置

默認情況下，神經元使用 [Logistic Sigmoid](http://en.wikipedia.org/wiki/Logistic_function) 作為擠壓/啟動功能，以及隨機偏差。您可以透過以下方式更改這些屬性：

```javascript
var A = new Neuron();
A.squash = Neuron.squash.TANH;
A.bias = 1;
```

有5個內建的壓縮功能，但你也可以建立自己的：

- [Neuron.squash.LOGISTIC](http://commons.wikimedia.org/wiki/File:SigmoidFunction.png)
- [Neuron.squash.TANH](http://commons.wikimedia.org/wiki/File:TanhFunction.jpg)
- [Neuron.squash.IDENTITY](http://en.wikipedia.org/wiki/File:Function-x.svg)
- [Neuron.squash.HLIM](http://commons.wikimedia.org/wiki/File:HardLimitFunction.png)
- [Neuron.squash.ReLU](http://i.stack.imgur.com/8CGlM.png)

在 [這裡](https://wagenaartje.github.io/neataptic/docs/methods/activation/) 查閱更多壓縮相關的方法