## Architect

### Perceptron（感知器）

---

這種架構允許你建立多層感知器，也被稱為前饋神經網路（feed-forward neural networks）。它們由一系列層組成，每個層完全連接到下一層。

![multilayer perceptron](http://www.codeproject.com/KB/dotnet/predictor/network.jpg "Multilayer Perceptron Architecture")

你必須提供至少 3 層（輸入、隱藏和輸出），但你可以使用儘可能多的隱藏層。這是一個感知器，在輸入層有 2 個神經元，隱藏層有 3 個神經元，輸出層有 1 個神經元：

```javascript
var myPerceptron = new Architect.Perceptron(2,3,1);
```

這是一個深層多層感知器，在輸入層有 2 個神經元，4 個隱藏層，每層有 10 個神經元，輸出層有 1 個神經元。

```javascript
var myPerceptron = new Architect.Perceptron(2, 10, 10, 10, 10, 1);
```

### LSTM（長短期記憶網路）

---

[長短期記憶](http://en.wikipedia.org/wiki/Long_short_term_memory) 是一種非常適合於從經驗中學習來區分、處理和預測時間序列的方法。

![long short-term memory](http://people.idsia.ch/~juergen/lstmcell4.jpg "Long Short-Term Memory Architecture")

要使用這種體系結構，你必須設置至少一個輸入層，一個記憶體塊程序集（由四層組成：輸入門、存儲單元、遺忘門和輸出門）和輸出層。

```javascript
var myLSTM = new Architect.LSTM(2,6,1);
```

你也可以設置很多層的記憶體塊：

```javascript
var myLSTM = new Architect.LSTM(2,4,4,4,1);
```

這個LSTM網路有 3 個記憶體塊組件，每個記憶體塊組件都有 4 個記憶體單元組成（輸入門、存儲單元、遺忘門和輸出門），請注意，每個塊仍然從前一層和第一層獲得輸入。如果你想要一個更有序的結構，看看 [Neataptic](https://github.com/wagenaartje/neataptic), Neataptic 還包括一個在訓練期間清除網路的選項，使其更好地用於序列預測。

### Liquid

---

Liquid 結構允許你創建 Liquid 機器。在這些網路中，神經元是隨機連接的。連接的遞迴性質使時變輸入變為網路節點啟動的時空模式。

要使用這種體系結構，必須設置輸入層的大小、池的大小、輸出層的大小、池中隨機連接的數量以及連接中的隨機門的數量。

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

[Hopfield](http://en.wikipedia.org/wiki/Hopfield_network) 架構用作內容可定址存儲器。他們被訓練來記憶模式，然後當向網路提供新的模式時，它將返回與被訓練記住的模式中最相似的模式。

```javascript
var hopfield = new Architect.Hopfield(10) // 為10位模式創建一個網路

// 教網路兩種不同的模式
hopfield.learn([
	[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
])

// 將新的模式輸入到網路中，它將返回與它被訓練來記住的最相似的模式。
hopfield.feed([0,1,0,1,0,1,0,1,1,1]) // [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
hopfield.feed([1,1,1,1,1,0,0,1,0,0]) // [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
```

你可以透過擴展網路類來建立自己的體系結構。可以查看 [範例](http://github.com/cazala/synaptic#examples) 以獲得有關這部分的更多訊息。