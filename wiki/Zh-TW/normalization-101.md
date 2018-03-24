**更新**：我們推薦使用[神經數據歸一化器](https://github.com/adadgio/neural-data-normalizer)來讓您的生活更輕鬆

我在發現 [這個問題](https://github.com/cazala/synaptic/issues/72) 之後建立了這個帖子，使用者詢問了如何向神經網路提供數據，而且這似乎對其他人有用，也是數據規範化的一個很好的例子。

範例訓練集如下：

```
[
  {name:'workout', duration:'120', enjoy: true, time:1455063275, tags:['gym', 'weights']},
  {name:'lunch', duration:'45', enjoy: false, time:1455063275, tags:['salad', 'wine']},
  {name:'sleep', duration:'420', enjoy: true, time:1455063275, tags:['bed', 'romance']}
]
```

神經網路的工作，你需要準備一個訓練集，包括輸入和他們所需的輸出。在這種情況下，將會是這樣的：

```
("workout",  120, ['gym', 'weights']) => 'is enjoyable'
("lunch",  45, ['salad', 'wine']) => 'is not enjoyable'
("sleep",  420, ['bed', 'romance']) => 'is enjoyable'
```

但神經網路不知道什麼是「鍛鍊」或 45 或['沙拉', '酒']。他們只理解一個單一的輸入，只包含 0 到 1 之間的值，它必須有一個固定的大小，所以所有的輸入具有相同的長度。所以你需要正規化你的輸入/輸出數據。

「名稱」輸入可以歸類為類別。假設你有三個類別：「鍛鍊」，「午餐」和「睡眠」，每一個都可以用一個標誌位表示（注意，這通常稱為二進位制化）。所以我們可以使用3位：

```
"workout" => 0, 0, 1
"lunch" => 0, 1, 0
"sleep" => 1, 0, 0
```

然後，可以將「持續時間」歸一化為 0 到 1 之間的值，並設置最大值並除以它。假設您的最大持續時間是 1000，那麼您的輸入將如下所示：

```
120 => 0.12
45 => 0.045
420 => 0.42
```

對於「標籤」類別，您可以再次使用類別，但要合併。所以我們假設你有6個類別：

```
gym => 0,0,0,0,0,1
weights => 0,0,0,0,1,0
salad => 0,0,0,1,0,0
wine => 0,0,1,0,0,0
bed => 0,1,0,0,0,0
romance => 1,0,0,0,0,0
```

那麼你的輸入將如下所示：

```
['gym', 'weights'] => 0,0,0,0,1,1
['salad', 'wine'] => 0,0,1,1,0,0
['bed', 'romance'] => 1,1,0,0,0,0
```

最後，「愉快」是最簡單的，因為它是一個布林值，它可以寫成：

```
true => 1
false => 0
```

把所有這些放在一起你的訓練集就像:

```
("workout",  120, ['gym', 'weights']) => 0,0,1 + 0.12, +  0,0,0,0,1,1 => [0,0,1,0.12,0,0,0,0,1,1]
("lunch",  45, ['salad', 'wine']) => 0,1,0 + 0.045, +  0,0,1,1,0,0 => [0,1,0,0.045,0,0,1,1,0,0]
("sleep",  420, ['bed', 'romance']) => 1,0,0 + 0,42 + 1,1,0,0,0,0 => [1,0,0,0.42,1,1,0,0,0,0]
```

那麼你的輸出就是：

```
true => 1 => [1]
false => 0 => [0]
true => 1 => [1]
```

現在，您需要將其轉換為 synaptic。你需要一個網路，在輸入層有 10 個神經元，在輸出層有1個神經元（因為這是你輸入和輸出的大小）。你可以選擇不同的架構。如果訓練集的順序很重要，則需要使用帶有上下文記憶體的網路，如 LSTM。如果順序不重要，那麼你應該使用 Perceptron，這是不知情的情況。

我的意思是，如果這樣的話：

```
("workout",  120, ['gym', 'weights']) => true
("lunch",  45, ['salad', 'wine']) => false
("sleep",  420, ['bed', 'romance']) => true
```

應或不應與此相同：

```
("lunch",  45, ['salad', 'wine']) => false 
("workout",  120, ['gym', 'weights']) => true 
("sleep",  420, ['bed', 'romance']) => true 
```

如果，比方說，之前的工作會改變睡眠設置輸出吃午餐的時候，你需要使用一個網路，記得以前的啟動（LSTM）。

但是為了保持簡單，我們假設集合的順序不重要，並使用感知器。

```js
var myNet = new Architect.Perceptron(10, 7, 1);
```

我用 10 個輸入、7 個隱藏神經元和 1 個輸出神經元來創建它。隱層神經元的數目不能猜測你直截了當地通常使用數字輸入和輸出之間的數量。

現在，你將你的訓練集送到網路的訓練器那裡：

```js
var trainingSet = [
  {
    input: [0,0,1,0.12,0,0,0,0,1,1],
    output: [1]
  },
  {
    input:  [0,1,0,0.045,0,0,1,1,0,0],
    output: [0]
  },
  {
    input:  [1,0,0,0.42,1,1,0,0,0,0],
    output: [1]
  }
]

var trainingOptions = {
  rate: .1,
  iterations: 20000,
  error: .005,
}

myNet.trainer.train(trainingSet, trainingOptions);
```

如果在結果中收到 `NaN`，請確認您的輸入/輸出值，並且輸入節點的數量不大於輸入數組的長度。

訓練選項應根據每個特殊情況進行調整，您可以在 [訓練器](https://github.com/cazala/synaptic/blob/master/wiki/Zh-TW/trainer.md) 頁面閱讀更多關於訓練器選項的訊息