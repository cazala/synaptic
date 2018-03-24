# Trainer

`Trainer`（訓練器）可以更容易地訓練任何網路，無論其架構如何。要創建一個 Trainer，你只需要提供一個 `Network (網路)` 來訓練。

```js
var trainer = new Trainer(myNetwork);
```

`Trainer` 還包含內建任務來測試您的網路的性能。

## train

下面的方法允許您訓練任何訓練集到網路，訓練集必須是包含具有輸入和輸出屬性的對象的數組，例如，以下是如何使用訓練器訓練 XOR 到網路的方法：

```js
var myNetwork = new Architect.Perceptron(2, 2, 1)
var trainer = new Trainer(myNetwork)

var trainingSet = [
  {
    input: [0,0],
    output: [0]
  },
  {
    input: [0,1],
    output: [1]
  },
  {
    input: [1,0],
    output: [1]
  },
  {
    input: [1,1],
    output: [0]
  },
]

trainer.train(trainingSet);
```

您還可以在對象中為訓練設置不同的選項作為第二個參數，如：

```js
trainer.train(trainingSet,{
	rate: .1,
	iterations: 20000,
	error: .005,
	shuffle: true,
	log: 1000,
	cost: Trainer.cost.CROSS_ENTROPY
});
```

### Options

- **rate**: 訓練網路的學習率。它可以是一個靜態的速率（只是一個數字），動態的（一組數字，根據疊代次數從一個到下一個）或者一個回調函數：`(iterations, error) => rate.`
- **iterations**: 最大疊代次數
- **error**: 最小誤差
- **shuffle**: 如果這是 `true` 的話，訓練集在每次疊代之後都會被洗牌，這對於訓練數據序列是有用的，對於具有上下文記憶的網路（例如 LSTM），排序是沒有意義的。
- **cost**: 你可以設置三種內建的成本函數用於訓練 (`Trainer.cost.CROSS_ENTROPY`, `Trainer.cost.MSE` 和 `Trainer.cost.BINARY`)，可以從交叉熵或均方誤差中進行選擇，你也可以使用你自己的成本函數（targetValues，outputValues）
- **log**： 輸出每 X 次疊代的錯誤和疊代。
- **schedule**: 您可以建立自訂的計劃任務，每執行 X 次疊代就會執行一次。它可用於建立自訂日誌，或基於傳遞給任務的數據（數據對象包括錯誤，疊代和當前學習速率）來計算分析。如果任務的返回值為 `true`，則訓練將被中止。這可以用來建立特殊條件來停止訓練（即如果錯誤開始增加）。
```js
schedule: {
	every: 500, // repeat this task every 500 iterations
	do: function(data) {
		// custom log
		console.log("error", data.error, "iterations", data.iterations, "rate", data.rate);
		if (someCondition)
			return true; // abort/stop training
	}
}
```

當訓練完成時，這個方法返回一個包含錯誤，疊代和訓練時間的對象。

## trainAsync

這種方法與訓練（train）的工作方式相同，但是它使用 WebWorker，所以訓練不會影響使用者界面（使用 `train` 方法進行的長時間培訓可能會凍結瀏覽器上的使用者介面，但使用 `trainAsync` 則不會發生這種情況） 。此方法在 node.js 中不起作用，並且可能不適用於每個瀏覽器（它必須支持 Blob 和 WebWorker）。

```js
var trainer = new Trainer(myNetwork);
trainer.trainAsync(set, options)
.then(results => console.log('done!', results)
```

它具有相同的簽名並支持與 `train` 相同的選項，但不返回訓練結果，而是返回一個可以解決訓練結果的 `Promise`

這是一個如何使用 `trainAsync` 方法訓練 XOR 的例子：

```js
var myNetwork = new Architect.Perceptron(2, 2, 1)
var trainer = new Trainer(myNetwork)

var trainingSet = [
  {
    input: [0,0],
    output: [0]
  },
  {
    input: [0,1],
    output: [1]
  },
  {
    input: [1,0],
    output: [1]
  },
  {
    input: [1,1],
    output: [0]
  },
]

trainer.trainAsync(trainingSet)
.then(results => console.log('done!', results))
```

## test

此方法接受與 `train(dataSet，options)` 相同的參數。它將疊代 `dataSet`，啟動網路。它返回經過的時間和錯誤（預設情況下為 MSE，但您可以在 `options(選項)` 中指定成本函數，與 `train()` 中的方法相同）。

## XOR

這種方法訓練XOR到網路，當你嘗試不同的體系結構，並且你想測試和比較它們的性能時，這個方法非常有用：

```js
var trainer = new Trainer(myNetwork);
trainer.XOR(); // {error: 0.004999821588193305, iterations: 21333, time: 111}
```

## DSR

該方法訓練網路以完成 [離散序列記憶](http://synaptic.juancazala.com/#/dsr)，這是測試神經網路中上下文記憶的任務。

```js
trainer.DSR({
	targets: [2,4],
	distractors: [3,5],
	prompts: [0,1],	
	length: 10	
});
```

## ERG

這種方法訓練網路通過 [嵌入式Reber語法](http://www.willamette.edu/~gorr/classes/cs449/reber.html) 測試。

`trainer.ERG();`

## timingTask

這個測試挑戰網路來完成一個 [計時任務](https://github.com/cazala/synaptic/issues/30#issuecomment-97624779)。