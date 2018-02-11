# Trainer

`Trainer`を使用するとアーキテクチャに関係なく、ネットワークに任意のトレーニングセットを訓練することが容易になります。 トレーニング用の`Network`を用意するだけでトレーナーを作成するができます。

`var trainer = new Trainer(myNetwork);`

`Trainer`にはネットワークのパフォーマンスをテストする組み込みタスクも含まれています。

train
=====

`train`ではトレーニングセットをネットワークに訓練することができます。 トレーニングセットは`input`プロパティと`output`プロパティを持つオブジェクトを含む配列です。たとえば、トレーナーを使用してXORを訓練する場合：

```javascript
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
こうやってさまざまなオプションを設定することもできます。

```javascript
trainer.train(trainingSet,{
	rate: .1,
	iterations: 20000,
	error: .005,
	shuffle: true,
	log: 1000,
	cost: Trainer.cost.CROSS_ENTROPY
});
```

#### オプション

- **rate**: ネットワークを訓練するための学習率です。 静的な数値、配列、または`(iterations, error) => rate`のコールバック関数を入力することができます。配列を入力すると反復回数に応じて1つから次の配列に移行します。
- **iterations**: 最大反復回数
- **error**: 最小誤差
- **shuffle**: `true`の場合、訓練セットは繰り返しごとにシャッフルされます。 LSTMのような、順序が意味を持たないデータシーケンスを訓練するのに便利です。
- **cost**: トレーニングに使用するコスト関数を設定できます。 `Trainer.cost.CROSS_ENTROPY`(クロスエントロピー)、`Trainer.cost.MSE`(平均二乗誤差)および`Trainer.cost.BINARY`(バイナリ)の3つの関数があります。独自のコスト関数を使用することもできます。(`targetValues`、`outputValues`)
- **log**: 設定された反復回数ごとにエラーと反復回数を "console.log"に出力します。
- **schedule**: -設定した回数の反復を実行するカスタムのスケジュールされたタスクを作成することができます。カスタムログを作成したり、タスクに渡されたデータに基づいて分析を計算したりすることができます。`data`オブジェクトには`error`、`iterations`(現在の学習率)が含まれます。タスクの戻り値が`true`の場合、トレーニングは中止されます。 エラーが増加し始めた時などにトレーニングを停止する特別な条件を作成するために使用できます。

```javascript
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

トレーニングが完了するとエラー、繰り返し、およびトレーニングの経過時間を含むオブジェクトを返します。

trainAsync
===========

`trainAsync`は`train`と同じように機能しますが、`WebWorker`を使用するため、ユーザーインターフェイスに影響しません。 `train`を使用した非常に長いトレーニングでは、ブラウザのUIがフリーズすることがありますが、`trainAsync`を使用してもそのようなことは起こりません。*node.js*では機能しません。 ブラウザは`Blob`と`WebWorker`をサポートしている必要があります。

```javascript
var trainer = new Trainer(myNetwork);
trainer.trainAsync(set, options)
.then(results => console.log('done!', results)
```

`train`と同じシグネチャとオプションをサポートしますが、トレーニング結果を返す代わりに結果に解決する`Promise`を返します

`trainAsync`を使用してXORをトレーニングする方法

```javascript
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

test
====

`train(dataSet, options)`と同じ引数を受け入れます。 データセットを繰り返してネットワークをアクティブにします。 経過時間とエラーを返します。 デフォルトでは平均二乗誤差ですが、`options`に`train()`と同じようにコスト関数を指定できます。

XOR
===

XORをネットワークに渡します。 異なるアーキテクチャーのパフォーマンスを比較したいときに便利です：

```javascript
var trainer = new Trainer(myNetwork);
trainer.XOR(); // {error: 0.004999821588193305, iterations: 21333, time: 111}
```

DSR
===

ネットワークを訓練して[Discrete Sequence Recall](http://synaptic.juancazala.com/#/dsr)を完成させます。 ニューラルネットワークにおけるコンテクストメモリをテストするためです。

```javascript
trainer.DSR({
	targets: [2,4],
	distractors: [3,5],
	prompts: [0,1],	
	length: 10	
});
```


ERG
===

ネットワークを訓練し[Embedded Reber Grammar](http://www.willamette.edu/~gorr/classes/cs449/reber.html)テストを完成させます。

`trainer.ERG();`


timingTask
==========

ネットワークが[timing task](https://github.com/cazala/synaptic/issues/30#issuecomment-97624779)を完成できるか試します。
