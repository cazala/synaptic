# Trainer

`Trainer(训练器)` 可以更容易地训练任何网络，无论其架构如何。要创建一个 Trainer，你只需要提供一个 `Network (网络)` 来训练。

```js
var trainer = new Trainer(myNetwork);
```

`Trainer` 还包含内置任务来测试您的网络的性能。

## train

下面的方法允许您训练任何训练集到网络，训练集必须是包含具有输入和输出属性的对象的数组，例如，以下是如何使用训练器训练 XOR 到网络的方法：

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

您还可以在对象中为训练设置不同的选项作为第二个参数，如：

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

- **rate**: 训练网络的学习率。它可以是一个静态的速率（只是一个数字），动态的（一组数字，根据迭代次数从一个到下一个）或者一个回调函数：`(iterations, error) => rate.`
- **iterations**: 最大迭代次数
- **error**: 最小误差
- **shuffle**: 如果这是真的话，训练集在每次迭代之后都会被洗牌，这对于训练数据序列是有用的，对于具有上下文记忆的网络（例如LSTM），排序是没有意义的。
- **cost**: 你可以设置三种内置的成本函数用于训练 (`Trainer.cost.CROSS_ENTROPY`, `Trainer.cost.MSE` 和 `Trainer.cost.BINARY`)，可以从交叉熵或均方误差中进行选择，你也可以使用你自己的成本函数（targetValues，outputValues）
- **log**： 输出每X次迭代的错误和迭代。
- **schedule**: 您可以创建自定义的计划任务，每执行X次迭代就会执行一次。它可用于创建自定义日志，或基于传递给任务的数据（数据对象包括错误，迭代和当前学习速率）来计算分析。如果任务的返回值为true，则训练将被中止。这可以用来创建特殊条件来停止训练（即如果错误开始增加）。
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

当训练完成时，这个方法返回一个包含错误，迭代和训练时间的对象。

## trainAsync

这种方法与列车(train)的工作方式相同，但是它使用 WebWorker，所以训练不会影响用户界面（使用train方法进行的长时间培训可能会冻结浏览器上的用户界面，但使用trainAsync不会发生这种情况） 。此方法在node.js中不起作用，并且可能不适用于每个浏览器（它必须支持Blob和WebWorker）。

```js
var trainer = new Trainer(myNetwork);
trainer.trainAsync(set, options)
.then(results => console.log('done!', results)
```

它具有相同的签名并支持与train相同的选项，但不返回训练结果，而是返回一个可以解决训练结果的 `Promise`

这是一个如何使用trainAsync方法训练XOR的例子：

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

此方法接受与 `train(dataSet，options)` 相同的参数。它将迭代 `dataSet`，激活网络。它返回经过的时间和错误（默认情况下，MSE，但您可以在 `options(选项)` 中指定成本函数，与 `train()` 中的方法相同）。

## XOR

这种方法训练XOR到网络，当你尝试不同的体系结构，并且你想测试和比较它们的性能时，这个方法非常有用：

```js
var trainer = new Trainer(myNetwork);
trainer.XOR(); // {error: 0.004999821588193305, iterations: 21333, time: 111}
```

## DSR

该方法训练网络以完成 [离散序列记忆](http://synaptic.juancazala.com/#/dsr)，这是测试神经网络中上下文记忆的任务。

```js
trainer.DSR({
	targets: [2,4],
	distractors: [3,5],
	prompts: [0,1],	
	length: 10	
});
```

## ERG

这种方法训练网络通过[嵌入式Reber语法](http://www.willamette.edu/~gorr/classes/cs449/reber.html)测试。

`trainer.ERG();`

## timingTask

这个测试挑战网络来完成一个[计时任务](https://github.com/cazala/synaptic/issues/30#issuecomment-97624779)。