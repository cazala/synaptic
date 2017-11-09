# 2.x API 设计稿

状态：草案

## 讨论部分

### 期望

Synaptic 出于多种目的被运用在多个地方

其中大部分与理论，其他语言或其他语言的经验转移有关。JS 不是数据科学中最受欢迎的语言，因此我们可能应该看看其他语言中好的方案并使用它们的实践。

我们有很多有用的统计资料：[https://github.com/showcases/machine-learning](https://github.com/showcases/machine-learning)，在这里我们可以看到顶级的 NN 解决方案：

- Scikit (Python)
- Caffe (C++)
- Keras (Python)

和很多其他的C++，Python和Java框架，还有那些必须被提到的（事实上，他们不在TOP5中，但记住，他们真的应用很广泛）如：Lasagne、FANN

在这个列表中我们没有列出像 Brain.js 和 Convnet.js 这样由著名的数据科学家 karpatny 创作的优秀的库，但是在这项工作中我们也希望得到他一些帮助。

如果我们查看所有这些解决方案，我们将看到以下特性：

- 多后端：CPU，GPU，一些分布式选项，有些甚至引入其他东西
- 多层类型：稠密（普通）、丢失、卷积、递归、激活是基本类型。
- 尽可能地声明并尽可能地配置。这与使用多个后端的事实有关，不能简单地使用lambda函数。

所有这些都是为了：

- 尽可能快
- 尽可能提高内存效率
- 尽可能简单

我们应该达到同样的目的。所以，我建议把Keras作为参考（因为它有最好的文档）。

下一部分是设计草图，随时可以更改，代表虚构（现在）设计状态。

现在这是一个乐观的设计，目的是尽可能多的选择。2.0版本不会包括所有的内容。

### 请求

@jocooler 的 [建议](https://github.com/cazala/synaptic/issues/140#issuecomment-247605457) 是使人可读的输出（以某种形式）转移到其他系统。JSON是完美的。

## 正式部分

### 全局设计

对于最终用户，提供以下文件：

```javascript
// 要使用的基本实体。预计将用作此类其他结构的抽象类。
import {
  Network,
  Layer,
  Trainer,
} from 'synaptic';

import {
  AsmJS,
  TensorFlow,
  WorkerAsmJS,
  WebCL,
} from 'synaptic/optimizers';

import {
  Dense,
  Activation,
  Dropout,

  Flatten,
  Reshape,
  Permute,


  Convolution1D,
  MaxPooling1D,
  AveragePooling1D,

  Convolution2D,
  MaxPooling2D,
  AveragePooling2D,

  GRU,
  LSTM
} from 'synaptic/layers';

import {
  Trainer, //same as in 'synaptic'
  objectives, //same as below, but as an object
  optimizers, //same as below, but as an object
} from 'synaptic/train'

import {
  mean_squared_error, 
  mean_absolute_error, 
  mean_absolute_percentage_error, 
  mean_squared_logarithmic_error, 
  squared_hinge,
  hinge, 
  binary_crossentropy, 
  categorical_crossentropy, 
  sparse_categorical_crossentropy, 
  kullback_leibler_divergence,

  iterations,
  time,

  any,
  every,
} from 'synaptic/train/objectives';

import {
  SGD, 
  RMSprop, 
  Adagrad, 
  Adadelta, 
  Adam, 
  Adamax, 
  Nadam,
} from 'synaptic/train/optimizers'

import {
  to_categorical,
} from 'synaptic/util'
```

### 网络的使用

这是一个 MNIST 训练实例。

```javascript
const train_network = new Network(
    new Convolution2D(32, 4, 4),
    new Dropout(.2)
    new Activation.Relu(),
    new Flatten(),
    new Dense(10),
    new Activation.Softmax(),   
);

// 重要信息: 现在这是一个异步操作！

await train_network.optimize(new WorkerAsmJS());

const trainer = new Trainer(train_network, {
    optimizer: RMSprop    
})

await trainer.train(train_input, to_categorical(train_output), {
    // any is imported from objectives
    objectives: objectives.any([
         objectives.categorical_crossentropy(.0005),
         objectives.every([
             objectives.iterations(5000),
             objectives.time(1000)
         ])
    ]),
    learning_rate: 1
});

assert.equal(await train_network.activate(test_input[0]), test_output[0])

//this is equal to something like test_input.map(network.activate) - just iteration over every entity

const weights = await train_network.export_weights()
const optimizer = await train_network.export_optimizer()

/* Let's imagine this is separate application already running */

const test_network = new Network(
    new Convolution2D(32, 4, 4),
    new Dropout(.2)
    new Activation.Relu(),
    new Flatten(),
    new Dense(10),
    new Activation.Softmax(),   
);

await test_network.import_weights(weights);

await test_network.import_optimizer(optimizer);

console.log(await train_network.activate_array(test_input));
```