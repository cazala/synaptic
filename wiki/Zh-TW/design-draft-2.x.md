# 2.x API 設計稿

狀態：草案

## 討論部分

### 期望

Synaptic 出於多種目的被運用在多個地方

其中大部分與理論，其他語言或其他語言的經驗轉移有關。JS 不是資料科學中最受歡迎的語言，因此我們可能應該看看其他語言中好的方案並使用它們的實踐。

我們有很多有用的統計資料：[https://github.com/showcases/machine-learning](https://github.com/showcases/machine-learning)，在這裡我們可以看到頂級的 NN 解決方案：

- Scikit (Python)
- Caffe (C++)
- Keras (Python)

和很多其他的 C++、Python 和 Java 框架，還有那些必須被提到的（事實上，他們不在前五名中，但記住，他們真的應用很廣泛）如：Lasagne、FANN

在這個列表中我們沒有列出像 Brain.js 和 Convnet.js 這樣由著名的數據科學家 karpatny 創作的優秀的庫，但是在這項工作中我們也希望得到他一些幫助。

如果我們查看所有這些解決方案，我們將看到以下特性：

- 多後端：CPU、GPU、一些分布式選項，有些甚至引入其他東西
- 多層類型：稠密（普通）、丟失、卷積、遞迴、啟動是基本類型。
- 儘可能地聲明並儘可能地配置。這與使用多個後端的事實有關，例如：不能簡單地使用 lambda 函數。

所有這些都是為了：

- 儘可能快
- 儘可能提高記憶體效率
- 儘可能簡單

我們應該達到同樣的目的。所以，我建議把 Keras 作為參考（因為它有最好的文件）。

下一部分是設計草圖，隨時可以更改，代表虛構（現在）設計狀態。

現在這是一個樂觀的設計，目的是儘可能多的選擇。2.0 版本不會包括所有的內容。

### 請求

@jocooler 的 [建議](https://github.com/cazala/synaptic/issues/140#issuecomment-247605457) 是使人可讀的輸出（以某種形式）轉移到其他系統。JSON 是完美的。

## 正式部分

### 全局設計

對於最終用戶，提供以下文件：

```javascript
// 要使用的基本實體。預計將用作此類其他結構的抽象類。
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

### 網路的使用

這是一個 MNIST 訓練例子。

```javascript
const train_network = new Network(
    new Convolution2D(32, 4, 4),
    new Dropout(.2)
    new Activation.Relu(),
    new Flatten(),
    new Dense(10),
    new Activation.Softmax(),   
);

// 重要訊息: 現在這是一個非同步操作！

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