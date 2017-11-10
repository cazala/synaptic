# Neural Networks 101

本文的目的是作为一个介绍，了解什么是神经网络，以及它们如何工作，没有所有的沉重的数学。如果你想要更高级的东西，或其他东西，你应该查看文章结尾处的链接

那么，让我们从基础开始

## Neuron（神经元）

神经元是神经网络的基本单元。在自然界中，神经元有许多树突（输入），细胞核（处理器）和轴突（输出）。当神经元激活时，它累积所有输入的输入，如果它超过一定的阈值，它通过轴突发出信号。关于神经元的重要事情是他们可以 `学习`。

人造神经元看起来更像这样：

*neuron j:*

![Artificial Neuron](http://i.imgur.com/d6T7K93.png "Artificial Neuron")

正如你所看到的，他们有几个输入，每个输入都有一个权重（特定连接的权重）。当人造神经元激活时，通过将所有输入的输入乘以其相应的连接权重来计算其状态。但神经元总是有一个额外的输入，偏差，总是1，并有自己的连接权重。这可以确保即使所有的输入都没有（全0），神经元中也会有激活。

![](http://latex.codecogs.com/gif.latex?s_j%20%3D%20%5Csum_%7Bi%7D%20w_%7Bij%7D.y_%7Bi%7D)

*其中y<sub>i</sub>是所有的输入（包括偏差）*

在计算出它的状态之后，神经元通过它的激活函数来传递它，这个函数对结果进行归一化（通常在0-1之间）

![](http://latex.codecogs.com/gif.latex?y_j%20%3D%20f_j%28S_j%29)

## Activation function(激活方法)

激活函数通常是一个Sigmoid函数，[Logistic](http://en.wikipedia.org/wiki/Logistic_function) 或 [双曲正切函数](http://mathworld.wolfram.com/HyperbolicTangent.html)。

![Logistic](http://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)

## Feed-forward Network (前馈网络)

这是最简单的架构，它由分层组织神经元和将一个层中的所有神经元连接到下一个神经元中的所有神经元组成，因此每层的输出（该层中的每个神经元的输出）成为下一层的输入。

![Feed-forward Network](http://i.imgur.com/3u9ORal.jpg?1)

第一层（输入层）接收来自环境的输入，激活，其输出作为下一层的输入。重复这个过程直到到达最后一层（输出层）。

### 那么神经网络如何进行学习？

神经网络通过训练来学习。用来做这个的算法被称为反向传播。给网络一个输入后，它将产生一个输出，下一步是教网络该输入（理想输出）应该是什么样的正确输出。网络将采取这个理想的输出，并开始调整权重，以产生一个更准确的输出下一次，从输出层开始，并倒退，直到到达输入层。因此，下一次我们向网络显示相同的输入时，它将使输出接近我们训练输出的理想输出。这个过程重复许多次迭代，直到我们认为理想输出和网络输出之间的误差足够小。

### 那么反向传播是如何工作的呢？

该算法使用 [梯度下降算法](http://en.wikipedia.org/wiki/Gradient_descent) 来调整权重。假设我们对网络输出中的某个权重与错误之间的关系进行了描述：

![](http://i.imgur.com/6VZeBpn.png)

这个算法计算出重量的实际值的梯度（也称为图像中的箭头）的瞬时斜率，并将其移向导致较低误差（图像中的红点）的方向。网络中每一个重量都会重复这个过程。

![Slope](http://latex.codecogs.com/gif.latex?slope%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_i_j%7D)

要计算梯度（斜率）并调整权重，我们使用Delta（δ）规则：

### Delta Rule

对于输出层（θ），使用*注入误差*（网络输出与理想或目标输出之间的差值）计算增量值。

![](http://latex.codecogs.com/gif.latex?E_%5CTheta%20%3D%20t%20-%20y_%5CTheta)

![](http://latex.codecogs.com/gif.latex?%5Cdelta%20_%5CTheta%20%3D%20E_%5CTheta%20.f%27%28s_%5CTheta%20%29)

*其中f'是激活函数的导数*

错误通过网络向后传播，直到达到输入层。

每一层使用来自先前计算的层的δ来计算其自己的δ

![](http://latex.codecogs.com/gif.latex?E_j%20%3D%20%5Csum%20%5Cdelta%20_k%20w_k_j)

![](http://latex.codecogs.com/gif.latex?%5Cdelta_j%20%3D%20E_j.f%27%28S_j%29)

我们使用delta来计算每个重量的梯度：

![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_j%7D%7B%5Cpartial%20w_i_j%7D%20%3D%20%5Cdelta_j%20.%20y_i)

现在我们可以使用反向传播算法更新权重：

![](http://latex.codecogs.com/gif.latex?%5CDelta%20w_i_j%20%3D%20%5Cvarepsilon%20%5Cfrac%7B%5Cpartial%20E_j%7D%7B%5Cpartial%20w_i_j%7D)

*其中ε是学习率*

## Recurrent Neural Networks (复发神经网络)

这些网络中的神经元具有自我连接（固定权重为1），可以让他们拥有某种短期记忆。

![Recurrent Neural Network](http://upload.wikimedia.org/wikipedia/commons/d/dd/RecurrentLayerNeuralNetwork.png)

过去激活的额外输入为网络提供了一些上下文信息，有助于在某些任务上产生更好的输出。这种网络在序列预测任务中被证明是非常有效的，尽管过去许多步骤都不能记住相关信息。

### 常量错误轮播

CEC由自连神经元（我们称之为*记忆细胞*）组成，具有**线性激活**功能。这有助于误差持续较长时间，修复了递归神经网络中的衰落梯度问题，这种问题通过推导压缩函数来缩放每次激活时的误差，使其随着时间的推移向后移动而呈指数减小或发散空间。这听起来很*酷*。

### Gates

有架构不仅使用神经元彼此之间的连接，而且调节流经这些连接的信息，这些架构被称为**二阶神经网络**。

保护存储器单元免受噪声输入和注入错误的一种方式是使用门来缩放存储器单元和输入/输出层之间的连接：

![](http://www.willamette.edu/~gorr/classes/cs449/figs/lstm.gif)

这就是Long Short Term Memory网络架构如何开始的。 LSTM是一个非常适合从经验中学习的体系结构，当重要事件之间存在很长时间的未知时间滞后时，可以对时间序列进行分类，处理和预测。

自从它的概念以来，它已经被改进了第三门，叫做Forget门，它调节存储单元的自连接，决定应该记住多少错误以及何时忘记，通过在每个门之后缩放来自单元状态的反馈时间步长。这保护了状态的分化和崩溃。

LSTM还增加了从存储单元到所有门的窥视孔连接，这提高了它们的性能，因为它们具有关于它们所保护的单元的信息。实际的LSTM架构如下所示：

![LSTM](http://i.imgur.com/JpF65wc.png)

递归神经网络和二阶神经网络的数学比前馈神经网络的方程稍微复杂一些。我不打算在这篇文章中讨论这个问题，但是如果你有兴趣，你应该阅读 [德里克·莫纳](http://www.overcomplete.net/papers/nn2012.pdf) 的论文。

## That's it

现在，您已经准备好开始成为神经网络大师的旅程，您可以先查看Synaptic的[演示](http://github.com/cazala/synaptic#demos)和[文档](https://github.com/cazala/synaptic/wiki#documentation)，查看[源代码](https://github.com/cazala/synaptic/tree/master/src)或阅读[这篇文章](http://www.overcomplete.net/papers/nn2012.pdf)。