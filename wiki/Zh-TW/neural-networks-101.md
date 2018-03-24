# Neural Networks 101

本文的目的是作為一個介紹，了解什麼是神經網路，以及它們如何運作，沒有那些的沉重的數學。如果你想要更進階的東西，或其他東西，你應該查看文章結尾處的連結

那麼，讓我們從基礎開始

## Neuron（神經元）

神經元是神經網路的基本單元。在自然界中，神經元有許多樹突（輸入），細胞核（處理器）和軸突（輸出）。當神經元啟動時，它累積所有輸入，如果它超過一定的閾值，它通過軸突發出信號。關於神經元的重要事情是他們可以 `學習`。

人造神經元看起來更像這樣：

*neuron j:*

![Artificial Neuron](http://i.imgur.com/d6T7K93.png "Artificial Neuron")

正如你所看到的，他們有幾個輸入，每個輸入都有一個權重（特定連接的權重）。當人造神經元啟動時，透過將所有輸入的輸入乘以其相應的連接權重來計算其狀態。但神經元總是有一個額外的輸入──偏差──總是 1，並有自己的連接權重。這可以確保即使所有的輸入都沒有（全為 0），神經元中也會有啟動。

![](http://latex.codecogs.com/gif.latex?s_j%20%3D%20%5Csum_%7Bi%7D%20w_%7Bij%7D.y_%7Bi%7D)

*其中y<sub>i</sub>是所有的輸入（包括偏差）*

在計算出它的狀態之後，神經元通過它的啟動函數來傳遞它，這個函數對結果進行正規化（通常在 0~1 之間）

![](http://latex.codecogs.com/gif.latex?y_j%20%3D%20f_j%28S_j%29)

## Activation function（啟動函式）

啟動函數通常是一個 Sigmoid 函數：[Logistic](http://en.wikipedia.org/wiki/Logistic_function) 或 [雙曲正切函數](http://mathworld.wolfram.com/HyperbolicTangent.html)。

![Logistic](http://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)

## Feed-forward Network (前饋網路)

這是最簡單的架構，它由分層組織神經元和將一個層中的所有神經元連接到下一個神經元中的所有神經元組成，因此每層的輸出（該層中的每個神經元的輸出）成為下一層的輸入。

![Feed-forward Network](http://i.imgur.com/3u9ORal.jpg?1)

第一層（輸入層）接收來自環境的輸入、啟動，其輸出作為下一層的輸入。重複這個過程直到到達最後一層（輸出層）。

### 那麼神經網路如何進行學習？

神經網路通過訓練來學習。用來做這個的算法被稱為**反向傳播**。給網路一個輸入後，它將產生一個輸出，下一步是教網路該輸入應該是什麼樣的正確輸出（理想輸出）。網路將採取這個理想的輸出，並開始調整權重，以產生一個更準確的輸出，從輸出層開始，並倒退，直到到達輸入層。因此，下一次我們向網路顯示相同的輸入時，它將使輸出接近我們訓練輸出的理想輸出。這個過程重複許多次疊代，直到我們認為理想輸出和網路輸出之間的誤差夠小。

### 那麼反向傳播是如何工作的呢？

該算法使用 [梯度下降算法](http://en.wikipedia.org/wiki/Gradient_descent) 來調整權重。假設我們對網路輸出中的某個權重與錯誤之間的關係進行了描述：

![](http://i.imgur.com/6VZeBpn.png)

這個算法計算出重量的實際值的梯度（也稱為圖像中的箭頭）的瞬時斜率，並將其移嚮導致較低誤差（圖像中的紅點）的方向。網路中每一個重量都會重複這個過程。

![Slope](http://latex.codecogs.com/gif.latex?slope%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_i_j%7D)

要計算梯度（斜率）並調整權重，我們使用Delta（δ）規則：

### Delta Rule

對於輸出層（θ），使用*注入誤差*（網路輸出與理想或目標輸出之間的差值）計算增量值。

![](http://latex.codecogs.com/gif.latex?E_%5CTheta%20%3D%20t%20-%20y_%5CTheta)

![](http://latex.codecogs.com/gif.latex?%5Cdelta%20_%5CTheta%20%3D%20E_%5CTheta%20.f%27%28s_%5CTheta%20%29)

*其中f'是啟動函式的導數*

錯誤透過網路向後傳播，直到達到輸入層。

每一層使用來自先前計算的層的 δ 來計算其自己的 δ

![](http://latex.codecogs.com/gif.latex?E_j%20%3D%20%5Csum%20%5Cdelta%20_k%20w_k_j)

![](http://latex.codecogs.com/gif.latex?%5Cdelta_j%20%3D%20E_j.f%27%28S_j%29)

我們使用 delta 來計算每個重量的梯度：

![](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E_j%7D%7B%5Cpartial%20w_i_j%7D%20%3D%20%5Cdelta_j%20.%20y_i)

現在我們可以使用反向傳播算法更新權重：

![](http://latex.codecogs.com/gif.latex?%5CDelta%20w_i_j%20%3D%20%5Cvarepsilon%20%5Cfrac%7B%5Cpartial%20E_j%7D%7B%5Cpartial%20w_i_j%7D)

*其中 ε 是學習率*

## Recurrent Neural Networks (復發神經網路)

這些網路中的神經元具有自我連接（固定權重為 1），可以讓他們擁有某種短期記憶。

![Recurrent Neural Network](http://upload.wikimedia.org/wikipedia/commons/d/dd/RecurrentLayerNeuralNetwork.png)

過去啟動的額外輸入為網路提供了一些上下文訊息，有助於在某些任務上產生更好的輸出。這種網路在序列預測任務中被證明是非常有效的，儘管過去許多步驟都不能記住相關訊息。

### 常量錯誤輪播

CEC 由自連神經元（我們稱之為*記憶細胞*）組成，具有**線性啟動**功能。這有助於誤差持續較長時間，修復了遞迴神經網路中的衰落梯度問題，這種問題通過推導壓縮函數來縮放每次啟動時的誤差，使其隨著時間的推移向後移動而呈指數減小或發散空間。這聽起來很*酷*。

### Gates（門）

有架構不僅使用神經元彼此之間的連接，而且調節流經這些連接的訊息，這些架構被稱為**二階神經網路**。

保護存儲器單元免受噪聲輸入和注入錯誤的一種方式是使用門來縮放存儲器單元和輸入/輸出層之間的連接：

![](http://www.willamette.edu/~gorr/classes/cs449/figs/lstm.gif)

這就是 Long Short Term Memory 網路架構如何開始的。 LSTM 是一個非常適合從經驗中學習的體系結構，當重要事件之間存在很長時間的未知時間滯後時，可以對時間序列進行分類，處理和預測。

自從它的概念以來，它已經被改進了第三門，叫做遺忘門（Forget gates），它調節存儲單元的自連接，決定應該記住多少錯誤以及何時忘記，通過在每個門之後縮放來自單元狀態的回饋時間步長。這保護了狀態的分化和崩潰。

LSTM 還增加了從存儲單元到所有門的窺視孔連接，這提高了它們的性能，因為它們具有關於它們所保護的單元的訊息。實際的 LSTM 架構如下所示：

![LSTM](http://i.imgur.com/JpF65wc.png)

遞迴神經網路和二階神經網路的數學比前饋神經網路的方程稍微複雜一些。我不打算在這篇文章中討論這個問題，但是如果你有興趣，你應該閱讀 [德里克·莫納](http://www.overcomplete.net/papers/nn2012.pdf) 的論文。

## 就是這樣

現在，您已經準備好開始成為神經網路大師的旅程，您可以先查看 Synaptic 的 [範例](https://github.com/NoobTW/synaptic/blob/master/README_Zh-TW.md#%E7%A4%BA%E7%AF%84%E7%94%A8%E4%BE%8B)和 [文件](https://github.com/cazala/synaptic/wiki#documentation)，查看 [原始碼](https://github.com/cazala/synaptic/tree/master/src)或閱讀 [這篇文章](http://www.overcomplete.net/papers/nn2012.pdf)。