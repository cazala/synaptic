## Layers (層)

通常你不會使用單個神經元，而是使用層代替。一層基本上是一組神經元，它們可以做幾乎和神經元一樣的工作，但它使寫程式過程更快。

要建立一個層，只需指定它的大小（該層中的神經元數量）：

```javascript
var myLayer = new Layer(5);
```
### project

一個層可以投射到另一層的連接。你必須提供層要連接和連接類型：

```javascript
var A = new Layer(5);
var B = new Layer(3);
A.project(B, Layer.connectionType.ALL_TO_ALL); // A 層中的所有神經元現在都在與B層中的所有神經元建立連接。
```

層也可以自我連接：

```javascript
A.project(A, Layer.connectionType.ONE_TO_ONE);
```

這裡有三種連接類型（`connectionType`）：

- `Layer.connectionType.ALL_TO_ALL`: 它把每一個神經元從 A 層連接到 B 層的每個神經元。
- `Layer.connectionType.ONE_TO_ONE`: 它把每一個神經元從 A 層連接到 B 層的一個神經元中。每層的大小必須相同
- `Layer.connectionType.ALL_TO_ELSE`: 僅在自我連接中有用。它把每個神經元從一層神經元連接到同一層的所有其他神經元，除了自身。如果用戶端是用於不同層之間的連接，它產生相同的結果 `ALL_TO_ALL`。

**NOTE**: 當連接兩個不同層時，連接類型預設是 `Layer.connectionType.ALL_TO_ALL`，當一個層連接到它自己時(如：`myLayer.project(myLayer`)，連接類型預設是`Layer.connectionType.ONE_TO_ONE`

該方法返回一個 `layerconnection` 項目對象，可由另一層門接受

### gate

一個層可以屏蔽兩個其他層之間的連接，或者層的自連接。

```javascript
var A = new Layer(5);
var B = new Layer(3);
var connection = A.project(B);

var C = new Layer(4);
C.gate(connection, Layer.gateType.INPUT_GATE); // now C gates the connection between A and B (input gate)
```

這裡有三種門類型（`gateType's`）:

- `Layer.gateType.INPUT_GATE`: 如果 C 層是 A 層和 B 層之間的門控連接，那麼 C 門的所有神經元都是B的輸入連接。
- `Layer.gateType.OUTPUT_GATE`: 如果 C 層是 A 層和 B 層之間的門控連接，那麼 C 門的所有神經元都來自A的輸出連接。
- `Layer.gateType.ONE_TO_ONE`: 如果 C 層是 A 層和 B 層之間的門控連接，那麼 C 神經元的每個神經元都有一個從A到B的連接，這對於門控自連接層是有用的。使用這種 gateType，A、B 和 C 的大小必須相同

### activate

當一個層啟動時，它會依次啟動該層中的所有神經元，並返回一個具有輸出的數組。它接受一系列的啟動作為參數（用於輸入層）：

```javascript
var A = new Layer(5);
var B = new Layer(3);
A.project(B);

A.activate([1,0,1,0,1]); // [1,0,1,0,1]
B.activate(); // [0.3280457, 0.83243247, 0.5320423]
```

### propagate(傳播)

在啟動後，你可以教給這個層正確的輸出應該是什麼（又名：訓練），這是由反向傳播誤差完成的。要使用傳播方法，必須提供學習率和目標值（浮動數組在 0 到 1 之間）。

例如，如果我想在 A 層啟動 [1,0,1,0,1] 時訓練 B 層輸出 [0,0]：

```javascript
var A = new Layer(5);
var B = new Layer(2);
A.project(B);

var learningRate = .3;

for (var i = 0; i < 20000; i++)
{
	// when A activates [1, 0, 1, 0, 1]
	A.activate([1,0,1,0,1]);

	// train B to activate [0,0]
	B.activate();
	B.propagate(learningRate, [0,0]);
}

// test it
A.activate([1,0,1,0,1]);
B.activate(); // [0.004606949693864496, 0.004606763721459169]
```

#### 壓縮函數和偏置

您可以使用方法集來設置圖層中所有神經元的壓縮函數和偏差

```javascript
myLayer.set({
	squash: Neuron.squash.TANH,
	bias: 0
})
```

### neurons （神經元）

使用 `neurons()` 方法以啟動順序返回層中所有神經元的數組。