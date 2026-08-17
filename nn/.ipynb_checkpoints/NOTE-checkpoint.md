# 神经网络形式化框架

## 1. 基本定义

对于一个神经网络 $N$：

- 设第 $i$ 层第 $n$ 个节点的第 $m$ 个参数为 $w_{nm}^i$
- 第 $\alpha$ 个样本的第 $\beta$ 个分量为 $x_\beta^\alpha$

---

## 2. 准内积映射

定义准内积映射 $\omega$，对任意数列 $a_n, b_m$：

$$
\omega: a_n, b_m \mapsto \sum_{n,m} a_n b_m \delta_{nm} = \sum_n a_n b_n
$$

记为：

$$
\omega(a_n, b_m)
$$

---

## 3. 神经网络中的准内积运算

### 3.1 第一层：参数和样本之间

（下文均按照传统张量符号书写）

$$
\omega(x_\beta^\alpha, w_{nm}^1) = \sum_m x_m^\alpha w_{nm}^1, \quad \text{s.w. } \omega_n^1
$$

### 3.2 不同层之间

$$
\omega(\omega_n^i, w_{nm'}^{i+1}) = \sum_m \omega_n^i w_{nm}^{i+1}, \quad \text{s.w. } \omega_n^{i+1}
$$

---

## 4. Sigmoid 函数

定义 Sigmoid 函数 $S$，对任意标量 $a$：

$$
S(a) = (1 + e^{\beta_i a})^{-1}
$$

其中，$\beta_i$ 为第 $i$ 层热力学逆温度。

---

## 5. 各层激活输出

### 5.1 第一层

$$
S(\omega_n^1) = (1 + e^{\beta_1 \sum_m x_m^\alpha w_{nm}^1})^{-1} = (1 + e^{\beta_1 \omega_n^1})^{-1} = S_n^1
$$

### 5.2 第二层

$$
S(\omega_n^2) = (1 + e^{\beta_1 \sum_m \omega_m^1 w_{nm}^2})^{-1} = S_n^2
$$

### 5.3 第 $i$ 层

$$
S(\omega_n^i) = (1 + e^{\beta_i \sum_m \omega_m^{i-1} w_{nm}^i})^{-1} = S_n^i
$$

---

## 6. Softmax 函数

定义 Softmax 函数 $A$，对任意矢量 $a_n$：

$$
A_n = \frac{e^{a_n}}{\sum_m e^{a_m}}
$$

在神经网络中，输出层：

$$
A_n^f = \frac{e^{S_n^f}}{\sum_m e^{S_m^f}}
$$

---

## 7. 编码协议与损失函数

定义一种输出层节点和 label 的编码关系，$\Theta$#-自然编码协议：

> 输出层第 $i$ 个 node 对应第 $n$ 个 label：
>
> $$
> y^n \leftrightarrow S_n^1
> $$

---

### 7.1 单个节点损失

定义衡量第 $n$ 个节点给出结果距离 label 偏差的损失函数：

$$
L(A_n^f) = -\sum_n \delta_{y_n, n} \log A_n^f = -\log A_{y_n}^f = -\log A_n^f = L_n
$$

### 7.2 输出层总损失

整个输出层的损失函数为：

$$
L = \sum_n L_n
$$

---

## 8. 梯度

### 8.1 输出层梯度

损失函数对输出层第 $n$ 个节点参数的第 $m$ 个分量 $w_{nm}^f$ 的梯度 $\nabla_{w_{nm}^f} L = d_{nm}^f$：

$$
d_{nm}^f = \frac{\partial L}{\partial A_n^f} \frac{\partial A_n^f}{\partial S_n^f} \frac{\partial S_n^f}{\partial \omega_n^f} \frac{\partial \omega_n^f}{\partial w_{nm}^f}
$$

所有 $d_{nm}^f$ 构成损失函数 $L$ 对输出层所有参数的梯度矩阵 $D^f$：

$$
D^f = 
\begin{bmatrix}
d_{1,1}^f & d_{1,2}^f & \cdots & d_{1,N-1}^f & d_{1,N}^f \\
d_{2,1}^f & d_{2,2}^f & \cdots & d_{2,N-1}^f & d_{2,N}^f \\
\vdots & \vdots & & \vdots & \vdots \\
d_{M-1,1}^f & d_{M-1,2}^f & \cdots & d_{M-1,N-1}^f & d_{M-1,N}^f \\
d_{M,1}^f & d_{M,2}^f & \cdots & d_{M,N-1}^f & d_{M,N}^f
\end{bmatrix}
$$

---

### 8.2 隐藏层梯度

第 $i$ 隐藏层第 $n$ 个节点的第 $m$ 个分量 $w_{nm}^i$ 的梯度 $d_{nm}^i$：

$$
d_{nm}^i = \frac{\partial L}{\partial A_n^f} \frac{\partial A_n^f}{\partial S_n^f} \frac{\partial S_n^f}{\partial \omega_n^f} \frac{\partial \omega_n^f}{\partial w_{nm}^{f-1}} \frac{\partial \omega_n^{f-1}}{\partial w_{nm}^{f-2}} \cdots \frac{\partial \omega_n^{i+1}}{\partial w_{nm}^{i}}
$$

同样构成类似的梯度矩阵 $D^i$。

---

## 9. 参数优化

用梯度下降法对第 $i$ 层全体参数矩阵 $W^i$ 进行优化：

$$
W_n^i = W_{n-1}^i - \alpha D_{n-1}^i
$$

---

## 10. 反向传播

那么也就是说，所谓向后传播，就是上面推的那样。而且每次计算参数，我们可以根据第 $i$ 层的结果，继续求解第 $i-1$ 层的梯度，这也相当于在下一层基础上，计算上一层梯度矩阵。

## 11. 张量网络视角

注意到映射 $\omega$ 满足：

$$
\omega(\omega_n^i, w_{nm'}^{i+1}) = \sum_m \omega_n^i w_{nm}^{i+1}, \quad \text{s.w. } \omega_n^{i+1}
$$

对 $N$ 层神经网络，若将第 $i$ 层所有 node 的参数写成二阶张量 $W_{\alpha_i, \beta_i}^i$，样本写成二阶张量 $X_{ij}$（全连接形 $X_{ij} = [x, x, x, x, x, x, \dots, x]$，也可以自然推广到任意阶张量），容易发现整个网络出现了：

$$
X W^1 W^2 W^3 \cdots W^N = Y
$$

的形式——一维 MPS 形式。其中每一层 node 连同每个节点内参数个数，构成了 $\alpha_i, \beta_i$ 的维度，即 bond dimension。

整个神经网络，即一个 MPO，将态 $X$ 映射为另一个态 $Y$ —— 即预测的结果。

---

### 11.1 符号对应关系

| 神经网络概念 | 张量网络概念 |
|--------------|--------------|
| 输入样本 $X$ | 初始量子态（或边界态） |
| 第 $i$ 层权重 $W^i$ | 局域张量（MPO 的 site tensor） |
| 第 $i$ 层 node 数 $N_i$ | 物理指标维度 |
| 节点内参数个数 $M_i$ | bond 维度（纠缠自由度） |
| 输出 $Y$ | 末态（测量结果） |
| 准内积 $\omega$ | 张量缩并（tensor contraction） |


### 11.2 物理含义

- **bond dimension** 对应网络的"表达能力"或"纠缠容量"
- 增加每层 node 数或参数个数，等价于增大 bond dimension，从而提升网络的表示能力
- 深度 $N$ 对应 MPS 的链长
- 训练过程等价于通过梯度下降优化 MPO 的局域张量，使输出态 $Y$ 逼近目标态

### 11.3 推广

该形式可自然推广至：
- **高维输入**：对应更高阶的张量网络（如 PEPS）
- **卷积结构**：对应具有平移不变性的 MPO
- **残差连接**：对应 MPO 的叠加态

---

## 12. 总结

整个框架从准内积映射 $\omega$ 出发，统一了：
1. **前向传播**：函数复合（Sigmoid + Softmax）
2. **损失度量**：交叉熵
3. **反向传播**：链式法则（梯度矩阵的逐层递推）
4. **参数优化**：梯度下降
5. **结构本质**：张量网络（MPS/MPO）

神经网络 $N$ 在这一框架下，被完全刻画为一个**可微分的、参数化的张量网络映射**。