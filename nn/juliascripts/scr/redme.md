# 从零实现神经网络 / Neural Networks from Scratch

**A from-scratch implementation of neural networks in Julia.**

本项目使用 Julia 从零实现神经网络。

本项目的目的并非替代成熟的机器学习框架，而是通过亲自实现神经网络的核心算法，深入理解其背后的数学原理与计算机制。

The purpose of this project is not to replace mature machine learning frameworks, but to develop a deeper understanding of the mathematical and computational mechanisms underlying neural networks through explicit implementation.

在实现过程中，我刻意避免使用自动微分（Automatic Differentiation）以及高层神经网络抽象，而是从底层计算出发，显式实现前向传播、Softmax、交叉熵损失、反向传播以及梯度下降等核心算法。

The implementation deliberately avoids automatic differentiation and high-level neural-network abstractions. Instead, the core algorithms—including forward propagation, softmax, cross-entropy loss, backpropagation, and gradient descent—are implemented explicitly from the underlying mathematical formulation.
