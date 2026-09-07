
## ✅ To-Do List

### Phase 1: 数据加载与预处理 ✅

- [x] `src/docker.jl` - 数据加载模块
  - [x] `load_datas()` - 文件筛选
  - [x] `bytes_to_int32()` - 字节转整数
  - [x] `convertor_image()` - 图像转矩阵
  - [x] `convert_label()` - 标签转向量
- [x] `src/datamanipulation.jl` - 数据操作模块
  - [x] `bias_terms()` - 添加偏置
  - [x] `normalize_batch()` - 归一化
  


### Phase 2: 神经网络基础

- [ ] `src/nn_method.jl` - 神经网络基础
  - [x] `structure()`神经网络层级结构 
  - [x] `connect()` 连接类型
  - [x] `sigmoid, relu, softmax, tanh`基本激活函数及导数映射
  - [ ] `forward()` - 前向传播
  - [ ] 损失函数 (MSE, CrossEntropy)
  - [ ] 梯度计算
  - [x] 参数更新 (SGD, Adam)

### Phase 3: Transformer 模型架构

- [ ] `src/transformer.jl` - Transformer 模型
  - [ ] `MultiHeadAttention` - 多头注意力
  - [ ] `PositionalEncoding` - 位置编码
  - [ ] `TransformerEncoder` - 编码器
  - [ ] `TransformerDecoder` - 解码器
  - [ ] `Transformer` - 完整模型
  - [ ] `forward()` - 前向传播

### Phase 4: 训练与测试

- [ ] `src/train.jl` - 训练循环
  - [ ] `train_epoch()` - 单轮训练
  - [ ] `train!()` - 完整训练流程
  - [ ] `save_model()` - 保存模型参数
  - [ ] `load_model()` - 加载模型参数
- [ ] `src/test.jl` - 测试与评估
  - [ ] `test()` - 模型测试
  - [ ] `accuracy()` - 准确率计算
  - [ ] `confusion_matrix()` - 混淆矩阵
  - [ ] `evaluate()` - 完整评估

### Phase 5: 脚本与集成

- [ ] `scripts/run.jl` - 主入口
  - [ ] 数据加载
  - [ ] 数据预处理
  - [ ] 模型创建
  - [ ] 训练
  - [ ] 测试
  - [ ] 可视化
- [ ] `scripts/train.jl` - 训练脚本
- [ ] `scripts/test.jl` - 测试脚本
- [ ] `scripts/visualize.jl` - 可视化脚本

### Phase 6: 文档与部署

- [ ] `README.md` - 项目说明
- [ ] `Project.toml` - 依赖管理
- [ ] 示例笔记本 (Jupyter/Pluto)

---

## 📝 Project.toml

```toml
name = "DPL_MNIST"
uuid = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
authors = ["KZ Yin"]
version = "0.1.0"

[deps]
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
BSON = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"