
include("/home/kzyin_norm/proj/d_dpl/nn/juliascripts/scr/dockers.jl")
include("/home/kzyin_norm/proj/d_dpl/nn/juliascripts/scr/datamapulation.jl")
include("/home/kzyin_norm/proj/d_dpl/nn/juliascripts/scr/nn_methods.jl")
using .Docker
using .DataMap
using .NN
using LinearAlgebra

# ============ 加载数据 ============
path = "../data/MNIST/raw"
train_files, test_files = load_datas(path, ["ubyte", "train", "t10k"])

datas_train_images = read(joinpath(path, train_files[1]))
datas_train_labels = read(joinpath(path, train_files[2]))

images_matrix = Matrix(renormlize(
                       bias_terms(
                           convertor_image(datas_train_images)
                       )
                   )')
labels_vector = 1 .+ Int.(convert_label(datas_train_labels))

# ============ 构建网络 ============
net = Network(3, [100, 400, 10], Function[ReLU, ReLU, softmax], false)
W = initial_weight(net, 28*28 + 1)



# ============ 训练设置 ============
X = images_matrix
y = labels_vector


∇_funcs = [dReLU, dReLU, nothing]

# 定义损失函数（用于 GradientDescent）
function Losstype(w)
    activations, _ = ForwardPass(X, w, net)
    return cross_entropy(activations[end], y)
end

# 定义梯度函数（用于 GradientDescent）
function ∇loss(w)
    activations, zs = ForwardPass(X, w, net)
    grads = Gradient(w, ∇_funcs, activations, zs, y, net)
    return grads
end

# ============ 开始训练 ============
println("\n" * "="^50)
println("开始训练...")
println("="^50)

lr = 1e-2
epochs = 200

final_w, history = GradientDescent(W, ∇loss, Losstype, lr, epochs)

# ============ 结果 ============
println("\n" * "="^50)
println("训练完成！")
println("初始损失: ", history[1, 2])
println("最终损失: ", history[end, 2])
println("="^50)

# 计算准确率
function accuracy(w, X, y)
    activations, _ = ForwardPass(X, w, net)
    y_pred = activations[end]
    pred_labels = [argmax(y_pred[i, :]) for i in 1:size(y_pred, 1)]
    return sum(pred_labels .== y) / length(y)
end

acc = accuracy(final_w, X, y)
println("训练准确率: ", acc * 100, "%")
