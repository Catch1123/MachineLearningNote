include("/home/kzyin_norm/proj/d_dpl/nn/juliascripts/scr/dockers.jl")
include("/home/kzyin_norm/proj/d_dpl/nn/juliascripts/scr/datamapulation.jl")
include("/home/kzyin_norm/proj/d_dpl/nn/juliascripts/scr/nn_methods.jl")

using .Docker
using .DataMap
using .NN

using LinearAlgebra
using Plots


# ============================================================
# 加载数据
# ============================================================

path = "../data/MNIST/raw"

train_files, test_files = load_datas(
    path,
    ["ubyte", "train", "t10k"]
)


# ============================================================
# 训练集
# ============================================================

datas_train_images = read(
    joinpath(path, train_files[1])
)

datas_train_labels = read(
    joinpath(path, train_files[2])
)

images_matrix = Matrix(
    renormlize(
        bias_terms(
            convertor_image(datas_train_images)
        )
    )'
)

labels_vector = 1 .+ Int.(
    convert_label(datas_train_labels)
)


# ============================================================
# 测试集
# ============================================================

datas_test_images = read(
    joinpath(path, test_files[1])
)

datas_test_labels = read(
    joinpath(path, test_files[2])
)

test_images_matrix = Matrix(
    renormlize(
        bias_terms(
            convertor_image(datas_test_images)
        )
    )'
)

test_labels_vector = 1 .+ Int.(
    convert_label(datas_test_labels)
)


println("训练集大小: ", size(images_matrix, 1))
println("测试集大小: ", size(test_images_matrix, 1))


# ============================================================
# 构建网络
# ============================================================

net = Network(
    5,
    [400, 400,600,300, 10],
    Function[ReLU,ReLU,ReLU, ReLU,softmax],
    false
)

W = initial_weight(
    net,
    28 * 28 + 1
)


# ============================================================
# 训练设置
# ============================================================

X = images_matrix
y = labels_vector

X_test = test_images_matrix
y_test = test_labels_vector


# 每一层对应的激活函数导数
∇_funcs = [
    dReLU,
    dReLU,
    dReLU,
    dReLU,
    nothing
]


# ============================================================
# Loss function
# ============================================================

function Losstype(w)

    activations, _ = ForwardPass(
        X,
        w,
        net
    )

    return cross_entropy(
        activations[end],
        y
    )
end


# ============================================================
# Gradient
# ============================================================

function ∇loss(w)

    activations, zs = ForwardPass(
        X,
        w,
        net
    )

    grads = Gradient(
        w,
        ∇_funcs,
        activations,
        zs,
        y,
        net
    )

    return grads
end


# ============================================================
# Accuracy
# ============================================================

function accuracy(w, X, y)

    activations, _ = ForwardPass(
        X,
        w,
        net
    )

    y_pred = activations[end]

    pred_labels = [
        argmax(y_pred[i, :])
        for i in 1:size(y_pred, 1)
    ]

    return sum(
        pred_labels .== y
    ) / length(y)
end


# ============================================================
# 开始训练
# ============================================================

println("\n" * "="^50)
println("开始训练...")
println("="^50)

lr = 1e-2
epochs = 100

final_w, history = GradientDescent(
    W,
    ∇loss,
    Losstype,
    lr,
    epochs
)


# ============================================================
# 训练结果
# ============================================================

println("\n" * "="^50)
println("训练完成！")
println("="^50)

println("初始损失: ", history[1, 2])
println("最终损失: ", history[end, 2])

train_acc = accuracy(
    final_w,
    X,
    y
)

test_acc = accuracy(
    final_w,
    X_test,
    y_test
)

println("训练准确率: ", train_acc * 100, "%")
println("测试准确率: ", test_acc * 100, "%")
println(
    "泛化差距: ",
    (train_acc - test_acc) * 100,
    "%"
)


# ============================================================
# 可视化
# ============================================================

println("\n" * "="^50)
println("生成可视化...")
println("="^50)


# ------------------------------------------------------------
# 1. Training Loss
# ------------------------------------------------------------

fig1 = plot(
    history[:, 1],
    history[:, 2],

    xlabel = "Epoch",
    ylabel = "Loss",

    title = "Training Loss",

    label = false,

    lw = 2,

    grid = true
)

display(fig1)

savefig(
    fig1,
    "loss_curve.png"
)

println("✅ 损失曲线已保存: loss_curve.png")


# ------------------------------------------------------------
# 2. Test Predictions
# ------------------------------------------------------------

function show_predictions(
    w,
    X,
    y,
    net,
    n = 10
)

    # 防止 n 超过测试集大小
    n = min(
        n,
        size(X, 1)
    )


    # -----------------------------
    # Forward pass
    # -----------------------------

    activations, _ = ForwardPass(
        X[1:n, :],
        w,
        net
    )

    y_pred = activations[end]


    # -----------------------------
    # Prediction
    # -----------------------------

    pred_labels = [
        argmax(y_pred[i, :]) - 1
        for i in 1:n
    ]

    true_labels = y[1:n] .- 1


    # -----------------------------
    # 每张图片单独创建 heatmap
    # -----------------------------

    plots = []

    for i in 1:n

        # X 的最后一列是 bias
        img = reshape(
            X[i, 1:end-1],
            28,
            28
        )

        p = heatmap(
            img,

            color = :grays,

            cbar = false,

            axis = false,

            aspect_ratio = :equal,

            title = "True: $(true_labels[i])  Pred: $(pred_labels[i])",

            titlefontsize = 10
        )

        push!(
            plots,
            p
        )
    end


    # -----------------------------
    # 组合成 2 × 5
    # -----------------------------

    return plot(
        plots...,

        layout = (2, 5),

        size = (1000, 450)
    )
end


fig2 = show_predictions(
    final_w,
    X_test,
    y_test,
    net,
    10
)

display(fig2)

savefig(
    fig2,
    "test_predictions.png"
)

println(
    "✅ 测试集预测结果已保存: test_predictions.png"
)


# ============================================================
# 完成
# ============================================================

println("\n" * "="^50)
println("🎉 可视化完成！")
println("="^50)

println("生成文件:")
println("  loss_curve.png")
println("  test_predictions.png")