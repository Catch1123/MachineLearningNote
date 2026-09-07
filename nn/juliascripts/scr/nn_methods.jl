module NN

export Network, ReLU, dReLU, softmax, initial_weight, ForwardPass, 
       cross_entropy, Gradient, GradientDescent

using Random
using Printf 
using Statistics
using LinearAlgebra
# ============ 网络结构 ============
struct Network
    layers::Int
    nodes::Vector{Int}
    actives::Vector{Function}
    transfomer::Bool

    function Network(layers::Int, nodes::Vector{Int}, actives::Vector{Function}, transfomer::Bool)
        @assert layers == length(actives) "The number of layers have to equal to the number of actives!"
        @assert all(nodes .> 0) "The nodes per layer have large than zero!"
        @assert nodes[end] == 10 "The output nodes of MNIST must be 10!"
        return new(layers, nodes, actives, false)
    end
end

# ============ 激活函数 ============
function ReLU(x)
    return max.(0, x)
end

function dReLU(x)
    return Float64.(x .> 0)
end

function softmax(x)
    exp_x = exp.(x .- maximum(x, dims=2))
    return exp_x ./ sum(exp_x, dims=2)
end

# ============ 权重初始化 ============
function initial_weight(topology::Network, sample_dim::Int)
    weights = Vector{Matrix{Float64}}()
    nodes = topology.nodes
    
    for l in 1:(topology.layers)
        if l == 1
            # He 初始化 (ReLU 专用)
            scale = sqrt(2.0 / sample_dim)
            push!(weights, randn(sample_dim, nodes[l]) * scale)
        else
            scale = sqrt(2.0 / nodes[l-1])
            push!(weights, randn(nodes[l-1], nodes[l]) * scale)
        end
    end
    return weights
end

# ============ 前向传播 ============
function ForwardPass(sample::AbstractVector, w::AbstractArray, topology::Network)
    A = sample
    for l in 1:topology.layers
        A = topology.actives[l](A * w[l])
    end
    return A
end

function ForwardPass(samples::AbstractMatrix, w::AbstractArray, topology::Network)
    activations = [samples]
    zs = []
    A = samples
    for l in 1:topology.layers
        z = A * w[l]
        push!(zs, z)
        A = topology.actives[l](z)
        push!(activations, A)
    end
    return activations, zs
end

# ============ 损失函数 ============
function cross_entropy(y_pred::Matrix{Float64}, y_true::Vector{Int})
    n = size(y_pred, 1)
    idx = CartesianIndex.(1:n, y_true)
    loss = -sum(log.(y_pred[idx] .+ 1e-15)) / n
    return loss
end

# ============ 反向传播 ============
function Gradient(w, ∇, activations, zs, y, topology::Network)
    la = topology.layers
    gradients = []
    
    # 输出层 δ (交叉熵 + softmax)
    y_pred = activations[end]
    n = size(y_pred, 1)
    δ = y_pred
    for i in 1:n
        δ[i, y[i]] -= 1.0
    end
    δ = δ ./ n
    
    # 从最后一层向前计算梯度矩阵
    for l in la:-1:1
        if l == la
            grad = activations[l]' * δ
        else
            δ = (δ * w[l+1]') .* ∇[l](zs[l])
            grad = activations[l]' * δ
        end
        push!(gradients, grad)
    end
    
    return reverse!(gradients)
end

# ============ 梯度下降优化器 ============
function GradientDescent(w::AbstractArray, ∇::Function, Losstype::Function, 
                         dw::Real, Iteration::Int)
    loss_history = Matrix{Float64}(undef, Iteration, 2)
    w_current = copy(w)
    for i in 1:Iteration
        grands = ∇(w_current)
        w_current = w_current - dw * grands
        loss = Losstype(w_current)
        loss_history[i, :] = [Float64(i), loss]
        grad_norm = sum([norm(g) for g in grands])
        @printf("Iter %4d: loss = %.8f, grad_norm = %.8f\n", i, loss, grad_norm)
    end
    return w_current, loss_history
end

# ============ 辅助函数 ============
function accuracy(w, X, y, topology::Network)
    activations, _ = ForwardPass(X, w, topology)
    y_pred = activations[end]
    pred_labels = [argmax(y_pred[i, :]) for i in 1:size(y_pred, 1)]
    return sum(pred_labels .== y) / length(y)
end

function predict(w, X, topology::Network)
    activations, _ = ForwardPass(X, w, topology)
    y_pred = activations[end]
    return [argmax(y_pred[i, :]) for i in 1:size(y_pred, 1)]
end

end  # module NN