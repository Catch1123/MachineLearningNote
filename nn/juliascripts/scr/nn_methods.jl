#定义网络框架结构
struct Network
    layers::Int #网络层数
    nodes::Vector{Int} #每层node个数
    actives::Vector{String} #每层激活函数类型
    transfomer::Bool = false #要不要子注意力机制


    function Network(layers::Int, nodes::Vector{Int},actives::Vector{String})
        @assert layers == length(actives) "The number of layers have to equal to the number of actives!"
        @assert all(nodes .> 0)  "The nodes per layer have large than zero!"
        return new(layers,nodes,actives,false)
    end
end


function connect(topolog::Network)
    # 直接初始化
end

function connect(topolog::Network, conn_form::String = "dense")
    #nn连接方式
end

net = Network(3,[2,2,1],["a","b","c"])