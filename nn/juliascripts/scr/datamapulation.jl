include("dockers.jl")
using  .Docker
using Statistics

using Plots
path = "../data/MNIST/raw"

train_files, test_files=load_datas(path,["ubyte","train","t10k"])
datas_train_images = read(joinpath(path,train_files[1]))
datas_train_labels = read(joinpath(path,train_files[2]))

images_matrix = convertor_image(datas_train_images)
labels_vector = convert_label(datas_train_labels)

num = 1
    test_examples = images_matrix[:,num]  # Get the num-th test example
    label_example = labels_vector[num]  # Get the corresponding label

    image = heatmap(reshape(test_examples, (28, 28))', 
        color=:grays, 
        title="MNIST Example Image",
        yflip=true,
        xlabel="Column", 
        ylabel="Row",
        aspect_ratio=:equal)
    display(image)
    println("Label of the example image: ", label_example)

typeof(images_matrix[:,2])
size(images_matrix)
size(labels_vector)


# 数据预处理，添加偏置项 bais terms

function bais_terms(X::Matrix)
        return vcat(X,ones(1,size(X,2)))
end

function renormlize(X::Matrix, methods::String = "meanstd")
        if methods == "meanstd"
            X_mean = mean(X)
            X_std = std(X)
            return (X .- X_mean) ./ X_std
        elseif methods == "minmax"
            X_min = minimum(X)
            X_max = maximum(X)
            return (X .- X_min) ./ (X_max - X_min)
        end
end

bais_images_matrix = bais_terms(images_matrix)
normalized_bais_images_matrix = renormlize(bais_images_matrix)

num = 1



image = heatmap(reshape(normalized_bais_images_matrix[:,num][1:784],28,28)', 
                        color=:grays,
                                yflip=true, 
                                        title = "label: $(labels_vector[num])")
display(image)