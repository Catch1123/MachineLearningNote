using Plots
function load_datas(path::String,
                    keys::Vector{String})

    files = filter(f -> endswith(f,keys[1]), readdir(path))
    train_files = filter(f -> occursin(keys[2], f), files)
    test_files = filter(f -> occursin(keys[3], f), files)
    return train_files, test_files
end

function bytes_to_int32(bytes::Vector{UInt8})
    return (Int(bytes[1])<<24 | Int(bytes[2])<<16 | Int(bytes[3])<<8 | Int(bytes[4]))
end


function convertor_image(bytes::Vector{UInt8})
    total_images = bytes_to_int32(bytes[5:8])    
    rows = bytes_to_int32(bytes[9:12])          
    cols = bytes_to_int32(bytes[13:16])        
    return reshape(bytes[17:end], ( rows * cols , total_images))
end

function convert_label(bytes::Vector{UInt8})
    total_labels = bytes_to_int32(bytes[5:8])    
    return bytes[9:end]
end


path = "/home/kzyin_norm/proj/d_dpl/nn/data/MNIST/raw"

train_files, test_files=load_datas(path,["ubyte","train","t10k"])
datas_train_images = read(joinpath(path,train_files[1]))
datas_train_labels = read(joinpath(path,train_files[2]))

images_martix = convertor_image(datas_train_images)
labels_vector = convert_label(datas_train_labels)


num = 100
test_examples = images_martix[:,num]  # Get the num-th test example
label_example = labels_vector[num]  # Get the corresponding label

image = heatmap(reshape(test_examples, (28, 28))', 
        color=:grays, 
        title="MNIST Example Image",
        xlabel="Column", 
        ylabel="Row",
        aspect_ratio=:equal)
display(image)
println("Label of the example image: ", label_example)



