"""
All rights reserved. This file is part of the DPL_MNIST project,
which is released under the MIT license.

Author: KZ Yin
"""


module Docker

export load_datas, bytes_to_int32, convertor_image, convert_label

"""
    load_datas(path, keys)

Filter files by suffix and keywords.

# Arguments
- `path::String`: Directory path
- `keys::Vector{String}`: [suffix, train_keyword, test_keyword]

# Returns
- `train_files`, `test_files`: Lists of matching filenames
"""
function load_datas(path::String,
                    keys::Vector{String})
                    
    # filter the files in the directory based ont he 
    files = filter(f -> endswith(f,keys[1]), readdir(path))
    train_files = filter(f -> occursin(keys[2], f), files)
    test_files = filter(f -> occursin(keys[3], f), files)
    return train_files, test_files
end



"""
    bytes_to_int32(bytes)

Convert 4 bytes in big-endian order to a 32-bit integer.

# Arguments
- `bytes::Vector{UInt8}`: Length-4 byte array

# Returns
- `Int`: Converted integer value
"""
function bytes_to_int32(bytes::Vector{UInt8})
    return (Int(bytes[1])<<24 | Int(bytes[2])<<16 | Int(bytes[3])<<8 | Int(bytes[4]))
end




"""
    convertor_image(bytes)

Convert MNIST image IDX format to image matrix.

# Arguments
- `bytes::Vector{UInt8}`: Raw byte array of MNIST image file

# Returns
- `Matrix{UInt8}`: Shape `(784, total_images)`, each column is a flattened 28×28 image
"""
function convertor_image(bytes::Vector{UInt8})
    total_images = bytes_to_int32(bytes[5:8])    
    rows = bytes_to_int32(bytes[9:12])          
    cols = bytes_to_int32(bytes[13:16])        
    return reshape(bytes[17:end], ( rows * cols , total_images))
end

"""
    convert_label(bytes)

Convert MNIST label IDX format to a vector of integers.

# Arguments
- `bytes::Vector{UInt8}`: Raw byte array of MNIST label file

# Returns
- `Vector{UInt8}`: Vector of label values
"""
function convert_label(bytes::Vector{UInt8})
    total_labels = bytes_to_int32(bytes[5:8])    
    return bytes[9:end]
end

# The following code block is for testing the main functions 
# of the module. It loads MNIST data, converts it to image and 
# label formats, and visualizes a sample image with its corresponding label.

if abspath(PROGRAM_FILE) == @__FILE__
# TESTING THE MAIN FUNCTIONS
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
        xlabel="Column", 
        ylabel="Row",
        aspect_ratio=:equal)
    display(image)
    println("Label of the example image: ", label_example)

end

end  # module Docker    


