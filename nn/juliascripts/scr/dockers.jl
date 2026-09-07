module Docker

export load_datas, bytes_to_int32, convertor_image, convert_label

function load_datas(path::String, keys::Vector{String})
    files = filter(f -> endswith(f, keys[1]), readdir(path))
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
    return reshape(bytes[17:end], (rows * cols, total_images))
end

function convert_label(bytes::Vector{UInt8})
    total_labels = bytes_to_int32(bytes[5:8])
    return bytes[9:end]
end

end  # module Docker