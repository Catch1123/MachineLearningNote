using Statistics

"""
    bias_terms(X::Matrix)

Add a bias row of ones to the input matrix.

# Arguments
- `X::Matrix`: Input matrix of size `(d, N)`

# Returns
- `Matrix`: Augmented matrix of size `(d+1, N)`
"""

function bias_terms(X::Matrix)
        return vcat(X,ones(1,size(X,2)))
end


"""
    renormlize(X::Matrix, methods::String="meanstd")

Normalize the matrix using either mean-std or min-max scaling.

# Arguments
- `X::Matrix`: Input matrix
- `methods::String`: "meanstd" or "minmax"

# Returns
- `Matrix`: Normalized matrix
"""
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


