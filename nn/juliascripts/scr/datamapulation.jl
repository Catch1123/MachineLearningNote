module DataMap

using Statistics
export  bias_terms, renormlize




function bias_terms(X::Matrix)
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


end