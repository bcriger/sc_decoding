#=
stabiliser_sampling.py is too slow, I'm going to check if Julia is faster.
=#
module stabiliser_sampling

using Iterators

export weight_dist

function weight_dist(stab_gens::Array{IntSet,1}, log::IntSet, coset_rep::IntSet)
    coset_log = symdiff(coset_rep, log)

    weights = zeros(Int64,50)

    for set in subsets(stab_gens)
        if set==[]
            idx = length(coset_log) + 1
            weights[idx] = weights[idx] + 1
        else
            stab = reduce(symdiff, set)
            idx = length(symdiff(stab, coset_log)) + 1 #one-countiiiing
            weights[idx] = weights[idx] + 1
        end
    end
    return weights
end

function weight_dist(stab_gens::Array{Int64,2}, log::Array{Int64,1}, coset_rep::Array{Int64,1})
    #=
        Using dense vectors/matrices to store 
    =#
    nq = size(log)[1]
    ng = size(stab_gens)[2] # number of generators
    weights = zeros(Int64, nq + 1)
    vecs = CartesianRange(CartesianIndex((0 for )))

end