using SimpleGF2

using PyCall
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport gf2_mat

"""
I'm going to solve all 3-by-4 systems and see if Python spits out the
same answers.
"""

lo = CartesianIndex((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
hi = CartesianIndex((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))

open("solved_system_table", "w") do phil
    for idx in CartesianRange(lo,hi)
        mat = copy(reshape(collect(idx.I), (3,4)))
        try
            println(gf2_mat.solve_augmented(mat) == Array{Int}(solve_augmented(Array{GF2}(mat))))
        catch
        end
    end
end
