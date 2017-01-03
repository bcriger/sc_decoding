import numpy as np
from scipy.special import binom

d_1 = lambda c1, c2: abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

def num_paths(start, end):
    """
    takes two tuples (x, y) and returns the number of square lattice
    minimum-length paths which go from start to end.
    """
    dx, dy = abs(start[0] - end[0]), abs(start[1] - end[1])
    return binom(dx + dy, dx)

def qubit_prob(a, b, nb_1, nb_2):
    """
    The probability of a qubit being on a syndrome path is given by 
    the fraction of paths which contain both neighbours of that qubit.
    """
    # re-order neighbours based on dist to a/b
    
    n_s, n_e = nb_1, nb_2 if d_1(nb_1, a) < d_1(nb_2, a) else nb_2, nb_1
    
    return num_paths(a, n_s) * num_paths (b, n_e) / num_paths(a, b)

def bbox_p_mat(crd_0, crd_1, mdl, sz):
    """
    Given a pair of co-ordinates, an iid (but not xz) error model, and
    a size, I'd like to generate a p_mat that handles all the qubits 
    inside the box.
    These boxes are disjoint, so I can sum when I'm done. 
    Then I'll do the fancy inverse and neg-log-odds.
    """
    
    pass

#---------------------------------------------------------------------#