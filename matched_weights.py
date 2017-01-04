import numpy as np
from scipy.special import binom
import decoding_2d as dc
import itertools as it

shifts = [(1, -1), (-1, -1), (1, 1), (-1, 1)] #data to ancilla

d_1 = lambda c1, c2: abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

def num_paths(start, end):
    """
    takes two tuples (x, y) and returns the number of square lattice
    minimum-length paths which go from start to end.
    """
    dx, dy = abs(start[0] - end[0]), abs(start[1] - end[1])
    return binom(dx + dy, dx)

def qubit_prob(a, b, nb_0, nb_1):
    """
    The probability of a qubit being on a syndrome path is given by 
    the fraction of paths which contain both neighbours of that qubit.
    """

    # re-order neighbours based on dist to a/b
    n_s, n_e = (nb_0, nb_1) if d_1(nb_0, a) < d_1(nb_1, a) else (nb_1, nb_0)
    
    return num_paths(a, n_s) * num_paths (b, n_e) / num_paths(a, b)

def p_v_prime(p_v, mdl, new_err):
    """
    Given a probability p_v that an error violating a given stabiliser
    is on the lattice, we calculate the probability that an error
    violating the other kind of stabiliser is on the same point.  
    """
    p_I, p_Z, p_X, p_Y = mdl
    new_err = new_err.lower()
    if new_err == 'x':
        return p_v * p_Y / (p_Z + p_Y) + (1. - p_v) * p_X / (p_I + p_X)
    elif new_err == 'z':
        return p_v * p_Y / (p_X + p_Y) + (1. - p_v) * p_Z / (p_I + p_Z)
    else:
        raise ValueError("new_err = {}".format(new_err))

def bbox_p_v_mat(crd_0, crd_1, vertices):
    """
    Given a pair of co-ordinates, an iid (but not xz) error model, and
    a size, I'd like to generate a p_mat that handles all the qubits 
    inside the box.
    These boxes are disjoint, so I can sum when I'm done. 
    Then I'll do the fancy inverse and neg-log-odds.

    Note that this function does a lot of stupid stuff, this is because
    of the rotated co-ordinate system.
    """
    vertices = sorted(vertices)
    p_mat = np.zeros((len(vertices), len(vertices)))
    crd_0, crd_1 = map(np.array, [crd_0, crd_1])
    crnrs = map(np.array, dc.corners(crd_0, crd_1))
    #I want the dist from the start to the corners
    deltas = [crnr - crd_0 for crnr in crnrs]
    
    # We're set up to handle paths with corners. What about straight
    # lines?
    if any(np.all(elem == np.array([0,0])) for elem in deltas):
        delta = [_ for _ in deltas if not(np.all(_ == np.array([0,0])))][0]
        n = abs(delta[0]) / 2 # ancilla-ancilla steps
        step = delta / n 
        for idx in range(n):
            p_mat[crd_0 + idx * step, crd_0 + (idx + 1) * step] = 1.

        return p_mat

    n_0 = abs(deltas[0][0]) / 2 # ancilla-ancilla steps
    step_0 = deltas[0] / n_0

    n_1 = abs(deltas[1][0]) / 2 # ancilla-ancilla steps
    step_1 = deltas[1] / n_1

    ancs_in_bbox = []
    for p in it.product(range(n_0 + 1), range(n_1 + 1)):
        new_anc = tuple(crd_0 + p[0] * step_0 + p[1] * step_1)
        if not(new_anc in ancs_in_bbox):
            ancs_in_bbox.append(new_anc)
    
    # this is dodgy, I'm going to invert a 2-by-2 matrix to get
    # co-ordinates
    step_mat = np.matrix(np.vstack([step_0, step_1]).T).I
    b = tuple(map(int, np.rint(step_mat * np.matrix(crd_1 - crd_0).T)))
    for pair in it.product(map(np.array, ancs_in_bbox), repeat=2):
        delta = pair[1] - pair[0]
        if np.all(delta == step_0) or np.all(delta == step_1):
            d_0, d_1 = pair[0] - crd_0, pair[1] - crd_0
            nb_0 = map(int, np.rint(step_mat * np.matrix(d_0).T))
            nb_1 = map(int, np.rint(step_mat * np.matrix(d_1).T))
            p_v = qubit_prob((0, 0), b, nb_0, nb_1)
            p_mat[vertices.index(tuple(pair[0])),
                  vertices.index(tuple(pair[1]))] = p_v        
    
    return p_mat

def matching_p_mat(match_lst, vertices, mdl, new_err):
    """
    lil wrapper for what's above, we first sum up all the p_v matrices
    from individual edges, then convert these to probabilities of
    error.
    By the time we get these vertices in the matching, the 
    boundary-boundary edges are gone, and everything's been converted
    to co-ordinates.
    """
    vertices = sorted(vertices)
    p_mat = np.zeros((len(vertices), len(vertices)))

    for pair in match_lst:
        p_mat += bbox_p_v_mat(pair[0], pair[1], vertices)

    for r, c in it.product(range(len(vertices)), repeat=2):
        p_mat[r, c] = p_v_prime(p_mat[r, c], mdl, new_err)

    return p_mat

def nn_edge_switch(crds):
    """
    In order to make a metric, I have to express a pair of adjacent X
    ancillas as a pair of adjacent Z ancillas which are next to the
    same qubit.
    This can be accomplished by transposing the co-ordinates. 
    For example, in the distance 5 layout, the adjacent Z ancillas
    at (2, 6) and (4, 8) are next to the same qubit as the X ancillas
    at (2, 8) and (4, 6). 
    This does not hold for long paths, so we have to perform this
    operation before doing the inversion trick. 
    """
    return ((crds[0][0], crds[1][1]), (crds[1][0], crds[0][1]))

#---------------------------------------------------------------------#