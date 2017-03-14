import numpy as np
from scipy.special import binom
import decoding_2d as dc
import itertools as it
import networkx as nx
from operator import mul
import sparse_pauli as sp

shifts = [(1, -1), (-1, -1), (1, 1), (-1, 1)] # data to ancilla
big_shifts = [(2, -2), (-2, -2), (2, 2), (-2, 2)] # ancilla to ancilla

#-----------------stuff that works for square bboxes------------------#

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

#---------------------------------------------------------------------#

#------------generic DiGraph algorithms for clipped bboxes------------#

def crds_to_digraph(crd_0, crd_1, vertices):
    """
    In order to count paths on some digraph, we first have to make
    that graph.
    """
    crnrs = map(np.array, dc.corners(crd_0, crd_1))
    deltas = [crnr - crd_0 for crnr in crnrs]

    g = nx.DiGraph()

    if any(np.all(elem == np.array([0,0])) for elem in deltas):
        delta = [_ for _ in deltas if not(np.all(_ == np.array([0, 0])))][0]
        n = abs(delta[0]) / 2 # ancilla-ancilla steps
        step = delta / n 
        for j in range(n):
            g.add_edge(tuple(crd_0 + j * step), tuple(crd_0 + (j + 1) * step))
        return g
    
    n_0 = abs(deltas[0][0]) / 2 # ancilla-ancilla steps
    step_0 = deltas[0] / n_0

    n_1 = abs(deltas[1][0]) / 2 # ancilla-ancilla steps
    step_1 = deltas[1] / n_1

    ancs_in_bbox = []
    for p in it.product(range(n_0 + 1), range(n_1 + 1)):
        new_anc = tuple(crd_0 + p[0] * step_0 + p[1] * step_1)
        if not(new_anc in ancs_in_bbox) and (new_anc in vertices):
            ancs_in_bbox.append(new_anc)

    g.add_nodes_from(ancs_in_bbox)
    for v, w in it.product(ancs_in_bbox, repeat=2):
        diff = (w[0] - v[0], w[1] - v[1])
        if diff in map(tuple, [step_0, step_1]):
            g.add_edge(v, w)

    return g

def num_paths_forward(g, v=None):
    """
    Recursive function that defines the forward path-counting
    algorithm.
    If you ever need to make this faster, you can check for values in
    the cache before recursing. 
    """
    if v is None: # node labels can cast to False, so don't do not(v).
        no_kids = [n for n in g.nodes() if list(g.successors(n)) == []]
        return sum([num_paths_forward(g, w) for w in no_kids])

    # base case
    if list(g.predecessors(v)) == []:
        g.node[v]['f_paths'] = 1
        return 1
    else:
        n_paths = sum([num_paths_forward(g, w) for w in g.predecessors(v)])
        g.node[v]['f_paths'] = n_paths
        return n_paths

def num_paths_backward(g, v=None):
    """
    ugly copypaste of the function above.
    """
    if v is None: # node labels can cast to False
        batmen = [n for n in g.nodes() if list(g.predecessors(n)) == []]
        if len(batmen) > 1:
            raise NotImplementedError("multiple source nodes")
        v = batmen[0]

    # base case
    if list(g.successors(v)) == []:
        g.node[v]['b_paths'] = 1
        return 1
    else:
        n_paths = sum([num_paths_backward(g, w) for w in g.successors(v)])
        g.node[v]['b_paths'] = n_paths
        return n_paths

def digraph_paths(g):
    """
    Given a DiGraph g, I'd like to decorate its edges with the number
    of paths from the source to the sink of the graph that traverse
    that edge. 
    """
    g_b_paths, g_f_paths = num_paths_backward(g), num_paths_forward(g)
    if g_b_paths != g_f_paths:
        raise ValueError("Graph has different number of paths in "
            "forward/backward directions: {}\n{}".format(g.nodes(),
                g.edges()))
    for e in g.edges():
        g[e[0]][e[1]]['p_path'] = float(g.node[e[0]]['f_paths'] * g.node[e[1]]['b_paths']) / g_f_paths

def path_prob(g, v=None):
    """
    Given a probability for an edge in a DAG to possess an error, 
    we can calculate the probability that a path joins the source and 
    sink node, by recursion, since:

    p_path(A, B) = sum_{v in predecessors(B)} p_path(v) * p_edge(v, B)
    
    TODO: Think up better variable names than p_path, p_edge, etc.

    """
    if v is None: # node labels can cast to False, so don't do not(v).
        no_kids = [n for n in g.nodes() if list(g.successors(n)) == []]
        return sum([path_prob(g, w) for w in no_kids])

    # base case
    if list(g.predecessors(v)) == []:
        g.node[v]['p_path'] = 1
        return 1
    else:
        total_prob = sum([path_prob(g, w) * g[w][v]['p_edge']
                                    for w in g.predecessors(v)])
        g.node[v]['p_path'] = total_prob
        return total_prob

#---------------------------------------------------------------------#

#-------------------------belief propagation, etc.--------------------#
def pauli_to_tpls(pauli, sprt):
    output_dict = {}
    # AAAAAAAAAAAAAAH
    for q in sprt:
        if q in pauli.x_set:
            if q in pauli.z_set:
                output_dict[q] = 3
            else:
                output_dict[q] = 2
        elif q in pauli.z_set:
            output_dict[q] = 1
        else:
            output_dict[q] = 0

    return output_dict.items()


def propagate_beliefs(g, n_steps):
    """
    To get decent marginal probabilities to feed in to multi-path
    matching, I'm going to try straight-forward Poulin/Chung BP. 
    """

    for _ in range(n_steps):
        # From check to qubit
        for v in _checks(g):
            #sum over local Pauli group
            check = g.node[v]['check']
            sprt = check.support()
            lpg = list(map(lambda p: pauli_to_tpls(p, sprt),
                        list(it.ifilter(lambda p: p.com(check) == g.node[v]['syndrome'],
                                sp.local_group(sprt)))))
            # print v
            for bit, pdx in it.product(sprt, range(4)):
                # print bit, pdx
                for big_p in lpg:
                    if (bit, pdx) in big_p:
                        summand = reduce(mul, [g.node[q]['mqc'][v][dx] 
                                            for q, dx in big_p if q != bit])
                        # print big_p, summand
                        g.node[v]['mcq'][bit][pdx] += summand
                # print '-------'

            # code above looks right, and sort of matches Poulin/Chung.
            # could be faster if we iterated over lpg once. 
            
            # code below wrong?
            # for elem in it.imap(lambda p: pauli_to_tpls(p, sprt), lpg):
            #     for tpl in elem:
            #         summand = reduce(mul, [g.node[q]['mqc'][v][dx] 
            #                                 for q, dx in elem if q != tpl[0]])
            #         g.node[v]['mcq'][tpl[0]][tpl[1]] += summand 
            
            # normalize
            for k in g.node[v]['mcq'].keys():
                g.node[v]['mcq'][k] /= sum(g.node[v]['mcq'][k])

        # From qubit to check
        for v in _bits(g):
            # product over neighbours not c
            chex = list(g[v].keys())
            for chek in chex:
                msgs = [g.node[_]['mcq'][v] for _ in chex if _ != chek]
                g.node[v]['mqc'][chek] = reduce(mul, msgs, g.node[v]['prior'])

            # normalize
            for k in g.node[v]['mqc'].keys():
                g.node[v]['mqc'][k] /= sum(g.node[v]['mqc'][k])

    pass # subroutine, you move me

def beliefs(g):
    """
    Takes an updated Tanner graph and outputs a dictionary taking qubit
    labels to probability distributions.
    """
    output = dict()
    
    for v in _bits(g):

        output[v] = reduce(mul,
                            (g.node[u]['mcq'][v] for u in g.neighbors(v)),
                                g.node[v]['prior'])

        output[v] /= sum(output[v])
    
    return output

def tanner_graph(stabilisers, error, mdl):
    """
    To propagate beliefs, we need to start a graph off with a set of 
    error probabilities on each qubit, and a value for the syndrome on
    each ancilla bit. 
    
    I assume each stabiliser is a sparse_pauli.Pauli.
    
    Probabilities over 1-bit Paulis are ordered I, Z, X, Y.

    """
    g = nx.Graph()
    for dx, s in enumerate(stabilisers):
        label = 's' + str(dx)
        g.add_node(
                    label,
                    check=s,
                    syndrome=error.com(s),
                    mcq={b: np.zeros_like(mdl) for b in s.support()},
                    partition='c'
                    )
        
        for bit in s.support():
            g.add_node(bit, prior=np.array(mdl), partition='b')
            g.add_edge(label, bit)

    for v in _bits(g):
        g.node[v]['mqc'] = {c: np.array(mdl) for c in g[v].keys()}
    
    return g

def _checks(graph):
    """
    NX doesn't have bipartite tools in automatically, so here are some
    hax.
    """
    return (n for n, d in graph.nodes(data=True)
                            if d['partition'] == 'c')

def _bits(graph):
    """
    NX doesn't have bipartite tools in automatically, so here are some
    hax.
    """
    return (n for n, d in graph.nodes(data=True)
                            if d['partition'] == 'b')

#---------------------------------------------------------------------#

#-------------------------------main things---------------------------#
def digraph_to_mat(g, vertices):
    digraph_paths(g) #subroutine
    p_mat = np.zeros((len(vertices), len(vertices)))
    for u, v in g.edges():
        r, c = vertices.index(u), vertices.index(v)
        edge_val = g[u][v]['p_path']
        p_mat[r, c] = edge_val
        p_mat[c, r] = edge_val
    return p_mat

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
    g = crds_to_digraph(crd_0, crd_1, vertices)
    return digraph_to_mat(g, vertices)

def bdy_p_v_mat(crd, bdy_verts, bulk_verts):
    """
    There are multiple shortest-length paths to the boundary.
    Pretty much as soon as you start trying to use re-weighting, you
    notice that the decoder is certain of a violation where in fact
    none has occurred.
    In order to determine where the errors are, we have to take
    multiple paths to the boundary into account. 
    We derive the probability for a violating error to be on a qubit in
    the case that a boundary point is selected at random, and then a
    path leading to that boundary point. 
    """
    vertices = sorted(bdy_verts + bulk_verts)
    bdy_dists = [dc.pair_dist(crd, v) for v in bdy_verts]
    d = min(bdy_dists)
    close_pts = [bdy_verts[_] for _ in range(len(bdy_verts))
                                            if bdy_dists[_] == d]
    g = reduce(nx.compose,
                [crds_to_digraph(crd, pt, vertices)
                    for pt in close_pts])

    return digraph_to_mat(g, vertices)

def matching_p_mat(match_lst, bdy_verts, bulk_verts, mdl, new_err):
    """
    lil wrapper for what's above, we first sum up all the p_v matrices
    from individual edges, then convert these to probabilities of
    error.
    By the time we get these vertices in the matching, the 
    boundary-boundary edges are gone, and everything's been converted
    to co-ordinates.
    """
    vertices = sorted(bdy_verts + bulk_verts)
    p_mat = np.zeros((len(vertices), len(vertices)))
    #TODO Check for overlap, bboxes are not supposed to share vertices.
    for thing in match_lst:
        if hasattr(thing[0], "__iter__"):
            # it's a list of two tuples
            new_mat = bbox_p_v_mat(thing[0], thing[1], vertices)
        elif len(thing) == 2:
            # it's a length-2 tuple
            new_mat = bdy_p_v_mat(thing, bdy_verts, bulk_verts)
    
        if (set(zip(*p_mat.nonzero())) & set(zip(*new_mat.nonzero()))):
            raise ValueError("bboxes can overlap, goto drawing board;")
        else:
            p_mat += new_mat

    #TODO Not all elements in the adj mat are edges. Many should get 0 prob, not p_v=0.
    for r, c in it.product(range(len(vertices)), repeat=2):
        if (vertices[r][0] - vertices[c][0], vertices[r][1] - vertices[c][1]) in big_shifts:
            p_mat[r, c] = p_v_prime(p_mat[r, c], mdl, new_err)

    return p_mat - np.diag(np.diag(p_mat)) #no self-loops

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