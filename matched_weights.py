import numpy as np
from scipy.special import binom
import decoding_2d as dc
import itertools as it
# from line_profiler import LineProfiler
import networkx as nx
from operator import mul, add
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
    gd_pts = list(vertices) + [crd_1] # allowed support on terminal if bdy
    crnrs = map(np.array, dc.corners(crd_0, crd_1))
    deltas = [crnr - crd_0 for crnr in crnrs]

    g = nx.DiGraph()

    # straight lines
    if any(np.all(elem == np.array([0, 0])) for elem in deltas):
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
        if not(new_anc in ancs_in_bbox) and (new_anc in gd_pts):
            ancs_in_bbox.append(new_anc)

    g.add_nodes_from(ancs_in_bbox)
    for v, w in it.product(ancs_in_bbox, repeat=2):
        diff = (w[0] - v[0], w[1] - v[1])
        if diff in map(tuple, [step_0, step_1]):
            g.add_edge(v, w)

    return g

def bdy_digraph(crd, vertices, bdy_pts):
    """
    Produces the union of graphs between a point and a specified set of
    boundary points. 
    You'll have to do this with both boundaries for each side, since
    either side can be likelier now. 
    We take the union over the individual graphs between the crd of
    interest and all the closest points from the bdy_pts. 
    """
    close_pts = close_bdy_pts(crd, bdy_pts)
    return reduce(nx.compose, [crds_to_digraph(crd, pt, vertices)
                                                for pt in close_pts])

#-----------------------------path counting---------------------------#

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

#-------------------------------------path probabilities--------------#

def p_anticom(belief, stab):
    if stab == 'X':
        return belief[1] + belief[3]
    elif stab == 'Z':
        return belief[2] + belief[3]
    else:
        raise ValueError("stab {} not recognized.".format(stab))

def record_beliefs(beliefs, g, layout, stab_type):
    """
    Given a dict of beliefs on data qubits ('beliefs'), a digraph
    taking us from one coord to another ('g'), and a layout object
    ('layout'), we put the appropriate beliefs (given by p_anticom) on 
    the edges of g as 'p_edge', so that we can run 'path_prob' later. 
    """
    for edge in list(g.edges()):
        q = tuple((c0 + c1) / 2
                    for c0, c1 in zip(edge[0], edge[1])) # intdiv
        
        p = p_anticom(beliefs[layout.map[q]], stab_type)
        
        g[edge[0]][edge[1]]['p_edge'] = p

    pass

def path_prob(g, v=None):
    """
    For some reason, I don't think the function `path_prob' is
    well-motivated. 
    I don't think it's calculating the right thing. 
    I've done a derivation in ReWeighting.tex that does calculate the
    right(ish) thing, which is the odds that there's a path joining two
    vertices given that it's either that or an empty bbox. 
    """
    if v is None: # node labels can cast to False, so don't do not(v).
        no_kids = [n for n in g.nodes() if list(g.successors(n)) == []]
        o = sum([path_prob(g, w) for w in no_kids])
        return o / (o + 1.) # terrible
    elif 'o_path' in g.node[v]:
        # caching
        return g.node[v]['o_path']
    elif list(g.predecessors(v)) == []:
        return 1
    else:
        # recurse
        odds = [path_prob(g, w) 
                * g[w][v]['p_edge'] / (1. - g[w][v]['p_edge']) 
                                    for w in g.predecessors(v)]
        g.node[v]['o_path'] = sum(odds)
        return g.node[v]['o_path']


#---------------------------------------------------------------------#

#----------------------edge weights-----------------------------------#

def nlo(p):
    if (p > 1) or (p < 0):
        raise ValueError("bad p: {}".format(p))
    return -np.log(p / (1. - p))

def bdy_edge(crd, vertices, bdy_1, bdy_2, beliefs, layout, stab_type):
    """
    Produces two bdy_digraphs, then assigns a close_pt and a weight
    based on the minimum over the two.
    To do this, we have to record the beliefs onto the subgraphs, 
    calculate path_prob for each, then return the fancy_weight of the
    higher probability. 
    """
    dg_1 = bdy_digraph(crd, vertices, bdy_1)
    dg_2 = bdy_digraph(crd, vertices, bdy_2)
    record_beliefs(beliefs, dg_1, layout, stab_type)
    record_beliefs(beliefs, dg_2, layout, stab_type)
    p_1, p_2 = map(path_prob, [dg_1, dg_2])
    if p_1 > p_2:
        wt = nlo(p_1)
        close_pt = close_bdy_pts(crd, bdy_1)[0]
    elif p_2 > p_1 :
        wt = nlo(p_2)
        close_pt = close_bdy_pts(crd, bdy_2)[0]
    else:
        raise RuntimeError("Boundary weight ambiguous "
                            "(I thought this would never happen)")
    return wt, close_pt

def bulk_wt(crd_0, crd_1, vertices, beliefs, layout, stab_type):
    """
    Let's handle business for bulk edges as well, just returning the
    weight.
    """
    dg = crds_to_digraph(crd_0, crd_1, vertices)
    record_beliefs(beliefs, dg, layout, stab_type)
    p = path_prob(dg)
    
    return nlo(p)

def input_beliefs(sim, err, bp_rounds=None):
    """
    bp_rounds defaults to 5*d.
    """
    layout = sim.layout
    stabs = dict(reduce(add, [layout.stabilisers()[_].items()
                                for _ in 'XZ']))
    mdl = sim.error_model.p_arr
    tg = tanner_graph(stabs, err, mdl) # it only uses syndromes
    
    if bp_rounds is None:
        bp_rounds = 5 * max(sim.dx, sim.dy)

    propagate_beliefs(tg, bp_rounds)
    
    return beliefs(tg)

def bp_graphs(sim, err, bp_rounds=None, precision=4):
    """
    Create a pair of graphs to use in independent x/z matching by:
     - running BP on the Tanner graph
     - message-passing to create edge weights based on min-len SAWs
    """
    layout = sim.layout
    blfs = input_beliefs(sim, err, bp_rounds)
    x_synd, z_synd = sim.syndromes(err)
    
    if sim.boundary_conditions == 'closed':
        x_bdy_1, x_bdy_2, z_bdy_1, z_bdy_2 = None, None, None, None
    else:
        x_bdy_1 = layout.boundary['x_left']
        x_bdy_2 = layout.boundary['x_right']
        z_bdy_1 = layout.boundary['z_top']
        z_bdy_2 = layout.boundary['z_bot']

    x_verts, z_verts = layout.x_ancs(), layout.z_ancs()

    # There seriously has to be a better way to do this
    svb1b2t = [(x_synd, x_verts, x_bdy_1, x_bdy_2, 'X'),
                (z_synd, z_verts, z_bdy_1, z_bdy_2, 'Z')]
    out_graphs = []
    for synd, vs, bdy_1, bdy_2, tp in svb1b2t:
        graph = nx.Graph()
        
        # ancilla-to-boundary edges
        if bdy_1 is not None:
            for pt in synd:
                crd = layout.map.inv[pt]
                wt, c_pt = bdy_edge(crd, vs, bdy_1, bdy_2,
                                    blfs, layout, tp)

                # NX does MAXIMUM-weight matching ----->------->----v
                graph.add_edge(pt, (pt, 'b'), close_pt=c_pt, weight=-wt)

        # pair edges
        if bdy_1 is not None:
            for p, q in it.combinations(synd, r=2):
                # boundary-boundary
                graph.add_edge((p, 'b'), (q, 'b'), weight=0.)
        
        for p, q in it.combinations(synd, r=2):
            # bulk
            c0, c1 = layout.map.inv[p], layout.map.inv[q]
            wt = bulk_wt(c0, c1, vs, blfs, layout, tp)
            # MAXIMUM-weight matching->-v
            graph.add_edge(p, q, weight=-wt)

        # Temporary: Clean up edge weights for comparison to integer
        # if list(graph.edges()):
        #     min_wt = min([edge[2]['weight'] for edge in graph.edges(data=True)])
        #     for u, v in graph.edges():
        #         graph[u][v]['weight'] += min_wt
        #     nrm = min(filter(lambda x: x !=0,
        #         [edge[2]['weight'] for edge in graph.edges(data=True)]))
        #     for u, v in graph.edges():
        #         graph[u][v]['weight'] /= nrm    

        out_graphs.append(graph.copy())
    
    return tuple(out_graphs)

#-------------------------belief propagation, etc.--------------------#
_order = {'Z': 1, 'X': 2, 'Y': 3}
def str_sprt_to_tpl(str_sprt, sprt):
    """
    pads a sparse_pauli.Pauli with identities, converting to a tuple
    which is ordered the same as the support.
    """
    pl_lst = [0 for _ in range(len(sprt))]
    
    for char, lbl in zip(*str_sprt):
        pl_lst[ sprt.index(lbl) ] = _order[char]
    
    return tuple(pl_lst)

def propagate_beliefs(g, n_steps):
    """
    To get decent marginal probabilities to feed in to multi-path
    matching, I'm going to try straight-forward Poulin/Chung BP. 
    """

    for _ in range(n_steps):

        check_to_qubit(g)
        qubit_to_check(g)

# This big block of variables is meant to pre-compute a lot of the
# stuff that goes into check_to_qubit for surface codes at import-time.

g_14 = [{0}, {1}, {2}, {3}]
g_24 = [{0, 1}, {1, 2}, {2, 3}]
g_12 = [{0}, {1}]
g_22 = [{0, 1}]

_xxxx_com = list(sp.generated_group(g_14, g_24))
_zzzz_com = list(sp.generated_group(g_24, g_14))
_xx_com   = list(sp.generated_group(g_12, g_22))
_zz_com   = list(sp.generated_group(g_22, g_12))

_xxxx_acom = [sp.Z([0]) * p for p in sp.generated_group(g_14, g_24)]
_zzzz_acom = [sp.X([0]) * p for p in sp.generated_group(g_24, g_14)]
_xx_acom   = [sp.Z([0]) * p for p in sp.generated_group(g_12, g_22)]
_zz_acom   = [sp.X([0]) * p for p in sp.generated_group(g_22, g_12)]

def tpl_lst(pauli_lst, n_bits):
    return [str_sprt_to_tpl(p.str_sprt_pair(), range(n_bits))
                                            for p in pauli_lst]

_lpg_wrap = {
                ('XXXX', 0): tpl_lst(_xxxx_com, 4),
                ('XXXX', 1): tpl_lst(_xxxx_acom, 4),
                ('ZZZZ', 0): tpl_lst(_zzzz_com, 4),
                ('ZZZZ', 1): tpl_lst(_zzzz_acom, 4),
                ('XX', 0):   tpl_lst(_xx_com, 2),
                ('XX', 1):   tpl_lst(_xx_acom, 2),
                ('ZZ', 0):   tpl_lst(_zz_com, 2),
                ('ZZ', 1):   tpl_lst(_zz_acom, 2)
            }

# profile = LineProfiler()
# @profile
def check_to_qubit(g):
    for v in _checks(g): 
        check = g.node[v]['check']
        synd = g.node[v]['syndrome']
        sprt = check[1]
        bs = range(len(sprt))
        
        mqc = np.array([g.node[q]['mqc'][v] for q in sprt])
        
        mcq = np.zeros_like(g.node[v]['mcq'].values())
        # factor sum to avoid excess multiplications
        if check[0] == 'XXXX':
            for b in bs:
                lst = [_ for _ in bs if _ != b]
                ix1, ix2, ix3 = [mqc[_][0] + mqc[_][2] for _ in lst]
                yz1, yz2, yz3 = [mqc[_][3] + mqc[_][1] for _ in lst]
                odd_2 = yz2 * ix3 + ix2 * yz3
                even_2 = yz2 * yz3 + ix2 * ix3
                odd_3 = yz1 * even_2 + ix1 * odd_2
                even_3 = ix1 * even_2 + yz1 * odd_2
                mcq[b, 0] += odd_3 if synd == 1 else even_3
                mcq[b, 2] += odd_3 if synd == 1 else even_3
                mcq[b, 1] += even_3 if synd == 1 else odd_3
                mcq[b, 3] += even_3 if synd == 1 else odd_3
        elif check[0] == 'ZZZZ':
            for b in bs:
                lst = [_ for _ in bs if _ != b]
                iz1, iz2, iz3 = [mqc[_][0] + mqc[_][1] for _ in lst]
                xy1, xy2, xy3 = [mqc[_][2] + mqc[_][3] for _ in lst]
                odd_2 = xy2 * iz3 + iz2 * xy3
                even_2 = xy2 * xy3 + iz2 * iz3
                odd_3 = xy1 * even_2 + iz1 * odd_2
                even_3 = iz1 * even_2 + xy1 * odd_2
                mcq[b, 0] += odd_3 if synd == 1 else even_3
                mcq[b, 2] += even_3 if synd == 1 else odd_3
                mcq[b, 1] += odd_3 if synd == 1 else even_3
                mcq[b, 3] += even_3 if synd == 1 else odd_3
        elif check[0] == 'XX':
            for b in bs:
                othr = [_ for _ in bs if _ != b][0]
                ix = mqc[othr][0] + mqc[othr][2]
                yz = mqc[othr][3] + mqc[othr][1]
                mcq[b, 0] += ix if synd == 0 else yz
                mcq[b, 1] += yz if synd == 0 else ix
                mcq[b, 2] += ix if synd == 0 else yz
                mcq[b, 3] += yz if synd == 0 else ix
        elif check[0] == 'ZZ':
            for b in bs:
                othr = [_ for _ in bs if _ != b][0]
                iz = mqc[othr][0] + mqc[othr][1]
                xy = mqc[othr][2] + mqc[othr][3]
                mcq[b, 0] += iz if synd == 0 else xy
                mcq[b, 1] += iz if synd == 0 else xy
                mcq[b, 2] += xy if synd == 0 else iz
                mcq[b, 3] += xy if synd == 0 else iz
        else:
            # sum over local Pauli group
            lpg = _lpg_wrap[(check[0], synd)]
            
            for elem in [zip(bs, _) for _ in lpg]:
                summand = reduce(mul, (mqc[tpl] for tpl in elem)) # slow
                for tpl in elem:
                    mcq[tpl] += summand / mqc[tpl]
        
        # normalize
        for b in bs:
            g.node[v]['mcq'][sprt[b]] = mcq[b, :] / sum(mcq[b, :])

def qubit_to_check(g):
    for v in _bits(g):
        # product over neighbours not c
        chex = list(g[v].keys())
        for chek in chex:
            msgs = [g.node[_]['mcq'][v] for _ in chex if _ != chek]
            g.node[v]['mqc'][chek] = reduce(mul, msgs, g.node[v]['prior'])

        # normalize
        for k in g.node[v]['mqc'].keys():
            g.node[v]['mqc'][k] /= sum(g.node[v]['mqc'][k])

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

    stabilisers have to be inserted in the form of a dict with entries
    label: check

    """
    g = nx.Graph()
    for dx, s in stabilisers.items():
        label = 's' + str(dx)
        g.add_node(
                    label,
                    check=s.str_sprt_pair(),
                    syndrome=error.com(s),
                    mcq={b: np.ones_like(mdl) for b in s.support()},
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

def close_bdy_pts(crd, bdy_verts):
    bdy_dists = [dc.pair_dist(crd, v) for v in bdy_verts]
    d = min(bdy_dists)
    return [bdy_verts[_] for _ in range(len(bdy_verts))
                                        if bdy_dists[_] == d]

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
    close_pts = close_bdy_pts(crd, bdy_verts)
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