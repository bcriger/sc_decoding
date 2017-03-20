import sparse_pauli as sp
import matched_weights as mw
from circuit_metric import SCLayoutClass as sc
from operator import add
import numpy as np
from copy import deepcopy

def two_bit_bp():
    """
    Reproduce Figure 2a from Poulin/Chung 2008, using the BP from
    matched_weights.py.
    """
    stabs = {2: sp.X([0, 1]), 3: sp.Z([0, 1])}
    err = sp.X([0])
    mdl = [0.9, 0.1/3, 0.1/3, 0.1/3]

    g = mw.tanner_graph(stabs, err, mdl)

    b_list = [g.node[1]['prior']] # symmetry sez: identical on both qubits

    for _ in range(10):
        mw.propagate_beliefs(g, 1)
        b_list.append(mw.beliefs(g)[1])

    return b_list

def sprt_paulis():
    """
    Just taking a look at some Paulis to see if they're being 
    reformatted correctly.
    """
    sprt = range(5)
    paulis = [sp.X([0, 2, 4]) * sp.Y([1]), sp.Z([0, 1]) * sp.X([1, 3])]
    return [mw.pauli_to_tpls(pauli, sprt) for pauli in paulis]

def yy_check():
    """
    If BP works, then we should be able to match a weight-2 Y error
    at d = 3
    """
    err = sp.Y([3, 8])
    layout = sc.SCLayout(3)
    mdl = [0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3]
    stabs = dict(reduce(add, [layout.stabilisers()[ltr].items() for ltr in 'XZ']))
    tg = mw.tanner_graph(stabs, err, mdl)
    mw.propagate_beliefs(tg, 15)
    b_list = mw.beliefs(tg)
    for key, val in b_list.items():
        print key, val
    # BP works even without the matching in this instance.

def xyx_check():
    """
    If BP works, than a chain consisting of X-Y-X errors will be
    corrected at d = 5.
    """
    layout = sc.SCLayout(5)
    err = sp.X([15, 24, 33]) * sp.Z([24])
    mdl = [0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3]
    # TODO finish later

def bdy_edge_test():
    """
    I'm going to put a single syndrome down on the d=3 SC, then see
    that it can match with either boundary depending on the beliefs we
    use.
    """
    layout = sc.SCLayout(3)
    d_bits = [layout.map[crd] for crd in layout.datas]
    dep_arr = np.array([0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3])

    crd = (4, 4)
    
    vertices = layout.x_ancs()
    
    bdy_1 = layout.boundary['x_left']
    bdy_2 = layout.boundary['x_right']
    
    flat_beliefs = {d: dep_arr for d in d_bits}
    
    peaked_beliefs = deepcopy(flat_beliefs)
    peaked_beliefs[8] = np.array([0., 1., 0., 0.])
    peaked_beliefs[9] = np.array([0., 1., 0., 0.])
    
    stab_type = 'X'
    tpl_1 = mw.bdy_edge(crd, vertices, bdy_1, bdy_2, flat_beliefs, layout, stab_type)
    tpl_2 = mw.bdy_edge(crd, vertices, bdy_1, bdy_2, peaked_beliefs, layout, stab_type)
    print 'flat: {}'.format(tpl_1)
    print 'peaked: {}'.format(tpl_2)
    pass 

#---------------------------------------------------------------------#