"""
The purpose of this script is to reproduce Figure 2a from Poulin/Chung
2008, using the BP from matched_weights.py
"""
import sparse_pauli as sp
import matched_weights as mw

def two_bit_bp():
    stabs = [sp.X([0, 1]), sp.Z([0, 1])]
    err = sp.X([0])
    mdl = [0.9, 0.1/3, 0.1/3, 0.1/3]

    g = mw.tanner_graph(stabs, err, mdl)

    b_list = [g.node[1]['prior']] # symmetry sez: identical on both qubits

    for _ in range(10):
        mw.propagate_beliefs(g, 1)
        b_list.append(mw.beliefs(g)[1])

    return b_list

def sprt_paulis():
    sprt = range(5)
    paulis = [sp.X([0, 2, 4]) * sp.Y([1]), sp.Z([0, 1]) * sp.X([1, 3])]
    return [mw.pauli_to_tpls(pauli, sprt) for pauli in paulis]
#---------------------------------------------------------------------#