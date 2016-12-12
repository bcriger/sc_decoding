import sparse_pauli as sp
import cProfile as prof
# if python 3 import profile as prof, blah blah
import numpy as np

def rand_set():
    return set(np.random.randint(50, size=(5,)))

def mul_paulis(set_pairs):
    "multiply random sparse paulis"
    for pr in set_pairs:
        _ = sp.Pauli(pr[0], {}) * sp.Pauli(pr[1], {})

def diff_sets(set_pairs):
    "just sym_diff the sets we use"
    for pr in set_pairs:
        _ = pr[0] ^ pr[1]

if __name__ == '__main__':
    set_pairs = [(rand_set(), rand_set()) for _ in range(10000)]
    prof.run("mul_paulis(set_pairs)", filename="mul_paulis.prof")
    prof.run("diff_sets(set_pairs)", filename="diff_sets.prof")