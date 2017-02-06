"""
I'm going to iterate over the entire stabiliser group of the d = 3
rotated SC to determine the minimum-weight Pauli that matches a given
syndrome.
"""

import circuit_metric.SCLayoutClass as sc
import sparse_pauli as sp 
import itertools as it
import decoding_2d as dc2
from operator import mul
from stabiliser_sampling import powerset

def mwe(sim, syndromes, stabs, logs):
    """
    Input: where XXXX and ZZZZ stabilisers are violated, respectively.
    Output: minimum-weight Pauli that matches the syndrome.
    """
    # Pauli must be a product of a dumb correction, a logical and a
    # stabiliser:
    dumb_corr = reduce(mul, sim.dumb_correction(syndromes), sp.Pauli())
    all_logs = [reduce(mul, _, sp.Pauli()) for _ in powerset(logs)]

    curr_corr = dumb_corr
    for stab_set in powerset(stabs):
        cs_prod = reduce(mul, stab_set, dumb_corr)
        for log in all_logs:
            test_corr = cs_prod * log
            if test_corr.weight() < curr_corr.weight():
                curr_corr = test_corr

    return curr_corr

def main(sim):
    xs = [sim.layout.map[_] for _ in sim.layout.x_ancs()]
    zs = [sim.layout.map[_] for _ in sim.layout.z_ancs()]
    x_combos = powerset(xs)
    z_combos = powerset(zs)

    stabs = sim.layout.stabilisers()
    stabs = [d.values() for d in stabs.values()][0] + [d.values() for d in stabs.values()][1]
    logs = sim.layout.logicals()
    table = dict()
    for synds in it.product(x_combos, z_combos):
        table[synds] = mwe(sim, synds, stabs, logs)
    return table

if __name__ == '__main__':
    test_sim = dc2.Sim2D(3, 3, 0.0)
    table = main(test_sim)