"""
We're going to calculate the probability that a certain coset is
correct, given a syndrome, by brute force.
This is a subroutine for RG decoding, and it can be used to generate
training data for the NN.  
"""

import numpy as np, itertools as it, circuit_metric as cm
import sparse_pauli as sp
from functools import reduce
from operator import mul, or_ as union
from scipy.special import betainc

#from the itertools cookbook
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))

def weight_dist(stab_gens, log, coset_rep):
    """
    If we iterate over an entire stabiliser group, we can calculate
    the weight of each Pauli of the form c * l * s for c a coset
    representative for the normaliser of the stabiliser, l a choice of 
    logical operator, and s a stabiliser.
    Colloquially, c is generated by a one-to-one map from the
    syndromes.

    The weight will range from 0 to nq, where nq is the number of
    qubits in the system, and there'll be many Paulis with each weight.
    Therefore, we return an array full of counts.  
    """

    # First get nq
    nq = len(reduce(union, (s.support() for s in stab_gens)))

    wt_counts = np.zeros((nq + 1,), dtype=np.int_)
    for subset in powerset(stab_gens):
        s = reduce(mul, subset, sp.Pauli())
        wt_counts[(s * log * coset_rep).weight()] += 1

    return wt_counts

def prob_integral(weight_counts, p_lo, p_hi):
    """
    The probability of a logical out given a syndrome coset rep in is
    sum_{s in stab_group} (p/(1-p))^{|c * l * s|} where c is the coset
    rep, and l is the logical.
    We're going to sample p over a uniform distribution from p_lo to
    p_hi, so we calculate the expected value of this probability over
    the distribution.
    """
    n = len(weight_counts) - 1
    return sum([
                    c * (betainc(w + 1, n - w + 1, p_hi) 
                        - betainc(w + 1, n - w + 1, p_lo))
                    for w, c in enumerate(weight_counts)
                ])/(p_hi - p_lo)

def single_prob(weight_counts, p):
    """
    For debugging purposes, I'd like to have a function that evaluates
    the coset probability at a single point in p-space, so that I can
    see whether the integral is off. This is going to get normalised
    anyway, so we can output probabilities that are off by an overall
    factor of (1 - p) ** n.
    """
    return sum([ c * (p / (1. - p)) ** w
                for w, c in enumerate(weight_counts)
        ])
    pass

def coset_prob(stab_gens, log, coset_rep, p_lo, p_hi):

    return prob_integral(weight_dist(stab_gens, log, coset_rep), p_lo, p_hi)

if __name__ == '__main__':
    test_layout = cm.SCLayoutClass.SCLayout(5)
    x_stabs = list(test_layout.stabilisers()['X'].values())
    log = test_layout.logicals()[0]
    p_lo, p_hi = 0.005, 0.05
    # prob_dist = [coset_prob(x_stabs, log, sp.Pauli(), p_lo, p_hi),
    #                 coset_prob(x_stabs, log, coset_rep, p_lo, p_hi)]
    c_rep_lst = [test_layout.map[q] for q in test_layout.datas]
    for idx in c_rep_lst:
        coset_rep = sp.Pauli({idx},{})
        prob_dist = [
                        single_prob(weight_dist(x_stabs, sp.Pauli(), coset_rep), 0.01),
                        single_prob(weight_dist(x_stabs, log, coset_rep), 0.01)
                    ]
        norm = sum(prob_dist)
        prob_dist = [p / norm for p in prob_dist]
        print('actual error (also coset rep): {}. logical probabilities: {}'.format(coset_rep, prob_dist))

#---------------------------------------------------------------------#
