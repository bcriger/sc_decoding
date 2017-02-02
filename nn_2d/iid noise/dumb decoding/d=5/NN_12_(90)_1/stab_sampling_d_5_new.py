import numpy as np, itertools as it, circuit_metric as cm
from functools import reduce
from operator import mul, or_ as union
from scipy.special import betainc
import SCLayoutClass
import sparse_pauli as sp
import decoding_2d as dc

"""
We're going to calculate the probability that a certain coset is
correct, given a syndrome, by brute force.
This is a subroutine for RG decoding, and it can be used to generate
training data for the NN.  
"""

#from the itertools cookbook
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))

def weight_dist(stab_gens, identity, log, coset_rep):
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

    wt_counts_I = np.zeros((nq + 1,), dtype=np.int_)
    wt_counts_L = np.zeros((nq + 1,), dtype=np.int_)
    
    i = 0
    for subset in powerset(stab_gens):
        i += 1
        s = reduce(mul, subset, sp.Pauli())
        wt_counts_I[(s * identity * coset_rep).weight()] += 1
        wt_counts_L[(s * log * coset_rep).weight()] += 1
        #if i % 1000 == 0:
        #    print(i)

    return [wt_counts_I, wt_counts_L]

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

    return [sum([ c * (p / (1. - p)) ** w for w, c in enumerate(weight_counts[0])]), 
	        sum([ c * (p / (1. - p)) ** w for w, c in enumerate(weight_counts[1])])]

def coset_prob(stab_gens, log, coset_rep, p_lo, p_hi):

    return prob_integral(weight_dist(stab_gens, log, coset_rep), p_lo, p_hi)

distance = 5
physprob = 0.08
sim_test = dc.Sim2D(distance, distance, physprob)
z_ancs_keys = list(sim_test.layout.z_ancs())
cycles = 0
samples = {}
x_anc_len = 12
test_layout = SCLayoutClass.SCLayout(distance)
x_stabs = list(test_layout.stabilisers()['X'].values())
log = test_layout.logicals()[0]

while cycles < 100000:
    cycles += 1
    rnd_err = sim_test.random_error()
    synds = sim_test.syndromes(rnd_err)
    
    list_z = [0] * len(z_ancs_keys)
    for k in synds[1]:
        key = sim_test.layout.map.inv[k]
        pos = z_ancs_keys.index(key)
        list_z[pos] = 1
        
    s = ''
    for i in range(x_anc_len):
        s += str(list_z[i])
		
    if s in samples:
        x = samples[s]
        samples[s] = [x[0], x[1]+1]
    else:
        dumb_x_corr, dumb_z_corr = sim_test.dumb_correction(synds)
        coset_rep = dumb_x_corr
        prob_dist = single_prob(weight_dist(x_stabs, sp.Pauli(), log, coset_rep), physprob)
        norm = sum(prob_dist)
        prob_dist = [p / norm for p in prob_dist]
		
        samples[s] = [prob_dist, 1]		
		
    if cycles % 100 == 0:
        print(len(samples), cycles)

sorted_samples = sorted(samples.items(), key=lambda e: e[1][1], reverse=True)

#for key, value in sorted_samples:
#    print(key, value[0], value[1])
with open('d='+str(distance)+'_p='+str(physprob)+'_samples.txt', 'w') as f:
    for key, value in sorted_samples:
        x = ''
        for i in range(x_anc_len):
            x += key[i] + ' '

        x += ' '
        x += str(value[0][0]) + ' '
        x += str(value[0][1]) + ' '
        x += str(value[1]) + '\n'
        f.write(x)
