import numpy as np
import sparse_pauli as sp

TOL = 10.**-14

class PauliErrorModel(object):
    """
    Small, dense Pauli Error Model. 

    Wraps an array and a list of lists.

    The array must be flat with 4**n real entries for n the length of
    each entry in `q_lsts`. These entries must sum to 1 to within
    tolerance TOL (set to 10**-14 by default).
    
    All the list has to do is cast to a list, and have equal-length
    elements.
    """

    def __init__(self, p_arr, q_lsts, check=True):
        
        #upcast if q_lsts is not nested
        if not hasattr(q_lsts[0], '__iter__'):
                q_lsts = [q_lsts]
        
        if check:
            p_arr = array_cast(p_arr)
            prob_sum_check(p_arr)
            for lst in q_lsts:
                if type(lst) is set:
                    raise TypeError("THOU SHALT NOT USE UN-ORDERED "
                        "TYPES FOR QUBIT LISTS")

            try: 
                q_lsts = list(q_lsts)
            except Exception as err:
                raise TypeError("Input list of qubit lists ({}) "
                    "does not cast to list.  ".format(q_lsts) + 
                    "Error: " + err.strerror)
            
            test_len = len(q_lsts[0])
            
            for lst in q_lsts:
                if len(lst) != test_len:
                    raise ValueError("All supports must be equal in"
                        " size. {} and {} have different "
                        "lengths.".format(q_lsts[0], lst)) 

            if len(p_arr) != 4 ** test_len:
                raise ValueError("Number of qubits inconsistent. Qubit "
                    "labels imply {} qubit(s), but".format(len(q_lsts)) + 
                    " there are {} probabilities.".format(len(p_arr)))

        self.p_arr = p_arr
        self.q_lsts = q_lsts

    def sample(self):
        """
        Produces a random number between 0 and 1, and iteratively 
        subtracts until the difference is less than the probability of
        a Pauli in the model, and returns that Pauli.
        I use int_sample to get an index from the array p_arr, then
        convert to a sparse_pauli using int_to_pauli. 
        """
        indxs = [int_sample(self.p_arr) for _ in self.q_lsts]
        return map(lambda args: int_to_pauli(*args),
                                zip(indxs, self.q_lsts))

    def iidxz_model(p, q_set):
        p_vec = [(1. - p)**2, p * (1. - p), p * (1. - p), p**2]
        return PauliErrorModel(p_vec, q_set)

    def x_flip(p, q_set):
        p_vec = [1. - p, 0., p, 0.]
        return PauliErrorModel(p_vec, q_set)
    
    def z_flip(p, q_set):
        p_vec = [1. - p, p, 0., 0.]
        return PauliErrorModel(p_vec, q_set)

class NoisyClifford(object):
    """
    Sometimes, a noisy Clifford operation is best represented not as a
    perfect Clifford followed by Pauli noise, but as a probability 
    distribution over the Cliffords themselves (see
    github.com/bcriger/rough_notes/Gates_With_T2.pdf for examples).

    I assume that such Cliffords will only act on 1-2 qubits, and that
    the probability distribution will have small support on the set of
    Cliffords (e.g. a Y_{90} with T_2 only has support on 4 out of 24
    Cliffords). 

    For NoisyClifford, you put in a p_arr (same format and checks as 
    the p_arr in PauliErrorModel) and an ordered list of Cliffords 
    (format TBD).
    """
    def __init__(self, p_arr, clifford_lst, check=True):
        if check:
            p_arr = array_cast(p_arr)
            prob_sum_check(p_arr)

        self.p_arr = p_arr
        #TODO: Check that the Cliffords are Cliffords?
        self.clifford_lst = clifford_lst
    
    def sample(self):
        return self.clifford_lst[int_sample(self.p_arr)]

#-----------------------convenience functions-------------------------#

def int_sample(probs):
    """
    This is a little fancy, so I'll explain it. 
    If we are given an array of probabilities [p_0, p_1, ..., p_n],
    we can sample, yielding k if a unit uniform deviate is
    within the interval (sum_{k<j}p_k, sum_{k<j}p_k + p_k). To 
    accomplish this, we first take such a sample, then subtract off
    p_k's as we proceed. In theory, distributions which are sorted 
    descending will be most efficiently sampled, but in practice it
    doesn't make a difference. We explicitly don't sort. 
    """
    value = np.random.rand()
    
    for idx, p in enumerate(probs):
        if value < p:
            return idx
        else:
            value -= p
    
    raise ValueError("Enumeration Failure (probability > 1?)") 

def int_to_pauli(intgr, lbl_lst):
    """
    This one needs a little explaining. I convert every integer with
    an even number of bits to a Pauli by splitting the integer into a
    left (big) half and a right (small) half. 
    Locations where the left half is 1 are elements of the `x_set`.
    Locations where the right half is 1 are elements of the `z_set`.
    """
    bits = bin(intgr)[2:].zfill(2 * len(lbl_lst))
    xs, zs = bits[ : len(lbl_lst)], bits[len(lbl_lst) : ]
    return sp.Pauli({l for l, b in zip(lbl_lst, xs) if b == '1'},
                    {l for l, b in zip(lbl_lst, zs) if b == '1'})

def array_cast(p_arr):
    """
    Nice check to see that something casts to an array
    """
    try: 
        return np.array(p_arr)
    except Exception as err:
        raise TypeError("Input array of probabilities ({}) "
            "does not cast to array.  ".format(p_arr) + 
            "Error: " + err.strerror)

def prob_sum_check(p_arr):
    delta = abs(sum(p_arr) - 1.)
    if delta > TOL:
        raise ValueError("Input probabilities must sum to" + 
            " unity. \n    Input: {}".format(p_arr) + 
            "\n    Difference: {}".format(delta))

def f_p_sample(faults_probs):
    """
    In order to generate a metric, we first produce a list of faults 
    that can occur at every timestep, along with the probability that
    they do. 
    """
    pass