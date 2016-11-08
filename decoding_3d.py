import circuit_metric as cm 
import cPickle as pkl
from collections import Iterable
import error_model as em
from circuit_metric.SCLayoutClass import LOCS
import progressbar as pb
import sparse_pauli as sp
from operator import mul

product = lambda itrbl: reduce(mul, itrbl)

class Sim3D(object):
    """
    Simulation object which contains information about fault-tolerant
    (2 + 1D) surface code simulations, extensible to other 2D codes in 
    principle.
    Methods follow a common pattern from my other simulation classes:
        __init__:   set simulation parameters
        run:        use parameters to populate sample list by Monte 
                    Carlo
        save:       pickle parameters and derived statistics
    """
    def __init__(self, d, n_meas, gate_error_model):
        """
        Produces a simulation object which can then be run, examined
        and saved.
        Inputs:
            d:                  code distance. Must be an integer
                                greater than 1.
            n_meas:             number of measurement rounds. Must be
                                an integer greater than 1.
            gate_error_model:   specification of which errors occur at
                                which timesteps. For an arbitrary
                                model, input a list of lists which is
                                the same length as the extractor of the
                                SCLayout object, less one ().
                                A sample from each PauliErrorModel
                                specified in element `j` will be
                                applied after timestep `j` of the 
                                extractor.
                                Two special error models, 'pq' and
                                'fowler' are permitted, they correspond
                                to internal models replicating the 
                                3DZ2RPGM and the 
                                Fowler/Stephens/Groszkowski paper. 
        """
        if not(isinstance(d, int)) or (d <= 1):
            raise ValueError("d must be an integer at least 2, "
                "{} entered.".format(d))
        
        if not(isinstance(n_meas, int)) or (n_meas <= 1):
            raise ValueError("n_meas must be an integer at least 2, "
                "{} entered.".format(n_meas))

        self.d = d
        self.layout = cm.SCLayoutClass.SCLayout(d)
        self.n_meas = n_meas

        #pre-fab error models 

        #TODO: Set up default error models for often-studied simulations
        if gate_error_model[0] == 'pq':
            p, q = gate_error_model[1:]
            gate_error_model = pq_model(self.layout.extractor(), p, q)
        elif gate_error_model[0] == 'fowler':
            p = gate_error_model[1]
            gate_error_model = fowler_model(self.layout.extractor(), p)
        else:
            #check that the gate_error_model is a list of lists of
            #`PauliErrorModel`s:
            if not isinstance(gate_error_model, Iterable):
                raise ValueError("Input gate_error_model is not "
                    "iterable:\n{}".format(gate_error_model))

            for idx, elem in enumerate(gate_error_model):
                if not isinstance(elem, Iterable):
                    raise ValueError(("Element {} of gate_error_model "
                        "is not iterable:\n{}").format(idx, elem))
                elif any(not(isinstance(_, em.PauliErrorModel)) for _ in elem):
                    raise ValueError(("Element {} of gate_error_model"
                        " contains non-error_model elements:\n{}"
                        ).format(idx, elem))

        self.gate_error_model = gate_error_model
        
        #extra derived properties
        self.errors = {'I' : 0, 'X' : 0, 'Y' : 0, 'Z' : 0}
        self.extractor = self.layout.extractor() #convenience        

    def history(self, final_perfect_rnd=True):
        """
        Produces a list of sparse_pauli.Paulis that track the error 
        through the `n_meas` measurement rounds. 
        """

        #ancillas (for restricting error to data bits)
        ancs = {self.layout.map[anc]
                for anc in sum(self.layout.ancillas.values(), ())}
        err_hist = []
        synd_hist = []
        #perfect (quiescent state) initialization
        err = sp.Pauli() 
        for meas_dx in xrange(self.n_meas):
            #just the ones
            synd = {'x': set(), 'z': set()}
            #run circuit
            for stp, mdl in zip(self.extractor, self.gate_error_model):
                #run timestep, then sample
                new_synds, err = cm.apply_step(stp, err)
                err *= product(_.sample() for _ in mdl)
                for ki in synd.keys():
                    synd[ki] |= new_synds[ki][1]
            
            #last round of circuit, because there are n-1 errs, n gates
            new_synds, err = cm.apply_step(self.extractor[-1], err)
            
            for ki in synd.keys():
                synd[ki] |= new_synds[ki][1]

            # remove remaining errors on ancilla qubits before append
            # (they contain no information)
            err.prep(ancs)
            
            synd_hist.append(synd)
            err_hist.append(err)

        return err_hist, synd_hist

    def correction(self, synd_hist, metric=None):
        """
        Given a set of recorded syndromes, returns a correction by
        minimum-weight perfect matching.
        In order to 'make room' for correlated decoders, X and Z 
        syndromes are passed in as a single object, and any splitting
        is performed inside this method.
        Also, a single correction Pauli is returned. 
        metric should be a function, so you'll have to wrap a matrix 
        in table-lookup if you want to use one.
        """
        if not(metric):
            pass #use manhattan dist
        
        flip_list = {'x': [], 'z': []}
        for key in 'xz':
            #first set of flips at first round
            flip_list[key] = [synd_hist[key][0]]
            for t, layer in enumerate(synd_hist[key][1:]):
                #diffs
                flip_list[key][t] = synd_hist[key][t] ^ synd_hist[key][t - 1] 
        #TODO FINISH



    def logical_error(self, final_error, corr):
        """
        Given an error and a correction, multiplies them and returns a
        single letter recording the resulting logical error (may be I,
        X, Y or Z)
        """
        anticom_dict = {
                        ( 0, 0 ) : 'I',
                        ( 0, 1 ) : 'X',
                        ( 1, 0 ) : 'Z',
                        ( 1, 1 ) : 'Y'
                    }
        x_bar, z_bar = self.layout.logicals()
        loop = final_error * corr
        x_com, z_com = x_bar.com(loop), z_bar.com(loop)

        return anticom_dict[ ( x_com, z_com ) ]

    def run(self, n_trials):
        """
        Repeats the following cycle `n_trials` times:
         + Generate a list of 'n_meas' cumulative errors
         + determine syndromes by checking stabilisers
         + make those syndromes into a graph with boundary vertices
         + match on that graph
         + check for a logical error by testing anticommutation with
           the logical paulis
        """
        bar = pb.ProgressBar()
        trials = bar(range(n_trials)) if progress else range(n_trials)
        for trial in trials:
            pass
        pass

    def save(self, flnm):
        pass


#-----------------------convenience functions-------------------------#
def pq_model(extractor, p, q):
    """
    Produces an error model list for a given syndrome extractor which
    is meant to replicate the toy model from DKLP 2001 where X, Z and 
    syndrome-bit flips occur with equal probability. 
    This is accomplished by putting independent X and Z errors after
    the first timestep, and syndrome errors of the appropriate type 
    (X or Z, opposite the measurement type) after the step before the 
    measurement.
    """

    err_list = [ [] for _ in range(len(extractor)-1) ]
    
    # data qubits
    ds = [tpl[1] for tpl in extractor[0] if tpl[0]=='I']
    err_list[0] = [em.iidxz_model(p, ds)]

    # flip ancillas immediately before measurement
    for t, timestep in enumerate(extractor):
        
        m_x, m_z = [[tp[1:] for tp in timestep if tp[0] == s]
                                 for s in ('M_X', 'M_Z')]

        # errors always come after gates, so measurement errors have to
        # go _back in time_:
        err_list[t-1].extend([ em.x_flip(q, m_z), em.z_flip(q, m_x) ])

    return err_list

def fowler_model(extractor, p):
    """
    Produces an error model for a given syndrome extractor which is
    meant to replicate the toy model from Fowler/Stephens/Groszkowski
    and/or Wang/Fowler/Hollenberg, where each single-qubit gate is 
    followed by depolarizing noise with each single qubit Pauli having
    probability p/3, and each two-qubit gate is followed by a two-bit
    depolarizing map where each non-I two-bit Pauli has probability 
    p/15.
    """
    err_list = [[] for _ in extractor]
    for t, timestep in enumerate(extractor):
        
        singles, doubles = [[tp[1:] for tp in timestep if tp[0] in LOCS[_]]
                            for _ in ['SINGLE_GATES', 'DOUBLE_GATES']]
        
        p_x, p_z, m_x, m_z = [[tp[1:] for tp in timestep if tp[0] == s]
                                 for s in ('P_X', 'P_Z', 'M_X', 'M_Z')]

        err_list[t].extend([
            em.depolarize(p, singles),
            em.pair_twirl(p, doubles),
            em.z_flip(p, p_x),
            em.x_flip(p, p_z)
            ])
        # errors always come after gates, so measurement errors have to
        # go _back in time_:
        err_list[t - 1].extend([
            em.x_flip(p, m_z),
            em.z_flip(p, m_x)
                    ])

    return err_list
