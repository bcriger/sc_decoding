import circuit_metric as cm 
from circuit_metric.SCLayoutClass import LOCS
from collections import Iterable
import cPickle as pkl
from decoding_2d import pair_dist
import error_model as em
from itertools import combinations
import networkx as nx
from operator import mul
import progressbar as pb
import sparse_pauli as sp

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
        synd_hist = {'X': [], 'Z': []}
        #perfect (quiescent state) initialization
        err = sp.Pauli() 
        for meas_dx in xrange(self.n_meas):
            #just the ones
            synd = {'X': set(), 'Z': set()}
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
            for key in 'XZ':
                synd_hist[key].append(synd[key])
            err_hist.append(err)

        if final_perfect_rnd:
            synd = {'X': set(), 'Z': set()}
            for ki, val in synd.items():
                for idx, stab in self.layout.stabilisers()[ki].items():
                    if err.com(stab) == 1:
                        val |= {idx}
            for key in 'XZ':
                synd_hist[key].append(synd[key])

        return err_hist, synd_hist

    def correction(self, synds, metric=None, bdy_info=None):
        """
        Given a set of recorded syndromes, returns a correction by
        minimum-weight perfect matching.
        In order to 'make room' for correlated decoders, X and Z 
        syndromes are passed in as a single object, and any splitting
        is performed inside this method.
        Also, a single correction Pauli is returned. 
        metric should be a function, so you'll have to wrap a matrix 
        in table-lookup if you want to use one.
        bdy_info is a function that takes a flip and returns the
        distance to the closest boundary point 
        """

        n = self.layout.n
        x = self.layout.map.inv

        if not(metric):
            metric = lambda flp_1, flp_2: self.manhattan_metric(flp_1, flp_2)
        
        if not bdy_info:
            bdy_info = lambda flp: self.manhattan_bdy_tpl(flp)

        flip_idxs = flat_flips(synds, n)
        
        # Note: 'X' syndromes are XXXX stabiliser measurement results.
        corr = sp.Pauli()
        for stab in 'XZ':
            
            verts = flip_idxs[stab]
            bdy_verts = ([(flip, 'b') for flip in verts])
            
            main_es = [(u, v, metric(u, v)) 
                                for u, v in combinations(verts, r=2)]
            bdy_es = [(u, v, bdy_info(u)[0])
                            for u, v in zip(verts, bdy_verts)]
            zero_es = [ (u, v, 0)
                            for u, v in combinations(bdy_verts, r=2)]
            
            graph = nx.Graph()
            graph.add_weighted_edges_from(main_es + bdy_es + zero_es)
            # Note: code reuse from decoding_2d.Sim2D
            matching = nx.max_weight_matching(graph, maxcardinality=True)
            # get rid of non-digraph duplicates 
            matching = [(u, v) for u, v in matching.items() if u < v]
            
            err = 'X' if stab == 'Z' else 'Z'

            for u, v in matching:
                if isinstance(u, int) & isinstance(v, int):
                    corr *= self.layout.path_pauli(x[u % n], x[v % n], err)
                elif isinstance(u, int) ^ isinstance(v, int):
                    vert = u if isinstance(u, int) else v
                    bdy_pt = bdy_info(vert)[1]
                    corr *= self.layout.path_pauli(bdy_pt, x[vert % n], err)
                else:
                    pass #both boundary points, no correction

        return corr

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

    def run(self, n_trials, progress=True, metric=None, bdy_info=None, final_perfect_rnd=True):
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
            err_hist, synd_hist = self.history(final_perfect_rnd)
            corr = self.correction(synd_hist, metric, bdy_info)
            log = self.logical_error(err_hist[-1], corr)
            self.errors[log] += 1
        pass

    def manhattan_metric(self, flip_a, flip_b):
        """
        Mostly for testing/demonstration, returns a function that takes
        you straight from flip _indices_ to edge weights for the 
        NetworkX maximum-weight matching. 
        
        I'm making a note here about whether to take the minimum
        between two cases for timelike syndrome weights. Given that the
        first round of syndromes count as flips, and the last round is 
        perfect, I'm only going to use paths "timelike between flips",
        and I won't give a pair a weight by taking each syndrome flip 
        out to the time boundary, just the space boundaries.   

        """
        n = self.layout.n
        crds = self.layout.map.inv

        # split into (round, idx) pairs:
        vert_a, idx_a = divmod(flip_a, n)
        vert_b, idx_b = divmod(flip_b, n)

        # horizontal distance between syndromes, from decoding_2d
        horz_dist = pair_dist(crds[idx_a], crds[idx_b])

        # vertical
        vert_dist = abs(vert_a - vert_b)

        return -(horz_dist + vert_dist)

    def manhattan_bdy_tpl(self, flp):
        """
        copypaste from decoding_2d.Sim2D.
        Returns an edge weight for NetworkX maximum-weight matching,
        hence the minus sign.
        """
        crds = self.layout.map.inv
        horz_dx = flp % self.layout.n
        crd = crds[horz_dx]
        
        min_dist = 4 * self.d #any impossibly large value will do
        err_type = 'Z' if crd in self.layout.x_ancs() else 'X'
        for pt in self.layout.boundary_points(err_type):
            new_dist = pair_dist(crd, pt)
            if new_dist < min_dist:
                min_dist, close_pt = new_dist, pt

        return -min_dist, close_pt

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

def flat_flips(synds, n):
    flip_list = {'X': [], 'Z': []}
    for err in 'XZ':
        flip_list[err] = [synds[err][0]]
        flip_list[err].extend([synds[err][t] ^ synds[err][t - 1]
                                 for t in range(1, len(synds[err]))])
    
    # Convert history of flips to a flat list by adding
    # n_qubits * t to each element
    flat_flips = {'X': [], 'Z': []}
    for err in 'XZ':
        for t, layer in enumerate(flip_list[err]):
            flat_flips[err].extend([flp + t * n for flp in layer])

    return flat_flips
