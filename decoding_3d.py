import circuit_metric as cm 
import cPickle as pkl
from collections import Iterable
import error_model as em
from cm.SCLayoutClass import LOCS

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
    
    def run(self, n_trials):
        """
        """
        for run_dx in xrange(n_trials):
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
    err_list[0] = [em.PauliErrorModel.iidxz_model(p, ds)]

    # flip ancillas at the end
    x_ms = [tpl[1] for tpl in extractor[0] if tpl[0]=='M_X']
    z_ms = [tpl[1] for tpl in extractor[0] if tpl[0]=='M_Z']
    err_list[-1].append(em.PauliErrorModel.z_flip(q, x_ms))
    err_list[-1].append(em.PauliErrorModel.x_flip(q, z_ms))

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
    err_list = []
    for t, timestep in enumerate(extractor):
        
        singles, doubles = (
            [b for a, b in timestep if a in LOCS[key]]
            for key in ['SINGLE_GATES', 'DOUBLE_GATES']
                        )
        
        p_x, p_z, m_x, m_z = [[b for a, b in timestep if a == s]
                                 for s in ('P_X', 'P_Z', 'M_X', 'M_Z')]

        err_list



