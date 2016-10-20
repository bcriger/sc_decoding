import circuit_metric as cm 
import cPickle as pkl

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
        #TODO: Set up default error models for often-studied simulations
        if gate_error_model == 'pq':
            pass
        elif gate_error_model == 'fowler':
            pass

        self.d = d
    
    def run(self, n_trials):
        """
        """
        for run_dx in xrange(n_trials):
            pass
        pass

    def save(self, flnm):
        pass