from circuit_metric.SCLayoutClass import SCLayout
import error_model as em 
import itertools as it
import networkx as nx
from operator import mul

class Sim2D(object):
    """
    This is a pretty bare-bones simulation of the 2D rotated surface
    code. We take a surface code of distance d, and put it up against 
    an IID X/Z error model with probability p. 
    """
    def __init__(self, d, p):
        #user-input properties
        self.d = d
        self.p = p
        
        #derived properties
        self.layout = SCLayout(d)
        self.error_model = em.PauliErrorModel([(1. - p)**2,
            p * (1. - p), p * (1. - p), p**2], 
            [[_] for _ in self.layout.datas])

    def random_error(self):
        return product(self.error_model.sample())
    
    def syndromes(self, error):
        x_synd = []
        z_synd = []
        
        for ltr, lst in zip('xz', [x_synd, z_synd]):
            for idx, stab in self.layout.stabilisers()[ltr].items():
                if com(error, stab) == 1:
                    lst.append(idx)

        return x_synd, z_synd

    def graph(syndrome):
        """
        returns a NetworkX graph from a given syndrome, on which you 
        can find the MAXIMUM weight perfect matching (This is what
        NetworkX does). We use negative edge weights to make this 
        happen.
        """
        crds = lambda idx: self.layout.map[:idx] #ugliness
        g = nx.Graph()
        
        #vertices directly from syndrome
        g.add_nodes_from(syndrome)
        g.add_weighted_edges_from(
            (v1, v2, -pair_dist(crds(v1), crds(v2))) for v1, v2 in 
            it.combinations(syndrome, 2)
            )
        
        #boundary vertices, edges from boundary distance
        
        
        #weight-0 edges between boundary vertices
        return g

    def correction(graph):
        """
        Given a syndrome graph with negative edge weights, finds the
        maximum-weight perfect matching and produces a
        sparse_pauli.Pauli 
        """
        x = lambda idx: self.layout.map[:idx] #ugliness
        matching = nx.max_weight_matching(graph, maxcardinality=True)
        # get rid of non-digraph duplicates 
        matching = [(k, v) for k, v in matching.items() if k < l]
        correction = product(path_pauli(x(k), x(v), layout.anc_type(k))
                                for k, v in matching)
        return correction
    
    def logical_error(error, x_corr, z_corr):
        """
        Given an error and a correction, multiplies them and returns a
        single letter recording the resulting logical error (may be I,
        X, Y or Z)
        """
        x_bar, z_bar = self.layout.logicals()
        pass

    def bdy_dist(self, crd):
        """
        Returns the minimum distance between the input co-ordinate and
        one of the two acceptable corner vertices, depending on 
        syndrome type (X or Z). 
        """
        return min(pair_dist(crd, pt) for pt in self.boundary_points())


#-----------------------convenience functions-------------------------#
product = lambda itrbl: return reduce(mul, itrbl)


def pair_dist(crd_0, crd_1):
    """
    Returns the distance between syndromes of the same type on the 
    rotated surface code. This distance is calculated by first taking
    as many steps as possible diagonally (the length of each of these 
    steps is 1), then finishing horizontally/vertically (each of these
    steps is length 2). 

    Note that syndromes at (2,2) and (4,4) in the layout are separated
    by a length-1 chain, because the squares are 2-by-2. 
    """
    diff = map(abs, (crd_0[0] - crd_1[0], crd_0[0] - crd_1[0]))
    diag_dist = min(diff)
    remainder = max([diff[0] - diag_dist[0], diff[1] - diag_dist[1]])
    return diag_dist / 2 + remainder

def path_pauli(crd_0, crd_1, anc_type):
    """
    Returns a minimum-length Pauli between two ancillas, given the type
    of STABILISER that they measure. 
    """
    diff = map(abs, (crd_0[0] - crd_1[0], crd_0[0] - crd_1[0]))
    diag_dist = min(diff)
    remainder = max([diff[0] - diag_dist[0], diff[1] - diag_dist[1]])
    # sort syndromes left-right
    crd_0, crd_1 = sorted([crd_0, crd_1], key = lambda x: x[0])

    h_shft = (2, 0)
    #decide whether to go up or down

    pass



#---------------------------------------------------------------------#


