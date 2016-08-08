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
            [[self.layout.map[_]] for _ in self.layout.datas])

    def random_error(self):
        return product(self.error_model.sample())
    
    def syndromes(self, error):
        x_synd = []
        z_synd = []
        
        for ltr, lst in zip('xz', [x_synd, z_synd]):
            for idx, stab in self.layout.stabilisers()[ltr].items():
                if error.com(stab) == 1:
                    lst.append(idx)

        return x_synd, z_synd

    def graph(self, syndrome):
        """
        returns a NetworkX graph from a given syndrome, on which you 
        can find the MAXIMUM weight perfect matching (This is what
        NetworkX does). We use negative edge weights to make this 
        happen.
        """
        crds = self.layout.map.inv 
        g = nx.Graph()
        
        #vertices directly from syndrome
        g.add_nodes_from(syndrome)
        g.add_weighted_edges_from(
            (v1, v2, -pair_dist(crds[v1], crds[v2]))
            for v1, v2 in 
            it.combinations(syndrome, 2)
            )
        
        #boundary vertices, edges from boundary distance
        for s in syndrome:
            g.add_edge(s, (s, 'b'),
                        weight=-self.bdy_dist(crds[s])[0],
                        close_pt=self.bdy_dist(crds[s])[1])
        
        #weight-0 edges between boundary vertices
        g.add_weighted_edges_from(
            ((v1, 'b'), (v2, 'b'), 0.)
            for v1, v2 in 
            it.combinations(syndrome, 2)
            )
        return g

    def correction(self, graph):
        """
        Given a syndrome graph with negative edge weights, finds the
        maximum-weight perfect matching and produces a
        sparse_pauli.Pauli 
        """
        x = self.layout.map.inv
        matching = nx.max_weight_matching(graph, maxcardinality=True)
        
        # get rid of non-digraph duplicates 
        matching = [(n1, n2) for n1, n2 in matching.items() if n1 < n2]
        
        pauli_lst = []
        for n1, n2 in matching:
            if isinstance(n1, int) & isinstance(n2, int):
                pauli_lst.append(self.path_pauli(x[n1], x[n2],
                                            self.layout.anc_type(n1)))
            elif isinstance(n1, int) ^ isinstance(n2, int):
                bdy_pt = g[n1][n2]['close_pt']
                vert = n1 if isinstance(n1, int) else n2
                pauli_lst.append(path_pauli(bdy_pt, x[vert],
                                            self.layout.anc_type(vert)))
            else:
                pass #both boundary points, no correction

        return product(pauli_lst)
    
    def logical_error(self, error, x_corr, z_corr):
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
        min_dist = 4 * self.d #any impossibly large value will do

        for pt in self.layout.boundary_points():
            new_dist = pair_dist(crd, pt)
            if new_dist < min_dist:
                min_dist, close_pt = new_dist, pt

        return min_dist, close_pt

    def path_pauli(self, crd_0, crd_1, anc_type):
        """
        Returns a minimum-length Pauli between two ancillas, given the
        type of STABILISER that they measure.

        This function is awkward, because it works implicitly on the
        un-rotated surface code, first finding a "corner" (a place on
        the lattice for the path to turn 90 degrees), then producing
        two diagonal paths on the rotated lattice that go to and from
        this corner. 
        """
        a, b, c, d = crd_0[0], crd_0[1], crd_1[0], crd_1[1]
        vs = [((d-b-c+a)/2, (b+d-c-a)/2),
                                (-(d-b+c-a)/2, -(-b-d-c-a)/2)]
        
        if vs[0] in sum(self.layout.ancillas.values(), ()):
            mid_vert = vs[0]
        else:
            mid_vert = vs[1]
        
        return map(self.layout.map, path_0 + path_1)


#-----------------------convenience functions-------------------------#
product = lambda itrbl: reduce(mul, itrbl)


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
    remainder = max([diff[0] - diag_dist, diff[1] - diag_dist])
    return diag_dist / 2 + remainder

#---------------------------------------------------------------------#


