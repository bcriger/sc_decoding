from circuit_metric.SCLayoutClass import SCLayout
import error_model as em 
import itertools as it
import networkx as nx
from operator import mul
from math import copysign
import sparse_pauli as sp

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
        self.errors = {'I' : 0, 'X' : 0, 'Y' : 0, 'Z' : 0}
        
    def random_error(self):
        return product(self.error_model.sample())
    
    def syndromes(self, error):
        x_synd = []
        z_synd = []
        
        for ltr, lst in zip('XZ', [x_synd, z_synd]):
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
                        weight=-self.bdy_info(crds[s])[0],
                        close_pt=self.bdy_info(crds[s])[1])
        
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
        
        #TODO: Dumb to check ancilla type every time, should be input.
        pauli_lst = []
        for n1, n2 in matching:
            if isinstance(n1, int) & isinstance(n2, int):
                pauli_lst.append(self.path_pauli(x[n1], x[n2],
                                            self.layout.anc_type(n1)))
            elif isinstance(n1, int) ^ isinstance(n2, int):
                bdy_pt = graph[n1][n2]['close_pt']
                vert = n1 if isinstance(n1, int) else n2
                pauli_lst.append(self.path_pauli(bdy_pt, x[vert],
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
        anticom_dict = {
                        ( 0, 0 ) : 'I',
                        ( 0, 1 ) : 'X',
                        ( 1, 0 ) : 'Z',
                        ( 1, 1 ) : 'Y'
                    }
        x_bar, z_bar = self.layout.logicals()
        loop = error * x_corr * z_corr
        x_com, z_com = x_bar.com(loop), z_bar.com(loop)

        return anticom_dict[ ( x_com, z_com ) ]

    def run(self, n_trials, verbose=False):
        """
        Repeats the following cycle `n_trials` times:
         + Generate a random error
         + determine syndromes by checking stabilisers
         + make those syndromes into a graph with boundary vertices
         + match on that graph
         + check for a logical error by testing anticommutation with
           the logical paulis
        """
        for trial in range(n_trials):
            err = self.random_error()
            x_synd, z_synd = self.syndromes(err)
            x_graph, z_graph = self.graph(x_synd), self.graph(z_synd)
            x_corr = self.correction(x_graph)
            z_corr = self.correction(z_graph)
            log = self.logical_error(err, x_corr, z_corr)
            self.errors[log] += 1
            if verbose:
                print "\n".join([
                    "error: {}".format(err),
                    "X syndrome: {}".format(x_synd),
                    "Z syndrome: {}".format(z_synd),
                    "X correction: {}".format(x_corr),
                    "Z correction: {}".format(z_corr),
                    "logical error: {}".format(log)
                    ])


    def bdy_info(self, crd):
        """
        Returns the minimum distance between the input co-ordinate and
        one of the two acceptable corner vertices, depending on 
        syndrome type (X or Z). 
        """
        min_dist = 4 * self.d #any impossibly large value will do
        anc_type = 'X' if crd in self.layout.x_ancs() else 'Z'
        for pt in self.layout.boundary_points(anc_type):
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
        vs = [
                ( ( d + c - b + a ) / 2, ( d + c + b - a ) / 2 ),
                ( ( d - c - b - a ) / -2, ( -d + c - b - a ) / -2 )
            ]
        
        if vs[0] in sum(self.layout.ancillas.values(), ()):
            mid_v = vs[0]
        else:
            mid_v = vs[1]
        
        pth_0, pth_1 = diag_pth(crd_0, mid_v), diag_pth(mid_v, crd_1)

        #path on lattice, uses idxs
        p = [self.layout.map[crd] for crd in pth_0 + pth_1]
        
        pl = sp.Pauli(p, []) if anc_type == 'Z' else sp.Pauli([], p)
        
        return pl 


#-----------------------convenience functions-------------------------#
product = lambda itrbl: reduce(mul, itrbl, sp.Pauli({},{}))


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
    diff = map(abs, (crd_0[0] - crd_1[0], crd_0[1] - crd_1[1]))
    diag_dist = min(diff)
    remainder = max([diff[0] - diag_dist, diff[1] - diag_dist])
    return diag_dist / 2 + remainder

def diag_pth(crd_0, crd_1):
    """
    Produces a path between two points which takes steps (\pm 2, \pm 2)
    between the two, starting (\pm 1, \pm 1) away from the first.
    """
    dx, dy = crd_1[0] - crd_0[0], crd_1[1] - crd_0[1]
    shft_x, shft_y = map(int, [copysign(1, dx), copysign(1, dy)])
    step_x, step_y = map(int, [copysign(2, dx), copysign(2, dy)])
    return zip(range(crd_0[0] + shft_x, crd_1[0], step_x), 
                range(crd_0[1] + shft_y, crd_1[1], step_y))
    

#---------------------------------------------------------------------#


