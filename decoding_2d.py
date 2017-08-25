# The import syntax changes slightly between python 2 and 3, so we
# need to detect which version is being used:
from sys import version_info
if version_info[0] == 3:
    PY3 = True
    from importlib import reload
    import pickle as pkl
elif version_info[0] == 2:
    PY3 = False
    import cPickle as pkl
else:
    raise EnvironmentError("sys.version_info refers to a version of "
        "Python neither 2 nor 3. This is not permitted. "
        "sys.version_info = {}".format(version_info))

from circuit_metric.SCLayoutClass import SCLayout, TCLayout, PCLayout
from decoding_utils import blossom_path, cdef_str
import error_model as em
import itertools as it
import matched_weights as mw
from math import copysign
import networkx as nx
import numpy as np
from operator import mul
import progressbar as pb
import sparse_pauli as sp

try:
    import matplotlib.pyplot as plt
except RuntimeError:
    pass #hope you don't want to draw

import tom.blossom_wrapper as bw

from cffi import FFI

from sys import version_info
if version_info.major == 3:
    from functools import reduce

_pauli_key = [None, sp.Z, sp.X, sp.Y] # trust me

class Sim2D(object):
    """
    This is a pretty bare-bones simulation of the 2D rotated surface
    code. We take a surface code of distance d, and put it up against
    an IID X/Z error model with probability p.
    """
    def __init__(self, dx, dy, p, useBlossom=True, boundary_conditions='rotated'):
        """
        I set optional BC here, for now you can set 'rotated' or
        'closed'.
        """
        #user-input properties
        self.dx = dx
        self.dy = dy
        self.useBlossom = useBlossom
        self.boundary_conditions = boundary_conditions

        #derived properties
        if boundary_conditions == 'open':
            self.layout = PCLayout(dx, dy)
            self.errors = {'I' : 0, 'X' : 0, 'Y' : 0, 'Z' : 0}
        elif boundary_conditions  == 'rotated':
            self.layout = SCLayout(dx, dy)
            self.errors = {'I' : 0, 'X' : 0, 'Y' : 0, 'Z' : 0}
        elif boundary_conditions == 'closed':
            self.layout = TCLayout(dx) #TODO dy
            self.errors = {
                            'II' : 0, 'IX' : 0, 'IY' : 0, 'IZ' : 0,
                            'XI' : 0, 'XX' : 0, 'XY' : 0, 'XZ' : 0,
                            'YI' : 0, 'YX' : 0, 'YY' : 0, 'YZ' : 0,
                            'ZI' : 0, 'ZX' : 0, 'ZY' : 0, 'ZZ' : 0
                        }

        self.error_model = em.PauliErrorModel([(1. - p)**2, p * (1. - p), p * (1. - p), p**2],
            [[self.layout.map[_]] for _ in self.layout.datas])

        if self.useBlossom:
            # load C blossom library
            self.ffi = FFI()
            self.blossom = self.ffi.dlopen(blossom_path)
            # print('Loaded lib {0}'.format(self.blossom))

            self.ffi.cdef(cdef_str)

    def random_error(self):
        return self.error_model.sample()

    def syndromes(self, error):
        x_synd = []
        z_synd = []

        for ltr, lst in zip('XZ', [x_synd, z_synd]):
            for idx, stab in self.layout.stabilisers()[ltr].items():
                if error.com(stab) == 1:
                    lst.append(idx)

        return x_synd, z_synd

    def dumb_correction(self, syndromes, origin=False):
        """
        Connects all detection events to the closest boundary of the
        appropriate type.
        Simple dumb decoder.
        Throughout this method, we will treat a Z syndrome as
        indicating a Z error.
        Note that these syndromes are supported on the X ancillas of
        the layout and vice versa.

        There's an optional argument, origin, that will just move
        all observed syndromes to the same corner of the lattice, near
        the origin.

        TODO: Make it work with toric BC?
        """
        corr_dict = {'Z': sp.Pauli(), 'X': sp.Pauli()}
        if origin:
            for err, synd, pt in zip ('ZX', syndromes, [(0, 0), (2, 0)]):
                crds = [self.layout.map.inv[_] for _ in synd]
                corr_dict[err] *= product([
                    self.path_pauli(_, pt, err)
                    for _ in crds
                    ])
        else:
            for err, synd in zip ('ZX', syndromes):
                crds = [self.layout.map.inv[_] for _ in synd]
                corr_dict[err] *= product([
                    self.path_pauli(_, self.bdy_info(_)[1], err)
                    for _ in crds
                    ])

        return corr_dict['X'], corr_dict['Z']

    def graph(self, syndrome, shadow=False, dist_func=None):
        """
        returns a NetworkX graph from a given syndrome, on which you
        can find the MAXIMUM weight perfect matching (This is what
        NetworkX does). We use negative edge weights to make this
        happen.
        """
        if self.boundary_conditions in ['closed', 'open']:
            l = None if self.boundary_conditions == 'open' else self.dx
            if dist_func is None:
                dist_func = lambda c0, c1: toric_dist(c0, c1, l)
                #Won't work for asymmetric toruses
        elif self.boundary_conditions == 'rotated':
            dist_func = dist_func if dist_func else pair_dist

        crds = self.layout.map.inv
        g = nx.Graph()

        #vertices directly from syndrome
        g.add_nodes_from(syndrome)
        g.add_weighted_edges_from(
            (v1, v2, -dist_func(crds[v1], crds[v2]))
            for v1, v2 in it.combinations(syndrome, 2) )

        if self.boundary_conditions in ['rotated', 'open']:
            #boundary vertices, edges from boundary distance
            for s in syndrome:
                b_info = self.bdy_info(crds[s])
                closest_pt = b_info[1]
                dist = dist_func(crds[s], closest_pt)
                g.add_edge(s, (s, 'b'), weight=-dist, close_pt=closest_pt)

            #weight-0 edges between boundary vertices
            g.add_weighted_edges_from(
                ((v1, 'b'), (v2, 'b'), 0.)
                for v1, v2 in
                it.combinations(syndrome, 2)
                )

        return g

    def correction(self, graph, err):
        """
        Given a syndrome graph with negative edge weights, finds the
        maximum-weight perfect matching and produces a
        sparse_pauli.Pauli
        """

        if self.useBlossom:
            #-----------  c processing
            # print('C Processing')

            # print( 'graph nodes: {0}'.format( graph.nodes() ) )
            # print( 'graph edges: {0}'.format( graph.edges() ) )

            node_num = nx.number_of_nodes(graph)
            edge_num = nx.number_of_edges(graph)
            # print( 'no of nodes : {0}, no of edges : {1}'.format(node_num,edge_num) )
            edges = self.ffi.new('Edge[%d]' % (edge_num) )
            cmatching = self.ffi.new('int[%d]' % (2 * node_num) )

            node2id = { val: index for index, val in enumerate(graph.nodes()) }
            id2node = {v: k for k, v in node2id.iteritems()}
            # print(node2id)

            e = 0
            for u,v in graph.edges():
                uid = int(node2id[u])
                vid = int(node2id[v])
                wt = -int(graph[u][v]['weight']) # weights from NetworkX
                # print('weight of edge[{0}][{1}] = {2}'.format( uid, vid, wt) )
                edges[e].uid = uid; edges[e].vid = vid; edges[e].weight = wt
                e += 1

            # print('printing edges before calling blossom')
            # for e in range(edge_num):
            #     print(edges[e].uid, edges[e].vid, edges[e].weight)

            retVal = self.blossom.Init()
            retVal = self.blossom.Process(node_num, edge_num, edges)
            # retVal = self.blossom.PrintMatching()
            nMatching = self.blossom.GetMatching(cmatching)
            retVal = self.blossom.Clean()

            pairs = []
            # print('recieved C matching :')
            for i in range(0, nMatching, 2):
                u, v = id2node[cmatching[i]], id2node[cmatching[i + 1]]
                pairs.append( (u, v) )
                # print( '{0}, {1} '.format(u,v) )

            #----------- end of c processing
        else:
            # Tom MWPM
            """
            bulk_vs = sorted(list([_ for _ in graph.nodes() if type(_) is int]))
            bdy_vs = sorted(list([_ for _ in graph.nodes() if type(_) is tuple]))
            
            if bulk_vs == []:
                return sp.I
            
            sz = len(bulk_vs) + 1
            weight_mat = np.zeros((sz, sz), dtype=np.int_)
            for r, c in it.product(range(1, sz), repeat=2):
                if r != c:
                    u, v = bulk_vs[r - 1], bulk_vs[c - 1]
                    weight_mat[r, c] = -graph[u][v]['weight']

            for dx in range(1, sz):
                u = bulk_vs[dx - 1]
                v = (u, 'b')
                weight_mat[0, dx] = -graph[u][v]['weight']
                weight_mat[dx, 0] = weight_mat[0, dx]

            # eliminate mixed sign
            min_wt = np.amin(weight_mat) - 1
            if min_wt != -1:
                for r, c in it.product(range(1, sz), repeat=2):
                    if weight_mat[r, c] != 0:
                        weight_mat[r, c] -= min_wt 
            
            # weight_mat = weight_mat.clip(0, np.inf)
            
            try:
                match_lst = bw.insert_wm(weight_mat)
            except:
                with open('error_wts.pkl', 'w') as phil:
                    pkl.dump(weight_mat, phil)
                raise ValueError("Tom's Blossom has gone wrong: "
                                    "weight_mat saved to error_wts.pkl.")
            
            redundant_pairs = [(n_lst[j], n_lst[k-1])
                                for j, k in enumerate(match_lst[1:])]

            """
            # """ NX MWPM
            matching = nx.max_weight_matching(graph, maxcardinality=True)
            redundant_pairs = matching.items()
            # """
            # get rid of non-digraph duplicates
            pairs = []
            for tpl in redundant_pairs:
                if tuple(reversed(tpl)) not in pairs:
                    pairs.append(tpl)
                    # print(tpl)


        x = self.layout.map.inv
        pauli_lst = []
        for u, v in pairs:
            if isinstance(u, int) & isinstance(v, int):
                pauli_lst.append(self.path_pauli(x[u], x[v], err))
            elif isinstance(u, int) ^ isinstance(v, int):
                bdy_pt = graph[u][v]['close_pt']
                vert = u if isinstance(u, int) else v
                pauli_lst.append(self.path_pauli(bdy_pt, x[vert], err))
            else:
                pass #both boundary points, no correction

        return product(pauli_lst)

    def graphAndCorrection(self, syndrome, err, dist_func=None,
                            return_matching=False):
        """
        Given a syndrome graph with edge weights, finds the
        maximum-weight perfect matching and produces a
        sparse_pauli.Pauli.

        Optional argument dist_func provides a way to customise
        distance calculations, you can put in a function that takes a
        pair of crds as arguments and spits out a (preferably small)
        positive number.

        Optional argument return_matching allows us to spit out a list
        of pairs of co-ordinates which can then be fed to re-weighting.
        """

        crds = self.layout.map.inv

        # calculate number of nodes and edges
        if self.boundary_conditions in ['rotated', 'open']:
            node_num = 2 * len(syndrome)
            edge_num = len(syndrome)
            for v1, v2 in it.combinations(syndrome, 2):
                edge_num = edge_num + 2 #TODO replace loop with n*(n-1)
            nodes = []
            for s in syndrome:
                n = str(s) + ', b'
                nodes.append(s)
                nodes.append(n)
        elif self.boundary_conditions == 'closed':
            node_num = len(syndrome)
            edge_num = (node_num * (node_num - 1))/2
            nodes = syndrome
            if dist_func is None:
                dist_func = lambda a, b: toric_dist(a, b, self.dx)

        # print( 'No of nodes : {0}, no of edges : {1}'.format(node_num,edge_num) )

        # generate nodes based on syndromes

        # print("nodes : " , nodes)

        # allocate edges and matching for c blossom
        edges = self.ffi.new('Edge[%d]' % (edge_num) )
        cmatching = self.ffi.new('int[%d]' % (2 * node_num) )

        # create node mapping
        node2id = { val: index for index, val in enumerate(nodes) }
        if PY3:
            id2node = {v: k for k, v in node2id.items()}
        else:
            id2node = {v: k for k, v in node2id.iteritems()}

        # generate edges
        e = 0
        for v1, v2 in it.combinations(syndrome, 2):
            uid = int(node2id[v1])
            vid = int(node2id[v2])

            if dist_func:
                wt = dist_func(crds[v1], crds[v2])
            else:
                wt = pair_dist(crds[v1], crds[v2])

            edges[e].uid = uid; edges[e].vid = vid; edges[e].weight = int(wt)

            e += 1

        if self.boundary_conditions == 'rotated':
            close_pts = {}
            for s in syndrome:
                v1 = s
                v2 = str(s) + ', b'
                uid = int(node2id[v1])
                vid = int(node2id[v2])
                bd_info = self.bdy_info(crds[s])
                close_pt = bd_info[1]
                if dist_func:
                    wt = dist_func(crds[s], close_pt)
                else:
                    wt = self.bdy_info(crds[s])[0]

                close_pts[(v1,v2)] = close_pt
                edges[e].uid = uid; edges[e].vid = vid; edges[e].weight = int(wt)
                e += 1

            for u, v in it.combinations(syndrome, 2):
                u1 = str(u)
                v1 = str(v)
                u2 = u1 + ', b'
                v2 = v1 + ', b'
                uid = int(node2id[u2])
                vid = int(node2id[v2])
                wt = 0
                edges[e].uid = uid; edges[e].vid = vid; edges[e].weight = wt
                e += 1

        # print( 'generated {0} edges.'.format(e) )

        # invoke c blossom
        retVal = self.blossom.Init()
        retVal = self.blossom.Process(node_num, edge_num, edges)
        # retVal = self.blossom.PrintMatching()
        nMatching = self.blossom.GetMatching(cmatching)
        retVal = self.blossom.Clean()

        pairs = []
        for i in range(0, nMatching, 2):
            u,v = id2node[cmatching[i]], id2node[cmatching[i+1]]
            pairs.append( (u,v) )

        pauli_lst = []
        matchups = []
        for u, v in pairs:
            if isinstance(u, int) & isinstance(v, int):
                pauli_lst.append(self.path_pauli(crds[u], crds[v], err))
                matchups.append((crds[u], crds[v]))
            elif isinstance(u, int) ^ isinstance(v, int):
                bdy_pt = close_pts[(u,v)]
                vert = u if isinstance(u, int) else v
                pauli_lst.append(self.path_pauli(bdy_pt, crds[vert], err))
                matchups.append(crds[vert]) #bdy matches roll solo
            else:
                pass #both boundary points, no correction

        return matchups if return_matching else product(pauli_lst)

    def beliefs(self, err, bp_rounds=None):
        """
        Produces propagated beliefs for a simulation, which we can use
        in bp_correction or else in run. 
        """
        return mw.input_beliefs(self, err, bp_rounds)
    
    def logical_error(self, error, x_corr, z_corr):
        """
        Given an error and a correction, multiplies them and returns a
        single letter recording the resulting logical error (may be I,
        X, Y or Z)
        """
        loop = error * x_corr * z_corr

        if self.boundary_conditions in ['rotated', 'open']:
            anticom_dict = {
                            ( 0, 0 ) : 'I',
                            ( 0, 1 ) : 'X',
                            ( 1, 0 ) : 'Z',
                            ( 1, 1 ) : 'Y'
                        }
            x_bar, z_bar = self.layout.logicals()
            com_tpl = x_bar.com(loop), z_bar.com(loop)
        elif self.boundary_conditions == 'closed':
            anticom_dict = {
                            (0, 0, 0, 0) : 'II',
                            (0, 0, 0, 1) : 'IX',
                            (0, 0, 1, 0) : 'XI',
                            (0, 0, 1, 1) : 'XX',
                            (0, 1, 0, 0) : 'IZ',
                            (0, 1, 0, 1) : 'IY',
                            (0, 1, 1, 0) : 'XZ',
                            (0, 1, 1, 1) : 'XY',
                            (1, 0, 0, 0) : 'ZI',
                            (1, 0, 0, 1) : 'ZX',
                            (1, 0, 1, 0) : 'YI',
                            (1, 0, 1, 1) : 'YX',
                            (1, 1, 0, 0) : 'ZZ',
                            (1, 1, 0, 1) : 'ZY',
                            (1, 1, 1, 0) : 'YZ',
                            (1, 1, 1, 1) : 'YY'
                        }

            logs = self.layout.logicals()
            com_tpl = tuple(op.com(loop) for op in logs)
        return anticom_dict[com_tpl]

    def run(self, n_trials, verbose=False, progress=True,
                dist_func=None, bp=False, bp_rounds=None):
        """
        Repeats the following cycle `n_trials` times:
         + Generate a random error
         + determine syndromes by checking stabilisers
         + make those syndromes into a graph with boundary vertices
         + match on that graph
         + check for a logical error by testing anticommutation with
           the logical paulis
        """
        bar = pb.ProgressBar()
        trials = range( int(n_trials) )
        trials = bar(trials) if progress else trials

        # self.layout.Print() # textual print of surface
        # self.layout.Draw() # graphical print of surface

        for trial in trials:
            err = self.random_error()
            x_synd, z_synd = self.syndromes(err)
            if bp:
                blfs = self.beliefs(err, bp_rounds=bp_rounds)
                # x_corr, z_corr = bp_correction(blfs)
                # if self.syndromes(x_corr * z_corr) != self.syndromes(err):
                # x_graph, z_graph = mw.bp_graphs(self, err, beliefs=blfs)
                x_graph, z_graph = mw.path_sum_graphs(self, err, beliefs=blfs)
                x_corr = self.correction(x_graph, 'Z')
                z_corr = self.correction(z_graph, 'X')
            else:            
                if self.useBlossom:
                    # without networkx interface (only with blossom)
                    x_corr = self.graphAndCorrection(x_synd, 'Z', dist_func=dist_func)
                    z_corr = self.graphAndCorrection(z_synd, 'X', dist_func=dist_func)
                else:
                    # with networkx interface (with/without blossom)
                    x_graph = self.graph(x_synd, dist_func=dist_func)
                    z_graph = self.graph(z_synd, dist_func=dist_func)
                    x_corr = self.correction(x_graph, 'Z')
                    z_corr = self.correction(z_graph, 'X')

            log = self.logical_error(err, x_corr, z_corr)
            self.errors[log] += 1
            if verbose:
                print_lst = [("error: {}",  err),
                    ("X syndrome: {}",  x_synd),
                    ("Z syndrome: {}",  z_synd),
                    ("X correction: {}",  x_corr),
                    ("Z correction: {}",  z_corr),
                    ("logical error: {}",  log)]
                if self.useBlossom == False:
                    print_lst.extend([
                        ("x_graph: {}", "\n".join(map(str,list(x_graph.edges(data=True))))),
                        ("z_graph: {}", "\n".join(map(str,list(z_graph.edges(data=True)))))
                        ])

                print("\nTrial status")
                print("\n".join([ st.format(obj)
                                for st, obj in print_lst]))

        # graphical print of surface with syndromes
        # self.layout.DrawSyndromes( x_graph.nodes(), z_graph.nodes() )

    def bdy_info(self, crd):
        """
        Returns the minimum distance between the input co-ordinate and
        one of the acceptable boundary vertices, depending on
        syndrome type (X or Z).
        """
        if self.boundary_conditions == 'closed':
            return None, None

        if self.boundary_conditions == 'open':
            dist_func = lambda c0, c1: toric_dist(c0, c1, None)
        elif self.boundary_conditions == 'rotated':
            dist_func = pair_dist
        
        min_dist = 4 * max(self.dx, self.dy) #any impossibly large value will do
        err_type = 'Z' if crd in self.layout.x_ancs() else 'X'
        for pt in self.layout.boundary_points(err_type):
            new_dist = dist_func(crd, pt)
            if new_dist < min_dist:
                min_dist, close_pt = new_dist, pt

        return min_dist, close_pt

    def path_pauli(self, crd_0, crd_1, err_type):
        """
        Returns a minimum-length Pauli between two ancillas, given the
        type of error that joins the two.

        This function is awkward, because it works implicitly on the
        un-rotated surface code, first finding a "corner" (a place on
        the lattice for the path to turn 90 degrees), then producing
        two diagonal paths on the rotated lattice that go to and from
        this corner.
        """
        err_type = err_type.upper()
        if self.boundary_conditions == 'open':
            pth_0 = [
                        (x, crd_0[1])
                        for x in short_seq(crd_0[0], crd_1[0], None)
                    ]
            pth_1 = [
                        (crd_1[0], y)
                        for y in short_seq(crd_0[1], crd_1[1], None)
                    ]
        if self.boundary_conditions == 'rotated':
            mid_v = diag_intersection(crd_0, crd_1, self.layout.ancillas.values())

            pth_0, pth_1 = diag_pth(crd_0, mid_v), diag_pth(mid_v, crd_1)

        elif self.boundary_conditions == 'closed':
            pth_0 = [
                        (x, crd_0[1])
                        for x in short_seq(crd_0[0], crd_1[0], self.dx)
                    ]
            pth_1 = [
                        (crd_1[0], y)
                        for y in short_seq(crd_0[1], crd_1[1], self.dy)
                    ]

        #path on lattice, uses idxs
        p = [self.layout.map[crd] for crd in list(pth_0) + list(pth_1)]
        pl = sp.X(p) if err_type == 'X' else sp.Z(p)
        return pl


#-----------------------convenience functions-------------------------#
product = lambda itrbl: reduce(mul, itrbl, sp.Pauli())

def short_seq(a, b, l):
    """
    I'm trying to figure out whether 'tis nobler to step from a to b
    around one direction on a torus l squares wide, or whether to go
    the other way.
    """
    d = abs(b - a)
    if (l is None) or (d < 2*l - d):
        return range(min(a, b) + 1, max(a, b), 2)
    else:
        return range(min(a, b) - 1, -1, -2) + range(max(a, b) + 1, 2*l, 2)

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
    mid_v = diag_intersection(crd_0, crd_1)
    diff_vs = [
                [abs(a - b) for a, b in zip(v, mid_v)]
                for v in [crd_0, crd_1]
            ]
    if (diff_vs[0][0] == diff_vs[0][1]) and (diff_vs[1][0] == diff_vs[1][1]):
        return (diff_vs[0][0] + diff_vs[1][0]) / 2 # intdiv
    else:
        raise ValueError("math don't work")

def toric_dist(crd_0, crd_1, l):
    """
    Because of the frame rotation when switching between closed and
    rotated BC, we have a separate distance function which takes into
    account the minimum length path on a torus.
    """
    dx, dy = abs(crd_0[0] - crd_1[0]), abs(crd_0[1] - crd_1[1])
    if l:
        return (min(dx, 2 * l - dx) + min(dy, 2 * l - dy)) / 2 # intdiv
    else:
        return dx + dy

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

def diag_intersection(crd_0, crd_1, ancs=None):
    """
    Returns an appropriate cornering point for a path on the lattice by
    testing potential corners to determine if they're on the lattice.
    """
    vs = corners(crd_0, crd_1)

    if ancs:
        if vs[0] in sum(list(ancs), []):
            mid_v = vs[0]
        else:
            mid_v = vs[1]
    else:
        mid_v = vs[0]

    return mid_v

def corners(crd_0, crd_1):
    """
    Uses a little linear algebra to determine where diagonal paths that
    step outward from ancilla qubits intersect.
    This allows us to reduce the problems of length-finding and
    path-making to a pair of 1D problems.
    """
    a, b, c, d = crd_0[0], crd_0[1], crd_1[0], crd_1[1]
    vs = [( int(( d + c - b + a ) / 2), int(( d + c + b - a ) / 2 )),
        ( int(( d - c - b - a ) / -2), int(( -d + c - b - a ) / -2 ))]
    return vs

def bp_correction(blfs):
    """
    Runs belief propagation and takes the argmax of the resulting
    beliefs. 
    If the syndromes of the actual error and the BP prediction
    match, we return a set of corrections (X and Z to conform with
    the rest of the simulation). 
    If not, we return None, so the calling method can pass the beliefs 
    """
    p_dct = {key: np.argmax(val) for key, val in blfs.items()}
    p_dct = {key: val for key, val in p_dct.items() if val != 0}
    pauli = sp.I
    for idx in range(1, 4):
        pauli *= _pauli_key[idx]([k for k in p_dct if p_dct[k] == idx])

    return pauli.xz_pair()

#---------------------------------------------------------------------#
