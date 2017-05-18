import networkx as nx
import matched_weights as mw
import decoding_2d as dc2
import operator as op
from circuit_metric.layout_utils import *
import sparse_pauli as sp

e_shifts = [shift for shift in mw.big_shifts if shift[0] == 2]
n_shifts = [shift for shift in mw.big_shifts if shift[1] == 2]

def bravyi_weight_graphs(sim, syndromes, cosets):
    """
    Takes as input:
     - a simulation object
     - x/z syndromes

    Returns:
     - NetworkX graphs with appropriate tiles as vertices, plus two
       extra to sum the boundaries

    Hardcoded directions corresponding to X/Z ancillas
    """
    layout = sim.layout

    # chex
    if layout.dx != layout.dy:
        raise ValueError('dx != dy')

    # contractions
    
    crd2int = layout.map
    bdy = layout.boundary
    ancs = layout.ancillas
    d = layout.dx

    # dumb corrections tell us where to put (1 - p) / p
    x_coset, z_coset = cosets
    
    # produce weighted, directed graphs
    mdl = sim.error_model.p_arr # IZXY ordering
    p_xy = mdl[2] + mdl[3]
    p_zy = mdl[1] + mdl[3]
    w_x = p_xy / (1. - p_xy)
    w_z = p_zy / (1. - p_zy)

    x_pts = summ([bdy[_] for _ in ('x_left', 'x_right')])
    x_pts += summ([ancs[_] for _ in ('x_top', 'x_sq', 'x_bot')])
    g_x = nx.DiGraph()
    
    g_x.add_nodes_from(bdy['x_left'])    

    frontier = set(g_x.nodes())
    for col in range(d):
        new_frontier = set()
        for pt in frontier:
            for shift in e_shifts:
                nu_pt = ad(pt, shift)
                if nu_pt in x_pts:
                    p = (1. - p_xy) if crd2int[av(pt, nu_pt)] in z_coset.support() else p_xy
                    g_x.add_edge(pt, nu_pt, p_edge = p)
                    new_frontier |= {nu_pt}

        frontier = new_frontier

    for pt in bdy['x_left']:
        g_x.add_edge('x_left', pt, p_edge = 0.5)

    for pt in bdy['x_right']:
        g_x.add_edge(pt, 'x_right', p_edge = 0.5)

    z_pts = summ([bdy[_] for _ in ('z_top', 'z_bot')])
    z_pts += summ([ancs[_] for _ in ('z_left', 'z_sq', 'z_right')])
    g_z = nx.DiGraph()

    # copypasta
    g_z.add_nodes_from(bdy['z_bot'])    

    frontier = set(g_z.nodes())
    for col in range(d):
        new_frontier = set()
        for pt in frontier:
            for shift in n_shifts:
                nu_pt = ad(pt, shift)
                if nu_pt in z_pts:
                    p = (1. - p_zy) if crd2int[av(pt, nu_pt)] in x_coset.support() else p_zy
                    g_z.add_edge(pt, nu_pt, p_edge = p)
                    new_frontier |= {nu_pt}

        frontier = new_frontier

    for pt in bdy['z_bot']:
        g_z.add_edge('z_bot', pt, p_edge = 0.5)

    for pt in bdy['z_top']:
        g_z.add_edge(pt, 'z_top', p_edge = 0.5)
    
    return g_x, g_z

def bravyi_run(sim, n_trials):
    """
    Basically a new run method for a dc2.Sim2D that uses bravyi weights
    """
    trials = range( int(n_trials) )

    for trial in trials:
        err = sim.random_error()
        syndromes = sim.syndromes(err)
        cosets = sim.dumb_correction(syndromes)
        g_x, g_z = bravyi_weight_graphs(sim, syndromes, cosets)
        # determine correctness
        p_z, p_x = mw.path_prob(g_x), mw.path_prob(g_z)
        l_x, l_z = sim.layout.logicals()
        x_corr, z_corr = cosets
        z_corr *= l_z if p_z > 0.5 else sp.I
        x_corr *= l_x if p_x > 0.5 else sp.I
        sim.errors[sim.logical_error(err, x_corr, z_corr)] += 1

    #return sim.errors

#------------------------convenience functions------------------------#

def av(crd_0, crd_1):
    """
    Averages two co-ordinates to find the spot in the middle.
    """
    return (crd_0[0] + crd_1[0]) / 2, (crd_0[1] + crd_1[1]) / 2

summ = lambda lst: reduce(op.add, lst)