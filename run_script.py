import circuit_metric as cm
import decoding_2d as dc2
import decoding_3d as dc3
import error_model as em
import itertools as it
import matched_weights as mw
import numpy as np
import pickle as pkl
from scipy.linalg import block_diag
from scipy.special import binom

SHIFTS = [(2, 2), (2, -2), (-2, 2), (-2, -2)]

def fancy_dist(d, p):
    """
    Turns the nice list/matrix description output by 
    circuit_metric.bit_flip_metric into a function with co-ordinates as
    arguments.
    """
    z_crds, z_mat = cm.bit_flip_metric(d, p, 'z')
    x_crds, x_mat = cm.bit_flip_metric(d, p, 'x')
    crds = x_crds + z_crds
    mat = block_diag(x_mat, z_mat)
    
    def dist_func(crd0, crd1):
        return mat[crds.index(crd0), crds.index(crd1)]

    return dist_func

def entropic_dist(l, p, precision=2):
    """
    I'm going to add a term to the toric distance to try to account
    for degeneracy in the minimum-weight paths.
    To a very crude first approximation, the probability of a
    minimum-length path between two syndromes will just be multiplied
    by the number of chains of that length. 
    Then, instead of |C| as the edge weight, we obtain
    |C| + ln(num_chains) / ln(p / (1 - p)).
    Note that, for now, I'm going to constrain the matching to use only
    minimum-length paths, even if the entropic term would make a longer
    path likelier.

    To make this function return an integer, I multiply the weight by
    10**precision and cast to integer.
    """
    log_odds = np.log(p / (1. - p))
    def dist_func(crd_0, crd_1):
        dx = abs(crd_0[0] - crd_1[0]) / 2  #intdiv
        dy = abs(crd_0[1] - crd_1[1]) / 2  #intdiv
        delta_x = min(dx, l - dx)
        delta_y = min(dy, l - dy)
        nc = binom(delta_x + delta_y, delta_x)
        entropic_correction = np.log(nc) / log_odds
        float_weight =  delta_x + delta_y + entropic_correction
        return int(float_weight * 10 ** precision)

    return dist_func

def run_batch(err_lo, err_hi, n_points, dists, n_trials, flnm, sim_type='iidxz', bc='rotated'):
    """
    Makes a bunch of simulation objects and runs them, based on input
    parameters. 
    """
    sim_type = sim_type.lower()
    if sim_type not in ['iidxz', 'dep', 'pq', 'circ']:
        raise ValueError("sim_type must be one of: ['iidxz', 'dep', 'pq', 'circ']")
    
    errs = np.linspace(err_lo, err_hi, n_points)
    output_dict = locals()

    for dist in dists:
        failures = []
        for err in errs:
            if sim_type in ['iidxz', 'dep']:
                # dist_func = fancy_dist(dist, err)
                # current_sim = dc2.Sim2D(dist, dist, err, useBlossom=False, boundary_conditions=bc)
                dist_func = entropic_dist(dist, err)
                current_sim = dc2.Sim2D(dist, dist, err, useBlossom=True, boundary_conditions=bc)
            elif sim_type == 'pq':
                current_sim = dc3.Sim3D(dist, dist, ('pq', err, err))
            elif sim_type == 'circ':
                current_sim = dc3.Sim3D(dist, dist, ('fowler', err))
            
            if sim_type == 'dep':
                current_sim.error_model = em.depolarize(err,
                    [[current_sim.layout.map[_]] for _ in current_sim.layout.datas])
                dist_func = fancy_dist(dist, 0.66666667 * err)

            current_sim.run(n_trials, progress=False, dist_func=dist_func)
            failures.append(current_sim.errors)
        output_dict.update({'failures ' + str(dist) : failures})

    with open(flnm, 'wb') as phil:
        pkl.dump(output_dict, phil)

    pass

def run_corr_dep(err_lo, err_hi, n_points, dists, n_trials, flnm):
    """
    special purpose loop to run the ad hoc correlated error decoder.
    """
    errs = np.linspace(err_lo, err_hi, n_points)
    output_dict = locals()
    for dist in dists:
        failures = []
        for err in errs:
            failures.append(corr_decode_test(dist, err, n_trials))
        output_dict.update({'failures ' + str(dist) : failures})

    with open(flnm, 'wb') as phil:
        pkl.dump(output_dict, phil)

    pass

def corr_decode_test(dist, p_dep, n_trials):
    """
    Let's knock together a re-weighting decoder for depolarizing
    errors.
    I'm going to do a round of independent matchings (X and Z), then a
    round of re-weighted matchings (X and Z). 
    """
    sim = dc2.Sim2D(dist, dist, p_dep)
    sim.error_model = em.depolarize(p_dep,
                    [[sim.layout.map[_]] for _ in sim.layout.datas])
    x_bdy_vs = sim.layout.boundary_points('z')
    x_bulk_vs = sim.layout.x_ancs()
    z_bdy_vs = sim.layout.boundary_points('x')
    z_bulk_vs = sim.layout.z_ancs()
    x_vs = sorted(x_bdy_vs + x_bulk_vs)
    z_vs = sorted(z_bdy_vs + z_bulk_vs)
    vs = x_vs + z_vs
    mdl = sim.error_model.p_arr
    for _ in range(n_trials):
        err = sim.random_error()
        x_synd, z_synd = sim.syndromes(err)
        sim.useBlossom = True
        x_matches = sim.graphAndCorrection(x_synd, 'z', return_matching=True)
        z_matches = sim.graphAndCorrection(z_synd, 'x', return_matching=True)
        x_mat = mw.matching_p_mat(x_matches, x_bdy_vs, x_bulk_vs, mdl, 'x')
        z_mat = mw.matching_p_mat(z_matches, z_bdy_vs, z_bulk_vs, mdl, 'z')
        temp_mat = np.zeros_like(x_mat) #x_mat/z_mat here both good.

        #put x_mat in temp_mat
        for r, c in it.product(range(len(x_vs)), repeat=2):
            crds = (x_vs[r], x_vs[c])
            diff = (crds[0][0] - crds[1][0], crds[0][1] - crds[1][1])
            if diff in SHIFTS:
                crds_p = mw.nn_edge_switch(crds)
                r_p, c_p = z_vs.index(crds_p[0]), z_vs.index(crds_p[1])
                temp_mat[r_p, c_p] = x_mat[r, c]

        #put z_mat in x_mat
        x_mat = np.zeros_like(x_mat) #x_mat/z_mat here both good.
        for r, c in it.product(range(len(x_vs)), repeat=2):
            crds = (z_vs[r], z_vs[c])
            diff = (crds[0][0] - crds[1][0], crds[0][1] - crds[1][1])
            if diff in SHIFTS:
                crds_p = mw.nn_edge_switch(crds)
                r_p, c_p = x_vs.index(crds_p[0]), x_vs.index(crds_p[1])
                x_mat[r_p, c_p] = z_mat[r, c]

        #put temp_mat in z_mat
        z_mat = temp_mat

        x_mat = cm.fancy_weights(x_mat)
        z_mat = cm.fancy_weights(z_mat)
        big_mat = block_diag(x_mat, z_mat)
        def dist_func(crd0, crd1):
            return big_mat[vs.index(crd0), vs.index(crd1)]

        sim.useBlossom = False
        x_graph = sim.graph(x_synd, dist_func=dist_func)
        z_graph = sim.graph(z_synd, dist_func=dist_func)
        x_corr = sim.correction(x_graph, 'Z')
        z_corr = sim.correction(z_graph, 'X')

        log = sim.logical_error(err, x_corr, z_corr)
        sim.errors[log] += 1
    
    return sim.errors

if __name__ == '__main__':
    from sys import argv
    err_lo, err_hi = map(float, argv[1:3])
    n_points = int(argv[3])
    dists = list(map(int, argv[4].split(' ')))
    n_trials = int(argv[5])
    flnm = argv[6]
    sim_type = argv[7] if len(argv) > 7 else 'iidxz'
    bc = 'closed'
    run_batch(err_lo, err_hi, n_points, dists, n_trials, flnm, sim_type, bc)
    # run_corr_dep(err_lo, err_hi, n_points, dists, n_trials, flnm)

#---------------------------------------------------------------------#