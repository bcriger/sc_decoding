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

def fancy_dist(d, p, precision=2, bc='rotated'):
    """
    Turns the nice list/matrix description output by 
    circuit_metric.bit_flip_metric into a function with co-ordinates as
    arguments.
    """
    z_crds, z_mat = cm.bit_flip_metric(d, p, 'z', bc, True, d)
    x_crds, x_mat = cm.bit_flip_metric(d, p, 'x', bc, True, d)
    crds = x_crds + z_crds
    mat = (10**precision * block_diag(x_mat, z_mat)).astype(np.int_)
    
    def dist_func(crd0, crd1):
        return mat[crds.index(crd0), crds.index(crd1)]

    return dist_func

def entropic_dist(l, p, precision=4):
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
    odds = p / (1. - p)
    log_odds = np.log(odds)
    def dist_func(crd_0, crd_1):
        dx = abs(crd_0[0] - crd_1[0]) / 2  #intdiv
        dy = abs(crd_0[1] - crd_1[1]) / 2  #intdiv
        dx = min(dx, l - dx) #uh-hahahahaha
        dy = min(dy, l - dy) #uh-hahahahaha
        nc = binom(dx + dy, dx)
        omega_ratio = float(dx * (dx + 1)**2 + dy * (dy + 1)**2) / float((dx + 1) * (dy + 1))
        entropic_correction = np.log(nc) / log_odds
        second_correction = np.log(1. + omega_ratio * odds ** 2) / log_odds
        float_weight =  dx + dy + entropic_correction + second_correction
        
        if precision is None:
            return float_weight
        else:
            return int(float_weight * 10 ** precision)

    return dist_func

def run_batch(err_lo, err_hi, n_points, dists, n_trials, flnm, sim_type='iidxz', bc='rotated', bp=False):
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
                # current_sim = dc2.Sim2D(dist, dist, err, useBlossom=False, boundary_conditions=bc)
                # dist_func = fancy_dist(dist, err, bc=bc)
                dist_func = entropic_dist(dist, err)
                if bc == 'open':
                    current_sim = dc2.Sim2D(dist, dist+1, err, useBlossom=False, boundary_conditions=bc)
                else:
                    current_sim = dc2.Sim2D(dist, dist, err, useBlossom=False, boundary_conditions=bc)
            elif sim_type == 'pq':
                current_sim = dc3.Sim3D(dist, dist, ('pq', err, err))
            elif sim_type == 'circ':
                current_sim = dc3.Sim3D(dist, dist, ('fowler', err))
                met_fun = dc3.nest_metric(current_sim, err)
                bdy_fun = dc3.nest_bdy_tpl(current_sim, err)
            
            if sim_type == 'dep':
                current_sim.error_model = em.depolarize(err,
                    [[current_sim.layout.map[_]] for _ in current_sim.layout.datas])
                dist_func = fancy_dist(dist, 0.66666667 * err)

            current_sim.run(n_trials, progress=False, bp=bp)
            # current_sim.run(n_trials, progress=False,
                            # metric=met_fun, bdy_info=bdy_fun)
            # current_sim.run(n_trials, progress=False, dist_func=dist_func)
            failures.append(current_sim.errors)
        output_dict.update({'failures ' + str(dist) : failures})

    with open(flnm, 'wb') as phil:
        pkl.dump(output_dict, phil)

    pass

if __name__ == '__main__':
    from sys import argv
    err_lo, err_hi = map(float, argv[1:3])
    n_points = int(argv[3])
    dists = list(map(int, argv[4].split(' ')))
    n_trials = int(argv[5])
    flnm = argv[6]
    sim_type = argv[7] if len(argv) > 7 else 'iidxz'
    bc = 'rotated'
    bp = False
    run_batch(err_lo, err_hi, n_points, dists, n_trials, flnm, sim_type, bc, bp)

#---------------------------------------------------------------------#
