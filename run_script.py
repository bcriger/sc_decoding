import decoding_2d as dc2
import decoding_3d as dc3
import error_model as em
import circuit_metric as cm
import numpy as np
import pickle as pkl
from scipy.linalg import block_diag

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

def run_batch(err_lo, err_hi, n_points, dists, n_trials, flnm, sim_type='iidxz'):
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
                dist_func = fancy_dist(dist, err)
                current_sim = dc2.Sim2D(dist, dist, err, useBlossom=False)
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

if __name__ == '__main__':
    from sys import argv
    err_lo, err_hi = map(float, argv[1:3])
    n_points = int(argv[3])
    dists = list(map(int, argv[4].split(' ')))
    n_trials = int(argv[5])
    flnm = argv[6]
    sim_type = argv[7] if len(argv) > 7 else 'iidxz'
    run_batch(err_lo, err_hi, n_points, dists, n_trials, flnm, sim_type)