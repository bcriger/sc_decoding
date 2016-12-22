import decoding_2d as dc2
import decoding_3d as dc3
import error_model as em
import numpy as np
import pickle as pkl

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
                current_sim = dc2.Sim2D(dist, dist, err)
            elif sim_type == 'pq':
                current_sim = dc3.Sim3D(dist, dist, ('pq', err, err))
            elif sim_type == 'circ':
                current_sim = dc3.Sim3D(dist, dist, ('fowler', err))
            
            if sim_type == 'dep':
                current_sim.error_model = em.depolarize(err,
                    [[current_sim.layout.map[_]] for _ in current_sim.layout.datas])

            current_sim.run(n_trials, progress=False)
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