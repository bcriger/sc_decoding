import decoding_2d as dc
import pickle as pkl
import numpy as np

def run_batch(err_lo, err_hi, n_points, dists, n_trials, flnm):
    """
    Makes a bunch of simulation objects and runs them, based on input
    parameters. 
    """

    errs = np.linspace(err_lo, err_hi, n_points)
    output_dict = locals()

    for dist in dists:
        failures = []
        for err in errs:
            current_sim = dc.Sim2D(dist, err)
            current_sim.run(n_trials, progress=False)
            failures.append(current_sim.errors)
        output_dict.update({'failures ' + str(dist) : failures})

    with open(flnm, 'wb') as phil:
        pkl.dump(output_dict, phil)

    pass

if __name__ == '__main__':
    from sys import argv
    print(argv)
    err_lo, err_hi = map(float, argv[1:3])
    n_points = int(argv[3])
    dists = list(map(int, argv[4].split(' ')))
    n_trials = int(argv[5])
    flnm = argv[6]
    run_batch(err_lo, err_hi, n_points, dists, n_trials, flnm)