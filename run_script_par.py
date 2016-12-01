import decoding_2d as dc2
import decoding_3d as dc3
import pickle as pkl
import numpy as np

def run_batch(err_lo, err_hi, n_points, dists, n_trials, flnm, sim_type='iidxz'):
    """
    Makes a bunch of simulation objects and runs them, based on input
    parameters.
    """
    sim_type = sim_type.lower()
    if sim_type not in ['iidxz', 'pq', 'circ']:
        raise ValueError("sim_type must be one of: ['iidxz', 'pq', 'circ']")

    errs = np.linspace(err_lo, err_hi, n_points)
    output_dict = locals()

    for dist in dists:
        failures = []
        for err in errs:
            if sim_type == 'iidxz':
                current_sim = dc2.Sim2D(dist, dist, err)
            elif sim_type == 'pq':
                current_sim = dc3.Sim3D(dist, dist, ('pq', err, err))
            elif sim_type == 'circ':
                raise NotImplementedError("Coming Soon!")

            current_sim.run(n_trials, progress=False)
            failures.append(current_sim.errors)
        output_dict.update({'failures ' + str(dist) : failures})

    with open(flnm, 'wb') as phil:
        pkl.dump(output_dict, phil)

    pass

from multiprocessing import Process
from timeit import default_timer as timer

def run_batch_par(err_lo, err_hi, n_points, dists, n_trials, flnm, sim_type, nThreads):
    jobs = []
    trialsPerThread = n_trials/nThreads

    for i in range(nThreads):
        fname = "out" + str(i) + ".dat"
        p = Process(target=run_batch, args=(err_lo, err_hi, n_points, dists, trialsPerThread, fname, sim_type,))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

if __name__ == '__main__':
    from sys import argv
    nThreads = 4
    n_trials =nThreads*16 # keep it a multiple of nThreads for now
    err_lo = 0.01
    err_hi = 0.03
    n_points = 30
    dists = [25]
    flnm = 'out.dat'
    sim_type = 'iidxz'

    # just for benchmarking/testing
    # comment for actual run
    print("Sequential execution ...")
    start = timer()
    run_batch(err_lo, err_hi, n_points, dists, n_trials, flnm, sim_type)
    end = timer()
    seqtime = (end-start)
    print("Sequential execution took : {0} ms".format(seqtime*1e3))

    print("Parallel execution ...")
    start = timer()
    run_batch_par(err_lo, err_hi, n_points, dists, n_trials, flnm, sim_type, nThreads)
    end = timer()
    partime = (end-start)
    print("Parallel execution on {0} threads took {1} ms ".format(nThreads,partime*1e3) )
