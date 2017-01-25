import run_script as rs
import plot_utils as pu
from pylab import plot
import numpy as np

err_lo, err_hi, n_points, dists, n_trials, flnm = 0.05, 0.15, 25, [3], 25000, 'dist_3_dep_test.pkl'
sim_type = 'dep'
bc = 'rotated'
rs.run_batch(err_lo, err_hi, n_points, dists, n_trials, flnm, sim_type=sim_type, bc=bc)
pu.error_rate_plot(flnm)
plot(np.linspace(err_lo, err_hi, n_points), np.linspace(err_lo, err_hi, n_points))