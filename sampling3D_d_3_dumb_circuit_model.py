from copy import deepcopy
import decoding_2d as dc2
import decoding_3d as dc3
import error_model as em
from functools import reduce
import itertools as it
import numpy as np
from operator import mul, or_ as union, xor as sym_diff
from run_script import fancy_dist
from scipy.special import betainc
import sparse_pauli as sp

d = 3
ms_rnd = 12
no_anc = int((d + 1) * (d - 1) / 2 )
p = 0.01
q = 0.01
samples = {}
z_ancs = [0,5,11,16]
x_ancs = [4,6,10,12]
flp = [0] * no_anc
flips_Z = [0] * (ms_rnd+1)
flips_X = [0] * (ms_rnd+1)
for i in range(ms_rnd+1):
    flips_Z[i] = deepcopy(flp)
    flips_X[i] = deepcopy(flp)
no_samples = 0
sim_test2d = dc2.Sim2D(d, d, p)
sim_test = dc3.Sim3D(d, ms_rnd, ('fowler', p), True)

#for cycl in range(1000000):
cycl = 0
while no_samples < 1000000:
    err, synd = sim_test.history(True)
    rnd_err_x = err[-1].x_set
    rnd_err_z = err[-1].z_set
    rd_er = sp.Pauli(list(rnd_err_x),list(rnd_err_z))
    #corr_blossom = sim_test.correction(synd, metric=None, bdy_info=None)
    #mwpm_log_err = sim_test.logical_error(err[-1], corr_blossom)
    synds_2d = sim_test2d.syndromes(rd_er)
    dumb_x_corr, dumb_z_corr = sim_test2d.dumb_correction(synds_2d, False)
    dumb_log_err = sim_test2d.logical_error(rd_er, dumb_x_corr, dumb_z_corr)
    
    #for j in range(d+1):
    for j in range(ms_rnd+1):
        for i in range(no_anc):
            if (z_ancs[i] in synd['Z'][j] and z_ancs[i] in synd['Z'][j+1]) or \
               (z_ancs[i] not in synd['Z'][j] and z_ancs[i] not in synd['Z'][j+1]):
                flips_Z[j][i] = 0
            else:
                flips_Z[j][i] = 1
    
    #for j in range(d+1):
    for j in range(ms_rnd+1):
        for i in range(no_anc):
            if (x_ancs[i] in synd['X'][j] and x_ancs[i] in synd['X'][j+1]) or \
               (x_ancs[i] not in synd['X'][j] and x_ancs[i] not in synd['X'][j+1]):
                flips_X[j][i] = 0
            else:
                flips_X[j][i] = 1
    
    s = ''
    #for j in range(d+1):
    for j in range(ms_rnd+1):
        for k in range(no_anc):
            s += str(flips_Z[j][k])
    
    #for j in range(d+1):
    for j in range(ms_rnd+1):
        for k in range(no_anc):
            s += str(flips_X[j][k])
    
    if s in samples:
        x = samples[s]
        if dumb_log_err == 'I':
            x[0] += 1
        elif dumb_log_err == 'X':
            x[1] += 1
        elif dumb_log_err == 'Z':
            x[2] += 1
        else:
            x[3] += 1
        samples[s] = [x[0], x[1], x[2], x[3], x[4]+1]
    else:
        no_samples += 1
        x = [0,0,0,0, 1]
        if dumb_log_err == 'I':
            x[0] = 1
        elif dumb_log_err == 'X':
            x[1] = 1
        elif dumb_log_err == 'Z':
            x[2] = 1
        else:
            x[3] = 1
        samples[s] = x
        
    cycl += 1            
    if cycl % 1000 == 0:
        print(no_samples, cycl)

print(no_samples)

sorted_samples = sorted(samples.items(), key=lambda e: e[1][4], reverse=True)
with open('d='+str(d)+'_p='+str(p)+'_'+str(len(samples))+'_samples_20_rounds.txt', 'w') as f:
    #pl = 2 * (d+1) * no_anc
    pl = 2 * (ms_rnd+1) * no_anc
    for key, value in sorted_samples:
        x = ''
        for i in range(pl):
            x += key[i] + ' '

        x += ' '
        x += str(value[0]) + ' '
        x += str(value[1]) + ' '
        x += str(value[2]) + ' '
        x += str(value[3]) + ' '
        x += str(value[4]) + '\n'
        f.write(x)