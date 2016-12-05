import decoding_2d as dc
import numpy as np
from operator import itemgetter

cnt = []
lst_z = []
list_train = []
list_target = []
dumb_logical = 0
cycles = 0
distance = 7
no_anc = 24#((distance * distance)-1)/2
samples = {}

while cycles < 1000000:
    cycles += 1
    physprob = 0.005
    sim_test = dc.Sim2D(distance, physprob)
    z_ancs_keys = list(sim_test.layout.z_ancs())

    rnd_err = sim_test.random_error()
    synds = sim_test.syndromes(rnd_err)

    list_z = [0] * len(z_ancs_keys)
    for k in synds[1]:
        key = sim_test.layout.map.inv[k]
        pos = z_ancs_keys.index(key)
        list_z[pos] = 1

    s = ''
    for i in range(no_anc):
        s += str(list_z[i])

    if s in samples:
        x = samples[s]
        samples[s] = [x[0], x[1]+1]
    else:
        dumb_x_corr, dumb_z_corr = sim_test.dumb_correction(synds)
        dumb_log_err = sim_test.logical_error(rnd_err, dumb_x_corr, dumb_z_corr)
        if dumb_log_err == 'X' or dumb_log_err == 'Y':
            dumb_logical = 1
        else:
            dumb_logical = 0
        samples[s] = [dumb_logical, 1]

    if cycles % 1000 == 0:
        print(len(samples), cycles)

sorted_samples = sorted(samples.items(), key=lambda e: e[1][1], reverse=True)

with open('d='+str(distance)+'_p='+str(physprob)+'_uniform_distr_samples.txt', 'w') as f:
    for key, value in sorted_samples:
        x = ''
        for i in range(no_anc):
            x += key[i] + ' '

        x += ' '
        x += str(value[0]) + ' '
        x += str(value[1]) + '\n'
        f.write(x)



