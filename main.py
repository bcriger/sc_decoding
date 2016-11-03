import decoding_2d as dc
import sparse_pauli as sp
import csv

sim_test = dc.Sim2D(5, 0.01)
x_ancs_keys = list(sim_test.layout.x_ancs())
z_ancs_keys = list(sim_test.layout.z_ancs())
logicals = sim_test.layout.logicals()
z_flips_list = []
nn_out_list = []
cnt = 0
nn_out = 0
cycles = 0
mwpm = []
dumb = []
neural = []

with open('d=5_p=0.01_x_logical_error_training123.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='|', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['error', 'z_synd', 'dumb', 'nn'])

    while cnt < 300:
        cycles += 1
        rnd_err = sim_test.random_error()
        synds = sim_test.syndromes(rnd_err)
        dumb_x_corr, dumb_z_corr = sim_test.dumb_correction(synds)
        check_coset_value = sim_test.logical_error(rnd_err, dumb_x_corr, dumb_z_corr)
        
        rnd_err_prime = rnd_err * dumb_z_corr * dumb_x_corr
        
        if check_coset_value == 'Y':
            rnd_err_double_prime = rnd_err_prime * logicals[0] * logicals[1]
            nn_out = 1
        elif check_coset_value == 'X':
            rnd_err_double_prime = rnd_err_prime * logicals[0]
            nn_out = 1  # used as output for training the neural network for z_syndromes input
        elif check_coset_value == 'Z':
            rnd_err_double_prime = rnd_err_prime * logicals[1]
            nn_out = 0
        elif check_coset_value == 'I':
            rnd_err_double_prime = rnd_err_prime
            nn_out = 0

        dumb.append(check_coset_value)

        # training part of neural network
        # lst_x = [0] * len(x_ancs_keys)
        lst_z = [0] * len(z_ancs_keys)

        # for j in x_synd:
        #    key = sim_test.key_from_value(sim_test.layout.map, j)
        #    pos = x_ancs_keys.index(key)
        #    lst_x[pos] = 1

        for k in synds[1]:
            key = sim_test.layout.map.inv[k]
            pos = z_ancs_keys.index(key)
            lst_z[pos] = 1

        z_flips = [lst_z]
        # x_flips = [lst_x]

        if z_flips not in z_flips_list:
            z_flips_list.append(z_flips)
            spamwriter.writerow([rnd_err, synds[1], check_coset_value, z_flips, nn_out])
            cnt += 1

        if cycles % 1000 == 0:
            print 'cnt', cnt, 'CYCLES', cycles
    print 'cnt', cnt, 'CYCLES', cycles
