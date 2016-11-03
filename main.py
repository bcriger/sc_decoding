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
    spamwriter.writerow(['error'] + ['z_synd'] + ['dumb'] + ['nn'])

    while cnt < 300:
        cycles += 1
        log_mwpm, x_synd, z_synd, x_corr_mwpm, z_corr_mwpm, rnd_err = sim_test.run(1, verbose=False, progress=False)
        dumb_x_corr, dumb_z_corr = sim_test.dumb_correction(x_synd, z_synd)
        rnd_err_prime = dumb_x_corr * dumb_z_corr  # output/correction of the dumb decoder
        rnd_err_prime_decomp = sim_test.XZ_decomposition(rnd_err_prime)
        check_coset_value = sim_test.logical_error(rnd_err, rnd_err_prime_decomp[0], rnd_err_prime_decomp[1])

        if check_coset_value == 'Y':
            #rnd_err_double_prime_x = rnd_err_prime * logicals[0]
            #rnd_err_double_prime_z = rnd_err_prime * logicals[1]
            rnd_err_double_prime = rnd_err_prime * logicals[0] * logicals[1]
            nn_out = 1
        else:
            if check_coset_value == 'X':
                rnd_err_double_prime_x = rnd_err_prime * logicals[0]
                nn_out = 1  # used as output for training the neural network for z_syndromes input
            else:
                rnd_err_double_prime_x = rnd_err_prime
                nn_out = 0  # used as output for training the neural network for z_syndromes input
            if check_coset_value == 'Z':
                rnd_err_double_prime_z = rnd_err_prime * logicals[1]
                nn_out = 0
            else:
                rnd_err_double_prime_z = rnd_err_prime
                nn_out = 0  # used as output for training the neural network for z_syndromes input
            if check_coset_value == 'I':
                rnd_err_double_prime_x = rnd_err_prime
                nn_out = 0

            rnd_err_double_prime = rnd_err_double_prime_x * rnd_err_double_prime_z

        dumb_corr_decomposition = sim_test.XZ_decomposition(rnd_err_double_prime)
        logical_dumb_error = sim_test.logical_error(rnd_err, dumb_corr_decomposition[0], dumb_corr_decomposition[1])
        dumb.append(logical_dumb_error)

        # training part of neural network
        # lst_x = [0] * len(x_ancs_keys)
        lst_z = [0] * len(z_ancs_keys)

        # for j in x_synd:
        #    key = sim_test.key_from_value(sim_test.layout.map, j)
        #    pos = x_ancs_keys.index(key)
        #    lst_x[pos] = 1

        for k in z_synd:
            key = sim_test.key_from_value(sim_test.layout.map, k)
            pos = z_ancs_keys.index(key)
            lst_z[pos] = 1

        z_flips = [lst_z]
        # x_flips = [lst_x]

        if z_flips not in z_flips_list:
            z_flips_list.append(z_flips)
            nn_out_list.append(nn_out)
            spamwriter.writerow([rnd_err] + [z_synd] + [logical_dumb_error] + [z_flips_list[cnt]] + [nn_out_list[cnt]])
            cnt += 1

        if cycles % 1000 == 0:
            print 'cnt', cnt, 'CYCLES', cycles
    print 'cnt', cnt, 'CYCLES', cycles
