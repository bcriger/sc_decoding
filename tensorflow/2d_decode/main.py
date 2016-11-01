import decoding_2d as dc
import csv
import sparse_pauli as sp

cnt = 0
cycles = 0
anc_list = []
cor_list = []
err_list = []
#rnd_list = []
sim_test = dc.Sim2D(5, 0.01)
x_ancs_keys = list(sim_test.layout.x_ancs())
z_ancs_keys = list(sim_test.layout.z_ancs())

max_iter = 2**(((sim_test.d**2)-1)/2)
lst_x_ancs = []
lst_z_ancs = []

with open('d=5_p=0.01_x_logical_dumb_4096.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='|', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    #spamwriter.writerow(['      flips       '] + ['logical errors'] + ['      random errors        '] + ['            corrections '])
    spamwriter.writerow(['   flips  '] + ['logical errors'] + ['corrections '])
    # while cycles < 200000:
    while len(anc_list) < max_iter: #d=3->cnt=16, d=5->cnt=4096, d=7->cnt=16777216
        cycles += 1
        ler, x_synd, z_synd, x_corr_mwpm, z_corr_mwpm, rnd_err = sim_test.run(1,verbose=False,progress=False)
        x_corr_dumb, z_corr_dumb = sim_test.dumb_decoder(x_synd, z_synd)
        #log = sim_test.logical_error(sim_test.random_error(), x_corr_mwpm*x_corr_dumb, z_corr_mwpm*z_corr_dumb)
        #log = sim_test.logical_error(sp.Pauli(), x_corr_mwpm * x_corr_dumb, z_corr_mwpm * z_corr_dumb)
        log = sim_test.logical_error(rnd_err, x_corr_dumb, z_corr_dumb)

        lst_x = [0] * len(x_ancs_keys)
        lst_z = [0] * len(z_ancs_keys)

        for i in x_synd:
            key = sim_test.key_from_value(sim_test.layout.map, i)
            pos = x_ancs_keys.index(key)
            lst_x[pos] = 1

        for i in z_synd:
            key = sim_test.key_from_value(sim_test.layout.map, i)
            pos = z_ancs_keys.index(key)
            lst_z[pos] = 1

        lst = lst_x + lst_z
        if lst_z not in anc_list:
            #rnd_list.append(sim_test.random_error())
            anc_list.append(lst_z)
            cor_list.append(z_corr_mwpm*z_corr_dumb)#x_corr_mwpm*x_corr_dumb)
            err_list.append(log)
            #spamwriter.writerow([anc_list[cnt]] + ['   '] + [err_list[cnt]] + ['  '] + [rnd_list[cnt]] + ['  '] + [cor_list[cnt]])
            spamwriter.writerow([anc_list[cnt]] + ['   '] + [err_list[cnt]] + ['  '] + [cor_list[cnt]])
            cnt += 1

        if cycles % 1000 == 0:
            print 'cnt', cnt, 'CYCLES', cycles

print 'cnt', cnt, 'CYCLES', cycles

