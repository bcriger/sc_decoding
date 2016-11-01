import decoding_2d as dc
import sparse_pauli as sp

def key_from_value(dict, value):
    for key, val in dict.iteritems():
        if val == value:
            return key

sim_test = dc.Sim2D(3, 0.1)

dumb_correction_z = []
dumb_correction_x = []
corr_dumb_dec_z = sp.Pauli()
corr_dumb_dec_x = sp.Pauli()

log, x_syndrome, z_syndrome = sim_test.run(1,verbose=True, progress=False)
print '------------------'

if z_syndrome != []:
    for i in z_syndrome:
        key = key_from_value(sim_test.layout.map, i)
        closest_z_bnd = sim_test.bdy_info(key)
        connect_to_z_bnd = sim_test.path_pauli(list(key), list(closest_z_bnd[1]), 'Z')
        dumb_correction_z.append(connect_to_z_bnd)

if x_syndrome != []:
    for i in x_syndrome:
        key = key_from_value(sim_test.layout.map, i)
        closest_x_bnd = sim_test.bdy_info(key)
        connect_to_x_bnd = sim_test.path_pauli(list(key), list(closest_x_bnd[1]), 'X')
        dumb_correction_x.append(connect_to_x_bnd)

if dumb_correction_z != []:
    for i in dumb_correction_z:
        corr_dumb_dec_z = corr_dumb_dec_z * i
if dumb_correction_x != []:
    for i in dumb_correction_x:
        corr_dumb_dec_x = corr_dumb_dec_x * i

dumb_correction = corr_dumb_dec_x * corr_dumb_dec_z

print 'corr_dumb_dec_x ', corr_dumb_dec_x
print 'corr_dumb_dec_z ', corr_dumb_dec_z
print 'dumb_correction ', dumb_correction
