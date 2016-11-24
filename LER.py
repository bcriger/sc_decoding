import decoding_2d as dc
import sparse_pauli as sp
import tensorflow as tf

sim_test = dc.Sim2D(5, 0.1)
z_ancs_keys = list(sim_test.layout.z_ancs())
logicals = sim_test.layout.logicals()
cycles = 0
mwpm_log_err = 0
mwpm = 0
dumb = 0
nn = 0

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

inputs = 12
outputs = 1
nodes = [24]

x_test = tf.placeholder(tf.float32, shape=[1, inputs])
W = []
W.append(weight_variable([inputs,nodes[0]]))
for i in range(len(nodes)-1):
	W.append(weight_variable([nodes[i],nodes[i+1]]))
W.append(weight_variable([nodes[-1],outputs]))

nodes_str = ''
bias=[]
for i in range(len(nodes)):
    nodes_str += '_' + str(nodes[i])
    bias.append(bias_variable([nodes[i]]))
bias.append(bias_variable([outputs]))

h1=[x_test]
for i in range(len(nodes)):
    h1.append(tf.nn.sigmoid(tf.matmul(h1[-1],W[i])+bias[i]))
y_test = tf.nn.sigmoid(tf.matmul(h1[-1], W[-1]) + bias[-1])

NUM_CORES = 8
sess_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)

with tf.Session(config=sess_config) as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, 'model'+nodes_str+'.ckpt')
	
    while mwpm < 5000:
#--------------- run SC cycle ------------------------------------------------------
        cycles += 1
        rnd_err = sim_test.random_error()
        synds = sim_test.syndromes(rnd_err)
#--------------- run mwpm ----------------------------------------------------------
        x_graph, z_graph = sim_test.graph(synds[0]), sim_test.graph(synds[1])
        x_corr = sim_test.correction(x_graph, 'Z')
        z_corr = sim_test.correction(z_graph, 'X')
        mwpm_log_err = sim_test.logical_error(rnd_err, x_corr, z_corr)
        if mwpm_log_err == 'X' or mwpm_log_err == 'Y':
            mwpm += 1
#--------------- run dumb ----------------------------------------------------------
        dumb_x_corr, dumb_z_corr = sim_test.dumb_correction(synds)
        dumb_log_err = sim_test.logical_error(rnd_err, dumb_x_corr, dumb_z_corr)
        if dumb_log_err == 'X' or dumb_log_err == 'Y':
            dumb += 1
#--------------- run nn ------------------------------------------------------------
        rnd_err_prime = dumb_z_corr * dumb_x_corr
    
        lst_z = [0] * len(z_ancs_keys)
        for k in synds[1]:
            key = sim_test.layout.map.inv[k]
            pos = z_ancs_keys.index(key)
            lst_z[pos] = 1
        
        predictions = sess.run(y_test, feed_dict={x_test:[lst_z]})
    
        if predictions[0][0] >= 0.5:
            x_corr_nn = rnd_err_prime * logicals[0]
        else:
            x_corr_nn = rnd_err_prime
    
        nn_log_err = sim_test.logical_error(rnd_err, x_corr_nn, sp.Pauli([], []))
        if nn_log_err == 'X' or nn_log_err == 'Y':
            nn += 1
        
        if cycles % 1000 == 0:
            print(mwpm, dumb, nn, cycles)
    print(mwpm, dumb, nn, cycles)
#--------------- start next cycle ------------------------------------------------------