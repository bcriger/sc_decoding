import decoding_2d as dc
import sparse_pauli as sp
import tensorflow as tf
import time
import random

random.seed(time.time())

physprob = 0.002
sim_test = dc.Sim2D(5, physprob)
print(physprob)
z_ancs_keys = list(sim_test.layout.z_ancs())
logicals = sim_test.layout.logicals()
cycles = 0
mwpm_log_err = 0
mwpm = 0
dumb = 0
nn = 0
nn_dumb = 0
temp_list = []
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

NUM_CORES = 1
sess_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)

sd = [sp.Pauli([2],[]), sp.Pauli([3],[]), sp.Pauli([4],[]), sp.Pauli([5],[]), sp.Pauli([6],[]), sp.Pauli([12],[]), sp.Pauli([13], []), 
           sp.Pauli([14],[]), sp.Pauli([15],[]), sp.Pauli([16],[]), sp.Pauli([22],[]), sp.Pauli([23],[]), sp.Pauli([24], []), sp.Pauli([25],[]), 
           sp.Pauli([26],[]), sp.Pauli([32],[]), sp.Pauli([33],[]), sp.Pauli([34],[]), sp.Pauli([35], []), sp.Pauli([36],[]), sp.Pauli([42],[]), 
           sp.Pauli([43],[]), sp.Pauli([44],[]), sp.Pauli([45],[]), sp.Pauli([46], []), sp.Pauli([2,3],[]), sp.Pauli([2,4],[]), sp.Pauli([2,5],[]), 
           sp.Pauli([2,6],[]), sp.Pauli([2,12],[]), sp.Pauli([2,13], []), sp.Pauli([2,14],[]), sp.Pauli([2,15],[]), sp.Pauli([2,16],[]), sp.Pauli([2,22],[]), 
           sp.Pauli([2,23],[]), sp.Pauli([2,24], []), sp.Pauli([2,25],[]), sp.Pauli([2,26],[]), sp.Pauli([2,32],[]), sp.Pauli([2,33],[]), sp.Pauli([2,34],[]),
           sp.Pauli([2,35], []), sp.Pauli([2,36],[]), sp.Pauli([2,42],[]), sp.Pauli([2,43],[]), sp.Pauli([2,44],[]), sp.Pauli([2,45],[]),
           sp.Pauli([2,46], []), sp.Pauli([3, 4], []), sp.Pauli([3, 5], []), sp.Pauli([3, 6], []), sp.Pauli([3, 12], []), sp.Pauli([3, 13], []), 
           sp.Pauli([3, 14], []), sp.Pauli([3, 15], []), sp.Pauli([3, 16], []), sp.Pauli([3, 22], []), sp.Pauli([3, 23], []),
           sp.Pauli([3, 24], []), sp.Pauli([3, 25], []), sp.Pauli([3, 26], []), sp.Pauli([3, 32], []), sp.Pauli([3, 33], []), sp.Pauli([3, 34], []),
           sp.Pauli([3, 35], []), sp.Pauli([3, 36], []), sp.Pauli([3, 42], []), sp.Pauli([3, 43], []), sp.Pauli([3, 44], []), sp.Pauli([3, 45], []),
           sp.Pauli([3, 46], []), sp.Pauli([4, 5], []), sp.Pauli([4, 6], []), sp.Pauli([4, 12], []), sp.Pauli([4, 13], []), sp.Pauli([4, 14], []), 
           sp.Pauli([4, 15], []), sp.Pauli([4, 16], []), sp.Pauli([4, 22], []), sp.Pauli([4, 23], []), sp.Pauli([4, 24], []), sp.Pauli([4, 25], []), 
           sp.Pauli([4, 26], []), sp.Pauli([4, 32], []), sp.Pauli([4, 33], []), sp.Pauli([4, 34], []), sp.Pauli([4, 35], []), sp.Pauli([4, 36], []), 
           sp.Pauli([4, 42], []), sp.Pauli([4, 43], []), sp.Pauli([4, 44], []), sp.Pauli([4, 45], []), sp.Pauli([4, 46], []), sp.Pauli([5, 6], []), 
           sp.Pauli([5, 12], []), sp.Pauli([5, 13], []), sp.Pauli([5, 14], []), sp.Pauli([5, 15], []), sp.Pauli([5, 16], []), sp.Pauli([5, 22], []), 
           sp.Pauli([5, 23], []), sp.Pauli([5, 24], []), sp.Pauli([5, 25], []), sp.Pauli([5, 26], []), sp.Pauli([5, 32], []), sp.Pauli([5, 33], []), 
           sp.Pauli([5, 34], []), sp.Pauli([5, 35], []), sp.Pauli([5, 36], []), sp.Pauli([5, 42], []), sp.Pauli([5, 43], []), sp.Pauli([5, 44], []), 
           sp.Pauli([5, 45], []), sp.Pauli([5, 46], []), sp.Pauli([6, 12], []), sp.Pauli([6, 13], []), sp.Pauli([6, 14], []), sp.Pauli([6, 15], []), 
           sp.Pauli([6, 16], []), sp.Pauli([6, 22], []), sp.Pauli([6, 23], []), sp.Pauli([6, 24], []), sp.Pauli([6, 25], []), sp.Pauli([6, 26], []), 
           sp.Pauli([6, 32], []), sp.Pauli([6, 33], []), sp.Pauli([6, 34], []), sp.Pauli([6, 35], []), sp.Pauli([6, 36], []), sp.Pauli([6, 42], []), 
           sp.Pauli([6, 43], []), sp.Pauli([6, 44], []), sp.Pauli([6, 45], []), sp.Pauli([6, 46], []), sp.Pauli([12, 13], []), sp.Pauli([12, 14], []), 
           sp.Pauli([12, 15], []), sp.Pauli([12, 16], []), sp.Pauli([12, 22], []), sp.Pauli([12, 23], []), sp.Pauli([12, 24], []), sp.Pauli([12, 25], []), 
           sp.Pauli([12, 26], []), sp.Pauli([12, 32], []), sp.Pauli([12, 33], []), sp.Pauli([12, 34], []), sp.Pauli([12, 35], []), sp.Pauli([12, 36], []), 
           sp.Pauli([12, 42], []), sp.Pauli([12, 43], []), sp.Pauli([12, 44], []), sp.Pauli([12, 45], []), sp.Pauli([12, 46], []), sp.Pauli([13, 14], []), 
           sp.Pauli([13, 15], []), sp.Pauli([13, 16], []), sp.Pauli([13, 22], []), sp.Pauli([13, 23], []), sp.Pauli([13, 24], []), sp.Pauli([13, 25], []), 
           sp.Pauli([13, 26], []), sp.Pauli([13, 32], []), sp.Pauli([13, 33], []), sp.Pauli([13, 34], []), sp.Pauli([13, 35], []), sp.Pauli([13, 36], []), 
           sp.Pauli([13, 42], []), sp.Pauli([13, 43], []), sp.Pauli([13, 44], []), sp.Pauli([13, 45], []), sp.Pauli([13, 46], []), sp.Pauli([14, 15], []), 
           sp.Pauli([14, 16], []), sp.Pauli([14, 22], []), sp.Pauli([14, 23], []), sp.Pauli([14, 24], []), sp.Pauli([14, 25], []), sp.Pauli([14, 26], []), 
           sp.Pauli([14, 32], []), sp.Pauli([14, 33], []), sp.Pauli([14, 34], []), sp.Pauli([14, 35], []), sp.Pauli([14, 36], []), sp.Pauli([14, 42], []), 
           sp.Pauli([14, 43], []), sp.Pauli([14, 44], []), sp.Pauli([14, 45], []), sp.Pauli([14, 46], []), sp.Pauli([15, 16], []), sp.Pauli([15, 22], []), 
           sp.Pauli([15, 23], []), sp.Pauli([15, 24], []), sp.Pauli([15, 25], []), sp.Pauli([15, 26], []), sp.Pauli([15, 32], []), sp.Pauli([15, 33], []),
           sp.Pauli([15, 34], []), sp.Pauli([15, 35], []), sp.Pauli([15, 36], []), sp.Pauli([15, 42], []), sp.Pauli([15, 43], []), sp.Pauli([15, 44], []), 
           sp.Pauli([15, 45], []), sp.Pauli([15, 46], []), sp.Pauli([16, 22], []), sp.Pauli([16, 23], []), sp.Pauli([16, 24], []), sp.Pauli([16, 25], []), 
           sp.Pauli([16, 26], []), sp.Pauli([16, 32], []), sp.Pauli([16, 33], []), sp.Pauli([16, 34], []), sp.Pauli([16, 35], []), sp.Pauli([16, 36], []), 
           sp.Pauli([16, 42], []), sp.Pauli([16, 43], []), sp.Pauli([16, 44], []), sp.Pauli([16, 45], []), sp.Pauli([16, 46], []), sp.Pauli([22, 23], []),
           sp.Pauli([22, 24], []), sp.Pauli([22, 25], []), sp.Pauli([22, 26], []), sp.Pauli([22, 32], []), sp.Pauli([22, 33], []), sp.Pauli([22, 34], []),
           sp.Pauli([22, 35], []), sp.Pauli([22, 36], []), sp.Pauli([22, 42], []), sp.Pauli([22, 43], []), sp.Pauli([22, 44], []), sp.Pauli([22, 45], []),
           sp.Pauli([22, 46], []), sp.Pauli([23, 24], []), sp.Pauli([23, 25], []), sp.Pauli([23, 26], []), sp.Pauli([23, 32], []), sp.Pauli([23, 33], []), 
           sp.Pauli([23, 34], []), sp.Pauli([23, 35], []), sp.Pauli([23, 36], []), sp.Pauli([23, 42], []), sp.Pauli([23, 43], []), sp.Pauli([23, 44], []), 
           sp.Pauli([23, 45], []), sp.Pauli([23, 46], []), sp.Pauli([24, 25], []), sp.Pauli([24, 26], []), sp.Pauli([24, 32], []), sp.Pauli([24, 33], []), 
           sp.Pauli([24, 34], []), sp.Pauli([24, 35], []), sp.Pauli([24, 36], []), sp.Pauli([24, 42], []), sp.Pauli([24, 43], []), sp.Pauli([24, 44], []), 
           sp.Pauli([24, 45], []), sp.Pauli([24, 46], []), sp.Pauli([25, 26], []), sp.Pauli([25, 32], []), sp.Pauli([25, 33], []), sp.Pauli([25, 34], []),
           sp.Pauli([25, 35], []), sp.Pauli([25, 36], []), sp.Pauli([25, 42], []), sp.Pauli([25, 43], []), sp.Pauli([25, 44], []), sp.Pauli([25, 45], []),
           sp.Pauli([25, 46], []), sp.Pauli([26, 32], []), sp.Pauli([26, 33], []), sp.Pauli([26, 34], []), sp.Pauli([26, 35], []), sp.Pauli([26, 36], []), 
           sp.Pauli([26, 42], []), sp.Pauli([26, 43], []), sp.Pauli([26, 44], []), sp.Pauli([26, 45], []), sp.Pauli([26, 46], []), sp.Pauli([32, 33], []), 
           sp.Pauli([32, 34], []), sp.Pauli([32, 35], []), sp.Pauli([32, 36], []), sp.Pauli([32, 42], []), sp.Pauli([32, 43], []), sp.Pauli([32, 44], []), 
           sp.Pauli([32, 45], []), sp.Pauli([32, 46], []), sp.Pauli([33, 34], []), sp.Pauli([33, 35], []), sp.Pauli([33, 36], []), sp.Pauli([33, 42], []), 
           sp.Pauli([33, 43], []), sp.Pauli([33, 44], []), sp.Pauli([33, 45], []), sp.Pauli([33, 46], []), sp.Pauli([34, 35], []), sp.Pauli([34, 36], []), 
           sp.Pauli([34, 42], []), sp.Pauli([34, 43], []), sp.Pauli([34, 44], []), sp.Pauli([34, 45], []), sp.Pauli([34, 46], []), sp.Pauli([35, 36], []), 
           sp.Pauli([35, 42], []), sp.Pauli([35, 43], []), sp.Pauli([35, 44], []), sp.Pauli([35, 45], []), sp.Pauli([35, 46], []), sp.Pauli([36, 42], []), 
           sp.Pauli([36, 43], []), sp.Pauli([36, 44], []), sp.Pauli([36, 45], []), sp.Pauli([36, 46], []), sp.Pauli([42, 43], []), sp.Pauli([42, 44], []), 
           sp.Pauli([42, 45], []), sp.Pauli([42, 46], []), sp.Pauli([43, 44], []), sp.Pauli([43, 45], []), sp.Pauli([43, 46], []), sp.Pauli([44, 45], []),
           sp.Pauli([44, 46], []),sp.Pauli([45,46], [])
           ]

samples = {}
m = 0
with open('d=5_uniform_distr_samples.txt', 'r') as f:
    for line in f: # iterate over each line
        m += 1
        data = line.split()

        s = ''
        for i in range(inputs):
            s += str(data[i])
        samples[s] = 0

        if m == 500:
            break
			
with tf.Session(config=sess_config) as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, 'model_d_5'+nodes_str+'.ckpt')
    #while len(temp_list) < 325:
    t_blossom = 0.0
    t_nn = 0.0
    cnt_times = 0
    while mwpm < 10:
#--------------- run SC cycle ------------------------------------------------------
        cycles += 1
        rnd_err = sim_test.random_error()
        synds = sim_test.syndromes(rnd_err)
#--------------- run mwpm ----------------------------------------------------------
        if synds[1] != [0] * len(z_ancs_keys):
           t = time.time()
           x_graph, z_graph = sim_test.graph(synds[0]), sim_test.graph(synds[1])
           x_corr = sim_test.correction(x_graph, 'Z')
           z_corr = sim_test.correction(z_graph, 'X')
           t = time.time() - t
           t_blossom += t
        else:
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

        s = ''
        for i in range(inputs):
            s += str(lst_z[i])

        if lst_z != [0] * len(z_ancs_keys):
            t = time.time()
            predictions = sess.run(y_test, feed_dict={x_test:[lst_z]})
            t = time.time() - t
            t_nn += t
            cnt_times += 1
        else:
            predictions = sess.run(y_test, feed_dict={x_test:[lst_z]})

        if predictions[0][0] >= 0.5:
            x_corr_nn = rnd_err_prime * logicals[0]
        else:
            x_corr_nn = rnd_err_prime
		
        nn_log_err = sim_test.logical_error(rnd_err, x_corr_nn, sp.Pauli([], []))
        if nn_log_err == 'X' or nn_log_err == 'Y':
            nn += 1
	
        if s not in samples:
            predictions_dumb = [[random.random()]] 
        else:
            predictions_dumb = predictions

        if predictions_dumb[0][0] >= 0.5:
            x_corr_nn_dumb = rnd_err_prime * logicals[0]
        else:
            x_corr_nn_dumb = rnd_err_prime

        nn_dumb_log_err = sim_test.logical_error(rnd_err, x_corr_nn_dumb, sp.Pauli([], []))
        if nn_dumb_log_err == 'X' or nn_dumb_log_err == 'Y':
            nn_dumb += 1
		
        if cycles % 1000 == 0:
            print(mwpm, dumb, nn, nn_dumb, cycles)
    print(mwpm, dumb, nn, nn_dumb, cycles)
    print('blossom=', t_blossom/cnt_times,'\nn=', t_nn/cnt_times)
#--------------- start next cycle ------------------------------------------------------