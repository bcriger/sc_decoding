import decoding_2d as dc
import sparse_pauli as sp
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import random

d = 5
p = 0.5
iterations = 10000
sim_test = dc.Sim2D(d, d, p)
z_ancs_keys = list(sim_test.layout.z_ancs())
logicals = sim_test.layout.logicals()
cycles = 0
mwpm_log_err = 0
plut_log_err = 0
mwpm = 0
dumb = 0
nn = 0

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.05, shape=shape)
    return tf.Variable(initial)

start_total_time = time.time()
inputs = 12
outputs = 2
nodes = [90]
'''
data_LUT = []
PLUT = 0
with open('d=3_p=0.08_samples.txt', 'r') as f:
    m = 0
    for line in f: # iterate over each line
        m += 1
        data_smpl = line.split() # split it by whitespace
        data_smpl = [int(d) for d in data_smpl[:inputs]]
        data_LUT.append(data_smpl)
        if m == 16:
            break
    f.close()
'''
x_test = tf.placeholder(tf.float32, shape=[None, inputs])
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
    #h1.append(tf.nn.tanh(tf.matmul(h1[-1],W[i])+bias[i]))
y_test = tf.nn.sigmoid(tf.matmul(h1[-1], W[-1]) + bias[-1])
#y_test = tf.nn.softmax(y_test)
#y_test = tf.nn.tanh(tf.matmul(h1[-1], W[-1]) + bias[-1])
#y_test = tf.argmax(y_test, dimension=1)

NUM_CORES = 8
sess_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)

#samples = {}
blossom_time = 0.0
nn_time = 0.0
valid_cycles = 0
#out_lut = 0
with tf.Session(config=sess_config) as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, 'd_5'+nodes_str+'_2968.ckpt')
    #start_blossom = time.time()
    while mwpm < iterations:
    #while len(samples) < 5000:
#--------------- run SC cycle ------------------------------------------------------
        cycles += 1
        rnd_err = sim_test.random_error()
        #print('err ', rnd_err)
        synds = sim_test.syndromes(rnd_err)
        #print('synds_Z ', synds[1])
#--------------- run mwpm ----------------------------------------------------------
        if synds[1] != []:
            valid_cycles += 1
            start_blossom = time.time()
            z_graph = sim_test.graph(synds[1])
            z_corr = sim_test.correction(z_graph, 'X')
            mwpm_log_err = sim_test.logical_error(rnd_err, sp.Pauli([],[]), z_corr)
            if mwpm_log_err == 'X' or mwpm_log_err == 'Y':
                mwpm += 1
            blossom_time += time.time() - start_blossom       
#--------------- run dumb ----------------------------------------------------------
        dumb_x_corr, dumb_z_corr = sim_test.dumb_correction(synds, False)
        #dumb_log_err = sim_test.logical_error(rnd_err, dumb_x_corr, dumb_z_corr)
        #if dumb_log_err == 'X' or dumb_log_err == 'Y':
        #    dumb += 1
#--------------- run nn ------------------------------------------------------------
        rnd_err_prime = dumb_z_corr * dumb_x_corr
        #print('rnd_err_prime ', rnd_err_prime)
        lst_z = [0] * len(z_ancs_keys)
        for k in synds[1]:
            key = sim_test.layout.map.inv[k]
            pos = z_ancs_keys.index(key)
            lst_z[pos] = 1
        '''
        if lst_z not in data_LUT:
            out_lut += 1
                        
        s = ''
        for i in range(inputs):
            s += str(lst_z[i])
		
        if s not in samples:
            samples[s] = [mwpm_log_err, dumb_log_err]
        '''    
        #-----------------------------------------------------
        start_nn = 0.0
        if synds[1] != []:
            start_nn = time.time()
            predictions = sess.run(y_test, feed_dict={x_test:[lst_z]})
        else:
            predictions = [[0] * outputs]
        
        #print(predictions[0])   
        
        if predictions[0][1] >= 0.5:
            x_corr_nn = rnd_err_prime * logicals[0]
        else:
            x_corr_nn = rnd_err_prime
        #print('x_corr_nn ', x_corr_nn)
        
        nn_log_err = sim_test.logical_error(rnd_err, x_corr_nn, sp.Pauli([], []))
        #print('nn_log_err ', nn_log_err)
        #print('--------------------')
        if nn_log_err == 'X' or nn_log_err == 'Y':
            nn += 1
        if start_nn > 0.0:
            nn_time += time.time() - start_nn
        #-----------------------------------------------------
        
        if cycles % 1000 == 0:
            print("p="+str(p),mwpm, nn, cycles)

end_total_time = time.time()
time_dif = end_total_time - start_total_time
print("-----------------------")
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
print(mwpm, nn, cycles)
print("blossom time : " + str(blossom_time/valid_cycles))
print("nn time      : " + str(nn_time/valid_cycles))
#--------------- start next cycle ------------------------------------------------------
'''
with open('d=5_p=0.09_blossom_dumb_4000.txt', 'w') as f:
    for j in samples:
        x = ''
        for i in range(12):
            x += j[i] + ' '

        x += ' '
        x += str(samples[j][0]) + ' '
        x += str(samples[j][1]) + '\n'
        f.write(x)
'''