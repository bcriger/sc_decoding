import decoding_2d as dc2
import decoding_3d as dc3
import error_model as em
from functools import reduce
import sparse_pauli as sp
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import random

d = 3
p=0.08
iterations = 5000
sim_test2d = dc2.Sim2D(d, d, p)
sim_test = dc3.Sim3D(d, d, ('pq', p, p), True)
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
d = 3
no_anc = int((d + 1) * (d - 1) / 2 )
inputs = int((d + 1) * no_anc)
outputs = 2
nodes = [768]
no_samples = 12211

z_ancs = [0,5,11,16]
p = 0.008
iterations = 5000

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
#y_test = tf.nn.softmax(y_test)
#y_test = tf.argmax(y_test, dimension=1)

NUM_CORES = 8
sess_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)
filename = 'd_'+str(d)+nodes_str+'_'+str(no_samples)+'.ckpt'
blossom_time = 0.0
nn_time = 0.0
valid_cycles = 0

with tf.Session(config=sess_config) as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    saver.restore(sess, filename)
    while mwpm < iterations:
#--------------- run SC cycle ------------------------------------------------------
        cycles += 1
        sim_test = dc3.Sim3D(d, d, ('pq', p, p), True)
        err_history, syndrome_history = sim_test.history()
        #rnd_err = sp.Pauli(err_history[2].x_set,[])
        rnd_err = err_history[2].x_set
        #print('rnd_err ', rnd_err)
        rd_er = sp.Pauli(rnd_err,[])
        #print('rd_er ', rd_er)
        synds = syndrome_history
        #print('synds ', synds)
#--------------- run mwpm ----------------------------------------------------------
        if synds != []:
            valid_cycles += 1
            start_blossom = time.time()
            corr_blossom = sim_test.correction(synds, metric=None, bdy_info=None)
            mwpm_log_err = sim_test.logical_error(err_history[-1], corr_blossom)
            if mwpm_log_err == 'X' or mwpm_log_err == 'Y':
                mwpm += 1
            blossom_time += time.time() - start_blossom       
#--------------- run nn ------------------------------------------------------------
        flips_Z = [0] * (d+1)* no_anc

        for j in range(d+1):
            for i in range(no_anc):
                if (z_ancs[i] in syndrome_history['Z'][j] and z_ancs[i] in syndrome_history['Z'][j+1]) or \
                   (z_ancs[i] not in syndrome_history['Z'][j] and z_ancs[i] not in syndrome_history['Z'][j+1]):
                    flips_Z[j*4+i] = 0
                else:
                    flips_Z[j*4+i] = 1
        
        #print('flips_Z ', flips_Z)
        
        synds_2d = sim_test2d.syndromes(rd_er)
        #print('synds_2d ', synds_2d)
        dumb_x_corr, dumb_z_corr = sim_test2d.dumb_correction(synds_2d, False)
        #print('dumb_x_corr ', dumb_x_corr)
        
        start_nn = 0.0
        if synds != []:
            start_nn = time.time()
            pred_corr_nn = sess.run(y_test, feed_dict={x_test:[flips_Z]})
        else:
            pred_corr_nn = [[0] * outputs]
    
        if pred_corr_nn[0][1] >= 0.5:
            x_corr_nn = dumb_x_corr * logicals[0]
        else:
            x_corr_nn = dumb_x_corr
            
        #print('pred_corr_nn ', pred_corr_nn)
        #print('x_corr_nn ', x_corr_nn, '   |    ', corr_blossom)
        #print(err_history[-1])
        nn_log_err = sim_test.logical_error(rd_er, x_corr_nn)
        if nn_log_err == 'X' or nn_log_err == 'Y':
            nn += 1
        if start_nn > 0.0:
            nn_time += time.time() - start_nn
        #-----------------------------------------------------
        #print('--------------------------------------------')
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