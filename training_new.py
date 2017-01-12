import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
from itertools import chain
import math

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1. / math.sqrt(float(shape[0])))#0.01, 0.05, 0.1
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1. / math.sqrt(float(shape[0])))
    # initial = tf.constant(0.05, shape=shape)
    return tf.Variable(initial)

distance = 5
inputs = 12
outputs = 2
nodes = [48]
first = 1
ff = 'd=5_p=0.09'
#x_anc_pos = [10, 8, 30, 28, 20, 18, 40, 38, 1, 0, 47, 48]
FLAGS_train = True	# True : for training (False : for prediction)
FLAGS_training_steps = 1000
FLAGS_checkpoint_steps = 100 # save every x iterations
print(nodes)

a = []
b = []
c = []

#with open(ff + '_samples.txt', 'r') as f:
with open(ff + '_blossom_dumb.txt', 'r') as f:
    m = 0
    for line in f: # iterate over each line
        m += 1
        data = line.split() # split it by whitespace
        blos = int(data[-2])
        dumb = [float(data[-4]), float(data[-3])]
        freq = [float(data[-1])]
        c.append(freq)     
        #if float(data[-3]) / float(data[-4]) >= 1.0:
        #    dumb = [1.0]
        #else:
        #    dumb = [0.0]
        '''
        if (blos == 1 and np.round_(dumb[1]) == 0) or (blos == 0 and np.round_(dumb[1]) == 1):
            dumb[0] = dumb[1]
            dumb[1] = float(data[-4])
        '''
        b.append(dumb)
        data = [float(d) for d in data[:inputs]]
        #for i in range(inputs):
        #    if data[i] == 0.0:
        #        data[i] = -1.0
        #data.append(data[0] and data[1] and data[8] and data[9])
        #data.append(data[0] and data[1] and data[4] and data[5])
        #data.append(data[4] and data[5] and data[6] and data[7])
        #data.append(data[6] and data[7] and data[10] and data[11])
        a.append(data)
        if m == 4000:#train + test
            break
    f.close()

#---------------- Get a mean value of 0 for each of the 12 bits of input ------------
# sum_inp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# curr_mean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# for bit in range(inputs):
#     for synd in range(len(a[:500])):
#         #sum_inp[bit] += a[synd][bit]
#         sum_inp[bit] += a[synd][bit] * c[synd][0]
#     #curr_mean[bit] = sum_inp[bit]/4000
#     curr_mean[bit] = sum_inp[bit]/749757
#     #print(curr_mean[bit])

# for bit in range(inputs):
#     for synd in range(len(a[:500])):
#         a[synd][bit] -= curr_mean[bit]
# #print(curr_mean)

# sum_inp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# for bit in range(inputs):
#     for synd in range(len(a[:500])):
#         #sum_inp[bit] += a[synd][bit]
#         sum_inp[bit] += a[synd][bit] * c[synd][0]
#     #curr_mean[bit] = sum_inp[bit]/4000
#     curr_mean[bit] = sum_inp[bit]/749757
#     #print(curr_mean[bit])
# #-----------------------------------------------------------------------------------

# #--------- Subtract all elements of the test set by the mean value calculated for the training set ---------
# for bit in range(inputs):
#     for synd in range(len(a[500:4000])):
#         a[synd][bit] -= curr_mean[bit]
# #-----------------------------------------------------------------------------------------------------------

# #--------- Get a standard deviation value of 1 for each of the 12 bits of input -----
# sum_col = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# var = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# std_dev = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# for bit in range(inputs):
#     for synd in range(len(a[:500])):
#         sum_col[bit] += (a[synd][bit] - curr_mean[bit])**2
#     var[bit] = sum_col[bit]/500
#     std_dev[bit] = np.sqrt(var[bit])
#     #print(std_dev[bit])

# for bit in range(inputs):
#     for synd in range(len(a[:500])):
#         a[synd][bit] = a[synd][bit] / std_dev[bit]
# #print(std_dev)

# sum_col = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# var = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# std_dev = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# for bit in range(inputs):
#     for synd in range(len(a[:500])):
#         sum_col[bit] += (a[synd][bit] - curr_mean[bit])**2
#     var[bit] = sum_col[bit]/500
#     std_dev[bit] = np.sqrt(var[bit])
#     #print(std_dev[bit])
# #-----------------------------------------------------------------------------------

# #--------- Divide all elements of the test set by the std_dev value calculated for the training set ---------
# for bit in range(inputs):
#     for synd in range(len(a[500:4000])):
#         a[synd][bit] = a[synd][bit] / std_dev[bit]
#-----------------------------------------------------------------------------------------------------------

'''
freq  = [1,12,55,55,50,50,30,5,0,0,0,0]
log   = [0, 0,35,35,30,30,15,3,0,0,0,0]
ident = [1,12,20,20,20,20,15,2,0,0,0,0]
new_a = []
new_b = []
I = 0
L = 0
for i in range(4000):
    sum = 0
    for j in range(inputs):
        sum += a[i][j]
    sum = int(sum)
    if freq[sum] > 0:
        if log[sum] > 0 and np.round_(b[i][1]) == 1:
            new_a.append(a[i])
            new_b.append(b[i])
            freq[sum] -= 1
            log[sum] -= 1
        elif ident[sum] > 0 and np.round_(b[i][0]) == 1:
            new_a.append(a[i])
            new_b.append(b[i])
            freq[sum] -= 1
            ident[sum] -= 1
      
for i in range(len(new_b)):
    I += np.round_(new_b[i][0])
    L += np.round_(new_b[i][1])

print(I,L)
#exit(0)
#print(len(new_a))
'''
#print(a[:15])
new_a = a[:500]
new_b = b[:500]
'''
for i in range(100,1000,9):
    new_a.append(a[i])
    new_b.append(b[i])
print(new_a[:10])
print(new_b[:10])
'''
#new_a = list(chain(a[:50], a[200:400]))
#new_b = list(chain(b[:50], b[200:400]))

new_a = np.array(new_a)
new_b = np.array(new_b)
train_inputs = new_a#a[300:508] 
train_targets = new_b#b[300:508]
test_inputs = np.array(a[500:1000])#a[208:1000]
test_targets = np.array(b[500:1000])#b[208:1000]
test_1_200_inp = np.array(a[1000:4000])#a[208:1000]
test_1_200_trg = np.array(b[1000:4000])#b[208:1000]
test_inputs_all = np.array(a)
test_targets_all = np.array(b)

x_data = tf.placeholder(tf.float32, shape=[None, inputs])
y_data = tf.placeholder(tf.float32, shape=[None, outputs])
#y_data_cls = tf.argmax(y_data, dimension=1)

W = []
W.append(weight_variable([inputs,nodes[0]]))
for i in range(len(nodes)-1):
	W.append(weight_variable([nodes[i],nodes[i+1]]))
W.append(weight_variable([nodes[-1],outputs]))
#print(np.array(W).shape)

nodes_str = ''
bias=[]
for i in range(len(nodes)):
    nodes_str += '_' + str(nodes[i])
    bias.append(bias_variable([nodes[i]]))
bias.append(bias_variable([outputs]))
#print(np.array(bias).shape)

h=[x_data]
for i in range(len(nodes)):
    h.append(tf.nn.sigmoid(tf.matmul(h[-1],W[i])+bias[i]))
    #h.append(tf.nn.tanh(tf.matmul(h[-1],W[i])+bias[i]))
#y_nn = tf.nn.sigmoid(tf.matmul(h[-1], W[-1]) + bias[-1])
y_nn = tf.matmul(h[-1], W[-1]) + bias[-1]
#y_nn = tf.nn.tanh(tf.matmul(h[-1], W[-1]) + bias[-1])
#print(y_nn.get_shape())

#keep_prob = tf.placeholder(tf.float32)
#y_pred = tf.nn.dropout(y_nn, 0.5)

#y_pred = tf.nn.softmax(y_nn)
y_pred = y_nn

#y_pred_cls = tf.argmax(y_pred, dimension=1)
#y_pred_cls = tf.argmax(y_nn, dimension=1)
#print(y_pred_cls.get_shape())
#cross_entropy = -tf.reduce_sum(y_data * tf.log(y_pred), reduction_indices=[1])
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_data)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_nn, labels=y_data)
#cross_entropy = tf.square(y_nn - y_data)
cost = tf.reduce_mean(cross_entropy)
train = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(cost)
#----- old code ----------------------------
#correct_prediction = tf.equal(tf.argmax(y_nn,1), tf.argmax(y_data,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#-------------------------------------------
#correct_prediction = tf.equal(y_pred_cls, y_data_cls)
#correct_prediction = tf.equal(tf.round(y_pred_cls), tf.round(y_data_cls))
correct_prediction = tf.equal(tf.round(y_pred[1]), tf.round(y_data[1]))
#y_pred = tf.with_dependencies([tf.assert_equal(y_pred, y_data)], y_pred) 
#with tf.control_dependencies([tf.assert_equal(y_pred, y_data)]):
#    y_pred = tf.identity(y_pred)
#accuracy = tf.reduce_mean(tf.cast(y_pred, tf.float32)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

NUM_CORES = 8  # Choose how many cores to use.
sess_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)

#------ test samples -----------
feed_dict_all = {x_data: test_inputs_all, y_data: test_targets_all}
feed_dict_test = {x_data: test_inputs, y_data: test_targets}
feed_dict_test200 = {x_data: test_1_200_inp, y_data: test_1_200_trg}
#------ train samples -----------
feed_dict_train = {x_data: train_inputs, y_data: train_targets}

#print(feed_dict_test200[y_data])
def myAccuracy(fd, data=False):
    predictions = sess.run(tf.nn.softmax(y_nn), feed_dict=fd)
    
    cnt = 0
    pl = len(predictions)
    for i in range(pl):
        if (predictions[i][0] > 0.5 and fd[y_data][i][0] > 0.5) or \
           (predictions[i][0] < 0.5 and fd[y_data][i][0] < 0.5):
        #if np.round_(predictions[i][0]) == np.round_(fd[y_data][i][0]) and \
        #   np.round_(predictions[i][1]) == np.round_(fd[y_data][i][1]):
            cnt += 1
        
        if data and i >= 480:    # last 20
            msg = "[{0:>12.9}, {1:>12.9}], [{2:>12.9}, {3:>12.9}],   {4:>5}"
            print(msg.format(predictions[i][0], fd[y_data][i][0], \
                             predictions[i][1], fd[y_data][i][1], cnt))
    return float(cnt)/pl



with tf.Session(config=sess_config) as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    
    start_time = time.time()
    i = 1
    #loss_prev = 0
    if first == 1:
        val_err = 100.0
        old_val_err = 100.0
        old_acc = 0.0
    else:
        saver.restore(sess, ff+'_model'+nodes_str+'.ckpt')
        err = sess.run(cost, feed_dict=feed_dict_train)
        ls = cost.eval(session=sess, feed_dict=feed_dict_test)
        #acc = sess.run(accuracy, feed_dict=feed_dict_train)
        msg = "Training Accuracy: {0:>6.2%}"
        acc = myAccuracy(feed_dict_test)
        old_acc = acc
        val_err = err
        old_val_err = err

        #acc = sess.run(accuracy, feed_dict=feed_dict_test)
        msg = "Testing Accuracy : {0:>6.2%}"
        print(msg.format(myAccuracy(feed_dict_test)))
        #old_err = sess.run(cost, feed_dict=feed_dict_train)
        
    while i <= FLAGS_training_steps:    #err > 0.001 and 
        i += 1
        sess.run(train, feed_dict=feed_dict_train)
        err = sess.run(cost, feed_dict=feed_dict_train)
        ls = cost.eval(session=sess, feed_dict=feed_dict_test)
        #acc = sess.run(accuracy, feed_dict=feed_dict_train)
        #if acc == 1.0:
        #    saver.save(sess, ff+'_model'+nodes_str+'.ckpt')
        #    break
        acc = myAccuracy(feed_dict_test)
            
        val_err = err
        if val_err < old_val_err:
            old_val_err = val_err
            #ls = cost.eval(session=sess, feed_dict=feed_dict_test)
            ##msg = "Iteration: {0:>6}, Error: {1:>12.9}, Loss: {2:>12.9}"
            ##print(msg.format(i, err, ls))
            ##saver.save(sess, ff+'_model'+nodes_str+'.ckpt')
        
        if  old_acc < acc:
            old_acc = acc
            saver.save(sess, ff+'_model'+nodes_str+'a.ckpt')
        
        if i % FLAGS_checkpoint_steps == 0:
            #acc = sess.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Error: {1:>6.2%}, Training Accuracy: {2:>6.2%}"
            print(msg.format(i, err, myAccuracy(feed_dict_train)))
            err = sess.run(cost, feed_dict=feed_dict_test)
            #acc = sess.run(accuracy, feed_dict=feed_dict_test)
            msg = "Optimization Iteration: {0:>6}, Testing Error:  {1:>6.2%}, Testing Accuracy:  {2:>6.2%}"
            print(msg.format(i, err, myAccuracy(feed_dict_test)))
    
    print("----------------------------------------")
    saver.save(sess, ff+'_model'+nodes_str+'.ckpt')
    saver.restore(sess, ff+'_model'+nodes_str+'.ckpt')
    err = sess.run(cost, feed_dict=feed_dict_train)
    #acc = sess.run(accuracy, feed_dict=feed_dict_train)
    msg = "Training Error: {0:>6.2%}, Training Accuracy: {1:>6.2%}"
    print(msg.format(err, myAccuracy(feed_dict_train, True)))
    err = sess.run(cost, feed_dict=feed_dict_test)
    #acc = sess.run(accuracy, feed_dict=feed_dict_test)
    msg = "Testing Error 300-1000: {0:>6.2%}, Testing Accuracy 300-1000: {1:>6.2%}"
    print(msg.format(err, myAccuracy(feed_dict_test)))
    err = sess.run(cost, feed_dict=feed_dict_test200)
    #acc = sess.run(accuracy, feed_dict=feed_dict_test200)
    msg = "Testing Error 1000-4000: {0:>6.2%}, Testing Accuracy 1000-4000: {1:>6.2%}"
    print(msg.format(err, myAccuracy(feed_dict_test200)))
    err = sess.run(cost, feed_dict=feed_dict_all)
    #acc = sess.run(accuracy, feed_dict=feed_dict_test200)
    msg = "Testing Error ALL: {0:>6.2%}, Testing Accuracy ALL: {1:>6.2%}"
    print(msg.format(err, myAccuracy(feed_dict_all)))
    #-----------------------------------------------------
    ##predictions = sess.run(y_nn, feed_dict=feed_dict_train)
    ##print(predictions[-20:])
    '''
    #saver.save(sess, ff+'_model'+nodes_str+'.ckpt')
    #print(err)
    #print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={x_data:test_inputs, y_data:test_targets}))
    #print("test loss %g" % loss.eval(session=sess, feed_dict={x_data:test_inputs, y_data:test_targets}))
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    '''
    #predictions = sess.run(y_pred, feed_dict={x_data:train_inputs})
    #predictions = sess.run(y_nn, feed_dict={x_data:train_inputs})
    #print(predictions[:20])
    #cnt = 0
    #for i in range(900):
    #    if np.round_(predictions[i][0]) == new_b[i][0]:
    #        cnt += 1
    #print(float(cnt)/9 '%')
    #predictions = sess.run(y_pred, feed_dict={x_data:test_inputs})
    #predictions = sess.run(y_nn, feed_dict={x_data:test_inputs})
    #print(predictions[200:220])
    