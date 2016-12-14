import tensorflow as tf
import numpy as np
import time
from datetime import timedelta

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.05, shape=shape)
    return tf.Variable(initial)

distance = 5
inputs = 12
outputs = 2
nodes = [12]
ff = 'd=5_p=0.01'
print(nodes)

a = []
b = []
with open(ff + '_samples.txt', 'r') as f:
    m = 0
    for line in f: # iterate over each line
        m += 1
        data = line.split() # split it by whitespace
        b.append([float(data[-3]), float(data[-2])])
        data = [float(d) for d in data[:inputs]]
        a.append(data)
        if m == 1000:#train + test
            break
    f.close()
a = np.array(a)
b = np.array(b)

train_inputs = a[:208] 
train_targets = b[:208]
test_inputs = a[208:1000]
test_targets = b[208:1000]

x_data = tf.placeholder(tf.float32, shape=[None, inputs])
y_data = tf.placeholder(tf.float32, shape=[None, outputs])
y_data_cls = tf.argmax(y_data, dimension=1)

W = []
W.append(weight_variable([inputs,nodes[0]]))
# for i in range(len(nodes)-1):
#   W.append(weight_variable([nodes[i],nodes[i+1]]))
W.append(weight_variable([nodes[-1],outputs]))

nodes_str = ''
bias=[]
for i in range(len(nodes)):
    nodes_str += '_' + str(nodes[i])
    bias.append(bias_variable([nodes[i]]))
bias.append(bias_variable([outputs]))

h=[x_data]
for i in range(len(nodes)):
    h.append(tf.nn.sigmoid(tf.matmul(h[-1], W[i]) + bias[i]))
y_nn = tf.nn.sigmoid(tf.matmul(h[-1], W[-1]) + bias[-1])

#y_pred = tf.nn.softmax(y_nn)
#y_pred_cls = tf.argmax(y_pred, dimension=1)
y_pred_cls = tf.argmax(y_nn, dimension=1)
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_data)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_nn, targets=y_data)
cost = tf.reduce_mean(cross_entropy)
#train = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(cross_entropy)
train = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(cost)
correct_prediction = tf.equal(tf.round(y_pred_cls), tf.round(y_data_cls))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

FLAGS_train = True  # True : for training (False : for prediction)
FLAGS_training_steps = 1000
FLAGS_checkpoint_steps = 50 # save every x iterations
NUM_CORES = 8  # Choose how many cores to use.
sess_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)
val_err = 0

#------ test samples -----------
feed_dict_test = {x_data: test_inputs, y_data: test_targets}
#------ train samples -----------
feed_dict_train = {x_data: train_inputs, y_data: train_targets}

with tf.Session(config=sess_config) as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    first = 1
    if FLAGS_train: # training
        start_time = time.time()
        i = 1
        loss_prev = 0
        
        if first == 1:
            err = 100.0
            old_err = 100.0
        else:
            saver.restore(sess, ff+'_model'+nodes_str+'.ckpt')
            err = sess.run(cost, feed_dict=feed_dict_train)
            old_err = sess.run(cost, feed_dict=feed_dict_train)
        
        while err > 0.001 and i <= FLAGS_training_steps:
            i += 1
            sess.run(train, feed_dict=feed_dict_train)
            err = sess.run(cost, feed_dict=feed_dict_train)
            #print(err)
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            
            if err < old_err:
                cnt = 0
                old_err = err
                ls = cost.eval(session=sess, feed_dict=feed_dict_test)
                msg = "Iteration: {0:>6}, Error: {1:>12.9}, Loss: {2:>12.9}"
                print(msg.format(i, err, ls))
                saver.save(sess, ff+'_model'+nodes_str+'.ckpt')
                #if ls > loss_prev:
                #    val_err += 1
                #loss_prev = ls
                #if val_err == 6:
                #    val_err = 0
                #    break
            
            if i % FLAGS_checkpoint_steps == 0:
                msg = "Optimization Iteration: {0:>6}, Training Error: {1:>6.2%}, Training Accuracy: {2:>6.2%}"
                print(msg.format(i, err, acc))
                err = sess.run(cost, feed_dict=feed_dict_test)
                acc = sess.run(accuracy, feed_dict=feed_dict_test)
                msg = "Optimization Iteration: {0:>6}, Testing Error:  {1:>6.2%}, Testing Accuracy:  {2:>6.2%}"
                print(msg.format(i, err, acc))

        #saver.save(sess, ff+'_model'+nodes_str+'.ckpt')
        err = sess.run(cost, feed_dict=feed_dict_train)
        acc = sess.run(accuracy, feed_dict=feed_dict_train)
        msg = "Optimization Iteration: {0:>6}, Training Error: {1:>6.2%}, Training Accuracy: {2:>6.2%}"
        print(msg.format(i, err, acc))
        err = sess.run(cost, feed_dict=feed_dict_test)
        acc = sess.run(accuracy, feed_dict=feed_dict_test)
        msg = "Optimization Iteration: {0:>6}, Testing Error: {1:>6.2%}, Testing Accuracy: {2:>6.2%}"
        print(msg.format(i, err, acc))
        #saver.save(sess, ff+'_model'+nodes_str+'.ckpt')
        #print(err)
        #print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={x_data:test_inputs, y_data:test_targets}))
        #print("test loss %g" % loss.eval(session=sess, feed_dict={x_data:test_inputs, y_data:test_targets}))
        end_time = time.time()
        time_dif = end_time - start_time
        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
        #predictions = sess.run(y_pred, feed_dict={x_data:train_inputs})
        predictions = sess.run(y_nn, feed_dict={x_data:train_inputs})
        print(predictions[:10])
        #predictions = sess.run(y_pred, feed_dict={x_data:test_inputs})
        predictions = sess.run(y_nn, feed_dict={x_data:test_inputs})
        print(predictions[:10])