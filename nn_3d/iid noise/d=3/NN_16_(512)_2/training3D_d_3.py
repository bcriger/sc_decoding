import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)#0.01, 0.05, 0.1
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.05, shape=shape)
    return tf.Variable(initial)

d = 3
no_anc = (d + 1) * (d - 1) / 2 
inputs = int((d + 1) * no_anc)
outputs = 2
nodes = [512]
first = 0
absolute = True
p = 0.04
#no_samples = 12211
no_samples = 10088
#x_anc_pos = [10, 8, 30, 28, 20, 18, 40, 38, 1, 0, 47, 48]
FLAGS_train = True	# True : for training (False : for prediction)
FLAGS_training_steps = 10000
FLAGS_checkpoint_steps = 20 # save every x iterations
print(nodes)

a = []
b = []

#with open('d='+str(d)+'_p='+str(p)+'_3D_'+str(no_samples)+'_samples.txt', 'r') as f:
with open('comparison.txt', 'r') as f:
    m = 0
    for line in f: # iterate over each line
        m += 1
        data = line.split() # split it by whitespace
        flips_Z = [int(k) for k in data[:inputs]]
        dumb = [int(k) for k in data[inputs:]]
        if dumb[-3] != dumb[-2]:
            a.append(flips_Z)
            b.append([dumb[-3]/dumb[-1], dumb[-2]/dumb[-1]])
        #log_err = [int(k) for k in data[inputs:inputs+2]]
        #b.append(log_err)

        if m == no_samples:
            break
no_samples = len(a)

train_inputs = np.array(a)
train_targets = np.array(b)
test_inputs = np.array(a[:623]) #623
test_targets = np.array(b[:623])

x_data = tf.placeholder(tf.float32, shape=[None, inputs])
y_data = tf.placeholder(tf.float32, shape=[None, outputs])
#y_data_cls = tf.argmax(y_data, dimension=1)

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

h=[x_data]
for i in range(len(nodes)):
    h.append(tf.nn.sigmoid(tf.matmul(h[-1],W[i])+bias[i]))
y_nn = tf.nn.sigmoid(tf.matmul(h[-1], W[-1]) + bias[-1])
#y_nn = tf.matmul(h[-1], W[-1]) + bias[-1]
#y_pred = y_nn
#y_nn = tf.nn.softmax(y_nn)  # for output
#---------------------------------------------------------
#y_nn_cls = tf.argmax(y_nn, dimension=1)
#cross_entropy = -tf.reduce_sum(y_data * tf.log(y_nn), reduction_indices=[1])
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_data)
cross_entropy = tf.square(y_nn - y_data)
cost = tf.reduce_mean(cross_entropy)
train = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cost)
#---------------------------------------------------------

NUM_CORES = 16  # Choose how many cores to use.
sess_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)

#------ train samples -----------
feed_dict_train = {x_data: train_inputs, y_data: train_targets}
#------ test samples -----------
feed_dict_test = {x_data: test_inputs, y_data: test_targets}

def myAccuracy(fd):
    predictions = sess.run(y_nn, feed_dict=fd)
    
    cnt = 0
    pl = len(predictions)
    for i in range(pl):
        cnt_dq_err = 0
        for k in range(outputs):
            if np.round_(predictions[i][k]) == np.round_(fd[y_data][i][k]):
                cnt_dq_err += 1
        if cnt_dq_err == outputs:
            cnt += 1
        
    return float(cnt)/pl

def myPrint(fd, start, end):    #[start, end)
    predictions = sess.run(y_nn, feed_dict=fd)
    print(fd[y_data][1])
    print(predictions[1])
    return
    msg = "[{0:>12.9}, {1:>12.9}], [{2:>12.9}, {3:>12.9}]"
    for i in range(len(predictions)):
        if i >= start and i < end:
            print(msg.format(predictions[i][0], fd[y_data][i][0], predictions[i][1], fd[y_data][i][1]))

filename = 'd_'+str(d) + nodes_str + '_' + str(no_samples)
with tf.Session(config=sess_config) as sess:
    #sess.run(tf.global_variables_initializer())
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    if first == 0:
        saver.restore(sess, filename + '.ckpt')

    msg = "{0:} Iteration: {1:>6}, Training Error: {2:>12.9}"
    msg1 = "{0:} Iteration: {1:>6}, Training Error: {2:>12.9}, Training Accuracy: {3:>6.2%}"
    msg2 = "Iteration: {0:>6}, Testing Error: {1:>12.9}, Testing Accuracy: {2:>6.2%}"
    i = 1
    old_acc = 0.0
    while i <= FLAGS_training_steps:
        sess.run(train, feed_dict=feed_dict_train)
        if absolute:
            acc = myAccuracy(feed_dict_train)
            if acc > old_acc:
                old_acc = acc
                saver.save(sess, filename + '_abs.ckpt')

        if i % (FLAGS_checkpoint_steps * 10) == 0:
            saver.save(sess, filename + '.ckpt')
            print(msg1.format(nodes_str, i, sess.run(cost, feed_dict=feed_dict_train), myAccuracy(feed_dict_train)))
            #print(msg2.format(i, sess.run(cost, feed_dict=feed_dict_test), myAccuracy(feed_dict_test)))
        elif i % FLAGS_checkpoint_steps == 0:
            print(msg1.format(nodes_str, i, sess.run(cost, feed_dict=feed_dict_train), myAccuracy(feed_dict_train)))
            #print(msg.format(nodes_str, i, sess.run(cost, feed_dict=feed_dict_train)))

        i += 1

    saver.save(sess, filename + '.ckpt')

    print("----------------------------------------")
    msg = "Training Error: {0:>12.9}, Training Accuracy: {1:>6.2%}"
    print(msg.format(sess.run(cost, feed_dict=feed_dict_train), myAccuracy(feed_dict_train)))
    print(msg2.format(i, sess.run(cost, feed_dict=feed_dict_test), myAccuracy(feed_dict_test)))
       
'''
    print("----------------------------------------")
    myPrint(feed_dict_train, 0, 20)
    print("----------------------------------------")
    myPrint(feed_dict_test, 14, 25)
'''