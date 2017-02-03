import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)#0.01, 0.05, 0.1
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.05, shape=shape)
    return tf.Variable(initial)

distance = 5
inputs = 12
outputs = 2
nodes = [90]
first = 1
#x_anc_pos = [10, 8, 30, 28, 20, 18, 40, 38, 1, 0, 47, 48]
FLAGS_train = True	# True : for training (False : for prediction)
FLAGS_training_steps = 10000
FLAGS_checkpoint_steps = 50 # save every x iterations
print(nodes)

a = []
b = []
samples = {}

with open('d=5_p=0.08_samples.txt', 'r') as f:
    m = 0
    for line in f: # iterate over each line
        m += 1
        data = line.split() # split it by whitespace
        dumb = [float(data[-3]), float(data[-2])]
        data = [int(d) for d in data[:inputs]]
        a.append(data)
        b.append(dumb)
        '''
        s = ''
        for i in range(inputs):
            s += str(data[i])

        if s in samples:
            x = samples[s]
            samples[s] = [x[0] + (dumb[0]+1)%2, x[1] + dumb[0]%2]
        else:
            samples[s] = [(dumb[0]+1)%2, dumb[0]%2]
        '''
        if m == 2968:
            break
    f.close()
'''
print(len(samples))
sorted_samples = sorted(samples.items(), key=lambda e: e[1][0] + e[1][1], reverse=True)

with open('d_5_dataset_million_smpl.txt', 'w') as f:
    #m = 0
    for key, value in sorted_samples:
        x = ''
        for i in range(inputs):
            x += key[i] + ' '

        x += ' '
        x += str(value[0]) + ' '
        x += str(value[1]) + '\n'
        f.write(x)

        #if value[0] * value[1] > 0:
        #    m += 1
            #print(key, value[0], value[1])
exit(0)
'''
train_inputs = np.array(a)
train_targets = np.array(b)
test_inputs = np.array(a)
test_targets = np.array(b)

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
#y_nn = tf.nn.softmax(y_nn)  # for output
#---------------------------------------------------------
#y_nn_cls = tf.argmax(y_nn, dimension=1)
#cross_entropy = -tf.reduce_sum(y_data * tf.log(y_nn), reduction_indices=[1])
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_nn, labels=y_data)
cross_entropy = tf.square(y_nn - y_data)
cost = tf.reduce_mean(cross_entropy)
train = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(cost)
#---------------------------------------------------------

NUM_CORES = 8  # Choose how many cores to use.
sess_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)

flnm = 'd_5' + nodes_str + '_' + str(len(train_inputs)) + '.ckpt'
#------ train samples -----------
feed_dict_train = {x_data: train_inputs, y_data: train_targets}
#------ test samples -----------
feed_dict_test = {x_data: test_inputs, y_data: test_targets}

def myAccuracy(fd):
    predictions = sess.run(y_nn, feed_dict=fd)
    
    cnt = 0
    pl = len(predictions)
    for i in range(pl):
        if np.round_(predictions[i][0]) == np.round_(fd[y_data][i][0]) and \
           np.round_(predictions[i][1]) == np.round_(fd[y_data][i][1]):
            cnt += 1
        
    return float(cnt)/pl

def myPrint(fd, start, end):    #[start, end)
    predictions = sess.run(y_nn, feed_dict=fd)
    msg = "[{0:>12.9}, {1:>12.9}], [{2:>12.9}, {3:>12.9}]"
    for i in range(len(predictions)):
        if i >= start and i < end:
            print(msg.format(predictions[i][0], fd[y_data][i][0], predictions[i][1], fd[y_data][i][1]))

with tf.Session(config=sess_config) as sess:
    #sess.run(tf.global_variables_initializer())
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    if first == 0:
        saver.restore(sess, flnm)

    msg = "Iteration: {0:>6}, Training Error: {1:>12.9}"
    msg1 = "Iteration: {0:>6}, Training Error: {1:>12.9}, Training Accuracy: {2:>6.2%}"
    msg2 = "Iteration: {0:>6}, Testing Error: {1:>12.9}, Testing Accuracy: {2:>6.2%}"
    i = 1
    while i <= FLAGS_training_steps:
        sess.run(train, feed_dict=feed_dict_train)

        if i % (FLAGS_checkpoint_steps * 10) == 0:
            saver.save(sess, flnm)
            print(msg1.format(i, sess.run(cost, feed_dict=feed_dict_train), myAccuracy(feed_dict_train)))
            #print(msg2.format(i, sess.run(cost, feed_dict=feed_dict_test), myAccuracy(feed_dict_test)))
        elif i % FLAGS_checkpoint_steps == 0:
            print(msg1.format(i, sess.run(cost, feed_dict=feed_dict_train), myAccuracy(feed_dict_train)))
            #print(msg.format(i, sess.run(cost, feed_dict=feed_dict_train)))

        i += 1

    saver.save(sess, flnm)

    print("----------------------------------------")
    msg = "Training Error: {0:>12.9}, Training Accuracy: {1:>6.2%}"
    print(msg.format(sess.run(cost, feed_dict=feed_dict_train), myAccuracy(feed_dict_train)))

'''
    print("----------------------------------------")
    myPrint(feed_dict_train, 0, 20)
    print("----------------------------------------")
    myPrint(feed_dict_test, 14, 25)
'''