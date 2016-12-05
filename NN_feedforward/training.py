import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)
distance = 9
inputs = 40
outputs = 1
nodes = [25]
print(nodes)

x_data = tf.placeholder(tf.float32, shape=[60158, inputs])
y_data = tf.placeholder(tf.float32, shape=[60158, outputs])
x_test = tf.placeholder(tf.float32, shape=[60158, inputs])
y_test = tf.placeholder(tf.float32, shape=[60158, outputs])

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
h1=[x_test]
for i in range(len(nodes)):
    h1.append(tf.nn.sigmoid(tf.matmul(h1[-1],W[i])+bias[i]))
y_pred = tf.nn.sigmoid(tf.matmul(h1[-1], W[-1]) + bias[-1])

# Minimize the mean squared errors.
correct_prediction = tf.equal(tf.argmax(y_nn,1), tf.argmax(y_data,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
loss = tf.reduce_mean(tf.square(y_nn - y_data))
optimizer = tf.train.AdamOptimizer(0.1)
train = optimizer.minimize(loss)

a = []
b = []
with open('d=9_uniform_distr_samples_new.txt', 'r') as f:
    m = 0
    for line in f: # iterate over each line
        m += 1
        data = line.split() # split it by whitespace
        a.append(data[:inputs])
        b.append([data[-2]])
        if m == 120316:#train + test
            break
    f.close()
a = np.array(a)
b = np.array(b)


train_inputs = a[:60158]     #np.swapaxes(a, 0, 1)[:99000]
train_targets = b[:60158]    #np.swapaxes(b, 0, 1)[:99000]
test_inputs = a[60158:120316]    #np.swapaxes(a, 0, 1)[1000:100000]
test_targets = b[60158:120316]    #np.swapaxes(b, 0, 1)[1000:100000]

FLAGS_train = True	# True : for training (False : for prediction)
FLAGS_training_steps = 50000
FLAGS_checkpoint_steps = 50 # save every x iterations
NUM_CORES = 8  # Choose how many cores to use.
sess_config = tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES, intra_op_parallelism_threads=NUM_CORES)
val_err = 0
with tf.Session(config=sess_config) as sess:
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    if FLAGS_train:	# training
        i = 0
        #saver.restore(sess, 'model_d_' + str(distance) + nodes_str+'.ckpt')
        err = 100.0#sess.run(loss,feed_dict={x_data:train_inputs, y_data:train_targets})
        #print(err)
        old_err = 100.0#sess.run(loss,feed_dict={x_data:train_inputs, y_data:train_targets})
        while err > 0.000001 and i <= FLAGS_training_steps:
            i += 1
            sess.run(train,feed_dict={x_data:train_inputs, y_data:train_targets})
            err = sess.run(loss,feed_dict={x_data:train_inputs, y_data:train_targets})
            if err < old_err:
                cnt = 0
                old_err = err
                ls = loss.eval(session=sess, feed_dict={x_data:test_inputs, y_data:test_targets})
                print(i, 'error=',err, 'loss=',ls)
                saver.save(sess, 'model_d_' + str(distance) + nodes_str+'.ckpt')
                if ls > loss_prev:
                    val_err += 1
                loss_prev = ls
                if val_err == 6:
                    val_err = 0
                    break

            if i % FLAGS_checkpoint_steps == 0:
                print('\t', i, err)			

        #saver.save(sess, 'model_d_' + str(distance) + nodes_str+'.ckpt')
        print(err)
        #print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={x_data:test_inputs, y_data:test_targets}))
        print("test loss %g" % loss.eval(session=sess, feed_dict={x_data:test_inputs, y_data:test_targets}))
        predictions = sess.run(y_nn, feed_dict={x_data:train_inputs})

        cnt = 0
        for i in range(len(predictions)):
            if round(predictions[i][0],0) == float(train_targets[i][0]):
                cnt += 1
        print(str((float(cnt)/len(predictions))*100)+'%', cnt)
    else:	# prediction
        saver.restore(sess, 'model_d_' + str(distance) + nodes_str+'.ckpt')
        err = sess.run(loss,feed_dict={x_data:train_inputs, y_data:train_targets})
        print(err)
        predictions = sess.run(y_pred, feed_dict={x_test:test_inputs})
        print(predictions)
        '''
        p = predictions[:]
        t = train_targets[:]
		#t = test_targets[:500]
        cnt = 0
        for i in range(len(p)):
            if round(p[i][0],0) == t[i][0]:
                cnt += 1
        print(str(float(cnt)/len(p)*100)+'%')
        '''