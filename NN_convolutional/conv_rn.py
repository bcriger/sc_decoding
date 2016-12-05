# https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html#build-a-multilayer-convolutional-network

import tensorflow as tf
import numpy as np
import random
import cPickle as cp

L = 8

batch_size = 200


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial,name=name)

def filter_const_one(shape): # for easy parity computation
    # check whether numerical accuracy will matter
    return tf.constant(1.0,shape=shape)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial,name=name)


# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')
#
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1], padding='VALID')
#
# def avg_pool_2x2(x):
#     return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
#                           strides=[1, 2, 2, 1], padding='VALID')
#

class BatchIterator:
    def __init__(self, data):
        self.test_size = 200
        self.anyons = data['anyons'].astype(np.float32)
        self.logicals = data['logicals'].astype(np.float32)
        self.rn_current = data['rn_current1'].astype(np.float32)
        self.batch_start = 0
        self.num_data = len(self.anyons) - self.test_size  # 1000 is the test set

    def __iter__(self):
        return self

    def test_batch(self):
        anyons_batch = self.anyons[self.num_data:,:,:,:]
        rn_current_batch = self.rn_current[self.num_data:,:,:,:]
        logicals_batch = self.logicals[self.num_data:].reshape(self.test_size,1)

        return anyons_batch, rn_current_batch, logicals_batch

    def next_batch(self):
        batch_end = self.batch_start + batch_size

        if batch_end >= self.num_data:  # temporarily solution
            self.batch_start = 0
            batch_end = self.batch_start + batch_size

        anyons_batch = self.anyons[self.batch_start:batch_end,:,:,:]
        rn_current_batch = self.rn_current[self.batch_start:batch_end,:,:,:]
        logicals_batch = self.logicals[self.batch_start:batch_end].reshape(batch_size,1)

        self.batch_start = batch_end
        return anyons_batch, rn_current_batch,logicals_batch


def load_data():
    with open('processed_data_rn_8_0.05_20000.pkl', 'r') as f:
        data = cp.load(f)

    batch_gen=BatchIterator(data)
    return batch_gen

def pre_training(batch_gen):

    num_filters=20

    conv1_size=3

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, L, L, 2])
        y_ = tf.placeholder(tf.float32, shape=[None, L/2,L/2, 2])

        W_conv1 = weight_variable([2, 2, 2, num_filters],name='W_conv1')
        b_conv1 = bias_variable([num_filters], name='b_conv1')


        h_conv1 = tf.nn.tanh(tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='VALID') + b_conv1)

        # print('h_conv1', h_conv1._shape)

        W_conv2 = weight_variable([conv1_size, conv1_size, num_filters, num_filters],name='W_conv2')
        b_conv2 = bias_variable([num_filters],name='b_conv2')


        h_conv2 = tf.nn.tanh(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

        W_conv3 = weight_variable([conv1_size, conv1_size, num_filters, 2],name='W_conv3')
        b_conv3 = bias_variable([2],name='b_conv3')


        y_conv = tf.nn.sigmoid(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

        # print('y_conv', y_conv._shape)

        cross_entropy = tf.reduce_mean(-y_ * tf.log(y_conv)- (1-y_)*tf.log(1-y_conv))

        #
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(5e-3).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.round(y_conv), tf.round(y_))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.initialize_all_variables())



        test_a, test_r,test_l = batch_gen.test_batch()
        for i in range(4000):
            a, r,l = batch_gen.next_batch()
            # a.resize((batch_size, L, L, 1))
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: a, y_: r})
                print("step %d, accuracy %g" % (i, train_accuracy))
                print("test accuracy %g" % accuracy.eval(feed_dict={x: test_a, y_: test_r}))
                # print("prediction", y_conv.eval(feed_dict={x: test_a[:40], y_: test_l[:40], keep_prob: 1.0}))
                # print(parity.eval(feed_dict={x:a[0:1,:,:,:], keep_prob: 1.0}))
            train_step.run(feed_dict={x: a, y_: r})

        saver = tf.train.Saver()
        save_path = saver.save(sess, "pre_training.ckpt")

    tf.reset_default_graph()

def training(batch_gen):


    num_filters=20

    conv1_size=3
    conv2_size=3

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, L, L, 2])
        y_ = tf.placeholder(tf.float32, shape=[None, 1])

#--------------------redefining the pre-training graph-----------------

        W_conv1 = weight_variable([2, 2, 2, num_filters],name='W_conv1')
        b_conv1 = bias_variable([num_filters], name='b_conv1')


        h_conv1 = tf.nn.tanh(tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='VALID') + b_conv1)

        # print('h_conv1', h_conv1._shape)

        W_conv2 = weight_variable([conv1_size, conv1_size, num_filters, num_filters],name='W_conv2')
        b_conv2 = bias_variable([num_filters],name='b_conv2')


        h_conv2 = tf.nn.tanh(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

        W_conv3 = weight_variable([conv1_size, conv1_size, num_filters, 2],name='W_conv3')
        b_conv3 = bias_variable([2],name='b_conv3')


        y_conv = tf.nn.sigmoid(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

        saver=tf.train.Saver()

        # print('y_conv', y_conv._shape)

#---------------------------pre-training graph finishes----------------------


        parity_filter=filter_const_one([2, 2, 1,1])
        parity_filter=tf.pad(parity_filter,[[0,0],[0,0],[0,1],[0,0]])

        print('parity_filter', parity_filter._shape)


        rn_anyons=tf.mod(tf.round(tf.nn.conv2d(x, parity_filter, strides=[1, 2, 2, 1], padding='VALID')),2)
        # round to make the numerical accuracy better???

        # To input information about the left and right boundary
        rn_boundary=np.zeros((batch_size,4,4,1))
        rn_boundary[:,:,0,0]=1
        rn_boundary[:,:,3,0]=1

        new_input=tf.concat(3,[rn_anyons,y_conv,rn_boundary])

        print('new_input_shape', new_input._shape)




        W_conv4 = weight_variable([conv2_size, conv2_size, 4, num_filters],name='W_conv4')
        b_conv4 = bias_variable([num_filters], name='b_conv4')

        h_conv4 = tf.nn.tanh(tf.nn.conv2d(new_input, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

        W_conv5 = weight_variable([conv2_size, conv2_size, num_filters, 10],name='W_conv5')
        b_conv5 = bias_variable([10],name='b_conv5')


        h_conv5 = tf.nn.tanh(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

        num_hidden1=10
        W_fc1 = weight_variable([10 * h_conv5._shape[1]._value ** 2, num_hidden1])
        b_fc1 = bias_variable([num_hidden1])

        h_conv5_flat = tf.reshape(h_conv5, [-1, 10 * h_conv5._shape[1]._value ** 2])
        h_fc1 = tf.nn.tanh(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

        W_fc2 = weight_variable([num_hidden1,1])
        b_fc2 = bias_variable([1])

        y_final = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)
        print('y_final', y_final._shape)



        cross_entropy = tf.reduce_mean(-y_ * tf.log(y_final)- (1-y_)*tf.log(1-y_final))

        train_step = tf.train.AdamOptimizer(5e-3).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.round(y_final), tf.round(y_))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # sess.run(tf.initialize_variables([W_conv4,W_conv5,b_conv4,b_conv5,W_fc1,W_fc2,b_fc1,b_fc2]))
        sess.run(tf.initialize_all_variables())

        saver.restore(sess, "pre_training.ckpt")



        test_a, test_r,test_l = batch_gen.test_batch()
        for i in range(4000):
            a, r,l = batch_gen.next_batch()
            # a.resize((batch_size, L, L, 1))
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: a, y_: l})
                print("step %d, accuracy %g" % (i, train_accuracy))
                print("test accuracy %g" % accuracy.eval(feed_dict={x: test_a, y_: test_l}))
                # print("prediction", y_conv.eval(feed_dict={x: test_a[:40], y_: test_l[:40], keep_prob: 1.0}))
                # print(parity.eval(feed_dict={x:a[0:1,:,:,:], keep_prob: 1.0}))
            train_step.run(feed_dict={x: a, y_: l})

    tf.reset_default_graph()
	
ld = load_data()