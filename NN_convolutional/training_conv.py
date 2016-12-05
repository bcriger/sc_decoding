import tensorflow as tf
import numpy as np
import time
from datetime import timedelta

distance = 7
num_anc = 24

class Samples:
    def __init__(self, inputs, targets):
        self.test_size = 1000
        self.syndromes = inputs
        self.logicals = targets
        self.num_data = len(self.syndromes)  

    def __iter__(self):
        return self

    def test_samples(self):
        syndromes_batch = self.syndromes[-self.test_size:self.num_data]
        logicals_batch = self.logicals[-self.test_size:self.num_data]
        return syndromes_batch, logicals_batch

    def train_samples(self):
        batch_end = self.num_data - self.test_size
        syndromes_batch = self.syndromes[:batch_end]
        logicals_batch = self.logicals[:batch_end]
        return syndromes_batch, logicals_batch

def load_data(file, num_samples):
    a = []
    b = []
    with open(file, 'r') as f:
        m = 0
        for line in f: # iterate over each line
            m += 1
            data = line.split() # split it by whitespace
            for i in range(num_anc+2):
                data[i] = int(data[i])
            a.append(data[:num_anc])
            b.append([data[-2]])
            if m == num_samples:#train + test
                break
        f.close()
    a = np.array(a)
    b = np.array(b)

    return Samples(a, b)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(length):
    initial = tf.constant(0.05, shape=[length])
    return tf.Variable(initial)

num_channels = 1
num_classes = 1

# Convolutional Layer 1.
filter_size1 = 4          # Convolution filters are 5 x 5 pixels.
num_filters1 = 10         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 4          # Convolution filters are 5 x 5 pixels.
num_filters2 = 20         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.
	
def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    shape = [filter_size, num_input_channels, num_filters]
    weights = weight_variable(shape=shape)
    print('weights: ', weights.get_shape())
    biases = bias_variable(length=num_filters)
    layer = tf.nn.conv1d(value=input,
                              filters=weights,
                              stride=2,
                              padding='SAME')
    layer += biases
    print('layer: ', layer.get_shape())
    if use_pooling:
        layer = tf.nn.pool(input=layer,
                           window_shape=[1],
                           pooling_type="MAX",
                           strides=[1],
                           padding='SAME')
    layer = tf.nn.sigmoid(layer)
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:3].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_sigmoid=True): # Use sigmoid

    weights = weight_variable(shape=[num_inputs, num_outputs])
    biases = bias_variable(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
    if use_sigmoid:
        layer = tf.nn.sigmoid(layer)
    return layer
	
x = tf.placeholder(tf.float32, shape=[None, num_anc], name='x')
print('x: ', x._shape)
x_input = tf.reshape(x, [-1, num_anc, num_channels])
print('x_input: ', x_input._shape)
y_true = tf.placeholder(tf.float32, shape=[None, 1], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x_input,
                                            num_input_channels=num_channels,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=True)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)
				   
layer_flat, num_features = flatten_layer(layer_conv2)
print('num_features ', num_features)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_sigmoid=True)
				
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_sigmoid=False)
		 
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
loss = tf.reduce_mean(tf.square(y_pred_cls - y_true_cls))

session = tf.Session()
session.run(tf.initialize_all_variables())

total_iterations = 0	

def optimize(num_iterations):
    global total_iterations
    start_time = time.time()
    #------ test samples -----------
    test_a, test_l = data.test_samples()
    feed_dict_test = {x: test_a, y_true: test_l}
    #------ train samples -----------
    x_batch, y_true_batch = data.train_samples()
    feed_dict_train = {x: x_batch, y_true: y_true_batch}

    saver = tf.train.Saver()
    
    for i in range(total_iterations, total_iterations + num_iterations):
        session.run(optimizer, feed_dict=feed_dict_train)
        err = session.run(loss, feed_dict=feed_dict_train)
        acc = session.run(accuracy, feed_dict=feed_dict_test)

        if acc == 1.0:
            saver.save(session, 'model_d_' + str(distance) + '.ckpt')
            break


        if i % 10 == 0:
            msg = "Optimization Iteration: {0:>6}, Training Error: {1:>6.2%}, Training Accuracy: {2:>6.2%}"
            print(msg.format(i + 1, err, acc))

    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

test_batch_size = 200
'''
def print_test_accuracy():
    test_a, test_l = ld.test_samples()
    num_test = len(test_a)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0

    while i < num_test:
        j = min(i + test_batch_size, num_test)
        a, l = ld.next_batch()
        inputs =  test_a[i:j]
        targets = test_l[i:j]
        feed_dict = {x: inputs, y_true: targets}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    cls_true = test_l
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
'''
data = load_data('d=7_uniform_distr_samples.txt', 20000)
optimize(num_iterations=1000)
#print_test_accuracy()