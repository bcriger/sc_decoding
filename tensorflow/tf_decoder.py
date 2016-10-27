from __future__ import print_function
import numpy as np

batch_x = np.array([])
batch_y = np.array([])

# Import data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_set_x = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0]])
train_set_y = np.array([[0], [0], [0], [0], [1], [1], [0], [1], [0], [0]])
#ds = Dataset(train_set)
eval_set_data  = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 0, 1]])
eval_set_labels = np.array([[0], [1], [1], [1], [1], [0]])

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 30
batch_size = 10          # mini batch for SGD
display_step = 1
total_batch = 1			# number_of_examples / batch_size

# Network Parameters
n_hidden_1 = 2 # 1st layer number of neurons (features)
n_input = 4  # 4 neurons (flips)
n_classes = 1 # 1 neuron (value = 0 or 1)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))   # cost = (pred - y)^2
cost = tf.reduce_mean((pred-y)**2)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        # Loop over all batches
        #for i in range(total_batch):
            #batch_x, batch_y = train_set.train.next_batch(batch_size)	# pick the first 5 from the train_set and then the next 5
        #for i in range(batch_size):
        #batch_x = train_set_x
        #batch_y = train_set_y
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: train_set_x, y: train_set_y})
        print('y', train_set_y)
        # Compute average loss
        avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: eval_set_data, y: eval_set_labels}))
