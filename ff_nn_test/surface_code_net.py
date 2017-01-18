"""
Builds a network to decode small instances of the surface code.
Implements the inference/loss/training pattern for model building.
1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
This file is not meant to be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# I'm making a dumb dataset which is just correct decoding instances.
NUM_CLASSES = 2

# I want to mess with arbitrary distance codes, decoding bit or phase flip individually
DISTANCE = 5
if not(DISTANCE % 2):
    raise ValueError("formula for number of syndromes won't work")
N_SYNDS = (DISTANCE ** 2 - 1) // 2 # only works for odd dist


def inference(synds, units_lst):
    """
    Build the decoding model up to where it may be used for inference.
    Args:
        synds: bit or scaled/shifted bit lists (just be consistent). Ought to be a matrix.
        units_lst: integer list, gives the width of each hidden layer. 
    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    
    size_lst = [N_SYNDS] + units_lst
    internal_state = synds
    
    for layer_idx in range(len(units_lst)):
        layer_name = 'hidden' + str(layer_idx)
        with tf.name_scope(layer_name):
            weights = tf.Variable(
                    tf.truncated_normal(size_lst[layer_idx:layer_idx + 2],
                                        stddev=1.0 / math.sqrt(float(size_lst[layer_idx]))),
                    name='weights')
            biases = tf.Variable(tf.zeros(size_lst[layer_idx + 1]), name='biases')
            internal_state = tf.nn.sigmoid(tf.matmul(internal_state, weights) + biases)
    
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
                tf.truncated_normal([size_lst[-1], NUM_CLASSES],
                                                        stddev=1.0 / math.sqrt(float(size_lst[-1]))),
                name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                                                 name='biases')
        logits = tf.matmul(internal_state, weights) + biases
    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
    Returns:
        loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    """
    Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))