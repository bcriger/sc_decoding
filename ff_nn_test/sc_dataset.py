"""
Functions for converting my nice matrices into MNIST-like data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pkl
import gzip

import numpy
from six.moves import xrange    # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

class DataSet(object):

    def __init__(self, inputs, labels, fake_data=False, one_hot=False,
                    dtype=dtypes.float32, reshape=True):
        """
        Construct a DataSet.
        one_hot arg is used only if fake_data is true.    `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                                            dtype)
        self._num_examples = inputs.shape[0]
        self._inputs = inputs
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """
        Return the next `batch_size` examples from this data set.
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._inputs = self._inputs[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._inputs[start:end], self._labels[start:end]


def read_data_sets(train_dir, one_hot=False, dtype=dtypes.float32,
                    reshape=True, validation_size=5000):
    with open('big_dataset_d5_p5.pkl', 'rb') as phil:
        data_dict = pkl.load(phil)
        big_input_mat = data_dict['input']
        big_label_set = data_dict['labels']
    
    trn_sz = 78000
    val_sz = 7000
    tst_sz = 15000
    
    train = DataSet(big_input_mat[:trn_sz, :], big_label_set[:trn_sz],
                                        dtype=dtype, reshape=reshape)
    validation = DataSet(big_input_mat[trn_sz:trn_sz + val_sz, :], big_label_set[trn_sz:trn_sz + val_sz],
                                        dtype=dtype, reshape=reshape)
    test = DataSet(big_input_mat[trn_sz + val_sz:, :], big_label_set[trn_sz + val_sz:],
                                        dtype=dtype, reshape=reshape)


    return base.Datasets(train=train, validation=validation, test=test)
