# -*- coding: utf-8 -*-
import sys
import h5py
import math
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 


def linear_function():

	np.random.seed(1)
	X = tf.constant(np.random.randn(3, 1), name = 'X')
	W = tf.constant(np.random.randn(4, 3), name = 'W')
	b = tf.constant(np.random.randn(4, 1), name = 'b')
	Y = tf.add(tf.matmul(W, X), b)

	sess = tf.Session()
	result = sess.run(Y)

	sess.close()
	return result

def load_dataset():

	train_dataset = h5py.File('datasets/train_signs.h5', 'r')
	train_set_x_orig = np.array(train_dataset['train_set_x'][:])
	train_set_y_orig = np.array(train_dataset['train_set_y'][:])

	test_dataset = h5py.File('datasets/test_signs.h5', 'r')
	test_set_x_orig = np.array(test_dataset['test_set_x'][:])
	test_set_y_orig = np.array(test_dataset['test_set_y'][:])

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig
	

if __name__ == '__main__':

	# load_dataset()
	print('result = ' + str(linear_function()))