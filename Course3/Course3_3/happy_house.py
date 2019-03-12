#-*- coding: utf-8 -*-

from __future__ import division
import os
import h5py
import numpy as np 
import tensorflow as tf

dataset_dir = 'datasets/'
train_happy_path = os.path.join(dataset_dir, 'train_happy.h5')
test_happy_path = os.path.join(dataset_dir, 'test_happy.h5')

train_happy = h5py.File(train_happy_path, 'r')
# print(list(train_happy.keys()))
train_set_x = np.array(train_happy['train_set_x'])
train_set_y = np.array(train_happy['train_set_y'])

train_set_num, height, width, channel = train_set_x.shape

# construct the network
with tf.name_scope('Input'):
	X = tf.placeholder(dtype = tf.float32, shape = (None, height, width, channel), name = 'X')
	y = tf.placeholder(dtype = tf.float32, shape = (None, 1), name = 'y')

with tf.name_scope('Layer1'):
	W_array1 = tf.get_variable(name = 'W_array1', shape = [3, 3, 3, 8], dtype = tf.float32,
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b_array1 = tf.get_variable(name = 'b_array1', shape = [8], dtype = tf.float32,
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	Z1 = tf.nn.conv2d(X, W_array1, strides = [1, 1, 1, 1], padding = 'SAME', name = 'Z1')
	# local response normalization before input into the activation function
	Z1 = tf.nn.lrn(input = Z1, depth_radius = 5, bias = 1, alpha = 1, beta = 0.5, name = 'Z1_LRN')  
	A1 = tf.nn.relu(Z1, name = 'A1')
	A1_pool = tf.nn.max_pool(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], 
		padding = 'SAME', name = 'A1_pool')

with tf.name_scope('Layer2'):
	W_array2 = tf.get_variable(name = 'W_array2', shape = [3, 3, 8, 16], dtype = tf.float32,
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b_array2 = tf.get_variable(name = 'b_array2', shape = [16], dtype = tf.float32,
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	Z2 = tf.
