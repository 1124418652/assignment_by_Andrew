#-*- coding: utf-8 -*-

from __future__ import division
import os
import h5py
import numpy as np 
import tensorflow as tf
# import matplotlib.pyplot as plt 

dataset_dir = 'datasets/'
train_happy_path = os.path.join(dataset_dir, 'train_happy.h5')
test_happy_path = os.path.join(dataset_dir, 'test_happy.h5')

train_happy = h5py.File(train_happy_path, 'r')
test_happy = h5py.File(test_happy_path, 'r')
# print(list(train_happy.keys()))

train_set_x = np.array(train_happy['train_set_x']) / 255
train_set_y = np.array(train_happy['train_set_y']).reshape([-1, 1]).astype(np.float32)
train_set_num, height, width, channel = train_set_x.shape
test_set_x = np.array(test_happy['test_set_x']) / 255
test_set_y = np.array(test_happy['test_set_y']).reshape([-1, 1]).astype(np.float32)
test_set_x.astype(np.float32)
# plt.imshow(train_set_x)
# plt.show()

# construct the network
with tf.name_scope('Input'):
	X = tf.placeholder(dtype = tf.float32, shape = (None, height, width, channel), name = 'X')
	y = tf.placeholder(dtype = tf.float32, shape = (None, 1), name = 'y')

with tf.name_scope('Conv_Layer1'):
	W_array1 = tf.Variable(tf.truncated_normal(shape = [3, 3, 3, 8], stddev = 0.01))
	b_array1 = tf.Variable(tf.constant(0, shape = [8]))
	Z1 = tf.nn.conv2d(X, W_array1, strides = [1, 1, 1, 1], padding = 'SAME', name = 'Z1')
	# local response normalization before input into the activation function
	Z1 = tf.nn.lrn(input = Z1, depth_radius = 5, bias = 1, alpha = 1, beta = 0.5, name = 'Z1_LRN')  
	A1 = tf.nn.relu(Z1, name = 'A1')
	A1_pool = tf.nn.max_pool(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], 
		padding = 'SAME', name = 'A1_pool')

with tf.name_scope('Conv_Layer2'):
	W_array2 = tf.Variable(tf.truncated_normal(shape = [3, 3, 8, 16], stddev = 0.01))
	b_array2 = tf.Variable(tf.constant(0, shape = [16]))
	Z2 = tf.nn.conv2d(A1_pool, W_array2, strides = [1, 1, 1, 1], padding = 'SAME', name = 'Z2')
	# local response normalization before input into the activation function
	Z2 = tf.nn.lrn(input = Z2, depth_radius = 5, bias = 1, alpha = 1, beta = 0.5, name = 'Z2_LRN')
	A2 = tf.nn.relu(Z2, name = 'A2')
	A2_pool = tf.nn.max_pool(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
		padding = 'SAME', name = 'A2_pool')

with tf.name_scope('Conv_Layer3'):
	W_array3 = tf.get_variable(name = 'W_array3', shape = [3, 3, 16, 32], dtype = tf.float32,
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b_array3 = tf.get_variable(name = 'b_array3', shape = [32], dtype = tf.float32,
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	Z3 = tf.nn.conv2d(A2_pool, W_array3, strides = [1, 1, 1, 1], padding = 'SAME', name = 'Z3')
	Z3 = tf.nn.lrn(input = Z3, depth_radius = 5, bias = 1, alpha = 1, beta = 0.5, name = 'Z3_LRN')
	A3 = tf.nn.relu(Z3, name = 'A3')
	A3_pool = tf.nn.max_pool(A3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
		padding = 'SAME', name = 'A3_pool')

with tf.name_scope('Full_Layer1'):
	A3_flatten = tf.layers.flatten(A3_pool, name = 'A3_flatten')
	W_array4 = tf.get_variable(name = 'W_array4', shape = [A3_flatten.shape[1], 1], dtype = tf.float32,
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b_array4 = tf.get_variable(name = 'b_array4', shape = [1], dtype = tf.float32,
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	matmul = tf.matmul(A3_flatten, W_array4)
	Z4 = matmul + b_array4
	A4 = tf.nn.sigmoid(Z4, name = 'A4')

# with tf.name_scope('Full_Layer2'):
# 	W_array5 = tf.get_variable(name = 'W_array5', shape = [A4.shape[1], 1], dtype = tf.float32,
# 		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
# 	b_array5 = tf.get_variable(name = 'b_array5', shape = [1], dtype = tf.float32,
# 		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
# 	matmul = tf.matmul(A4, W_array5)
# 	Z5 = matmul + b_array5
# 	A5 = tf.nn.sigmoid(Z5, name = 'A5')

Loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = A4), name = 'cost')
train_step = tf.train.AdamOptimizer(1e-4).minimize(Loss)

init = tf.global_variables_initializer()
prediction = tf.cast(A4 >= 0.5, dtype = tf.float32)
accuracy = 1 - tf.reduce_mean(tf.abs(y - prediction))

with tf.Session() as sess:
	sess.run(init)
	writer = tf.summary.FileWriter('logs', sess.graph)
	for epoch in range(2000):
		sess.run(train_step, feed_dict = {X: train_set_x, y: train_set_y})
		print("Iteration: %d , cost: %.6f, train accuracy: %f, test accuracy: %.4f" \
			% (epoch, Loss.eval(feed_dict = {X: train_set_x, y: train_set_y}), \
			accuracy.eval(feed_dict = {X: train_set_x, y: train_set_y}),
			accuracy.eval(feed_dict = {X: test_set_x, y: test_set_y})))
		print(Z4.eval(feed_dict = {X: train_set_x, y: train_set_y}).max())
		# print(W_array1.eval())