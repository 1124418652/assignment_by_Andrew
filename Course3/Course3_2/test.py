#-*- coding: utf-8 -*-
import os
import h5py
import numpy as np 
import tensorflow as tf


dataset_dir = 'datasets/'

def load_dataset(path):

	if not os.path.exists(path):
		raise ValueError('The path {0!r} is not exists!'.format(path))

	train_data_path = os.path.join(path, 'train_signs.h5')
	test_data_path = os.path.join(path, 'test_signs.h5')
	train_data = h5py.File(train_data_path, 'r')
	test_data = h5py.File(test_data_path, 'r')

	train_set_x = train_data['train_set_x'][:]
	train_set_y = train_data['train_set_y'][:]
	test_set_x = test_data['test_set_x'][:]
	test_set_y = test_data['test_set_y'][:]
	
	return train_set_x, train_set_y, test_set_x, test_set_y

def convert2onehot(y, num_classes = 6):

	y_onehot = np.zeros((y.shape[0], num_classes))
	for i, value in enumerate(y):
		y_onehot[i, value] = 1 
	return y_onehot

train_set_x, train_set_y, test_set_x, test_set_y = load_dataset(dataset_dir)
train_set_x = (train_set_x / 255).astype(np.float32)
test_set_x = (test_set_x / 255).astype(np.float32) 
train_set_y_onehot = convert2onehot(train_set_y, 6)
test_set_y_onehot = convert2onehot(test_set_y, 6)

def create_placeholders(n_height, n_width, n_channels, n_y_classes):
	"""
	Creates the placeholders for the tensorflow session.

	Arguments:
	n_height -- scalar, height of an input image
	n_width -- scalar, width of an input image
	n_channels -- scalar, channels of an input image
	n_y_classes -- scalar, the number of classes you want to classify

	Returns:
	X -- placeholder for the data input, of shape [None, n_height, n_width, n_channels]
	y -- placeholder for the input labels, of shape [None, n_y_classes]
	"""

	with tf.name_scope('input'):
		X = tf.placeholder(tf.float32, shape = (None, n_height, n_width, n_channels), 
			name = 'image_input')
		y = tf.placeholder(tf.float32, shape = (None, n_y_classes), name = 'labels_input')

	return X, y

X, y = create_placeholders(train_set_x.shape[1], train_set_x.shape[2], train_set_x.shape[3], 6)

keep_prob = tf.placeholder(tf.float32)

# def forward_propogation(X, keep_prob):
# 	"""
# 	Implements the forward propogation for the model:
# 	CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL 
# 	-> FLATTERN -> FULLYCONNECT1 -> RELU -> FULLYCONNECT2
# 	-> RELU -> SOFTMAX

# 	Arguments:
# 	X -- input dataset placeholder, of shape (batch_size, n_height, n_width, n_channels)

# 	Returns:
# 	prediction -- the output of model calculation
# 	"""

# 	# layer1: convolotion layer with relu and max pool
with tf.name_scope('CONV1D'):
	W_conv1 = tf.get_variable(name = 'W_conv1', shape = [4, 4, 3, 8], 
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b_conv1 = tf.get_variable(name = 'b_conv1', shape = [8],
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	Z_conv1 = tf.nn.conv2d(X, W_conv1, strides = [1, 1, 1, 1], padding = 'SAME', name = 'Z_conv1')\
		+ b_conv1
	A_conv1 = tf.nn.relu(Z_conv1, name = 'A_conv1')
	A_conv1_pool = tf.nn.max_pool(A_conv1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], 
		padding = 'SAME', name = 'A_conv1_pool')

with tf.name_scope('CONV2D'):
	W_conv2 = tf.get_variable(name = 'W_conv2', shape = [2, 2, 8, 16], 
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b_conv2 = tf.get_variable(name = 'b_conv2', shape = [16], 
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	Z_conv2 = tf.nn.conv2d(A_conv1_pool, W_conv2, strides = [1, 1, 1, 1], padding = 'SAME',
		name = 'Z_conv2') + b_conv2
	A_conv2 = tf.nn.relu(Z_conv2, name = 'A_conv2')
	A_conv2_pool = tf.nn.max_pool(A_conv2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1],
		padding = 'SAME', name = 'A_conv2_pool')

A_conv2_flatten = tf.layers.flatten(A_conv2_pool, name = 'A_conv2_flatten')

with tf.name_scope('FULLY_CONNECT1'):
	W_fully1 = tf.get_variable(name = 'W_fully1', shape = (A_conv2_flatten.shape[1], 512),
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b_fully1 = tf.get_variable(name = 'b_fully1', shape = [512], 
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	matmul1 = tf.matmul(A_conv2_flatten, W_fully1)
	Z_fully1 = tf.add(matmul1, b_fully1, name = 'Z_fully1')
	A_fully1 = tf.nn.relu(Z_fully1, name = 'A_fully1')
	A_fully1_prob = tf.nn.dropout(A_fully1, keep_prob, name = 'A_fully1_prob')

with tf.name_scope('FULLY_CONNECT2'):
	W_fully2 = tf.get_variable(name = 'W_fully2', shape = (A_fully1_prob.shape[1], 6),
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	b_fully2 = tf.get_variable(name = 'b_fully2', shape = [6], 
		initializer = tf.contrib.layers.xavier_initializer(seed = 0))
	matmul2 = tf.matmul(A_fully1_prob, W_fully2)
	Z_fully2 = tf.add(matmul2, b_fully2, name = 'Z_fully2')
	prediction = tf.nn.softmax(Z_fully2, name = 'prediction')



# def calculate_cost(prediction, y):
with tf.name_scope('cost'):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))
	# return cross_entropy

# def train(cross_entropy):
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

		
# def test(prediction, y):
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)), tf.float32))
	# return accuracy

if __name__ == '__main__':

	# 准备训练数据集和测试数据集
	

	# 准备网络的运行以及准确率的计算
	
	
	# prediction = forward_propogation(X, keep_prob)
	# cross_entropy = calculate_cost(prediction, y)
	# train_step = train(cross_entropy)
	# accuracy = test(prediction, y)

	print(train_set_y_onehot.max())
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config = config) as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter('logs/', sess.graph)
		for epoch in range(1001):
			_, cost = sess.run([train_step, cross_entropy], feed_dict = {X: train_set_x, y: train_set_y_onehot, keep_prob: 0.7})
			acc = sess.run(accuracy, feed_dict = {X: train_set_x, y: train_set_y_onehot, keep_prob: 1})
			acc1 = sess.run(accuracy, feed_dict = {X: test_set_x, y: test_set_y_onehot, keep_prob: 1})
			print("Iteration %d, accuracy of train set: %f, accuracy of test set %f" %(epoch, acc, acc1))
			# print('cost: ', cost)