#-*- coding: utf-8 -*-
import os
import h5py
import numpy as np 


__all__ = ['load_dataset', 'random_mini_batches', 'convert_to_onehot']

def load_dataset():
	train_file_path = 'datasets/train_signs.h5'
	test_file_path = 'datasets/test_signs.h5'
	train_file = h5py.File(train_file_path, 'r')
	test_file = h5py.File(test_file_path, 'r')
	train_set_x = np.array(list(train_file['train_set_x'])) / 255.0
	train_set_y = np.array(list(train_file['train_set_y']))
	test_set_x = np.array(list(test_file['test_set_x'])) / 255.0
	test_set_y = np.array(list(test_file['test_set_y']))

	return train_set_x, train_set_y, test_set_x, test_set_y

def random_mini_batches(train_set_x, train_set_y, batch_size = 128, seed = 0):
	m = train_set_x.shape[0]
	mini_batches = []
	np.random.seed(seed)
	permutation = np.random.permutation(m).tolist()
	shuffled_X = train_set_x[permutation, :, :, :]
	shuffled_y = train_set_y[permutation, :, :, :]
	num_complete_minibatches = m // batch_size
	for k in range(num_complete_minibatches):
		mini_batch_X = shuffled_X[k * batch_size: (k+1) * batch_size, :, :, :]
		mini_batch_y = shuffled_y[k * batch_size: (k+1) * batch_size, :]
		mini_batches.append((mini_batch_X, mini_batch_y))
	if m % batch_size != 0:
		mini_batch_X = shuffled_X[num_complete_minibatches * batch_size: m, :, :, :]
		mini_batch_y = shuffled_y[num_complete_minibatches * batch_size: m, :]
		mini_batches.append((mini_batch_X, mini_batch_y))
	return mini_batches

def convert_to_onehot(y, classes = 6):
	Y = np.eye(classes)[y.reshape(-1)]
	return Y


if __name__ == '__main__':
	load_dataset()