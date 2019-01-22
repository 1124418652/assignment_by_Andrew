# -*- coding: utf-8 -*-
import h5py
import numpy as np 
import matplotlib.pyplot as plt 


class DNN():
	"""
	Deep learning neural network
	"""

	def __init__(self, data_features, num_layers):
		self.data_features = data_features
		self.num_layers = num_layers

	def initial_parameters_deep(self, sample_num, 
									  layer_type_list, 
									  node_num_list):
		assert(self.num_layers == len(layer_type_list))
		assert(self.num_layers == len(node_num_list))
		self.m = sample_num
		self.layer_type_list = layer_type_list
		self.node_num_list = node_num_list

		np.random.seed(3)
		parameters = {}
		parameters['W1'] = np.random.randn(node_num_list[0], 
										   self.data_features) / np.sqrt(self.m)
		parameters['b1'] = np.zeros((node_num_list[0], 1))
		for l in range(1, self.num_layers):
			node_num_before = node_num_list[l - 1]
			node_num = node_num_list[l]
			parameters['W' + str(l + 1)] = np.random.randn(node_num, node_num_before)\
										/ np.sqrt(node_num_before)
			parameters['b' + str(l + 1)] = np.zeros((node_num, 1))

		return parameters

	def linear_forward(self, A_prev, W, b):
		Z = np.dot(W, A_prev) + b 
		assert(Z.shape == (W.shape[0], A_prev.shape[1]))
		return Z 

	def activate_forward(self, A_prev, W, b, activation_type):
		assert(A_prev.shape[0] == W.shape[1])
		Z = self.linear_forward(A_prev, W, b)
		
		if 'sigmod' == activation_type:
			A = 1 / (1 + np.exp(-Z))
			dA_dZ = np.multiply(A, 1 - A)
		
		elif 'relu' == activation_type:
			A = np.maximum(Z, 0)
			dA_dZ = np.where(Z >= 0, 1, 0)

		elif 'leaky relu' == activation_type:
			A = np.maximum(Z, 0.01 * Z)
			dA_dZ = np.where(Z >=0, 1, 0.01)

		elif 'tanh' == activation_type:
			A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
			dA_dZ = 1 - np.power(A, 2)

		else:
			return

		assert(A.shape == dA_dZ.shape)
		cache = {'A_prev': A_prev,
				 'W': W,
				 'activation_type': activation_type,
				 'dA_dZ': dA_dZ}

		return A, cache

	def forward_propogation(self, X, parameters):
		assert(x.shape == (self.data_features, self.m))
		A_prev = X
		A_list = {}       # the list of array A from forward propogation
		cache_list = {}   # the list of cache from forward propogation
		A_list['A0'] = A_prev

		for l in range(1, self.num_layers + 1):
			W = parameters['W' + str(l)]
			b = parameters['b' + str(l)]
			activation_type = self.layer_type_list[l - 1]
			A, cache = self.activate_forward(A_prev, W, b, activation_type)
			A_list['A' + str(l)] = A 
			cache_list['cache' + str(l)] = cache
			A_prev = A 

		forward_cache = {'A_list': A_list, 'cache_list': cache_list}
		return A, forward_cache

	def calculate_cost(self, A, y):
		assert(A.shape[1] == len(y))
		log_prop = np.dot(y, np.log(A).T) + np.dot(1 - y, np.log(1 - A).T) / self.m
		cost = -np.squeeze(log_prop)
		return cost 

	def linear_backward(self, dZ, cache):
		pass

	def activate_backward(self):
		pass

	def update_parameters(self):
		pass

	def backward_propogation(self):
		pass

	def model_training(self, X, y, num_iterations = 100, 
					   learning_rate = 0.1, 
					   lamda = 0.1,
					   print_cost = False):
		"""
		training struct for every iterations: 
			forward_propogation -> calculate_cost
			-> backward_propogation -> update_parameters
		"""
		sample_num = X.shape[1]
		parameters = self.initial_parameters_deep(sample_num, layer_type_list, node_num_list)

	def predict(self):
		pass


if __name__ == '__main__':
	nn = DNN(10, 3)
	parameters = nn.initial_parameters_deep(sample_num = 1000, node_num_list = [20, 10, 1], layer_type_list = 
		['relu', 'relu', 'sigmod'])
	print(parameters)