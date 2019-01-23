# -*- coding: utf-8 -*-
import h5py
import numpy as np 
import matplotlib.pyplot as plt 


class DNN():
	"""
	Deep learning neural network
	"""

	def __init__(self, num_layers, layer_type_list = None, node_num_list = None):
		self.num_layers = num_layers
		self.layer_type_list = layer_type_list
		self.node_num_list  = node_num_list

	def initial_parameters_deep(self, sample_num, 
									  data_features):
		self.data_features = data_features
		assert(self.num_layers == len(self.layer_type_list))
		assert(self.num_layers == len(self.node_num_list))
		self.m = sample_num

		np.random.seed(3)
		parameters = {}
		parameters['W1'] = np.random.randn(self.node_num_list[0], 
										   self.data_features) / np.sqrt(self.m)
		parameters['b1'] = np.zeros((self.node_num_list[0], 1))
		for l in range(1, self.num_layers):
			node_num_before = self.node_num_list[l - 1]
			node_num = self.node_num_list[l]
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
		# assert(X.shape == (self.data_features, self.m))
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

	def calculate_cost(self, A, y, activation_type):
		assert(A.shape[1] == y.shape[1])
		if 'sigmod' == activation_type:
			log_prop = np.dot(y, np.log(A).T) + np.dot(1 - y, np.log(1 - A).T)
			# log_prop = np.sum(np.multiply(y, np.log(A)) + np.multiply(1 - y, np.log(1 - A)))
			cost = -np.squeeze(log_prop) / self.m
		else:
			cost = np.sum(np.power(A - y, 2)) / (2 * self.m)

		return cost 

	def linear_backward(self, dZ, A_prev, W):
		assert(A_prev.shape[0] == W.shape[1])
		dA_prev = np.dot(W.T, dZ)
		dW = np.dot(dZ, A_prev.T)
		db = np.sum(dZ, axis = 1, keepdims = True)

		return dA_prev, dW, db

	def activate_backward(self, dA, dA_dZ):
		assert(dA.shape == dA_dZ.shape)
		dZ = np.multiply(dA, dA_dZ)
		return dZ

	def update_parameters(self, parameters, grads, learning_rate, lamda):
		
		for l in range(1, self.num_layers + 1):
			W = parameters['W' + str(l)]
			b = parameters['b' + str(l)]
			dW = grads['dW' + str(l)]
			db = grads['db' + str(l)]

			W -= learning_rate * (dW + lamda * W)
			b -= learning_rate * db
			# print(dW)
			parameters['W' + str(l)] = W 
			parameters['b' + str(l)] = b

		return parameters

	def backward_propogation(self, A, y, cache_list):
		cache = cache_list['cache' + str(self.num_layers)]
		A_prev = cache['A_prev']
		activation_type = cache['activation_type']
		W = cache['W']
		dA_dZ = cache['dA_dZ']

		# the cost function will be changed turns to the activation_function of last layer
		if 'sigmod' == cache['activation_type']:     
			dZ = (A - y) / self.m 
		else:
			dZ = np.multiply(A - y, dA_dZ) / self.m

		dA, dW, db = self.linear_backward(dZ, A_prev, W)    # it's the dA of this layer's previous layer
		grads = {'dW' + str(self.num_layers): dW,
				 'db' + str(self.num_layers): db}

		for l in range(self.num_layers - 1, 0, -1):
			cache = cache_list['cache' + str(l)]
			A_prev = cache['A_prev']
			dA_dZ = cache['dA_dZ']
			W = cache['W']
			dZ = self.activate_backward(dA, dA_dZ)
			dA_prev, dW, db = self.linear_backward(dZ, A_prev, W)
			dA = dA_prev

			grads['dW' + str(l)] = dW
			grads['db' + str(l)] = db

		return grads

	def model_training(self, X, y,
					   num_iterations = 10000, 
					   learning_rate = 0.1, 
					   lamda = 0.01,
					   print_cost = False):
		"""
		training struct for every iterations: 
			forward_propogation -> calculate_cost
			-> backward_propogation -> update_parameters
		"""
		data_features, sample_num = X.shape
		parameters = self.initial_parameters_deep(sample_num, data_features)
		for i in range(num_iterations):
			A, forward_cache = self.forward_propogation(X, parameters)
			A_list = forward_cache['A_list']
			cache_list = forward_cache['cache_list']
			cost = self.calculate_cost(A, y, self.layer_type_list[-1])
			grads = self.backward_propogation(A, y, cache_list)
			parameters = self.update_parameters(parameters, grads, learning_rate, lamda)

			if True == print_cost and 0 == i % 10:
				print("the cost after iterate %d: %f" %(i, cost))

		return parameters

	def predict(self, parameters, X):
		num_layers = int(len(parameters) / 2)
		A, forward_cache = self.forward_propogation(X, parameters)
		prediction = np.where(A >=0.5, 1, 0)
		return prediction


def plot_decision_boundary(model, X, y):
	x_min, x_max = X[0, :].min() - 1, X[1, :].max() + 1
	y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
	h = 0.01
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = model(np.c_[xx.ravel(), yy.ravel()].T)
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral)
	plt.xlabel("feature1")
	plt.ylabel("feature2")
	plt.scatter(X[0, :], X[1, :], c = y[0],
				linewidth = 1, edgecolors = (0, 0, 0), 
				cmap = plt.cm.Spectral, alpha = 0.8)
	plt.show()

if __name__ == '__main__':
	from planar_utils import load_planar_dataset
	X, y = load_planar_dataset()
	nn = DNN(3, ['relu','relu', 'relu'], [20, 10, 1])
	parameters = nn.model_training(X, y, print_cost = True)

	plot_decision_boundary(lambda x: nn.predict(parameters, x), X, y)