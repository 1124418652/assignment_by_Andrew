# -*- coding: utf-8 -*-
import numpy as np 


class NN():

	def initialize_parameters(self, sample_num, feature_num, hidden_layer_nodes):
		np.random.seed(2)
		W1 = np.random.randn(hidden_layer_nodes, feature_num) * 0.01
		b1 = np.zeros((hidden_layer_nodes, 1))
		W2 = np.random.randn(1, hidden_layer_nodes) * 0.01
		b2 = np.zeros((1, 1))

		parameters = {'W1': W1,
					  'b1': b1,
					  'W2': W2,
					  'b2': b2}

		return parameters

	def forward_propogation(self, X, parameters):
		W1 = parameters['W1']
		b1 = parameters['b1']
		W2 = parameters['W2']
		b2 = parameters['b2']

		assert(X.shape[0] == W1.shape[1])

		Z1 = np.dot(W1, X) + b1
		# A1 = np.maximum(Z1, 0)
		A1 = np.tanh(Z1)
		assert(A1.shape[0] == W2.shape[1])
		Z2 = np.dot(W2, A1) + b2
		A2 = 1 / (1 + np.exp(-Z2))

		cache = {'Z1': Z1,
				 'A1': A1,
				 'Z2': Z2,
				 'A2': A2}

		return A2, cache

	def calculate_cost(self, A, y):
		assert(A.shape == y.shape)
		m = y.shape[1]
		log_prop = np.dot(y, np.log(A).T) + np.dot(1 - y, np.log(1 - A).T)
		cost = np.squeeze(-log_prop / m)
		return cost

	def backward_propogation(self, parameters, cache, X, y):
		W1 = parameters['W1']
		b1 = parameters['b1']
		W2 = parameters['W2']
		b2 = parameters['b2']

		A2 = cache['A2']
		Z2 = cache['Z2']
		A1 = cache['A1']
		Z1 = cache['Z1']

		dZ2 = (A2 - y) / y.shape[1]
		dW2 = np.dot(dZ2, A1.T)
		db2 = np.sum(dZ2, axis = 1, keepdims = True)
		dA1 = np.dot(W2.T, dZ2)
		dZ1 = np.multiply(dA1, 1 - np.power(A1, 2))
		dW1 = np.dot(dZ1, X.T)
		db1 = np.sum(dZ1, axis = 1, keepdims = True)

		grads = {'dW1': dW1,
				 'db1': db1,
				 'dW2': dW2,
				 'db2': db2}

		return grads

	def update_parameters(self, parameters, grads, learning_rate):
		W1 = parameters['W1']
		b1 = parameters['b1']
		W2 = parameters['W2']
		b2 = parameters['b2']

		dW1 = grads['dW1']
		db1 = grads['db1']
		dW2 = grads['dW2']
		db2 = grads['db2']

		W1 -= learning_rate * dW1
		b1 -= learning_rate * db1
		W2 -= learning_rate * dW2
		b2 -= learning_rate * db2

		parameters = {'W1': W1,
					  'b1': b1,
					  'W2': W2,
					  'b2': b2}

		return parameters

	def model_training(self, X, y, hidden_layer_nodes, learning_rate,\
					   iteration = 1000, print_cost = True):
		feature_num, sample_num = X.shape
		parameters = self.initialize_parameters(sample_num, feature_num, hidden_layer_nodes)
		costs = []
		for i in range(iteration):
			A2, cache = self.forward_propogation(X, parameters)
			cost = self.calculate_cost(A2, y)
			grads = self.backward_propogation(parameters, cache, X, y)
			parameters = self.update_parameters(parameters, grads, learning_rate)

			if print_cost and 0 == i % 100:
				print("cost after iterated of %d : %f" %(i, cost))
				costs.append(cost)
		return parameters, costs

	def predict(self, parameters, X):
		prediction, cache = self.forward_propogation(X, parameters)
		prediction = np.where(prediction >= 0.5, 1, 0)
		return prediction


class DNN():
	"""
	network structure: (L-1) * relu + sigmod
	"""

	def initialize_parameters(self, layer_dims):
		"""
		input:
			layer_dims: type of list, 
						(features_num, layer1_nodes, layer2_nodes,..., layerL_nodes)
		return:
			parameters
		"""
		
		np.random.seed(2)
		layer_num = len(layer_dims) - 1
		parameters = {} 
		for l in range(1, layer_num + 1):
			parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1])\
									   * 0.01
			parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

		return parameters

	def activate_forward(self, A_prev, W, b, activation):
		"""
		return:
			A: the output of this layer
			dA_dZ: the value of dA / dZ, which will be used in backward propagation
		"""

		assert(A_prev.shape[0] == W.shape[1])
		assert(W.shape[0] == b.shape[0])
		Z = np.dot(W, A_prev) + b

		if 'relu' == activation:
			A = np.maximum(Z, 0) 
			dA_dZ = np.where(Z >= 0, 1, 0)
		elif 'leaky relu' == activation:
			A = np.maximum(Z, 0.01 * Z)
			dA_dZ = np.where(Z >= 0, 1, 0.01)
		elif 'tanh' == activation:
			A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
			dA_dZ = (1 - np.power(A, 2))
		elif 'sigmod' == activation:
			A = 1 / (1 + np.exp(-Z))
			dA_dZ = np.multiply(A, 1 - A)

		return A, dA_dZ

	def L_model_forward(self, X, parameters, activation_list):
		"""
		input:
			X: the dataset
			parameters: the parameters of this network
						{'W1': W1, 'b1': b1, ..., 'WL': WL, 'bL': bL}
			activation_list: list type, activation function's type

		return:
			AL: the output of this network
			caches: the dictionary contains (Z1, A1, Z2, A2, ..., ZL, AL)
		"""

		layer_num = len(activation_list)
		W = parameters['W1']
		b = parameters['b1']
		activation = activation_list[0]
		A, dA_dZ = self.activate_forward(X, W, b, activation)
		A_prev = A 
		caches = {'A0': X,
				  'A1': A,
				  'dA_dZ1': dA_dZ}

		for l in range(2, layer_num + 1):
			W = parameters['W' + str(l)]
			b = parameters['b' + str(l)]
			activation = activation_list[l - 1]
			A, dA_dZ = self.activate_forward(A_prev, W, b, activation)
			caches['A' + str(l)] = A
			caches['dA_dZ' + str(l)] = dA_dZ
			A_prev = A 

		return A, caches

	def compute_cost(self, AL, y):
		"""
		return:
			cost
		"""

		m = AL.shape[1]
		assert(AL.shape == y.shape)
		logprops = np.dot(y, np.log(AL.T)) + np.dot(1 - y, np.log(1 - AL).T)
		cost = -np.squeeze(logprops / m)
		return cost

	def L_model_backward(self, y, parameters, caches, activation_list):
		"""
		input:
			y: the label of this dataset, dimemsions is (1, sample_num)
			caches: the caches from L_model_forward
		return:
			grads
		"""

		m = y.shape[1]
		layer_num = len(activation_list)
		A = caches['A' + str(layer_num)]
		A_prev = caches['A' + str(layer_num - 1)]
		W = parameters['W' + str(layer_num)]

		if 'sigmod' == activation_list[-1]:    # calculate the last layer's backword propagation
			dZ = (A - y) / m 
		else:
			dA_dZ = caches['dA_dZ' + str(layer_num)]
			dZ = np.multiply(A - y, dA_dZ) / m 

		dA = np.dot(W.T, dZ)
		dW = np.dot(dZ, A_prev.T)
		db = np.sum(dZ, axis = 1, keepdims = True)

		grads = {'dW' + str(layer_num): dW,
				 'db' + str(layer_num): db}

		for l in range(layer_num - 1, 0, -1):

			A_prev = caches['A' + str(l - 1)]
			dA_dZ = caches['dA_dZ' + str(l)]
			W = parameters['W' + str(l)]
			b = parameters['b' + str(l)]

			dZ = np.multiply(dA, dA_dZ)
			dA_prev = np.dot(W.T, dZ)
			dW = np.dot(dZ, A_prev.T)
			db = np.sum(dZ, axis = 1, keepdims = True)

			grads['dW' + str(l)] = dW 
			grads['db' + str(l)] = db 

			dA = dA_prev

		return grads

	def update_parameters(self, parameters, grads, learning_rate):
		"""
		return:
			parameters
		"""
		
		layer_num = int(len(parameters) // 2)
		for l in range(1, layer_num + 1):
			parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
			parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

		return parameters

	def model_training(self, X, y, layer_dims, activation_list = None, iteration = 100, 
					   learning_rate = 0.01, print_cost = True):
		"""
		initialize_parameters >> for every iteration:
		L_model_forward -> compute_cost
		-> L_model_backward -> update_parameters

		return:
			parameteres
		"""
		
		feature_num, m = X.shape
		assert(X.shape[1] == y.shape[1])
		assert(X.shape[0] == layer_dims[0])
		if not activation_list:
			activation_list = ['relu'] * (len(layer_dims) - 2) + ['sigmod']

		assert(len(activation_list) == len(layer_dims) - 1)
		parameters = self.initialize_parameters(layer_dims)
		costs = []

		for i in range(iteration):
			A, caches = self.L_model_forward(X, parameters, activation_list)
			cost = self.compute_cost(A, y)
			grads = self.L_model_backward(y, parameters, caches, activation_list)
			parameters = self.update_parameters(parameters, grads, learning_rate)

			if print_cost and 0 == i % 100:
				print("Cost after iterations of %d : %f" %(i, cost))

				costs.append(cost)

		return parameters, costs

	def predict(self, X, parameters, activation_list):

		layer_num = len(parameters) // 2
		A, caches = self.L_model_forward(X, parameters, activation_list)
		prediction = np.where(A >= 0.5, 1, 0)
		return prediction


class Improve_DNN(DNN):

	def initialize_parameters_zeros(self, layer_dims):

		layer_num = len(layer_dims) - 1
		parameters = {}

		for l in range(1, layer_num + 1):
			parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
			parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

		return parameters

	def initalize_parameters_random(self, layer_dims):

		layer_num = len(layer_dims) - 1
		parameters = {}
		np.random.seed(3)

		for l in range(1, layer_num + 1):
			parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 10 
			parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

		return parameters

	def initialize_parameters_he(self, layer_dims):

		layer_num = len(layer_dims) - 1
		parameters = {}
		np.random.seed(3)

		for l in range(1, layer_num + 1):
			parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1])\
									   * np.sqrt(2. / layer_dims[l - 1])
			parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

		return parameters

	def model_training(self, X, y, layer_dims, activation_list = None, 
					   iteration = 100, init_type = 'he', learning_rate = 0.01, 
					   print_cost = True):

		feature_num, m = X.shape
		assert(X.shape[1] == y.shape[1])
		assert(X.shape[0] == layer_dims[0])

		if not activation_list:
			activation_list = ['relu'] * (len(layer_dims) - 2) + ['sigmod']

		if 'zeros' == init_type:
			parameters = self.initialize_parameters_zeros(layer_dims)
		elif 'random' == init_type:
			parameters = self.initalize_parameters_random(layer_dims)
		elif 'he' == init_type:
			parameters = self.initialize_parameters_he(layer_dims)

		costs = []
		for i in range(iteration):
			A, caches = self.L_model_forward(X, parameters, activation_list)
			cost = self.compute_cost(A, y)
			grads = self.L_model_backward(y, parameters, caches, activation_list)
			parameters = self.update_parameters(parameters, grads, learning_rate)

			if print_cost and 0 == i % 100:
				print("Cost after iterations of %d : %f" %(i, cost))

				costs.append(cost)

		return parameters, costs