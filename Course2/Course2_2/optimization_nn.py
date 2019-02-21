#-*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt 


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


class DNN_with_mini_batch(DNN):

	def random_mini_batches(self, X, y, mini_batch_size = 64, seed = 0):
		
		np.random.seed(seed)
		m = X.shape[1]
		mini_batches = []
		permutation = list(np.random.permutation(m))
		shuffled_X = X[:, permutation]
		shuffled_y = y[:, permutation].reshape((1, m))
		num_complete_minibatches = m // mini_batch_size

		for k in range(0, num_complete_minibatches):
			mini_batch_X = shuffled_X[:, k * mini_batch_size: 
										 (k + 1) * mini_batch_size]
			mini_batch_y = shuffled_y[:, k * mini_batch_size:
										 (k + 1) * mini_batch_size]
			mini_batches.append((mini_batch_X, mini_batch_y))

		if m % mini_batch_size != 0:
			mini_batch_X = shuffled_X[:, num_complete_minibatches * \
										 mini_batch_size: ]
			mini_batch_y = shuffled_y[:, num_complete_minibatches * \
										 mini_batch_size: ]
			mini_batches.append((mini_batch_X, mini_batch_y))

		return mini_batches

	def model_training(self, X, y, layer_dims, activation_list = None, \
					   batch_size = None, seed = 0, iteration = 100, \
					   learning_rate = 0.01, print_cost = True):

		feature_num, m = X.shape
		assert(X.shape[1] == y.shape[1])
		assert(X.shape[0] == layer_dims[0])
		if not activation_list:
			activation_list = ['relu'] * (len(layer_dims) - 2) + ['sigmod']

		parameters = self.initialize_parameters(layer_dims)
		costs = []

		if not batch_size:
			for i in range(iteration):
				A, caches = self.L_model_forward(X, parameters, activation_list)
				cost = self.compute_cost(A, y)
				grads = self.L_model_backward(y, parameters, caches, activation_list)
				parameters = self.update_parameters(parameters, grads, learning_rate)

				if print_cost and 0 == i % 100:
					print("Cost after iterations of %d : %f" %(i, cost))

				costs.append(cost)

			return parameters, costs

		else:
			costs = []
			for i in range(iteration):
				mini_batches = self.random_mini_batches(X, y, batch_size, seed)
				# print(mini_batches)
				for (mini_batch_X, mini_batch_y) in mini_batches:
					# print("i = ", i)
					# print(mini_batch_X)
					A, caches = self.L_model_forward(mini_batch_X, parameters, activation_list)
					cost = self.compute_cost(A, mini_batch_y)
					grads = self.L_model_backward(mini_batch_y, parameters, caches, activation_list)
					parameters = self.update_parameters(parameters, grads, learning_rate)

				if print_cost and 0 == i % 100:
					print("Cost after iteration of %d : %f" %(i, cost))

				costs.append(cost)

			return parameters, costs


def plot_decision_boundary(model, X, y):

	x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
	y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
	h = 0.01
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = model(np.c_[xx.ravel(), yy.ravel()].T)
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral)
	plt.ylabel('x2')
	plt.xlabel('x1')
	plt.scatter(X[0, :], X[1, :], c = y[0], cmap = plt.cm.Spectral, alpha = 0.8,
				linewidth = 1, edgecolors = (0, 0, 0))
	plt.show()

def load_dataset(is_plot = True):

	np.random.seed(3)
	import sklearn
	import sklearn.datasets
	train_X, train_y = sklearn.datasets.make_moons(n_samples = 300, noise = .2)
	if is_plot:
		plt.scatter(train_X[:, 0], train_X[:, 1], c = train_y, s = 40, cmap = plt.cm.Spectral)
	train_X = train_X.T
	train_y = train_y.reshape((1, train_y.shape[0]))

	return train_X, train_y


if __name__ == '__main__':

	train_X, train_y = load_dataset()
	layer_dims = [train_X.shape[0], 10, 7, 1]
	activation_list = ['relu'] * (len(layer_dims) - 2) + ['sigmod']
	dnn = DNN_with_mini_batch()
	parameters, costs = dnn.model_training(train_X, train_y, layer_dims, iteration = 50000,
					   learning_rate = 0.01, batch_size = 64)
	plot_decision_boundary(lambda x: dnn.predict(x, parameters, activation_list),
						   train_X, train_y)
	plt.plot(costs)
	plt.show()

	# mini_batches = dnn.random_mini_batches(train_X, train_y)
	# for (mini_batch_X, mini_batch_y) in mini_batches:
	# 	print(mini_batch_X, mini_batch_y)