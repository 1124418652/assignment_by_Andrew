# -*- coding: utf-8 -*-
import numpy as np 


class NN():

	def initialize_parameters(self, sample_num, feature_num, hidden_layer_nodes):
		np.random.seed(2)
		W1 = np.random.randn(hidden_layer_nodes, feature_num) / sample_num
		b1 = np.zeros((hidden_layer_nodes, 1))
		W2 = np.random.randn(1, hidden_layer_nodes) / sample_num
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
		A1 = np.maximum(Z1, 0)
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
		dZ1 = np.multiply(dA1, np.where(Z1 >= 0, 1, 0))
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

		for i in range(iteration):
			A2, cache = self.forward_propogation(X, parameters)
			cost = self.calculate_cost(A2, y)
			grads = self.backward_propogation(parameters, cache, X, y)
			parameters = self.update_parameters(parameters, grads, learning_rate)

			if print_cost and 0 == i % 10:
				print("cost after iterated of %d : %f" %(i, cost))

		return parameters

	def predict(self, parameters, X):
		prediction, cache = self.forward_propogation(X, parameters)
		prediction = np.where(prediction >= 0.5, 1, 0)
		return prediction

