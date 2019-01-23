# -*- coding: utf-8 -*-
import sklearn
import numpy as np 
import matplotlib.pyplot as plt 
from planar_utils import load_planar_dataset

X, y = load_planar_dataset()

def show_planar_data(X, y):
	plt.scatter(X[0, :], X[1, :], c = y[0], cmap = plt.cm.Spectral)
	plt.xlabel('feature1')
	plt.ylabel('feature2')
	plt.show()

def sigmod(X, w, b):
	assert(X,shape[0] == w.shape[1])
	return 1 / (1 + np.exp(-w * X + b))

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


class NN():
	"""
	Three layer neural network
	"""

	def __init__(self, hidden_size = 4, hidden_type = 'tanh', output_type = 'sigmod'):
		self.n_hidden = hidden_size
		self.hidden_type = hidden_type
		self.output_type = output_type

	def layer_size(self, X, y):
		self.n_input, self.m = X.shape
		self.n_output = y.shape[0]

	def initialize_parameters(self):
		np.random.seed(2)  # 基于种子产生随机数，从而保证每次的结果一致
		W1 = np.random.randn(self.n_hidden, self.n_input) / self.m 
		b1 = np.zeros((self.n_hidden, 1))
		W2 = np.random.randn(self.n_output, self.n_hidden) / self.m 
		b2 = np.zeros((self.n_output, 1))

		parameters = {"W1": W1,
					  "b1": b1,
					  "W2": W2,
					  "b2": b2}

		return parameters

	def _activate(self, Z, activate_func_type = 'sigmod'):
		if 'sigmod' == activate_func_type:
			A = 1 / (1 + np.exp(-Z))
			dA_dZ = np.multiply(A, (1 - A))
		elif 'tanh' == activate_func_type:
			A = np.tanh(Z)
			dA_dZ = 1 - np.multiply(A, A)
		elif 'relu' == activate_func_type:
			A = np.where(Z > 0, Z, 0)
			dA_dZ = np.where(Z >= 0, 1, 0)

		return A, dA_dZ

	def forward_propagation(self, X, parameters):
		W1 = parameters['W1']
		b1 = parameters['b1']
		W2 = parameters['W2']
		b2 = parameters['b2']

		Z1 = np.dot(W1, X) + b1 
		A1, dA_dZ1 = self._activate(Z1, activate_func_type = self.hidden_type)
		Z2 = np.dot(W2, A1) + b2 
		A2, dA_dZ2 = self._activate(Z2, activate_func_type = self.output_type)

		assert(A2.shape == (1, X.shape[1]))

		cache = {'Z1': Z1,
				 'A1': A1,
				 'dA_dZ1': dA_dZ1,
				 'Z2': Z2,
				 'A2': A2,
				 'dA_dZ2': dA_dZ2}

		# print('A1:', A1, '\nA2:', A2)

		return A2, cache

	def calculate_cost(self, A2, y, parameters, lamda):
		assert(A2.shape == y.shape)
		m = y.shape[1]
		W1 = parameters['W1']
		W2 = parameters['W2']

		# print(A2)

		# logprobs = np.multiply(y, np.log(A2 + 10e-8)) + np.multiply((1 - y), np.log(1 - A2))
		logprobs = np.power(A2 - y, 2)
		# print(logprobs)
		cost = np.sum(logprobs) / m\
			   + lamda / (2 * m) *\
			   (np.sum(np.power(W1, 2)) + np.sum(np.power(W2, 2)))
		cost = np.squeeze(cost)
		return cost

	def backward_propagation(self, parameters, cache, X, y):
		m = y.shape[1]
		W1 = parameters['W1']
		W2 = parameters['W2']
		A1 = cache['A1']
		dA_dZ1 = cache['dA_dZ1']
		A2 = cache['A2']
		dA_dZ2 = cache['dA_dZ2']

		if 'sigmod' == self.output_type:
			dZ2 = (A2 - y) / m
		else:
			dZ2 = np.multiply((A2 - y), dA_dZ2) / m

		# print(dZ2)

		dW2 = np.dot(dZ2, A1.T)
		db2 = np.sum(dZ2, axis = 1, keepdims = True)
		dA1 = np.dot(W2.T, dZ2)
		dZ1 = np.multiply(dA1, dA_dZ1)
		dW1 = np.dot(dZ1, X.T)
		db1 = np.sum(dZ1, axis = 1, keepdims = True)

		grads = {'dW1': dW1,
				 'db1': db1,
				 'dW2': dW2,
				 'db2': db2}
		# print(dW2)
		return grads 

	def update_parameters(self, parameters, grads, learning_rate = 0.001, lamda = 0):
		W1 = parameters['W1']
		b1 = parameters['b1']
		W2 = parameters['W2']
		b2 = parameters['b2']

		dW1 = grads['dW1']
		db1 = grads['db1']
		dW2 = grads['dW2']
		db2 = grads['db2']

		W1 = W1 - learning_rate * (dW1 + 1 / self.m * lamda * W1)
		b1 = b1 - learning_rate * db1
		W2 = W2 - learning_rate * (dW2 + 1 / self.m * lamda * W2)
		b2 = b2 - learning_rate * db2 
		# print(dW2 * learning_rate)

		parameters = {'W1': W1,
					  'b1': b1,
					  'W2': W2,
					  'b2': b2}

		return parameters

	def training_model(self, X, y, num_iterations = 1000, lamda = 0.1,\
					   learning_rate = 0.1, print_cost = False):
		np.random.seed(3)
		self.layer_size(X, y)
		parameters = self.initialize_parameters()

		for i in range(num_iterations):
			A2, cache = self.forward_propagation(X, parameters)
			
			cost = self.calculate_cost(A2, y, parameters, lamda)
			grads = self.backward_propagation(parameters, cache, X, y)
			parameters = self.update_parameters(parameters, grads, learning_rate, lamda)

			if print_cost and 0 == i % 10:
				print('Cost after iteration %i : %f' %(i, cost))

		return parameters

	def predict(self, parameters, X):
		prediction, cache = self.forward_propagation(X, parameters)
		prediction = np.where(prediction >= 0.5, 1, 0)
		return prediction


if __name__ == '__main__':

	nn = NN(20, 'tanh', 'relu')
	parameters = nn.training_model(X, y, 1500, 0.1, 0.5, True)

	prediction = nn.predict(parameters, X)

	plot_decision_boundary(lambda x: nn.predict(parameters, x), X, y)
	show_planar_data(X, y)