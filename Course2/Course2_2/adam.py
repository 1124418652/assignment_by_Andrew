# -*- coding: utf-8 -*-
import math
import numpy as np 
import matplotlib.pyplot as plt
from optimization_nn import DNN, DNN_with_mini_batch
from optimization_nn import plot_decision_boundary, load_dataset


class DNN_with_adam(DNN_with_mini_batch):

	def initialize_adam(self, parameters):

		L = len(parameters) // 2
		v = {}
		s = {}

		for l in range(L):
			v['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
			v['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
			s['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
			s['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)

		return v, s 

	def update_parameters_with_adam(self, parameters, grads, v, 
									s, t, learning_rate, beta1, 
									beta2, epsilon = 1e-8):

		L = len(parameters) // 2
		v_corrected = {}
		s_corrected = {}

		for l in range(L):
			v['dW' + str(l + 1)] = (1 - beta1) * grads['dW' + str(l + 1)] +\
				beta1 * v['dW' + str(l + 1)]
			v['db' + str(l + 1)] = (1 - beta1) * grads['db' + str(l + 1)] +\
				beta1 * v['db' + str(l + 1)]
			s['dW' + str(l + 1)] = (1 - beta2) * (grads['dW' + str(l + 1)] ** 2) +\
				beta2 * s['dW' + str(l + 1)]
			s['db' + str(l + 1)] = (1 - beta2) * (grads['db' + str(l + 1)] ** 2) +\
				beta2 * s['db' + str(l + 1)]
			v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - math.pow(beta1, t))
			v_corrected['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - math.pow(beta1, t))
			s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - math.pow(beta2, t))
			s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - math.pow(beta2, t))

			parameters['W' + str(l + 1)] -= learning_rate * v_corrected['dW' + str(l + 1)] /\
				np.sqrt(s_corrected['dW' + str(l + 1)] + epsilon)
			parameters['b' + str(l + 1)] -= learning_rate * v_corrected['db' + str(l + 1)] /\
				np.sqrt(s_corrected['db' + str(l + 1)] + epsilon)

		return parameters, v, s

	def model_training(self, X, y, layer_dims, activation_list = None, 
					   batch_size = None, iteration = 100, learning_rate = 0.01,
					   beta1 = 0.9, beta2 = 0.999, print_cost = True):

		feature_num, m = X.shape
		assert(X.shape[1] == y.shape[1])
		assert(X.shape[0] == layer_dims[0])
		if not activation_list:
			activation_list = ['relu'] * (len(layer_dims) - 2) + ['sigmod']

		parameters = self.initialize_parameters(layer_dims)
		v, s = self.initialize_adam(parameters)
		costs = []

		if not batch_size:
			for i in range(iteration):
				A, caches = self.L_model_forward(X, parameters, activation_list)
				cost = self.compute_cost(A, y)
				grads = self.L_model_backward(y, parameters, caches, activation_list)
				parameters, v, s = self.update_parameters_with_adam(parameters, grads,
					v, s, i+1, learning_rate, beta1, beta2)

				if print_cost and 0 == i % 100:
					print("Cost after iteration of %d : %f" %(i, cost))

				costs.append(cost)

			return parameters, costs

if __name__ == '__main__':

	train_X, train_y = load_dataset()
	layer_dims = [train_X.shape[0], 10, 7, 1]
	activation_list = ['relu'] * (len(layer_dims) - 2) + ['sigmod']
	nn = DNN_with_adam()
	parameters, costs = nn.model_training(train_X, train_y, layer_dims, activation_list, 
		None, 10000, 0.0007)
	plot_decision_boundary(lambda x: nn.predict(x, parameters, activation_list),
						   train_X, train_y)
	plt.plot(costs)
	plt.show()
