# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
from optimization_nn import DNN, DNN_with_mini_batch
from optimization_nn import plot_decision_boundary, load_dataset


class DNN_with_momentum(DNN_with_mini_batch):

	def initialize_velocity(self, parameters):

		L = len(parameters) // 2
		v = {}
		for l in range(L):
			v['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
			v['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)

		return v

	def update_parameters_with_momentum(self, parameters, grads, v, beta,
										learning_rate):

		L = len(parameters) // 2
		for l in range(L):
			v['dW' + str(l + 1)] = beta * grads['dW' + str(l + 1)] + (1 - beta)\
								   * v['dW' + str(l + 1)]
			v['db' + str(l + 1)] = beta * grads['db' + str(l + 1)] + (1 - beta)\
								   * v['db' + str(l + 1)]
			parameters['W' + str(l + 1)] -= learning_rate * v['dW' + str(l + 1)]
			parameters['b' + str(l + 1)] -= learning_rate * v['db' + str(l + 1)] 

		return parameters, v 

	def model_training(self, X, y, layer_dims, activation_list = None, 
					   iteration = 100, learning_rate = 0.01, beta = 0.1,
					   batch_size = None, print_cost = True):

		feature_num, m = X.shape
		assert(X.shape[1] == y.shape[1])
		assert(X.shape[0] == layer_dims[0])

		if not activation_list:
			activation_list = ['relu'] * (len(layer_dims) - 2) + ['sigmod']

		parameters = self.initialize_parameters(layer_dims)
		v = self.initialize_velocity(parameters)
		costs = []

		if not batch_size:
			for i in range(iteration):
				A, caches = self.L_model_forward(X, parameters, activation_list)
				cost = self.compute_cost(A, y)
				grads = self.L_model_backward(y, parameters, caches, activation_list)
				parameters, v = self.update_parameters_with_momentum(parameters, 
					grads, v, beta, learning_rate)

				if print_cost and 0 == i % 100:
					print("Cost after iteration of %d : %f" %(i, cost))

				costs.append(cost)

			return parameters, costs


if __name__ == '__main__':
	
	train_X, train_y = load_dataset()
	layer_dims = [train_X.shape[0], 10, 7, 1]
	activation_list = ['relu'] * (len(layer_dims) - 2) + ['sigmod']
	nn = DNN_with_momentum()
	parameters, costs = nn.model_training(train_X, train_y, layer_dims, activation_list,
										  10000, 0.1, beta = 0.1)
	plot_decision_boundary(lambda x: nn.predict(x, parameters, activation_list),
						   train_X, train_y)
	plt.plot(costs)
	plt.show()
