# -*- coding: utf-8 -*-
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
			v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / ()