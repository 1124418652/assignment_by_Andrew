# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
from planar_utils import load_planar_dataset

def layer_sizes(X, y):

	n_x = X.shape[0]
	n_h = 4 
	n_y = y.shape[0]

	return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
	
	np.random.seed(2)
	W1 = np.random.randn(n_h, n_x) * 0.01
	b1 = np.zeros((n_h, 1))
	W2 = np.random.randn(n_y, n_h) * 0.01
	b2 = np.zeros((n_y, 1))

	parameters = {'W1': W1,
				  'b1': b1,
				  'W2': W2,
				  'b2': b2}

	return parameters

def forward_propagation(X, parameters):

	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']

	Z1 = np.dot(W1, X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = 1 / (1 + np.exp(-Z2))

	cache = {'Z1': Z1,
			 'A1': A1,
			 'Z2': Z2,
			 'A2': A2}

	return A2, cache

def compute_cost(A2, y, parameters):
	
	m = y.shape[1]
	logprops = np.multiply(np.log(A2), y) + np.multiply(np.log(1 - A2), 1 - y)
	cost = -np.sum(logprops) / m
	cost = np.squeeze(cost)

	return cost

def backward_propagation(parameters, cache, X, y):

	m = X.shape[1]
	W1 = parameters['W1']
	W2 = parameters['W2']

	A1 = cache['A1']
	A2 = cache['A2']

	dZ2 = A2 - y
	dW2 = np.dot(dZ2, A1.T) / m
	db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
	dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
	dW1 = np.dot(dZ1, X.T) / m
	db1 = np.sum(dZ1, axis = 1, keepdims = True) / m 

	grads = {'dW1': dW1,
			 'db1': db1,
			 'dW2': dW2,
			 'db2': db2}

	return grads

def update_parameters(parameters, grads, learning_rate = 1.2):

	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']

	dW1 = grads['dW1']
	db1 = grads['db1']
	dW2 = grads['dW2']
	db2 = grads['db2']

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	parameters = {'W1': W1,
				  'b1': b1,
				  'W2': W2,
				  'b2': b2}

	return parameters

def nn_model(X, y, n_h, num_iterations = 10000, print_cost = True):

	np.random.seed(3)
	n_x = layer_sizes(X, y)[0]
	n_y = layer_sizes(X, y)[2]

	parameters = initialize_parameters(n_x, n_h, n_y)
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']

	for i in range(0, num_iterations):
		A2, cache = forward_propagation(X, parameters)
		cost = compute_cost(A2, y, parameters)
		grads = backward_propagation(parameters, cache, X, y)
		parameters = update_parameters(parameters, grads)

		if print_cost and i % 10 == 0:
			print("Cost after iteration %i: %f" %(i, cost))

	return parameters

def predict(parameters, X):

	A2, cache = forward_propagation(X, parameters)
	prediction = np.round(A2)

	return prediction

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0], cmap=plt.cm.Spectral)
    plt.show()



if __name__ == '__main__':

	X, y = load_planar_dataset()
	parameters = nn_model(X, y, 50)
	prediction = predict(parameters, X)
	plot_decision_boundary(lambda x: predict(parameters, x), X, y)