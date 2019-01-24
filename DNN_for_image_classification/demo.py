# -*- coding: utf-8 -*-
import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image 
from dnn_units import NN
import matplotlib.pyplot as plt

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

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

def load_data(file_path, type = 'training'):
	if not os.path.exists(file_path):
		return

	with h5py.File(file_path, 'r') as fr:
		if 'training' == type:
			train_set = list(fr['train_set_x'])
			train_label = list(fr['train_set_y'])
			return train_set, train_label

		elif 'testing' == type:
			test_set = list(fr['test_set_x'])
			test_label = list(fr['test_set_y'])
			return test_set, test_label

def show_image(img, row = 10, col = 10):
	if len(img.shape) > 3:
		img = img[: row * col]
		img = np.concatenate(img, axis = 0).reshape((row, col, 64, 64, 3))
		img = np.transpose(img, (0, 2, 1, 3, 4)).reshape(row * 64, col * 64, 3)
		plt.imshow(img)
		plt.show()
	else:
		plt.imshow(img)
		plt.show()


if __name__ == '__main__':

	training_path = 'datasets/train_catvnoncat.h5'
	testing_path = 'datasets/test_catvnoncat.h5'
	train_set, train_label = load_data(training_path)
	test_set, test_label = load_data(testing_path, 'testing')

	train_set = np.array(train_set)
	test_set = np.array(test_set)
	train_label = np.array(train_label)
	test_label = np.array(test_label)


	# show_image(train_set)
	# print(train_label.reshape((209,1)).T.shape)
	# nn = NN()
	# nn.model_training(train_set.reshape((209, -1)).T / 255 ,\
	# 				  train_label.reshape((209, 1)).T, 10, 0.1)

	X, y = load_planar_dataset()
	nn = NN()
	parameters = nn.model_training(X, y, 20, 0.5, 10000)
	plot_decision_boundary(lambda x: nn.predict(parameters, x), X, y)
