# -*- coding: utf-8 -*-
import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image 


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

	show_image(train_set)