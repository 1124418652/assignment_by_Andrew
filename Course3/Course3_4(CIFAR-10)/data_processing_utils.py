# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import struct
import argparse
import tarfile
import scipy
import tensorflow as tf

from six.moves import urllib


parser = argparse.ArgumentParser(description = 'Prepare the data for nn training')
parser.add_argument('--batch_size', type = int, default = 100,
					help = "Number of images to process in a batch")
parser.add_argument('--data_dir', type = str, 
					help = "Path to the CIFAR-10 data directory")


DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


args = parser.parse_args()

if args.batch_size:
	batch_size = args.batch_size
else:
	batch_size = 128

if args.data_dir:
	data_dir = args.data_dir
else:
	data_dir = os.path.join('.', 'dataset')


def maybe_download_and_extract():
	"""
	Download and extract the tarball from Alex's website.
	"""

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)
	
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(data_dir, filename)

	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			# 定义一个回调函数用于在 urlretrieve 中打印信息
			sys.stdout.write('>> Downloading %s %.1f%%\n' % (filename,
				count * block_size / total_size * 100.0))
			sys.stdout.flush()

		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
		statinfo = os.stat(filepath)
		print('Successfully download ', filename, statinfo.st_size, ' bytes.\n')

	extracted_dir_path = os.path.join(data_dir, 'cifar-10-batches-bin')
	if not os.path.exists(extracted_dir_path):
		tarfile.open(filepath, 'r:gz').extractall(data_dir)


def load_data_from_binary_file(filepath = os.path.join(data_dir, 'cifar-10-batches-bin', 'data_batch_1.bin'), 
							   img_num = 10000, 
							   img_size = 3072, 
							   label_size = 1):

	if not os.path.exists(filepath):
		raise ValueError("Can't find file %s" % (file))

	datasets = []
	labels = []
	format_str = '>{0}B'.format(img_size + label_size)
	with open(filepath, 'rb') as fr:
		buffer = fr.read()
		for i in range(img_num):
			offset = struct.calcsize(format_str) * i  # begin from 0
			tmp = struct.unpack_from(format_str, buffer, offset)
			labels.append(tmp[0])
			datasets.append(tmp[1:])

	return datasets, labels


def convert_one_hot(y, num_classes):

	y_one_hot = np.zeros((y.shape[-1], num_classes))
	for i in range(y.shape[-1]):
		y_one_hot[i][y[i]] = 1 
	return y_one_hot


def read_cifar10(filename_queue):
	"""
	Reads and parses examples from CIFAR10 data files.

	Recommendation: if you want N-way read parallelism call this function
	N times. This will give you N independent Readers reading different 
	files & positions within those files, which will give better mixing of 
	examples.

	Args:
	filename_queue: A queue of strings with the filenames to read from.
	
	Returns:
	An object representing a single example, with the following fields:
	height: number of rows in the result (32)
	width: number of columns in the result (32)
	depth: number of color channels in the result (3)
	key: a scalar string Tensor describing the filename & record number
		 for this example.
	label: an int32 Tensor with the label in the range 0..9
	uint8image: a [height, width, depth] uint8 Tensor with the image data
	"""

	class CIFAR10Record(object):
		pass

	result = CIFAR10Record()

	label_bytes = 1 # 2 for CIFAR-100
	
	result.height = 32
	result.width = 32
	result.depth = 3

	# 1 bytes for one pixel(0~255, 8bits)
	image_bytes = result.height * result.width * result.depth
	record_bytes = label_bytes + image_bytes

	# 读取固定长度字节数信息(针对bin文件使用FixedLengthRecordReader读取比较合适)
	# 在下次读取时会自动从当前位置往后读,而不会从头开始
	reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
	result.key, value = reader.read(filename_queue)

	# convert from a binary string to a vector of uint8 that is record_bytes long.
	record_bytes = tf.decode_raw(value, tf.uint8)

	# get the label(the first byte in vector Tensor record_bytes) by funtion
	# tf.stride_slice(Tensor, begin, end) and convert the data type from 
	# tf.uint8 to tf.int32
	result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), 
		tf.int32)

	# the remaining bytes after the label represent the image, which we
	# reshape from [depth * height * width] to [depth, height, width].
	depth_major = tf.reshape(
		tf.strided_slice(record_bytes, [label_bytes], 
						[label_bytes + image_bytes]),
		[result.depth, result.height, result.width])

	result.uint8image = tf.transpose(depth_major, [1, 2, 0])

	return result


def _generate_image_and_label_batch(image, label, min_queue_examples, 
									batch_size, shuffle):
	"""
	Construct a queue batch of images and labels

	Parameters:
	image: 3-D Tensor of [height, width, 3] of type.float32
	label: 1-D Tensor of type.int32
	min_queue_examples: int32, minimum number of samples to retain in 
						the queue that provides of batches of examples.
	batch_size: Number of images per batch.
	shuffle: boolean indicating whether to use a shuffling queue

	Returns:
	images: Images. 4D tensor of [batch_size, height, width, 3] size.
	labels: Labels. 1D tensor of [batch_size] size.						
	"""

	# create a queue that shuffles the examples, and then read 'batch_size'
	# images + labels from the example queue.
	num_preprocess_threads = 16
	if shuffles:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size = batch_size,
			num_threads = num_preprocess_threads,
			capacity = min_queue_examples + 3 * batch_size, # the maximum number of elements in the queue
			min_after_dequeue = min_queue_examples)   # dequeue after there are min_queue_examples
	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size = batch_size,
			num_threads = num_preprocess_threads,
			capacity = min_queue_examples + 3 * batch_size)

	# display the training images in the visualizer
	tf.summary.image('images', images)
	return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
	"""
	Construct distorted input for CIFAR training using the Reader ops.
	use the methods which were metioned in Alex's paper
	
	Parameters:
	data_dir: Path to the CIRAR-10 data directory
	batch_size: Number of images per batch.

	Returns:
	images: Images. 4D tensor of [batch_size, height, width, 3]
	labels: Labels. 1D tensor of [batch_size].
	"""

	filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %i)
				 for i in range(1, 6)]
	for f in filenames:
		if not os.path.exists(f):
			raise ValueError('Failed to find file: ' + f)
	
	# create a queue that produces the filenames to read
	filename_queue = tf.train.string_input_producer(filenames)		

	# read examples from files in the filename queue.
	read_input = read_cifar10(filename_queue)
	reshape_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Randomly crop a [height, width] section of the image
	distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
	# Randomly flip the image horizontally
	distorted_image = tf.image.random_flip_left_right(distorted_image)
	distorted_image = tf.image.random_brightness(distorted_image,
												 max_delta = 63)
	distorted_image = tf.image.random_contrast(distorted_image, 
											   lower = 0.2, upper = 1.8)
	# Subtract off the mean and divide by the variance of the pixels
	float_image = tf.image.per_image_standardization(distorted_image)
	# set the shapes of tensors
	float_image.set_shape([height, width, 3])
	read_input.label.set_shape([1])

	# Ensure that the random shuffling has good mixing properties
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 
							 min_fraction_of_examples_in_queue)

	print("""Filling queue with %d CIFAR images before starting to train.
			 This will take a few minutes."""%min_queue_examples)

	# Generate a batch of images and labels by building up a queue of examples
	return _generate_image_and_label_batch(float_image, read_input.label, 
		min_queue_examples, batch_size, shuffle = True)



# if __name__ == '__main__':
# 		maybe_download_and_extract()	