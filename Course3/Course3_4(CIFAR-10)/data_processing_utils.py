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

from six.moves import urllib

parser = argparse.ArgumentParser(description = 'Prepare the data for nn training')
parser.add_argument('--batch_size', type = int, 
					help = "Number of images to process in a batch")
parser.add_argument('--data_dir', type = str, 
					help = "Path to the CIFAR-10 data directory")
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

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

def load_data_from_binary_file(filepath, img_num = 10000, img_size = 3072, label_size = 1):

	if not os.path.exists(filepath):
		raise ValueError("Can't find file %s" % (file))

	datasets = []
	labels = []
	format_str = '>{0}B'.format(img_size + label_size)
	with open(filepath, 'rb') as fr:
		buffer = fr.read()
		for i in range(img_num):
			offset = struct.calcsize(format_str) * i
			tmp = struct.unpack_from(format_str, buffer, offset)
			labels.append(tmp[0])
			datasets.append(tmp[1:])

	print(len(datasets))

load_data_from_binary_file(os.path.join(data_dir, 'cifar-10-batches-bin', 'data_batch_1.bin'))
