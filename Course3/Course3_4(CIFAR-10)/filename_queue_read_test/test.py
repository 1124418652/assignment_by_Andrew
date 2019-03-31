#-*- coding: utf-8 -*-
import tensorflow as tf 

filename = ['A.jpg', 'B.jpg', 'C.jpg']
filename_queue = tf.train.string_input_producer(filename, 
	num_epochs = 5, shuffle = True)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
v = tf.decode_raw(value, tf.int8)
data1 = tf.strided_slice(v, [0], [1])

print(key, value)
with tf.Session() as sess:
	tf.local_variables_initializer().run()
	threads = tf.train.start_queue_runners(sess = sess)
	print("data1", sess.run(data1))
	i = 0
	while True:
		i += 1
		image_data = sess.run(value)

		with open('read/test%d.jpg' %i, 'wb') as fw:
			fw.write(image_data)