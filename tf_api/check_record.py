# Use this script to see tensorflow record file format visually. 

import tensorflow as tf
'''
for example in tf.python_io.tf_record_iterator("data/train.record"):
	result = tf.train.Example.FromString(example)
	print result
	break
'''
#f = open('train.txt', 'w')
#for example in tf.python_io.tf_record_iterator("data/test.record"):
	#result = tf.train.Example.FromString(example)
	#print result
	#print type(result)
	#f.write(result)
#f.close

record_iterator = tf.python_io.tf_record_iterator("data/test.record")

with tf.Session() as sess:
	for string_record in record_iterator:
		example = tf.train.Example()
		example.ParseFromString(string_record)
		print (example)