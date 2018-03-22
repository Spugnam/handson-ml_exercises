#!/usr/local/bin//python3

import numpy as np
import tensorflow as tf

tf.reset_default_graph()

filter_primes = np.array([2., 3., 5., 7., 11., 13.], dtype=np.float32)
X = tf.constant(np.arange(1, 13+1, dtype=np.float32).reshape([1, 1, 13, 1]))
filters = tf.constant(filter_primes.reshape(1, 6, 1, 1))

valid_conv = tf.nn.conv2d(X, filters, strides=[1, 1, 5, 1], padding='VALID')
same_conv = tf.nn.conv2d(X, filters, strides=[1, 1, 5, 1], padding='SAME')

with tf.Session() as sess:
    print("VALID:\n", valid_conv.eval())
    print("SAME:\n", same_conv.eval())
# ('VALID:\n', array([[[[184.],
#          [389.]]]], dtype=float32))
# ('SAME:\n', array([[[[143.],
#          [348.],
#          [204.]]]], dtype=float32))

# Verification
print("VALID:")
print(np.array([1, 2, 3, 4, 5, 6]).T.dot(filter_primes))
print(np.array([6, 7, 8, 9, 10, 11]).T.dot(filter_primes))
# VALID:
# 184.0
# 389.0

print("SAME:")
print(np.array([0, 1, 2, 3, 4, 5]).T.dot(filter_primes))
print(np.array([5, 6, 7, 8, 9, 10]).T.dot(filter_primes))
print(np.array([10, 11, 12, 13, 0, 0]).T.dot(filter_primes))
# SAME:
# 143.0
# 348.0
# 204.0
