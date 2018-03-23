#!/usr/local/bin//python3

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_sample_image


# def plot_image(image):
#     plt.imshow(image, cmap="gray", interpolation="nearest")
#     plt.axis("off")

# def plot_color_image(image):
#     plt.imshow(image.astype(np.uint8),interpolation="nearest")
#     plt.axis("off")

seed = 42
tf.reset_default_graph()
tf.set_random_seed(seed)
np.random.seed(seed)

# Load sample images
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape

print("printing original images")
for i in range(2):
    # plt.imshow(dataset[i], cmap="gray", interpolation="nearest")  # grayscale
    plt.imshow(dataset[i].astype(np.uint8), interpolation="nearest")
    plt.axis("off")
    plt.show()

"""
Shapes
images
    [height, width, channels]  # channels: e.g. RGB
mini-batch
    [mini-batch size, height, width, channels]
weights of convolutional layer
    [fh, fw, fn', fn]
    fh, fw: height and width of receptive field
    fn': number of feature maps in previous layer l-1
    fn: number of feature maps in layer l
filters or convolution kernels
    [height, width, channels, num_filters]
feature map
    layer of neurons with the same filter
    one convolutional layers is a stack of feature maps

Number of parameters in convolution layer l:
    ( kernel size (height * width)
        * num of channels (if input) or num of feature maps in layer l-1
        + 1 for bias )
        * number of feature maps in layer l

output dimension:
    (input_w / stride_w) * (input_h / stride_h) * num feature map
"""

##################
# tf.nn.conv2d
##################
print("using tf.nn.conv2d...")

# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line

# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
# None: will be number of samples
convolution = tf.nn.conv2d(X, filters, strides=[1, 2, 2, 1], padding="SAME")

# or
with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

for image_index in (0, 1):
    for feature_map_index in (0, 1):  # vertical and horizontal filter
        image = output[image_index, :, :, feature_map_index]
        plt.imshow(image, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.show()

##################
# tf.layers.conv2d
##################
print("using tf.layers.conv2d...")

# tf.layers.conv2d filters are created automatically
tf.reset_default_graph()

X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
# creates 2 7*7 feature maps with 2*2 strides
convolution2 = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2, 2],
                                padding="SAME")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    output = sess.run(convolution2, feed_dict={X: dataset})

for image_index in (0, 1):
    for feature_map_index in (0, 1):  # vertical and horizontal filter
        image = output[image_index, :, :, feature_map_index]
        plt.imshow(image, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.show()

################
# tf.nn.max_pool
################
print("using tf.nn.max_pool...")

tf.reset_default_graph()
X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)

max_pool = tf.nn.max_pool(X,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding="VALID")

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})

plt.imshow(output[0].astype(np.uint8))  # plot the output for the 1st image
plt.show()
