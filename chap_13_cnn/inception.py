#!/usr/local/bin//python3

import os
import sys
import time
from datetime import datetime
import csv
import re
import itertools as it
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

"""Check inception_v4 instead for the right way to feed jpg pictures
using urllib and tf.image.decode_jpeg
"""

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# restore pretrained model
INCEPTION_V3_CHECKPOINT_PATH = "./checkpoints/inception_v3.ckpt"
# get class name
CLASS_NAME_REGEX = re.compile(r"^n\d+\s+(.*)\s*$", re.M | re.U)


def load_class_names():
    with open(os.path.join("./checkpoints/imagenet_class_names.txt"), "rb") as f:
        content = f.read().decode("utf-8")
        return CLASS_NAME_REGEX.findall(content)


class_names = ["background"] + load_class_names()


reset_graph()
X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="X")
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(  # builds inception model
        X, num_classes=1001, is_training=False)
predictions = end_points["Predictions"]
saver = tf.train.Saver()

# inspect model loaded
# Print name and shape of each tensor.
print("Layers")
for k, v in end_points.items():
    print('name = {}, shape = {}'.format(v.name, v.get_shape()))

# Print name and shape of parameter nodes  (values not yet initialized)
# print("\n")
# print("Parameters")
# for v in slim.get_model_variables():
#     print('name = {}, shape = {}'.format(v.name, v.get_shape()))

# prediction tests
# with tf.Session() as sess:
#     saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
#     predictions_val = predictions.eval(feed_dict={X: X_test})
#
# predictions_val = predictions.eval(feed_dict={X: X_test})
#
# most_likely_class_index = np.argmax(predictions_val)
# most_likely_class_index
# class_names[most_likely_class_index]
#
# # top 5 prediction
# top_5 = np.argpartition(predictions_val[0], -5)[-5:]
# top_5 = reversed(top_5[np.argsort(predictions_val[0][top_5])])
# for i in top_5:
#     print("{0}: {1:.2f}%".format(class_names[i], 100 * predictions_val[0][i]))

# image dimensions
width = 299
height = 299
channels = 3

image_filepath = "/Users/Quentin/Documents/Projects/popsy/data/raw_urls/data-with-images-000000000000.csv"

f = csv.DictReader(open(image_filepath, 'r'))
# odict_keys(['category_name', 'title', 'country', 'image'])

for row_dict in it.islice(f, 0, 30):
    print(row_dict['title'], row_dict['country'])

# OrderedDict([('category_name', '559150b3531b3b92438b4574'), ('title', 'Ap Nascente na Pelinca com Duas Vagas na Garagem'), ('country', 'BR'), ('image', 'https://lh3.goo')])
# OrderedDict([('category_name', '55906905531b3b93438b456e'), ('title', 'DVD docteur who saison 5.'), ('country', 'FR'), ('image', 'http://lh3.goo')])


# predictions on csv file of image urls
with tf.Session() as sess:
    saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
    for row_dict in it.islice(f, 26, 28):
        url = row_dict['image']
        data = requests.get(url).content
        im = Image.open(BytesIO(data))
        # im.format  # JPEG
        # im.mode  # RGB
        # im.size  # (435, 500)
        im.resize((299, 299)).save('im_resized.jpg')
        # TODO remove saving step
        im_mpimg = mpimg.imread('./im_resized.jpg', format=None)[:, :, :channels]
        # im_mpimg.shape  # (299, 299, 3)
        im_mpimg = 2 * (im_mpimg/255) - 1  # resize to [-1, 1] range
        # predictions
        X_test = im_mpimg.reshape(-1, height, width, channels)
        predictions_val = predictions.eval(feed_dict={X: X_test})
        most_likely_class_index = np.argmax(predictions_val)
        class_names[most_likely_class_index]
        print("Label: {}\tGuess: {}".format(
            row_dict['title'],
            class_names[most_likely_class_index]), end='\n')
        plt.imshow(im)
        plt.axis("off")
        plt.show()




# EoF
