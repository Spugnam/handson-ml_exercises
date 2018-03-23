#!/usr/local/bin//python3

import os
import sys
from datetime import datetime
from functools import partial
import numpy as np
import tensorflow as tf


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def neuron_layer(input_layer, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(input_layer.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)  # helps convergence
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(input_layer, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


def fetch_batch(X, y, batch_size=100):
    n = X.shape[0]
    indices = np.random.randint(n, size=batch_size)
    X_batch = X[indices].reshape(-1, height, width, 1)
    y_batch = y[indices]
    return X_batch, y_batch


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


if __name__ == "__main__":
    ##############
    # Construction
    ##############

    # parameters
    # network
    reset_graph()
    height = 28
    width = 28
    channels = 1
    n_hidden = 100  # for dense hidden layers
    n_outputs = 10

    # training
    learning_rate = 0.01
    n_epochs = 30
    batch_size = 2048
    best_loss = np.infty
    epochs_without_progress = 0
    max_epochs_without_progress = 10
    batch_norm_momentum = 0.9
    dropout_rate = 0.5

    X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    training = tf.placeholder_with_default(False, shape=(), name="training")

    # neural network
    layers = []
    with tf.name_scope("CNN"):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        my_batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=training,
            momentum=batch_norm_momentum)
        my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=he_init)
        my_dropout_layer = partial(
            tf.layers.dropout,
            rate=dropout_rate,
            training=training)

        # creates 2 7*7 feature maps with 2*2 strides
        conv1 = tf.layers.conv2d(
            X, filters=100, kernel_size=7, strides=(1, 1), padding="SAME",
            activation=tf.nn.relu, name="conv1")
        print("conv1 format: {}".format(conv1.get_shape()))
        conv2 = tf.layers.conv2d(
            conv1, filters=100, kernel_size=7, strides=(1, 1), padding="SAME",
            activation=tf.nn.relu, name="conv2")
        print("conv2 format: {}".format(conv2.get_shape()))

        # conv2 output: height * width * channels * 10 * 10
        conv2_r = tf.reshape(conv2, [-1, 784 * 10])
        hidden2 = my_dense_layer(
            conv2_r, n_hidden, name="hidden2")
        hidden2_drop = my_dropout_layer(hidden2)
        bn2 = tf.nn.elu(my_batch_norm_layer(hidden2_drop))
        logits_before_bn = my_dense_layer(bn2, n_outputs, name="outputs")
        logits = my_batch_norm_layer(logits_before_bn)
        # not softmax for optimization

    # loss function
    with tf.name_scope("loss"):
        xentropy =\
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                           logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
        loss_summary = tf.summary.scalar('log_loss', loss)

    # optimizer
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    # evaluation metric
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, k=1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    ###########
    # Execution
    ###########
    # import MNIST
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("tmp/data/")
    X_train = mnist.train.images  # (55000, 784)
    X_test = mnist.test.images  # (10000, 784)
    X_valid = mnist.validation.images  # (5000, 784)
    y_train = mnist.train.labels
    y_test = mnist.test.labels
    y_valid = mnist.validation.labels

    # logging
    if len(sys.argv) > 1:
        run_description = sys.argv[1]
    else:
        run_description = ""
        print("No log description entered")
    logdir = log_dir("mnist_dnn_" + run_description)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    checkpoint_path = "./tmp/my_deep_mnist_model.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = "./my_deep_mnist_model"

    # for batch normalization
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # training
    with tf.Session() as sess:
        if os.path.isfile(checkpoint_epoch_path):
            # if checkpoint file exists, restore model and load epoch number
            with open(checkpoint_epoch_path, "rb") as f:
                start_epoch = int(f.read())
            print("Training was interrupted. Continuing at epoch", start_epoch)
            saver.restore(sess, checkpoint_path)
        else:
            start_epoch = 0
            sess.run(init)

        for epoch in range(start_epoch, n_epochs):
            for iteration in range(len(X_train) // batch_size):
                X_batch, y_batch = fetch_batch(X_train, y_train,
                                               batch_size)
                sess.run([training_op, extra_update_ops],
                         feed_dict={X: X_batch, y: y_batch})
            # metrics
            acc_train = accuracy.eval(
                feed_dict={X: X_train.reshape(-1, height, width, 1),
                           y: y_train})
            acc_val = accuracy.eval(
                feed_dict={X: X_test.reshape(-1, height, width, 1),
                           y: y_test})
            print(epoch, "Train accuracy: {}, Test accuracy: {}".format(
                acc_train, acc_val))
            accuracy_val, loss_val, accuracy_summary_str, loss_summary_str =\
                sess.run([accuracy, loss, accuracy_summary, loss_summary],
                         feed_dict={X: X_valid.reshape(-1, height, width, 1),
                                    y: y_valid})
            file_writer.add_summary(accuracy_summary_str, epoch)
            file_writer.add_summary(loss_summary_str, epoch)
            print("Epoch:", epoch,
                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                  "\tValidation Loss: {:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 1
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break
        os.remove(checkpoint_epoch_path)  # remove intermediate run

    # restore saved model
    # with tf.Session() as sess:
    #     saver.restore(sess, final_model_path)
    #     accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
    # accuracy_val
