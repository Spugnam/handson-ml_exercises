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
    X_batch = X[indices]
    y_batch = y[indices]
    return X_batch, y_batch


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


def dnn(inputs, num_layers, units, activation=tf.nn.elu,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        name=None):
    """ Creates dense neural network
    Defaults: he init and ELU activation
    """
    with tf.variable_scope(name, "dnn"):
        # with tf.name_scope("dnn"):
        for layer in range(num_layers):
            inputs = tf.layers.dense(inputs,
                                     units=units,
                                     kernel_initializer=kernel_initializer,
                                     name="hidden%i" % layer)
    return inputs

if __name__ == "__main__":
    ##############
    # Construction
    ##############
    # parameters #

    # network
    reset_graph()
    n_inputs = 28*28
    num_layers = 5  # number of hidden layers
    n_hidden = 100  # number of units per hidden layer
    n_outputs = 5
    # training
    learning_rate = 0.01
    n_epochs = 30
    batch_size = 2048
    best_loss = np.infty
    epochs_without_progress = 0
    max_epochs_without_progress = 10
    batch_norm_momentum = 0.9
    dropout_rate = 0.5

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    training = tf.placeholder_with_default(False, shape=(), name="training")

    # neural network #

    # initializations
    he_init = tf.contrib.layers.variance_scaling_initializer()
    X_drop = tf.layers.dropout(X, rate=dropout_rate, training=training)

    my_batch_norm_layer = partial(
        tf.layers.batch_normalization,
        training=training,
        momentum=batch_norm_momentum)

    my_dense_layer = partial(
        tf.layers.dense,
        kernel_initializer=he_init)

    # build network
    layers = dnn(X_drop, num_layers=num_layers, units=n_hidden,
                 kernel_initializer=he_init)
    logits_before_bn = my_dense_layer(layers, n_outputs, name="outputs")
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

    # only train digits 0 to 4
    idx_train_0_4 = np.isin(y_train, list(range(5)))
    idx_test_0_4 = np.isin(y_test, list(range(5)))
    idx_valid_0_4 = np.isin(y_valid, list(range(5)))

    X_train_0_4 = X_train[idx_train_0_4]
    X_test_0_4 = X_test[idx_test_0_4]
    X_valid_0_4 = X_valid[idx_valid_0_4]
    y_train_0_4 = y_train[idx_train_0_4]
    y_test_0_4 = y_test[idx_test_0_4]
    y_valid_0_4 = y_valid[idx_valid_0_4]

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
            for iteration in range(len(X_train_0_4) // batch_size):
                X_batch, y_batch = fetch_batch(X_train_0_4, y_train_0_4,
                                               batch_size)
                sess.run([training_op, extra_update_ops],
                         feed_dict={X: X_batch, y: y_batch})
            # metrics
            acc_train = accuracy.eval(feed_dict={X: X_train_0_4,
                                                 y: y_train_0_4})
            loss_train = loss.eval(feed_dict={X: X_train_0_4,
                                              y: y_train_0_4})
            print("Epoch:", epoch,
                  "\tTrain accuracy: {:.3f}%".format(acc_train * 100),
                  "\tTrain Loss: {:.5f}".format(loss_train))
            # accuracy_test, loss_test, _, _ =\
            #     sess.run([accuracy, loss, accuracy_summary, loss_summary],
            #              feed_dict={X: X_test_0_4, y: y_test_0_4})
            # print("Epoch:", epoch,
            #       "\Test accuracy: {:.3f}%".format(accuracy_test * 100),
            #       "\tTest Loss: {:.5f}".format(loss_test))
            accuracy_val, loss_val, accuracy_summary_str, loss_summary_str =\
                sess.run([accuracy, loss, accuracy_summary, loss_summary],
                         feed_dict={X: X_valid_0_4, y: y_valid_0_4})
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
