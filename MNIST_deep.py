#!/usr/local/bin//python3

import os
from datetime import datetime
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
    reset_graph()
    n_inputs = 28*28
    n_hidden1 = 400
    n_hidden2 = 150
    n_outputs = 10
    learning_rate = 0.005
    n_epochs = 100
    batch_size = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    # neural network
    with tf.name_scope("DNN"):
        hidden1 = neuron_layer(X, n_hidden1,
                               name="hidden1", activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1, n_hidden2,
                               name="hidden2", activation=tf.nn.relu)
        logits = neuron_layer(hidden2, n_outputs, name="output")
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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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
    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels.astype("int")
    y_test = mnist.test.labels.astype("int")
    X_valid = mnist.validation.images  # 5000 digits
    y_valid = mnist.validation.labels  # 5000 digits

    # logging
    logdir = log_dir("mnist_dnn")
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = "./my_deep_mnist_model"

    # training
    best_loss = np.infty
    epochs_without_progress = 0
    max_epochs_without_progress = 10

    # os.remove(checkpoint_epoch_path)

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
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            # acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            # acc_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
            # print(epoch, "Train accuracy: {}, Test accuracy: {}".format(
            #     acc_train, acc_val))
            accuracy_val, loss_val, accuracy_summary_str, loss_summary_str =\
                sess.run([accuracy, loss, accuracy_summary, loss_summary],
                         feed_dict={X: X_valid, y: y_valid})
            file_writer.add_summary(accuracy_summary_str, epoch)
            file_writer.add_summary(loss_summary_str, epoch)
            # if epoch % 5 == 0:
            print("Epoch:", epoch,
                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                  "\tLoss: {:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break
        os.remove(checkpoint_epoch_path)  # remove intermediate run

    # restore saved model
    # with tf.Session() as sess:
    #     saver.restore(sess, final_model_path)
    #     accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
    # accuracy_val
