#!/usr/local/bin//python3

import os
import sys
import time
import json
from datetime import datetime
from functools import partial
import numpy as np
import tensorflow as tf


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def fetch_batch(X, y, batch_size=100):
    n = X.shape[0]
    indices = np.random.randint(n, size=batch_size)
    X_batch = X[indices].reshape(-1, height, width, 1)
    y_batch = y[indices]
    return X_batch, y_batch


def log_dir(root="", desc=""):
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    module = os.path.basename(__file__).split('.')[0]  # filename w/o extension
    if len(desc) > 0:
        desc = "-" + desc
    name = module + desc + "-run-" + now
    return "{}/{}/".format(root, name)


def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value
            in zip(gvars, tf.get_default_session().run(gvars))}


def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(
        gvar_name + "/Assign") for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op
                   in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name]
                 for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


if __name__ == "__main__":
    ##############
    # Construction
    ##############

    # parameters
    # log parameters
    if len(sys.argv) > 1:
        run_description = sys.argv[1]
    else:
        run_description = ""
    params_logdir = log_dir(root="parameters", desc=run_description)
    os.makedirs(params_logdir)

    from collections import defaultdict
    params = defaultdict(dict)
    params

    # network
    # params['input']['height'] = 28
    # params['input']['width'] = 28
    # params['input']['channels'] = 1

    width = 28
    height = 28
    channels = 1

    params['conv1']['filters'] = 32
    params['conv1']['kernel_size'] = [1, 1]
    params['conv1']['strides'] = [1, 1]
    params['conv1']['padding'] = "SAME"
    params['conv1']['name'] = "conv1"

    params['conv2']['filters'] = 64
    params['conv2']['kernel_size'] = [1, 1]
    params['conv2']['strides'] = [2, 2]
    params['conv2']['padding'] = "SAME"
    params['conv2']['name'] = "conv2"

    # params['pool3']['filters'] = params['conv2']['filters']
    params['pool3']['ksize'] = [1, 2, 2, 1]
    params['pool3']['strides'] = [1, 2, 2, 1]
    params['pool3']['padding'] = "VALID"
    params['pool3']['name'] = "pool3"

    params['dens4']['units'] = 128
    params['dens4']['name'] = 'dens4'
    params['dens4_drop']['rate'] = 0.

    # filters = 4
    # ksize = 3
    # stride = 1
    # n_hidden = 100  # for dense hidden layers
    n_outputs = 10

    # training
    params['learning_rate'] = 0.01
    n_epochs = 30
    params['batch_size'] = 2048
    best_loss = np.infty
    epochs_without_progress = 0
    max_epochs_without_progress = 10
    batch_norm_momentum = 0.9
    # dropout_rate = 0.0

    # save parameters
    with open(os.path.join(params_logdir, 'params.json'), 'w') as f:
        json.dump(params, f)

    reset_graph()

    with tf.name_scope("inputs"):
        X = tf.placeholder(shape=(None, height, width, channels),
                           dtype=tf.float32, name="X")
        y = tf.placeholder(tf.int64, shape=(None), name="y")
        training = tf.placeholder_with_default(False, shape=(), name="training")

    # neural network
    with tf.name_scope("CNN"):
        he_init = tf.contrib.layers.\
            variance_scaling_initializer(dtype=tf.float32)
        my_batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=training,
            momentum=batch_norm_momentum)
        my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=he_init)
        # my_dropout_layer = partial(
        #     tf.layers.dropout,
        #     rate=dropout_rate,
        #     training=training)

        conv1 = tf.layers.conv2d(X, **params['conv1'], activation=tf.nn.elu)
        # print("conv1 shape: {}".format(conv1.get_shape()))
        conv2 = tf.layers.conv2d(conv1, **params['conv2'], activation=tf.nn.elu)
        # print("conv2 shape: {}".format(conv2.get_shape()))

        pool3 = tf.nn.max_pool(conv2, **params['pool3'])
        # print("pool3 shape: {}".format(pool3.get_shape()))

        # calculate dimensions out of conv2
        _w = int(width / (params['conv1']['strides'][0] *
                          params['conv2']['strides'][0] *
                          params['pool3']['strides'][1]))
        _h = int(height / (params['conv1']['strides'][1] *
                           params['conv2']['strides'][1] *
                           params['pool3']['strides'][2]))

        pool3_flat =\
            tf.reshape(pool3, shape=(-1, _w * _h * params['conv2']['filters']))
        # print("pool3_flat shape: {}".format(pool3_flat.get_shape()))

        dens4 = my_dense_layer(pool3_flat, **params['dens4'])
        dens4_drop = tf.layers.dropout(dens4, **params['dens4_drop'])
        # print("dens4_drop.dtype: {}".format(dens4_drop.get_shape()))

        bn5 = tf.nn.relu(my_batch_norm_layer(dens4_drop))
        # issue 8535: needs float32, doesn't handle float16
        logits_before_bn = my_dense_layer(bn5, n_outputs, name="outputs")
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
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
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
    tf_logdir = log_dir(root="tf_logs", desc=run_description)
    # tensorboard
    file_writer = tf.summary.FileWriter(tf_logdir, tf.get_default_graph())
    # save models
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
            for iteration in range(len(X_train) // params['batch_size']):
                X_batch, y_batch = fetch_batch(X_train, y_train,
                                               params['batch_size'])
                sess.run([training_op, extra_update_ops],
                         feed_dict={X: X_batch, y: y_batch})
            # metrics
            acc_train = accuracy.eval(
                feed_dict={X: X_train.reshape(-1, height, width, 1), y: y_train})
            acc_test = accuracy.eval(
                feed_dict={X: X_test.reshape(-1, height, width, 1), y: y_test})
            print(epoch, "Train acc: {}, Test acc: {}".format(acc_train, acc_test))

            accuracy_val, loss_val, accuracy_summary_str, loss_summary_str =\
                sess.run([accuracy, loss, accuracy_summary, loss_summary],
                         feed_dict={X: X_valid.reshape(-1, height, width, 1),
                                    y: y_valid})
            print("Epoch:", epoch,
                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100),
                  "\tValidation Loss: {:.5f}".format(loss_val))
            file_writer.add_summary(accuracy_summary_str, epoch)
            file_writer.add_summary(loss_summary_str, epoch)
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))

            if loss_val < best_loss:
                # saver.save(sess, final_model_path)
                best_loss = loss_val
                best_model_params = get_model_params()  # cache best params
            else:
                epochs_without_progress += 1
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break

        # restore parameters from best run
        if best_model_params:
            restore_model_params(best_model_params)
        saver.save(sess, final_model_path)
        os.remove(checkpoint_epoch_path)  # remove intermediate run

    # restore saved model
    # with tf.Session() as sess:
    #     saver.restore(sess, final_model_path)
    #     accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})
    # accuracy_val
