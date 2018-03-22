#!/usr/local/bin//python3

import os
import sys
from datetime import datetime
# from functools import partial
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden_layers=5, units=100,
                 optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.01, batch_size=100, activation=tf.nn.elu,
                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  # noqa
                 batch_norm_momentum=None, dropout_rate=None,
                 random_state=None):
        """Initialize the DNNClassifier by simply storing all the
        hyperparameters.
        """
        self.n_hidden_layers = n_hidden_layers
        self.units = units
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._session = None

    def _dnn(self, inputs):
        """Build the hidden layers, with support for batch normalization and
        dropout."""
        for layer in range(self.n_hidden_layers):
            if self.dropout_rate:
                inputs = tf.layers.dropout(inputs, self.dropout_rate,
                                           training=self._training)
            inputs = tf.layers.dense(inputs, self.units,
                                     kernel_initializer=self.kernel_initializer,
                                     name="hidden%d" % (layer + 1))
            if self.batch_norm_momentum:
                inputs = tf.layers.batch_normalization(
                    inputs,
                    momentum=self.batch_norm_momentum,
                    training=self._training)
            inputs = self.activation(inputs, name="hidden%d_out" % (layer + 1))
        return inputs

    def _build_graph(self, n_inputs, n_outputs):
        """Called by fit method
        Saves parameters for access by other methods
        """
        if self.random_state is not None:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)

        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None), name="y")

        # if self.batch_norm_momentum or self.dropout_rate:
        self._training = tf.placeholder_with_default(False, shape=(),
                                                     name='training')
        # else:
        #     self._training = None

        dnn_outputs = self._dnn(X)

        logits = tf.layers.dense(dnn_outputs, n_outputs,
                                 kernel_initializer=self.kernel_initializer,
                                 name="logits")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")

        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                  logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
        loss_summary = tf.summary.scalar('log_loss', loss)

        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available through instance variables
        self._X, self._y = X, y
        self._Y_proba, self._loss = Y_proba, loss
        self._training_op, self._accuracy = training_op, accuracy
        self._loss_summary, self._accuracy_summary =\
            loss_summary, accuracy_summary
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        """Get all variable values (used for early stopping, faster than saving
        to disk)
        """
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in
                zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """Set all variables to the given values (for early stopping,
        faster than loading from disk)
        """
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(
            gvar_name + "/Assign") for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1]
                       for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name]
                     for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, y, n_epochs=100, X_valid=None, y_valid=None):
        """Fit the model to the training set. If X_valid and y_valid are
        provided, use early stopping.
        """
        self.close_session()

        # infer n_inputs and n_outputs from the training set.
        n_inputs = X.shape[1]
        self.classes_ = np.unique(y)
        n_outputs = len(self.classes_)

        """
        Translate the labels vector to a vector of sorted class indices,
        containing integers from 0 to n_outputs - 1.
        For example, if y is equal to [8, 8, 9, 5, 7, 6, 6, 6], then the
        sorted class labels (self.classes_) will be equal to [5, 6, 7, 8, 9],
        and the labels vector
        will be translated to [3, 3, 4, 0, 2, 1, 1, 1]
        """
        self.class_to_index_ = {label: index
                                for index, label in enumerate(self.classes_)}
        y = np.array([self.class_to_index_[label]
                      for label in y], dtype=np.int32)

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph(n_inputs, n_outputs)
            # extra ops for batch normalization
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # needed in case of early stopping
        max_checks_without_progress = 20
        checks_without_progress = 0
        best_loss = np.infty
        best_params = None

        # logging
        # program description (first argument)
        if len(sys.argv) > 1:
            run_description = sys.argv[1]
        else:
            run_description = ""
            print("No log description entered")
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        logdir = "tf_logs/mnist_dnn-" + run_description + "run-" + now
        # tensorboard logs
        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

        # recover intermediate runs
        checkpoint_path = "./tmp/my_deep_mnist_model.ckpt"
        checkpoint_epoch_path = checkpoint_path + ".epoch"
        final_model_path = "./my_deep_mnist_model"

        # Now train the model!
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            if os.path.isfile(checkpoint_epoch_path):
                # if checkpoint file exists, restore model and load epoch
                with open(checkpoint_epoch_path, "rb") as f:
                    start_epoch = int(f.read())
                print("Training interrupted. Continuing at epoch", start_epoch)
                self._saver.restore(sess, checkpoint_path)
            else:
                start_epoch = 0
                self._init.run()

            for epoch in range(n_epochs):
                rnd_idx = np.random.permutation(len(X))
                for rnd_indices in np.array_split(rnd_idx,
                                                  len(X) // self.batch_size):
                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]
                    feed_dict = {self._training: True,
                                 self._X: X_batch, self._y: y_batch}
                    sess.run(self._training_op, feed_dict=feed_dict)
                    if extra_update_ops:
                        sess.run(extra_update_ops, feed_dict=feed_dict)
                if X_valid is not None and y_valid is not None:
                    loss_val, acc_val, loss_summary, accuracy_summary =\
                        sess.run([self._loss, self._accuracy,
                                  self._loss_summary, self._accuracy_summary],
                                 feed_dict={self._X: X_valid, self._y: y_valid})
                    self._saver.save(sess, checkpoint_path)
                    file_writer.add_summary(accuracy_summary, epoch)
                    file_writer.add_summary(loss_summary, epoch)

                    with open(checkpoint_epoch_path, "wb") as f:
                        f.write(b"%d" % (epoch + 1))
                    if loss_val < best_loss:
                        self._saver.save(sess, final_model_path)
                        best_params = self._get_model_params()
                        best_loss = loss_val
                        checks_without_progress = 0
                    else:
                        checks_without_progress += 1
                    print("{}\tValidation loss: {:.6f}\
                    \tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(
                            epoch, loss_val, best_loss, acc_val * 100))
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping!")
                        break
                else:
                    loss_train, acc_train = sess.run(
                        [self._loss, self._accuracy],
                        feed_dict={self._X: X_batch, self._y: y_batch})
                    print("{}\tLast training batch loss:\
                          {:.6f}\tAccuracy: {:.2f}%".format(
                            epoch, loss_train, acc_train * 100))
            # If we used early stopping then rollback to the best model found
            if best_params:
                self._restore_model_params(best_params)
            os.remove(checkpoint_epoch_path)  # remove intermediate run
            return self

    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" %
                                 self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._Y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices], np.int32)

    def save(self, path):
        self._saver.save(self._session, path)


if __name__ == "__main__":
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

    # single run
    dnn_clf = DNNClassifier(
                 n_hidden_layers=5, units=100,
                 optimizer_class=tf.train.AdamOptimizer,
                 learning_rate=0.01, batch_size=512,
                 activation=tf.nn.elu,
                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  # noqa
                 batch_norm_momentum=None,
                 dropout_rate=0.5,
                 random_state=42)

    dnn_clf.fit(X_train_0_4, y_train_0_4,
                n_epochs=1000,
                X_valid=X_valid_0_4, y_valid=y_valid_0_4)

    print("Starting prediction...")
    y_pred = dnn_clf.predict(X_test_0_4)
    print("Test accuracy on trained model: {}".format(accuracy_score(
        y_test_0_4, y_pred)))

    # randomized search
    # param_distribs = {
    #     "units": [70, 100, 130, 160],
    #     "batch_size": [10, 50, 100, 500],
    #     "learning_rate": [0.005, 0.01, 0.05, 0.1],
    #     "activation": [tf.nn.relu, tf.nn.elu]
    # }
    # print("Parameters used for randomized search: {}".format(param_distribs))
    #
    # rnd_search = RandomizedSearchCV(
    #     DNNClassifier(random_state=42),
    #     param_distribs, n_iter=50,
    #     random_state=42, verbose=2)
    #
    # fit_params = {"X_valid": X_valid_0_4,
    #               "y_valid": y_valid_0_4}
    #
    # print("Starting randomized search...")
    # # rnd_search.fit(X_train_0_4, y_train_0_4, **fit_params)
    # rnd_search.fit(X_train_0_4, y_train_0_4,
    #                n_epochs=1000,
    #                X_valid=X_valid_0_4, y_valid=y_valid_0_4)
    #
    # print("Best parameters: {}".format(rnd_search.best_params_))
    # print("Starting prediction...")
    # y_pred = rnd_search.predict(X_test_0_4)
    # print("Test accuracy on best model: {}".format(accuracy_score(
    #     y_test_0_4, y_pred)))
    # rnd_search.best_estimator_.save("./best_gridsearch_mnist_model_0_to_4")

    # Best parameters: {'units': 70, 'learning_rate': 0.005, 'batch_size': 100,
    #                   'activation': <function elu at 0x10abb4158>}
    # Test accuracy on best model: 0.9920217941233703
