#!/usr/local/bin//python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

matplotlib.use("MacOSX", warn=False, force=True)


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    plt.legend()


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def fetch_batch(X, y, epoch, batch_index, batch_size=100):
    n = X.shape[0]
    # print("n = ".format(n))
    n_batches = int(np.ceil(n / batch_size))
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(n, size=batch_size)
    indices
    X_batch = X[indices]
    y_batch = y.reshape(-1, 1)[indices]
    return X_batch, y_batch


def main():
    #############
    # Load data
    #############
    from sklearn.datasets import make_moons
    X_moon, y_moon = make_moons(n_samples=100, noise=0.15, random_state=42)
    X_dtype = X_moon.dtype  # to coerce y to same dtype (float64)
    # X_moon_bias = np.c_[np.ones((n, 1)), X_moon]  # done by poly
    print("Loaded {} points".format(X_moon.shape[0]))

    # plot data
    print("Plotting dataset")
    plot_dataset(X_moon, y_moon, [-1.5, 2.5, -1, 1.5])
    plt.show()

    y_moon = y_moon.reshape(-1, 1).astype('float64')

    # Add polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    pf = PolynomialFeatures(degree=4,
                            interaction_only=False,
                            include_bias=True)
    X_moon_bias = pf.fit_transform(X_moon, y=None)
    n, p = X_moon_bias.shape

    ##########
    # training
    ##########

    # logging
    from datetime import datetime
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    # train/ test split
    ratio = 0.8
    train_indices = int(ratio * n)
    X_train = X_moon_bias[:train_indices]
    y_train = y_moon[:train_indices]
    X_test = X_moon_bias[train_indices:]
    y_test = y_moon[train_indices:]

    # parameters
    reset_graph()
    n_epochs = 5000
    learning_rate = .01
    batch_size = 10
    n_batches = int(x=np.ceil(n / batch_size))

    X = tf.placeholder(dtype=X_dtype, shape=(None, p), name="X")
    y = tf.placeholder(dtype=X_dtype, shape=(None, 1), name="y")
    y.dtype
    theta = tf.Variable(tf.random_uniform([p, 1], -1, 1,
                                          dtype=X_dtype, seed=42),
                        name="theta", dtype=X_dtype)
    logits = tf.matmul(X, theta, name="logits")
    y_proba = tf.sigmoid(logits)
    loss = tf.losses.log_loss(y, y_proba)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    loss_summary = tf.summary.scalar('Loss', loss)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init, feed_dict=None, run_metadata=None)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(X_train, y_train, epoch,
                                               batch_index, batch_size)
                feed_dict = {X: X_batch, y: y_batch}
                if batch_index % 10 == 0:
                    summary_str = loss_summary.eval(feed_dict=feed_dict)
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, global_step=step)
                sess.run(training_op,
                         feed_dict=feed_dict,
                         run_metadata=tf.RunMetadata())
            if epoch % 100 == 0:
                print("Epoch: {}, Logloss: {}".format(
                    epoch,
                    loss.eval(feed_dict={X: X_moon_bias, y: y_moon})))

        y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
        best_theta = theta.eval()
        save_path = saver.save(sess, "saved_models/moon_regression.ckpt")

    print("Best theta = {}".format(best_theta), end='\n')

    #########
    # Results
    #########

    y_pred = (y_proba_val >= 0.5)
    from sklearn.metrics import precision_score, recall_score  # noqa
    precision = precision_score(y_test, y_pred)
    print("precision: {}".format(precision))
    recall = recall_score(y_test, y_pred)
    print("recall: {}".format(recall))

    # plot results
    y_pred_idx = y_pred.reshape(-1)  # a 1D array rather than a column vector
    plt.plot(X_test[y_pred_idx, 1],
             X_test[y_pred_idx, 2], 'go', label="Positive")
    plt.plot(X_test[~y_pred_idx, 1],
             X_test[~y_pred_idx, 2], 'r^', label="Negative")
    plt.legend()
    plt.show()

    #########
    # restore
    #########
    # with tf.Session() as sess:
    #     saver.restore(sess, save_path)
    #     best_theta_restored = theta.eval()
    #     print("best_theta_restored: {}".format(best_theta_restored))


if __name__ == "__main__":
    main()
