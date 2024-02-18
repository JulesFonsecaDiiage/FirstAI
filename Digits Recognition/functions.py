import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def train(X_train, Y_train, X_valid, Y_valid, layers, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """function that builds, trains, and saves a neural network classifier"""
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
        y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')
        y_pred = create_layer(x, layers, activations)
        loss = tf.losses.softmax_cross_entropy(y, y_pred)
        train_op = tf.train.AdamOptimizer(alpha).minimize(loss)
        accuracy = tf.metrics.accuracy(tf.argmax(y, 1), tf.argmax(y_pred, 1))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy[1])
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('/tmp/tensorflow', sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for i in range(iterations + 1):
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            if i % 100 == 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(
                    sess.run(loss, feed_dict={x: X_train, y: Y_train})))
                print("\tTraining Accuracy: {}".format(
                    sess.run(accuracy, feed_dict={x: X_train, y: Y_train})))
                print("\tValidation Cost: {}".format(
                    sess.run(loss, feed_dict={x: X_valid, y: Y_valid})))
                print("\tValidation Accuracy: {}".format(
                    sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})))
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            if i % 100 == 0:
                summary = sess.run(merged, feed_dict={x: X_train, y: Y_train})
                writer.add_summary(summary, i)
        save_path = saver.save(sess, save_path)
    return save_path