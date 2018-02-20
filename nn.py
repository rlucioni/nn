"""
mnist classifiers

TODO: https://www.tensorflow.org/tutorials/layers
"""
import random

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class ModelBase:
    def __init__(self, learning_rate=0.5):
        # define and run variable init operation
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)

    def load(self):
        # mnist.train: 55k points of training data
        # mnist.test: 10k points of test data
        # mnist.validation: 5k points of validation data
        return input_data.read_data_sets('mnist_data/', one_hot=True)

    def train(self, steps=1000, batch_size=100):
        pass

    def test(self):
        pass

    def predict(self):
        pass

    def humanize(self, label):
        number, _ = max(enumerate(label), key=lambda x: x[1])

        return number


class BadModel(ModelBase):
    """
    https://www.tensorflow.org/versions/r0.12/tutorials/mnist/beginners/
    """
    def __init__(self, learning_rate=0.5):
        self.mnist = self.load()

        # placeholder for flattened 28x28 = 784 pixel intensity vectors
        self.x = tf.placeholder(tf.float32, shape=[None, 784])

        # placeholder for correct labels
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        # variables to be learned
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        # define model
        y = tf.matmul(self.x, W) + b

        # define loss
        # cross-entropy explained at http://colah.github.io/posts/2015-09-Visual-Information/
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=y))

        # define optimizer
        # more optimizers at https://www.tensorflow.org/api_guides/python/train#optimizers
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

        # get a list of booleans comparing predicted labels to correct labels
        self.predicted_labels = tf.argmax(y, 1)
        correct_labels = tf.argmax(self.y_, 1)
        correct_predictions = tf.equal(self.predicted_labels, correct_labels)

        # determine fraction of correct predictions
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        super().__init__()

    def train(self, steps=1000, batch_size=100):
        for step in range(steps):
            # feed batches of random training points to the train_step (stochastic training)
            batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
            feed_dict = {
                self.x: batch_xs,
                self.y_: batch_ys,
            }

            if step % 100 == 0:
                train_accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
                print(f'step {step}, training accuracy {train_accuracy:.3f}')

            self.sess.run(self.train_step, feed_dict=feed_dict)

    def test(self):
        feed_dict = {
            self.x: self.mnist.test.images,
            self.y_: self.mnist.test.labels,
        }
        test_accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
        print(f'test accuracy {test_accuracy:.3f}')

    def predict(self):
        index = random.randrange(len(self.mnist.validation.labels))
        label = self.mnist.validation.labels[index]

        print(f'actual {self.humanize(label)}')

        feed_dict = {
            self.x: self.mnist.validation.images[index:index + 1],
        }
        predicted = self.sess.run(self.predicted_labels, feed_dict=feed_dict)

        print(f'predicted {predicted[0]}')


class GoodModel(ModelBase):
    """
    https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/
    """
    def __init__(self, learning_rate=1e-3):
        self.mnist = self.load()

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        # first convolutional layer
        W_conv1 = self.weight_var([5, 5, 1, 32])
        b_conv1 = self.bias_var([32])
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # second convolutional layer
        W_conv2 = self.weight_var([5, 5, 32, 64])
        b_conv2 = self.bias_var([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # densely connected layer
        W_fc1 = self.weight_var([7 * 7 * 64, 1024])
        b_fc1 = self.bias_var([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # readout layer
        W_fc2 = self.weight_var([1024, 10])
        b_fc2 = self.bias_var([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=y_conv))
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

        self.predicted_labels = tf.argmax(y_conv, 1)
        correct_labels = tf.argmax(self.y_, 1)
        correct_predictions = tf.equal(self.predicted_labels, correct_labels)

        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        super().__init__()

    def weight_var(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)

        return tf.Variable(initial)

    def bias_var(self, shape):
        initial = tf.constant(0.1, shape=shape)

        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def train(self, steps=1000, batch_size=100, dropout=0.5):
        for step in range(steps):
            batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)

            if step % 100 == 0:
                feed_dict = {
                    self.x: batch_xs,
                    self.y_: batch_ys,
                    self.keep_prob: 1.0,
                }
                train_accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
                print(f'step {step}, training accuracy {train_accuracy:.3f}')

            feed_dict = {
                self.x: batch_xs,
                self.y_: batch_ys,
                self.keep_prob: dropout,
            }
            self.sess.run(self.train_step, feed_dict=feed_dict)

    def test(self):
        feed_dict = {
            self.x: self.mnist.test.images,
            self.y_: self.mnist.test.labels,
            self.keep_prob: 1.0,
        }
        test_accuracy = self.sess.run(self.accuracy, feed_dict=feed_dict)
        print(f'test accuracy {test_accuracy:.3f}')

    def predict(self):
        index = random.randrange(len(self.mnist.validation.labels))
        label = self.mnist.validation.labels[index]

        print(f'actual {self.humanize(label)}')

        feed_dict = {
            self.x: self.mnist.validation.images[index:index + 1],
            self.keep_prob: 1.0,
        }
        predicted = self.sess.run(self.predicted_labels, feed_dict=feed_dict)

        print(f'predicted {predicted[0]}')
