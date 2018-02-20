"""
simple MNIST classifier

https://www.tensorflow.org/tutorials/layers
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/beginners/
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
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
    def __init__(self, learning_rate=0.5):
        self.mnist = self.load()

        # placeholder for flattened 28x28 = 784 pixel intensity vectors
        self.x = tf.placeholder(tf.float32, shape=[None, 784])

        # placeholder for correct labels
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        # variables to be learned
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        # model
        y = tf.nn.softmax(tf.matmul(self.x, W) + b)

        # loss to be minimized
        # cross-entropy explained at http://colah.github.io/posts/2015-09-Visual-Information/
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(y), reduction_indices=[1]))

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
            # feed batches of 100 random training points to the train_step (stochastic training)
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
