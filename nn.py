"""
simple MNIST classifier

https://www.tensorflow.org/tutorials/layers
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/beginners/
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax.py
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# mnist.train: 55,000 points of training data
# mnist.test: 10,000 points of test data
# mnist.validation: 5,000 points of validation data
mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

# placeholder for flattened 28x28 = 784 pixel intensity vectors
x = tf.placeholder(tf.float32, shape=[None, 784])

# variables to be learned
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# placeholder for correct labels
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# loss to be minimized
# cross-entropy explained at http://colah.github.io/posts/2015-09-Visual-Information/
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# more optimizers at https://www.tensorflow.org/api_guides/python/train#optimizers
learning_rate = 0.5
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# get a list of booleans comparing predicted labels to correct labels
predicted_labels = tf.argmax(y, 1)
correct_labels = tf.argmax(y_, 1)
correct_predictions = tf.equal(predicted_labels, correct_labels)

# determine fraction of correct predictions
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# define and run variable init operation
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

steps = 1000
for step in range(steps):
    # feed batches of 100 random training points to the train_step (stochastic training)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    feed_dict = {
        x: batch_xs,
        y_: batch_ys,
    }

    sess.run(train_step, feed_dict=feed_dict)

    if step % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
        print(f'step {step}, training accuracy {train_accuracy:.3f}')

feed_dict = {
    x: mnist.test.images,
    y_: mnist.test.labels,
}
test_accuracy = sess.run(accuracy, feed_dict=feed_dict)
print(f'test accuracy {test_accuracy:.3f}')
