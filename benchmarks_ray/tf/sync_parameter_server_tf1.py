from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import ray
import time

from tensorflow.keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data


import argparse
import numpy as np

import ray
import ray.experimental.tf_utils

from tensorflow.keras.datasets import mnist

import ray.services

ray.init(address="auto")


res = ray.cluster_resources()
res_keys = res.keys()
nodes = []

for e in res_keys:
    if ('node' in e):
        nodes.append(e)

local_hostname = ray.services.get_node_ip_address()
driver_node_id = f"node:{local_hostname}"

# # Check to make sure the node id resource exists
assert driver_node_id in nodes
nodes.remove(driver_node_id)
worker_node_id_1 = nodes[0]
worker_node_id_2 = nodes[1]


def download_mnist_retry(seed=0, max_num_retries=20):
    # return mnist.load_data()
    import tensorflow as tf
    for _ in range(max_num_retries):
        try:
            return input_data.read_data_sets(
                "MNIST_data", one_hot=True, seed=seed)
        except tf.errors.AlreadyExistsError:
            time.sleep(1)
    raise Exception("Failed to download MNIST.")



class SimpleCNN(object):
    def __init__(self, learning_rate=1e-4):
        import tensorflow as tf
        with tf.Graph().as_default():

            # Create the model
            self.x = tf.compat.v1.placeholder(tf.float32, [None, 784])

            # Define loss and optimizer
            self.y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])

            # Build the graph for the deep net
            self.y_conv, self.keep_prob = deepnn(self.x)

            with tf.compat.v1.name_scope("loss"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.stop_gradient(self.y_), logits=self.y_conv)
            self.cross_entropy = tf.reduce_mean(input_tensor=cross_entropy)

            with tf.compat.v1.name_scope("adam_optimizer"):
                self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
                self.train_step = self.optimizer.minimize(self.cross_entropy)

            with tf.compat.v1.name_scope("accuracy"):
                correct_prediction = tf.equal(
                    tf.argmax(input=self.y_conv, axis=1), tf.argmax(input=self.y_, axis=1))
                correct_prediction = tf.cast(correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(input_tensor=correct_prediction)

            self.sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(
                    intra_op_parallelism_threads=1,
                    inter_op_parallelism_threads=1))
            self.sess.run(tf.compat.v1.global_variables_initializer())

            # Helper values.

            self.variables = ray.experimental.tf_utils.TensorFlowVariables(
                self.cross_entropy, self.sess)

            self.grads = self.optimizer.compute_gradients(self.cross_entropy)
            self.grads_placeholder = [(tf.compat.v1.placeholder(
                "float", shape=grad[1].get_shape()), grad[1])
                                      for grad in self.grads]
            self.apply_grads_placeholder = self.optimizer.apply_gradients(
                self.grads_placeholder)

    def compute_update(self, x, y):
        # TODO(rkn): Computing the weights before and after the training step
        # and taking the diff is awful.
        weights = self.get_weights()[1]
        self.sess.run(
            self.train_step,
            feed_dict={
                self.x: x,
                self.y_: y,
                self.keep_prob: 0.5
            })
        new_weights = self.get_weights()[1]
        return [x - y for x, y in zip(new_weights, weights)]

    def compute_gradients(self, x, y):
        return self.sess.run(
            [grad[0] for grad in self.grads],
            feed_dict={
                self.x: x,
                self.y_: y,
                self.keep_prob: 0.5
            })

    def apply_gradients(self, gradients):
        feed_dict = {}
        for i in range(len(self.grads_placeholder)):
            feed_dict[self.grads_placeholder[i][0]] = gradients[i]
        self.sess.run(self.apply_grads_placeholder, feed_dict=feed_dict)

    def compute_accuracy(self, x, y):
        return self.sess.run(
            self.accuracy,
            feed_dict={
                self.x: x,
                self.y_: y,
                self.keep_prob: 1.0
            })

    def set_weights(self, variable_names, weights):
        self.variables.set_weights(dict(zip(variable_names, weights)))

    def get_weights(self):
        weights = self.variables.get_weights()
        return list(weights.keys()), list(weights.values())

def deepnn(x):
    import tensorflow as tf

    """deepnn builds the graph for a deep net for classifying digits.

    Args:
        x: an input tensor with the dimensions (N_examples, 784), where 784 is
            the number of pixels in a standard MNIST image.

    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with
            values equal to the logits of classifying the digit into one of 10
            classes (the digits 0-9). keep_prob is a scalar placeholder for the
            probability of dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images
    # are grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.compat.v1.name_scope("reshape"):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.compat.v1.name_scope("conv1"):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.compat.v1.name_scope("pool1"):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.compat.v1.name_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.compat.v1.name_scope("pool2"):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.compat.v1.name_scope("fc1"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.compat.v1.name_scope("dropout"):
        keep_prob = tf.compat.v1.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, 1 - (1 - (keep_prob)))

    # Map the 1024 features to 10 classes, one for each digit
    with tf.compat.v1.name_scope("fc2"):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob

def conv2d(x, W):
    import tensorflow as tf

    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    import tensorflow as tf

    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool2d(
        input=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def weight_variable(shape):
    import tensorflow as tf

    """weight_variable generates a weight variable of a given shape."""
    [init]ial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    import tensorflow as tf

    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

@ray.remote
class ParameterServer(object):
    def __init__(self, learning_rate):
        import tensorflow as tf
        self.net = SimpleCNN(learning_rate=learning_rate)

    def apply_gradients(self, *gradients):
        self.net.apply_gradients(np.mean(gradients, axis=0))
        return self.net.variables.get_flat()

    def get_weights(self):
        return self.net.variables.get_flat()


@ray.remote
class Worker(object):
    def __init__(self, worker_index, batch_size=50):
        import tensorflow as tf
        self.worker_index = worker_index
        self.batch_size = batch_size
        self.mnist = download_mnist_retry(seed=worker_index)
        #print(self.mnist)
        self.net = SimpleCNN()

    def compute_gradients(self, weights):
        import tensorflow as tf
        #print('weight shape:', weights.shape)
        self.net.variables.set_flat(weights)
        xs, ys = self.mnist.train.next_batch(self.batch_size)
        ret = self.net.compute_gradients(xs,ys)
        #print('gradient shape:', len(ret))
        return ret


num_workers = 7
net = SimpleCNN()

local_ps = ParameterServer.options(resources={worker_node_id_1: 0.01})
workers_1 = Worker.options(resources={worker_node_id_1: 0.01})
#workers_2 = Worker.options(resources={worker_node_id_2: 0.01})

ps = local_ps.remote(1e-4 * num_workers)
# Create workers.
workers = [worker_1.remote(worker_index)
               for worker_index in range(int(num_workers))]
#workers += [workers_2.remote(worker_index)
#               for worker_index in range(int(num_workers/2), num_workers)]

# Download MNIST.
mnist = download_mnist_retry()

i = 0
#time.sleep(20)

current_weights = ps.get_weights.remote()
start = time.time()
while (i<1000):
   # Compute and apply gradients.
    gradients = [worker.compute_gradients.remote(current_weights)
                 for worker in workers]
    current_weights = ps.apply_gradients.remote(*gradients)

    if i % 10 == 0:
       # Evaluate the current model.
    #    net.variables.set_flat(ray.get(current_weights))
    #    test_xs, test_ys = mnist.test.next_batch(1000)
    #    accuracy = net.compute_accuracy(test_xs, test_ys)
    #    print("Iteration {}: accuracy is {}".format(i, accuracy))
       print("Elapsed(s): ", time.time()-start)
    i+=1