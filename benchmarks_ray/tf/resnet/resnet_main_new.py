"""ResNet training script, with some code from
https://github.com/tensorflow/models/tree/master/resnet.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import ray
import tensorflow as tf

import test_model

import time



# Tensorflow must be at least version 1.2.0 for the example to work.
# tf_major = int(tf.__version__.split(".")[0])
# tf_minor = int(tf.__version__.split(".")[1])
# if (tf_major < 1) or (tf_major == 1 and tf_minor < 2):
#     raise Exception("Your Tensorflow version is less than 1.2.0. Please "
#                     "update Tensorflow to the latest version.")

parser = argparse.ArgumentParser(description="Run the ResNet example.")
parser.add_argument(
    "--dataset",
    default="cifar10",
    type=str,
    help="Dataset to use: cifar10 or cifar100.")
parser.add_argument(
    "--train_data_path",
    default="/home/ubuntu/resnet_tf1/cifar-10-batches-bin/data_batch*",
    type=str,
    help="Data path for the training data.")
parser.add_argument(
    "--eval_data_path",
    default="/home/ubuntu/resnet_tf1/cifar-10-batches-bin/test_batch.bin",
    type=str,
    help="Data path for the testing data.")
parser.add_argument(
    "--eval_dir",
    default="/tmp/resnet-model/eval",
    type=str,
    help="Data path for the tensorboard logs.")
parser.add_argument(
    "--eval_batch_count",
    default=50,
    type=int,
    help="Number of batches to evaluate over.")
parser.add_argument(
    "--num_gpus",
    default=0,
    type=int,
    help="Number of GPUs to use for training.")
parser.add_argument(
    "--redis-address",
    default=None,
    type=str,
    help="The Redis address of the cluster.")

FLAGS = parser.parse_args()

# Determines if the actors require a gpu or not.
use_gpu = 1 if int(FLAGS.num_gpus) > 0 else 0


# @ray.remote
# def get_data(path, size, dataset):
#     import tensorflow as tf

#     # Retrieves all preprocessed images and labels using a tensorflow queue.
#     # This only uses the cpu.
#     #os.environ["CUDA_VISIBLE_DEVICES"] = ""
#     images=[]
#     labels=[]
#     with tf.device("/cpu:0"):
#         dataset = cifar_input.build_data(path, size, dataset)
#         sess = tf.Session()
#         images, labels = sess.run(dataset)
#         sess.close()
#         return images, labels

@ray.remote(num_cpus=1)
class ParameterServer(object):
     def __init__(self):
        import tensorflow as tf
        with tf.device("/cpu:0"):
            self.model = test_model.TFBenchModel()

    def apply_gradients(self, *gradients):
        self.model.apply_gradients(np.mean(gradients, axis=0))
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()


@ray.remote(num_gpus=use_gpu)
class ResNetTrainActor(object):
    def __init__(self, num_gpus, ind):
        import tensorflow as tf
        # if num_gpus > 0:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        #         [str(i) for i in ray.get_gpu_ids()])

        # We seed each actor differently so that each actor operates on a
        # different subset of data.
        # if num_gpus > 0:
        #     tf.set_random_seed(ray.get_gpu_ids()[0] + 1)
        # else:
            # Only a single actor in this case.
        tf.set_random_seed(ind+1)

        with tf.device("/gpu:0" if num_gpus > 0 else "/cpu:0"):
            # Build the model.
            # images, labels = cifar_input.build_input(data, hps.batch_size,
            #                                          dataset, False)
            self.model = test_model.TFBenchModel()
            self.steps = 10

    def compute_gradients(self, weights):
        # This method sets the weights in the network, trains the network
        # self.steps times, and returns the new weights.

        #print("------ weights: ", weights)
        self.model.set_weights(weights)
        #for i in range(self.steps):
            #self.model.variables.sess.run(self.model.train_op)
        #    self.model.variables.sess.run(self.model.get_loss())
        #return self.model.get_weights()
        # TODO: need to do anything with batches here?
        ret = self.model.compute_gradients(self.model.get_feed_dict())
        return ret

    def get_weights(self):
        # Note that the driver cannot directly access fields of the class,
        # so helper methods must be created.
        print("hello!")
        #weights = self.model.variables.get_flat()
        #print(weights)
        #return list(weights.keys()), list(weights.values())
        #return weights
        return self.model.get_weights()




'''
@ray.remote
class ResNetTestActor(object):
    def __init__(self, data, dataset, eval_batch_count, eval_dir):
        import tensorflow as tf
        #os.environ["CUDA_VISIBLE_DEVICES"] = ""
        hps = resnet_model.HParams(
            batch_size=100,
            num_classes=100 if dataset == "cifar100" else 10,
            min_lrn_rate=0.0001,
            lrn_rate=0.1,
            num_residual_units=5,
            use_bottleneck=False,
            weight_decay_rate=0.0002,
            relu_leakiness=0.1,
            optimizer="mom",
            num_gpus=0)
        with tf.device("/cpu:0"):
            # Builds the testing network.
            images, labels = cifar_input.build_input(data, hps.batch_size,
                                                     dataset, False)
            self.model = resnet_model.ResNet(hps, images, labels, "eval")
            self.model.build_graph()
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            self.model.variables.set_session(sess)
            init = tf.global_variables_initializer()
            sess.run(init)

            # Initializing parameters for tensorboard.
            self.best_precision = 0.0
            self.eval_batch_count = eval_batch_count
            self.summary_writer = tf.summary.FileWriter(eval_dir, sess.graph)
        # The IP address where tensorboard logs will be on.
        self.ip_addr = ray.services.get_node_ip_address()

    def accuracy(self, weights, train_step):
        # Sets the weights, computes the accuracy and other metrics
        # over eval_batches, and outputs to tensorboard.
        import tensorflow as tf
        self.model.variables.set_weights(weights)
        total_prediction, correct_prediction = 0, 0
        model = self.model
        sess = self.model.variables.sess
        for _ in range(self.eval_batch_count):
            summaries, loss, predictions, truth = sess.run(
                [model.summaries, model.cost, model.predictions, model.labels])

            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        self.best_precision = max(precision, self.best_precision)
        precision_summ = tf.Summary()
        precision_summ.value.add(tag="Precision", simple_value=precision)
        self.summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(
            tag="Best Precision", simple_value=self.best_precision)
        self.summary_writer.add_summary(best_precision_summ, train_step)
        self.summary_writer.add_summary(summaries, train_step)
        tf.logging.info("loss: %.3f, precision: %.3f, best precision: %.3f" %
                        (loss, precision, self.best_precision))
        self.summary_writer.flush()
        return precision

    def get_ip_addr(self):
        # As above, a helper method must be created to access the field from
        # the driver.
        return self.ip_addr
'''

def train():
    num_gpus = FLAGS.num_gpus
    ray.init(address="auto")

    # if FLAGS.redis_address is None:
    #     ray.init(num_gpus=num_gpus)
    # else:
    #     ray.init(redis_address=FLAGS.redis_address)
    #train_data = get_data.remote(FLAGS.train_data_path, 10000, FLAGS.dataset)
    #test_data = get_data.remote(FLAGS.eval_data_path, 1000, FLAGS.dataset)
    # Creates an actor for each gpu, or one if only using the cpu. Each actor
    # has access to the dataset.
    #if FLAGS.num_gpus > 0:

    num_actors=1

    train_actors = [
        ResNetTrainActor.remote(num_gpus, i)
        for i in range(num_actors)
    ]
    ps = ParameterServer.remote()

    weight_id = ps.get_weights.remote()

    #else:
    #    train_actors = [ResNetTrainActor.remote(train_data, FLAGS.dataset, 0)]
    #test_actor = ResNetTestActor.remote(test_data, FLAGS.dataset,
    #                                    FLAGS.eval_batch_count, FLAGS.eval_dir)
    # print("The log files for tensorboard are stored at ip {}.".format(
    #     ray.get(test_actor.get_ip_addr.remote())))
    step = 0
    #weight_id = train_actors[0].get_weights.remote()
    #acc_id = test_actor.accuracy.remote(weight_id, step)
    # Correction for dividing the weights by the number of gpus.
    #print(ray.get(weight_id))


    if num_gpus == 0:
        num_gpus = 1
    print("Starting training loop. Use Ctrl-C to exit.")
    #time.sleep(60)
    try:

        # TODO: make this PS style - also make sure different inputs are being used
        while (step<100):
            print("step: ", step)
            all_gradients = ray.get([
                actor.compute_gradients.remote(weight_id) for actor in train_actors
            ])

            #print(all_gradients)
            current_weights = ps.apply_gradients.remote(*all_gradients)

            '''
            mean_weights = {
                k: (sum(weights[k] for weights in all_weights) / num_gpus)
                for k in all_weights[0]
            }
            weight_id = ray.put(mean_weights)
            '''

            step += 1
            if step % 10 == 0:
                # Retrieves the previously computed accuracy and launches a new
                # testing task with the current weights every 200 steps.
                #acc = ray.get(acc_id)
                #acc_id = test_actor.accuracy.remote(weight_id, step)
                acc=0
                print("Step {}: {:.6f}".format(step , acc))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    train()
