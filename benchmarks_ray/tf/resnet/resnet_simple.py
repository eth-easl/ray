
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import ray
import tensorflow as tf

import cifar_input
import resnet_model

def get_data(path, size, dataset):
    # Retrieves all preprocessed images and labels using a tensorflow queue.
    # This only uses the cpu.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with tf.device("/cpu:0"):
        dataset = cifar_input.build_data(path, size, dataset)
        sess = tf.Session()
        images, labels = sess.run(dataset)
        sess.close()
        return images, labels


def init(data, dataset, num_gpus):
        if num_gpus > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ["0"]
        hps = resnet_model.HParams(
            batch_size=128,
            num_classes=100 if dataset == "cifar100" else 10,
            min_lrn_rate=0.0001,
            lrn_rate=0.1,
            num_residual_units=5,
            use_bottleneck=False,
            weight_decay_rate=0.0002,
            relu_leakiness=0.1,
            optimizer="mom",
            num_gpus=num_gpus)

        # We seed each actor differently so that each actor operates on a
        # different subset of data.
        # if num_gpus > 0:
        #     tf.set_random_seed(ray.get_gpu_ids()[0] + 1)
        # else:
        #     # Only a single actor in this case.
        tf.set_random_seed(1)

        with tf.device("/gpu:0" if num_gpus > 0 else "/cpu:0"):
            # Build the model.
            images, labels = cifar_input.build_input(data, hps.batch_size,
                                                     dataset, False)
            model = resnet_model.ResNet(hps, images, labels, "train")
            model.build_graph()
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            #self.model.variables.set_session(sess)
            tf.compat.v1.keras.backend.set_session(sess)
            init = tf.global_variables_initializer()
            sess.run(init)
            steps = 10


train_data = get_data(FLAGS.train_data_path, 50000, FLAGS.dataset)
test_data = get_data(FLAGS.eval_data_path, 10000, FLAGS.dataset)


init(train_data, "cifar-10", 1)