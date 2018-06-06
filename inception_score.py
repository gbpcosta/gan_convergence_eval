'''
From https://github.com/tsc2017/inception-score
Code derived from
https://github.com/openai/improved-gan/blob/master/inception_score/model.py

Args:
    images: A numpy array with values ranging from -1 to 1 and shape in the
            form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be
            arbitrary.
    splits: The number of splits of the images, default is 10.
Returns:
    mean and standard deviation of the inception across the splits.
'''

import tensorflow as tf
import os
import sys
import functools
import numpy as np
import math
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
tfgan = tf.contrib.gan


def inception_logits(batch, num_splits=1):
    with tf.variable_scope("inception", reuse=tf.AUTO_REUSE):
        preprocessed_images = tfgan.eval.preprocess_image(batch)
        logits = tfgan.eval.run_inception(preprocessed_images)
    return logits


def get_inception_probs(images, batch_size=50, gpu_id='0'):
    ds_images = \
        tf.data.Dataset.from_tensor_slices((images)) \
        .shuffle(5000) \
        .apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    ds_images_iterator = ds_images.make_initializable_iterator()

    ds_images_init_op = ds_images_iterator.initializer
    ds_images_next = ds_images_iterator.get_next()

    logits = inception_logits(ds_images_next)

    gpu_options = tf.GPUOptions(visible_device_list=gpu_id,
                                allow_growth=True)
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
        gpu_options=gpu_options,
        allow_soft_placement=True)

    with tf.Session(config=session_conf) as sess:
        sess.run(ds_images_init_op)
        preds = []
        while True:
            start_time = time.time()
            try:
                pred = sess.run(logits)
                preds.append(pred)
            except tf.errors.OutOfRangeError:
                preds = np.concatenate(preds, 0)
                preds = np.exp(preds)/np.sum(np.exp(preds), 1, keepdims=True)
                break

    return preds


def compute_score(logits):
    kl = logits * \
        (tf.log(logits) - tf.log(
            tf.expand_dims(tf.reduce_mean(logits, axis=0), axis=0)))
    kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
    return tf.exp(kl)


def logits2score(logits, batch_size=10000, gpu_id='0'):
    ds_logits = \
        tf.data.Dataset.from_tensor_slices((logits)) \
        .shuffle(50000) \
        .apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # .batch(batch_size)

    ds_logits_iterator = ds_logits.make_initializable_iterator()

    ds_logits_init_op = ds_logits_iterator.initializer
    ds_logits_next = ds_logits_iterator.get_next()

    scores = compute_score(ds_logits_next)

    gpu_options = tf.GPUOptions(visible_device_list=gpu_id,
                                allow_growth=True)
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
        gpu_options=gpu_options,
        allow_soft_placement=True)

    with tf.Session(config=session_conf) as sess:
        sess.run(ds_logits_init_op)
        inception_scores = []
        while True:
            start_time = time.time()
            try:
                inception_score = sess.run(scores)
                inception_scores.append(inception_score)
            except tf.errors.OutOfRangeError:
                # inception_scores = np.array(inception_scores)
                # print("TESTE ", inception_scores.shape)
                break

    return np.mean(inception_scores), np.std(inception_scores)


def get_inception_score(images):
    assert(type(images) == np.ndarray)

    logits = get_inception_probs(images)
    mean, std = logits2score(logits)
    # Reference values: 11.34 for 49984 CIFAR-10 training set images,
    # or mean=11.31, std=0.08 if in 10 splits (default).
    return mean, std
