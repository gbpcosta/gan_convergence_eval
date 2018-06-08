# Based on:
# https://github.com/tensorflow/models/blob/master/research/gan/cifar/util.py

import tensorflow as tf
tfgan = tf.contrib.gan


def get_frechet_inception_distance(real_images, generated_images,
                                   num_batches=100):
    """ Get Frechet Inception Distance between real and generated images.
    Args:
    real_images: Real images minibatch. Shape [batch size, width, height,
      channels. Values are in [-1, 1].
    generated_images: Generated images minibatch. Shape [batch size, width,
      height, channels]. Values are in [-1, 1].
    batch_size: Python integer. Batch dimension.
    num_inception_images: Number of images to run through Inception at once.
    Returns:
    Frechet Inception distance. A floating-point scalar.
    Raises:
    ValueError: If the minibatch size is known at graph construction time, and
      doesn't batch `batch_size`.
    """

    # Resize input images.
    preprocessed_real = tfgan.eval.preprocess_image(real_images)
    preprocessed_g = tfgan.eval.preprocess_image(generated_images)

    # Compute Frechet Inception Distance.
    fid = tfgan.eval.frechet_inception_distance(
      preprocessed_real, preprocessed_g,
      num_batches=num_batches)

    return fid
