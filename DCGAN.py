# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import dataset
from tqdm import tqdm, trange
from utils.utils import check_folder
from utils.utils import save_images

from GAN import GAN

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
plt.switch_backend("Agg")

slim = tf.contrib.slim


class DCGAN(GAN):
    model_name = "DCGAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 compute_metrics_it, checkpoint_dir, result_dir,
                 log_dir, bot, redo, verbosity):
        super().__init__(sess, epoch, batch_size, z_dim, dataset_name,
                         compute_metrics_it, checkpoint_dir, result_dir,
                         log_dir, bot, redo, verbosity)

        if self.dataset_name in ['mnist', 'fashion-mnist']:
            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

        elif self.dataset_name in ['celeba']:
            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # architecture hyper parameters
            self.repeat_num = int(np.log2(self.input_height)) - 2
            self.hidden_num = 128
            self.data_format = 'NHWC'

        else:
            raise NotImplementedError

    def discriminator(self, x, is_training=True, reuse=False):
        if self.dataset_name in ['mnist', 'fashion-mnist']:
            # Network Architecture is exactly same as in infoGAN
            # (https://arxiv.org/abs/1606.03657)
            # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
            with tf.variable_scope('discriminator', reuse=reuse) as vs:

                net = slim.conv2d(x, 64, 4, 2,
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                  biases_initializer=tf.constant_initializer(0.0),
                                  activation_fn=tf.nn.leaky_relu)

                net = slim.batch_norm(
                    slim.conv2d(net, 128, 4, 2, activation_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                biases_initializer=tf.constant_initializer(0.0)),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.leaky_relu)

                net = slim.flatten(net)

                net = slim.batch_norm(
                    slim.fully_connected(net, 1024,
                                         activation_fn=None,
                                         weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0)),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.leaky_relu)

                out_logit = \
                    slim.fully_connected(net, 1,
                                         activation_fn=None,
                                         weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))

                out = tf.nn.sigmoid(out_logit)

                d_vars = tf.contrib.framework.get_variables(vs)
                return out, out_logit, d_vars

        if self.dataset_name in ['celeba']:

            with tf.variable_scope('discriminator', reuse=reuse) as vs:

                net = slim.conv2d(x, 64, 5, 2,
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                  biases_initializer=tf.constant_initializer(0.0),
                                  activation_fn=tf.nn.leaky_relu,
                                  data_format=self.data_format)

                net = slim.batch_norm(
                    slim.conv2d(net, 128, 5, 2,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                biases_initializer=tf.constant_initializer(0.0),
                                activation_fn=tf.nn.leaky_relu,
                                data_format=self.data_format),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.leaky_relu)

                net = slim.batch_norm(
                    slim.conv2d(net, 256, 5, 2,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                biases_initializer=tf.constant_initializer(0.0),
                                activation_fn=tf.nn.leaky_relu,
                                data_format=self.data_format),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.leaky_relu)

                net = slim.batch_norm(
                    slim.conv2d(net, 512, 5, 2,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                biases_initializer=tf.constant_initializer(0.0),
                                activation_fn=tf.nn.leaky_relu,
                                data_format=self.data_format),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.leaky_relu)

                net = slim.flatten(net)

                out_logit = \
                    slim.fully_connected(net, 1,
                                         activation_fn=None,
                                         weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0))

                out = tf.nn.sigmoid(out_logit)

                d_vars = tf.contrib.framework.get_variables(vs)
                return out, out_logit, d_vars
        else:
            raise NotImplementedError

    def generator(self, z, is_training=True, reuse=False):
        if self.dataset_name in ['mnist', 'fashion-mnist']:
            # Network Architecture is exactly same as in infoGAN
            # (https://arxiv.org/abs/1606.03657)
            # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
            with tf.variable_scope('generator', reuse=reuse) as vs:

                net = slim.batch_norm(
                    slim.fully_connected(z, 1024, activation_fn=None,
                                         weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0)),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.relu)

                net = slim.batch_norm(
                    slim.fully_connected(net,
                                         128 *
                                         (self.input_width // 4) *
                                         (self.input_height // 4),
                                         activation_fn=None,
                                         weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0)),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.relu)

                net = tf.reshape(net,
                                 [-1,
                                  (self.input_width // 4),
                                  (self.input_height // 4),
                                  128])

                net = slim.batch_norm(
                    slim.conv2d_transpose(
                        net, 64, 4, 2,
                        activation_fn=None,
                        data_format=self.data_format,
                        weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        biases_initializer=tf.constant_initializer(0.0)),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.relu)

                out = slim.conv2d_transpose(
                    net, self.c_dim, 4, 2,
                    activation_fn=tf.nn.sigmoid,
                    data_format=self.data_format,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    biases_initializer=tf.constant_initializer(0.0))

                g_vars = tf.contrib.framework.get_variables(vs)
                return out, g_vars

        elif self.dataset_name in ['celeba']:
            with tf.variable_scope('generator', reuse=reuse) as vs:

                net = slim.fully_connected(z, 1024 * 4 * 4,
                                           activation_fn=tf.nn.relu,
                                           weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                           biases_initializer=tf.constant_initializer(0.0))

                net = tf.reshape(net, [-1, 4, 4, 1024])

                net = slim.batch_norm(
                    slim.conv2d_transpose(net, 512, 5, 2,
                                          activation_fn=None,
                                          weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                          biases_initializer=tf.constant_initializer(0.0)),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.relu)

                net = slim.batch_norm(
                    slim.conv2d_transpose(net, 256, 5, 2,
                                          activation_fn=None,
                                          weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                          biases_initializer=tf.constant_initializer(0.0)),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.relu)

                net = slim.batch_norm(
                    slim.conv2d_transpose(net, 128, 5, 2,
                                          activation_fn=None,
                                          weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                          biases_initializer=tf.constant_initializer(0.0)),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.relu)

                out = slim.batch_norm(
                    slim.conv2d_transpose(net, self.c_dim, 4, 2,
                                          activation_fn=None,
                                          weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                          biases_initializer=tf.constant_initializer(0.0)),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.tanh)

                g_vars = tf.contrib.framework.get_variables(vs)
                return out, g_vars

        else:
            raise NotImplementedError

    def define_loss_fn(self):
        """ Loss Function """

        # output of D for real and fake images
        G, self.g_vars = self.generator(self.z, is_training=True, reuse=False)

        D_real, D_real_logits, self.d_vars = \
            self.discriminator(self.inputs,
                               is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = \
            self.discriminator(G,
                               is_training=True, reuse=True)

        # D_out, D_logits, d_vars = \
        #     self.discriminator(tf.concat([self.inputs, G], axis=0),
        #                        is_training=True, reuse=False)
        #
        # D_real, D_fake = \
        #     tf.split(self.ds.denorm_img(D_out), 2)
        # D_real_logits, D_fake_logits = \
        #     tf.split(self.ds.denorm_img(D_logits), 2)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_real_logits, labels=tf.ones_like(D_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_fake_logits, labels=tf.zeros_like(D_fake)))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        """ Summary """
        d_loss_real_sum = tf.summary.scalar('d_loss_real', d_loss_real)
        d_loss_fake_sum = tf.summary.scalar('d_loss_fake', d_loss_fake)
        d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
        g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def define_optimizers(self):
        """ Training """
        # divide trainable variables into a group for D and a group for G
        # t_vars = tf.trainable_variables()
        # d_vars = [var for var in t_vars if 'd_' in var.name]
        # g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            self.d_optim = \
                tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=self.d_vars)
            self.g_optim = \
                tf.train.AdamOptimizer(self.learning_rate,
                                       beta1=self.beta1) \
                .minimize(self.g_loss, var_list=self.g_vars)

    def build_model(self):
        self.define_input()
        self.define_loss_fn()
        self.define_optimizers()
        self.define_test_sample()
        self.define_mmd_comp()
        self.define_inception_score_input()

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' +
                                            self.model_name, self.sess.graph)

        # restore check-point if it exits
        start_epoch, start_batch_id, counter = self.verify_checkpoint()

        # plot variables
        plot_d_loss = []
        plot_g_loss = []
        plot_logMMD = []
        plot_inception_score = []
        first_it = counter

        # loop for epoch
        start_time = time.time()

        for epoch in trange(start_epoch, self.epoch, position=1):
            batch_number = 0

            pbar = tqdm(total=self.num_batches, position=0)

            self.sess.run([self.training_init_op])
            while True:
                try:
                    # update D and G networks
                    _, summary_str_d, d_loss, \
                        _, summary_str_g, g_loss = \
                        self.sess.run([
                            self.d_optim, self.d_sum, self.d_loss,
                            self.g_optim, self.g_sum, self.g_loss])
                    self.writer.add_summary(summary_str_d, counter)
                    self.writer.add_summary(summary_str_g, counter)

                    plot_d_loss.append(d_loss)
                    plot_g_loss.append(g_loss)

                    # update training status
                    counter += 1
                    batch_number += 1
                    pbar.update(1)

                    if np.mod(counter, self.compute_metrics_it) == 0:
                        plot_logMMD.append(self.compute_mmd()[0])

                        inception_mean, inception_std = \
                            self.compute_inception_score()
                        plot_inception_score.append(inception_mean)

                    if self.verbosity >= 4:
                        print('Epoch: [%2d] [%4d] time: %4.4f,'
                              ' d_loss: %.8f, g_loss: %.8f'
                              % (epoch, batch_number,
                                 time.time() - start_time,
                                 d_loss, g_loss))

                    # save test results for every 300 steps
                    if self.verbosity >= 3 and \
                       self.dataset_name in \
                            ['mnist', 'fashion-mnist', 'celeba'] and \
                       np.mod(counter, 300) == 0:
                        self.save_test_sample(epoch, batch_number)

                except tf.errors.OutOfRangeError:
                    pbar.close()
                    break

            if self.verbosity >= 2:
                print('Epoch [%02d]: time: %4.4f,'
                      ' d_loss: %.8f, g_loss: %.8f'
                      % (epoch, time.time() - start_time,
                         np.mean(plot_d_loss[-self.batch_size:]),
                         np.mean(plot_g_loss[-self.batch_size:])))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading
            # pre-trained model
            start_batch_id = 0

            # plot loss and evaluation metrics
            metrics_its = list(range(
               first_it // self.compute_metrics_it,
               counter,
               self.compute_metrics_it))[1:]

            self.plot_metrics(
                [(plot_d_loss, plot_g_loss),
                 plot_logMMD,
                 plot_inception_score],
                iterations_list=[list(range(first_it, counter)),
                                 metrics_its,
                                 metrics_its],
                metric_names=[('Discriminator loss', 'Generator loss'),
                              'log(MMD)',
                              'Inception Score'],
                n_cols=1,
                legend=[True, False, False],
                x_label=['Iteration', 'Iteration', 'Iteration'],
                y_label=['Loss', 'log(MMD)', 'Inception Score (Average)'])

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        # self.save(self.checkpoint_dir, counter)
