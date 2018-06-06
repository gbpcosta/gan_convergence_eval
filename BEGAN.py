# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import dataset
from tqdm import tqdm
from utils.utils import check_folder
from utils.utils import save_images

from GAN import GAN

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
plt.switch_backend('Agg')

slim = tf.contrib.slim


class BEGAN(GAN):
    model_name = 'BEGAN'     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 compute_metrics_it, checkpoint_dir, result_dir,
                 log_dir, bot, redo, verbosity):
        super().__init__(sess, epoch, batch_size, z_dim, dataset_name,
                         compute_metrics_it, checkpoint_dir, result_dir,
                         log_dir, bot, redo, verbosity)

        if self.dataset_name in ['mnist', 'fashion-mnist']:
            # BEGAN Parameter
            self.gamma = 0.75
            self.lamda = 0.001

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # architecture hyper parameters
            self.code_dim = 32

        elif self.dataset_name in ['celeba']:
            # BEGAN Parameter
            self.gamma = 0.5
            self.lamda = 0.001

            # train
            self.learning_rate = 0.00001
            self.beta1 = 0.5

            # architecture hyper parameters
            self.repeat_num = int(np.log2(self.input_height)) - 2
            self.hidden_num = 128
            self.data_format = 'NHWC'
            self.code_dim = 64

        else:
            raise NotImplementedError

    def discriminator(self, x, is_training=True, reuse=False):
        if self.dataset_name in ['mnist', 'fashion-mnist']:
            # It must be Auto-Encoder style architecture
            # Architecture : (64)4c2s-FC32_BR-FC64*14*14_BR-(1)4dc2s_S
            with tf.variable_scope('discriminator', reuse=reuse) as vs:
                # net = tf.nn.relu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
                net = slim.conv2d(x, 64, 4, 2,
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                  biases_initializer=tf.constant_initializer(0.0),
                                  activation_fn=tf.nn.relu)

                # net = tf.reshape(net, [self.batch_size, -1])
                net = slim.flatten(net)

                # code = tf.nn.relu(bn(linear(net, 32, scope='d_fc2'),
                #                      is_training=is_training, scope='d_bn2'))
                net = code = slim.batch_norm(
                    slim.fully_connected(net, self.code_dim,
                                         activation_fn=None,
                                         weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                         biases_initializer=tf.constant_initializer(0.0)),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.relu)

                # net = tf.nn.relu(
                #   bn(linear(code, 64 * 14 * 14, scope='d_fc3'),
                #      is_training=is_training, scope='d_bn3'))
                net = slim.batch_norm(
                    slim.fully_connected(
                        net,
                        64 *
                        (self.input_width // 2) *
                        (self.input_height // 2),
                        activation_fn=None,
                        weights_initializer=tf.random_normal_initializer(stddev=0.02),
                        biases_initializer=tf.constant_initializer(0.0)),
                    is_training=is_training,
                    center=True, scale=True,
                    epsilon=1e-5,
                    decay=0.9,
                    updates_collections=None,
                    activation_fn=tf.nn.relu)

                # net = tf.reshape(net, [self.batch_size, 14, 14, 64])
                net = tf.reshape(net,
                                 [-1,
                                  (self.input_width // 2),
                                  (self.input_height // 2),
                                  64])

                # out = tf.nn.sigmoid(deconv2d(net,
                #   [self.batch_size, 28, 28, 1],
                #   4, 4, 2, 2, name='d_dc4'))
                out = slim.conv2d_transpose(net, self.c_dim, 4, 2,
                                            weights_initializer=tf.random_normal_initializer(stddev=0.02),
                                            biases_initializer=tf.constant_initializer(0.0),
                                            activation_fn=tf.nn.sigmoid)

                d_vars = tf.contrib.framework.get_variables(vs)
                return out, code, d_vars

        elif self.dataset_name in ['celeba']:
            # Architecture from: https://git.io/vhqBv
            with tf.variable_scope('discriminator', reuse=reuse) as vs:
                # Encoder
                net = slim.conv2d(x, self.hidden_num, 3, 1,
                                  activation_fn=tf.nn.elu,
                                  data_format=self.data_format)

                prev_channel_num = self.hidden_num
                for idx in range(self.repeat_num):
                    channel_num = self.hidden_num * (idx + 1)

                    net = slim.conv2d(net, channel_num, 3, 1,
                                      activation_fn=tf.nn.elu,
                                      data_format=self.data_format)

                    net = slim.conv2d(net, channel_num, 3, 1,
                                      activation_fn=tf.nn.elu,
                                      data_format=self.data_format)

                    if idx < self.repeat_num - 1:
                        net = slim.conv2d(net, channel_num, 3, 2,
                                          activation_fn=tf.nn.elu,
                                          data_format=self.data_format)

                net = tf.reshape(net,
                                 [-1,
                                  np.prod(
                                    [(self.input_width // 8),
                                     (self.input_height // 8),
                                     channel_num])])
                code = net = slim.fully_connected(net, self.code_dim,
                                                  activation_fn=None)

                # Decoder
                num_output = int(np.prod(
                    [(self.input_width // 8),
                     (self.input_height // 8),
                     self.hidden_num]))

                net = slim.fully_connected(net, num_output, activation_fn=None)
                net = tf.reshape(net,
                                 [-1,
                                  (self.input_width // 8),
                                  (self.input_height // 8),
                                  self.hidden_num])

                for idx in range(self.repeat_num):
                    net = slim.conv2d(net, self.hidden_num, 3, 1,
                                      activation_fn=tf.nn.elu,
                                      data_format=self.data_format)

                    net = slim.conv2d(net, self.hidden_num, 3, 1,
                                      activation_fn=tf.nn.elu,
                                      data_format=self.data_format)

                    if idx < self.repeat_num - 1:
                        # upscale image (height and width) by a factor of 2
                        net = \
                            tf.image.resize_nearest_neighbor(
                                net,
                                tf.multiply(tf.shape(net)[1:3], (2, 2)))

                out = slim.conv2d(net, self.c_dim, 3, 1, activation_fn=None,
                                  data_format=self.data_format)

                d_vars = tf.contrib.framework.get_variables(vs)
                return out, code, d_vars
        else:
            raise NotImplementedError

    def generator(self, z, is_training=True, reuse=False):
        if self.dataset_name in ['mnist', 'fashion-mnist']:
            # Network Architecture is exactly same as in infoGAN
            # (https://arxiv.org/abs/1606.03657)
            # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
            with tf.variable_scope('generator', reuse=reuse) as vs:
                # net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'),
                #                     is_training=is_training, scope='g_bn1'))
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

                # net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'),
                #                  is_training=is_training, scope='g_bn2'))
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

                # net = tf.reshape(net, [self.batch_size, 7, 7, 128])
                net = tf.reshape(net,
                                 [-1,
                                  (self.input_width // 4),
                                  (self.input_height // 4),
                                  128])

                # net = tf.nn.relu(
                #     bn(deconv2d(net,
                #                 [self.batch_size, 14, 14, 64], 4, 4, 2, 2,
                #                 name='g_dc3'),
                #         is_training=is_training, scope='g_bn3'))
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

                # out = tf.nn.sigmoid(deconv2d(net,
                #   [self.batch_size, 28, 28, 1],
                #                              4, 4, 2, 2, name='g_dc4'))
                out = slim.conv2d_transpose(
                    net, self.c_dim, 4, 2,
                    data_format=self.data_format,
                    weights_initializer=tf.random_normal_initializer(stddev=0.02),
                    biases_initializer=tf.constant_initializer(0.0),
                    activation_fn=tf.nn.sigmoid)

                g_vars = tf.contrib.framework.get_variables(vs)
                return out, g_vars

        elif self.dataset_name in ['celeba']:
            # Architecture from: https://git.io/vhqBv
            with tf.variable_scope('generator', reuse=reuse) as vs:
                num_output = int(np.prod(
                    [(self.input_width // 8),
                     (self.input_height // 8),
                     self.hidden_num]))
                net = slim.fully_connected(z, num_output, activation_fn=None)
                net = tf.reshape(net,
                                 [-1,
                                  (self.input_width // 8),
                                  (self.input_height // 8),
                                  self.hidden_num])

                for idx in range(self.repeat_num):
                    net = slim.conv2d(net, self.hidden_num, 3, 1,
                                      activation_fn=tf.nn.elu,
                                      data_format=self.data_format)

                    net = slim.conv2d(net, self.hidden_num, 3, 1,
                                      activation_fn=tf.nn.elu,
                                      data_format=self.data_format)

                    if idx < self.repeat_num - 1:
                        # upscale image (height and width) by a factor of 2
                        net = \
                            tf.image.resize_nearest_neighbor(
                                net,
                                tf.multiply(tf.shape(net)[1:3], (2, 2)))

                out = slim.conv2d(net, self.c_dim, 3, 1, activation_fn=None,
                                  data_format=self.data_format)

                g_vars = tf.contrib.framework.get_variables(vs)
                return out, g_vars
        else:
            raise NotImplementedError

    def define_loss_fn(self):
        ''' Loss Function '''

        # output of D for fake images
        G, self.g_vars = self.generator(self.z, is_training=True, reuse=False)

        # D_out, D_code, d_vars = \
        #     self.discriminator(tf.concat([self.inputs, G], axis=0),
        #                        is_training=True, reuse=False)
        #
        # self.D_real_img, self.D_fake_img = \
        #     tf.split(self.ds.denorm_img(D_out), 2)
        # self.D_real_code, self.D_fake_code = \
        #     tf.split(D_code, 2)
        #
        # D_real_err = tf.reduce_mean(
        #     tf.abs(self.D_real_img -
        #            self.ds.denorm_img(self.inputs)))
        # D_fake_err = tf.reduce_mean(
        #     tf.abs(self.D_fake_img -
        #            self.ds.denorm_img(G)))

        self.D_real_img, self.D_real_code, self.d_vars = \
            self.discriminator(self.inputs,
                               is_training=True, reuse=False)
        self.D_fake_img, self.D_fake_code, _ = \
            self.discriminator(G,
                               is_training=True, reuse=True)

        D_real_err = \
            tf.sqrt(2 *
                    tf.nn.l2_loss(self.D_real_img -
                                  self.inputs)) \
            / self.batch_size
        D_fake_err = \
            tf.sqrt(2 *
                    tf.nn.l2_loss(self.D_fake_img -
                                  G)) \
            / self.batch_size

        # get loss for discriminator
        self.d_loss = D_real_err - self.k * D_fake_err

        # get loss for generator
        self.g_loss = D_fake_err

        # convergence metric
        self.M = D_real_err + tf.abs(self.gamma * D_real_err - D_fake_err)

        # operation for updating k
        self.update_k = self.k.assign(
            self.k + self.lamda *
            (self.gamma * D_real_err - D_fake_err))

        ''' Summary '''
        d_loss_real_sum = tf.summary.scalar('d_error_real', D_real_err)
        d_loss_fake_sum = tf.summary.scalar('d_error_fake', D_fake_err)
        d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
        g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        M_sum = tf.summary.scalar('M', self.M)
        k_sum = tf.summary.scalar('k', self.k)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.p_sum = tf.summary.merge([M_sum, k_sum])

    def define_optimizers(self):
        ''' Training '''
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
                tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)\
                .minimize(self.g_loss, var_list=self.g_vars)

    def build_model(self):
        ''' BEGAN variable '''
        self.k = tf.Variable(0., trainable=False)

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
        plot_M = []
        plot_k_value = []
        plot_logMMD = []
        plot_inception_score = []
        first_it = counter

        # loop for epoch
        start_time = time.time()

        for epoch in tqdm(range(start_epoch, self.epoch), position=1):
            batch_number = 0

            pbar = tqdm(total=self.num_batches, position=0)

            self.sess.run(self.training_init_op)
            while True:
                try:
                    # update D and G networks
                    _, summary_str_d, d_loss, \
                        _, summary_str_g, g_loss, \
                        _, summary_str_k, M_value, k_value = \
                        self.sess.run([
                            self.d_optim, self.d_sum, self.d_loss,
                            self.g_optim, self.g_sum, self.g_loss,
                            self.update_k, self.p_sum, self.M, self.k])
                    self.writer.add_summary(summary_str_d, counter)
                    self.writer.add_summary(summary_str_g, counter)
                    self.writer.add_summary(summary_str_k, counter)

                    plot_d_loss.append(d_loss)
                    plot_g_loss.append(g_loss)
                    plot_M.append(M_value)
                    plot_k_value.append(k_value)

                    # display training status
                    counter += 1
                    pbar.update(1)
                    batch_number += 1

                    if np.mod(counter, self.compute_metrics_it) == 0:
                        plot_logMMD.append(self.compute_mmd()[0])

                        inception_mean, inception_std = \
                            self.compute_inception_score()
                        plot_inception_score.append(inception_mean)

                    if self.verbosity >= 4:
                        print('Epoch: [%2d] [%4d / %4d] time: %4.4f,'
                              ' d_loss: %.8f, g_loss: %.8f'
                              % (epoch, batch_number, self.num_batches,
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
                      ' d_loss: %.8f, g_loss: %.8f,'
                      ' M: %.8f, k: %.8f'
                      % (epoch, time.time() - start_time,
                         np.mean(plot_d_loss[-self.batch_size:]),
                         np.mean(plot_g_loss[-self.batch_size:]),
                         np.mean(plot_M[-self.batch_size:]),
                         k_value))

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
                 plot_M,
                 plot_k_value,
                 plot_logMMD,
                 plot_inception_score],
                [list(range(first_it, counter)),
                 list(range(first_it, counter)),
                 list(range(first_it, counter)),
                 metrics_its,
                 metrics_its],
                metric_names=[('Discriminator loss', 'Generator loss'),
                              'M',
                              'k',
                              'log(MMD)',
                              'Inception Score'],
                n_cols=2,
                legend=[True, False, False, False, False],
                x_label='Iteration',
                y_label=['Loss', 'M Value', 'k Value',
                         'log(MMD)', 'Inception Score'],
                fig_wsize=22, fig_hsize=16)

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)
