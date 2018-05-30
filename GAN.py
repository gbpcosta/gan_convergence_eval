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

import mmd
# import inception_score

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
plt.switch_backend("Agg")

slim = tf.contrib.slim


class GAN(object):
    model_name = "GAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 checkpoint_dir, result_dir, log_dir, bot, redo, verbosity):
        self.sess = sess
        self.dataset_name = dataset_name.lower()
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.bot = bot
        self.redo = redo
        self.verbosity = verbosity

        if self.dataset_name in ['mnist', 'fashion-mnist']:
            # parameters
            self.input_height = 28
            self.input_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.ds = dataset.MNIST(self.batch_size)
            self.num_batches = self.ds.N_TRAIN_SAMPLES // self.batch_size

            # architecture hyper parameters
            self.data_format = 'NHWC'

        elif self.dataset_name in ['celeba']:
            # parameters
            self.input_height = 64
            self.input_width = 64

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 3

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load CelebA
            self.ds = dataset.CelebA(self.batch_size)
            self.num_batches = self.ds.N_TRAIN_SAMPLES // self.batch_size

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
            with tf.variable_scope("discriminator", reuse=reuse) as vs:

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
            with tf.variable_scope("generator", reuse=reuse) as vs:

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

    def build_model(self):
        """ Graph Input """

        # images
        # create general iterator
        self.iterator = tf.data.Iterator.from_structure(self.ds.output_types,
                                                        self.ds.output_shapes)

        self.training_init_op = \
            self.iterator.make_initializer(self.ds.train_ds)

        if self.ds.valid_ds is not None:
            self.validation_init_op = \
                self.iterator.make_initializer(self.ds.valid_ds)

        if self.ds.test_ds is not None:
            self.testing_init_op = \
                self.iterator.make_initializer(self.ds.test_ds)

        next_element, _ = self.iterator.get_next()
        self.inputs = next_element

        # noises
        self.z = tf.random_normal([self.batch_size, self.z_dim])

        """ Loss Function """

        # output of D for real and fake images
        G, g_vars = self.generator(self.z, is_training=True, reuse=False)

        D_real, D_real_logits, d_vars = \
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

        """ Training """
        # divide trainable variables into a group for D and a group for G
        # t_vars = tf.trainable_variables()
        # d_vars = [var for var in t_vars if 'd_' in var.name]
        # g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            self.d_optim = \
                tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = \
                tf.train.AdamOptimizer(5*self.learning_rate*5,
                                       beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.sample_z = tf.constant(
            np.random.normal(loc=0.0, scale=1.0,
                             size=(self.batch_size, self.z_dim))
            .astype(np.float32))

        fake_images, _ = self.generator(self.sample_z, is_training=False,
                                        reuse=True)
        self.fake_images = self.ds.denorm_img(fake_images)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

        """ MMD """
        aux_1 = tf.reshape(G, [-1, self.input_width * self.input_height *
                               self.c_dim])

        aux_2 = tf.reshape(self.inputs, [-1, self.input_width *
                                         self.input_height * self.c_dim])

        self.log_mmd = tf.log(mmd.rbf_mmd2(aux_1, aux_2))

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' +
                                            self.model_name, self.sess.graph)

        # restore check-point if it exits
        if not self.redo:
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            if could_load:
                start_epoch = (int)(checkpoint_counter / self.num_batches)
                start_batch_id = \
                    checkpoint_counter - start_epoch * self.num_batches
                counter = checkpoint_counter
                if self.verbosity >= 1:
                    print("[*] Load SUCCESS")
            else:
                start_epoch = 0
                start_batch_id = 0
                counter = 1
                if self.verbosity >= 1:
                    print("[!] Load failed...")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            if self.verbosity >= 1:
                print("[!] Redo!")

        # plot variables
        plot_d_loss = []
        plot_g_loss = []
        plot_logMMD = []
        # plot_inception_score = []
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
                        logMDD_value = \
                        self.sess.run([
                            self.d_optim, self.d_sum, self.d_loss,
                            self.g_optim, self.g_sum, self.g_loss,
                            self.log_mmd])
                    self.writer.add_summary(summary_str_d, counter)
                    self.writer.add_summary(summary_str_g, counter)

                    plot_d_loss.append(d_loss)
                    plot_g_loss.append(g_loss)
                    plot_logMMD.append(logMDD_value)

                    # """ Inception Score """
                    # samples = self.sess.run(self.fake_images)
                    # samples = samples * 255
                    #
                    # if samples.shape[-1] == 1:
                    #     samples = np.tile(samples, reps=3)
                    #
                    # acc_samples_inception_score.extend(list(samples))

                    # display training status
                    counter += 1
                    pbar.update(1)
                    batch_number += 1
                    if self.verbosity >= 4:
                        print("Epoch: [%2d] [%4d] time: %4.4f,"
                              " d_loss: %.8f, g_loss: %.8f"
                              % (epoch, batch_number,
                                 time.time() - start_time,
                                 d_loss, g_loss))

                    # save training results for every 300 steps
                    if self.verbosity >= 3 and \
                       self.dataset_name in \
                            ['mnist', 'fashion-mnist', 'celeba'] and \
                       np.mod(counter, 300) == 0:
                        samples = \
                            self.sess.run(self.fake_images)

                        tot_num_samples = min(self.sample_num, self.batch_size)
                        manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                        manifold_w = int(np.floor(np.sqrt(tot_num_samples)))

                        save_images(
                            samples[:manifold_h * manifold_w, :, :, :],
                            [manifold_h, manifold_w],
                            os.path.join(
                                check_folder(os.path.join(os.getcwd(),
                                                          self.result_dir,
                                                          self.model_dir)),
                                self.model_name +
                                '_train_{:04d}_{:04d}.png'
                                .format(epoch, batch_number)))

                        if self.bot is not None:
                            self.bot.send_file(
                                os.path.join(os.getcwd(),
                                             self.result_dir, self.model_dir,
                                             self.model_name +
                                             '_train_{:04d}_{:04d}.png'
                                             .format(epoch, batch_number)))

                except tf.errors.OutOfRangeError:
                    pbar.close()
                    break

            if self.verbosity >= 2:
                print("Epoch [%02d]: time: %4.4f,"
                      " d_loss: %.8f, g_loss: %.8f"
                      % (epoch, time.time() - start_time,
                         np.mean(plot_d_loss[-self.batch_size:]),
                         np.mean(plot_g_loss[-self.batch_size:])))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading
            # pre-trained model
            start_batch_id = 0

            # plot loss and evaluation metrics
            # self.plot_loss(plot_d_loss, plot_g_loss, plot_M,
            #                plot_logMMD, first_it, counter)
            self.plot_metrics([(plot_d_loss, plot_g_loss), plot_logMMD],
                              list(range(first_it, counter)),
                              metric_names=[("Discriminator loss",
                                             "Generator loss"), "log(MMD)"],
                              n_cols=1,
                              legend=[True, False],
                              x_label="Iteration",
                              y_label=["Loss", "log(MMD)"])

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        if self.dataset_name in ['mnist', 'fashion-mnist', 'celeba']:
            tot_num_samples = min(self.sample_num, self.batch_size)
            image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

            """ random condition, random noise """
            samples = self.sess.run(self.fake_images)

            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :],
                        [image_frame_dim, image_frame_dim],
                        os.path.join(
                            check_folder(os.path.join(os.getcwd(),
                                                      self.result_dir,
                                                      self.model_dir)),
                            self.model_name + '_epoch%03d' % epoch +
                            '_test_all_classes.png'))

            if self.bot is not None:
                self.bot.send_file(
                    os.path.join(os.getcwd(), self.result_dir, self.model_dir,
                                 self.model_name + '_epoch%03d' % epoch +
                                 '_test_all_classes.png'))
        else:
            raise NotImplementedError

    def plot_metrics(self, metrics_list, iterations_list,
                     metric_names=None, n_cols=2, legend=False, x_label=None,
                     y_label=None, wspace=None, hspace=None,
                     fig_wsize=16, fig_hsize=16):
        # cmap=plt.cm.tab20
        assert isinstance(metrics_list, (list, tuple)) and \
            not isinstance(metrics_list, str)

        # fig, ax1 = plt.subplots(1,1, figsize=(10,8))
        fig = plt.figure(figsize=(fig_hsize, fig_wsize))

        grid_cols = n_cols
        grid_rows = int(np.ceil(len(metrics_list) / n_cols))

        gs = GridSpec(grid_rows, grid_cols)
        if wspace is not None and hspace is not None:
            gs.update(wspace=wspace, hspace=hspace)
        elif wspace is not None:
            gs.update(wspace=wspace)
        elif hspace is not None:
            gs.update(hspace=hspace)

        n_plots = len(metrics_list)

        for ii, metric in enumerate(metrics_list):
            # if isinstance(first_it, (list, tuple)) and \
            #    isinstance(it_counter, (list, tuple)):
            #     list_it = range(first_it[ii], it_counter[ii])
            # else:
            #     list_it = range(first_it, it_counter)

            # if (n_plots % n_cols != 0) or (ii // n_cols == grid_rows):
            ax = plt.subplot(gs[ii // n_cols, ii % n_cols])
            # else:

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

            if isinstance(metric[0], (list, tuple)):
                lines = []
                for jj, submetric in enumerate(metric):
                    if metric_names is not None:
                        label = metric_names[ii][jj]
                    else:
                        label = "line_%01d" % jj
                    line, = ax.plot(iterations_list, submetric,
                                    color='C%d' % jj,
                                    label=label)
                    lines.append(line)
            else:
                if metric_names is not None:
                    label = metric_names[ii]
                else:
                    label = "line_01"
                line, = ax.plot(iterations_list, metric, color='C0',
                                label=label)
                lines = [line]

            if (not isinstance(legend, (list, tuple)) and legend) or \
                    (isinstance(legend, (list, tuple)) and legend[ii]):
                lg = ax.legend(handles=lines,
                               bbox_to_anchor=(1.0, 1.0),
                               loc="upper left")
                bbox_extra_artists = (lg, )
            else:
                bbox_extra_artists = None

            if x_label is not None and not isinstance(x_label, (list, tuple)):
                ax.set_xlabel(x_label, color='k')
            elif isinstance(x_label, (list, tuple)):
                ax.set_xlabel(x_label[ii], color='k')

            # Make the y-axis label, ticks and tick labels
            # match the line color.
            if y_label is not None and not isinstance(y_label, (list, tuple)):
                ax.set_ylabel(y_label, color='k')
            elif isinstance(y_label, (list, tuple)):
                ax.set_ylabel(y_label[ii], color='k')
            ax.tick_params('y', colors='k')

            # lg = ax2.legend(handles=[M_line], bbox_to_anchor=(1.0, 1.0),
            # loc="lower left")
            # lg = ax2.legend(handles=[MMD_line], bbox_to_anchor=(1.0, 1.0),
            # loc="lower left")

        plt.savefig(
            os.path.join(check_folder(self.result_dir + '/' + self.model_dir),
                         "metrics.png"), dpi=300,
            bbox_extra_artists=bbox_extra_artists, bbox_inches='tight')
        plt.close(fig)

        if self.bot is not None:
            self.bot.send_file(
                os.path.join(os.getcwd(), self.result_dir,
                             self.model_dir, "metrics.png"))

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir,
                                      self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir,
                                     self.model_name + '.model'),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        if self.verbosity >= 1:
            print("[*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir,
                                      self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,
                               os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(
                re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

            if self.verbosity >= 1:
                print("[*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            if self.verbosity >= 1:
                print("[*] Failed to find a checkpoint")
            return False, 0
