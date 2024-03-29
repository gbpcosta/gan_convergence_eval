# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import mmd
from tqdm import tqdm

from ops import *
from utils import *

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
plt.switch_backend("Agg")


class CGAN(object):
    model_name = "CGAN"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name,
                 checkpoint_dir, result_dir, log_dir, bot, verbosity):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.bot = bot
        self.verbosity = verbosity

        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.y_dim = 10         # dimension of condition-vector (label)
            self.c_dim = 1

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        elif dataset_name == 'synt_data':
            raise NotImplementedError
        elif dataset_name == 'synt_data_noisy':
            raise NotImplementedError
        elif dataset_name == 'omniglot_models_hog':
            # parameters
            self.input_height = 1
            self.input_width = 3529
            self.output_height = 1
            self.output_width = 3529

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # BEGAN Parameter
            self.gamma = 0.75
            self.lamda = 0.001

            # train
            self.learning_rate = 0.0002
            self.beta1 = 0.5

            # test
            self.sample_num = 16  # number of generated images to be saved

            # load mnist
            self.data_X, self.data_y = load_omniglot_models(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    def discriminator(self, x, y, is_training=True, reuse=False):
        if self.dataset_name == 'mnist' or \
           self.dataset_name == 'fashion-mnist':
            # Network Architecture is exactly same as in infoGAN
            # (https://arxiv.org/abs/1606.03657)
            # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
            with tf.variable_scope("discriminator", reuse=reuse):

                # merge image and label
                y = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(x, y)

                net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='d_conv1'))
                net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'),
                               is_training=is_training, scope='d_bn2'))
                net = tf.reshape(net, [self.batch_size, -1])
                net = lrelu(bn(linear(net, 1024, scope='d_fc3'),
                               is_training=is_training, scope='d_bn3'))
                out_logit = linear(net, 1, scope='d_fc4')
                out = tf.nn.sigmoid(out_logit)

                return out, out_logit, net
        else:
            raise NotImplementedError

    def generator(self, z, y, is_training=True, reuse=False):
        if self.dataset_name == 'mnist' or \
           self.dataset_name == 'fashion-mnist':
            # Network Architecture is exactly same as in infoGAN
            # (https://arxiv.org/abs/1606.03657)
            # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
            with tf.variable_scope("generator", reuse=reuse):

                # merge noise and label
                z = concat([z, y], 1)

                net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'),
                                    is_training=is_training, scope='g_bn1'))
                net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'),
                                    is_training=is_training, scope='g_bn2'))
                net = tf.reshape(net, [self.batch_size, 7, 7, 128])
                net = tf.nn.relu(
                    bn(deconv2d(net, [self.batch_size, 14, 14, 64],
                                4, 4, 2, 2, name='g_dc3'),
                       is_training=is_training, scope='g_bn3'))

                out = tf.nn.sigmoid(
                    deconv2d(net, [self.batch_size, 28, 28, 1],
                             4, 4, 2, 2, name='g_dc4'))

                return out
        else:
            raise NotImplementedError

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims,
                                     name='real_images')

        # labels
        self.y = tf.placeholder(tf.float32, [bs, self.y_dim], name='y')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(
            self.inputs, self.y, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = \
            self.discriminator(G, self.y, is_training=True, reuse=True)

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
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # optimizers
        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = \
                tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = \
                tf.train.AdamOptimizer(self.learning_rate*5, beta1=self.beta1)\
                .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = \
            self.generator(self.z, self.y, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

        """ MMD """
        self.generated_samples = tf.placeholder(tf.float32,
                                                [None, self.output_height,
                                                 self.output_width,
                                                 self.c_dim],
                                                name="mmd_generatedsamples")
        self.training_data = tf.placeholder(tf.float32,
                                            [None, self.input_height,
                                             self.input_width, self.c_dim],
                                            name="mmd_trainingdata")

        aux_1 = tf.reshape(self.generated_samples,
                           [-1, self.output_width * self.output_height *
                            self.c_dim])

        aux_2 = tf.reshape(self.training_data,
                           [-1, self.input_width * self.input_height *
                            self.c_dim])

        self.log_mmd = tf.log(mmd.rbf_mmd2(aux_1, aux_2))

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = \
            np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        self.test_labels = self.data_y[0:self.batch_size]

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = \
            tf.summary.FileWriter(self.log_dir + '/' + self.model_name,
                                  self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * \
                self.num_batches
            counter = checkpoint_counter
            if self.verbosity >= 1:
                print("[*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            if self.verbosity >= 1:
                print("[!] Load failed...")

        # plot variables
        plot_d_loss = []
        plot_g_loss = []
        plot_M = []
        plot_logMMD = []
        first_it = counter

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in tqdm(range(start_batch_id, self.num_batches)):
                batch_images = \
                    self.data_X[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_labels = \
                    self.data_y[idx * self.batch_size:
                                (idx + 1) * self.batch_size]
                batch_z = \
                    np.random.uniform(-1, 1, [self.batch_size,
                                      self.z_dim]).astype(np.float32)

                # update D network
                _, summary_str, d_loss = \
                    self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                  feed_dict={self.inputs: batch_images,
                                             self.y: batch_labels,
                                             self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = \
                    self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                  feed_dict={self.y: batch_labels,
                                             self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                plot_d_loss.append(d_loss)
                plot_g_loss.append(g_loss)
                # plot_M.append(M_value)

                # display training status
                counter += 1

                # if np.mod(counter, 100) == 0:
                samples = self.sess.run(self.fake_images,
                                        feed_dict={self.z: self.sample_z,
                                                   self.y: self.test_labels})

                logMDD_value = \
                    self.sess.run(self.log_mmd,
                                  feed_dict={self.generated_samples: samples,
                                             self.training_data: batch_images})

                plot_logMMD.append(logMDD_value)

                # save training results for every 300 steps
                if (self.dataset_name == 'mnist' or
                    self.dataset_name == 'fashion-mnist') \
                        and np.mod(counter, 300) == 0:
                    samples = \
                        self.sess.run(self.fake_images,
                                      feed_dict={self.z: self.sample_z,
                                                 self.y: self.test_labels})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))

                    save_images(
                        samples[:manifold_h * manifold_w, :, :, :],
                        [manifold_h, manifold_w],
                        './' +
                        check_folder(self.result_dir + '/' + self.model_dir)
                        + '/' + self.model_name +
                        '_train_{:04d}_{:04d}.png'.format(epoch, idx))

                    if self.verbosity >= 3 and self.bot is not None:
                        self.bot.send_file(
                            os.path.join(self.result_dir, self.model_dir,
                                         self.model_name +
                                         '_train_{:04d}_{:04d}.png'
                                         .format(epoch, idx)))

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
        if self.dataset_name == 'mnist' or \
           self.dataset_name == 'fashion-mnist':
            tot_num_samples = min(self.sample_num, self.batch_size)
            image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

            """ random condition, random noise """
            y = np.random.choice(self.y_dim, self.batch_size)
            y_one_hot = np.zeros((self.batch_size, self.y_dim))
            y_one_hot[np.arange(self.batch_size), y] = 1

            z_sample = np.random.uniform(-1, 1, size=(self.batch_size,
                                         self.z_dim))

            samples = self.sess.run(self.fake_images,
                                    feed_dict={self.z: z_sample,
                                               self.y: y_one_hot})

            save_images(
                samples[:image_frame_dim * image_frame_dim, :, :, :],
                [image_frame_dim, image_frame_dim],
                check_folder(self.result_dir + '/' + self.model_dir) + '/' +
                self.model_name + '_epoch%03d' % epoch +
                '_test_all_classes.png')

            if self.bot is not None:
                self.bot.send_file(
                    os.path.join(self.result_dir, self.model_dir,
                                 self.model_name + '_epoch%03d' % epoch +
                                 '_test_all_classes.png'))

            # specified condition, random noise

            n_styles = 10  # must be less than or equal to self.batch_size

            np.random.seed()
            si = np.random.choice(self.batch_size, n_styles)

            for l in range(self.y_dim):
                y = np.zeros(self.batch_size, dtype=np.int64) + l
                y_one_hot = np.zeros((self.batch_size, self.y_dim))
                y_one_hot[np.arange(self.batch_size), y] = 1

                samples = self.sess.run(self.fake_images, feed_dict={
                                        self.z: z_sample, self.y: y_one_hot})
                save_images(
                    samples[:image_frame_dim * image_frame_dim, :, :, :],
                    [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) +
                    '/' + self.model_name + '_epoch%03d' % epoch +
                    '_test_class_%d.png' % l)

                if self.bot is not None:
                    self.bot.send_file(
                        os.path.join(self.result_dir, self.model_dir,
                                     self.model_name + '_epoch%03d' % epoch +
                                     '_test_class_%d.png' % l))

                samples = samples[si, :, :, :]

                if l == 0:
                    all_samples = samples
                else:
                    all_samples = np.concatenate((all_samples, samples),
                                                 axis=0)

            """ save merged images to check style-consistency """
            canvas = np.zeros_like(all_samples)
            for s in range(n_styles):
                for c in range(self.y_dim):
                    canvas[s * self.y_dim + c, :, :, :] = \
                        all_samples[c * n_styles + s, :, :, :]

            save_images(canvas, [n_styles, self.y_dim],
                        check_folder(self.result_dir + '/' + self.model_dir) +
                        '/' + self.model_name + '_epoch%03d' % epoch +
                        '_test_all_classes_style_by_style.png')

            if self.bot is not None:
                self.bot.send_file(
                    os.path.join(self.result_dir, self.model_dir,
                                 self.model_name + '_epoch%03d' % epoch +
                                 '_test_all_classes_style_by_style.png'))
        else:
            raise NotImplementedError

    def plot_metrics(self, metrics_list, iterations_list,
                     metric_names=None, n_cols=2, legend=False, x_label=None,
                     y_label=None, wspace=None, hspace=None):
        # cmap=plt.cm.tab20
        assert isinstance(metrics_list, (list, tuple)) and \
            not isinstance(metrics_list, str)

        # fig, ax1 = plt.subplots(1,1, figsize=(10,8))
        fig = plt.figure(figsize=(12, 16))

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
                os.path.join(self.result_dir, self.model_dir, "metrics.png"))

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

        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        if self.verbosity >= 1:
            print("[*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir,
                                      self.model_dir, self.model_name)

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
