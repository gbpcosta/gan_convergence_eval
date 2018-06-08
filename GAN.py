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

import mmd
import inception_score
import fid

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
                 compute_metrics_it, checkpoint_dir, result_dir,
                 log_dir, gpu_id, bot, redo, verbosity):
        self.sess = sess
        self.dataset_name = dataset_name.lower()
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.compute_metrics_it = compute_metrics_it
        self.gpu_id = gpu_id
        self.bot = bot
        self.redo = redo
        self.verbosity = verbosity

        if self.dataset_name in ['mnist']:
            # parameters
            self.input_height = 28
            self.input_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.ds = dataset.MNIST(self.batch_size)
            self.num_batches = self.ds.N_TRAIN_SAMPLES // self.batch_size

            # architecture hyper parameters
            self.data_format = 'NHWC'

        elif self.dataset_name in ['fashion-mnist']:
            # parameters
            self.input_height = 28
            self.input_width = 28

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 1

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load mnist
            self.ds = dataset.FASHION_MNIST(self.batch_size)
            self.num_batches = self.ds.N_TRAIN_SAMPLES // self.batch_size

            # architecture hyper parameters
            self.data_format = 'NHWC'

        elif self.dataset_name in ['celeba']:
            # parameters
            self.input_height = 64
            self.input_width = 64

            self.z_dim = z_dim         # dimension of noise-vector
            self.c_dim = 3

            # test
            self.sample_num = 64  # number of generated images to be saved

            # load CelebA
            self.ds = dataset.CelebA(self.batch_size)
            self.num_batches = self.ds.N_TRAIN_SAMPLES // self.batch_size

        else:
            raise NotImplementedError

    def discriminator(self, x, is_training=True, reuse=False):
        raise NotImplementedError("method discriminator must be implemented")

    def generator(self, z, is_training=True, reuse=False):
        raise NotImplementedError("method generator must be implemented")

    def define_input(self):
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

        self.inputs, _ = self.iterator.get_next()

        # noises
        self.z = tf.random_normal([self.batch_size, self.z_dim])

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

    def define_test_sample(self):
        """" Testing """
        # for test
        self.sample_z = tf.constant(
            np.random.normal(loc=0.0, scale=1.0,
                             size=(self.batch_size, self.z_dim))
            .astype(np.float32))

        fake_images, _ = self.generator(self.sample_z, is_training=False,
                                        reuse=True)
        self.fake_images = self.ds.denorm_img(fake_images)

    def save_test_sample(self, epoch, batch_number):
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

    def define_mmd_comp(self, n_batches=100):
        """ MMD """
        ds_mmd_real = self.ds.test_ds

        ds_mmd_real = ds_mmd_real.shuffle(5000).apply(
            tf.contrib.data.batch_and_drop_remainder(
                n_batches))
        ds_mmd_real_iterator = ds_mmd_real.make_initializable_iterator()

        self.ds_mmd_real_init_op = ds_mmd_real_iterator.initializer
        ds_mmd_real_next, _ = ds_mmd_real_iterator.get_next()
        ds_mmd_real_next = tf.reshape(
            ds_mmd_real_next,
            shape=[-1, self.input_height, self.input_width, self.c_dim])

        mmd_z = tf.constant(
            np.random.normal(loc=0.0, scale=1.0,
                             size=(n_batches * self.batch_size, self.z_dim))
            .astype(np.float32))

        ds_mmd_z = \
            tf.data.Dataset.from_tensor_slices((mmd_z)) \
            .apply(tf.contrib.data.batch_and_drop_remainder(
                n_batches * self.batch_size))

        ds_mmd_z_iterator = ds_mmd_z.make_initializable_iterator()

        self.ds_mmd_z_init_op = ds_mmd_z_iterator.initializer
        ds_mmd_z_next = ds_mmd_z_iterator.get_next()

        mmd_generated_images, _ = \
            self.generator(ds_mmd_z_next,
                           is_training=False,
                           reuse=True)

        self.log_mmd = mmd.get_mmd(mmd_generated_images,
                                   ds_mmd_real_next,
                                   log=True)

    def compute_mmd(self):
        """ MMD """
        self.sess.run([self.ds_mmd_z_init_op,
                       self.ds_mmd_real_init_op])
        logMDD_value = \
            self.sess.run([self.log_mmd])

        return logMDD_value

    def define_inception_score_input(self):
        """ Inception Score """
        # "We find that it’s important to evaluate the metric on a large
        # enough number of samples (i.e. 50k) as part of this metric measures
        # diversity"
        inception_z = np.random.normal(loc=0.0, scale=1.0,
                                       size=(50000,
                                             self.z_dim)).astype(np.float32)

        ds_inception_z = \
            tf.data.Dataset.from_tensor_slices((inception_z)) \
            .shuffle(50000) \
            .apply(tf.contrib.data.batch_and_drop_remainder(500))

        ds_inception_z_iterator = ds_inception_z.make_initializable_iterator()

        self.ds_inception_z_init_op = ds_inception_z_iterator.initializer
        ds_inception_z_next = ds_inception_z_iterator.get_next()

        inception_images, _ = \
            self.generator(ds_inception_z_next,
                           is_training=False,
                           reuse=True)

        if self.c_dim == 1:
            self.inception_images = \
                    tf.image.grayscale_to_rgb(
                        self.ds.denorm_img(inception_images))

        else:
            self.inception_images = \
                    self.ds.denorm_img(inception_images)

    def compute_inception_score(self):
        """ Inception score """
        self.sess.run([self.ds_inception_z_init_op])

        if self.verbosity >= 3:
            print('[!] Computing inception score. '
                  'This may take a while...')

        inception_images = []
        while True:
            try:
                inception_images_batch = \
                    self.sess.run([self.inception_images])
                inception_images.extend(inception_images_batch)
            except tf.errors.OutOfRangeError:
                break

        inception_images = \
            np.concatenate(inception_images, axis=0)

        return inception_score.get_inception_score(inception_images,
                                                   gpu_id=self.gpu_id)

    def define_fid_input(self, n_batches=10):
        """ FID """
        fid_z = np.random.normal(loc=0.0, scale=1.0,
                                 size=(50000, self.z_dim)).astype(np.float32)

        ds_fid_z = \
            tf.data.Dataset.from_tensor_slices((fid_z)) \
            .shuffle(5000) \
            .apply(tf.contrib.data.batch_and_drop_remainder(n_batches *
                                                            self.batch_size))

        ds_fid_z_iterator = ds_fid_z.make_initializable_iterator()

        self.ds_fid_z_init_op = ds_fid_z_iterator.initializer
        ds_fid_z_next = ds_fid_z_iterator.get_next()

        fid_images, _ = \
            self.generator(ds_fid_z_next,
                           is_training=False,
                           reuse=True)

        ds_fid_real = self.ds.test_ds

        ds_fid_real = ds_fid_real.shuffle(5000).apply(
            tf.contrib.data.batch_and_drop_remainder(
                n_batches))
        ds_fid_real_iterator = ds_fid_real.make_initializable_iterator()

        self.ds_fid_real_init_op = ds_fid_real_iterator.initializer
        ds_fid_real_next, _ = ds_fid_real_iterator.get_next()

        ds_fid_real_next = tf.reshape(
            ds_fid_real_next,
            shape=[-1, self.input_height, self.input_width, self.c_dim])

        if self.c_dim == 1:
            self.fid_z_images = \
                tf.image.grayscale_to_rgb(
                    self.ds.denorm_img(fid_images))
            self.fid_real_images = \
                tf.image.grayscale_to_rgb(
                    self.ds.denorm_img(ds_fid_real_next))
        else:
            self.fid_z_images = \
                self.ds.denorm_img(fid_images)
            self.fid_real_images = \
                self.ds.denorm_img(ds_fid_real_next)

    def compute_fid(self, num_batches=10):
        """ FID """
        self.sess.run([self.ds_fid_z_init_op,
                       self.ds_fid_real_init_op])

        if self.verbosity >= 3:
            print('[!] Computing FID. '
                  'This may take a while...')

        fid_value = self.sess.run(
            fid.get_frechet_inception_distance(self.fid_real_images,
                                               self.fid_z_images,
                                               num_batches=num_batches))

        return fid_value

    def build_model(self):
        self.define_input()
        self.define_loss_fn()
        self.define_optimizers()
        self.define_test_sample()
        self.define_mmd_comp()
        self.define_inception_score_input()
        self.define_fid_input()

    def verify_checkpoint(self):
        # restore check-point if it exits
        if not self.redo:
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            if could_load:
                start_epoch = (int)(checkpoint_counter / self.num_batches)
                start_batch_id = \
                    checkpoint_counter - start_epoch * self.num_batches
                counter = checkpoint_counter
                if self.verbosity >= 1:
                    print('[*] Load SUCCESS')
            else:
                start_epoch = 0
                start_batch_id = 0
                counter = 1
                if self.verbosity >= 1:
                    print('[!] Load failed...')
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            if self.verbosity >= 1:
                print('[!] Redo!')

        return start_epoch, start_batch_id, counter

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' +
                                            self.model_name, self.sess.graph)

        start_epoch, start_batch_id, counter = self.verify_checkpoint()

        # plot variables
        plot_d_loss = []
        plot_g_loss = []
        plot_logMMD = []
        plot_inception_score = []
        plot_FID = []
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
                        plot_logMMD.append(self.compute_mmd())

                        inception_mean, inception_std = \
                            self.compute_inception_score()
                        plot_inception_score.append(inception_mean)

                        fid_value = self.compute_fid()
                        plot_FID.append(fid_value)

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
                 plot_inception_score,
                 plot_FID],
                iterations_list=[list(range(first_it, counter)),
                                 metrics_its,
                                 metrics_its,
                                 metrics_its],
                metric_names=[('Discriminator loss', 'Generator loss'),
                              'log(MMD)',
                              'Inception Score',
                              'FID'],
                n_cols=2,
                legend=[True, False, False, False],
                x_label='Iteration',
                y_label=['Loss', 'log(MMD)',
                         'Inception Score (Average)', 'FID'],
                fig_wsize=22, fig_hsize=16)

            # save model
            # self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

        # save model for final step
        # self.save(self.checkpoint_dir, counter)

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

    def visualize_data(self, epoch, sample_max=5):
        if self.dataset_name in ['mnist', 'fashion-mnist', 'celeba']:
            tot_num_samples = min(self.sample_num, self.batch_size)
            image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

            samples = self.sess.run(self.ds.denorm_img(self.inputs))

            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :],
                        [image_frame_dim, image_frame_dim],
                        check_folder(self.result_dir + '/' + self.model_dir) +
                        '/' + self.model_name + '_epoch%03d' % epoch +
                        '_training_data.png')

            if self.bot is not None:
                self.bot.send_file(
                    os.path.join(self.result_dir, self.model_dir,
                                 self.model_name + '_epoch%03d' % epoch +
                                 '_training_data.png'))
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
                    line, = ax.plot(iterations_list[ii], submetric,
                                    color='C%d' % jj,
                                    label=label)
                    lines.append(line)
            else:
                if metric_names is not None:
                    label = metric_names[ii]
                else:
                    label = "line_01"
                line, = ax.plot(iterations_list[ii], metric, color='C0',
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
