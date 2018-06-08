# Source: https://git.io/vhTze

import sys
import os
import numpy as np
import tensorflow as tf
import pathlib
import glob
import gzip


class MNIST:
    TARGET_DIR = '_datasets/mnist/'
    N_TRAIN_SAMPLES = 50000
    N_VALID_SAMPLES = 10000
    N_TEST_SAMPLES = 10000

    @staticmethod
    def denorm_img(norm_imgs):
        return norm_imgs * 255

    @staticmethod
    def _load_dataset(labels=True):
        if not os.path.exists(MNIST.TARGET_DIR):
            os.makedirs(MNIST.TARGET_DIR)

        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, MNIST.TARGET_DIR+filename)

        def norm_img(img):
            img = img / 255
            return np.float32(img)

        def load_mnist_images(filename):
            if not os.path.exists(MNIST.TARGET_DIR+filename):
                download(filename)

            with gzip.open(MNIST.TARGET_DIR+filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)

            data = data.reshape(-1, 1, 28, 28).transpose(0, 2, 3, 1)

            return norm_img(data)

        def load_mnist_labels(filename):
            if not os.path.exists(MNIST.TARGET_DIR+filename):
                download(filename)

            with gzip.open(MNIST.TARGET_DIR+filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)

            label = data.reshape(-1)

            return label

        if labels:
            train_ims, train_labels = \
                load_mnist_images('train-images-idx3-ubyte.gz'), \
                load_mnist_labels('train-labels-idx1-ubyte.gz')

            test_ims, test_labels = \
                load_mnist_images('t10k-images-idx3-ubyte.gz'), \
                load_mnist_labels('t10k-labels-idx1-ubyte.gz')

            return train_ims, train_labels, test_ims, test_labels
        else:
            train_ims = load_mnist_images('train-images-idx3-ubyte.gz')
            test_ims = load_mnist_images('t10k-images-idx3-ubyte.gz')

            return train_ims, test_ims

    def __init__(self, batch_size, train_epoch=None):
        self.ims, self.labels, self.test_ims, self.test_labels = \
            MNIST._load_dataset()

        self.train_ims = self.ims[:50000]
        self.train_labels = self.labels[:50000]
        self.valid_ims = self.ims[50000:]
        self.valid_labels = self.labels[50000:]

        self.train_ds = \
            tf.data.Dataset.from_tensor_slices(
                (self.train_ims, self.train_labels)) \
            .shuffle(5000) \
            .apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        # .repeat(train_epoch) \

        self.valid_ds = \
            tf.data.Dataset.from_tensor_slices(
                (self.valid_ims, self.valid_labels)) \
            .apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (self.test_ims, self.test_labels)) \
            .apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        # train_iterator = train_ds.make_initializable_iterator()
        # valid_iterator = valid_ds.make_initializable_iterator()

        # self.train_data_init_op, self.train_data_op = \
        #     train_iterator.initializer, train_iterator.get_next()
        # self.valid_data_init_op, self.valid_data_op = \
        #     valid_iterator.initializer, valid_iterator.get_next()

        self.output_types = self.train_ds.output_types
        self.output_shapes = self.train_ds.output_shapes


class FASHION_MNIST:
    TARGET_DIR = '_datasets/fashion-mnist/'
    N_TRAIN_SAMPLES = 50000
    N_VALID_SAMPLES = 10000
    N_TEST_SAMPLES = 10000

    @staticmethod
    def denorm_img(norm_imgs):
        return norm_imgs * 255

    @staticmethod
    def _load_dataset(labels=True):
        if not os.path.exists(FASHION_MNIST.TARGET_DIR):
            os.makedirs(FASHION_MNIST.TARGET_DIR)

        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        def download(filename, source='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'):
            print("Downloading %s" % filename)
            urlretrieve(source + filename, FASHION_MNIST.TARGET_DIR+filename)

        def norm_img(img):
            img = img / 255
            return np.float32(img)

        def load_mnist_images(filename):
            if not os.path.exists(FASHION_MNIST.TARGET_DIR+filename):
                download(filename)

            with gzip.open(FASHION_MNIST.TARGET_DIR+filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)

            data = data.reshape(-1, 1, 28, 28).transpose(0, 2, 3, 1)

            return norm_img(data)

        def load_mnist_labels(filename):
            if not os.path.exists(FASHION_MNIST.TARGET_DIR+filename):
                download(filename)

            with gzip.open(FASHION_MNIST.TARGET_DIR+filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)

            label = data.reshape(-1)

            return label

        if labels:
            train_ims, train_labels = \
                load_mnist_images('train-images-idx3-ubyte.gz'), \
                load_mnist_labels('train-labels-idx1-ubyte.gz')

            test_ims, test_labels = \
                load_mnist_images('t10k-images-idx3-ubyte.gz'), \
                load_mnist_labels('t10k-labels-idx1-ubyte.gz')

            return train_ims, train_labels, test_ims, test_labels
        else:
            train_ims = load_mnist_images('train-images-idx3-ubyte.gz')
            test_ims = load_mnist_images('t10k-images-idx3-ubyte.gz')

            return train_ims, test_ims

    def __init__(self, batch_size, train_epoch=None):
        self.ims, self.labels, self.test_ims, self.test_labels = \
            FASHION_MNIST._load_dataset()

        self.train_ims = self.ims[:50000]
        self.train_labels = self.labels[:50000]
        self.valid_ims = self.ims[50000:]
        self.valid_labels = self.labels[50000:]

        self.train_ds = \
            tf.data.Dataset.from_tensor_slices(
                (self.train_ims, self.train_labels)) \
            .shuffle(5000) \
            .apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        # .repeat(train_epoch) \

        self.valid_ds = \
            tf.data.Dataset.from_tensor_slices(
                (self.valid_ims, self.valid_labels)) \
            .apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        self.test_ds = tf.data.Dataset.from_tensor_slices(
            (self.test_ims, self.test_labels)) \
            .apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

        # train_iterator = train_ds.make_initializable_iterator()
        # valid_iterator = valid_ds.make_initializable_iterator()

        # self.train_data_init_op, self.train_data_op = \
        #     train_iterator.initializer, train_iterator.get_next()
        # self.valid_data_init_op, self.valid_data_op = \
        #     valid_iterator.initializer, valid_iterator.get_next()

        self.output_types = self.train_ds.output_types
        self.output_shapes = self.train_ds.output_shapes


class CelebA:
    TARGET_DIR = '_datasets/CelebA'
    N_TRAIN_SAMPLES = 162770
    N_VALID_SAMPLES = 19867
    N_TEST_SAMPLES = 19962

    @staticmethod
    def maybe_download_and_extract():
        from utils import celeba_download as d

        def check_avail():
            p = pathlib.Path(CelebA.TARGET_DIR)

            if not p.exists():
                return 0
            i = p/'img_align_celeba'

            if not i.exists():
                return 1
            i = p/'splits'

            if not i.exists():
                return 2
            return 3

        fns = [d.prepare_data_dir, d.download_celeb_a, d.add_splits]
        start = check_avail()
        for fn in fns[start:]:
            fn(CelebA.TARGET_DIR)

    @staticmethod
    def denorm_img(norm_imgs):
        return tf.clip_by_value((norm_imgs + 1)*127.5, 0, 255)

    @staticmethod
    def load_img_and_preprocess(filename):

        def norm_img(im):
            im = im/127.5 - 1.
            return tf.cast(im, tf.float32)

        image_string = tf.read_file(filename)
        im = tf.image.decode_image(image_string, channels=3)
        im = tf.image.crop_to_bounding_box(im, 50, 25, 128, 128)
        im = tf.image.resize_images(im, [64, 64])
        im = tf.reshape(im, [64, 64, 3])
        im = norm_img(im)

        return im, tf.constant(-1, tf.int32)

    def __init__(self, batch_size, train_epoch=None):
        CelebA.maybe_download_and_extract()

        def _make_ds(set_name):
            files = glob.glob(str(
                pathlib.Path(CelebA.TARGET_DIR)/'splits'/set_name/'*.*'))

            ds = \
                tf.data.Dataset.from_tensor_slices(files) \
                .map(CelebA.load_img_and_preprocess, num_parallel_calls=4)

            if set_name == 'train':
                ds = ds.shuffle(5000) \
                       .apply(
                        tf.contrib.data.batch_and_drop_remainder(batch_size))
                # .repeat(train_epoch) \
            else:
                ds = \
                    ds.apply(
                        tf.contrib.data.batch_and_drop_remainder(batch_size))

            return ds

        self.train_ds = _make_ds('train')
        self.valid_ds = _make_ds('valid')
        self.test_ds = _make_ds('test')

        # train_iterator = train_ds.make_initializable_iterator()
        # valid_iterator = _make_ds('valid').make_initializable_iterator()
        # test_iterator = _make_ds('test').make_initializable_iterator()
        #
        # self.train_data_init_op, self.train_data_op = \
        #     train_iterator.initializer, train_iterator.get_next()
        # self.valid_data_init_op, self.valid_data_op = \
        #     valid_iterator.initializer, valid_iterator.get_next()
        # self.test_data_init_op, self.test_data_op = \
        #     test_iterator.initializer, test_iterator.get_next()

        self.output_types = self.train_ds.output_types
        self.output_shapes = self.train_ds.output_shapes


if __name__ == "__main__":
    def test_ds(ds):
        sess = tf.InteractiveSession()

        iterator = tf.data.Iterator.from_structure(ds.output_types,
                                                   ds.output_shapes)

        training_init_op = iterator.make_initializer(ds.train_ds)
        next_element = iterator.get_next()
        print(next_element[0].shape)

        sess.run([training_init_op])
        for _ in range(10):
            ims, labels = sess.run(next_element)
            print(ims.shape, labels)

        valid_labels = []
        try:
            valid_init_op = iterator.make_initializer(ds.valid_ds)
            sess.run([valid_init_op])

            while(True):
                ims, labels = sess.run(next_element)
                valid_labels.append(labels)

        except tf.errors.OutOfRangeError:
            valid_labels = np.concatenate(valid_labels, axis=0)
            labels, count = np.unique(valid_labels, return_counts=True)
            print('# of validation images: %d' % len(valid_labels))
            print('labels/count')
            for l, c in zip(labels, count):
                print('%d: %d' % (l, c))

        sess.close()

    mnist = MNIST(16)
    test_ds(mnist)

    fashion_mnist = FASHION_MNIST(16)
    test_ds(fashion_mnist)

    celeba = CelebA(16)
    test_ds(celeba)
