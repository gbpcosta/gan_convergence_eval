"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
import pandas as pd
from time import gmtime, strftime
from six.moves import xrange
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import gzip

from keras.datasets import cifar100

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
plt.switch_backend("Agg")


base_dir = "/home/DADOS1/gabriel/BEPE/model_GAN/datasets"


def load_synt_data(dataset_name, norm=False, random_seed=42):
    dataset_key = dataset_name.rsplit("_", 1)[1]
    dataset_name = dataset_name.rsplit("_", 1)[0]

    if dataset_key == "noisy":
        dataset_key = "noisy_models2"

    if dataset_key == "normal":
        dataset_key = "models"

    dataset_file = os.path.join(base_dir, dataset_name, dataset_name + ".h5")

    data = pd.read_hdf(dataset_file, dataset_key)

    X = data.values[:, :, np.newaxis, np.newaxis]
    y = np.zeros(data.shape[0], dtype=np.int32)

    if norm:
        X_min = np.min(X)
        X_max = np.max(X)
        X = (X - X_min) / (X_max - X_min)
        X -= 0.5
    else:
        X_min = None
        X_max = None

    np.random.seed(random_seed)
    np.random.shuffle(X)
    np.random.seed(random_seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), np.unique(y).shape[0]+1), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X, y_vec, X_min, X_max


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, axcb, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.sign(np.matmul(np.c_[xx.ravel(), yy.ravel()], clf))
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_contours_disc(ax, axcb, xx, yy, Z_final, n_levels=100, **params):
    # Z_train,
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    # n_levels = min(n_levels, len(set(np.append(Z_final, Z_train))))

    minv = Z_final.min()  # np.minimum(Z_final.min(), Z_train.min())
    maxv = Z_final.max()  # Z_train.max()
    levels = np.arange(minv, maxv, ((maxv - minv)/n_levels))
    # np.append( , maxv)
    # print levels

    out = ax.contourf(xx, yy, Z_final, levels, extend="both", **params)
    cbar = plt.colorbar(out, cax=axcb)
    cbar.set_label("Discriminator value")

    return out, cbar


def load_omniglot_test(dataset_name, eval=False):
    dataset_key = dataset_name.split("_")[2]
    dataset_name = dataset_name.split("_")[0]

    if eval:
        dataset_key = "eval_" + dataset_key

    if dataset_key == "raw":
        dataset_key += "_28"

    dataset_file = \
        os.path.join(base_dir, dataset_name, dataset_name + "_features.h5")

    data = pd.read_hdf(dataset_file, dataset_key)

    # X = data.iloc[:,2:].values
    # y = pd.Categorical(data.iloc[:,0])

    # np.random.seed(random_seed)
    # np.random.shuffle(X)
    # np.random.seed(random_seed)
    # np.random.shuffle(y)

    # y_vec = np.zeros((len(y), np.unique(y.codes).shape[0]), dtype=np.float)
    # for i, label in enumerate(y.codes):
    #     y_vec[i, y.codes[i]] = 1.0

    return data  # X, y


def load_omniglot_models(dataset_name, min_ap=None, random_seed=42, norm=None):
    dataset_key = dataset_name.split("_")[2]
    dataset_name = dataset_name.split("_")[0]

    if dataset_key == "raw":
        dataset_key = "svm_ovr_" + dataset_key + "_28"

    if dataset_key == "hog":
        dataset_key = "svm_ovr_" + dataset_key

    dataset_file = os.path.join(base_dir, dataset_name, dataset_name + ".h5")

    data = pd.read_hdf(dataset_file, dataset_key)

    if min_ap is not None:
        data = data[data.iloc[:, -1] >= min_ap]

    X = data.iloc[:, :-2].values  # [:, :, np.newaxis, np.newaxis]
    # if dataset_key == "svm_ovr_raw_28":
    #     X[:, 783, : , :] = X[:, 784, :, :]
    #     X = np.reshape(X[:, :784, : , :], [-1, 28, 28, 1])
    y = pd.Categorical(data.iloc[:, -2])

    if norm is not None:
        if norm == "znorm":
            X_min = X.mean(axis=0, keepdims=True)
            X_max = X.std(axis=0, keepdims=True)

            X = (X - X_min)/(X_max)
        elif norm == "self":
            X_min = X.min(axis=1, keepdims=True)
            X_max = X.max(axis=1, keepdims=True)

            X = (X - X_min) / (X_max - X_min)
        else:
            X_min = X.min()
            X_max = X.max()

            X = (X - X_min) / (X_max - X_min)

    X = X[:, :, np.newaxis, np.newaxis]

    np.random.seed(random_seed)
    np.random.shuffle(X)
    np.random.seed(random_seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), np.unique(y.codes).shape[0]), dtype=np.float)
    for i, label in enumerate(y.codes):
        y_vec[i, y.codes[i]] = 1.0

    return X, y_vec, X_min, X_max


def load_cifar_resblock(dataset_name, layer, random_seed=42,
                        norm=None, per_class=False):

    dataset_file = os.path.join(base_dir, dataset_name.split('_')[0],
                                dataset_name.rsplit('_', 1)[0] + ".h5")

    if layer == "conv20" or layer == "conv21":
        if not per_class:
            raise NotImplementedError
            # data = pd.read_hdf(dataset_file, layer)
            #
            # X = np.concatenate([np.array(data.iloc[:,2].values.tolist()),
            # np.array(data.iloc[:,3].values.tolist())[:, np.newaxis, :]],
            # axis=1)
            #
            # X = X[:, :, :, np.newaxis]
        else:

            data = pd.read_hdf(dataset_file, "softmax")

            used_classes = np.unique(
                [item for l in data.iloc[:, 0].values for item in l])
            test_classes = sorted(list(
                set(xrange(0, 100)) - set(used_classes)))

            data = pd.read_hdf(dataset_file, "%s_per_kernel" % layer)

            X = data.iloc[:, :].values

            X = X[:, :, np.newaxis, np.newaxis]

    elif layer == "bn20" or layer == "bn21":
        raise NotImplementedError

    elif layer == "softmax":
        if not per_class:
            data = pd.read_hdf(dataset_file, "softmax")

            used_classes = np.unique(
                [item for l in data.iloc[:, 0].values for item in l])
            test_classes = sorted(list(
                set(xrange(0, 100)) - set(used_classes)))

            X = np.concatenate(
                [np.array(data.iloc[:, 2].values.tolist()),
                 np.array(data.iloc[:, 3].values.tolist())[:, np.newaxis, :]],
                axis=1)

            X = X[:, :, :, np.newaxis]
        else:
            data = pd.read_hdf(dataset_file, "softmax_per_class")

            used_classes = np.unique(data.iloc[:, 0].values).astype(np.int64)
            test_classes = sorted(list(set(xrange(0, 100)) -
                                       set(used_classes)))

            X = data.iloc[:, 1:].values

            X = X[:, :, np.newaxis, np.newaxis]

    if norm is not None:
        X_min = X.min()
        X_max = X.max()

        X = (X - X_min) / (X_max - X_min)

    np.random.seed(random_seed)
    np.random.shuffle(X)

    return X, X_min, X_max, used_classes, test_classes


def load_cifar_resnet(dataset_name, random_seed=42, norm=None, per_class=False):

    dataset_file = os.path.join(base_dir, dataset_name.split('_')[0],
                                dataset_name + ".h5")

    if not per_class:
        data = pd.read_hdf(dataset_file, "classifiers")

        used_classes = np.unique(
            [item for l in data.iloc[:, 0].values for item in l])
        test_classes = sorted(list(set(xrange(0, 100)) - set(used_classes)))

        X = np.concatenate(
            [np.array(data.iloc[:, 2].values.tolist()),
             np.array(data.iloc[:, 3].values.tolist())[:, np.newaxis, :]],
            axis=1)

        X = X[:, :, :, np.newaxis]
    else:
        data = pd.read_hdf(dataset_file, "per_class")

        used_classes = np.unique(data.iloc[:, 0].values).astype(np.int64)
        test_classes = sorted(list(set(xrange(0, 100)) - set(used_classes)))

        X = data.iloc[:, 1:].values

        X = X[:, :, np.newaxis, np.newaxis]

    if norm is not None:
        X_min = X.min()
        X_max = X.max()

        X = (X - X_min) / (X_max - X_min)

    np.random.seed(random_seed)
    np.random.shuffle(X)

    return X, X_min, X_max, used_classes, test_classes


def to_per_class(dataset_name):
    dataset_file = os.path.join(base_dir, dataset_name.split('_')[0],
                                dataset_name + ".h5")

    data = pd.read_hdf(dataset_file, "classifiers")

    W = data.iloc[:, 2].tolist()
    b = data.iloc[:, 3].tolist()
    classes = data.iloc[:, 0].tolist()

    conc = [np.concatenate([classes[ii].reshape(1, -1),
                            W[ii], b[ii].reshape(1, -1)], axis=0)
            for ii in range(len(W))]

    data = []
    for ii in range(len(conc)):
        data.extend(np.split(conc[ii], conc[ii].shape[1], axis=1))

    data = np.array(data).squeeze()
    data = pd.DataFrame(data)

    data.to_hdf(dataset_file, "per_class")


def load_cifar_resnet_nomaxpool(dataset_name, random_seed=42, norm=None,
                                per_class=False):

    dataset_file = os.path.join(base_dir, dataset_name.split('_')[0],
                                dataset_name + ".h5")

    if not per_class:
        data = pd.read_hdf(dataset_file, "classifiers")

        used_classes = \
            np.unique([item for l in data.iloc[:, 0].values for item in l])
        test_classes = sorted(list(set(xrange(0, 100)) - set(used_classes)))

        X = np.concatenate(
            [np.array(data.iloc[:, 2].values.tolist()),
             np.array(data.iloc[:, 3].values.tolist())[:, np.newaxis, :]],
            axis=1)

        X = X[:, :, :, np.newaxis]
    else:
        data = pd.read_hdf(dataset_file, "per_class")

        used_classes = np.unique(data.iloc[:, 0].values).astype(np.int64)
        test_classes = sorted(list(set(xrange(0, 100)) - set(used_classes)))

        X = data.iloc[:, 1:-1].values
        X = np.concatenate(
            [X, data.iloc[:, -1].repeat(8*8).values.reshape([-1, 8*8])],
            axis=1)

        X = X[:, :, np.newaxis, np.newaxis]

    if norm is not None:
        X_min = X.min()
        X_max = X.max()

        X = (X - X_min) / (X_max - X_min)

    np.random.seed(random_seed)
    np.random.shuffle(X)

    return X, X_min, X_max, used_classes, test_classes


def load_cifar_test(dataset_name, eval=False):
    dataset_key = dataset_name.split("_", 1)[1]
    dataset_name = dataset_name.split("_", 1)[0]
    #
    # if eval:
    #     dataset_key = "eval_" + dataset_key
    #
    # if dataset_key == "raw":
    #     dataset_key += "_28"
    #
    #
    dataset_file = \
        os.path.join(base_dir, dataset_name, dataset_name + "_features.h5")
    #
    data = pd.read_hdf(dataset_file, "test_" + dataset_key)

    # X = data.iloc[:,2:].values
    # y = pd.Categorical(data.iloc[:,0])

    # np.random.seed(random_seed)
    # np.random.shuffle(X)
    # np.random.seed(random_seed)
    # np.random.shuffle(y)

    # y_vec = np.zeros((len(y), np.unique(y.codes).shape[0]), dtype=np.float)
    # for i, label in enumerate(y.codes):
    #     y_vec[i, y.codes[i]] = 1.0

    return data  # X, y


def print_top_evaluation(clfs, test_data, k, s_ind, savedir, dataset_name,
                         n_classes=10, reshape=False, bias=True):
    # if reshape:
    #     aux = np.zeros((clfs.shape[0], (clfs.shape[1] * clfs.shape[2])+1))
    #     aux[:,:-2] = np.reshape(clfs, [-1, 784])[:,:-1]
    #     aux[:, -1] = np.reshape(clfs, [-1, 784])[:, -1]
    #     clfs = aux

    if dataset_name == 'omniglot_models_raw' or \
       dataset_name == 'omniglot_models_hog':

        clfs_results = (np.dot(test_data.values[:, 2:],
                               clfs[s_ind, :-1]) + clfs[s_ind, -1])

        max_k = test_data.iloc[clfs_results.argsort()[-k:][::-1], 0]
        max_k_filenames = test_data.iloc[clfs_results.argsort()[-k:][::-1], 1]

        min_k = test_data.iloc[clfs_results.argsort()[:k], 0]
        min_k_filenames = test_data.iloc[clfs_results.argsort()[:k], 1]

        main_dir = base_dir + '/omniglot/images/images_background'

        # Four axes, returned as a 2-d array
        fig, axarr = plt.subplots(2, k, figsize=(20, 10))
        for i in range(k):
            im_path = os.path.join(main_dir, max_k.iloc[i].rsplit('_', 1)[0],
                                   max_k.iloc[i].rsplit('_', 1)[1],
                                   max_k_filenames.iloc[i])

            # n_im = max_k.index[i] - data[data[0] == max_k.iloc[i]].index[0]
            # im_path = os.path.join(im_path, os.walk(im_path).next()[2][n_im])

            im = mpimg.imread(im_path)
            axarr[0, i].imshow(im, cmap="gray")
            if i % 2 == 0:
                axarr[0, i].set_title(max_k.iloc[i], fontsize=9)
            else:
                axarr[0, i].set_title(max_k.iloc[i], fontsize=9,
                                      position=(0.5, -0.3))

            im_path = os.path.join(main_dir, min_k.iloc[i].rsplit('_', 1)[0],
                                   min_k.iloc[i].rsplit('_', 1)[1],
                                   min_k_filenames.iloc[i])

            # n_im = min_k.index[i] - data[data[0] == min_k.iloc[i]].index[0]
            # im_path = os.path.join(im_path, os.walk(im_path).next()[2][n_im])

            im = mpimg.imread(im_path)
            axarr[1, i].imshow(im, cmap="gray")
            if i % 2 == 0:
                axarr[1, i].set_title(min_k.iloc[i], fontsize=10)
            else:
                axarr[1, i].set_title(min_k.iloc[i], fontsize=10,
                                      position=(0.5, -0.3))

            # Fine-tune figure; hide x ticks for top plots and
            # y ticks for right plots
            plt.setp([a.get_xticklabels() for a in axarr[:, i]], visible=False)
            plt.setp([a.get_yticklabels() for a in axarr[:, i]], visible=False)

    elif dataset_name == 'cifar100_resnet20v1' or \
            dataset_name == 'cifar100_resnet20v1_nomaxpool':

        if bias:
            clfs_results = (np.dot(test_data.values[:, 2:],
                                   clfs[s_ind, :-1]) + clfs[s_ind, -1])
        else:
            clfs_results = np.dot(test_data.values[:, 2:], clfs[s_ind, :-1])

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        clfs_results = softmax(clfs_results)
        # print(clfs_results.max(axis=1), clfs_results.min(axis=1))

        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(n_classes, (2*k+1))

        for ii in range(n_classes):
            max_k_idx = clfs_results[:, ii].argsort()[-k:][::-1]
            min_k_idx = clfs_results[:, ii].argsort()[:k]

            max_k = test_data.iloc[max_k_idx, 0].values
            max_k_filenames = test_data.iloc[max_k_idx, 1].values

            min_k = test_data.iloc[min_k_idx, 0].values
            min_k_filenames = test_data.iloc[min_k_idx, 1].values

            (x_train, y_train), (x_test, y_test) = cifar100.load_data()

            # s_train = np.argsort(np.squeeze(y_train))
            s_test = np.argsort(np.squeeze(y_test))

            # x_train = x_train[s_train]
            # y_train = y_train[s_train]
            x_test = x_test[s_test]
            y_test = y_test[s_test]

            # Four axes, returned as a 2-d array
            # fig, axarr = plt.subplots(n_classes, 2*k,
            # figsize=(4*n_classes,4*k))
            for jj in range(k):
                ax_max = plt.subplot(gs[ii, jj])

                im = x_test[int(max_k[jj]*100+max_k_filenames[jj])]
                ax_max.imshow(im)
                ax_max.set_title(max_k[jj], fontsize=9)

                plt.setp([ax_max.get_xticklabels(), ax_max.get_xticklines()],
                         visible=False)
                plt.setp([ax_max.get_yticklabels(), ax_max.get_yticklines()],
                         visible=False)

                ax_min = plt.subplot(gs[ii, jj+k+1])

                im = x_test[int(min_k[jj]*100+min_k_filenames[jj])]
                ax_min.imshow(im)
                ax_min.set_title(min_k[jj], fontsize=9)

                plt.setp([ax_min.get_xticklabels(), ax_min.get_xticklines()],
                         visible=False)
                plt.setp([ax_min.get_yticklabels(), ax_min.get_yticklines()],
                         visible=False)

    plt.savefig(os.path.join(savedir, "max_min_%d_sample_%d.png" % (k, s_ind)))
    print("[*] Results saved to %s" %
          os.path.join(savedir, "max_min_%d_sample_%d.png" % (k, s_ind)))
    plt.close(fig)


def load_mnist(dataset_name):
    data_dir = os.path.join(base_dir, dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz',
                        60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz',
                        10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width, resize_height=64,
              resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width, resize_height=64,
              resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width,
                                    resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image,
                                            [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


""" Drawing Tools """
# borrowed from
# https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb


def save_scattered_image(z, id, z_range_x, z_range_y,
                         name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o',
                edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
