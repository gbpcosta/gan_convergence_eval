import os
import numpy as np
import pandas as pd


def check_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def eq_spaced_points(n, r=1):
    polypoints = np.linspace(start=0, stop=2*np.pi, num=n+1)
    polypoints = polypoints[:-1]

    circx = r * np.sin(polypoints)
    circy = r * np.cos(polypoints)

    return zip(*[circx, circy])


def create_gaussian_modes(n_modes, std=0.1,
                          sample_size=500, r=1):
    centroids = eq_spaced_points(n_modes, r=r)

    dataset = np.zeros([n_modes * sample_size, 3])
    for ii, centroid in enumerate(centroids):
        sample = np.random.normal(centroid, std, (sample_size, 2))
        labels = np.repeat(ii, sample.shape[0])[:, np.newaxis]

        sample = np.concatenate([sample, labels], axis=1)
        dataset[ii*sample_size:(ii+1)*sample_size] = sample

    return dataset
