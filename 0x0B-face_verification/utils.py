#!/usr/bin/env python3
"""
Face verification utils
"""

import os
import numpy as np
import cv2
import glob
import csv


def load_images(images_path, as_array=True):
    """
    Function to load images
    """
    images = []
    filenames = []

    image_path = glob.glob(images_path + "/*")
    image_path.sort()
    for path in image_path:
        img_name = path.split("/")[-1]
        filenames.append(img_name)

    for path in image_path:
        image = cv2.imread(path)
        new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(new_img)

    if as_array is True:
        images = np.array(images)

    return images, filenames


def load_csv(csv_path, params={}):
    """
    a list of lists representing the contents found in csv_path
    """
    csv_list = []
    with open(csv_path, newline='') as csvfile:
        content = csv.reader(csvfile, params)
        for row in content:
            csv_list.append(row)
    return csv_list


def save_images(path, images, filenames):
    """save images"""
    try:
        os.chdir(path)
        for name, img in zip(filenames, images):
            cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        os.chdir('../')
        return True

    except FileNotFoundError:
        return False


def generate_triplets(images, filenames, triplet_names):
    """
    images is a numpy.ndarray of shape (i, n, n, 3) containing the aligned
    images in the dataset
    """
    names = [name.split('.')[0] for name in filenames]
    A = []
    P = []
    N = []
    for a, p, n in triplet_names:
        try:
            A.append(images[names.index(a)])
            P.append(images[names.index(p)])
            N.append(images[names.index(n)])
        except ValueError:
            continue
    return [np.asarray(A), np.asarray(P), np.asarray(N)]
