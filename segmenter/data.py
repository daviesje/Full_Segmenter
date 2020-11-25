# -*- coding: utf-8 -*-
"""
Package containing setup of datasets for the segmenter
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib

#Ceiling division for masks
def ceildiv(a,b):
    return -(-a // b)

class Constants:
    def __init__(self):
        self.img_size = (128, 128)
        self.img_channels = 3
        self.mask_channels = 1
        self.n_labels = 7

constants = Constants()

def transform(datapoint):
    # TODO: put more transformations, add calls in loading
    if tf.random.uniform() > 0.5:
        datapoint[0] = tf.image.flip_left_right(datapoint[0])
        datapoint[1] = tf.image.flip_left_right(datapoint[1])

#TODO a more modular data prepare function
def prepare(dataset,format='png',augment=False,mask=False):
    return

def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=constants.img_channels)
    img = tf.image.convert_image_dtype(img, tf.float32)

    #img = tf.image.resize(img, constants.img_size)

    return img

def load_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=constants.mask_channels)
    
    #each png class is separated by 16 z-levels
    mask = mask[...,0] // 16

    if constants.n_labels == 4:
        #seagrass, newdead and dead set to background
        check = tf.math.logical_or(mask==3,mask==5)
        check = tf.cast(tf.math.logical_not(tf.math.logical_or(check,mask==6)),tf.uint8)
        #shuffle bleached back to index 3
        check2 = tf.cast(mask==k,tf.uint8)
        mask = mask*check - check2

    elif constants.n_labels == 2:
        #only glare
        check = tf.cast(mask == 1,tf.uint8)
        mask = mask * check

    mask = tf.one_hot(mask, constants.n_labels)

    #mask = tf.image.resize(mask, constants.img_size)

    return mask


# Tensorflow dataset loading (DOES NOT NEED TO FIT IN MEMORY)
def load_images(data_dir, img_size=(128, 128), img_channels=3, mask_channels=1, n_labels=2):
    # setting dimensions etc.
    data_dir = pathlib.Path(data_dir)
    constants.img_size = img_size
    constants.img_channels = img_channels
    constants.mask_channels = mask_channels
    constants.n_labels = n_labels

    # file list -> dataset
    x_train = tf.data.Dataset.list_files(str(data_dir / 'train_tiles/tile*.png'),shuffle=False)
    y_train = tf.data.Dataset.list_files(str(data_dir / 'train_masks/mask*.png'),shuffle=False)
    x_val = tf.data.Dataset.list_files(str(data_dir / 'val_tiles/tile*.png'),shuffle=False)
    y_val = tf.data.Dataset.list_files(str(data_dir / 'val_masks/mask*.png'),shuffle=False)
    x_test = tf.data.Dataset.list_files(str(data_dir / 'test_tiles/tile*.png'),shuffle=False)
    y_test = tf.data.Dataset.list_files(str(data_dir / 'test_masks/mask*.png'),shuffle=False)


    # list dataset -> image dataset
    x_train = x_train.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y_train = y_train.map(load_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_val = x_val.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y_val = y_val.map(load_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_test = x_test.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y_test = y_test.map(load_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train = tf.data.Dataset.zip((x_train, y_train))
    val = tf.data.Dataset.zip((x_val, y_val))
    test = tf.data.Dataset.zip((x_test, y_test))

    info = {}
    image_count = len(list(data_dir.glob('train_tiles/*.png')))
    info['train_count'] = image_count
    val_count = len(list(data_dir.glob('val_tiles/*.png')))
    info['val_count'] = val_count
    test_count = len(list(data_dir.glob('test_tiles/*.png')))
    info['test_count'] = test_count

    return train, val, test, info
