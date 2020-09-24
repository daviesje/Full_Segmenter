# -*- coding: utf-8 -*-
"""
Package containing setup of datasets for the segmenter
"""

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
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
    mask = mask[...,0] // 16
    
    mask = tf.one_hot(mask, constants.n_labels)

    #mask = tf.image.resize(mask, constants.img_size)

    return mask

def load_pets_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], constants.img_size)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], constants.img_size)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
        # MORE TRANSFORMATIONS HERE

    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1

    return input_image, input_mask


def load_pets_test(datapoint):
    input_image = tf.image.resize(datapoint['image'], constants.img_size)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], constants.img_size)

    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1

    return input_image, input_mask


def load_pets():
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    train = dataset['train'].map(load_pets_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = dataset['test'].map(load_pets_test)

    return train, test, info


# Tensorflow dataset loading (DOES NOT NEED TO FIT IN MEMORY)
def load_images(data_dir, img_size=(128, 128), img_channels=3, mask_channels=1, n_labels=2):
    # setting dimensions etc.
    data_dir = pathlib.Path(data_dir)
    constants.img_size = img_size
    constants.img_channels = img_channels
    constants.mask_channels = mask_channels
    constants.n_labels = n_labels

    # file list -> dataset
    x_train = tf.data.Dataset.list_files(str(data_dir / 'train_tiles/t*.png'),shuffle=False)
    x_test = tf.data.Dataset.list_files(str(data_dir / 'test_tiles/t*.png'),shuffle=False)
    y_train = tf.data.Dataset.list_files(str(data_dir / 'train_masks/m*.png'),shuffle=False)
    y_test = tf.data.Dataset.list_files(str(data_dir / 'test_masks/m*.png'),shuffle=False)


    # list dataset -> image dataset
    x_train = x_train.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y_train = y_train.map(load_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_test = x_test.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    y_test = y_test.map(load_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train = tf.data.Dataset.zip((x_train, y_train))
    test = tf.data.Dataset.zip((x_test, y_test))

    info = {}
    image_count = len(list(data_dir.glob('train_tiles/*.png')))
    info['train_count'] = image_count
    test_count = len(list(data_dir.glob('test_tiles/*.png')))
    info['test_count'] = test_count

    return train, test, info


# keras generator method
def get_image_generators(data_dir, batch_size):
    image_datagen = ImageDataGenerator(rescale=1. / 255)

    mask_datagen = ImageDataGenerator(rescale=1,
                                      dtype=int)
    seed = 123

    train_image_generator = image_datagen.flow_from_directory(
        'tiles_seg/train_frames/',
        batch_size=batch_size,
        class_mode=None,
        color_mode='rgb',
        seed=seed)

    train_mask_generator = mask_datagen.flow_from_directory(
        'tiles_seg/train_masks/',
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=seed)

    val_image_generator = image_datagen.flow_from_directory(
        'tiles_seg/val_frames/',
        batch_size=batch_size,
        class_mode=None,
        color_mode='rgb',
        seed=seed)

    val_mask_generator = mask_datagen.flow_from_directory(
        'tiles_seg/val_masks/',
        batch_size=batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=seed)

    train_generator = zip(train_image_generator, train_mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)

    return train_generator, val_generator
