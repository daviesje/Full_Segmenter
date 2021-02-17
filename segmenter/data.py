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
        self.n_labels = 2
        self.z_steps = 255 #z difference between categories

constants = Constants()

def scale_randomizer(image, scale, method='bilinear'):
    '''
    take an image and a scale > 1.
    '''    
    paddings = tf.constant([[1, 1],[1,1],[0,0]]) * scale #dont pad channels, pad other dims by (scale) pixels
    input_image = tf.pad(image, paddings, mode='CONSTANT')
    input_image = tf.image.resize(input_image, constants.img_size, method=method)
    
    return input_image

def augment(input_image,input_mask):
    # TODO: put more transformations, add calls in loading
    if tf.random.uniform(()) > 0.5:
        #FLIP
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    rot = tf.random.uniform(shape=(),minval=0,maxval=3,dtype=tf.dtypes.int32)
    input_image = tf.image.rot90(input_image,k=rot)
    input_mask = tf.image.rot90(input_mask,k=rot)

    if tf.random.uniform(()) > 0.2:
        #COLOR AUGMENTATION
        input_image = tf.image.random_brightness(input_image, 0.2)
        input_image = tf.image.random_saturation(input_image, 0.5, 2)
        input_image = tf.image.random_hue(input_image, 0.2)
        input_image = tf.image.random_contrast(input_image, 0.5, 2)

    if tf.random.uniform(()) > 0.2:
        #SCALE - ZOOM OUT AND PAD TILE TO 256x256 WITH 0
        #up to a doubling of size
        scale = tf.random.uniform((), 0, constants.img_size[0], dtype='int32')
        input_image = scale_randomizer(input_image, scale,'bilinear')
        input_mask = scale_randomizer(input_mask, scale, 'bilinear')
    
        #decide if glare or bg based on interpolation at 0.5
        #rotation and scale changes can blur the segmentations so we have to
        #draw the line somewhere
        if constants.n_labels == 1:
            input_mask = tf.math.floor(input_mask + 0.5)
            input_mask = tf.image.convert_image_dtype(input_mask, tf.uint8)
            input_mask = input_mask // constants.z_steps
    
    input_image = tf.clip_by_value(input_image, 0.0, 1.0)

    return input_image, input_mask

def joint_parser(img_path,mask_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=constants.img_channels)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=constants.mask_channels)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)

    #divide by brightness steps per category
    mask = mask[...,0] // constants.z_steps
    mask = mask[...,None]

    if constants.n_labels > 1:
        mask = tf.one_hot(mask, constants.n_labels)
        #mask = tf.squeeze(mask)

    #mask = tf.image.resize(mask, constants.img_size)
    #mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)

    return img, mask



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

    train = tf.data.Dataset.zip((x_train, y_train))
    val = tf.data.Dataset.zip((x_val, y_val))
    test = tf.data.Dataset.zip((x_test, y_test))

    # list dataset -> image dataset
    train = train.map(joint_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val = val.map(joint_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = test.map(joint_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #TODO: put behind flag
    train = train.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    info = {}
    image_count = len(list(data_dir.glob('train_tiles/*.png')))
    info['train_count'] = image_count
    val_count = len(list(data_dir.glob('val_tiles/*.png')))
    info['val_count'] = val_count
    test_count = len(list(data_dir.glob('test_tiles/*.png')))
    info['test_count'] = test_count

    return train, val, test, info
