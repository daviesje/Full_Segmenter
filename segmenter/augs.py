from Constants import *
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
#np.set_printoptions(threshold=np.inf)

#KEVEN'S DATA LOADING CODE, SUBSETS ADAPTED INTO DATA.PY

constants = Constants()

def one_hot(img, mask):

    mask_f = mask[...,0]    
    mask_f = tf.one_hot(mask_f, constants.N_CLASSES)

    return img, mask_f

def joint_parser(img_path, mask_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)
    #mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)

    return image, mask

def construct_dataset(path_dir, label, fraction, NUM_FILES = constants.N_FILES, seed = constants.SEED_LIST):
    """
    if no particular masks to train on
    pre-shuffle the directory
    avoid calling whole dataset into RAM - give a precomputed number for number of files, maybe problematic if you have many labels
    """
    #seed = constants.SEED_LIST

    image_set = tf.data.Dataset.list_files(f'{path_dir}/{label}/images/*.png', seed = seed)
    mask_set  = tf.data.Dataset.list_files(f'{path_dir}/{label}/masks/*.png', seed = seed) 

    dataset_set = tf.data.Dataset.zip((image_set, mask_set))

    split = int(fraction * NUM_FILES/2)

    val_dataset = dataset_set.take(split * 2)
    #test_dataset = dataset_set.take(split)
    test_dataset = dataset_set.take(0)

    train_dataset  = dataset_set.skip(int(split*2))

    #convert filenames to images/masks

    train_dataset = train_dataset.map(joint_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(joint_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset  = test_dataset.map(joint_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, split, NUM_FILES - int(split*2)

@tf.function
def normalize(input_image, input_mask):
    '''
    Scale input images into [0,1] from RGB [0,255]
    Use this if mask labels are {1,2,..n} -> {0,1,..,n-1}
    '''

    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.uint8)

    return input_image, input_mask

@tf.function
def image_pre_blur(image, mask, sigma):
    input_image = tf.image.resize(image, (constants.IMG_SIZE, constants.IMG_SIZE))
    input_mask = tf.image.resize(mask, (constants.IMG_SIZE, constants.IMG_SIZE))

    if tf.random.uniform(()) > 0.1:
        input_image = tfa.image.gaussian_filter2d(input_image, filter_shape=[3,3], sigma=sigma, padding='CONSTANT')

    return input_image, input_mask


@tf.function
def load_image_train(image, mask):
    #FOR JAMES - CHANGE TO tf.random.uniform < 1 TO ADD THE PARTICULAR AUGMENTATION:

    input_image = tf.image.resize(image, (constants.IMG_SIZE, constants.IMG_SIZE))
    input_mask = tf.image.resize(mask, (constants.IMG_SIZE, constants.IMG_SIZE))
    
    if tf.random.uniform(()) > 0.5:
        #FLIP
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    if tf.random.uniform(()) > 1:
        #SCALE - ZOOM OUT AND PAD TILE TO 256x256 WITH 0
        scale = tf.random.uniform((), 0, constants.IMG_SIZE, dtype='int32')
        input_image = scale_randomizer(input_image, scale)
        input_mask = scale_randomizer(input_mask, scale, 'nearest')

    if tf.random.uniform(()) > 1:
        #COLOR AUGMENTATION
        input_image = tf.image.random_brightness(input_image, 0.05)
        input_image = tf.image.random_saturation(input_image, 0.6, 1.6)
        input_image = tf.image.random_hue(input_image, 0.2)
        input_image = tf.image.random_contrast(input_image, 0.6, 1.6)

    if tf.random.uniform(()) > 1:
    #BLUR FILTERS, GAUSSIAN AND MEDIAN BLUR
    #stupidest interaction between tfa gaussian_filter and tf.dataset.map. gaussian filter has python code which does not float well with tf graphs.
    #doing this the dumb way

        roll_dice1 = tf.random.uniform(())
        roll_dice = tf.random.uniform(())

        if roll_dice1 > 0.3:
            if roll_dice < 0.2:
                input_image = gauss_blur(input_image, 0.4)

            if (roll_dice > 0.2) & (roll_dice < 0.4):
                input_image = gauss_blur(input_image, 0.8)

            if (roll_dice > 0.4) & (roll_dice < 0.6):
                input_image = gauss_blur(input_image, 1.2)

            if (roll_dice > 0.6) & (roll_dice < 0.8):
                input_image = gauss_blur(input_image, 1.6)

            if roll_dice > 0.8:
                input_image = gauss_blur(input_image, 2.0)

        if roll_dice1 < 0.3:
            if roll_dice < 0.5:
                input_image = median_blur(input_image, 3)

            if roll_dice > 0.5:
                input_image = median_blur(input_image, 5)


    input_image, input_mask = normalize(input_image, input_mask)
    input_image = tf.clip_by_value(input_image, 0.0, 1.0)

    return input_image, input_mask

def gauss_blur(input_image, sigma):
    return tfa.image.gaussian_filter2d(input_image, filter_shape=[3,3], sigma=sigma, padding='CONSTANT')

def median_blur(input_image, pixel):
    return tfa.image.median_filter2d(input_image, filter_shape=[pixel,pixel], padding='CONSTANT')

def scale_randomizer(image, scale, method='bilinear'):
    '''
    take an image and a scale > 1.
    '''

    #input_image = tf.image.resize_with_crop_or_pad(image, int(constants.IMG_SIZE * scale), int(constants.IMG_SIZE * scale))    

    paddings = tf.constant([[1, 1],[1,1],[0,0]]) * scale
    input_image = tf.pad(image, paddings, mode='CONSTANT')
    input_image = tf.image.resize(input_image, (constants.IMG_SIZE, constants.IMG_SIZE), method=method)
    
    return input_image


def load_image_test(image, mask):
    input_image = tf.image.resize(image, (constants.IMG_SIZE, constants.IMG_SIZE))
    input_mask = tf.image.resize(mask, (constants.IMG_SIZE, constants.IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

if __name__ == "__main__":
    
    N_FILES = 512
    IMG_SIZE=256
    N_CHANNELS = 3
    N_CLASSES = 3

    PATH_DIR = '/Users/keven/x-sensing/good/paper_images/ml_tiles/tiles'
    LABEL = 'test'

    a, b, TRAIN_LENGTH, TEST_LENGTH = construct_dataset(PATH_DIR, LABEL, 0.8, NUM_FILES = N_FILES)

    test_dataset = b.map(load_image_test)
    test_dataset = test_dataset.repeat()
    test_dataset = test_dataset.batch(32)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print(test_dataset)

    for aaa,bbb in test_dataset.take(1):
        q, r = aaa, bbb

    i = 4
    print(np.sum(r[i]))
    print(r[i][:,:,0])

    def display(display_list):
        plt.figure(figsize=(6, 6))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')

        plt.show()

    display([q[i], r[i]])