"""
Create masks for all images in ../input/test_images

Pretrained Model Fn for multiclass is model_mres_3class.hdf5
Pretrained Model for glint is model_glare_retrained.hdf5 

All images you would like to segment should be placed into INPUT_DIR
Outputs (png of image/mask and data array of mask) will be placed into OUTPUT_DIR

data array of mask can be loaded via np.genfromtxt

WARNING: Ensure that you have sufficient RAM to load your image files. If not, splice your image into more managable sections.
"""

import sys
import glob
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from context import segmenter
from segmenter import unet_model, neuralNet
from tensorflow.keras import backend as K
from matplotlib import gridspec
from PIL import Image


MODEL_DIR = f'../models/'
MODEL_FN = f'model_mres_3class.hdf5'

parser = argparse.ArgumentParser()
parser.add_argument("-i", action='store_true', help="input filepath")
parser.add_argument("-o", action='store_true', help="output filepath")
parser.add_argument("-v", action='store_true', help="verbose")

args = parser.parse_args()
verbose = args.v

if args.i:
    INPUT_DIR = args.i
else:
    INPUT_DIR = f'../input/test_images'

if args.o:
    OUTPUT_DIR = args.o
else:
    OUTPUT_DIR = f'../input/test_images'

def create_mask(pred_mask):
    """
    Multi-class, take argmax across class dimension.
    Return format: (H, W, 1, 1)
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def simple_crop(image):
    """
    Ensures that dimensions are divisble by 32 (2^5), for 5 layers. This is such that
    you do not end up with fractional pixels during the pooling layers
    """
    d1,_ = np.divmod(image.shape[0], 32)
    d2,_ = np.divmod(image.shape[1], 32)

    return d1*32, d2*32

num_output = 4

test_list = sorted(glob.glob(f'{INPUT_DIR}/*'))

assert len(test_list) > 0, 'no images in the specified input folder'

model_base = tf.keras.models.load_model(f'{MODEL_DIR}{MODEL_FN}', compile=False)

model = neuralNet.mRES_net(input_size=(None, None, 3), n_output=num_output, dropout=0.5)
model.set_weights(model_base.get_weights())

if verbose:
    model.summary()
    print('')
    model_base.summary()


if verbose:
    print('----')
    print(f'There are {len(test_list)} test images')
    print('----')


for test_fn in test_list:
    image = tf.io.read_file(f'{test_fn}')
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize_with_crop_or_pad(image, *simple_crop(image))

    if verbose:
        print(f'Predicting mask for {test_fn}')
        print(f'Resized shape: ({image.shape})')

    pred_img = model.predict(image[tf.newaxis,...])
    pred_img_mask = create_mask(pred_img)[0]

    n_label = num_output

    cmap = plt.get_cmap('viridis', n_label)
    clim = [-0.5,n_label-0.5]
    gs = gridspec.GridSpec(1,4, wspace=0., hspace=0., width_ratios=[0.32,0.32,0.02,0.02])

    fig = plt.figure(figsize=(20,6))
    ax = fig.add_subplot(gs[:,0])
    ax2 = fig.add_subplot(gs[:,1])
    axc = fig.add_subplot(gs[:,-1])

    ax.imshow(image)
    ax.set_title('image')
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

    test = ax2.imshow(pred_img_mask, clim=clim, cmap=cmap)
    ax2.set_title('net')
    ax2.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

    cb = plt.colorbar(test, cax = axc, ticks=range(num_output))
    cb.ax.set_yticklabels(['Background','Coral','Bleached Coral','Glare'])

    gs.update(wspace=0, hspace=0)

    plt.savefig(f'{OUTPUT_DIR}/{test_fn}_predict.png', dpi=400, bbox_inches='tight')
    np.savetxt(f'{OUTPUT_DIR}/{test_fn}_mask.dat', pred_img_mask, header='#background = 0, coral = 1, bleached = 2, glare = 3')
