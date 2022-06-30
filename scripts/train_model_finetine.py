"""
Finetune a given model
Freeze the lower layers of a net (you may want to use model.summary() to taake a look at the architecture) and set to untraininable.

USAGE: python train_model_finetune.py PATH_TO_FINETUNED_MODEL(optional)

Default is set to (256,256) input image size with 4 classes. You can change these to your own image sizes, but ensure that they are consistent
and the dimensions are divisible by 32. This latter requirement is due to the neural net architecture

Make sure the input is placed at the corresponding location: ../input/train_images
and that paired images and masks have the **SAME FILENAMES** prefixed with tile_{FILENAME} or mask_{FILENAME} respectively.

A retiling script is provided in ../segmenter/tile_images.py, however you are free to use your own.

Specifics and changes to the data loading pipeline (plus choice of augmentations) can be found in ../segmenter/data.py under the load_images function.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import context
from segmenter import unet_model, data, outputs, multires_unet, neuralNet
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", action='store_true', help="path to model you want to finetune")

args = parser.parse_args()

input_model = args.i

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

IMG_SIZE = (256, 256)
OUTPUT_CHANNELS = 4
IMG_CHANNELS = 3

DEFAULT_MODEL_DIR = f'../models/'
DEFAULT_MODEL_FN = f'model_mres_3class.hdf5'


input_size = IMG_SIZE + (IMG_CHANNELS,)
model_suffix = 'new_model_finetune'

train, val, test, info = data.load_images('../input/train_images/', img_size=IMG_SIZE, n_labels=OUTPUT_CHANNELS)

TRAIN_LENGTH = info['train_count']
print(f'debug: train images {TRAIN_LENGTH}')
print(f'debug: val images {info["val_count"]}')
print(f'debug: test images {info["test_count"]}')

def fix_shape(images, labels):
    """
    Making this code compatible with previous code. Mask in the previous code required (B, H, W, 1) for binary classification of glare. Here we remove axis=3 and reshape
    into 4 for number of classes 
    """
    images.set_shape([None, None, None, 3])
    labels = tf.squeeze(labels, axis=3)
    labels.set_shape([None, None, None, OUTPUT_CHANNELS])
    return images, labels

for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
    ibuf = sample_image.numpy()
    mbuf = sample_mask.numpy()
    vals,_ = tf.unique(mbuf.flatten())
    # display([sample_image, sample_mask])
    print('train')
    print(f'debug: image size, {ibuf.shape}, dtype {ibuf.dtype}, min {ibuf.min()},max {ibuf.max()}, avg {ibuf.mean()}')
    print(f'debug: mask size,dtype,vals {mbuf.shape, mbuf.dtype, vals.numpy()}')

def dice_coef(y_true, y_pred, smooth=1e-6, ignore_back=False, num_classes=OUTPUT_CHANNELS):
    """
    Dice coefficient to calculate dice loss. Just a metric you can define to look at if you like.
    Ignore background is False by default
    """
    if ignore_back == True:
        y_true_f = K.flatten(y_true[...,1:])
        y_pred_f = K.flatten(y_pred[...,1:])
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        return K.mean((2. * intersection / (K.sum(y_true_f + y_pred_f, axis=-1) + smooth)))

    else:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f, axis=-1)
        return K.mean((2. * intersection) / (K.sum(y_true_f + y_pred_f, axis=-1) + smooth)) #divide result by classes

def weighted_cce(y_true, y_pred):
    """
    Weighted Categorical Crossentropy: Define your weight importance below
    """
    weights = np.array([1,1.6,5,5])    #class weights here
    weights = weights.reshape((1,1,1,4))
    kWeights = K.constant(weights)

    yWeights = kWeights * y_true         #shape (batch, n, m, class_labels)
    yWeights = K.sum(yWeights, axis=-1)  #shape (batch, n, m), each pixel is 'weighed' if they match

    loss = K.categorical_crossentropy(y_true, y_pred) #shape (batch, n, m), unweighed cc at each pixel
    wLoss = yWeights * loss

    return wLoss

if input_model:
    print('loadin')
    model = tf.keras.models.load_model(sys.argv[1], compile=False)
else:
    print('using default in ../models')
    model = tf.keras.models.load_model(f'{DEFAULT_MODEL_DIR}{DEFAULT_MODEL_FN}', compile=False)

#Layer number threshold where lower layers are frozen.
#Use model.layers to get an idea of the total number of layers
FINE_TUNE_LAYER = 160

model.trainable = True
for layer in model.layers[:FINE_TUNE_LAYER]:
    layer.trainable = False

#learning_rate set relatively low (lower than 1e-4)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=weighted_cce,
              metrics=['accuracy', dice_coef])

EPOCHS = 100
BATCH_SIZE = 4
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

#If your computer is unable to cache the buffer, then remove .cache() and reduce TRAIN_LENGTH to something managable.
train_dataset = train.cache().shuffle(TRAIN_LENGTH).batch(BATCH_SIZE).map(fix_shape).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = val.batch(BATCH_SIZE).map(fix_shape)

#Callbacks to optimize training. e.g. save checkpoints, log metrics, terminate if model stops learning.
checkpoint = ModelCheckpoint(f'../models/checkpoint_{model_suffix}.hdf5', monitor='val_loss',
                             verbose=1, save_best_only=True)

csv_logger = CSVLogger(f'../logs/log_{model_suffix}.out', separator=',')

earlystopping = EarlyStopping(monitor='val_loss', verbose=1,
                              min_delta=0.0005, patience=5)

plateau = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.00005, verbose=1)

callbacks_list = [checkpoint, csv_logger, plateau, earlystopping]

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=None,
                          validation_data=val_dataset,
                          callbacks=callbacks_list)

model.save(f'../models/{model_suffix}.hdf5')

