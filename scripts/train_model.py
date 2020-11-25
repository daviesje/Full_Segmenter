# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:28:50 2020

Segmenter Training Functions

@author: jed12
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import context
from segmenter import unet_model, data, outputs, multires_unet
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
import sys
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


#tf.config.experimental.set_visible_devices([], 'GPU')

IMG_SIZE = (256, 256)
OUTPUT_CHANNELS = 2
IMG_CHANNELS = 3
input_size = IMG_SIZE + (IMG_CHANNELS,)

train, val, test, info = data.load_images('../multiclass_seg/re_tiling/', img_size=IMG_SIZE, n_labels=OUTPUT_CHANNELS)

TRAIN_LENGTH = info['train_count']
print(f'debug: train images {TRAIN_LENGTH}')
print(f'debug: val images {info["val_count"]}')
print(f'debug: test images {info["test_count"]}')
BATCH_SIZE = 4
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


def my_weighted_loss(lossfunc, y_true, y_pred, weights=None):
    """scale loss based on class weights
    """
    #BROKEN
    if weights is None:
        _,_,counts = tf.unique_with_counts(y_true)
        weights = tf.size(y_true)/counts/tf.size(counts)

    weights = tf.reduce_sum(weights * y_true, axis=-1)

    # compute (unweighted) softmax cross entropy loss
    unweighted_losses = lossfunc(y_true, y_pred)
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)
    return loss

for image, mask in train.take(1):
    sample_image, sample_mask = image, mask
    ibuf = sample_image.numpy()
    mbuf = sample_mask.numpy()
    # display([sample_image, sample_mask])
    print('train')
    print(f'debug: image size,min,max,avg {ibuf.shape, ibuf.dtype, ibuf.min(), ibuf.max(), ibuf.mean()}')
    print(f'debug: mask size,min,max,avg {mbuf.shape, mbuf.dtype, mbuf.min(), mbuf.max(), mbuf.mean(axis=(0,1))}')

for image, mask in test.take(1):
    sample_image, sample_mask = image, mask
    ibuf = sample_image.numpy()
    mbuf = sample_mask.numpy()
    # display([sample_image, sample_mask])
    print('test')
    print(f'debug: image size,min,max,avg {ibuf.shape, ibuf.dtype, ibuf.min(), ibuf.max(), ibuf.mean()}')
    print(f'debug: mask size,min,max,avg {mbuf.shape, mbuf.dtype, mbuf.min(), mbuf.max(), mbuf.mean(axis=(0,1))}')

train_dataset = train.cache().shuffle(TRAIN_LENGTH).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val.batch(BATCH_SIZE)
test_dataset = test.batch(BATCH_SIZE)

if len(sys.argv) > 1:
    model = tf.keras.models.load_model(sys.argv[1], custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
else:
    model = unet_model.unet(input_size=input_size, n_output=OUTPUT_CHANNELS
                            , n_base=16, batchnorm=True, dropout=0.15)
    #model = multires_unet.mr_unet(input_size=input_size, n_output=OUTPUT_CHANNELS
    #                        , batchnorm=False, dropout=0, n_layers=4)


model.summary()

#weights_arr = np.array([0.256,0.012,0.279,0.014,0.023,0.301,0.116])
weights_arr = np.array([0.750,0.250])
weights_arr = (1/weights_arr)
weights_arr = weights_arr / weights_arr.min()

loss = lambda x, y : my_weighted_loss(tf.keras.losses.categorical_crossentropy,x,y,
                                      weights=weights_arr)

#loss = tf.keras.losses.categorical_crossentropy

model.compile(optimizer='adam',
              loss=loss,
              metrics=['accuracy'])

EPOCHS = 20
VAL_SUBSPLITS = 8
VALIDATION_STEPS = info['test_count'] // BATCH_SIZE // VAL_SUBSPLITS

checkpoint = ModelCheckpoint('./models/checkpoint.hdf5', monitor='val_loss',
                             verbose=1, save_best_only=True)

csv_logger = CSVLogger('./log_retiled_drop.out', separator=',')

earlystopping = EarlyStopping(monitor='val_loss', verbose=1,
                              min_delta=0.0001, patience=10)

plateau = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)

callbacks_list = [checkpoint, csv_logger, plateau, earlystopping]

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=None,
                          validation_data=val_dataset,
                          callbacks=callbacks_list)

model.save('./models/model_unet_2_retiled_drop.hdf5')

outputs.show_predictions(model,test.shuffle(100),num=6,interactive=False)

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
epochs = range(len(loss))

fig = plt.figure()
ax=fig.add_subplot(211)
ax.plot(epochs, loss, 'r', label='Training loss')
ax.plot(epochs, val_loss, 'bo', label='Validation loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss Value')
ax.set_ylim([0,loss[0]])
ax.legend()
ax=fig.add_subplot(212)
ax.plot(epochs, acc, 'r', label='Training acc')
ax.plot(epochs, val_acc, 'bo', label='Validation acc')
ax.set_xlabel('Epoch')
ax.set_ylabel('accuracy')
ax.set_ylim([0,1])
ax.legend()
fig.savefig('./plots/history_temp.png')
