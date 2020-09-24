# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:28:50 2020

Segmenter Training Functions

@author: jed12
"""
import context
from segmenter import unet_model, data, outputs
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot as plt
import sys
import numpy as np

IMG_SIZE = (256, 256)
OUTPUT_CHANNELS = 7
IMG_CHANNELS = 3
input_size = IMG_SIZE + (IMG_CHANNELS,)

# train, test, info = data.load_pets()
train, test, info = data.load_images('../multiclass_seg/', img_size=IMG_SIZE, n_labels=OUTPUT_CHANNELS)

# TRAIN_LENGTH = info.splits['train'].num_examples
TRAIN_LENGTH = info['train_count']
print(f'debug: train images {TRAIN_LENGTH}')
print(f'debug: test images {info["test_count"]}')
BATCH_SIZE = 32
BUFFER_SIZE = 512
#loss = tf.keras.losses.SparseCategoricalCrossentropy()
loss = tf.keras.losses.CategoricalCrossentropy()
#loss = tf.keras.losses.BinaryCrossentropy()
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
#weights = {0: 1, 1: 100}
#weights = np.array([1,100])[None,:]

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

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

# image_batch, label_batch = next(iter(test_dataset))
# show_batch(image_batch.numpy(), label_batch.numpy())
# quit()

if len(sys.argv) > 1:
    model = tf.keras.models.load_model(sys.argv[1], custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
else:
    model = unet_model.unet(input_size=input_size, n_output=OUTPUT_CHANNELS
                            , n_base=16, batchnorm=False, dropout=0.3)

model.summary()

#loss = lambda x, y : my_weighted_loss(tf.keras.losses.categorical_crossentropy,x,y,
#                                      weights=np.array([1,75]))
loss = tf.keras.losses.categorical_crossentropy

model.compile(optimizer='adam',
              loss=loss,
              metrics=['accuracy'])

EPOCHS = 5
VAL_SUBSPLITS = 5
# VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
VALIDATION_STEPS = info['test_count'] // BATCH_SIZE // VAL_SUBSPLITS

checkpoint = ModelCheckpoint('./models/checkpoint.hdf5', monitor='val_loss',
                             verbose=1, save_best_only=True)

csv_logger = CSVLogger('./log.out', separator=',')

earlystopping = EarlyStopping(monitor='val_loss', verbose=1,
                              min_delta=0.005, patience=10)

plateau = ReduceLROnPlateau(factor=0.1, patience=4, min_lr=0.0001, verbose=1)

callbacks_list = [checkpoint, csv_logger, plateau, earlystopping]

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=None,
                          validation_data=test_dataset,
                          callbacks=callbacks_list)

model.save('./models/model_multi.hdf5')

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig('./plots/history.png')

outputs.show_predictions(model,test.shuffle(100),num=6,interactive=False)
