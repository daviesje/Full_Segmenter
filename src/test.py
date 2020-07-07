# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:28:50 2020

@author: jed12
"""
from src import unet_model
import sys
import data
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping,ReduceLROnPlateau

IMG_SIZE = (128,128,3)
OUTPUT_CHANNELS = 3

train,test,info = data.load_pets()

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE


def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.savefig('./predictions.png')


for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
#display([sample_image, sample_mask])

train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

model = unet_model.unet(input_size=IMG_SIZE,n_output=3,n_base=16)
#plot_model(model, show_shapes=True)
#model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

#show_predictions(test_dataset,3)

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 10
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

checkpoint = ModelCheckpoint('./models/checkpoint.hdf5',monitor='val_accuracy',
                             verbose=1,save_best_only=True)

csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor='val_acc',verbose=1,
                              min_delta=0.005,patience=5)

plateau = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)

callbacks_list = [checkpoint, csv_logger, earlystopping,plateau]

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=callbacks_list)

model.save('./models/model.hdf5')

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig('./history.png')

show_predictions(test_dataset,3)
