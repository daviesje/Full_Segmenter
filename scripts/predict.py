"""
Predict from images
"""

from src import data, outputs
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import sys

model = tf.keras.models.load_model(sys.argv[1]
                                   ,custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
#model = tf.keras.models.load_model('./models/checkpoint.hdf5')


# LOAD IMAGES
data_dir = '../tiles_seg/'
train,test,_ = data.load_images(data_dir,(256,256))

for image, mask in test.take(1):
    sample_image, sample_mask = image, mask
    ibuf = sample_image.numpy()
    mbuf = sample_mask.numpy()
    # display([sample_image, sample_mask])
    print('test')
    print(f'debug: image size,min,max,avg {ibuf.shape, ibuf.dtype, ibuf.min(), ibuf.max(), ibuf.mean()}')
    print(f'debug: mask size,min,max,avg {mbuf.shape, mbuf.dtype, mbuf.min(), mbuf.max(), mbuf.mean()}')

ntake = 6
image_arr = outputs.show_predictions(model,test.shuffle(100),ntake,interactive=False)
image_arr = outputs.show_predictions(model,train.shuffle(100),ntake,interactive=False)

true_pos = 0
true_neg = 0
fals_pos = 0
fals_neg = 0

mask_frac = np.array([])
pred_frac = np.array([])

for image, mask in test:
    pred, mask = outputs.create_mask(model,image,mask)

    # PER PIXEL ANALYSIS
    true_pos += np.sum(np.logical_and(pred==1,mask==1))
    true_neg += np.sum(np.logical_and(pred==0,mask==0))
    fals_pos += np.sum(np.logical_and(pred==1,mask==0))
    fals_neg += np.sum(np.logical_and(pred==0,mask==1))

    # PER IMAGE ANALYSIS (% glare)
    mask_frac = np.append(mask_frac,np.sum(mask==1)/mask.numpy().size)
    pred_frac = np.append(pred_frac,np.sum(pred==1)/pred.numpy().size)

print(f'True positives : {true_pos}')
print(f'False positives : {fals_pos}')
print(f'True negatives : {true_neg}')
print(f'False negatives : {fals_neg}')

print(pred_frac[mask_frac==0])

idx = mask_frac != 0
print(idx.shape)
mask_frac = mask_frac[idx]
print(idx.shape)
pred_frac = pred_frac[idx]

print(f'average excess glare fraction : {((pred_frac - mask_frac)/mask_frac).mean()}')
print(f'RMS excess glare fraction : {np.sqrt((((pred_frac - mask_frac)/mask_frac) ** 2).mean())}')

fig = plt.figure()
ax = fig.add_subplot(121)
ax.scatter(mask_frac,(pred_frac-mask_frac)/mask_frac,s=20)

ax.plot([0.,0.1],[0.,0.],'k:')
ax.set_xlabel('glare fraction')
ax.set_ylabel('excess prediction glare (% of glare fraction)')

ax = fig.add_subplot(122)
ax.scatter(mask_frac,(pred_frac-mask_frac),s=20)

ax.plot([0.,0.1],[0.,0.],'k:')
ax.set_xlabel('glare fraction')
ax.set_ylabel('excess prediction glare (% of image)')

plt.show()