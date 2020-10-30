"""
Predict from images
"""

#from .context.segmenter import data, outputs
from context import segmenter
from segmenter import data, outputs
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import sys
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


n_categ = 7

model = tf.keras.models.load_model(sys.argv[1]
                                   ,custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
#model = tf.keras.models.load_model('./models/checkpoint.hdf5')

#Test read
image = np.array(Image.open('../multiclass_seg/test_masks/mask_0106_251.png',mode='r'))
print(np.unique(image.reshape(65536,1),return_counts=True,axis=0))

mask = tf.io.read_file('../multiclass_seg/test_masks/mask_0106_251.png')
mask = tf.image.decode_image(mask,channels=1)
mask = mask[...,0] // 16
mask = tf.one_hot(mask, n_categ)
print(np.unique(mask.numpy().reshape(65536,n_categ),return_counts=True,axis=0))


# LOAD IMAGES
data_dir = '../multiclass_seg/'
train,test,info = data.load_images(data_dir,(256,256),n_labels=n_categ)
ntest = info['test_count']
ntrain = info['train_count']

for image, mask in test.take(1):
    sample_image, sample_mask = image, mask
    ibuf = sample_image.numpy()
    mbuf = sample_mask.numpy()
    # display([sample_image, sample_mask])
    print('test')
    print(f'debug: image size,min,max,avg {ibuf.shape, ibuf.dtype, ibuf.min(), ibuf.max(), ibuf.mean()}')
    print(f'debug: mask size,min,max,avg {mbuf.shape, mbuf.dtype, mbuf.min(), mbuf.max(), mbuf.mean(axis=(0,1))}')

ntake = 6


nthrsh = 1
thresholds = np.ones((nthrsh,n_categ))

image_arr = outputs.show_predictions(model,test.shuffle(100),ntake,interactive=False)
#image_arr = outputs.show_predictions(model,train.shuffle(1000),ntake,interactive=False)
quit()

error_matrix = np.zeros((nthrsh,n_categ,n_categ))

mask_frac = np.zeros((nthrsh,ntest,n_categ))
pred_frac = np.zeros((nthrsh,ntest,n_categ))

for i, t in enumerate(thresholds):
    weights = t
    print(f'%{i}th threshold, {t}')
    #TODO: this is really inefficient, will separate create_mask into output
    #  generation and prediction, to generate one mask and apply several weights
    j = 0
    for image, mask in test:
        if j%25==0: print(j)
        #print(np.unique(mask.numpy().reshape(65536,n_categ),return_counts=True,axis=0))
        pred, mask = outputs.create_mask(model,image,mask,weights=weights)


        # PER PIXEL ANALYSIS
        for mpix,ppix in zip(mask.numpy().flatten(),pred.numpy().flatten()):
            error_matrix[i,mpix,ppix] += 1

        # PER IMAGE ANALYSIS (% glare)
        for ii in range(n_categ):
            mask_frac[i,j,ii] = np.sum(mask==ii)/mask.numpy().size
            pred_frac[i,j,ii] = np.sum(pred==ii)/pred.numpy().size
        j = j + 1

xbase = np.linspace(0,1,num=50)
zero = np.zeros(xbase.shape)
one = np.ones(xbase.shape)

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(ncols=1,nrows=2,figure=fig)
ax = fig.add_subplot(gs[0])
ax.plot(xbase,zero,'k:')
for i in range(nthrsh):
    xplot = mask_frac[i,...,1]
    yplot = (pred_frac[i,...,1] - mask_frac[i,...,1])
    
    print(yplot.min(),yplot.max(),yplot.mean())
    ax.scatter(xplot,yplot,s=2)
ax.set_ylabel('excess frac of image')

ax = fig.add_subplot(gs[1])
ax.plot(xbase,one,'k:')
for i in range(nthrsh):
    idx = mask_frac[i,...,1] > 0
    xplot = mask_frac[i,idx,1]
    yplot = (np.fabs(pred_frac[i,idx,1] - mask_frac[i,idx,1]))/mask_frac[i,idx,1]

    print(yplot.min(),yplot.max(),yplot.mean())
    ax.scatter(xplot,yplot,s=2)

ax.set_yscale('log')
ax.set_ylabel('excess frac / mask frac')
ax.set_xlabel('mask frac')

fig.savefig('./plots/excess.png')

pred_sums = error_matrix.sum(axis=1)[:,None,:]
mask_sums = error_matrix.sum(axis=2)[:,:,None]
np.set_printoptions(precision=3,suppress=True)
print("total pixels (row == real) (col == pred)")
print(error_matrix)
print("------------")
print("normalised by prediction")
print(error_matrix/pred_sums*100)
print("------------")
print("normalised by mask")
print(error_matrix/mask_sums*100)
print("------------")
print("mask fractions")
print(mask_frac.mean(axis=(0,1)))
print("------------")
print("pred fractions")
print(pred_frac.mean(axis=(0,1)))
print("------------")
print('histogram')
print(np.histogram(mask_frac[:,:,1],bins=np.linspace(0,1,num=11))[0])

plt.show()