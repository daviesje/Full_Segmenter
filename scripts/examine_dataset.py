import matplotlib
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np

import context
from segmenter import data

allwhite = lambda x: np.all(x==255,axis=-1)

data_dir = '../multiclass_seg/x-sensing/'

train, val, test, info = data.load_images(data_dir,img_size=(256,256),img_channels=3,mask_channels=1,n_labels=2)

data.constants.n_labels = 2



gs = gridspec.GridSpec(2,5)
fig = plt.figure(figsize=(10,4))


for i,dpoint in enumerate(train.shuffle(1000).take(5)):
    img, mask = dpoint
    ax = fig.add_subplot(gs[0,i])
    ax.imshow(img,interpolation='none')

    ax = fig.add_subplot(gs[1,i],sharex=ax,sharey=ax)
    ax.imshow(mask[...,1],interpolation='none')

plt.show()
