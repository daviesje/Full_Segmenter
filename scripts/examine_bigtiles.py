import matplotlib
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np

import context
from segmenter import data, tile_images

allwhite = lambda x: np.all(x==255,axis=-1)

data.constants.n_labels = 2

gs = gridspec.GridSpec(2,3)

b_tile = '../multiclass_seg/Bleaching_glare_map_big_02_08.png'
b_mask = '../multiclass_seg/Bleaching_glare_polygon_big_02_08.png'
m_tile = '../multiclass_seg/main0509.png'
m_mask = '../multiclass_seg/mask_cube0509'

m_tile = tile_images.read_image(m_tile)
b_tile = tile_images.read_image(b_tile)
m_mask = np.loadtxt(m_mask,dtype=int,delimiter=' ',comments='#',ndmin=2)
b_mask = tile_images.read_image(b_mask)
#np.loadtxt(b_mask,dtype=int,delimiter=' ',comments='#',ndmin=2)

#glare only
check = m_mask == 1
m_mask = m_mask * check


fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(gs[0,0])
ax.set_ylabel('multiclass glare')
ax.set_title('tiles')
ax.imshow(m_tile,interpolation='none')
ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax = fig.add_subplot(gs[0,1],sharex=ax,sharey=ax)
ax.set_title('masks')
ax.imshow(m_mask,interpolation='none')
ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax = fig.add_subplot(gs[0,2],sharex=ax,sharey=ax)
ax.set_title('all white pixels')
ax.imshow(allwhite(m_tile),interpolation='none')
ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax = fig.add_subplot(gs[1,0],sharex=ax,sharey=ax)
ax.set_ylabel('x-sensing')
ax.imshow(b_tile,interpolation='none')
ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax = fig.add_subplot(gs[1,1],sharex=ax,sharey=ax)
ax.imshow(b_mask,interpolation='none')
ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
ax = fig.add_subplot(gs[1,2],sharex=ax,sharey=ax)
ax.imshow(allwhite(b_tile),interpolation='none')
ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

plt.show()