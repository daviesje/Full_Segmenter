import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import gridspec

#TODO: generating a mask for each weight set is inefficient, do something better
def create_mask(model,image,mask=None,weights=None):
    if len(image.shape) == 3:
        image = image[None,...]
        
    pred_mask = model.predict(image)
    #if mask is not None:
    #    if not tf.reduce_all(pred_mask.shape[-3:] == mask.shape[-3:]):
    #        print(f'shape error: mask {mask.shape} pred {pred_mask.shape}')
    #        #quit()
    
    #TODO: maybe not hardcode the shape (although it will never not be this)
    if weights is None:
        if pred_mask.shape[-1] == 1:
            weights = 0.5 #this default is hardcoded assuming a sigmoid activation
        else:
            weights = tf.ones([1,1,1,pred_mask.shape[-1]])
    else:
        if pred_mask.shape[-1] != 1:
            weights = tf.reshape(tf.constant(weights),shape=[1,1,1,pred_mask.shape[-1]])
    
    #categorical case
    if pred_mask.shape[-1] > 1:
        #clip to [0,1] with weights
        #pred_mask = tf.maximum(tf.minimum(tf.multiply(pred_mask,weights),1),0)
        pred_mask = tf.multiply(pred_mask,weights)

        buf = tf.argmax(pred_mask, axis=-1)
        
        if mask is not None:
            mask = tf.argmax(mask, axis=-1)
    #binary case
    else:
        buf = tf.squeeze((pred_mask > weights))
    
    return buf, mask

def to_colors(mask,cmapstr=None):
    if cmapstr is None:
        cmapstr = 'viridis'

    cmap = plt.get_cmap(cmapstr,int(tf.math.reduce_max(mask)) + 1)

    mask = cmap(mask)
    
    return mask
    

#create a plot of images, true masks and predicted masks (3 by num grid)
def show_predictions(model, dataset, num=1,weights=None,interactive=False):
    image_arr = []
    for image, mask in dataset.take(num):
        pbuf, mbuf = create_mask(model,image,mask=mask,weights=weights)
        image_arr.append([image, tf.squeeze(mbuf), tf.squeeze(pbuf)])
        print(mbuf.shape,pbuf.shape)
        print(mbuf.numpy().min(),mbuf.numpy().max())

    if len(mbuf.shape) == 3:
        n_categ = max(mask.shape[-1],2)
    else:
        n_categ = 2

    cmap = plt.get_cmap('viridis',n_categ)

    rows = 3
    cols = num
    gs = gridspec.GridSpec(rows, cols,hspace=0,wspace=0)
    #clim takes -0.5 to align the colorbar labels
    clim = [-0.5,n_categ-0.5]

    #categories = ['bg','glare','coral','newdead','bleach','seagrass','dead']
    #categories = ['bg','glare','coral','bleach']
    categories = ['bg','glare']

    fig = plt.figure(figsize=(4,(rows/(cols))*4))
    for i, X in enumerate(image_arr):
        ax = fig.add_subplot(gs[0, i])
        if i == 0: ax.set_ylabel('image')
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        ax.imshow(X[0],interpolation='none')
        ax = fig.add_subplot(gs[1, i])
        if i == 0: ax.set_ylabel('mask')
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        ax.imshow(X[1],clim=clim,cmap=cmap,interpolation='none')
        ax = fig.add_subplot(gs[2, i])
        if i == 0: ax.set_ylabel('pred')
        ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        im = ax.imshow(X[2],clim=clim,cmap=cmap,interpolation='none')

    fig.subplots_adjust(hspace=0,wspace=0,left=0.05,right=0.95,top=0.95,bottom=0.05)
    cb = plt.colorbar(im,ax=fig.axes,ticks=range(n_categ))
    #cb.ax.set_yticklabels(categories)

    if interactive:
        plt.show()
    else:
        fig.savefig('./plots/predictions_2.png')
    return image_arr
  