import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import gridspec

#TODO: generating a mask for each weight set is inefficient, do something better
def create_mask(model,image,mask=None,weights=None):
    if len(image.shape) == 3:
        image = image[None,...]
        
    pred_mask = model.predict(image)
    
    #TODO: maybe not hardcode the shape (although it will never not be this)
    if weights is None:
        weights = tf.ones([1,1,1,pred_mask.shape[-1]])
    else:
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
        buf = (pred_mask > weights).squeeze()
    return buf, mask

def to_colors(mask,cmapstr=None):

    if cmapstr is None:
        cmapstr = 'viridis'

    cmap = plt.get_cmap(cmapstr,int(tf.math.reduce_max(mask)) + 1)

    mask = cmap(mask)
    
    return mask
    

#TODO: use create_mask here instead of doing it all again
def show_predictions(model, dataset, num=1,weights=None,interactive=False):
    image_arr = []
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image[None,...])
        print(f'debug: prediction size,min,max,avg ',
              f'{pred_mask.shape, pred_mask.dtype}',
              f'{pred_mask.min(), pred_mask.max()}',
              f'{pred_mask.mean(axis=(0,1,2))}')

        #TODO: maybe not hardcode the shape (although it will never not be this)
        if weights is None:
            weights = tf.ones([1,1,1,pred_mask.shape[-1]])
        else:
            weights = tf.reshape(tf.constant(weights),shape=[1,1,1,pred_mask.shape[-1]])

        #categorical
        if pred_mask.shape[-1] > 1:
            #clip to [0,1] with weights
            pred_mask = tf.maximum(tf.minimum(tf.multiply(pred_mask,weights),1),0)
            buf = tf.argmax(pred_mask[0],axis=-1)
            mask = tf.argmax(mask,axis=-1)
        #binary
        else:
            #weights will be single float in this case
            buf = (pred_mask[0] > weights).squeeze()
        image_arr.append([image, mask, buf])
    # PLOT IMAGE, LABELS, PREDICTION

    cmap = plt.get_cmap('viridis',pred_mask.shape[-1])

    rows = 3
    cols = num
    gs = gridspec.GridSpec(rows, cols,hspace=0,wspace=0)
    clim = [-0.5,pred_mask.shape[-1]-0.5]
    #categories = ['bg','glare','coral','newdead','bleach','seagrass','dead']
    #categories = ['bg','glare','coral','bleach']
    categories = ['bg','glare']

    fig = plt.figure(figsize=(cols*2, rows*2))
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

    cb = plt.colorbar(im,ax=fig.axes,ticks=range(pred_mask.shape[-1]))
    cb.ax.set_yticklabels(categories)

    fig.savefig('./plots/predictions_2.png')
    return image_arr
