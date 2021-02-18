import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import gridspec

#TODO: generating a mask for each weight set is inefficient, do something better
def create_mask(model,image,truth=None,weights=None):
    if len(image.shape) == 3:
        image = image[None,...]
        
    pred_mask = model.predict(image)
    pred_mask = pred_mask.numpy()
    truth = truth.numpy()

    print(f'initial mask shape: {pred_mask.shape}')

    #binary refers to where the output shape is one with 2 classes
    binary = pred_mask.shape[-1] == 1

    if mask is not None:
        if not tf.reduce_all(pred_mask.shape[1:] == truth.shape[1:]):
            print(f'shape error: mask {truth.shape} pred {pred_mask.shape}')
            quit()
    
    n_weights = weights.shape[0]

    #TODO: maybe not hardcode the shape (although it will never not be this)
    #set default weights/threshold
    if weights is None:
        if binary:
            weights = 0.5 #this default is assuming a sigmoid activation
        else:
            weights = np.ones([1,1,1,pred_mask.shape[-1]])
    else:
        if not binary:
            weights = weights.reshape((n_weights,1,1,pred_mask.shape[-1]))
    
    #out_shape = np.insert(pred_mask.shape[1:],0,n_weights)
    #predictions = np.full(out_shape,-1)
    #cast probabilities to class number
    if not binary:
        #categorical case
        pred_mask = pred_mask * weights

        pred_mask = np.argmax(pred_mask, axis=-1)
        
        if truth is not None:
            truth = tf.argmax(truth, axis=-1)
    else:
        #binary case
        pred_mask = pred_mask > weights
        if truth is not None:
            truth = truth > weights

    print(f'final shape {pred_mask.shape} {truth.shape}')

    return pred_mask, truth
    

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
  