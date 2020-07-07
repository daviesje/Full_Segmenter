import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import gridspec

def create_mask(model,image,mask):
    pred_mask = model.predict(image[None, ...])
    if pred_mask.shape[-1] > 1:
        buf = tf.argmax(pred_mask[0], axis=-1)
        mask = tf.argmax(mask, axis=-1)
    else:
        buf = (pred_mask[0] > 0.5).squeeze()

    return buf, mask

def show_predictions(model, dataset, num=1,interactive=False):
    image_arr = []
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image[None,...])
        print(f'debug: prediction size,min,max,avg ',
              f'{pred_mask.shape, pred_mask.dtype}',
              f'{pred_mask.min(), pred_mask.max()}',
              f'{pred_mask.mean(axis=(0,1,2))}')
        if pred_mask.shape[-1] > 1:
            buf = tf.argmax(pred_mask[0],axis=-1)
            mask = tf.argmax(mask,axis=-1)
        else:
            buf = (pred_mask[0] > 0.5).squeeze()
        image_arr.append([image, mask, buf])
    # PLOT IMAGE, LABELS, PREDICTION
    rows = 3
    cols = num
    gs = gridspec.GridSpec(rows, cols)

    clim = [0,1]
    fig = plt.figure(figsize=(6, 6))
    for i, X in enumerate(image_arr):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(X[0])
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(X[1],clim=clim)
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(X[2],clim=clim)

    if interactive:
        plt.show()

    fig.savefig('./plots/predictions.png')
    return image_arr
