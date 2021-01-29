"""
Predict from images
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
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

n_categ = 2

model = tf.keras.models.load_model(sys.argv[1]
                                   ,custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
#model = tf.keras.models.load_model(sys.argv[1])

# LOAD IMAGES
data_dir = '../multiclass_seg/x-sensing/'
_,_,test,info = data.load_images(data_dir,(256,256),n_labels=n_categ,mask_channels=1)
ntest = info['test_count']
ntrain = info['train_count']
print(f'total {ntest} testing images')

ntake = 6
thresholds = np.array([ [20.,1.],[10.,1.],[5.,1.],
                        [3.,1.],[2.,1.],[1,1.],
                        [1.,2.],[1.,3.],[1.,5.],
                        [1.,10.],[1.,20.],[1.,100.]])

#thresholds = np.array([[0.,0.2,0.4,0.6,0.8,1.]])
#thresholds = np.array([[1.,1.]])
                        
nthrsh = thresholds.shape[0]
#thresholds = np.ones((nthrsh,n_categ))
#thresholds = [None]
                        
image_arr = outputs.show_predictions(model,test.shuffle(1000),ntake,interactive=True)

error_matrix = np.zeros((nthrsh,n_categ,n_categ))

mask_frac = np.zeros((nthrsh,ntest,n_categ))
pred_frac = np.zeros((nthrsh,ntest,n_categ))

for i, t in enumerate(thresholds):
    weights = t
    print(f'%{i}th threshold, {t}')
    #TODO: this is really inefficient, will separate create_mask into output
    #  generation and prediction, to generate one mask and apply several weights
    j = 0
    for image, mask in test.shuffle(1000):
        #if j%25==0: print(j)
        #print(np.unique(mask.numpy().reshape(65536,n_categ),return_counts=True,axis=0))
        pred, mask = outputs.create_mask(model,image,mask,weights=weights)
        

        mbuf = mask.numpy()
        pbuf = pred.numpy()
        vals,_ = tf.unique(mbuf.flatten())
        valsp,_ = tf.unique(pbuf.flatten())

        for n_m in range(n_categ):
            #PER IMAGE SUMS
            mask_frac[i,j,n_m] = np.sum(mbuf==n_m)/mbuf.size
            pred_frac[i,j,n_m] = np.sum(pbuf==n_m)/pbuf.size
            
            # ERROR MATRIX
            for n_p in range(n_categ):
                buf = np.sum(np.logical_and(mbuf==n_m,pbuf==n_p))
                error_matrix[i,n_m,n_p] += buf

        j = j + 1

xbase = np.linspace(0,1,num=50)
zero = np.zeros(xbase.shape)
one = np.ones(xbase.shape)

pred_sums = error_matrix.sum(axis=1)[:,None,:]
mask_sums = error_matrix.sum(axis=2)[:,:,None]

fals_pos = error_matrix[:,0,1]
true_pos = error_matrix[:,1,1]

#false pos / pred pos
PRECISION = true_pos/pred_sums[:,0,1]
#true pos / mask pos
RECALL = true_pos/mask_sums[:,1,0]
print(PRECISION)
print(RECALL)

#FPR = [0,1]/([0,1]+ [0,0]) == false positives / mask negatives
#TPR = [1,1]/([1,1] + [1,0]) == true positives / mask positives

FPR = fals_pos/mask_sums[:,0,0]
TPR = true_pos/mask_sums[:,1,0]

total_acc = (error_matrix[:,0,0]+error_matrix[:,1,1])/error_matrix.sum(axis=(1,2))

np.set_printoptions(precision=5,suppress=True)
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
print(mask_frac.mean(axis=1))
print("------------")
print("pred fractions")
print(pred_frac.mean(axis=1))
print("------------")
print("accuracy")
print(total_acc)
print("------------")

print('histogram')
print(np.histogram(mask_frac[:,:,1],bins=np.linspace(0,1,num=11))[0])

if nthrsh > 1:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(FPR,TPR,color='blue', marker='o'
        , linestyle='solid', linewidth=2, markersize=5)
    ax.set_xlabel('% False Positive Rate')
    ax.set_ylabel('% True Positive Rate')

    fig.savefig('./plots/ROC.png')
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(PRECISION,RECALL,color='blue', marker='o'
        , linestyle='solid', linewidth=2, markersize=5)
    ax.set_xlabel('% Precision')
    ax.set_ylabel('% Recall')

    fig.savefig('./plots/ROC_2.png')

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(ncols=1,nrows=1,figure=fig)
ax = fig.add_subplot(gs[0])
ax.plot(xbase,xbase,'k:')
for i in range(nthrsh):
    if np.any(thresholds[i,:] != 1):
        continue
    xplot = mask_frac[i,...,1]
    yplot = (pred_frac[i,...,1])
    
    print(yplot.min(),yplot.max(),yplot.mean())
    ax.scatter(xplot,yplot,s=2)

ax.set_ylabel('predicted glare (% of image)')
ax.set_xlabel('glare in mask (% of image)')

fig.savefig('./plots/fractions.png')

plt.show()