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
from scipy.integrate import simps

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

n_categ = 1

model = tf.keras.models.load_model(sys.argv[1]
                                   ,custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
#model = tf.keras.models.load_model(sys.argv[1])

# LOAD IMAGES
data_dir = '../multiclass_seg/another_one/'
_,_,test,info = data.load_images(data_dir,(256,256),n_labels=n_categ,mask_channels=1)
ntest = info['test_count']
print(f'total {ntest} testing images')

sigmoid = lambda x : 1/(1+np.exp(-x))

ntake = 6                    
nthrsh = 40
thrshmax = 4
thrshmin = -14
if n_categ > 1:
    thresholds = np.logspace(np.log10(thrshmin),np.log10(thrshmax),num=nthrsh,base=10)
    buf = np.ones(thresholds.shape)
    #thresholds = np.concatenate(thresholds,buf,axis=1)
    #thresholds = np.insert(thresholds,0,np.array([0,1]),axis=0)
    #thresholds = np.append(thresholds,np.array([1,0]),axis=0)
else:
    thresholds = np.linspace(thrshmin,thrshmax,nthrsh-2)
    thresholds = sigmoid(thresholds)
    thresholds = np.insert(thresholds,0,0)
    thresholds = np.append(thresholds,1)

print(f'thresholds = {thresholds}')
#thresholds = np.array([0.19])
#nthrsh = 1

image_arr = outputs.show_predictions(model,test.shuffle(5000),ntake,interactive=True,weights=np.array([0.19]))
quit()

n_dim = max(2,n_categ)

error_matrix = np.zeros((nthrsh,n_dim,n_dim),dtype=int)

mask_frac = np.zeros((nthrsh,ntest,n_dim))
pred_frac = np.zeros((nthrsh,ntest,n_dim))

j = 0
#TODO: batches of images instead of singles
for image, mask in test:
    if j%100==0: print(j)
    #print(np.unique(mask.numpy().reshape(65536,n_categ),return_counts=True,axis=0))
    pred, truth = outputs.create_mask(model,image,mask,weights=thresholds)

    for n_m in range(n_dim):
        #PER IMAGE SUMS
        #print(np.sum(truth==n_m,axis=(1,2,3)))
        #print(np.sum(pred==n_m,axis=(1,2,3)))
        mask_frac[:,j,n_m] = np.sum(truth==n_m,axis=(1,2,3))/np.product(truth.shape[1:])
        pred_frac[:,j,n_m] = np.sum(pred==n_m,axis=(1,2,3))/np.product(pred.shape[1:])
            
        # ERROR MATRIX
        for n_p in range(n_dim):
            buf = np.sum(np.logical_and(truth==n_m,pred==n_p),axis=(1,2,3))
            error_matrix[:,n_m,n_p] += buf

    j = j + 1

xbase = np.linspace(0,1,num=50)

pred_sums = error_matrix.sum(axis=1)
mask_sums = error_matrix.sum(axis=2)

fals_pos = error_matrix[:,0,1]
true_pos = error_matrix[:,1,1]

#Precision == true pos / pred pos
PRECISION = true_pos/pred_sums[:,1]
#mark no positives as 1 precision (no false positives)
PRECISION[np.isnan(PRECISION)] = 1
#Recall == true pos / mask pos
RECALL = true_pos/mask_sums[:,1]
print("Precision")
print(PRECISION*100)
print("Recall")
print(RECALL*100)

#FPR = [0,1]/([0,1]+ [0,0]) == false positives / mask negatives
#TPR = [1,1]/([1,1] + [1,0]) == true positives / mask positives

FPR = fals_pos/mask_sums[:,0]
TPR = true_pos/mask_sums[:,1]

total_acc = (error_matrix[:,0,0]+error_matrix[:,1,1])/error_matrix.sum(axis=(1,2))

print(FPR*100)

np.set_printoptions(precision=5,suppress=True)
print("total pixels (row == real) (col == pred)")
#print(error_matrix)
print("------mask norm------")
#print(error_matrix/mask_sums[:,:,None])
print("-----pred norm-------")
#print(error_matrix/pred_sums[:,None,:])
print("------------")
print("mask fractions")
#print(mask_frac.mean(axis=1))
print("------------")
print("pred fractions")
#print(pred_frac.mean(axis=1))
print("------------")
print("accuracy")
#print(total_acc)
print("------------")

#remove same-x values from integration
pvr_start = 0
pvr_end = nthrsh
if np.any(RECALL == 1):
    pvr_start = np.nonzero(RECALL == 1)[0][-1]
if np.any(RECALL == 0):
    pvr_end = np.nonzero(RECALL == 0)[0][0]
roc_start = 0
roc_end = nthrsh
if np.any(FPR == 1):
    roc_start = np.nonzero(FPR == 1)[0][-1]
if np.any(FPR == 0):
    roc_end = np.nonzero(FPR == 0)[0][0]

print(pvr_start,pvr_end)
print(roc_start,roc_end)
PRECISION = PRECISION[pvr_start:pvr_end]
RECALL = RECALL[pvr_start:pvr_end]
TPR = TPR[roc_start:roc_end]
FPR = FPR[roc_start:roc_end]

AUC_ROC = simps(TPR[::-1],FPR[::-1])
AUC_PVR = simps(PRECISION[::-1],RECALL[::-1])

print(f"AUC for ROC = {AUC_ROC} ||  for PVR = {AUC_PVR}")

#print('histogram')
#print(np.histogram(mask_frac[:,:,1],bins=np.linspace(0,1,num=11))[0])

if nthrsh > 1:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(FPR,TPR,color='blue', marker='o'
        , linestyle='solid', linewidth=2, markersize=5)
    ax.set_xlabel('% False Positive Rate')
    ax.set_ylabel('% True Positive Rate')
    ax.set_title(f'ROC curve, AUC = {AUC_ROC}')

    #fig.savefig('./plots/ROC.png')
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(RECALL,PRECISION,color='blue', marker='o'
        , linestyle='solid', linewidth=2, markersize=5)
    ax.set_ylabel('% Precision')
    ax.set_xlabel('% Recall')
    ax.set_title(f'Precision vs Recall curve, AUC = {AUC_PVR}')

    #fig.savefig('./plots/ROC_2.png')

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(ncols=1,nrows=1,figure=fig)
ax = fig.add_subplot(gs[0])
ax.plot(xbase,xbase,'k:')
for i in range(nthrsh):
    if np.any(thresholds[i] != 1):
        continue
    xplot = mask_frac[i,...,1]
    yplot = (pred_frac[i,...,1])
    
    ax.scatter(xplot,yplot,s=2)

ax.set_ylabel('predicted glare (% of image)')
ax.set_xlabel('glare in mask (% of image)')

#fig.savefig('./plots/fractions.png')

plt.show()