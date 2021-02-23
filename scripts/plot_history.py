import numpy as np
from matplotlib import pyplot as plt

#cols: 0=epoch, 1=accuracy, 2=loss, 3=val_accuracy, 4=val_loss
history = np.loadtxt('log_retiled_binary.out',dtype=float,delimiter=',',comments='#',ndmin=2)

epochs = history[:,0]
acc = history[:,1]
loss = history[:,2]
val_acc = history[:,3]
val_loss = history[:,4]

acc = np.log10(1 - acc)
val_acc = np.log10(1 - val_acc)
loss = np.log10(loss)
val_loss = np.log10(val_loss)

fig = plt.figure()
ax=fig.add_subplot(211)
ax.plot(epochs, loss, 'r', label='Training')
ax.plot(epochs, val_loss, 'bo', label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('Log10 (Loss Value)')
#ax.set_ylim([0,loss[0]])
ax.legend()
ax=fig.add_subplot(212)
ax.plot(epochs, acc, 'r', label='Training')
ax.plot(epochs, val_acc, 'bo', label='Validation')
ax.set_xlabel('Epoch')
ax.set_ylabel('Log10 (1 - Accuracy)')
#ax.set_ylim([0,1])
ax.legend()
fig.tight_layout()
fig.savefig('./plots/history_binary.png')

