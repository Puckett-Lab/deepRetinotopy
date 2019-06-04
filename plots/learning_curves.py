import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


text_file=open('/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle/output4_lr_models.txt')
#text_file=open('/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle/output_3105-2_2_model16noDO.txt')
lines=text_file.readlines()
epochs=[]
MAE_train=[]
MAE_test=[]

for x in lines:
    epochs.append(x.split(',')[0][7:])
    MAE_train.append(x.split(',')[2][12:])
    MAE_test.append(x.split(',')[3][11:][:-1])
text_file.close()

for i in range(len(MAE_train)):
    MAE_train[i]=float(MAE_train[i])
    MAE_test[i]=float(MAE_test[i])

plt.figure()
#plt.scatter(np.arange(1000),MAE_train[1000:2000],marker='.')
plt.scatter(np.arange(1000),MAE_train[0:1000],marker='.')
#plt.scatter(np.arange(1000),MAE_test[1000:2000],marker='.')
plt.scatter(np.arange(1000),MAE_test[0:1000],marker='.')




#plt.scatter(np.arange(1000),MAE_train[0:1000],marker='.')
#plt.scatter(np.arange(1000),MAE_test[0:1000],color='red',marker='.')

plt.xlim(0,  1000)
plt.ylim(0, 100)
plt.axhline(y=28, color='r', linestyle='-')
plt.show()