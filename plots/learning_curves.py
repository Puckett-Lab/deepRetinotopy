import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


text_file=open('/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle/output_models5l_batchnorm.txt')
lines=text_file.readlines()
epochs=[]
MSE_train=[]
MSE_test=[]

for x in lines:
    epochs.append(x.split(',')[0][7:])
    MSE_train.append(x.split(',')[2][12:])
    MSE_test.append(x.split(',')[3][11:][:-1])
text_file.close()

for i in range(len(MSE_train)):
    MSE_train[i]=float(MSE_train[i])
    MSE_test[i]=float(MSE_test[i])


plt.scatter(np.arange(1000),MSE_test[0:1000])
plt.xlim(0,  1000)
plt.ylim(20, 100)
plt.show()