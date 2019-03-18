import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


text_file=open('/home/uqfribe1/PycharmProjects/DEEP-fMRI/output/test_out_model4_1000_nothresh_6layers.txt')
lines=text_file.readlines()
epochs=[]
MSE=[]
MAE=[]

for x in lines:
    epochs.append(x.split(',')[0][7:])
    MSE.append(x.split(',')[1][8:])
    MAE.append(x.split(',')[2][7:][:-1])
text_file.close()

for i in range(len(MSE)):
    MSE[i]=float(MSE[i])
    MAE[i]=float(MAE[i])


plt.scatter(np.arange(1000),MAE[0:1000])
plt.show()