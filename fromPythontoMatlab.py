import scipy.io
import torch
import numpy as np
import os.path as osp

path='/home/uqfribe1/PycharmProjects/pytorch_geometric/my_stuff/my_own_data/raw/my_version_retinotopy'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']

number_cortical_nodes = int(64984)
number_hemi_nodes = int(number_cortical_nodes / 2)

with open(osp.join(path, 'list_subj')) as fp:
    subjects = fp.read().split("\n")
subjects = subjects[0:len(subjects) - 1]
'''
output = np.array(torch.tensor(np.reshape(curv['x' + subjects[0] + '_curvature'][0][0][number_hemi_nodes:number_cortical_nodes].reshape(
                (number_hemi_nodes)), (-1, 1)), dtype=torch.float))

scipy.io.savemat('/home/uqfribe1/PycharmProjects/pytorch_geometric/my_stuff/testing',mdict={'output':output,'output2':output},appendmat=True)'''