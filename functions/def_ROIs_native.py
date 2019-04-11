import numpy as np
import os.path as osp
import scipy.io


def roi(list_of_labels):
    # Defining number of nodes


    list_primary_visual_areas_L = np.zeros([6,136849])
    list_primary_visual_areas_R = np.zeros([6,139486])

    for i in range(len(list_of_labels)):
        list_primary_visual_areas_L[i] = np.reshape(scipy.io.loadmat(
            osp.join(osp.dirname(osp.realpath(__file__)), '..', 'labels/102311_native_Wang2015',
                     list_of_labels[i] + '_label_L_102311_native.mat'))[list_of_labels[i]+'_L'],(-1))
        list_primary_visual_areas_R[i] = np.reshape(scipy.io.loadmat(
            osp.join(osp.dirname(osp.realpath(__file__)), '..', 'labels/102311_native_Wang2015',
                     list_of_labels[i] + '_label_R_102311_native.mat'))[list_of_labels[i] + '_R'], (-1))


    final_mask_L = np.sum(list_primary_visual_areas_L, axis=0)
    final_mask_R = np.sum(list_primary_visual_areas_R, axis=0)

    index_L_mask = [i for i, j in enumerate(final_mask_L) if j == 1]
    index_R_mask = [i for i, j in enumerate(final_mask_R) if j == 1]

    return final_mask_L, final_mask_R, index_L_mask, index_R_mask
