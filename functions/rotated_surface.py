import numpy as np

# Explain

rotated_info = open('/home/uqfribe1/Desktop/Rotation/SurfToSurf.1D')
lines = rotated_info.readlines()

def rotate_roi(index_L_mask):
    new_index = []
    old_index = []
    for i in range(14,len(lines)):
        old_index.append(int(lines[i][:6].strip()))
        new_index.append(int(lines[i][6:-4].strip()))

    new_L_mask_indexes = []
    for j in range(len(index_L_mask)):
        new_L_mask_indexes.append(new_index[index_L_mask[j]])

    mask = np.zeros(32492,)
    mask[new_L_mask_indexes] = 1
    return mask, new_L_mask_indexes


# #
# # Selecting all visual areas (Wang2015) plus V1-3 fovea
# import os.path as osp
# import scipy.io
# import torch
#
# from torch_geometric.data import InMemoryDataset
# from read.read_HCPdata_rotated import read_HCP
# from functions.labels import labels
# from functions.def_ROIs_WangParcelsPlusFovea import roi
# from functions.rotated_surface import rotate_roi
#
# path = '/home/uqfribe1/PycharmProjects/deepRetinotopy/data/raw/converted'
# label_primary_visual_areas = ['ROI']
# final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
#     label_primary_visual_areas)
#
# faces_R = labels(scipy.io.loadmat(osp.join(path, 'tri_faces_R.mat'))[
#                      'tri_faces_R'] - 1, index_R_mask)
# faces_L = labels(scipy.io.loadmat(osp.join(path, 'tri_faces_L.mat'))[
#                      'tri_faces_L'] - 1, index_L_mask)
#
#
# rot_roi, bla = rotate_roi(index_L_mask)
#
# roi = np.reshape(np.where(rot_roi == 1),(-1,1))
#
# mask_corrected = []
# for i in range(len(roi)):
#     mask_corrected.append(np.where(bla == roi[i]))
# np.reshape(mask_corrected,(-1,))
#
# data = read_HCP(path, Hemisphere='Left', index=0,
#                 surface='mid', visual_mask_L=final_mask_L,rotated_mask_L=rot_roi,
#                 visual_mask_R=final_mask_R, faces_L=faces_L,
#                 faces_R=faces_R, myelination=True,
#                 prediction='polarAngle')
# import numpy as np
# import scipy.io
# import os.path as osp
# import torch
#
# from functions.def_ROIs_WangParcelsPlusFovea import roi
# from nilearn import plotting
# from functions.rotated_surface import rotate_roi
# from functions.labels import labels
#
# path = '/home/uqfribe1/PycharmProjects/DEEP-fMRI/data/raw/converted'
# curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
# background = np.reshape(curv['x191841_curvature'][0][0][0:32492], (-1))
# threshold = 1  # threshold for the curvature map
#
# label_primary_visual_areas = ['ROI']
# final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
#     label_primary_visual_areas)
#
# faces_R = labels(scipy.io.loadmat(osp.join(path, 'tri_faces_R.mat'))[
#                      'tri_faces_R'] - 1, index_R_mask)
# faces_L = labels(scipy.io.loadmat(osp.join(path, 'tri_faces_L.mat'))[
#                      'tri_faces_L'] - 1, index_L_mask)
#
#
# rot_roi, bla = rotate_roi(index_L_mask)
#
# roi = np.reshape(np.where(rot_roi == 1),(-1,1))
# mask_corrected = []
# for i in range(len(roi)):
#     mask_corrected.append(np.where(bla == roi[i]))
# mask_corrected = np.reshape(mask_corrected,(-1,))
#
#
#
#
# pred = np.zeros((32492, 1))
#
# pred[rot_roi == 1] = np.reshape(np.array(test_dataset[0].x).T[0],(-1,1)) + threshold + 2
# pred[mask_corrected] = np.reshape(np.array(test_dataset[0].x).T[0],(-1,1)) + threshold + 2
#
#
# # Empirical map
# view = plotting.view_surf(
#     surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)),
#                        'data/raw/original/S1200_7T_Retinotopy_9Zkk'
#                        '/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k'
#                        '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
#     surf_map=np.reshape(pred[0:32492], (-1)),
#      cmap='gist_gray', black_bg=False, symmetric_cmap=False,
#      vmin=1, vmax=3.5,
#      threshold=threshold)
# view.open_in_browser()