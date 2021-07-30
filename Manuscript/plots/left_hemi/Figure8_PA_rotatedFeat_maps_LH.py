import numpy as np
import scipy.io
import os.path as osp
import torch

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from nilearn import plotting
from Retinotopy.functions.rotated_surface import rotate_roi

subject_index = 7

hcp_id = ['617748', '191336', '572045', '725751', '198653',
          '601127', '644246', '191841', '680957', '157336']

path = './../../../Retinotopy/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background = np.reshape(
    curv['x' + hcp_id[subject_index] + '_curvature'][0][0][0:32492], (-1))

threshold = 1  # threshold for the curvature map

# Background setting
nocurv = np.isnan(background)
background[nocurv == 1] = 0
background[background < 0] = 0
background[background > 0] = 1

# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)

pred = np.zeros((32492, 1))

# Loading the predictions
predictions = torch.load(
    './../../testset_results/left_hemi'
    '/testset-rotatedROI_deepRetinotopy_PA_LH.pt',
    map_location='cpu')

pred[final_mask_L == 1] = np.reshape(
    np.array(predictions['Predicted_values'][subject_index]),
    (-1, 1))

# # To generate the mean predicted map, just uncomment the
# # following lines:
# pred_mean = []
# measured_mean = []
# for i in range(10):
#     pred_mean.append(np.reshape(np.array(predictions['Predicted_values'][i]),
#                                 (-1, 1)))
# pred[final_mask_L == 1] = np.mean(pred_mean, 0)

# Rescaling
pred = np.array(pred)
minus = pred > 180
sum = pred < 180
pred[minus] = pred[minus] - 180 + threshold
pred[sum] = pred[sum] + 180 + threshold
pred = np.array(pred)

# Masking
pred[final_mask_L != 1] = 0

# Predicted map
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(pred[0:32492], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=361)
view.open_in_browser()

# Plotting the curvature and myelin values
curvature = np.zeros((32492, 1))
curvature[final_mask_L == 1] = \
    np.reshape(
        np.reshape(np.array(predictions['Input'][subject_index]), (-1, 2)).T[
            0] + threshold + 2, (-1, 1))

# Curvature
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(curvature[0:32492], (-1)),
    cmap='gist_gray', black_bg=False, symmetric_cmap=False,
    vmin=1, vmax=3.5,
    threshold=threshold)
view.open_in_browser()

# Myelin
myelin = np.zeros((32492, 1))
myelin[final_mask_L == 1] = \
    np.reshape(
        np.reshape(np.array(predictions['Shuffled_myelin'][subject_index]), (-1, 2)).T[
            1] + threshold, (-1, 1))

view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf'
                       '.gii'),
    surf_map=np.reshape(myelin[0:32492], (-1)),
    cmap='gist_gray', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmin=1, vmax=3)
view.open_in_browser()
