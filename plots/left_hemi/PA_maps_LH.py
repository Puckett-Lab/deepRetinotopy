import numpy as np
import scipy.io
import os.path as osp
import torch

from functions.def_ROIs_WangParcelsPlusFovea import roi
from nilearn import plotting

subject_index = 7

hcp_id = ['617748', '191336', '572045', '725751', '198653',
          '601127', '644246', '191841', '680957', '157336']

path = './../../data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background = np.reshape(
    curv['x' + hcp_id[subject_index] + '_curvature'][0][0][0:32492], (-1))

threshold = 1  # threshold for the curvature map

# Background settings
nocurv = np.isnan(background)
background[nocurv == 1] = 0
background[background < 0] = 0
background[background > 0] = 1

# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)

pred = np.zeros((32492, 1))
measured = np.zeros((32492, 1))

# curv_plot = background[final_mask_L == 1] # Plotting the curvature values


# Loading the predictions
predictions = torch.load(
    './../../testset_results/left_hemi'
    '/testset-pred_deepRetinotopy_PA_LH.pt',
    map_location='cpu')

pred[final_mask_L == 1] = np.reshape(
    np.array(predictions['Predicted_values'][subject_index]),
    (-1, 1))

measured[final_mask_L == 1] = np.reshape(
    np.array(predictions['Measured_values'][subject_index]),
    (-1, 1))

# # To generate the mean predicted and mean empirical maps, just uncomment the
# # following lines:
# pred_mean = []
# measured_mean = []
# for i in range(10):
#     pred_mean.append(np.reshape(np.array(predictions['Predicted_values'][i]),
#                                 (-1, 1)))
#     measured_mean.append(np.reshape(np.array(predictions[
#     'Measured_values'][i]),
#                                              (-1, 1)))
# pred[final_mask_L == 1] = np.mean(pred_mean, 0)
# measured[final_mask_L == 1] = np.mean(measured_mean, 0)


# Rescaling
pred = np.array(pred)
minus = pred > 180
sum = pred < 180
pred[minus] = pred[minus] - 180 + threshold
pred[sum] = pred[sum] + 180 + threshold
pred = np.array(pred)

measured = np.array(measured)
minus = measured > 180
sum = measured < 180
measured[minus] = measured[minus] - 180 + threshold
measured[sum] = measured[sum] + 180 + threshold
measured = np.array(measured)

# Masking
measured[final_mask_L != 1] = 0
pred[final_mask_L != 1] = 0

# Empirical map
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                       'data/raw/original/S1200_7T_Retinotopy_9Zkk'
                       '/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(measured[0:32492], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=361)
view.open_in_browser()

# Predicted map
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../..',
                       'data/raw/original/S1200_7T_Retinotopy_9Zkk'
                       '/S1200_7T_Retinotopy181/MNINonLinear/fsaverage_LR32k'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(pred[0:32492], (-1)), bg_map=background,
    cmap='gist_rainbow_r', black_bg=False, symmetric_cmap=False,
    threshold=threshold, vmax=361)
view.open_in_browser()
