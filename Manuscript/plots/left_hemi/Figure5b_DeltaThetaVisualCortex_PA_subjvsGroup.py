import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
import scipy.io
import os.path as osp
import torch

from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.functions.error_metrics import smallest_angle

subject_index = 0

hcp_id=['617748','191336','572045','725751','198653',
        '601127','644246','191841','680957','157336']

path = './../../../Retinotopy/data/raw/converted'
curv = scipy.io.loadmat(osp.join(path, 'cifti_curv_all.mat'))['cifti_curv']
background = np.reshape(curv['x'+hcp_id[subject_index]+'_curvature'][0][0][0:32492], (-1))

threshold = 1

nocurv = np.isnan(background)
background[nocurv == 1] = 0

# Predictions generated with 4 sets of features (pred = intact features)
models = ['pred', 'rotatedROI', 'shuffled-myelincurv', 'constant']

mean_delta = [] # error
mean_across = [] # individual variability
for m in range(len(models)):
    prediction = torch.load(
        './../../testset_results/left_hemi/testset-' +
        models[m] + '_deepRetinotopy_PA_LH.pt', map_location='cpu')

    theta_withinsubj = []
    theta_acrosssubj_pred = []

    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        label_primary_visual_areas)
    ROI1 = np.zeros((32492, 1))
    ROI1[final_mask_L == 1] = 1

    mask = ROI1
    mask = mask[ROI1 == 1]

    theta_pred_across_temp = []
    for i in range(len(prediction['Predicted_values'])):
        # Compute angle between predicted and empirical predictions within subj
        if i == subject_index:
            # Loading predicted values
            pred = np.reshape(np.array(prediction['Predicted_values'][i]), (-1, 1))
            measured = np.reshape(np.array(prediction['Measured_values'][subject_index]), (-1, 1))

            # Rescaling polar angles to match the right visual field (
            # left hemisphere)
            minus = pred > 180
            sum = pred < 180
            pred[minus] = pred[minus] - 180
            pred[sum] = pred[sum] + 180
            pred = np.array(pred) * (np.pi / 180)

            minus = measured > 180
            sum = measured < 180
            measured[minus] = measured[minus] - 180
            measured[sum] = measured[sum] + 180
            measured = np.array(measured) * (np.pi / 180)

            # Computing delta theta
            theta = smallest_angle(pred, measured)
            theta_withinsubj.append(theta) # Prediction error

        if i != subject_index:
            # Compute angle between predicted and empirical predictions
            # across subj
            # Loading predicted values
            pred = np.reshape(np.array(prediction['Predicted_values'][i]), (-1, 1))
            pred2 = np.reshape(np.array(prediction['Predicted_values'][subject_index]), (-1, 1))

            # Rescaling polar angles to match the correct visual field (left
            # hemisphere)
            minus = pred > 180
            sum = pred < 180
            pred[minus] = pred[minus] - 180
            pred[sum] = pred[sum] + 180
            pred = np.array(pred) * (np.pi / 180)

            minus = pred2 > 180
            sum = pred2 < 180
            pred2[minus] = pred2[minus] - 180
            pred2[sum] = pred2[sum] + 180
            pred2 = np.array(pred2) * (np.pi / 180)

            # Computing delta theta
            theta_pred = smallest_angle(pred, pred2)
            theta_pred_across_temp.append(theta_pred) # Individual variability

    theta_acrosssubj_pred.append(np.mean(theta_pred_across_temp, axis=0))

    mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
    mean_theta_acrosssubj_pred = np.mean(np.array(theta_acrosssubj_pred),
                                         axis=0)

    mean_delta.append(mean_theta_withinsubj[mask == 1])
    mean_across.append(mean_theta_acrosssubj_pred[mask == 1])

mean_delta = np.reshape(np.array(mean_delta), (len(models), -1))
mean_across = np.reshape(np.array(mean_across), (len(models), -1))

# Generating plots
# Select predictions generated with a given set of features
model_index = np.where(np.array(models) == 'pred')

# Region of interest
delta_theta = np.ones((32492, 1))
delta_theta[final_mask_L == 1] = (np.reshape(mean_delta[model_index],
                                             (3267, 1)) + threshold)
delta_theta[final_mask_L != 1] = 0

delta_across = np.ones((32492, 1))
delta_across[final_mask_L == 1] = (np.reshape(mean_across[model_index],
                                              (3267, 1)) + threshold)
delta_across[final_mask_L != 1] = 0

# Error map
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(delta_theta[0:32492], (-1)), bg_map=background,
    cmap='Reds', black_bg=False, symmetric_cmap=False, threshold=threshold,
    vmax=75 + threshold)
view.open_in_browser()

# Individual variability map
view = plotting.view_surf(
    surf_mesh=osp.join(osp.dirname(osp.realpath(__file__)), '../../..',
                       'Retinotopy/data/raw/surfaces'
                       '/S1200_7T_Retinotopy181.L.sphere.32k_fs_LR.surf.gii'),
    surf_map=np.reshape(delta_across[0:32492], (-1)), bg_map=background,
    cmap='Blues', black_bg=False, symmetric_cmap=False, threshold=threshold,
    vmax=75 + threshold)
view.open_in_browser()
