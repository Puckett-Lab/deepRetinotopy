import numpy as np
import torch
import scipy.stats
import pandas as pd

from Retinotopy.functions.def_ROIs_WangParcels import roi as roi2
from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.functions.plusFovea import add_fovea_R

models = ['pred']
eccentricity_mask = np.reshape(np.load('./../../plots/output/MaskEccentricity_'
                                       'above1below8ecc_RH.npz')['list'], (-1))

# Average map
ecc_average = np.load('./../../plots/output/AverageEccentricityMap_RH.npz')[
    'list']
# DeepRetinotopy predictions
predictions = torch.load(
    './../../testset_results/right_hemi/testset-pred_deepRetinotopy_ecc_RH.pt',
    map_location='cpu')  #

# Create lists with empirical and predicted (average or deepRetinotopy) maps
empirical_maps = []
predicted_maps_deepRetinotopy = []
predicted_maps_average = []
for i in range(10):
    empirical_maps.append(np.array(predictions['Measured_values'][i]))
    predicted_maps_deepRetinotopy.append(
        np.array(predictions['Predicted_values'][i]))
    predicted_maps_average.append(np.reshape(ecc_average.T, (-1)))

empirical_maps = np.array(empirical_maps).T
predicted_maps_deepRetinotopy = np.array(predicted_maps_deepRetinotopy).T
predicted_maps_average = np.array(predicted_maps_average).T

# Compute vertex-wise correlation between predicted and empirical values
# across participants
vertex_wise_corr_deepRetinotopy = []
vertex_wise_corr_average = []
for i in range(len(predicted_maps_average)):
    vertex_wise_corr_average.append(
        scipy.stats.pearsonr(empirical_maps[i], predicted_maps_average[i])[0])
    vertex_wise_corr_deepRetinotopy.append(
        scipy.stats.pearsonr(empirical_maps[i],
                             predicted_maps_deepRetinotopy[i])[0])

vertex_wise_corr_deepRetinotopy = np.array(vertex_wise_corr_deepRetinotopy)
vertex_wise_corr_average = np.array(vertex_wise_corr_average)

# ROI settings
label_primary_visual_areas = ['ROI']
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
    label_primary_visual_areas)
ROI1 = np.zeros((32492, 1))
ROI1[final_mask_R == 1] = 1

# Higher order visual areas
# Visual areas
visual_areas = [
    ['hV4', 'VO1', 'VO2', 'PHC1', 'PHC2', 'V3a', 'V3b', 'LO1', 'LO2', 'TO1',
     'TO2', 'IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1']]
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
    visual_areas[0])
higher_order_visual_areas = np.zeros((32492, 1))
higher_order_visual_areas[final_mask_R == 1] = 1

# Final mask
mask = ROI1 + higher_order_visual_areas
mask = mask[ROI1 == 1]

# Explained variance
vertex_wise_expVar_model_higherOrder = \
    vertex_wise_corr_deepRetinotopy[eccentricity_mask][
        mask[eccentricity_mask] > 1] ** 2
vertex_wise_expVar_average_higherOrder = \
    vertex_wise_corr_average[eccentricity_mask][
        mask[eccentricity_mask] > 1] ** 2

# Early visual cortex
label_primary_visual_areas = ['V1d', 'V1v', 'fovea_V1', 'V2d', 'V2v',
                              'fovea_V2', 'V3d', 'V3v', 'fovea_V3']
V1, V2, V3 = add_fovea_R(label_primary_visual_areas)
primary_visual_areas = np.sum(
    [np.reshape(V1, (-1, 1)), np.reshape(V2, (-1, 1)),
     np.reshape(V3, (-1, 1))], axis=0)

# Final mask
mask = ROI1 + np.reshape(primary_visual_areas, (32492, 1))
mask = mask[ROI1 == 1]

# Explained variance
vertex_wise_expVar_model_earlyVisualCortex = \
    vertex_wise_corr_deepRetinotopy[eccentricity_mask][
        mask[eccentricity_mask] > 1] ** 2
vertex_wise_expVar_average_earlyVisualCortex = \
    vertex_wise_corr_average[eccentricity_mask][
        mask[eccentricity_mask] > 1] ** 2

# Dorsal early visual cortex
visual_areas = [
    ['V1d', 'V2d', 'V3d']]
final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
    visual_areas[0])
primary_visual_areas = np.zeros((32492, 1))
primary_visual_areas[final_mask_R == 1] = 1

# Final mask
mask = ROI1 + primary_visual_areas
mask = mask[ROI1 == 1]

# Explained variance
vertex_wise_expVar_model_dorsalEarlyVisualCortex = \
    vertex_wise_corr_deepRetinotopy[eccentricity_mask][
        mask[eccentricity_mask] > 1] ** 2
vertex_wise_expVar_average_dorsalEarlyVisualCortex = \
    vertex_wise_corr_average[eccentricity_mask][
        mask[eccentricity_mask] > 1] ** 2

summary = pd.DataFrame({'Region of interest': ['Dorsal V1-3 - Average',
                                               'Early visual cortex - '
                                               'Average',
                                               'Higher order visual '
                                               'areas - Average',
                                               'Dorsal V1-3 - Model',
                                               'Early visual cortex - '
                                               'Model',
                                               'Higher order visual '
                                               'areas - Model'
                                               ],
                        'Mean': [np.mean(
                            vertex_wise_expVar_average_dorsalEarlyVisualCortex),
                            np.mean(
                                vertex_wise_expVar_average_earlyVisualCortex),
                            np.mean(
                                vertex_wise_expVar_average_higherOrder),
                            np.mean(
                                vertex_wise_expVar_model_dorsalEarlyVisualCortex),
                            np.mean(
                                vertex_wise_expVar_model_earlyVisualCortex),
                            np.mean(
                                vertex_wise_expVar_model_higherOrder)
                        ],
                        'Std': [np.std(
                            vertex_wise_expVar_average_dorsalEarlyVisualCortex),
                            np.std(
                                vertex_wise_expVar_average_earlyVisualCortex),
                            np.std(
                                vertex_wise_expVar_average_higherOrder),
                            np.std(
                                vertex_wise_expVar_model_dorsalEarlyVisualCortex),
                            np.std(
                                vertex_wise_expVar_model_earlyVisualCortex),
                            np.std(
                                vertex_wise_expVar_model_higherOrder)]}
                       )
print(summary)
