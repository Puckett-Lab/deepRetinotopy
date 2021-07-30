import numpy as np
import os
import torch

from Retinotopy.functions.def_ROIs_WangParcels import roi as roi2
from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.functions.error_metrics import smallest_angle
from Retinotopy.functions.plusFovea import add_fovea_R


def PA_difference(model):
    """Function to determine the difference between empirical
    and predicted polar angle values for higher order visual areas, early
    visual cortex and dorsal V1-3.

    Args:
        model (str): 'deepRetinotopy' or 'average'

    Output: .npz files in ./../output named:
        'ErrorPerParticipant_PA_RH_WangParcels_' + str(model) + '_1-8.npz'
        'ErrorPerParticipant_PA_RH_EarlyVisualCortex_' + str(model) +
        '_1-8.npz'
        'ErrorPerParticipant_PA_RH_dorsalV1-3_' + str(model) + '_1-8.npz'

    """
    visual_areas = [
        ['hV4', 'VO1', 'VO2', 'PHC1', 'PHC2', 'V3a', 'V3b', 'LO1', 'LO2',
         'TO1',
         'TO2', 'IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1']]
    models = ['pred']

    eccentricity_mask = np.reshape(
        np.load('./../../plots/output/MaskEccentricity_'
                'above1below8ecc_RH.npz')['list'], (-1))
    # Average map
    PA_average = np.load('./../../plots/output/AveragePolarAngleMap_RH.npz')[
        'list']

    # Benson14 template
    Benson14_predictions = \
    np.load('./../../testset_results/benson14/benson14_testPrediction_PA_rh.npz')[
        'list']
    for k in range(len(visual_areas)):
        mean_delta = []
        for m in range(len(models)):
            predictions = torch.load(
                './../../testset_results/right_hemi/testset-' +
                models[m] + '_deepRetinotopy_PA_RH.pt', map_location='cpu')

            # ROI settings
            label_primary_visual_areas = ['ROI']
            final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
                label_primary_visual_areas)
            ROI1 = np.zeros((32492, 1))
            ROI1[final_mask_R == 1] = 1

            # Visual areas
            final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
                visual_areas[k])
            primary_visual_areas = np.zeros((32492, 1))
            primary_visual_areas[final_mask_R == 1] = 1

            # Final mask
            mask = ROI1 + primary_visual_areas
            mask = mask[ROI1 == 1]

            theta_withinsubj = []
            for j in range(len(predictions['Predicted_values'])):
                for i in range(len(predictions['Predicted_values'])):
                    if i == j:
                        # Loading predicted values
                        # Polar angles
                        if model == 'deepRetinotopy':
                            pred = np.reshape(
                                np.array(predictions['Predicted_values'][i]),
                                (-1, 1))
                        if model == 'average':
                            pred = np.reshape(np.array(PA_average), (-1, 1))
                        if model == 'Benson14':
                            pred = np.reshape(
                                np.array(Benson14_predictions[i]),
                                (-1, 1))
                        measured = np.reshape(
                            np.array(predictions['Measured_values'][j]),
                            (-1, 1))

                        # Rescaling
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
                        theta = smallest_angle(pred[eccentricity_mask],
                                               measured[eccentricity_mask])
                        theta_withinsubj.append(
                            theta[mask[eccentricity_mask] > 1])
            mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=1)
            mean_delta.append(mean_theta_withinsubj)
        mean_delta = np.reshape(np.array(mean_delta), (1, -1))

    np.savez('./../output/ErrorPerParticipant_PA_RH_WangParcels_' + str(
        model) + '_1-8.npz',
             list=np.reshape(theta_withinsubj, (10, -1)))

    # Early visual cortex
    label_primary_visual_areas = ['V1d', 'V1v', 'fovea_V1', 'V2d', 'V2v',
                                  'fovea_V2', 'V3d', 'V3v', 'fovea_V3']
    V1, V2, V3 = add_fovea_R(label_primary_visual_areas)
    primary_visual_areas = np.sum(
        [np.reshape(V1, (-1, 1)), np.reshape(V2, (-1, 1)),
         np.reshape(V3, (-1, 1))], axis=0)

    mean_delta_2 = []
    for m in range(len(models)):
        predictions = torch.load(
            './../../testset_results/right_hemi/testset-' +
            models[m] + '_deepRetinotopy_PA_RH.pt', map_location='cpu')

        # ROI settings
        label_primary_visual_areas = ['ROI']
        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
            label_primary_visual_areas)
        ROI1 = np.zeros((32492, 1))
        ROI1[final_mask_R == 1] = 1

        mask = ROI1 + np.reshape(primary_visual_areas, (32492, 1))
        mask = mask[ROI1 == 1]

        theta_withinsubj = []
        for j in range(len(predictions['Predicted_values'])):
            for i in range(len(predictions['Predicted_values'])):
                if i == j:
                    # Polar angle
                    if model == 'deepRetinotopy':
                        pred = np.reshape(
                            np.array(predictions['Predicted_values'][i]),
                            (-1, 1))
                    if model == 'average':
                        pred = np.reshape(np.array(PA_average), (-1, 1))
                    if model == 'Benson14':
                        pred = np.reshape(
                            np.array(Benson14_predictions[i]),
                            (-1, 1))
                    measured = np.reshape(
                        np.array(predictions['Measured_values'][j]),
                        (-1, 1))

                    # Rescaling
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
                    theta = smallest_angle(pred[eccentricity_mask],
                                           measured[eccentricity_mask])
                    theta_withinsubj.append(theta[mask[eccentricity_mask] > 1])
        mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=1)
        mean_delta_2.append(mean_theta_withinsubj)
    mean_delta_2 = np.reshape(np.array(mean_delta_2), (1, -1))

    np.savez('./../output/ErrorPerParticipant_PA_RH_EarlyVisualCortex_' + str(
        model) + '_1-8'
                 '.npz', list=np.reshape(theta_withinsubj, (10, -1)))

    # Dorsal early visual cortex
    visual_areas = [
        ['V1d', 'V2d', 'V3d']]
    for k in range(len(visual_areas)):
        mean_delta = []
        for m in range(len(models)):
            predictions = torch.load(
                './../../testset_results/right_hemi/testset-' +
                models[m] + '_deepRetinotopy_PA_RH.pt', map_location='cpu')

            # ROI seetings
            label_primary_visual_areas = ['ROI']
            final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
                label_primary_visual_areas)
            ROI1 = np.zeros((32492, 1))
            ROI1[final_mask_R == 1] = 1

            # Visual areas
            final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
                visual_areas[k])
            primary_visual_areas = np.zeros((32492, 1))
            primary_visual_areas[final_mask_R == 1] = 1

            # Final mask
            mask = ROI1 + primary_visual_areas
            mask = mask[ROI1 == 1]

            theta_withinsubj = []
            for j in range(len(predictions['Predicted_values'])):
                for i in range(len(predictions['Predicted_values'])):
                    if i == j:
                        # Polar angle
                        if model == 'deepRetinotopy':
                            pred = np.reshape(
                                np.array(predictions['Predicted_values'][i]),
                                (-1, 1))
                        if model == 'average':
                            pred = np.reshape(np.array(PA_average), (-1, 1))
                        if model == 'Benson14':
                            pred = np.reshape(
                                np.array(Benson14_predictions[i]),
                                (-1, 1))
                        measured = np.reshape(
                            np.array(predictions['Measured_values'][j]),
                            (-1, 1))

                        # Rescaling
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
                        theta = smallest_angle(pred[eccentricity_mask],
                                               measured[eccentricity_mask])
                        theta_withinsubj.append(
                            theta[mask[eccentricity_mask] > 1])
            mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=1)
            mean_delta.append(mean_theta_withinsubj)
        mean_delta = np.reshape(np.array(mean_delta), (1, -1))

    np.savez('./../output/ErrorPerParticipant_PA_RH_dorsalV1-3_' + str(
        model) + '_1-8.npz',
             list=np.reshape(theta_withinsubj, (10, -1)))


# Create an output folder if it doesn't already exist
directory = './../output'
if not os.path.exists(directory):
    os.makedirs(directory)

PA_difference('average')
PA_difference('deepRetinotopy')
PA_difference('Benson14')
