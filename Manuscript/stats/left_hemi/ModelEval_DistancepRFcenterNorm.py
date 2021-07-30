import numpy as np
import torch
import os

from Retinotopy.functions.def_ROIs_WangParcels import roi as roi2
from Retinotopy.functions.def_ROIs_WangParcelsPlusFovea import roi
from Retinotopy.functions.plusFovea import add_fovea
from Retinotopy.functions.error_metrics import distance_PolarCoord


def pRF_distance(model):
    """Function to determine the difference between empirical
    and predicted pRF center location for higher order visual areas, early
    visual cortex and dorsal V1-3.

    Args:
        model (str): 'deepRetinotopy' or 'average'

    Output: .npz files in ./../output named:
        'ErrorPerParticipant_pRFcenter_LH_WangParcels_' + str(model) +
        '_1-8.npz'
        'ErrorPerParticipant_pRFcenter_LH_EarlyVisualCortex_' + str(model) +
        '_1-8.npz'
        'ErrorPerParticipant_pRFcenter_LH_dorsalV1-3_' + str(model) +
        '_1-8.npz'

    """
    visual_areas = [
        ['hV4', 'VO1', 'VO2', 'PHC1', 'PHC2', 'V3a', 'V3b', 'LO1', 'LO2',
         'TO1',
         'TO2', 'IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1']]
    models = ['pred']

    eccentricity_mask = np.reshape(
        np.load('./../../plots/output/MaskEccentricity_'
                'above1below8ecc_LH.npz')['list'], (-1))

    # Average maps
    PA_average = np.load('./../../plots/output/AveragePolarAngleMap_LH.npz')[
        'list']
    ecc_average = \
        np.load('./../../plots/output/AverageEccentricityMap_LH.npz')[
            'list']

    for k in range(len(visual_areas)):
        mean_delta = []
        for m in range(len(models)):
            predictions_PA = torch.load(
                './../../testset_results/left_hemi/testset-' +
                models[m] + '_deepRetinotopy_PA_LH.pt', map_location='cpu')
            predictions_ecc = torch.load(
                './../../testset_results/left_hemi/testset-' +
                models[m] + '_deepRetinotopy_ecc_LH.pt', map_location='cpu')

            # ROI settings
            label_primary_visual_areas = ['ROI']
            final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
                label_primary_visual_areas)
            ROI1 = np.zeros((32492, 1))
            ROI1[final_mask_L == 1] = 1

            # Visual areas
            final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
                visual_areas[k])
            higher_order_areas = np.zeros((32492, 1))
            higher_order_areas[final_mask_L == 1] = 1

            # Final mask
            mask = ROI1 + higher_order_areas
            mask = mask[ROI1 == 1]

            theta_withinsubj = []
            for j in range(len(predictions_PA['Predicted_values'])):
                for i in range(len(predictions_PA['Predicted_values'])):
                    if i == j:
                        # Loading predicted values
                        # Polar angles
                        if model == 'deepRetinotopy':
                            pred_PA = np.reshape(
                                np.array(
                                    predictions_PA['Predicted_values'][i]),
                                (-1, 1))
                        if model == 'average':
                            pred_PA = np.reshape(np.array(PA_average), (-1, 1))
                        measured_PA = np.reshape(
                            np.array(predictions_PA['Measured_values'][j]),
                            (-1, 1))

                        # Eccentricity
                        if model == 'deepRetinotopy':
                            pred_ecc = np.reshape(
                                np.array(
                                    predictions_ecc['Predicted_values'][i]),
                                (-1, 1))
                        if model == 'average':
                            pred_ecc = np.reshape(np.array(ecc_average),
                                                  (-1, 1))
                        measured_ecc = np.reshape(
                            np.array(predictions_ecc['Measured_values'][j]),
                            (-1, 1))

                        # Rescaling polar angles to match the right visual
                        # field
                        # (left hemisphere)
                        minus = pred_PA > 180
                        sum = pred_PA < 180
                        pred_PA[minus] = pred_PA[minus] - 180
                        pred_PA[sum] = pred_PA[sum] + 180
                        pred_PA = np.array(pred_PA) * (np.pi / 180)

                        minus = measured_PA > 180
                        sum = measured_PA < 180
                        measured_PA[minus] = measured_PA[minus] - 180
                        measured_PA[sum] = measured_PA[sum] + 180
                        measured_PA = np.array(measured_PA) * (np.pi / 180)

                        # Computing pRF center difference
                        # Changing ecc values below 1 to 1 to avoid infinity
                        # result
                        new_measured_ecc = np.zeros(np.shape(measured_ecc))
                        for i in range(len(measured_ecc)):
                            if measured_ecc[i] < 1:
                                new_measured_ecc[i] = 1
                            else:
                                new_measured_ecc[i] = measured_ecc[i]
                        # pRF center difference
                        distance = distance_PolarCoord(
                            new_measured_ecc[eccentricity_mask],
                            pred_ecc[eccentricity_mask],
                            measured_PA[[eccentricity_mask]],
                            pred_PA[[eccentricity_mask]]) / new_measured_ecc[
                                       eccentricity_mask]
                        theta_withinsubj.append(
                            distance[mask[eccentricity_mask] > 1])

            mean_theta_withinsubj = np.mean(np.array(theta_withinsubj),
                                            axis=1)  # mean with subj
            mean_delta.append(mean_theta_withinsubj)
        mean_delta = np.reshape(np.array(mean_delta), (1, -1))

    np.savez('./../output/ErrorPerParticipant_pRFcenter_LH_WangParcels_' + str(
        model) + '_1-8.npz', list=np.reshape(theta_withinsubj, (10, -1)))

    # Early visual cortex
    label_primary_visual_areas = ['V1d', 'V1v', 'fovea_V1', 'V2d', 'V2v',
                                  'fovea_V2', 'V3d', 'V3v', 'fovea_V3']
    V1, V2, V3 = add_fovea(label_primary_visual_areas)
    primary_visual_areas = np.sum(
        [np.reshape(V1, (-1, 1)), np.reshape(V2, (-1, 1)),
         np.reshape(V3, (-1, 1))], axis=0)

    mean_delta_2 = []
    for m in range(len(models)):
        predictions_PA = torch.load(
            './../../testset_results/left_hemi/testset-' +
            models[m] + '_deepRetinotopy_PA_LH.pt', map_location='cpu')
        predictions_ecc = torch.load(
            './../../testset_results/left_hemi/testset-' +
            models[m] + '_deepRetinotopy_ecc_LH.pt', map_location='cpu')

        # ROI settings
        label_primary_visual_areas = ['ROI']
        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
            label_primary_visual_areas)
        ROI1 = np.zeros((32492, 1))
        ROI1[final_mask_L == 1] = 1

        mask = ROI1 + np.reshape(primary_visual_areas, (32492, 1))
        mask = mask[ROI1 == 1]

        theta_withinsubj = []
        for j in range(len(predictions_PA['Predicted_values'])):
            for i in range(len(predictions_PA['Predicted_values'])):
                if i == j:
                    # Polar angle
                    if model == 'deepRetinotopy':
                        pred_PA = np.reshape(
                            np.array(predictions_PA['Predicted_values'][i]),
                            (-1, 1))
                    if model == 'average':
                        pred_PA = np.reshape(np.array(PA_average), (-1, 1))
                    measured_PA = np.reshape(
                        np.array(predictions_PA['Measured_values'][j]),
                        (-1, 1))

                    # Eccentricity
                    if model == 'deepRetinotopy':
                        pred_ecc = np.reshape(
                            np.array(predictions_ecc['Predicted_values'][i]),
                            (-1, 1))
                    if model == 'average':
                        pred_ecc = np.reshape(np.array(ecc_average), (-1, 1))
                    measured_ecc = np.reshape(
                        np.array(predictions_ecc['Measured_values'][j]),
                        (-1, 1))

                    # Rescaling polar angles to match the right visual field (
                    # left hemisphere)
                    minus = pred_PA > 180
                    sum = pred_PA < 180
                    pred_PA[minus] = pred_PA[minus] - 180
                    pred_PA[sum] = pred_PA[sum] + 180
                    pred_PA = np.array(pred_PA) * (np.pi / 180)

                    minus = measured_PA > 180
                    sum = measured_PA < 180
                    measured_PA[minus] = measured_PA[minus] - 180
                    measured_PA[sum] = measured_PA[sum] + 180
                    measured_PA = np.array(measured_PA) * (np.pi / 180)

                    # Computing pRF center difference
                    # Changing ecc values below 1 to 1 to avoid infinity result
                    new_measured_ecc = np.zeros(
                        np.shape(measured_ecc))
                    for i in range(len(measured_ecc)):
                        if measured_ecc[i] < 1:
                            new_measured_ecc[i] = 1
                        else:
                            new_measured_ecc[i] = measured_ecc[i]
                    # pRF center difference
                    distance = distance_PolarCoord(
                        new_measured_ecc[eccentricity_mask],
                        pred_ecc[eccentricity_mask],
                        measured_PA[
                            [eccentricity_mask]],
                        pred_PA[[eccentricity_mask]]) / new_measured_ecc[
                                   eccentricity_mask]

                    theta_withinsubj.append(
                        distance[mask[eccentricity_mask] > 1])
        mean_theta_withinsubj = np.mean(np.array(theta_withinsubj),
                                        axis=1)  # mean with subj
        mean_delta_2.append(mean_theta_withinsubj)
    mean_delta_2 = np.reshape(np.array(mean_delta_2), (1, -1))
    np.savez(
        './../output/ErrorPerParticipant_pRFcenter_LH_EarlyVisualCortex_' +
        str(
            model) + '_1-8.npz',
        list=np.reshape(theta_withinsubj, (10, -1)))

    # Dorsal early visual cortex
    visual_areas = [
        ['V1d', 'V2d', 'V3d']]
    for k in range(len(visual_areas)):
        mean_delta = []
        for m in range(len(models)):
            predictions_PA = torch.load(
                './../../testset_results/left_hemi/testset-' +
                models[m] + '_deepRetinotopy_PA_LH.pt', map_location='cpu')
            predictions_ecc = torch.load(
                './../../testset_results/left_hemi/testset-' +
                models[m] + '_deepRetinotopy_ecc_LH.pt', map_location='cpu')

            # ROI seetings
            label_primary_visual_areas = ['ROI']
            final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
                label_primary_visual_areas)
            ROI1 = np.zeros((32492, 1))
            ROI1[final_mask_L == 1] = 1

            # Visual areas
            final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
                visual_areas[k])
            primary_visual_areas = np.zeros((32492, 1))
            primary_visual_areas[final_mask_L == 1] = 1

            # Final mask
            mask = ROI1 + primary_visual_areas
            mask = mask[ROI1 == 1]

            theta_withinsubj = []
            for j in range(len(predictions_PA['Predicted_values'])):
                for i in range(len(predictions_PA['Predicted_values'])):

                    if i == j:
                        # Polar angle
                        if model == 'deepRetinotopy':
                            pred_PA = np.reshape(
                                np.array(
                                    predictions_PA['Predicted_values'][i]),
                                (-1, 1))
                        if model == 'average':
                            pred_PA = np.reshape(np.array(PA_average), (-1, 1))
                        measured_PA = np.reshape(
                            np.array(predictions_PA['Measured_values'][j]),
                            (-1, 1))

                        # Eccentricity
                        if model == 'deepRetinotopy':
                            pred_ecc = np.reshape(
                                np.array(
                                    predictions_ecc['Predicted_values'][i]),
                                (-1, 1))
                        if model == 'average':
                            pred_ecc = np.reshape(np.array(ecc_average),
                                                  (-1, 1))
                        measured_ecc = np.reshape(
                            np.array(predictions_ecc['Measured_values'][j]),
                            (-1, 1))

                        # Rescaling polar angles to match the right visual
                        # field
                        # (left hemisphere)
                        minus = pred_PA > 180
                        sum = pred_PA < 180
                        pred_PA[minus] = pred_PA[minus] - 180
                        pred_PA[sum] = pred_PA[sum] + 180
                        pred_PA = np.array(pred_PA) * (np.pi / 180)

                        minus = measured_PA > 180
                        sum = measured_PA < 180
                        measured_PA[minus] = measured_PA[minus] - 180
                        measured_PA[sum] = measured_PA[sum] + 180
                        measured_PA = np.array(measured_PA) * (np.pi / 180)

                        # Computing pRF center difference
                        # Changing ecc values below 1 to 1 to avoid infinity
                        # result
                        new_measured_ecc = np.zeros(np.shape(measured_ecc))
                        for i in range(len(measured_ecc)):
                            if measured_ecc[i] < 1:
                                new_measured_ecc[i] = 1
                            else:
                                new_measured_ecc[i] = measured_ecc[i]
                        # pRF center difference
                        distance = distance_PolarCoord(
                            new_measured_ecc[eccentricity_mask],
                            pred_ecc[eccentricity_mask],
                            measured_PA[
                                [eccentricity_mask]],
                            pred_PA[[eccentricity_mask]]) / new_measured_ecc[
                                       eccentricity_mask]
                        theta_withinsubj.append(
                            distance[mask[eccentricity_mask] > 1])
            mean_theta_withinsubj = np.mean(np.array(theta_withinsubj),
                                            axis=1)  # mean with subj
            mean_delta.append(mean_theta_withinsubj)
        mean_delta = np.reshape(np.array(mean_delta), (1, -1))
    np.savez('./../output/ErrorPerParticipant_pRFcenter_LH_dorsalV1-3_' + str(
        model) + '_1-8.npz', list=np.reshape(theta_withinsubj, (10, -1)))


# Create an output folder if it doesn't already exist
directory = './../output'
if not os.path.exists(directory):
    os.makedirs(directory)

pRF_distance('average')
pRF_distance('deepRetinotopy')
