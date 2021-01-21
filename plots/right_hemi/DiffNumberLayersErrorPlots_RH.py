import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd

from functions.def_ROIs_WangParcels import roi as roi2
from functions.def_ROIs_WangParcelsPlusFovea import roi
from functions.plusFovea import add_fovea
from functions.least_difference_angles import smallest_angle

visual_areas = [['hV4'], ['VO1', 'VO2', 'PHC1', 'PHC2'], ['V3a', 'V3b'],
                ['LO1', 'LO2', 'TO1', 'TO2'],
                ['IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1']]
number_layers = ['9', '10', '11', '12', '14', '16', '18', '20']

sns.set_style("whitegrid")
# Higher order visual areas
for k in range(len(visual_areas)):
    mean_delta = np.zeros((5, 8))
    mean_across = np.zeros((5, 8))

    l = 0
    while l < 5:
        mean_delta_temp = []
        mean_across_temp = []

        for m in range(8):
            if m < 4:
                prediction = torch.load(
                    '/home/uqfribe1/Desktop/Wiener/July/output_RH'
                    '/model4_nothresh_RH_' + str(
                        m + 9) +
                    'layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_' + str(
                        l + 1) + '_output_epoch100.pt', map_location='cpu')

                theta_withinsubj = []
                theta_acrosssubj_pred = []


                label_primary_visual_areas = ['ROI']
                final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
                    label_primary_visual_areas)
                ROI1 = np.zeros((32492, 1))
                ROI1[final_mask_R == 1] = 1

                final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
                    visual_areas[k])
                primary_visual_areas = np.zeros((32492, 1))
                primary_visual_areas[final_mask_R == 1] = 1

                mask = ROI1 + primary_visual_areas
                mask = mask[ROI1 == 1]

                for j in range(len(prediction['Predicted_values'])):
                    theta_pred_across_temp = []

                    for i in range(len(prediction['Predicted_values'])):
                        if i == j:
                            # Loading predicted values
                            pred = np.reshape(
                                np.array(prediction['Predicted_values'][i]), (-1, 1))
                            measured = np.reshape(np.array(prediction['Measured_values'
                                                             ''][j]), (-1, 1))

                            # Rescaling polar angles to match the right
                            # visual field (left hemisphere)
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
                            theta_withinsubj.append(theta)

                        if i != j:
                            # Loading predicted values
                            pred = np.reshape(
                                np.array(prediction['Predicted_values'][i]), (-1, 1))
                            pred2 = np.reshape(
                                np.array(prediction['Predicted_values'][j]), (-1, 1))

                            # Rescaling polar angles to match the right
                            # visual field (left hemisphere)
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
                            theta_pred_across_temp.append(theta_pred)

                    theta_acrosssubj_pred.append(
                        np.mean(theta_pred_across_temp, axis=0))

                mean_theta_withinsubj = np.mean(np.array(theta_withinsubj),
                                                axis=0)
                mean_theta_acrosssubj_pred = np.mean(
                    np.array(theta_acrosssubj_pred), axis=0)

                mean_delta_temp.append(
                    np.mean(mean_theta_withinsubj[mask > 1]))
                mean_across_temp.append(
                    np.mean(mean_theta_acrosssubj_pred[mask > 1]))

            else:
                prediction = torch.load(
                    '/home/uqfribe1/Desktop/Wiener/July/output_RH'
                    '/model4_nothresh_RH_' + str(
                        12 + (
                                    m - 4) * 2) +
                    'layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_' + str(
                        l + 1) + '_output_epoch200.pt', map_location='cpu')

                theta_withinsubj = []
                theta_acrosssubj_pred = []

                label_primary_visual_areas = ['ROI']
                final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
                    label_primary_visual_areas)
                ROI1 = np.zeros((32492, 1))
                ROI1[final_mask_R == 1] = 1

                final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
                    visual_areas[k])
                primary_visual_areas = np.zeros((32492, 1))
                primary_visual_areas[final_mask_R == 1] = 1

                mask = ROI1 + primary_visual_areas
                mask = mask[ROI1 == 1]

                for j in range(len(prediction['Predicted_values'])):
                    theta_pred_across_temp = []

                    for i in range(len(prediction['Predicted_values'])):
                        if i == j:
                            # Loading predicted values
                            pred = np.reshape(
                                np.array(prediction['Predicted_values'][i]), (-1, 1))
                            measured = np.reshape(
                                np.array(prediction['Measured_values'][j]), (-1, 1))

                            # Rescaling polar angles to match the correct
                            # visual field (left hemisphere)
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
                            theta_withinsubj.append(theta)

                        if i != j:
                            # Loading predicted values
                            pred = np.reshape(
                                np.array(prediction['Predicted_values'][i]), (-1, 1))
                            pred2 = np.reshape(
                                np.array(prediction['Predicted_values'][j]), (-1, 1))

                            # Rescaling polar angles to match the right
                            # visual field (left hemisphere)
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
                            theta_pred_across_temp.append(theta_pred)

                    theta_acrosssubj_pred.append(
                        np.mean(theta_pred_across_temp, axis=0))

                mean_theta_withinsubj = np.mean(np.array(theta_withinsubj),
                                                axis=0)
                mean_theta_acrosssubj_pred = np.mean(
                    np.array(theta_acrosssubj_pred), axis=0)

                mean_delta_temp.append(
                    np.mean(mean_theta_withinsubj[mask > 1]))
                mean_across_temp.append(
                    np.mean(mean_theta_acrosssubj_pred[mask > 1]))

        mean_delta[l] = np.array(mean_delta_temp)
        mean_across[l] = np.array(mean_across_temp)
        l += 1
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    data = np.concatenate([[mean_across[0], number_layers,
                            len(number_layers) * ['Between predicted maps']],
                           [mean_across[1], number_layers,
                            len(number_layers) * ['Between predicted maps']],
                           [mean_across[2], number_layers,
                            len(number_layers) * ['Between predicted maps']],
                           [mean_across[3], number_layers,
                            len(number_layers) * ['Between predicted maps']],
                           [mean_across[4], number_layers,
                            len(number_layers) * ['Between predicted maps']],
                           [mean_delta[0], number_layers,
                            len(number_layers) * [
                                'Between predicted map and ground truth']],
                           [mean_delta[1], number_layers,
                            len(number_layers) * [
                                'Between predicted map and ground truth']],
                           [mean_delta[2], number_layers,
                            len(number_layers) * [
                                'Between predicted map and ground truth']],
                           [mean_delta[3], number_layers,
                            len(number_layers) * [
                                'Between predicted map and ground truth']],
                           [mean_delta[4], number_layers,
                            len(number_layers) * [
                                'Between predicted map and ground truth']]],
                          axis=1)
    df = pd.DataFrame(columns=['$\Delta$$\t\Theta$', 'Layers', 'label'],
                      data=data.T)
    df['$\Delta$$\t\Theta$'] = df['$\Delta$$\t\Theta$'].astype(float)
    palette = ['dimgray', 'lightgray']
    ax = sns.boxplot(y='$\Delta$$\t\Theta$', x='Layers', order=number_layers,
                     hue='label', data=df, palette=palette)
    title = ['V4', 'Ventral', 'V3a/b', 'Lateral', 'Parietal']
    ax.set_title(title[k])
    legend = plt.legend()
    legend.remove()
    plt.ylim([0, 70])
    # plt.savefig('PAdif_cluster' + str(k + 1) + '_RH.svg')
    plt.show()

# Early visual cortex
label_primary_visual_areas = ['V1d', 'V1v', 'fovea_V1', 'V2d', 'V2v',
                              'fovea_V2', 'V3d', 'V3v', 'fovea_V3']
V1, V2, V3 = add_fovea(label_primary_visual_areas)
primary_visual_areas = np.sum(
    [np.reshape(V1, (-1, 1)), np.reshape(V2, (-1, 1)),
     np.reshape(V3, (-1, 1))], axis=0)
label = ['Early visual cortex']

fig = plt.figure()

mean_delta_2 = np.zeros((5, 8))
mean_across_2 = np.zeros((5, 8))

l = 0
while l < 5:
    mean_delta_temp = []
    mean_across_temp = []

    for m in range(8):
        if m < 4:
            prediction = torch.load(
                '/home/uqfribe1/Desktop/Wiener/July/output_RH'
                '/model4_nothresh_RH_' + str(
                    m + 9) +
                'layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_' + str(
                    l + 1) + '_output_epoch100.pt', map_location='cpu')

            theta_withinsubj = []
            theta_acrosssubj_pred = []

            label_primary_visual_areas = ['ROI']
            final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
                label_primary_visual_areas)
            ROI1 = np.zeros((32492, 1))
            ROI1[final_mask_R == 1] = 1
            mask = ROI1 + np.reshape(primary_visual_areas, (32492, 1))
            mask = mask[ROI1 == 1]

            for j in range(len(prediction['Predicted_values'])):
                theta_pred_across_temp = []

                for i in range(len(prediction['Predicted_values'])):
                    if i == j:
                        # Loading predicted values
                        pred = np.reshape(np.array(prediction['Predicted_values'][i]),
                                          (-1, 1))
                        measured = np.reshape(np.array(prediction['Measured_values'][
                                                           j]), (-1, 1))

                        # Rescaling polar angles to match the correct visual
                        # field (left hemisphere)
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
                        theta_withinsubj.append(theta)

                    if i != j:
                        # Loading predicted values
                        pred = np.reshape(np.array(prediction['Predicted_values'][i]),
                                          (-1, 1))
                        pred2 = np.reshape(np.array(prediction['Predicted_values'][j]),
                                           (-1, 1))

                        # Rescaling polar angles to match the correct visual
                        # field (left hemisphere)
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
                        theta_pred_across_temp.append(theta_pred)

                theta_acrosssubj_pred.append(
                    np.mean(theta_pred_across_temp, axis=0))

            mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
            mean_theta_acrosssubj_pred = np.mean(
                np.array(theta_acrosssubj_pred), axis=0)

            mean_delta_temp.append(np.mean(mean_theta_withinsubj[mask > 1]))
            mean_across_temp.append(
                np.mean(mean_theta_acrosssubj_pred[mask > 1]))

        else:

            a = torch.load(
                '/home/uqfribe1/Desktop/Wiener/July/output_RH'
                '/model4_nothresh_RH_' + str(
                    12 + (
                                m - 4) * 2) +
                'layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_' + str(
                    l + 1) + '_output_epoch200.pt', map_location='cpu')

            theta_withinsubj = []
            theta_acrosssubj_pred = []


            label_primary_visual_areas = ['ROI']
            final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
                label_primary_visual_areas)
            ROI1 = np.zeros((32492, 1))
            ROI1[final_mask_R == 1] = 1
            mask = ROI1 + np.reshape(primary_visual_areas, (32492, 1))
            mask = mask[ROI1 == 1]

            for j in range(len(prediction['Predicted_values'])):
                theta_pred_across_temp = []

                for i in range(len(prediction['Predicted_values'])):
                    if i == j:
                        # Loading predicted values
                        pred = np.reshape(np.array(prediction['Predicted_values'][i]),
                                          (-1, 1))
                        measured = np.reshape(
                            np.array(prediction['Measured_values'][j]), (-1, 1))

                        # Rescaling polar angles to match the right visual
                        # field (left hemisphere)
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
                        theta_withinsubj.append(theta)

                    if i != j:
                        # Loading predicted values
                        pred = np.reshape(np.array(prediction['Predicted_values'][i]),
                                          (-1, 1))
                        pred2 = np.reshape(np.array(prediction['Predicted_values'][j]),
                                           (-1, 1))

                        # Rescaling polar angles to match the right visual
                        # field (left hemisphere)
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
                        theta_pred_across_temp.append(theta_pred)

                theta_acrosssubj_pred.append(
                    np.mean(theta_pred_across_temp, axis=0))

            mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
            mean_theta_acrosssubj_pred = np.mean(
                np.array(theta_acrosssubj_pred), axis=0)

            mean_delta_temp.append(np.mean(mean_theta_withinsubj[mask > 1]))
            mean_across_temp.append(
                np.mean(mean_theta_acrosssubj_pred[mask > 1]))

    mean_delta_2[l] = np.array(mean_delta_temp)
    mean_across_2[l] = np.array(mean_across_temp)
    l += 1

ax = fig.add_subplot(1, 1, 1)
data = np.concatenate([[mean_across_2[0], number_layers,
                        len(number_layers) * ['Between predicted maps']],
                       [mean_across_2[1], number_layers,
                        len(number_layers) * ['Between predicted maps']],
                       [mean_across_2[2], number_layers,
                        len(number_layers) * ['Between predicted maps']],
                       [mean_across_2[3], number_layers,
                        len(number_layers) * ['Between predicted maps']],
                       [mean_across_2[4], number_layers,
                        len(number_layers) * ['Between predicted maps']],
                       [mean_delta_2[0], number_layers, len(number_layers) * [
                           'Between predicted map and ground truth']],
                       [mean_delta_2[1], number_layers, len(number_layers) * [
                           'Between predicted map and ground truth']],
                       [mean_delta_2[2], number_layers, len(number_layers) * [
                           'Between predicted map and ground truth']],
                       [mean_delta_2[3], number_layers, len(number_layers) * [
                           'Between predicted map and ground truth']],
                       [mean_delta_2[4], number_layers, len(number_layers) * [
                           'Between predicted map and ground truth']]], axis=1)
df = pd.DataFrame(columns=['$\Delta$$\t\Theta$', 'Layers', 'label'],
                  data=data.T)
df['$\Delta$$\t\Theta$'] = df['$\Delta$$\t\Theta$'].astype(float)
palette = ['dimgray', 'lightgray']
ax = sns.boxplot(y='$\Delta$$\t\Theta$', x='Layers', order=number_layers,
                 hue='label', data=df, palette=palette)
ax.set_title('' + label[0])
plt.legend(loc='upper right')
plt.ylim([0, 70])
# plt.savefig('PAdif_EarlyVisualArea_RH.svg')
plt.show()
