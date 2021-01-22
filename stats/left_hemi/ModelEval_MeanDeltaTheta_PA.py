import numpy as np
import torch

from functions.def_ROIs_WangParcels import roi as roi2
from functions.def_ROIs_WangParcelsPlusFovea import roi
from functions.plusFovea import add_fovea
from functions.least_difference_angles import smallest_angle

visual_areas = [
    ['hV4', 'VO1', 'VO2', 'PHC1', 'PHC2', 'V3a', 'V3b', 'LO1', 'LO2', 'TO1',
     'TO2', 'IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1']]
models = ['pred']
models_name = ['Default']

# # Uncomment to evaluate the performance of the average map
# PA_average = np.load('./../../plots/output/AveragePolarAngleMap_LH.npz')['list']
for k in range(len(visual_areas)):
    mean_delta = []
    for m in range(len(models)):
        predictions = torch.load(
            './../../testset_results/left_hemi/testset-' +
            models[m] + '_Model3_PA_LH.pt', map_location='cpu')

        theta_withinsubj = []

        label_primary_visual_areas = ['ROI']
        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
            label_primary_visual_areas)
        ROI1 = np.zeros((32492, 1))
        ROI1[final_mask_L == 1] = 1

        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
            visual_areas[k])
        primary_visual_areas = np.zeros((32492, 1))
        primary_visual_areas[final_mask_L == 1] = 1

        mask = ROI1 + primary_visual_areas
        mask = mask[ROI1 == 1]

        for j in range(len(predictions['Predicted_values'])):
            for i in range(len(predictions['Predicted_values'])):
                if i == j:
                    # Loading predicted values
                    pred = np.reshape(np.array(predictions['Predicted_values'][i]),
                                      (-1, 1))
                    measured = np.reshape(np.array(predictions['Measured_values'][j]),
                                          (-1, 1))

                    # # Uncomment the line bellow to evaluate the performance
                    # # of the average map
                    # pred = np.reshape(np.array(PA_average), (-1, 1))

                    # Rescaling polar angles to match the right visual field
                    # (left hemisphere)
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
        mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
        mean_delta.append(mean_theta_withinsubj[mask > 1])
    mean_delta = np.reshape(np.array(mean_delta), (1, -1))

# np.savez('./../output/ErrorPerParticipant_WangParcels_model.npz',list=np.reshape(theta_withinsubj,(10,-1))[:,mask>1])
# np.savez('./../output/ErrorPerParticipant_WangParcels_averageMap.npz',list=np.reshape(theta_withinsubj,(10,-1))[:,mask>1])


# Primary visual cortex
label_primary_visual_areas = ['V1d', 'V1v', 'fovea_V1', 'V2d', 'V2v',
                              'fovea_V2', 'V3d', 'V3v', 'fovea_V3']
V1, V2, V3 = add_fovea(label_primary_visual_areas)
primary_visual_areas = np.sum(
    [np.reshape(V1, (-1, 1)), np.reshape(V2, (-1, 1)),
     np.reshape(V3, (-1, 1))], axis=0)
label = ['Early visual cortex']

mean_delta_2 = []
for m in range(len(models)):
    predictions = torch.load(
        './../../testset_results/left_hemi/testset-' +
        models[m] + '_Model3_PA_LH.pt', map_location='cpu')

    theta_withinsubj = []

    label_primary_visual_areas = ['ROI']
    final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
        label_primary_visual_areas)
    ROI1 = np.zeros((32492, 1))
    ROI1[final_mask_L == 1] = 1
    mask = ROI1 + np.reshape(primary_visual_areas, (32492, 1))
    mask = mask[ROI1 == 1]

    for j in range(len(predictions['Predicted_values'])):
        for i in range(len(predictions['Predicted_values'])):
            if i == j:
                # Loading predicted values
                pred = np.reshape(np.array(predictions['Predicted_values'][i]), (-1, 1))
                measured = np.reshape(np.array(predictions['Measured_values'][j]),
                                      (-1, 1))

                # # Uncomment the line bellow to evaluate the performance of
                # # the average map
                # pred = np.reshape(np.array(PA_average), (-1, 1))

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
                theta_withinsubj.append(theta)
    mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
    mean_delta_2.append(mean_theta_withinsubj[mask > 1])

mean_delta_2 = np.reshape(np.array(mean_delta_2), (1, -1))

# np.savez('./../output/ErrorPerParticipant_EarlyVisualCortex_model.npz',list=np.reshape(theta_withinsubj,(10,-1))[:,mask>1])
# np.savez('./../output/ErrorPerParticipant_EarlyVisualCortex_averageMap.npz',list=np.reshape(theta_withinsubj,(10,-1))[:,mask>1])

mean_all = np.mean(np.concatenate((mean_delta[0], mean_delta_2[0])))
std_all = np.std(np.concatenate((mean_delta[0], mean_delta_2[0])))

print(
    f'Mean error and std in early visual cortex (V1, V2, V3) including the '
    f'fovea: {np.mean(mean_delta_2[0])}, {np.std(mean_delta_2[0])}')
print(
    f'Mean error and std in higher order areas (Wang et al., 2015):'
    f' {np.mean(mean_delta[0])}, {np.std(mean_delta[0])}')



# Dorsal early visual cortex
visual_areas = [
    ['V1d', 'V2d', 'V3d']]
models = ['pred']
models_name = ['Default']

for k in range(len(visual_areas)):
    mean_delta = []
    for m in range(len(models)):
        predictions = torch.load(
            './../../testset_results/left_hemi/testset-' +
            models[m] + '_Model3_PA_LH.pt', map_location='cpu')

        theta_withinsubj = []

        label_primary_visual_areas = ['ROI']
        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi(
            label_primary_visual_areas)
        ROI1 = np.zeros((32492, 1))
        ROI1[final_mask_L == 1] = 1

        final_mask_L, final_mask_R, index_L_mask, index_R_mask = roi2(
            visual_areas[k])
        primary_visual_areas = np.zeros((32492, 1))
        primary_visual_areas[final_mask_L == 1] = 1

        mask = ROI1 + primary_visual_areas
        mask = mask[ROI1 == 1]

        # Compute angle between predicted and empirical predictions across subj
        for j in range(len(predictions['Predicted_values'])):
            for i in range(len(predictions['Predicted_values'])):

                if i == j:
                    # Loading predicted values
                    pred = np.reshape(np.array(predictions['Predicted_values'][i]),
                                      (-1, 1))
                    measured = np.reshape(np.array(predictions['Measured_values'][j]),
                                          (-1, 1))

                    # # Uncomment the line bellow to evaluate the performance of
                    # # the average map
                    # pred = np.reshape(np.array(PA_average), (-1, 1))

                    # Rescaling polar angles to match the right visual field
                    # (left hemisphere)
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

        mean_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)
        mean_delta.append(mean_theta_withinsubj[mask > 1])
    mean_delta = np.reshape(np.array(mean_delta), (1, -1))

print(
    f'Mean error and std in dorsal early visual cortex (V1, V2, V3) not including the '
    f'fovea: {np.mean(mean_delta[0])}, {np.std(mean_delta[0])}')

# np.savez('./../output/ErrorPerParticipant_dorsalV1-3_model.npz',list=np.reshape(theta_withinsubj,(10,-1))[:,mask>1])
# np.savez('./../output/ErrorPerParticipant_dorsalV1-3_averageMap.npz',list=np.reshape(theta_withinsubj,(10,-1))[:,mask>1])