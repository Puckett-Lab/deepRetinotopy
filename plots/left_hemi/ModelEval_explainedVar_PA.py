import numpy as np
import torch
import scipy.stats

from functions.def_ROIs_WangParcels import roi as roi2
from functions.def_ROIs_WangParcelsPlusFovea import roi
from functions.plusFovea import add_fovea

visual_areas = [
    ['hV4', 'VO1', 'VO2', 'PHC1', 'PHC2', 'V3a', 'V3b', 'LO1', 'LO2', 'TO1',
     'TO2', 'IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1']]
models = ['pred']
models_name = ['Default']

# # Uncomment to evaluate the performance of the average map
# PA_average = np.load('./../output/AveragePolarAngleMap_LH.npz')['list']
for k in range(len(visual_areas)):
    mean_delta = []
    for m in range(len(models)):
        predictions = torch.load(
            './../../testset_results/left_hemi/testset-' +
            models[m] + '_Model3_PA_LH.pt', map_location='cpu')

        theta_withinsubj = []
        theta_acrosssubj = []
        theta_acrosssubj_pred = []
        theta_acrosssubj_emp = []

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
                    pred = np.reshape(
                        np.array(predictions['Predicted_values'][i]),
                        (-1, 1))
                    measured = np.reshape(
                        np.array(predictions['Measured_values'][j]),
                        (-1, 1))

                    # # Uncomment the line bellow to evaluate the
                    # performance of the average map
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

                    # Computing correlation between predicted and empirical
                    # values
                    corr_theta = np.corrcoef(np.reshape(pred[mask > 1], (-1)),
                                             np.reshape(measured[mask > 1],
                                                        (-1)))[0, 1]
                    theta_withinsubj.append(corr_theta)

        # np.savez('./output/corr_higherOrder_LH_PA_Model.npz',
        #          list=np.reshape(theta_withinsubj, (10, -1)))
        # np.savez('./output/corr_higherOrder_LH_PA_averageMap.npz',
        # list=np.reshape(theta_withinsubj,(10,-1)))

        corr_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)

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

    # Compute angle between predicted and empirical predictions across subj
    for j in range(len(predictions['Predicted_values'])):
        for i in range(len(predictions['Predicted_values'])):
            if i == j:
                # Loading predicted values
                pred = np.reshape(np.array(predictions['Predicted_values'][i]),
                                  (-1, 1))
                measured = np.reshape(
                    np.array(predictions['Measured_values'][j]),
                    (-1, 1))

                # # Uncomment the line bellow to evaluate the performance of
                # # the average map
                # pred = np.reshape(np.array(PA_average), (-1, 1))

                # Rescaling polar angles to match the correct visual field (
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

                # Computing correlation between predicted and empirical
                # values
                corr_theta = np.corrcoef(np.reshape(pred[mask > 1], (-1)),
                                         np.reshape(measured[mask > 1], (-1)))[
                    0, 1]
                theta_withinsubj.append(corr_theta)

    # np.savez('./output/corr_earlyVisualCortex_LH_PA_averageMap.npz',
    #          list=np.reshape(theta_withinsubj, (10, -1)))
    # np.savez('./output/corr_earlyVisualCortex_LH_PA_Model.npz',
    #          list=np.reshape(theta_withinsubj, (10, -1)))

    corr_theta_withinsubj = np.mean(np.array(theta_withinsubj), axis=0)

# Loading the data
corr_EarlyVisualCortex_model = np.reshape(np.array(
    np.load('./../output/corr_earlyVisualCortex_LH_PA_Model.npz')['list']),
    (10, -1))
corr_EarlyVisualCortex_average = np.reshape(np.array(
    np.load('./../output/corr_earlyVisualCortex_LH_PA_averageMap.npz')[
        'list']), (10, -1))
corr_higherOrder_model = np.reshape(np.array(
    np.load('./../output/corr_higherOrder_LH_PA_Model.npz')['list']), (10, -1))
corr_higherOrder_average = np.reshape(np.array(
    np.load('./../output/corr_higherOrder_LH_PA_averageMap.npz')['list']),
    (10, -1))

print(f'Mean explained variance and std of model in early visual cortex:'
      f'{np.mean(corr_EarlyVisualCortex_model ** 2)}, {np.std(corr_EarlyVisualCortex_model ** 2)}')
print(f'Mean explained variance and std of average in early visual cortex:'
      f'{np.mean(corr_EarlyVisualCortex_average ** 2)}, {np.std(corr_EarlyVisualCortex_average ** 2)}')
test = scipy.stats.ttest_rel(corr_EarlyVisualCortex_model,
                             corr_EarlyVisualCortex_average)
print(test)

print(f'Mean explained variance and std of model in higher order areas:'
      f'{np.mean(corr_higherOrder_model ** 2)}, {np.std(corr_higherOrder_model ** 2)}')
print(f'Mean explained variance and std of average in higher order areas:'
      f'{np.mean(corr_higherOrder_average ** 2)}, {np.std(corr_higherOrder_average ** 2)}')
test = scipy.stats.ttest_rel(corr_higherOrder_model, corr_higherOrder_average)
print(test)
