import numpy as np
import pandas as pd
import sys
import os

sys.path.append('..')

models=['PA']

# Create an output folder if it doesn't already exist
directory = './../output'
if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(len(models)):
    # Loading the data
    error_dorsalEarlyVisualCortex_Benson14_fit1 = np.reshape(np.array(
        np.load(
            './../output/ErrorPerParticipant_PA_LH_dorsalV1-3_Benson14_1-8.npz')[
            'list']),
        (10, -1))
    error_dorsalEarlyVisualCortex_Benson14_fit2 = np.reshape(np.array(
        np.load(
            './../output/ErrorPerParticipant_PA_LH_dorsalV1-3_Benson14_1-8_fit2'
            '.npz')[
            'list']),
        (10, -1))
    error_dorsalEarlyVisualCortex_Benson14_fit3 = np.reshape(np.array(
        np.load(
            './../output/ErrorPerParticipant_PA_LH_dorsalV1-3_Benson14_1-8_fit3.npz')[
            'list']),
        (10, -1))

    error_dorsalEarlyVisualCortex_model_fit1 = np.reshape(np.array(
        np.load('./../output/ErrorPerParticipant_PA_LH_dorsalV1-3_deepRetinotopy_1-8.npz')['list']),
        (10, -1))
    error_dorsalEarlyVisualCortex_model_fit2 = np.reshape(np.array(
        np.load('./../output/ErrorPerParticipant_PA_LH_dorsalV1-3_deepRetinotopy_1-8_fit2.npz')['list']),
        (10, -1))
    error_dorsalEarlyVisualCortex_model_fit3 = np.reshape(np.array(
        np.load('./../output/ErrorPerParticipant_PA_LH_dorsalV1-3_deepRetinotopy_1-8_fit3.npz')['list']),
        (10, -1))

    error_dorsalEarlyVisualCortex_average_fit1 = np.reshape(np.array(
        np.load('./../output/ErrorPerParticipant_PA_LH_dorsalV1-3_average_1-8.npz')['list']),
        (10, -1))
    error_dorsalEarlyVisualCortex_average_fit2 = np.reshape(np.array(
        np.load('./../output/ErrorPerParticipant_PA_LH_dorsalV1-3_average_1-8_fit2.npz')['list']),
        (10, -1))
    error_dorsalEarlyVisualCortex_average_fit3 = np.reshape(np.array(
        np.load('./../output/ErrorPerParticipant_PA_LH_dorsalV1-3_average_1-8_fit3.npz')['list']),
        (10, -1))

    error_dorsalEarlyVisualCortex_fit1_fit2 = np.reshape(np.array(
        np.load(
            './../output/ErrorPerParticipant_PA_LH_dorsalV1-3_fit1_1-8_fit2.npz')[
            'list']),
        (10, -1))
    error_dorsalEarlyVisualCortex_fit1_fit3 = np.reshape(np.array(
        np.load(
            './../output/ErrorPerParticipant_PA_LH_dorsalV1-3_fit1_1-8_fit3.npz')[
            'list']),
        (10, -1))


    # Summary table
    summary = pd.DataFrame({'Region of interest': ['Dorsal V1-3 Fit1 - Average',
                                                   'Dorsal V1-3 Fit2 - Average',
                                                   'Dorsal V1-3 Fit3 - Average',
                                                   'Dorsal V1-3 Fit1 - Model',
                                                   'Dorsal V1-3 Fit2 - Model',
                                                   'Dorsal V1-3 Fit3 - Model',
                                                   'Dorsal V1-3 Fit1 - '
                                                   'Benson14',
                                                   'Dorsal V1-3 Fit2 - '
                                                   'Benson14',
                                                   'Dorsal V1-3 Fit3 - '
                                                   'Benson14',
                                                   ],
                            'Mean': [np.mean(np.mean(error_dorsalEarlyVisualCortex_average_fit1, axis=1)),
                                     np.mean(np.mean(error_dorsalEarlyVisualCortex_average_fit2, axis=1)),
                                     np.mean(np.mean(error_dorsalEarlyVisualCortex_average_fit3, axis=1)),
                                     np.mean(np.mean(error_dorsalEarlyVisualCortex_model_fit1, axis=1)),
                                     np.mean(np.mean(error_dorsalEarlyVisualCortex_model_fit2, axis=1)),
                                     np.mean(np.mean(error_dorsalEarlyVisualCortex_model_fit3, axis=1)),
                                     np.mean(np.mean(
                                         error_dorsalEarlyVisualCortex_Benson14_fit1,
                                         axis=1)),
                                     np.mean(np.mean(
                                         error_dorsalEarlyVisualCortex_Benson14_fit2,
                                         axis=1)),
                                     np.mean(np.mean(
                                         error_dorsalEarlyVisualCortex_Benson14_fit3,
                                         axis=1))
                                 ],
                            'Std': [np.std(np.mean(error_dorsalEarlyVisualCortex_average_fit1, axis=1)),
                                     np.std(np.mean(error_dorsalEarlyVisualCortex_average_fit2, axis=1)),
                                     np.std(np.mean(error_dorsalEarlyVisualCortex_average_fit3, axis=1)),
                                     np.std(np.mean(error_dorsalEarlyVisualCortex_model_fit1, axis=1)),
                                     np.std(np.mean(error_dorsalEarlyVisualCortex_model_fit2, axis=1)),
                                     np.std(np.mean(error_dorsalEarlyVisualCortex_model_fit3, axis=1)),
                                    np.std(np.mean(
                                        error_dorsalEarlyVisualCortex_Benson14_fit1,
                                        axis=1)),
                                    np.std(np.mean(
                                        error_dorsalEarlyVisualCortex_Benson14_fit2,
                                        axis=1)),
                                    np.std(np.mean(
                                        error_dorsalEarlyVisualCortex_Benson14_fit3,
                                        axis=1)),
                                    ]}
                           )
    print(summary)

    print(np.mean(np.mean(error_dorsalEarlyVisualCortex_fit1_fit2, axis=1)),
          np.std(np.mean(error_dorsalEarlyVisualCortex_fit1_fit2, axis=1)))
    print(np.mean(np.mean(error_dorsalEarlyVisualCortex_fit1_fit3, axis=1)),
          np.std(np.mean(error_dorsalEarlyVisualCortex_fit1_fit3, axis=1)))


    # Table for Jamovi
    stats_table = pd.DataFrame(
        {'Dorsal V1-3 Fit1 - Average': np.mean(error_dorsalEarlyVisualCortex_average_fit1,
                                  axis=1),
         'Dorsal V1-3 Fit2 - Average':
                  np.mean(error_dorsalEarlyVisualCortex_average_fit2,
                                  axis=1),
         'Dorsal V1-3 Fit3 - Average':
                  np.mean(error_dorsalEarlyVisualCortex_average_fit3,
                                  axis=1),
         'Dorsal V1-3 Fit1 - Model':
                  np.mean(error_dorsalEarlyVisualCortex_model_fit1,
                                  axis=1),
         'Dorsal V1-3 Fit2 - Model':
                  np.mean(error_dorsalEarlyVisualCortex_model_fit2,
                                  axis=1),
         'Dorsal V1-3 Fit3 - Model':
                  np.mean(error_dorsalEarlyVisualCortex_model_fit3,
                                  axis=1),
         'Dorsal V1-3 Fit1 - Benson14':
             np.mean(error_dorsalEarlyVisualCortex_Benson14_fit1,
                     axis=1),
         'Dorsal V1-3 Fit2 - Benson14':
             np.mean(error_dorsalEarlyVisualCortex_Benson14_fit2,
                     axis=1),
         'Dorsal V1-3 Fit3 - Benson14':
             np.mean(error_dorsalEarlyVisualCortex_Benson14_fit3,
                     axis=1)
         }
        )
    stats_table.to_excel('./../output/modelVSaverageVSbenson_fit1-3_dorsalV1-3_PA_LH.xlsx')
    print(stats_table)

