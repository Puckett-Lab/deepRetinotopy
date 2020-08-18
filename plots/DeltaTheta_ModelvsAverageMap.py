import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the data
error_EarlyVisualCortex_model = np.reshape(np.array(
    np.load('../ErrorPerParticipant_EarlyVisualCortex_model.npz')['list']),
    (10, -1))
error_higherOrder_model = np.reshape(np.array(
    np.load('../ErrorPerParticipant_WangParcels_model.npz')['list']), (10, -1))

error_EarlyVisualCortex_average = np.reshape(np.array(
    np.load('../ErrorPerParticipant_EarlyVisualCortex_averageMap.npz')[
        'list']), (10, -1))
error_higherOrder_average = np.reshape(np.array(
    np.load('../ErrorPerParticipant_WangParcels_averageMap.npz')['list']),
    (10, -1))

# Reformatting data from early visual cortex
data_earlyVisualCortex = np.concatenate([
    [np.mean(error_EarlyVisualCortex_average, axis=1),
     np.shape(error_EarlyVisualCortex_average)[0] * [
         'Average Map'],
     np.shape(error_EarlyVisualCortex_average)[0] * [
         'Early Visual Cortex']
     ],
    [np.mean(error_EarlyVisualCortex_model, axis=1),
     np.shape(error_EarlyVisualCortex_model)[0] * ['Model'],
     np.shape(error_EarlyVisualCortex_model)[0] * [
         'Early Visual Cortex']]],
    axis=1)

df_1 = pd.DataFrame(
    columns=['Mean Error', 'Prediction', 'Area'],
    data=data_earlyVisualCortex.T)
df_1['Mean Error'] = df_1['Mean Error'].astype(float)

print(df_1)

# Reformatting data from higher order visual areas
data_HigherOrder = np.concatenate([
    [np.mean(error_higherOrder_average, axis=1),
     np.shape(error_higherOrder_average)[0] * [
         'Average Map'],
     np.shape(error_higherOrder_average)[0] * [
         'Higher Order Visual Areas']],
    [np.mean(error_higherOrder_model, axis=1),
     np.shape(error_higherOrder_model)[0] * ['Model'],
     np.shape(error_higherOrder_model)[0] * [
         'Higher Order Visual Areas']]
],
    axis=1)

df_2 = pd.DataFrame(
    columns=['Mean Error', 'Prediction', 'Area'],
    data=data_HigherOrder.T)
df_2['Mean Error'] = df_2['Mean Error'].astype(float)

# Generate the plot
title = ['Early visual cortex', 'Higher order visual areas']
fig = plt.figure(figsize=(10, 5))
for i in range(2):
    fig.add_subplot(1, 2, i + 1)
    ax = sns.swarmplot(x='Prediction', y='Mean Error',
                       data=eval('df_' + str(i + 1)),
                       palette='colorblind')
    plt.title(title[i])

    # Prediction error for the same participants
    for j in range(10):
        x = [eval('df_' + str(i + 1))['Prediction'][j],
             eval('df_' + str(i + 1))['Prediction'][j + 10]]
        y = [eval('df_' + str(i + 1))['Mean Error'][j],
             eval('df_' + str(i + 1))['Mean Error'][j + 10]]
        ax.plot(x, y, color='black', alpha=0.1)

plt.show()

# Plotting the difference
# data = pd.DataFrame({'DiffEarly': np.mean(
#     error_EarlyVisualCortex_average,
#     axis=1) - np.mean(error_EarlyVisualCortex_model,
#                       axis=1),
#                      'DiffHigher': np.mean(error_higherOrder_average,
#                                            axis=1) - np.mean(
#                          error_higherOrder_model, axis=1)})
#
# data = pd.melt(data)
# sns.swarmplot(x='variable', y='value',
#               data=data,
#               palette='colorblind')
# plt.show()
#
