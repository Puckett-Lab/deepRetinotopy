import numpy as np
import matplotlib.pyplot as plt
import torch


a=torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle/model4_nothresh_rotated_12layers_smoothL1lossR2_curvnmyelin_ROI1_k25_batchnorm_dropout010_2_output_epoch150.pt',map_location='cpu')
theta_withinsubj=[]
theta_acrosssubj=[]

#Compute angle between predicted and empirical predictions across subj
for j in range(len(a['Predicted_values'])):
    for i in range(len(a['Predicted_values'])):
        # Compute angle between predicted and empical predictions within subj
        if i==j:
            #Loading predicted values
            pred=np.reshape(np.array(a['Predicted_values'][i]),(-1,1))
            measured=np.reshape(np.array(a['Measured_values'][j]),(-1,1))

            #Rescaling polar angles to match the right visual field (left hemisphere)
            minus=pred>180
            sum=pred<180
            pred[minus]=pred[minus]-180
            pred[sum]=pred[sum]+180
            pred=np.array(pred)*(np.pi/180)


            minus=measured>180
            sum=measured<180
            measured[minus]=measured[minus]-180
            measured[sum]=measured[sum]+180
            measured=np.array(measured)*(np.pi/180)

            #Computing delta theta, angle between vector defined predicted value and empirical value
            theta=np.arccos(np.cos(pred)*np.cos(measured)-np.sin(pred)*np.sin(measured))
            theta=theta*(180/np.pi)
            theta_withinsubj.append(theta)

        # Compute angle between predicted and empical predictions across subj
        if i != j:
            # Loading predicted values
            pred = np.reshape(np.array(a['Predicted_values'][i]), (-1, 1))
            measured = np.reshape(np.array(a['Measured_values'][j]), (-1, 1))

            # Rescaling polar angles to match the right visual field (left hemisphere)
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

            # Computing delta theta, angle between vector defined predicted value and empirical value
            theta = np.arccos(np.cos(pred) * np.cos(measured) - np.sin(pred) * np.sin(measured))
            theta = theta * (180 / np.pi)
            theta_acrosssubj.append(theta)

mean_theta_acrosssubj=np.mean(np.array(theta_acrosssubj),axis=0)
mean_theta_withinsubj=np.mean(np.array(theta_withinsubj),axis=0)
plt.hist(mean_theta_acrosssubj,bins=100)
plt.xlim(0, 180)
plt.show()
