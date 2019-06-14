import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns

a=torch.load('/home/uqfribe1/PycharmProjects/DEEP-fMRI/polarAngle/model4_nothresh_rotated_12layers_arccos_curvnmyelin_ROI1_k25_batchnorm_dropout010_output_epoch100.pt',map_location='cpu')

theta_withinsubj=[]
theta_acrosssubj=[]
theta_acrosssubj_emp=[]

R2_thr=[]

#Compute angle between predicted and empirical predictions across subj
for j in range(len(a['Predicted_values'])):
    theta_across_temp=[]
    theta_emp_across_temp=[]
    R2_thr.append(np.reshape(np.array(a['R2'][j]), (-1)))



    for i in range(len(a['Predicted_values'])):
        # Compute angle between predicted and empirical predictions within subj
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
            theta=np.arccos(np.cos(pred)*np.cos(measured)+np.sin(pred)*np.sin(measured))
            theta=theta*(180/np.pi)
            theta_withinsubj.append(theta)


        if i != j:
            # Compute angle between predicted and empical predictions across subj
            # Loading predicted values
            pred = np.reshape(np.array(a['Predicted_values'][i]), (-1, 1))
            pred2= np.reshape(np.array(a['Predicted_values'][j]), (-1, 1))
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
            theta = np.arccos(np.cos(pred) * np.cos(measured) + np.sin(pred) * np.sin(measured))
            theta = theta * (180 / np.pi)
            theta_across_temp.append(theta)

            # Computing delta theta, angle between vector defined predicted value and empirical value
            theta_emp = np.arccos(np.cos(pred) * np.cos(pred2) + np.sin(pred) * np.sin(pred2))
            theta_emp = theta_emp * (180 / np.pi)
            theta_emp_across_temp.append(theta_emp)

    theta_acrosssubj.append(np.mean(theta_across_temp,axis=0))
    theta_acrosssubj_emp.append(np.mean(theta_emp_across_temp, axis=0))



R2_thr=np.mean(R2_thr,axis=0)

mean_theta_acrosssubj=np.mean(np.array(theta_acrosssubj),axis=0)
mean_theta_withinsubj=np.mean(np.array(theta_withinsubj),axis=0)
mean_theta_acrosssubj_emp=np.mean(np.array(theta_acrosssubj_emp),axis=0)

sns.violinplot(data=[np.reshape(mean_theta_withinsubj[R2_thr>40],(-1)),np.reshape(mean_theta_acrosssubj[R2_thr>40],(-1)),np.reshape(mean_theta_acrosssubj_emp[R2_thr>40],(-1))])
plt.show()

sns.violinplot(data=[np.reshape(mean_theta_withinsubj[R2_thr>40],(-1)),np.reshape(mean_theta_acrosssubj[R2_thr>40],(-1))])
plt.show()
