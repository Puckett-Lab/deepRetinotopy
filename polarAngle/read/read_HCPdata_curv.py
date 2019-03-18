import scipy.io
import numpy as np
import torch
import os.path as osp

from torch_geometric.data import Data

def read_HCP(path,Hemisphere=None,index=None,surface=None,threshold=None,eccentricity=None,polar_angle=None):
    # Loading the measures
    curv = scipy.io.loadmat(osp.join(path,'cifti_curv_all.mat'))['cifti_curv']
    eccentricity = scipy.io.loadmat(osp.join(path,'cifti_eccentricity_all.mat'))['cifti_eccentricity']
    polarAngle = scipy.io.loadmat(osp.join(path,'cifti_polarAngle_all.mat'))['cifti_polarAngle']
    R2 = scipy.io.loadmat(osp.join(path,'cifti_R2_all.mat'))['cifti_R2']

    # Loading list of subjects
    with open(osp.join(path,'..','..','list_subj')) as fp:
        subjects = fp.read().split("\n")
    subjects = subjects[0:len(subjects) - 1]

    #Defining number of nodes
    number_cortical_nodes=int(64984)
    number_hemi_nodes=int(number_cortical_nodes/2)


    if Hemisphere=='Right':
        #Loading connectivity of triangles
        faces=scipy.io.loadmat(osp.join(path,'tri_faces_R.mat'))['tri_faces_R']-1 #in matlab, list indexing stars from 1
        faces=torch.tensor(faces.T,dtype=torch.long)        #Transforming data to torch data type

        if surface=='mid':
        #Coordinates of the Right hemisphere vertices
            pos=torch.tensor(scipy.io.loadmat(osp.join(path,'mid_pos_R.mat'))['mid_pos_R'].reshape((number_hemi_nodes,3)),dtype=torch.float)
        if surface=='sphere':
            pos=torch.tensor(curv['pos'][0][0][number_hemi_nodes:number_cortical_nodes].reshape((number_hemi_nodes,3)),dtype=torch.float)

        #Measures for the Right hemisphere
        R2_values = torch.tensor(np.reshape(R2['x' + subjects[index] + '_fit1_r2_msmall'][0][0][number_hemi_nodes:number_cortical_nodes].reshape((number_hemi_nodes)),(-1,1)),dtype=torch.float)
        curvature = torch.tensor(np.reshape(
            curv['x' + subjects[index] + '_curvature'][0][0][number_hemi_nodes:number_cortical_nodes].reshape(
                (number_hemi_nodes)), (-1, 1)), dtype=torch.float)
        eccentricity_values = torch.tensor(np.reshape(
            eccentricity['x' + subjects[index] + '_fit1_eccentricity_msmall'][0][0][
            number_hemi_nodes:number_cortical_nodes].reshape((number_hemi_nodes)), (-1, 1)), dtype=torch.float)
        polarAngle_values = torch.tensor(np.reshape(polarAngle['x' + subjects[index] + '_fit1_polarangle_msmall'][0][0][
                                                    number_hemi_nodes:number_cortical_nodes].reshape(
            (number_hemi_nodes)), (-1, 1)), dtype=torch.float)

        nocurv=np.isnan(curvature)
        curvature[nocurv==1] = 0

        noR2=np.isnan(R2_values)
        R2_values[noR2==1]=0

        condition=R2_values < threshold
        condition2=np.isnan(eccentricity_values)
        condition3=np.isnan(polarAngle_values)


        eccentricity_values[condition == 1] = curvature[condition==1]
        eccentricity_values[condition2 == 1] = curvature[condition2==1]

        polarAngle_values[condition==1] = curvature[condition==1]
        polarAngle_values[condition3 == 1] = curvature[condition3==1]

        data=Data(x=curvature,y=polarAngle_values,pos=pos)
        data.face=faces
        data.R2 = R2_values


    if Hemisphere=='Left':
        #Loading connectivity of triangles
        faces=scipy.io.loadmat(osp.join(path,'tri_faces_L.mat'))['tri_faces_L']-1
        faces=torch.tensor(faces.T,dtype=torch.long)        #Transforming data to torch data type

        # Coordinates of the Left hemisphere vertices
        if surface=='mid':
            pos=torch.tensor(scipy.io.loadmat(osp.join(path,'mid_pos_L.mat'))['mid_pos_L'].reshape((number_hemi_nodes,3)),dtype=torch.float)
        if surface== 'sphere':
            pos = torch.tensor(curv['pos'][0][0][0:number_hemi_nodes].reshape((number_hemi_nodes, 3)),dtype=torch.float)

        # Measures for the Left hemisphere
        R2_values = torch.tensor(np.reshape(R2['x' + subjects[index] + '_fit1_r2_msmall'][0][0][0:number_hemi_nodes].reshape((number_hemi_nodes)),(-1,1)),dtype=torch.float)
        curvature=torch.tensor(np.reshape(
            curv['x' + subjects[index] + '_curvature'][0][0][0:number_hemi_nodes].reshape(
                (number_hemi_nodes)), (-1, 1)), dtype=torch.float)
        eccentricity_values= torch.tensor(np.reshape(eccentricity['x' + subjects[index] + '_fit1_eccentricity_msmall'][0][0][0:number_hemi_nodes].reshape((number_hemi_nodes)), (-1, 1)), dtype=torch.float)
        polarAngle_values= torch.tensor(np.reshape(polarAngle['x' + subjects[index] + '_fit1_polarangle_msmall'][0][0][0:number_hemi_nodes].reshape((number_hemi_nodes)),(-1,1)),dtype=torch.float)

        nocurv=np.isnan(curvature)
        curvature[nocurv==1] = 0

        noR2=np.isnan(R2_values)
        R2_values[noR2==1]=0


        condition=R2_values < threshold
        condition2=np.isnan(eccentricity_values)
        condition3=np.isnan(polarAngle_values)


        eccentricity_values[condition == 1] = curvature[condition==1]
        eccentricity_values[condition2 == 1] = curvature[condition2==1]

        polarAngle_values[condition==1] = curvature[condition==1]
        polarAngle_values[condition3 == 1] = curvature[condition3==1]

        data = Data(x=curvature, y=polarAngle_values, pos=pos)
        data.face = faces
        data.R2 = R2_values

    return data
