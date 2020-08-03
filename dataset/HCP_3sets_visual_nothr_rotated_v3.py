import os.path as osp
import scipy.io
from functions.def_ROIs_WangParcels import roi
import torch
from torch_geometric.data import InMemoryDataset
from read.read_HCPdata_visual_nothr_rotated import read_HCP
from functions.labels import labels

#Generates the training and test set separately

#Version that includes all ventral-temporal and dorsal-lateral areas

class Retinotopy(InMemoryDataset):
    url = 'https://balsa.wustl.edu/study/show/9Zkk'
    def __init__(self,
                 root,
                 set=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 n_examples=None,
                 myelination=None,
                 prediction=None):
        self.myelination=myelination
        self.prediction=prediction
        self.n_examples = int(n_examples)
        super(Retinotopy, self).__init__(root, transform, pre_transform, pre_filter)
        self.set=set
        if self.set=='Train':
            path = self.processed_paths[0]
        elif self.set == 'Development':
            path = self.processed_paths[1]
        else:
            path = self.processed_paths[2]
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        return 'S1200_7T_Retinotopy_9Zkk.zip'

    @property
    def processed_file_names(self):
        if self.myelination==True:
            if self.prediction == 'eccentricity':
                return ['training_ecc_visual_nothresh_rotated_myelincurv_v3.pt', 'development_ecc_visual_nothresh_rotated_myelincurv_v3.pt',
                        'test_ecc_visual_nothresh_rotated_myelincurv_v3.pt']
            else:
                return ['training_PA_visual_nothresh_rotated_myelincurv_v3.pt', 'development_PA_visual_nothresh_rotated_myelincurv_v3.pt',
                        'test_PA_visual_nothresh_rotated_myelincurv_v3.pt']
        else:
            if self.prediction=='eccentricity':
                return ['training_ecc_visual_nothresh_rotated_v3.pt','development_ecc_visual_nothresh_rotated_v3.pt','test_ecc_visual_nothresh_rotated_v3.pt']
            else:
                return ['training_PA_visual_nothresh_rotated_v3.pt','development_PA_visual_nothresh_rotated_v3.pt','test_PA_visual_nothresh_rotated_v3.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download S1200_7T_Retinotopy_9Zkk.zip from {} and '
            'move it to {} and execute SettingDataset.sh'.format(self.url, self.raw_dir))

    def process(self):
        #extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        path=osp.join(self.raw_dir, 'converted')
        data_list=[]

        # Selecting only V1,V2 and V3
        label_primary_visual_areas = ['V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v','hV4','VO1','VO2','PHC1','PHC2','V3a','V3b','LO1','LO2','TO1','TO2','IPS0','IPS1','IPS2','IPS3']
        final_mask_L, final_mask_R, index_L_mask, index_R_mask= roi(label_primary_visual_areas)

        faces_R = labels(scipy.io.loadmat(osp.join(path,'tri_faces_R.mat'))['tri_faces_R']-1, index_R_mask)
        faces_L = labels(scipy.io.loadmat(osp.join(path, 'tri_faces_L.mat'))['tri_faces_L'] - 1, index_L_mask)



        for i in range(0,self.n_examples):
            data=read_HCP(path,Hemisphere='Left',index=i,surface='mid',visual_mask_L=final_mask_L,visual_mask_R=final_mask_R,faces_L=faces_L,faces_R=faces_R,myelination=self.myelination,prediction=self.prediction)
            if self.pre_transform is not None:
                data=self.pre_transform(data)
            data_list.append(data)

        train = data_list[0:int(161)]
        dev = data_list[int(161):int(171)]
        test = data_list[int(171):len(data_list)]

        torch.save(self.collate(train),self.processed_paths[0])
        torch.save(self.collate(dev), self.processed_paths[1])
        torch.save(self.collate(test), self.processed_paths[2])
