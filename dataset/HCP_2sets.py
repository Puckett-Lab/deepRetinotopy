import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from read.read_HCPdata import read_HCP

#Generates the training and test set separately


class Retinotopy(InMemoryDataset):
    url = 'https://balsa.wustl.edu/study/show/9Zkk'
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 n_examples=None):
        self.n_examples = int(n_examples)
        super(Retinotopy, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)


    @property
    def raw_file_names(self):
        return 'S1200_7T_Retinotopy_9Zkk.zip'

    @property
    def processed_file_names(self):
        return ['training.pt','development.pt','test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download S1200_7T_Retinotopy_9Zkk.zip from {} and '
            'move it to {} and execute SettingDataset.sh'.format(self.url, self.raw_dir))

    def process(self):
        #extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        path=osp.join(self.raw_dir, 'converted')
        data_list=[]
        for i in range(0,self.n_examples):
            data=read_HCP(path,Hemisphere='Left',index=i,surface='mid',threshold=2.2)
            if self.pre_transform is not None:
                data=self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list[:145]),self.processed_paths[0])
        torch.save(self.collate(data_list[145:]), self.processed_paths[1])


