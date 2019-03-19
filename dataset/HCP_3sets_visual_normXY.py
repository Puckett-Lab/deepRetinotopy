import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset
from read.read_HCPdata_visual import read_HCP

#Generates the training and test set separately


class Retinotopy(InMemoryDataset):
    url = 'https://balsa.wustl.edu/study/show/9Zkk'
    def __init__(self,
                 root,
                 set=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 n_examples=None,prediction=None):
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
        return ['training_visual_norm.pt','development_visual_norm.pt','test_visual_norm.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download S1200_7T_Retinotopy_9Zkk.zip from {} and '
            'move it to {} and execute SettingDataset.sh'.format(self.url, self.raw_dir))

    def process(self):
        #extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        path=osp.join(self.raw_dir, 'converted')
        data_list=[]
        for i in range(0,self.n_examples):
            data=read_HCP(path,Hemisphere='Left',index=i,surface='mid',threshold=2.2,prediction=self.prediction)
            if self.pre_transform is not None:
                data=self.pre_transform(data)
            data.y=(data.y-torch.mean(data.y))/torch.std(data.y)
            data.Ymean=torch.mean(data.y)
            data.Ystd=torch.std(data.y)

            data.x = (data.x - torch.mean(data.x)) / torch.std(data.x)
            data.Xmean = torch.mean(data.x)
            data.Xstd = torch.std(data.x)

            data_list.append(data)

        train = data_list[0:int(round(len(data_list) * 0.6))]
        dev = data_list[int(round(len(data_list) * 0.6)):int(round(len(data_list) * 0.8))]
        test = data_list[int(round(len(data_list) * 0.8)):len(data_list)]

        torch.save(self.collate(train),self.processed_paths[0])
        torch.save(self.collate(dev), self.processed_paths[1])
        torch.save(self.collate(test), self.processed_paths[2])
