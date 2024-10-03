import os
import pickle

import torch
import dgl
from torchvision import transforms
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from __init__ import *

IP_num_classes = 17
PU_num_classes = 9

#Indian_pines
class MyFSPGraphDataset(DGLDataset):

    def __init__(self, name, path_save_graph, path_save_Sub, url=None, raw_dir=None, save_dir=None, force_reload=False, verbose=True):
        self.ori_data_name = name
        self.path_save_graph = path_save_graph
        self.path_save_Sub = path_save_Sub
        # self.Gname = os.listdir(os.path.join(rf'C:\Users\HP\PycharmProjects\Transformer\data\Block_graph_data_AedFCM\{name}_32\Graph_set'))
        self.Gname = os.listdir(os.path.join(self.path_save_graph))
        self.img_name = os.listdir(os.path.join(rf'C:\Users\HP\PycharmProjects\Transformer\data\split_block\{name}_32\img_slide'))
        self.label_name = os.listdir(os.path.join(rf'C:\Users\HP\PycharmProjects\Transformer\data\split_block\{name}_32\label_slide'))
        self.AedFCM_Subspace_list_name = os.listdir(os.path.join(self.path_save_Sub))

    def download(self):
        pass

    def process(self):
        pass

    def __len__(self):
        # 获取文件夹中文件的数量
        return len(self.Gname)

    def __getitem__(self, idx):
        SubGraph_name = self.Gname[idx]
        label_name = self.label_name[idx]
        img_name = self.img_name[idx]
        Subspace_list_name = self.AedFCM_Subspace_list_name[idx]

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_path = os.path.join(rf'C:\Users\HP\PycharmProjects\Transformer\data\split_block\{self.ori_data_name}_32\img_slide', img_name)
        Graph_path = os.path.join(self.path_save_graph, SubGraph_name)
        label_path = os.path.join(rf'C:\Users\HP\PycharmProjects\Transformer\data\split_block\{self.ori_data_name}_32\label_slide', label_name)
         Subspace_list_path = os.path.join(self.path_save_Sub, Subspace_list_name)

        img = np.load(img_path)
        H, W, C = img.shape
        img = np.reshape(img, (H * W, C))
        img = img.T
        img = img.astype(np.float32)

        SubGraphs = dgl.load_graphs(Graph_path)
        subgraphs = SubGraphs[0]

        label = np.load(label_path)
        label = label.astype(np.float32)
        label = np.reshape(label, (H * W, 1))

        with open(Subspace_list_path, 'rb') as f:
            Subspace_list = pickle.load(f)
        list_num = len(Subspace_list) #4
        F = Subspace_list[0]
        for i in range(list_num - 1):
            F = np.concatenate((F, Subspace_list[i + 1]))

        return subgraphs, img, label, Subspace_list, F

    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass

    @property
    def num_tasks(self):
        return 17


