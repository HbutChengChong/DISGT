import math

import torch
import torch.nn as nn
import dgl
from tqdm import tqdm
import torch.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs
import scipy.io as sio
import numpy as np
import pickle
import skfuzzy as fuzz
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform
from Final_DSP import FSP
from __init__ import *

device = ''
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
def fuzzy_entropy(membership_matrix):
    entropy_values = -np.sum(membership_matrix * np.log2(membership_matrix), axis=0)
    average_entropy = np.mean(entropy_values)
    return average_entropy

class CONSTRUCT_SUBGRAPH(object):
    def __init__(self, data_name, Subspace, split_x, split_y, path_save_graph, path_save_Sub, deltas, lambdas, p, threshold):
        # self.Subspace = math.ceil(band_select_num/10)
        self.Subspace = Subspace
        self.data_name = data_name
        self.threshold = threshold
        self.deltas = deltas
        self.lambdas = lambdas
        self.p = p

        self.path_in_ori = PATH_ori
        self.path_in = PATH_block
        self.path_graph_out = path_save_graph# for P2P
        self.path_Sub_out = path_save_Sub# for P2P

        self.split_x = split_x
        self.split_y = split_y

        self.Subspace_list_DSP = []  # for DSP cluster
        self.SubGraph_list_DSP = []

        self.read_split_data()
        self.height, self.width, self.bands_num = self.data.shape

        self.index = torch.arange(self.bands_num)
        self.subgraph_bands_num_mean = int(self.bands_num / self.Subspace)

        self.DSP_data = self.data
        self.data = np.reshape(self.data, (self.height * self.width, self.bands_num))
        self.transpose_matrix = np.transpose(self.data) # 200*(32*32)

        self.DSP()
        self.Construct_Graph_set_DSP()
        self.save_graph_DSP()

    def readData(self):
        if self.data_name == 'Indian_pines':
            self.data_ori = sio.loadmat(self.path_in_ori + 'Indian_pines_corrected.mat')['indian_pines_corrected']
            self.labels_ori = sio.loadmat(self.path_in_ori + 'Indian_pines_gt.mat')['indian_pines_gt']
        elif self.data_name == 'PaviaU':
            self.data_ori = sio.loadmat(self.path_in_ori + 'PaviaU.mat')['paviaU']
            self.labels_ori = sio.loadmat(self.path_in_ori + 'PaviaU_gt.mat')['paviaU_gt']
        elif self.data_name == 'KSC':
            self.data_ori = sio.loadmat(self.path_in_ori + 'KSC.mat')['KSC']
            self.labels_ori = sio.loadmat(self.path_in_ori + 'KSC_gt.mat')['KSC_gt']
        elif self.data_name == 'Salinas':
            self.data_ori = sio.loadmat(self.path_in_ori + 'Salinas_corrected.mat')['salinas_corrected']
            self.labels_ori = sio.loadmat(self.path_in_ori + 'Salinas_gt.mat')['salinas_gt']
        elif self.data_name == 'washington':
            self.data_ori = sio.loadmat(self.path_in_ori + 'washington.mat')['washington_datax']
            self.labels_ori = sio.loadmat(self.path_in_ori + 'washington_gt.mat')['washington_labelx']
        elif self.data_name == 'Houston':
            self.data_ori = sio.loadmat(self.path_in_ori + 'Houstondata.mat')['Houstondata']
            self.labels_ori = sio.loadmat(self.path_in_ori + 'Houstonlabel.mat')['Houstonlabel']
        elif self.data_name == 'Houston_1':
            self.data_ori = sio.loadmat(self.path_in_ori + 'hou_0.mat')['hou_data']
            self.labels_ori = sio.loadmat(self.path_in_ori + 'hou_0_change_label.mat')['hou_label']
        elif self.data_name == 'Hanchuan':
            self.data_ori = sio.loadmat(self.path_in_ori + 'Hanchuan.mat')['Hanchuan']
            self.labels_ori = sio.loadmat(self.path_in_ori + 'Hanchuan_gt.mat')['Hanchuan_gt']
        self.data_ori = np.float64(self.data_ori)
        self.labels_ori = np.array(self.labels_ori).astype(float)

    def read_split_data(self):
        if self.data_name == 'Indian_pines':
            self.data = np.load(self.path_in+'Indian_Pines_32/img_slide/_{}-{}_.npy'.format(self.split_x, self.split_y))
            self.labels = np.load(self.path_in+'Indian_Pines_32/label_slide/_{}-{}_.npy'.format(self.split_x, self.split_y))
        elif self.data_name == 'Salinas':
            self.data = np.load(self.path_in + 'Salinas_32/img_slide/_{}-{}_.npy'.format(self.split_x, self.split_y))
            self.labels = np.load(self.path_in + 'Salinas_32/label_slide/_{}-{}_.npy'.format(self.split_x, self.split_y))
        elif self.data_name == 'Hanchuan':
            self.data = np.load(self.path_in + 'Hanchuan_32/img_slide/_{}-{}_.npy'.format(self.split_x, self.split_y))
            self.labels = np.load(self.path_in + 'Hanchuan_32/label_slide/_{}-{}_.npy'.format(self.split_x, self.split_y))
        elif self.data_name == 'PaviaU':
            self.data = np.load(self.path_in + 'PaviaU_32/img_slide/_{}-{}_.npy'.format(self.split_x, self.split_y))
            self.labels = np.load(self.path_in + 'PaviaU_32/label_slide/_{}-{}_.npy'.format(self.split_x, self.split_y))
            # self.data = np.float64(self.data)
            # self.labels = np.array(self.labels).astype(float)
    def Mexican_hat(self, x, x_prime, delta):
        product_term = torch.prod(1 - (x - x_prime) ** 2 / delta ** 2)
        exp_term = torch.exp(-torch.sum((x - x_prime) ** 2) / (2 * delta ** 2))
        return product_term * exp_term


    def DSP(self):
        v, u, _, _ = FSP(self.DSP_data, self.Subspace, delta=self.deltas, lambdas=self.lambdas, p=self.p).Fuzzy_Subspace_Partition()
        # interact
        for i in range(self.Subspace):
            labels = np.where(u[i, :] > self.threshold)[0]
            self.Subspace_list_DSP.append(labels)

        # # Uninteract
        # labels = np.argmax(u, axis=0)
        # print('uninteract!!!!!!')
        # for i in range(self.Subspace):
        #     members = np.where(labels == i)[0]
        #     self.Subspace_list_DSP.append(members)

        return self.Subspace_list_DSP

    def Construct_Graph(self):
        Graph = self.transpose_matrix #200*(32*32)
        node_num = self.bands_num #200
        self.allG.add_nodes(node_num)
        # self.allG.edata['efeats'] = torch.ones(40000, 1)
        Graph = torch.Tensor(Graph)
        features = Graph
        self.allG.ndata['ndata'] = features
        for i in range(node_num):
            for j in range(node_num):
                # corr = nn.functional.cosine_similarity(features[i], features[j], dim=0)
                self.allG = dgl.add_edges(self.allG, i, j,
                                          {'efeat': torch.Tensor(self.Pearsonresult[i][j].reshape(1, 1))})
        # self.allG = dgl.add_reverse_edges(self.allG)
        return self.allG

    def Construct_SubGraph_DSP(self, Subgraph_NO):
        Subset_Band = self.transpose_matrix.take(self.Subspace_list_DSP[Subgraph_NO], axis=0)
        SubGraph = Subset_Band
        sub_node_num = Subset_Band.shape[0]
        g = dgl.DGLGraph().to(device)
        g.add_nodes(sub_node_num)
        SubGraph = torch.Tensor(SubGraph).to(device)
        features = SubGraph
        g.ndata['feat'] = features
        g.ndata['NO.'] = torch.Tensor(self.Subspace_list_DSP[Subgraph_NO].reshape(sub_node_num, 1)).to(device)
        for i in range(sub_node_num):
            for j in range(sub_node_num):
                if i == j:
                    g = dgl.add_edges(g, i, j,
                                      {'efeat': torch.Tensor([[1.0]]).to(device)})
                else:
                    a = self.Subspace_list_DSP[Subgraph_NO][i]
                    b = self.Subspace_list_DSP[Subgraph_NO][j]
                    delta = math.exp(- abs(a - b))
                    pre_corr = np.corrcoef(Subset_Band[i], Subset_Band[j])[0, 1]
                    if np.isnan(pre_corr):
                        # print("NaN!")
                        corr = np.array(delta)
                    else:
                        corr = pre_corr ** 2 * delta
                    # corr = self.Mexican_hat(x, x_prime, 1)
                    g = dgl.add_edges(g, i, j,
                                      {'efeat': torch.Tensor(corr.reshape(1, 1)).to(device)})
                # corr = torch.nn.functional.cosine_similarity(features[i], features[j], dim=0)
                # if corr > 0.5:
                #     g.add_edges(i, j)
        # newg = dgl.add_reverse_edges(g)
        return g
    def Construct_Graph_set_DSP(self):
        subgraph_num = self.Subspace
        for i in tqdm(range(subgraph_num)):
            g = self.Construct_SubGraph_DSP(i)
            self.SubGraph_list_DSP.append(g)
        return self.SubGraph_list_DSP
    def save_graph_DSP(self):
        print("Save Subgraph...")
        dgl.data.utils.save_graphs(
            self.path_graph_out + f"Subgraphs_set_{self.Subspace}_{self.split_x}_{self.split_y}.dgl",
            self.SubGraph_list_DSP)
        print("Saved Subgraph!")
        print("Save Partition...")
        if os.path.exists(
                self.path_Sub_out + f'Partition_{self.split_x}_{self.split_y}.pkl'):
            pass
        else:
            print("Saved Partition")
            with open(
                    self.path_Sub_out + f'Partition_{self.split_x}_{self.split_y}.pkl',
                    'wb') as file:
                pickle.dump(self.Subspace_list_DSP, file)

