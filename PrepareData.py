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
from FSP import FSP
from __init__ import *

device = ''
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
print(device)
def fuzzy_entropy(membership_matrix):
    entropy_values = -np.sum(membership_matrix * np.log2(membership_matrix), axis=0)
    average_entropy = np.mean(entropy_values)
    return average_entropy

class CONSTRUCT_SUBGRAPH(object):
    def __init__(self, data_name, band_select_num, split_x, split_y, path_save_graph, path_save_Sub):
        self.Subspace = math.ceil(band_select_num/10)
        # self.Subspace = 5
        self.data_name = data_name

        self.path_in_ori = PATH_ori
        self.path_in = PATH_block

        # self.path_out = PATH_block_out
        # self.path_out_fuzzy = PATH_block_out_fuzzy
        # self.path_out_spectral = PATH_block_out_Spectral
        # self.path_out_spectral_fuzzy = PATH_block_out_Spectral_Fuzzy
        # self.path_out_AedFCM = PATH_block_out_AedFCM
        self.path_graph_out = path_save_graph# for P2P

        # self.path_pre = PATH_block_pre_out
        # self.path_pre_fuzzy = PATH_block_pre_out_fuzzy
        # self.path_pre_spectral = PATH_block_pre_out_Spectral
        # self.path_pre_spectral_fuzzy = PATH_block_pre_out_Spectral_Fuzzy
        # self.path_pre_AedFCM = PATH_block_pre_out_AedFCM
        self.path_Sub_out = path_save_Sub# for P2P

        self.split_x = split_x
        self.split_y = split_y

        # self.SubGraph_list = []#for ASPS
        # self.Subspace_list = []#for fuzzy_entropy
        # self.Subspace_list_Spectral = [] #for Spectral cluster
        # self.Subspace_list_Spectral_Fuzzy = []  # for Spectral_fuzzy cluster
        self.Subspace_list_FSP = []  # for AedFCM cluster

        # self.SubGraph_list_Fuzzy = []
        # self.SubGraph_list_Spectral = []
        # self.SubGraph_list_Spectral_fuzzy = []
        self.SubGraph_list_FSP = []

        '''self.readData()'''
        self.read_split_data()

        '''self.height_ori, self.width_ori, self.bands_num_ori = self.data_ori.shape'''
        self.height, self.width, self.bands_num = self.data.shape

        self.index = torch.arange(self.bands_num)
        self.subgraph_bands_num_mean = int(self.bands_num / self.Subspace)
        # self.G = dgl.DGLGraph()
        '''self.allG = dgl.DGLGraph()'''

        # 将数据集转换为二维矩阵形式
        '''self.data_ori = np.reshape(self.data_ori, (self.height_ori * self.width_ori, self.bands_num_ori))'''
        self.FSP_data = self.data
        self.data = np.reshape(self.data, (self.height * self.width, self.bands_num))
        '''self.transpose_matrix_ori = np.transpose(self.data_ori) #200*(145*145)'''
        self.transpose_matrix = np.transpose(self.data) # 200*(32*32)

        '''self.Euclidean_Distance_Ori()
        self.Euclidean_Distance()
        self.PearsonMatrix_Ori()
        self.PearsonMatrix()
        self.Dij_Matrix_Ori()
        self.Dij_Matrix()'''

        '''self.ASPS()'''
        '''self.Fuzzy_entropy()'''
        '''self.Spectral_cluster()'''
        '''self.Spectral_Fuzzy_cluster()'''
        self.FSP()

        '''self.Construct_Graph()'''
        '''self.Construct_Graph_set()'''
        '''self.Construct_Graph_set_Fuzzy_Ent()'''
        '''self.Construct_Graph_set_Spectral_Fuzzy()'''
        self.Construct_Graph_set_FSP()

        # self.save_graph()
        '''self.save_graph_Fuzzy_Ent()'''
        self.save_graph_FSP()

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

    def Euclidean_Distance_Ori(self):
        # 输出欧式距离矩阵
        self.similarMatrix_ori = self.transpose_matrix_ori
        self.distance_ori = pdist(self.similarMatrix_ori, metric='euclidean')
        self.distance_matrix_ori = squareform(self.distance_ori)
        # return distance_matrix

    def Euclidean_Distance(self):
        # 输出欧式距离矩阵
        self.similarMatrix = self.transpose_matrix
        self.distance = pdist(self.similarMatrix, metric='euclidean')
        self.distance_matrix = squareform(self.distance)
        # return distance_matrix

    def PearsonMatrix_Ori(self):
        # 输出皮尔逊系数矩阵
        self.Pearsonresult_ori = np.zeros((self.bands_num_ori, self.bands_num_ori))
        # 计算每两行之间的皮尔逊系数
        for i in range(self.bands_num_ori):
            for j in range(i, self.bands_num_ori):
                if i == j:
                    self.Pearsonresult_ori[i, j] = 1.0  # 对角线上的元素为1
                else:
                    pearson = np.corrcoef(self.transpose_matrix_ori[i], self.transpose_matrix_ori[j])[0, 1]
                    self.Pearsonresult_ori[i, j] = pearson
                    self.Pearsonresult_ori[j, i] = pearson  # 结果矩阵是对称矩阵，需要同时更新result[i,j]和result[j,i]
        return self.Pearsonresult_ori

    def PearsonMatrix(self):
        # 输出皮尔逊系数矩阵
        self.Pearsonresult = np.zeros((self.bands_num, self.bands_num))
        # 计算每两行之间的皮尔逊系数
        for i in range(self.bands_num):
            for j in range(i, self.bands_num):
                if i == j:
                    self.Pearsonresult[i, j] = 1.0  # 对角线上的元素为1
                else:
                    pearson = np.corrcoef(self.transpose_matrix[i], self.transpose_matrix[j])[0, 1]
                    self.Pearsonresult[i, j] = pearson
                    self.Pearsonresult[j, i] = pearson  # 结果矩阵是对称矩阵，需要同时更新result[i,j]和result[j,i]
        return self.Pearsonresult

    def Dij_Matrix_Ori(self):
        self.Dijresult_ori = np.subtract(1, abs(self.Pearsonresult_ori))

    def Dij_Matrix(self):
        self.Dijresult = np.subtract(1, abs(self.Pearsonresult))
    def FSP(self):
        v, u, _, _ = FSP(self.FSP_data, self.Subspace).Fuzzy_Subspace_Partition()
        labels = np.argmax(u, axis=0)

        # 输出每个数据点所属的子空间
        for i in range(self.Subspace):
            members = np.where(labels == i)[0]
            self.Subspace_list_FSP.append(members)

        return self.Subspace_list_FSP

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

    def Construct_SubGraph_FSP(self, Subgraph_NO):
        Subset_Band = self.transpose_matrix.take(self.Subspace_list_FSP[Subgraph_NO], axis=0)
        SubGraph = Subset_Band
        sub_node_num = Subset_Band.shape[0]
        g = dgl.DGLGraph().to(device)
        print(g.device)
        g.add_nodes(sub_node_num)
        SubGraph = torch.Tensor(SubGraph).to(device)
        features = SubGraph
        g.ndata['feat'] = features
        g.ndata['NO.'] = torch.Tensor(self.Subspace_list_FSP[Subgraph_NO].reshape(sub_node_num, 1)).to(device)
        for i in range(sub_node_num):
            for j in range(sub_node_num):
                if i == j:
                    g = dgl.add_edges(g, i, j,
                                      {'efeat': torch.Tensor([[1.0]]).to(device)})
                else:
                    corr = np.corrcoef(Subset_Band[i], Subset_Band[j])[0, 1]
                    g = dgl.add_edges(g, i, j,
                                      {'efeat': torch.Tensor(corr.reshape(1, 1)).to(device)})

                # corr = torch.nn.functional.cosine_similarity(features[i], features[j], dim=0)
                # if corr > 0.5:
                #     g.add_edges(i, j)
        # newg = dgl.add_reverse_edges(g)
        return g
    def Construct_Graph_set_FSP(self):
        subgraph_num = self.Subspace
        for i in tqdm(range(subgraph_num)):
            g = self.Construct_SubGraph_FSP(i)
            self.SubGraph_list_FSP.append(g)
        return self.SubGraph_list_FSP

    def save_graph_FSP(self):
        print("Save Subgraph...")
        dgl.data.utils.save_graphs(
            self.path_graph_out + f"Subgraphs_set_{self.Subspace}_{self.split_x}_{self.split_y}.dgl",
            self.SubGraph_list_FSP)
        print("Saved Subgraph!")
        print("--Save Partition")
        # with open(
        #         self.path_Sub_out + f'Partition_{self.split_x}_{self.split_y}.pkl',
        #         'wb') as file:
        #     pickle.dump(self.Subspace_list_AedFCM, file)
        if os.path.exists(
                self.path_Sub_out + f'Partition_{self.split_x}_{self.split_y}.pkl'):
            pass
        else:
            print("--Save Partition")
            with open(
                    self.path_Sub_out + f'Partition_{self.split_x}_{self.split_y}.pkl',
                    'wb') as file:
                pickle.dump(self.Subspace_list_FSP, file)

        # print("Saved pre data!")


'''data_name = 'Indian_pines'
# data_name = 'Salinas'
# data_name = 'Hanchuan'
band_select_num = 32

# construct_subgraph = CONSTRUCT_SUBGRAPH(data_name, band_select_num, 0, 0)
torch.set_printoptions(precision=7)
# print(construct_subgraph.Partition)
# print(construct_subgraph.transpose_matrix[int(construct_subgraph.Partition[1]):int(construct_subgraph.Partition[1 + 1])])
# print(construct_subgraph.transpose_matrix[int(construct_subgraph.Partition[1]):int(construct_subgraph.Partition[1 + 1])].shape)
# print(construct_subgraph.Construct_SubGraph(1))
# # print(construct_subgraph.transpose_matrix)




for i in range(5):
    for j in range(5):
        construct_subgraph = CONSTRUCT_SUBGRAPH(data_name, band_select_num, i, j)
        print("--"*30, end='')
        print(f"saved {i}_{j}")'''
# construct_subgraph = CONSTRUCT_SUBGRAPH(data_name, band_select_num, 0, 0)
# print("Data shape")
# print(construct_subgraph.data.shape)
# print("Label shape")
# print(construct_subgraph.labels.shape)
# print("Partition ")
# print(construct_subgraph.Partition)
# print("Nodes number of each batch ")
# print(construct_subgraph.G.batch_num_nodes())
# print("SubGraph information ")
# print(construct_subgraph.G)
# print("Graph information ")
# print(construct_subgraph.allG)
# # # print(construct_subgraph.index)


# nx_G = construct_subgraph.Construct_SubGraph(1).to_networkx()
# pos = nx.spring_layout(nx_G)
# nx.draw(nx_G,pos,with_labels=True,node_color='r',edge_color='b',width=1,alpha=0.7)
# plt.show()