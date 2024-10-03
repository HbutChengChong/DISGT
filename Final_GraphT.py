#32*32
import torch
import dgl
import os
import pickle
import random
import dgl.sparse as dglsp
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from __init__ import *
import matplotlib.pyplot as plt
import time
from dgl.data.utils import makedirs, save_info, load_info
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image

from dgl.nn.pytorch.conv.graphconv import GraphConv
from dgl.dataloading import GraphDataLoader
from Final_MyDSPGraphDataset import MyFSPGraphDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# GT_num = 5
# band_select_num = 36

band_num = 200 #IP
# band_num = 204 #Salinas
# band_num = 103 #PaviaU
# band_num = 274 #Hanchuan


pos_enc_size = 32
epoch_num = 50
def Black_Box(G, band_select_num, band_num):
    subgraph_num = []
    # select_num = []
    select_num_int = []
    # select_num_ceil = []
    select_num_float = []
    for i, subgraph in enumerate(G):
        subgraph_num.append(subgraph.number_of_nodes())
        select_num_int.append(int(band_select_num * subgraph.number_of_nodes() / band_num))
        select_num_float.append((band_select_num * subgraph.number_of_nodes() / band_num))
        select_num_diff = [select_num_float[i] - select_num_int[i] for i in range(len(select_num_float))]

    select_num_diff_numpy = np.array(select_num_diff)
    select_num_diff_sort = np.argsort(-select_num_diff_numpy)
    band_select_sum = sum(select_num_int)
    diff = band_select_num - band_select_sum
    i = 0
    while diff > 0:
        select_num_int[select_num_diff_sort[i]] += 1
        diff -= 1
        i += 1
    return subgraph_num, select_num_int


class SparseMHA(nn.Module):
    def __init__(self, hidden_size=256, num_heads=8):
        super(SparseMHA, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_trans = nn.Linear(hidden_size, hidden_size)
        self.k_trans = nn.Linear(hidden_size, hidden_size)
        self.v_trans = nn.Linear(hidden_size, hidden_size)
        self.out_trans = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h):
        N = len(h)
        q = self.q_trans(h).reshape(N, self.head_dim, self.num_heads)
        k = self.k_trans(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_trans(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling

        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))
        attn = attn.softmax()
        out = dglsp.bspmm(attn, v)
        out = self.out_trans(out.reshape(N, -1))

        return out

class GTLayer(nn.Module):
    def __init__(self,hidden_size=256, num_heads=8):
        super(GTLayer, self).__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size*2)
        self.FFN2 = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, A, h):
        h1 = h
        h = self.MHA(A, h)
        h = self.batchnorm1(h + h1)

        h2 = h
        h = F.relu(self.FFN1(h))
        h = self.FFN2(h)
        h = h + h2
        h = self.batchnorm2(h)

        return h

class GraphTransformer(nn.Module):
    def __init__(self, out_size=1, hidden_size=256, pos_enc_size=32, num_layer=8, num_heads=8):
        super(GraphTransformer, self).__init__()
        self.atom_encoder = AtomEncoder(hidden_size)
        self.GCN = GraphConv(1024, 256)
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.layers = nn.ModuleList([GTLayer(hidden_size, num_heads) for _ in range(num_layer)])
        self.pool = dglnn.SumPooling()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, out_size),
            # nn.Softmax()
            nn.Sigmoid()
        )

    def forward(self, g, X, pos_enc):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        E = g.edata['efeat'].view(-1)
        A = dglsp.spmatrix(indices, shape=(N, N), val=E)
        h = self.GCN(g, X) + self.pos_linear(pos_enc)
        for layer in self.layers:
            h = layer(A, h)
        # h = self.pool(g, h)

        return self.predictor(h)

class SGT(nn.Module):
    def __init__(self, len):
        super(SGT, self).__init__()
        self.lens = len
        self.GT_layers = nn.ModuleList([GraphTransformer() for _ in range(self.lens)])

    def forward(self, G, band_select_num, F):#this G is the gg = G[0]
        if len(G) == 1:
            subgraph_num = band_num
            select_num = band_select_num
        else:
            subgraph_num, select_num = Black_Box(G, band_select_num, F.shape[0])
        for i in range(len(G)):
            G[i] = G[i].to(device)
        select_band_interact_index = torch.tensor([], device=device)
        select_band_interact_weight = torch.tensor([], device=device)
        pred_ori_num = 0
        i = 0
        # for i in range(len(G)):
        for GT_layer in self.GT_layers:
            pred = GT_layer(G[i], G[i].ndata['feat'].long(), G[i].ndata["PE"])
            if len(G) != 1:
                pred_num = select_num[i]
            else:
                pred_num = select_num
            pred_weight, indices = pred.topk(k=pred_num, dim=0, largest=True, sorted=True)
            indices = indices + (pred_ori_num)


            select_band_interact_index = torch.cat((select_band_interact_index, indices), 0)
            select_band_interact_weight = torch.cat((select_band_interact_weight, pred_weight), 0)
            if len(G) != 1:
                pred_ori_num += subgraph_num[i]
            else:
                pred_ori_num = subgraph_num
            i += 1
        return select_band_interact_index, select_band_interact_weight

class ReconstructNet(nn.Module):
    def __init__(self, GT_num,  hidden_size, out_size=band_num):
        super(ReconstructNet, self).__init__()
        self.SGT = nn.DataParallel(SGT(len=GT_num))
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, out_size),
        )

    def forward(self, G, img, F, band_select_num):

        INDEX = torch.arange(band_num).reshape(band_num, 1).to(device)  # Salinas
        F = torch.transpose(F, 0, 1)
        index, select_band_ori_weight = self.SGT(G, band_select_num, F)
        index = index.long()
        select_band_ori = F[index.view(-1)]
        mask = torch.zeros_like(INDEX, dtype=torch.bool).to(device)
        mask[select_band_ori] = True  # 掩码向量
        input_size = torch.sum(mask).item()
        selected_img = img[:, mask.squeeze(), :]
        selected_img = torch.transpose(selected_img, 1, 2)
        if input_size != band_select_num:
            zero_column = torch.zeros(1, 1024, band_select_num - input_size).to(device)
            selected_img = torch.cat([selected_img, zero_column], dim=2)
        predict = self.predictor(selected_img)
        return predict, select_band_ori, select_band_ori_weight


def train(band_select_num, GT_num, model_save_path, data_name, path_graph, path_sub):

    dataset = MyFSPGraphDataset(data_name, path_graph, path_sub)
    numclasses = dataset.num_tasks
    dataloader = GraphDataLoader(dataset, batch_size=1, shuffle=True)
    model = ReconstructNet(GT_num=GT_num, hidden_size=band_select_num).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fcn = nn.MSELoss()
    loss_list = []
    epoch_list = []
    min_loss = 1000
    model.train()
    for epoch in range(epoch_num):
        total_loss = 0.0
        for gs, img, label, SPlist, F in tqdm(dataloader):
            img = img.to(device)
            F = F.to(device)
            optimizer.zero_grad()
            for q, subgraph in enumerate(gs):
                subgraph.ndata["PE"] = dgl.laplacian_pe(subgraph, k=pos_enc_size, padding=True)
            pred, select_band, _ = model(gs, img, F, band_select_num)
            # Counter[select_band] += 1
            compare_img = torch.transpose(img, 1, 2)
            loss = loss_fcn(pred, compare_img)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        loss_list.append(avg_loss)
        epoch_list.append(epoch)
        print(f"Epoch: {epoch:03d}, Loss: {avg_loss:.4f}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)


def Output_band(band_select_num, GT_num, model_save_path, select_band_save_path, data_name, path_graph, path_sub):
    Counter = torch.zeros(band_num, 1).to(device)  # Salinas
    dataset = MyFSPGraphDataset(data_name, path_graph, path_sub)
    numclasses = dataset.num_tasks
    test_dataloader = GraphDataLoader(dataset, batch_size=1, shuffle=True)
    model = ReconstructNet(GT_num=GT_num, hidden_size=band_select_num).to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    i = 0
    for gs, img, label, SPlist, F in tqdm(test_dataloader):
        img = img.to(device)
        F = F.to(device)
        for q, subgraph in enumerate(gs):
            subgraph.ndata["PE"] = dgl.laplacian_pe(subgraph, k=pos_enc_size, padding=True)
            # subgraph.to(device)
        pred, select_band, select_band_weight = model(gs, img, F, band_select_num)
        counter = torch.zeros(band_num, 1).to(device)
        if i % 8==0:
            counter.scatter_(0, select_band, select_band_weight)
            Counter += counter
        i += 1
    Counter = Counter / i

    sorted_indices = torch.argsort(Counter.view(-1), descending=True).to(device)
    select_band_unsorted = sorted_indices[:band_select_num]
    sorted_counts = Counter[select_band_unsorted]
    sorted_bands, sorted_index = torch.sort(select_band_unsorted)
    np.save(select_band_save_path, sorted_bands.cpu())
    print("Select Bands:", sorted_bands)

