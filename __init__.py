import os
import copy
import math
import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral
from functools import reduce
import plotly.express as px
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split



default_path = ''

PATH_ori = os.getcwd()+'/data/ori_data/'

PATH_pre = os.getcwd()+'/data/pre_data/'

PATH_graph = os.getcwd()+'/data/graph_data/'

PATH_block = os.getcwd()+'/data/split_block/'


PATH_block_out = os.getcwd()+'/data/Block_graph_data/'

PATH_block_out_fuzzy = os.getcwd()+'/data/Block_graph_data_Fuzzy_ent/'

PATH_block_out_Spectral = os.getcwd()+'/data/Block_graph_data_Spectral_cluster/'

PATH_block_out_Spectral_Fuzzy = os.getcwd()+'/data/Block_graph_data_Spectral_Fuzzy/'

PATH_block_out_DSP = os.getcwd()+'/data/Block_graph_data_DSP/'



PATH_block_pre_out = os.getcwd()+'/data/Block_pre_data/'

PATH_block_pre_out_fuzzy = os.getcwd()+'/data/Block_pre_data_Fuzzy_ent/'

PATH_block_pre_out_Spectral = os.getcwd()+'/data/Block_pre_data_Spectral_cluster/'

PATH_block_pre_out_Spectral_Fuzzy = os.getcwd()+'/data/Block_pre_data_Spectral_Fuzzy/'

PATH_block_pre_out_DSP = os.getcwd()+'/data/Block_pre_data_DSP/'

# print(os.getcwd()) #C:\Users\HP\PycharmProjects\Transformer