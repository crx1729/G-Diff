import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset

def get_graph(dict_data, n):
    transfer_sep = [[], []]
    for u in dict_data.keys():
        item_sep = dict_data[u]
        transfer_sep[0].extend(item_sep[:-1])
        transfer_sep[1].extend(item_sep[1:])
    g = torch.sparse_coo_tensor(transfer_sep, torch.ones(len(transfer_sep[0]), dtype=torch.float32), (n, n))
    d = 1 / g.to_dense().sum(dim=1)
    d[torch.isinf(d)] = 0
    d = torch.diag(d)
    return torch.sparse.mm(g.t(), d)

def data_load(train_path, valid_path, test_path, w_min, w_max):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid

    n_user, n_item = np.r_[np.array(train_list), np.array(valid_list), np.array(test_list)].max(axis=0)
    n_user, n_item = n_user + 1, n_item + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_weight = []
    train_list = []
    for uid in train_dict:
        int_num = len(train_dict[uid])
        weight = np.linspace(w_min, w_max, int_num)  # Temporal-award Training
        # weight = get_weight(w_min, w_max, int_num)
        train_weight.extend(weight)
        for iid in train_dict[uid]:
            train_list.append([uid, iid])
    train_list = np.array(train_list)
    ii_graph = get_graph(train_dict, n_item)
    train_data_temp = sp.csr_matrix((train_weight, (train_list[:, 0], train_list[:, 1])),
                                    dtype='float64', shape=(n_user, n_item))

    train_data_ori = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))

    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    return train_data_temp, train_data_ori, valid_y_data, test_y_data, n_user, n_item, ii_graph

def get_weight(w_min, w_max, int_num):
    a = (1 + int_num) ** (1/(w_max - w_min))
    m = np.arange(1, int_num+1)
    weight = w_min + (np.log2(1 + m) / np.log2(a))
    return weight

class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)
