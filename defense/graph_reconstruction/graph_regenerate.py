from __future__ import division
from __future__ import print_function

import math
import os
import time
import argparse
from math import ceil

import numpy as np
import pandas as pd
import torch_geometric
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from mask.add_diagonal_matrix import diagnoal_matrix, matrix_add, \
    add_diagonal_and_normalize_edge, degree_limit_index, self_connecting
from mask.compute_scores import keep_edges_add_many_edge_from_zero_priD
from mask.mata_grads import compute_matrix_grads

import random

from model.GCN import GCN
from utils.matrix_operation import normalize_edge
from utils.train import train, test
from utils.utils import replace_elements



def graph_regenerate_different(features,dense_matrix,labels,idx_train,idx_val,idx_test,hidden,dropout,lr,weight_decay,prune,degree_limit,device,pri_d):

    A_hat = add_diagonal_and_normalize_edge(dense_matrix,device)
    edge_num=torch.count_nonzero(dense_matrix)
    print("number_edges:",edge_num)

    dense_matrix=self_connecting(dense_matrix,device)
    matrix_vec = torch.cat([score.flatten() for score in dense_matrix])
    origin_matrix_vec_non_zero_indices = torch.nonzero(matrix_vec)

    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout).to(device)


    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    print(features.size(), A_hat.size(),labels.size())
    train(100, features, A_hat, labels, idx_train, idx_val, model, optimizer)
    test(features, A_hat, labels, idx_test, model)

    # generate the new label
    output = model(features, A_hat)
    new_label_all = replace_elements(labels, output.max(1)[1], idx_test)  # B覆盖A，对应idx的位置上

    # Model and optimizer
    model = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout).to(device)

    # model = GCN_onelayer(nfeat=features.shape[1],
    #             nhid=hidden,
    #             nclass=labels.max().item() + 1,
    #             dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)


    num_to_keep = math.ceil((1-prune)*torch.count_nonzero(dense_matrix))

    keep_adj=torch.eye(len(labels))  #construct the initial mask

    # keep_adj = torch.tensor(torch.eye(len(labels)), requires_grad=True).to(device)

    A_hat,_ = normalize_edge(keep_adj,device)
    # 用对角先训练
    train(100, features, A_hat, labels, idx_train, idx_val, model, optimizer)
    degree_limit_indexs = torch.tensor([], dtype=torch.long).to(device)

    torch.cuda.empty_cache()

    print("num_to_keep:",num_to_keep)

    for i in range(num_to_keep):

        train(5, features, A_hat, labels, idx_train, idx_val, model, optimizer)

        # the gradient of adj
        matrix_grads = compute_matrix_grads(A_hat, features, new_label_all, model, idx_train, idx_test,device)

        edge_num_gen = 1

        keep_adj= keep_edges_add_many_edge_from_zero_priD(matrix_grads, keep_adj, device, edge_num_gen,degree_limit_indexs,origin_matrix_vec_non_zero_indices,matrix_vec,pri_d)

        non_zero_indices = torch.nonzero(keep_adj)

        non_zero_count = non_zero_indices.size(0)

        A_hat,D = normalize_edge(keep_adj,device)

        if degree_limit < 100:
            degree_limit_indexs = degree_limit_index(D, degree_limit)

        test(features, A_hat, labels, idx_test, model)

    # retrain：
    t_total = time.time()
    # Model and optimizer
    model2 = GCN(nfeat=features.shape[1],
                nhid=hidden,
                nclass=labels.max().item() + 1,
                dropout=dropout).to(device)

    optimizer = optim.Adam(model2.parameters(),
                           lr=lr, weight_decay=weight_decay)

    epochs = 100
    train(epochs, features, A_hat, labels, idx_train, idx_val, model2, optimizer)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    acc_test = test(features, A_hat, labels, idx_test, model2)

    return acc_test.cpu(),edge_num.cpu(),num_to_keep,model2,A_hat

