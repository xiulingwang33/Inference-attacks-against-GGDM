from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops

from data.dataload import load_data,load_data_att_defense

from graph_reconstruction.graph_regenerate import graph_regenerate_different
from graph_sparsification.graph_sparsification import graph_sparsification
from utils.utils import split_dataset

from collections import defaultdict
from torch_geometric.data import Data

# CUDA_VISIBLE_DEVICES=2 python main-defense-preprocessing.py --dataset MUTAG --prune 0.1
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='regen',choices=['regen','spars','train','train_with_perturb'])
    parser.add_argument('--dataset', type=str, default='ENZYMES'
                            ,choices=['MUTAG','ENZYMES','IMDB-BINARY','IMDB-MULTI','imdb-binary', 'imdb-multi','mutag', 'imdb-binary', 'imdb-multi','enzymes','proteins'])
    parser.add_argument('--device', type=str, default='cuda',choices=['cpu','cuda'])
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='units dropout')
    parser.add_argument('--prune', type=float, default=0.5,
                        help='prune how many edges')
    parser.add_argument('--ratio_of_train_set', type=float, default='0.1')
    parser.add_argument('--priD', type=float, default='1.0')
    parser.add_argument('--degree_limit', type=int, default='101')
    parser.add_argument('--perturb_method', type=str, default='RR'
                        ,choices=['RR', 'LAP', 'LAP-Group','regen'])
    parser.add_argument('--eps', type=float, default=1.0)
    parser.add_argument('--convert_x', help='whether to convert original node features to one-hot degree features',
                        type=bool, default=False)
    parser.add_argument('--datapath', type=str, default='XXX',
                        help='The input path of data.')

    args = parser.parse_args()
    algorithm=args.algorithm
    device=args.device
    dataset_name=args.dataset
    seed=args.seed
    ratio_of_train_set=args.ratio_of_train_set
    hidden=args.hidden
    dropout=args.dropout
    lr=args.lr
    weight_decay=args.weight_decay
    prune=args.prune
    priD=args.priD
    degree_limit=args.degree_limit
    attack=args.attacks
    np.random.seed(seed)
    torch.manual_seed(seed)
    eps=args.eps
    perturb_method=args.perturb_method

    dataset=load_data_att_defense(dataset_name,args.datapath,args.convert_x)

    results=[]
    model_list=[]
    regen_adj_list=[]
    dense_matrix_list=[]
    features_list=[]
    labels_list=[]

    # print(dataset[0])
    # exit()

    for samp in dataset:
        # print(samp.x,samp.y)
        print(samp.edge_index)

        # edge_index_with_self_loops, _ = add_self_loops(samp.edge_index)
        # exit()

        dense_matrix = torch_geometric.utils.to_dense_adj(samp.edge_index)[0]
        # dense_matrix = torch_geometric.utils.to_dense_adj(edge_index_with_self_loops)[0]
        # print(torch_geometric.utils.to_dense_adj(samp.edge_index))
        print(dense_matrix.size())

        # labels=dataset.x.to(device)
        labels=torch.argmax(samp.x, dim=1).to(device)

        node_attributes=np.ones((len(labels),3))
        features=torch.tensor(node_attributes, dtype=torch.float).to(device)

        # print(labels)
        # exit()

        idx_train,idx_val,idx_test=split_dataset(labels,ratio_of_train_set,seed)

        if algorithm=='regen':
            acc_test,num_priv_edges,num_rengen_edges,model,regen_adj=graph_regenerate_different(features, dense_matrix, labels, idx_train, idx_val, idx_test, hidden, dropout, lr,weight_decay, prune, degree_limit,device,priD)

        # pd.DataFrame([acc_test,num_priv_edges,num_rengen_edges]).to_csv(f"{File_Path_Csv}/acc.csv",index=False, header=False) \
        results.append([acc_test,num_priv_edges,num_rengen_edges])

        print(acc_test,num_priv_edges,num_rengen_edges,regen_adj,dense_matrix,torch.count_nonzero(dense_matrix),torch.count_nonzero(regen_adj))

        model_list.append(model.state_dict())
        regen_adj_list.append(regen_adj)
        dense_matrix_list.append(dense_matrix)
        features_list.append(features)
        labels_list.append(labels)

        # exit()

    File_Path_Csv = os.getcwd() + f"/result/{algorithm}/{dataset_name}/"
    if not os.path.exists(File_Path_Csv):
        os.makedirs(File_Path_Csv)

    name = ["acc_test","num_priv_edges","num_rengen_edges"]
    result = pd.DataFrame(columns=name, data=results)
    result.to_csv(
        File_Path_Csv + 'acc.csv')

    torch.save(model_list, f"{File_Path_Csv}/model_{prune}.pt")
    torch.save(regen_adj_list, f"{File_Path_Csv}/regen_edge_{prune}.pth")
    torch.save(dense_matrix_list, f"{File_Path_Csv}/priv_edge_{prune}.pth")
    torch.save(features_list, f"{File_Path_Csv}/features_{prune}.pth")
    torch.save(labels_list, f"{File_Path_Csv}/labels_{prune}.pth")

if __name__ == '__main__':
     main()