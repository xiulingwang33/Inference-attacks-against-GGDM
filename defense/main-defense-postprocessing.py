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
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data


from graph_reconstruction.graph_regenerate import graph_regenerate_different
from graph_sparsification.graph_sparsification import graph_sparsification
from utils.utils import split_dataset

from collections import defaultdict
from torch_geometric.data import Data
import pickle

# CUDA_VISIBLE_DEVICES=0 python main-defense-postprocessing.py --dataset enzymes --prune 0.1
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
    parser.add_argument('--datapath', type=str, default='/mnt/diskLv/luo/FedStar-main/lib/Data/',
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

    # dataset=load_data_att_defense(dataset_name,args.datapath,args.convert_x)
    # print(dataset[1],dataset[1].edge_index,dataset[1].x)

    data_ =dataset_name.lower()

    if data_ == 'mutag':
        res = 'XXX'
        graph_s_gen_dir0 = res + "XXX"
        graph_s_gen_dir1 = res + "XXX"

        graph_s_gen_dirs = [graph_s_gen_dir0,graph_s_gen_dir1]
        num_class = 2


    elif data_ == 'imdb-binary':
        res = 'XXX'
        graph_s_gen_dir0 = res + "XXX"
        graph_s_gen_dir1 = res + "XXX"

        graph_s_gen_dirs = [graph_s_gen_dir0, graph_s_gen_dir1]
        num_class = 2

    elif data_ == 'imdb-multi':
        res = 'XXX'
        graph_s_gen_dir0 = res + "XXX"
        graph_s_gen_dir1 = res + "XXX"
        graph_s_gen_dir2 = res + "XXX"

        graph_s_gen_dirs = [graph_s_gen_dir0, graph_s_gen_dir1, graph_s_gen_dir2]
        num_class = 3

    elif data_ == 'enzymes':
        res = 'XXX'
        graph_s_gen_dir0 = res + "XXX"
        graph_s_gen_dir1 = res + "XXX"
        graph_s_gen_dir2 = res + "XXX"
        graph_s_gen_dir3 = res + "XXX"
        graph_s_gen_dir4 = res + "XXX"
        graph_s_gen_dir5 = res + "XXX"
        graph_s_gen_dirs = [graph_s_gen_dir0, graph_s_gen_dir1, graph_s_gen_dir2,graph_s_gen_dir3, graph_s_gen_dir4, graph_s_gen_dir5]

        num_class = 6

    elif data_ == 'proteins':
        res = 'XXX'
        graph_s_gen_dir0 = res + "XXX"
        graph_s_gen_dir1 = res + "XXX"
        graph_s_gen_dirs = [graph_s_gen_dir0, graph_s_gen_dir1]
        num_class = 2


    for ii in range(num_class):


        results = []
        model_list = []
        regen_adj_list = []
        dense_matrix_list = []
        features_list = []
        labels_list = []


        graph_dir = graph_s_gen_dirs[ii]
        # print(graph_dir )

        graph_s_gen_dir = f"{graph_dir}/sample_data/{str(ii)}/"
        print(graph_s_gen_dir)
        files = os.listdir(graph_s_gen_dir)
        # print(files)
        # division = 500

        for file in files:
            graph_s_gen_dir_f = os.path.join(graph_s_gen_dir, file)
            f2 = open(graph_s_gen_dir_f, 'rb')
            graph_s_gen = pickle.load(f2, encoding='latin1')
            print(len(graph_s_gen))

            for g_s in graph_s_gen:
                # print(g_)
                result = []
                model_lis = []
                regen_adj_lis = []
                dense_matrix_lis = []
                features_lis = []
                labels_lis = []
                for g_ in g_s:
                    degrees = dict(g_.degree())
                    # degree_array = np.array([degrees[node] for node in g_.nodes()])
                    #
                    # print(g_.number_of_nodes())

                    # print(g_.nodes())
                    # exit()

                    labels =[]

                    edges=list(g_.edges())
                    edge_index = np.array(edges).T

                    num_nodes=max(max(max(edge_index[0]),max(edge_index[1]))+1,g_.number_of_nodes())
                    for nd in range(num_nodes):
                        if nd in g_.nodes():
                            labels.append(g_.degree()[nd])
                        else:
                            labels.append(0)

                    # print(num_nodes)


                    node_attributes = np.ones((len(labels), 3))

                    # data = Data(x=node_attributes, edge_index=edge_index)
                    edge_index=torch.tensor(edge_index, dtype=torch.long).to(device)

                    # self_loops = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
                    # edge_index = torch.cat([edge_index, self_loops.to(device)], dim=1)

                    dense_matrix = torch_geometric.utils.to_dense_adj(edge_index)[0]
                    print(dense_matrix)

                    labels = torch.tensor(labels, dtype=torch.long).to(device)

                    features = torch.tensor(node_attributes, dtype=torch.float).to(device)

                    idx_train, idx_val, idx_test = split_dataset(labels, ratio_of_train_set, seed)

                    if algorithm == 'regen':
                        acc_test, num_priv_edges, num_rengen_edges, model, regen_adj = graph_regenerate_different(
                            features, dense_matrix, labels, idx_train, idx_val, idx_test, hidden, dropout, lr,
                            weight_decay, prune, degree_limit, device, priD)

                    # pd.DataFrame([acc_test,num_priv_edges,num_rengen_edges]).to_csv(f"{File_Path_Csv}/acc.csv",index=False, header=False) \
                    results.append([acc_test, num_priv_edges, num_rengen_edges])

                    print(acc_test, num_priv_edges, num_rengen_edges, regen_adj, dense_matrix,
                          torch.count_nonzero(dense_matrix), torch.count_nonzero(regen_adj))

                    model_lis.append(model.state_dict())
                    regen_adj_lis.append(regen_adj)
                    dense_matrix_lis.append(dense_matrix)
                    features_lis.append(features)
                    labels_lis.append(labels)

                model_list.append(model_lis)
                regen_adj_list.append(regen_adj_lis)
                dense_matrix_list.append(dense_matrix_lis)
                features_list.append(features_lis)
                labels_list.append(labels_lis)

                # exit()

                File_Path_Csv = os.getcwd() + f"/postprocessing/{algorithm}/{dataset_name}/"
                if not os.path.exists(File_Path_Csv):
                    os.makedirs(File_Path_Csv)


                torch.save(model_list, f"{File_Path_Csv}/model_{ii}_{prune}.pt")
                torch.save(regen_adj_list, f"{File_Path_Csv}/regen_edge_{ii}_{prune}.pth")
                torch.save(dense_matrix_list, f"{File_Path_Csv}/priv_edge_{ii}_{prune}.pth")
                torch.save(features_list, f"{File_Path_Csv}/features_{ii}_{prune}.pth")
                torch.save(labels_list, f"{File_Path_Csv}/labels_{ii}_{prune}.pth")


if __name__ == '__main__':
     main()