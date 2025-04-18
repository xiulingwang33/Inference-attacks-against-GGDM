import os,sys
import pickle
import json
import random
sys.path.append("..")
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, degree
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.spatial import distance
from utils import split_data, use_node_attributes, convert_to_nodeDegreeFeatures
from multiprocessing import Pool
from setupGC import _randChunk
from AnonymousWalkKernel import GraphKernel
from generate_graphAWE import generate_awe_v2
import random
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import networkx as nx
import pickle as pkl



def analyze_struct_feature_proteins_():
    num_clients = 3
    datasets = ["mutag", "enzymes", "imdb-binary", "imdb-multi"]
    # datasets = ["imdb-binary"]
    result_all = []
    ress=[]
    for data_ in datasets:

        with open('./plot_%s/dens_gen.pkl' % (data_),
                  'rb') as f:
            den_gen=pkl.load(f)

        with open('./plot_%s/dens_orig.pkl' % (data_),
                  'rb') as f:
            den_orig=pkl.load(f)

        with open('./plot_%s/dgr_gen.pkl' % (data_),
                  'rb') as f:
            dgr_gen=pkl.load(f)

        with open('./plot_%s/dgr_orig.pkl' % (data_),
                  'rb') as f:
            dgr_orig=pkl.load(f)
        # args.dataset_name = data_
        # res = '/mnt/diskLv/luo/GDSS-master/'
        if data_ == 'mutag':

            num_class = 2


        elif data_ == 'imdb-binary':

            num_class = 2

        elif data_ == 'imdb-multi':

            num_class = 3

        elif data_ == 'enzymes':

            num_class = 6

        if data_ == 'mutag':
            graph_s_gen_dir0_0 = "./mutag/edp-gnn_mutag__Dec-26-00-02-28_914610/62-28-0-0/"
            graph_s_gen_dir0_1 = "./mutag/edp-gnn_mutag__Dec-26-00-02-28_914610/62-21-0-1/"

            graph_s_gen_dirs0 = [graph_s_gen_dir0_0]
            graph_s_gen_dirs1 = [graph_s_gen_dir0_1]

            graph_s_gen_dirs_s = [graph_s_gen_dirs0, graph_s_gen_dirs1]

            num_class = 2


        elif data_ == 'imdb-binary':
            graph_s_gen_dir0_0 = "./imdb-binary/edp-gnn_imdb-binary__Jan-06-12-56-33_2197873/266-49-0-0-3/"
            graph_s_gen_dir0_1 = "./imdb-binary/edp-gnn_imdb-binary__Jan-06-12-56-33_2197873/266-72-0-1-3/"
            graph_s_gen_dirs0 = [graph_s_gen_dir0_0]
            graph_s_gen_dirs1 = [graph_s_gen_dir0_1]

            graph_s_gen_dirs0 = [graph_s_gen_dir0_0]
            graph_s_gen_dirs1 = [graph_s_gen_dir0_1]

            graph_s_gen_dirs_s = [graph_s_gen_dirs0, graph_s_gen_dirs1]

            num_class = 2

        data_dir = './'
        file_name = data_ + '-train_feats_label_edge_list'
        file_path = os.path.join(data_dir, file_name)
        # feature_set = set()
        with open(file_path, 'rb') as f:
            feats_list_train0, label_list_train0, edge_list_train0 = pickle.load(f)

            for i in range(np.shape(label_list_train0)[0]):

                if np.max(edge_list_train0[i]) >= np.shape(feats_list_train0[i])[0]:
                    print('error1')
            feats_list_train = [ft_train for ft_train in feats_list_train0]
            label_list_train = [lb_train for lb_train in label_list_train0]
            edge_list_train = [eg_train for eg_train in edge_list_train0]

        # Train Dataset split to Clients
        startup = 0
        Client_list = []
        division = int(len(label_list_train) / num_clients)
        print(division)

        # exit()

        num_label=set(label_list_train)
        print(num_label)

        g_all={}


        g_all_num_nodes={}
        for i in range(num_clients):
            for j in range(num_class):
                g_all[str(i)+'-'+str(j)]=[]
                g_all_num_nodes[str(i)+'-'+str(j)] = []

            client_data_x = feats_list_train[startup:division + startup]
            # print(np.shape(client_data_x))
            # exit()
            # print(np.shape(x_gen[i]))
            client_data_y = label_list_train[startup:division + startup]
            client_data_edge = edge_list_train[startup:division + startup]

            for kk in range(len(client_data_y)):

                # print(client_data_edge[kk])

                g=nx.Graph()
                # print('***',np.shape(client_data_x[kk]))
                g.add_nodes_from(list(range(np.shape(client_data_x[kk])[0])))
                g.add_edges_from(list(client_data_edge[kk].T))

                kk_l=client_data_y[kk]

                g_all[str(i)+'-'+str(kk_l)].append(g)
                g_all_num_nodes[str(i)+'-'+str(kk_l)].append(np.shape(client_data_x[kk])[0])

        for ii in range(num_clients):
            for jj in range(num_class):

                graph_dir = graph_s_gen_dirs_s[jj][ii]

                graph_s_gen_dir = os.path.join(graph_dir, "sample/sample_data")
                files = os.listdir(graph_s_gen_dir)
                print(files)
                division = 500
                file_=[]
                for file in files:
                    if "_sample.pkl" in file:
                        file_.append(file)

                for file in file_:
                    graph_s_gen_dir_f = os.path.join(graph_s_gen_dir, file)
                    f2 = open(graph_s_gen_dir_f, 'rb')
                    graph_s_gen = pickle.load(f2, encoding='latin1')

                    num_nodes = []

                    for g in graph_s_gen:
                        num_nodes.append(g.number_of_nodes())
                    # print(num_nodes)
                    # print(set(num_nodes))
                    # exit()

                    g_orig_node_idx = g_all_num_nodes[str(ii) + '-' + str(jj)]

                    node_num_inter=set(g_orig_node_idx)&set(num_nodes)

                    for num in set(node_num_inter):

                        print(den_gen)

                        den_gen_=den_gen[str(ii) + '-' + str(jj)+'-'+str(num)]
                        den_orig_=den_orig[str(ii) + '-' + str(jj)+'-'+str(num)]

                        dgr_gen_ = dgr_gen[str(ii) + '-' + str(jj) + '-' + str(num)]
                        dgr_orig_ = dgr_orig[str(ii) + '-' + str(jj) + '-' + str(num)]

                        print(np.mean(np.array(den_gen_)),np.mean(np.array(den_orig_)))
                        print(np.mean(np.array(dgr_gen_ )),np.mean(np.array(dgr_orig_)))

                        ress.append([data_,ii,jj,num,np.mean(np.array(den_gen_)),np.mean(np.array(den_orig_)),np.mean(np.array(dgr_gen_ )),np.mean(np.array(dgr_orig_))])


            name = ['data', 'client', 'class','num', 'den_gen', 'den_orig', 'dgr_gen', 'dgr_orig']
            result = pd.DataFrame(columns=name, data=ress)
            result.to_csv('./outputs/heteroAnalysis/property.csv')




if __name__ == "__main__":
    """ get the distribution of difference between end nodes of edges base on node label / node degree """

    analyze_struct_feature_proteins_()
