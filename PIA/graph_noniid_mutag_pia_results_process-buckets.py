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
import itertools


def analyze_struct_feature_proteins_():
    num_clients = 3
    num_buck = 10
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
                # print(files)

                division = 500
                file_=[]
                for file in files:
                    if "_sample.pkl" in file:
                        file_.append(file)

                # print(file_)
                # exit()
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

                    den_gen_=[]
                    den_orig_=[]
                    dgr_gen_=[]
                    dgr_orig_=[]

                    for num in set(node_num_inter):
                        # print(den_gen)
                        if str(ii) + '-' + str(jj)+'-'+str(num) not in den_gen.keys():
                            continue
                        den_gen_.append(den_gen[str(ii) + '-' + str(jj)+'-'+str(num)])
                        den_orig_.append(den_orig[str(ii) + '-' + str(jj)+'-'+str(num)])

                        dgr_gen_.append(dgr_gen[str(ii) + '-' + str(jj) + '-' + str(num)])
                        dgr_orig_.append(dgr_orig[str(ii) + '-' + str(jj) + '-' + str(num)])

                    den_gen_ =list(itertools.chain.from_iterable(den_gen_))
                    den_orig_ = list(itertools.chain.from_iterable(den_orig_))
                    dgr_gen_ =list(itertools.chain.from_iterable(dgr_gen_))
                    dgr_orig_ =list(itertools.chain.from_iterable(dgr_orig_))

                    # den_min=min(min(den_gen_),min(den_orig_))
                    # dgr_min=min(min(dgr_gen_),min(dgr_orig_))
                    #
                    # den_max=max(max(den_gen_),max(den_orig_))
                    # dgr_max=max(max(dgr_gen_),max(dgr_orig_))

                    den_min=min(den_orig_)
                    dgr_min=min(dgr_orig_)

                    den_max=max(den_orig_)
                    dgr_max=max(dgr_orig_)


                    den_interv=(den_max-den_min)/num_buck
                    dgr_interv=(dgr_max-dgr_min)/num_buck

                    den_min_idx = []
                    dgr_min_idx = []
                    den_max_idx = []
                    dgr_max_idx = []

                    buck_dic_den_gen={}
                    buck_dic_dgr_gen={}
                    buck_dic_den_orig={}
                    buck_dic_dgr_orig={}

                    for idx in range(num_buck):
                        den_min_idx.append(den_min+den_interv*idx)
                        dgr_min_idx.append(dgr_min+dgr_interv*idx)
                        den_max_idx.append(den_min+den_interv*(idx+1))
                        dgr_max_idx.append(dgr_min+dgr_interv*(idx+1))
                        buck_dic_den_gen[idx]=[]
                        buck_dic_dgr_gen[idx]=[]
                        buck_dic_den_orig[idx]=[]
                        buck_dic_dgr_orig[idx]=[]

                    for den in den_gen_:
                        idx_=int((den-den_min)/den_interv)
                        if idx_==num_buck:
                            print('1',idx_)
                            idx_-=1
                        if idx_ > num_buck - 1:
                            idx_=num_buck - 1
                        if idx_ < 0:
                            idx_=0
                        buck_dic_den_gen[idx_].append(den)

                    for den in den_orig_:
                        idx_ = int((den - den_min) / den_interv)
                        if idx_==num_buck:
                            print(idx_)
                            idx_-=1
                        if idx_ > num_buck - 1:
                            idx_=num_buck - 1
                        if idx_ < 0:
                            idx_=0
                        buck_dic_den_orig[idx_].append(den)

                    for dgr in dgr_gen_:
                        idx_ = int((dgr - dgr_min) / dgr_interv)
                        if idx_==num_buck:
                            print(idx_)
                            idx_-=1
                        if idx_ > num_buck - 1:
                            idx_=num_buck - 1
                        if idx_ < 0:
                            idx_=0
                        buck_dic_dgr_gen[idx_].append(dgr)

                    for dgr in dgr_orig_:
                        idx_ = int((dgr - dgr_min) / dgr_interv)

                        if idx_==num_buck:
                            print(idx_)
                            idx_-=1
                        if idx_ > num_buck - 1:
                            idx_=num_buck - 1
                        if idx_ < 0:
                            idx_=0
                        buck_dic_dgr_orig[idx_].append(dgr)

                    num_buck_den_gen_list=[]
                    num_buck_den_orig_list=[]
                    num_buck_dgr_gen_list=[]
                    num_buck_dgr_orig_list=[]
                    for idx in range(num_buck):

                        num_buck_den_gen_list.append(len(buck_dic_den_gen[idx])/len(den_gen_))
                        num_buck_dgr_gen_list.append(len(buck_dic_dgr_gen[idx])/len(dgr_gen_))
                        num_buck_den_orig_list.append(len(buck_dic_den_orig[idx])/len(den_orig_))
                        num_buck_dgr_orig_list.append(len(buck_dic_dgr_orig[idx])/len(dgr_orig_))


                    print(ii,jj,data_)
                    print(num_buck_den_gen_list)
                    print(num_buck_den_orig_list)
                    print(num_buck_dgr_gen_list)
                    print(num_buck_dgr_orig_list)


                    ress.append([data_,f'client:{ii}',f'class:{jj}'])
                    ress.append(num_buck_den_gen_list)
                    ress.append(num_buck_den_orig_list)
                    ress.append(num_buck_dgr_gen_list)
                    ress.append(num_buck_dgr_orig_list)

                # name = ['data', 'client', 'class','num', 'den_gen', 'den_orig', 'dgr_gen', 'dgr_orig']
                result = pd.DataFrame(data=ress)
                result.to_csv(f'./outputs/heteroAnalysis/property-bucket-{data_}-{num_buck}.csv')




if __name__ == "__main__":
    """ get the distribution of difference between end nodes of edges base on node label / node degree """
    # # # main_distribution_difference_nodeLabelsAndDegrees()
    # # # main_KStest_distDiff_nodeLabels()
    # inpath = './outputs/featureStats/'
    # dict_dist_diff_nodeLabel = json.load(open(os.path.join(inpath, "distribution_diff_nodeLabels_edgewise.txt")))
    # # # print(dict_dist_diff_nodeLabel["MUTAG"])
    # datasets = list(dict_dist_diff_nodeLabel.keys())
    # # ds_pairs = []
    # # for i in range(len(datasets)):
    # #     for j in range(i, len(datasets)):
    # #         ds_pairs.append((datasets[i], datasets[j]))
    # # with Pool(6) as p:
    # #     p.map(main_ratio_KStest_distDiff_nodeLabels, ds_pairs)
    #
    # suffix = 'nodeLabels'
    # aggregate_ratioFiles(datasets, inpath, suffix)
    #
    # inpath = './outputs/featureStats'
    # dict_dist_diff_degrs = json.load(open(os.path.join(inpath, "distribution_diff_nodeDegrees_edgewise.txt")))
    # datasets = list(dict_dist_diff_degrs.keys())
    # ds_pairs = []
    # for i in range(len(datasets)):
    #     for j in range(i, len(datasets)):
    #         ds_pairs.append((datasets[i], datasets[j]))
    # with Pool(6) as p:
    #     p.map(main_ratio_KStest_distDiff_nodeDegrees, ds_pairs)


    # """ UPDATING: to use KL divergence (JS distance) """
    # analyze_struct_feature_mixtiny()
    #
    # # # PROTEINS
    analyze_struct_feature_proteins_()
    #
    # """ Compare the clustering results between GCFL & GCFL+ """
    # compare_struct_feature_mix()
    # calc_similarity_clusterwise_mix()


    # analyze_struct_feature_heterogeneity_()