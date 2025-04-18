import os
import sys

import networkx as nx
import time
import pickle
import numpy as np
import random

import pandas as pd
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from AnonymousWalkKernel import GraphKernel
from AnonymousWalkKernel import AnonymousWalks

import multiprocessing as mp
from multiprocessing import Pool

import sklearn
from scipy.spatial import distance


def load_dataset_graph(data_,data_dir='data', file_name=None, need_set=False):
    file_name = data_ + '-train_feats_label_edge_list'
    file_path = os.path.join(data_dir, file_name)
    # feature_set = set()
    with open(file_path, 'rb') as f:
        feats_list,label_list,edge_list = pickle.load(f)

    # print(np.shape(np.array(edge_list))[0])
    print(np.shape(np.array(label_list))[0])
    print(np.shape(np.array(feats_list))[0])

    num_nodes_list=[]
    adj_list=[]
    g_list=[]

    for ii in range(np.shape(np.array(label_list))[0]):
        num_nodes_list.append(np.shape(feats_list[ii])[0])
        g=nx.Graph()
        g.add_nodes_from(list(range(np.shape(feats_list[ii])[0])))
        g.add_edges_from(list(edge_list[ii].T))
        adj=np.array(nx.adjacency_matrix(g).todense())
        adj_list.append(adj)
        g_list.append(g)


    return feats_list,adj_list,label_list,g_list


def read_graphs(dataset):
    gk = GraphKernel()
    # tudataset = TUDataset(datapath, dataset)
    for i, graph in enumerate(dataset):
        # print(graph)
        # exit()

        # g = to_networkx(graph, to_undirected=True)
        gk.read_graph_from_nx(graph)
    # print(len(gk.graphs), nx.info(gk.graphs[0]))
    return gk


def generate_awe(g, klens=(3, 4, 5)):
    aw = AnonymousWalks(g)

    long_embeddings = []
    for klen in klens:
        # print(f"Anonymous walks of length {klen}:")
        # start = time.time()
        embedding, meta = aw.embed(steps=klen, method='exact', keep_last=True, verbose=False)

        # if klen == 3:
        #     embedding, meta = aw.embed(steps=klen, method='exact', keep_last=True, verbose=False)
        # else:
        #     embedding, meta = aw.embed(steps=klen, method='sampling', keep_last=True, verbose=False, MC=100, delta=0.01, eps=0.1)

        # embedding, meta = aw.embed(steps=klen, method='sampling', keep_last=True, verbose=False, MC=100, delta=0.01, eps=0.1)
        # embedding, meta = aw.embed(steps=klen, method='exact', keep_last=True, verbose=False)
        # finish = time.time()

        # aws = meta['meta-paths']
        # print('Computed Embedding of {} dimension in {:.3f} sec.'.format(len(aws), finish - start))
        long_embeddings.extend(embedding)

    return long_embeddings


def generate_awe_v2(g, klens=(3, 4, 5)):
    aw = AnonymousWalks(g)

    long_embeddings = []
    for klen in klens:
        # print(f"Anonymous walks of length {klen}:")
        # start = time.time()
        embedding, meta = aw.embed(steps=klen, method='sampling', keep_last=True, verbose=False, MC=100, delta=0.01, eps=0.1)
        # finish = time.time()

        # aws = meta['meta-paths']
        # print('Computed Embedding of {} dimension in {:.3f} sec.'.format(len(aws), finish - start))
        long_embeddings.extend(embedding)

    return long_embeddings
def runner2(ds,graph_s_gen,ii,jj,kk,idx,Flag):
    datapath = './data'
    outpath = './outputs/AWEs_'
    # print('@@@', graph_s_gen)
    gk = read_graphs(graph_s_gen)


    klens = (3, 4, 5)

    embs_list=[]

    df = pd.DataFrame()
    print("Dataset: {}; #graphs: {}".format(ds, len(gk.graphs)))
    for i, g in enumerate(gk.graphs):
        # if (i+1)%10 == 0:
        print(f"  > {i+1}th graph")
        embs = generate_awe(g, klens)
        df[i] = embs
        embs_list.append(embs)

    # print('###',df)
    # exit()
    df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{ii}_{jj}_{kk}_{idx}_{klens[0]}-{klens[-1]}_sampling_{Flag}.csv'), index=False)
    print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{ii}_{jj}_{kk}_{idx}_{klens[0]}-{klens[-1]}_sampling_{Flag}.csv'))

    return embs_list

def runner(ds,graph_s_gen,ii,jj,kk,Flag):
    datapath = './data'
    outpath = './outputs/AWEs_'
    # print('@@@', graph_s_gen)
    gk = read_graphs(graph_s_gen)


    klens = (3, 4, 5)

    embs_list=[]

    df = pd.DataFrame()
    print("Dataset: {}; #graphs: {}".format(ds, len(gk.graphs)))
    for i, g in enumerate(gk.graphs):
        # if (i+1)%10 == 0:
        print(f"  > {i+1}th graph")
        embs = generate_awe(g, klens)
        df[i] = embs
        embs_list.append(embs)

    # print('###',df)
    # exit()
    df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{ii}_{jj}_{kk}_{klens[0]}-{klens[-1]}_sampling_{Flag}.csv'), index=False)
    print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{ii}_{jj}_{kk}_{klens[0]}-{klens[-1]}_sampling_{Flag}.csv'))

    return embs_list


def sim_metrics(emb1, emb2):
    # similarity_matrix1 = sklearn.metrics.pairwise.cosine_similarity(embed, embed2)
    #
    # similarity_matrix = sklearn.metrics.pairwise.euclidean_distances(embed, embed2)
    # similarity_matrix2 = np.exp(-similarity_matrix)
    sim1 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 0.0000000000000000000000000000001)

    sim2 = np.dot(emb1, emb2)

    sim3 = np.linalg.norm(np.array(emb1) - np.array(emb2))

    sim4 = np.exp(-sim3)

    sim5=distance.jensenshannon(emb1, emb2)

    return sim1,sim2,sim3,sim4,sim5



if __name__ == "__main__":
    num_clients=3
    datasets = ["mutag","imdb-binary", "imdb-multi","enzymes"]
    # datasets = ["mutag"]



    for data_ in datasets:
        if data_ == 'mutag':
            res = './mutag/edp-gnn_mutag__Feb-28-13-10-40_215397/'
            graph_s_gen_dir0_0_0 = res + "/62-28-0-0/"
            graph_s_gen_dir0_0_1 = res + "/62-28-0-0/"
            graph_s_gen_dir0_0_2 = res + "/62-26-0-0/"
            graph_s_gen_dir0_1_0 = res + "/62-28-0-1/"
            graph_s_gen_dir0_1_1 = res + "/62-28-0-1/"
            graph_s_gen_dir0_1_2 = res + "/62-26-0-1/"

            graph_s_gen_dirs0_0 = [graph_s_gen_dir0_0_0, graph_s_gen_dir0_0_1, graph_s_gen_dir0_0_2]
            graph_s_gen_dirs0_1 = [graph_s_gen_dir0_1_0, graph_s_gen_dir0_1_1, graph_s_gen_dir0_1_2]

            num_class = 2

            graph_s_gen_dirs_s = [[graph_s_gen_dirs0_0, graph_s_gen_dirs0_1]]


        elif data_ == 'imdb-binary':
            res = './imdb-binary/edp-gnn_imdb-binary__Feb-28-13-13-37_221586/'
            graph_s_gen_dir0_0_0 = res + "/266-49-0-0/"
            graph_s_gen_dir0_0_1 = res + "/266-59-0-0/"
            graph_s_gen_dir0_0_2 = res + "/266-55-0-0/"
            graph_s_gen_dir0_1_0 = res + "/266-49-0-1/"
            graph_s_gen_dir0_1_1 = res + "/266-59-0-1/"
            graph_s_gen_dir0_1_2 = res + "/266-55-0-1/"

            graph_s_gen_dirs0_0 = [graph_s_gen_dir0_0_0, graph_s_gen_dir0_0_1, graph_s_gen_dir0_0_2]
            graph_s_gen_dirs0_1 = [graph_s_gen_dir0_1_0, graph_s_gen_dir0_1_1, graph_s_gen_dir0_1_2]

            num_class = 2

            graph_s_gen_dirs_s = [[graph_s_gen_dirs0_0, graph_s_gen_dirs0_1]]

        data_dir = './'
        file_name = data_ + '-train_feats_label_edge_list'
        file_path = os.path.join(data_dir, file_name)
        # feature_set = set()
        with open(file_path, 'rb') as f:
            feats_list, label_list, edge_list = pickle.load(f)


        num_nodes_list = []
        adj_list = []

        for ii in range(np.shape(np.array(label_list))[0]):
            num_nodes_list.append(np.shape(feats_list[ii])[0])
            g = nx.Graph()
            g.add_nodes_from(list(range(np.shape(feats_list[ii])[0])))
            g.add_edges_from(list(edge_list[ii].T))
            # adj = np.array(nx.adjacency_matrix(g).todense())
            adj_list.append(g)

        graph_list_ft_ = feats_list
        graph_list_adj_ = adj_list
        graph_list_label_ = label_list

        step = int(len(graph_list_label_) / 3)

        den_orig = {}
        den_gen = {}
        dgr_orig = {}
        dgr_gen = {}
        # ress1 = {}
        # ress2 = {}
        ress1 = []
        ress2 = []

        for ii in range(num_clients):
            graph_list_adj = graph_list_adj_[ii * step:(ii + 1) * step]
            graph_list_ft = graph_list_ft_[ii * step:(ii + 1) * step]
            graph_list_label = graph_list_label_[ii * step:(ii + 1) * step]

            label_set = set(list(graph_list_label))
            class_idx = []

            train_graph_list_ft_list = []
            test_graph_list_ft_list = []
            train_graph_list_adj_list = []
            test_graph_list_adj_list = []
            for l in list(label_set):
                # print(l,np.where(np.array(label_list_)==l))
                idx = np.where(np.array(graph_list_label) == l)[0]
                class_idx.append(len(idx))
                # print(len(idx))

                train_graph_list_adj = np.array(graph_list_adj)[idx]
                test_graph_list_adj = train_graph_list_adj

                train_graph_list_ft = np.array(graph_list_ft)[idx]
                test_graph_list_ft = train_graph_list_ft

                train_graph_list_ft_list.append(train_graph_list_ft)
                train_graph_list_adj_list.append(train_graph_list_adj)

            test_graph_list_ft_list = train_graph_list_ft_list
            test_graph_list_adj_list = train_graph_list_adj_list

            for jj in range(num_class):
                train_graph_list_adj = train_graph_list_adj_list[jj]
                test_graph_list_adj = test_graph_list_adj_list[jj]
                train_graph_list_ft = train_graph_list_ft_list[jj]
                test_graph_list_ft = test_graph_list_ft_list[jj]
                # if ii==0 and jj==0:
                #     continue
                if ii!=0:
                    continue
                if jj!=0:
                    continue

                print(ii,jj)

                # embs_list_orig = runner(data_, test_graph_list_adj[0:30], ii, jj, ii, 'orig')
                klens = (3, 4, 5)
                ds=data_
                Flag = 'orig'

                embs_list_orig = pd.read_csv(
                    f"./outputs/AWEs_/AWEs_{ds}_{ii}_{jj}_{ii}_{klens[0]}-{klens[-1]}_sampling_{Flag}.csv",
                    index_col=None, header=0)

                for kk in range(num_clients):

                    # if os.path.exists('./res2_729-black/%s/sim2_%s_%s_%s' % (data_, str(ii), str(jj), str(kk))):
                    #    continue

                    # if ii==kk:
                    #     continue

                    # if ii==jj==kk==0:
                    #     continue

                    graph_dir = graph_s_gen_dirs_s[ii][jj][kk]
                    # print(graph_dir )

                    graph_s_gen_dir = f"{graph_dir}/sample_data/{str(ii)}{str(jj)}{str(kk)}/"
                    print(graph_s_gen_dir)
                    files = os.listdir(graph_s_gen_dir)
                    # print(files)
                    # division = 500
                    for file in files:
                        graph_s_gen_dir_f = os.path.join(graph_s_gen_dir, file)
                        f2 = open(graph_s_gen_dir_f, 'rb')
                        graph_s_gen = pickle.load(f2, encoding='latin1')

                    num_samples = min(100, len(test_graph_list_adj))
                    embs_list_gens={}

                    for idx in range(len(graph_s_gen)):
                        # if idx!=21:
                        #     continue
                        print(ii,jj,kk,idx)
                        # embs_list_gen=runner2(data_, graph_s_gen[idx], ii, jj,kk,idx, 'gen')
                        Flag='gen'

                        if not os.path.exists(f"./outputs/AWEs_/AWEs_{ds}_{ii}_{jj}_{kk}_{idx}_{klens[0]}-{klens[-1]}_sampling_{Flag}.csv"):
                            exit()

                        embs_list_gen = pd.read_csv(
                            f"./outputs/AWEs_/AWEs_{ds}_{ii}_{jj}_{kk}_{idx}_{klens[0]}-{klens[-1]}_sampling_{Flag}.csv",
                            index_col=None, header=0)

                        embs_list_gens[idx]=embs_list_gen

                    # print('****',embs_list_gens)
                    # print('****',embs_list_orig)
                    #
                    # exit()


                    for idx in range(len(graph_s_gen)):
                        # print(embs_list_orig,len(embs_list_orig))
                        # print(embs_list_orig[str(0)])
                        g_i1 = embs_list_orig[str(idx)]
                        # print(g_i1)
                        g_i2s = embs_list_gens[idx]
                        # if g_i1.number_of_edges() == g_i2.number_of_edges():
                        # print('ooooooo')
                        # print('###', g_i2s,np.shape(g_i2s))

                        for idx2 in range(np.shape(g_i2s)[1]):
                            g_i2=g_i2s[str(idx2)]
                            # print('###2',g_i2)
                            # exit()

                            # print('%%%',g_i1, g_i2)

                            res = sim_metrics( g_i1, g_i2)

                            # ress1[str(ii) + '_' + str(jj)+ '_' + str(kk)] = res

                            ress1.append([ii, jj, kk, res[0], res[1], res[2], res[3], res[4]])

                    # print(len(embs_list_gens))

                    for idx in range(len(embs_list_gens)):
                        g_i2_0 = embs_list_gens[idx]

                        for idx_1 in range(np.shape(g_i2_0)[1]-1):
                            g_i2_1 = g_i2_0[str(idx_1)]

                            for idx_2 in range(idx_1+1,np.shape(g_i2_0)[1]):

                                g_i2_2 = g_i2_0[str(idx_2)]

                                # print(g_i2_1, g_i2_2)

                                res = sim_metrics( g_i2_1, g_i2_2)
                                # ress2[str(ii) + '_' + str(jj)+ '_' + str(kk)] = res
                                ress2.append([ii, jj, kk, res[0], res[1], res[2], res[3], res[4]])

                    # ress2[str(i1) + '_' + str(i2)] = res2

                    # np.save('./res/match_%s_%s_%s_%s.npy' % (str(ii), str(jj), str(num), args.dataset_name),ress1)
                    # np.save('./res2/match_%s_%s_%s_%s.npy' % (str(ii), str(jj), str(num), args.dataset_name), ress2)

                    save_dir=f'./res2_729-black/{data_}/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    with open('./res2_729-black/%s/sim1_%s_%s_%s' % (data_, str(ii), str(jj), str(kk)),
                              'wb') as f:
                        pickle.dump(ress1, f)

                    with open('./res2_729-black/%s/sim2_%s_%s_%s' % (data_, str(ii), str(jj), str(kk)),
                              'wb') as f:
                        pickle.dump(ress2, f)


