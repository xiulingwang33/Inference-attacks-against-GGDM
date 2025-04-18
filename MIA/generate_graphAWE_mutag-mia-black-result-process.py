

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
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix



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
def runner2(ds,graph_s_gen,ii,jj,idx,Flag):
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
    df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{ii}_{jj}_{idx}_{klens[0]}-{klens[-1]}_sampling_{Flag}.csv'), index=False)
    print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{ii}_{jj}_{idx}_{klens[0]}-{klens[-1]}_sampling_{Flag}.csv'))

    return embs_list

def runner(ds,graph_s_gen,ii,jj,Flag):
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
    df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{ii}_{jj}_{klens[0]}-{klens[-1]}_sampling_{Flag}.csv'), index=False)
    print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{ii}_{jj}_{klens[0]}-{klens[-1]}_sampling_{Flag}.csv'))

    return embs_list

#
# def runner2(graphs):
#     ds = "COLLAB"
#     gk = GraphKernel()
#     for i, graph in enumerate(graphs):
#         g = to_networkx(graph, to_undirected=True)
#         gk.read_graph_from_nx(g)
#
#     outpath = './outputs/AWEs'
#     klens = (3, 4, 5)
#
#     df = pd.DataFrame()
#     print("Dataset: {}-{}; #graphs: {}".format(s, ds, len(gk.graphs)))
#     for i, g in enumerate(gk.graphs):
#         # if (i+1)%10 == 0:
#         print(f"  > {i + 1}th graph")
#         embs = generate_awe(g, klens)
#         df[i] = embs
#
#     print(df)
#     df.to_csv(os.path.join(outpath, f'{s}-AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'), index=False)
#     print("Wrote to file:", os.path.join(outpath, f'{s}-AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'))


def sim_metrics(emb1, emb2):
    # similarity_matrix1 = sklearn.metrics.pairwise.cosine_similarity(embed, embed2)
    #
    # similarity_matrix = sklearn.metrics.pairwise.euclidean_distances(embed, embed2)
    # similarity_matrix2 = np.exp(-similarity_matrix)
    sim1 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 0.0000000000000000000000000000001)

    sim2 = np.dot(emb1, emb2)

    sim3 = np.linalg.norm(np.array(emb1) - np.array(emb2))

    sim4 = np.exp(-sim3)

    return sim1,sim2,sim3,sim4



if __name__ == "__main__":
    num_clients=3
    datasets = ["mutag","imdb-binary", "imdb-multi","enzymes"]
    # datasets = ["mutag"]
    result_all1 = []
    result_all2 = []

    ft_num = 7

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

        print(np.shape(np.array(label_list))[0])
        print(np.shape(np.array(feats_list))[0])

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
        # ress1 = []
        # ress2 = []
        mem1={}
        non_mem1={}
        mem2={}
        non_mem2={}

        for ii in range(num_clients):
            for jj in range(num_class):
                for kk in range(num_clients):

                    if not os.path.exists('./res2_729-black/%s/sim1_%s_%s_%s' % (data_, str(ii), str(jj), str(kk))):
                        continue
                    if not os.path.exists('./res2_729-black/%s/sim2_%s_%s_%s' % (data_, str(ii), str(jj), str(kk))):
                        continue
                    with open('./res2_729-black/%s/sim1_%s_%s_%s' % (data_, str(ii), str(jj), str(kk)),
                              'rb') as f:
                        ress1=pickle.load(f)

                    with open('./res2_729-black/%s/sim2_%s_%s_%s' % (data_, str(ii), str(jj), str(kk)),
                              'rb') as f:
                        ress2=pickle.load(f)

                    ress1=np.array(ress1)
                    ress2=np.array(ress2)

                    ress1_ii_idx=np.where(ress1[:,0]==ii)[0]
                    ress1_jj_idx=np.where(ress1[:,1]==jj)[0]
                    ress1_kk_idx=np.where(ress1[:,2]==kk)[0]

                    # print(ress1_ii_idx,ress1_jj_idx,ress1_kk_idx)

                    ress1_idx=np.intersect1d(ress1_ii_idx, ress1_jj_idx)
                    ress1_idx=np.intersect1d(ress1_idx,ress1_kk_idx)

                    ress2_ii_idx=np.where(ress2[:,0]==ii)[0]
                    ress2_jj_idx=np.where(ress2[:,1]==jj)[0]
                    ress2_kk_idx=np.where(ress2[:,2]==kk)[0]

                    ress2_idx=np.intersect1d(ress2_ii_idx, ress2_jj_idx)
                    ress2_idx=np.intersect1d(ress2_idx,ress2_kk_idx)

                    ress1_=ress1[:,3:ft_num][ress1_idx]
                    ress2_=ress2[:,3:ft_num][ress2_idx]

                    if ii==kk:
                        mem1[f'{ii}-{jj}-{kk}']=ress1_
                        mem2[f'{ii}-{jj}-{kk}']=ress2_

                    else:
                        non_mem1[f'{ii}-{jj}-{kk}']=ress1_
                        non_mem2[f'{ii}-{jj}-{kk}']=ress2_

        print(mem1.keys())
        print(mem2.keys())
        print(non_mem1.keys())
        print(non_mem2.keys())
        for ii in range(num_clients):
            for jj in range(num_class):
                if f'{ii}-{jj}-{ii}' not in mem1.keys():
                    continue
                print('$$$$')
                mem1_ = mem1[f'{ii}-{jj}-{ii}']
                mem2_ = mem2[f'{ii}-{jj}-{ii}']
                # mem1_ = np.array(mem1_)[:,[0,0]]
                # mem2_ = np.array(mem2_)[:,[0,0]]


                for kk in range(num_clients):
                    if kk==ii:
                        continue

                    if f'{ii}-{jj}-{kk}' not in non_mem1.keys():
                        continue

                    print('****')
                    non_mem1_=non_mem1[f'{ii}-{jj}-{kk}']
                    non_mem2_=non_mem2[f'{ii}-{jj}-{kk}']

                    # non_mem1_ = np.array(non_mem1_)[:, [0, 0]]
                    # non_mem2_ = np.array(non_mem2_)[:, [0, 0]]


                    # print(mem1_)
                    # print(mem2_)
                    # print(non_mem1_)
                    # print(non_mem2_)
                    #
                    # exit()

                    label_mem_ = np.ones((np.shape(mem1_)[0], np.shape(mem1_)[1]))
                    label_non_mem_ = np.zeros((np.shape(non_mem1_)[0], np.shape(non_mem1_)[1]))

                    index_mem=int(np.shape(mem1_)[0]/30)
                    index_non_mem=int(np.shape(non_mem1_)[0]/30)


                    mem=mem1_
                    non_mem=non_mem1_
                    label_mem=label_mem_
                    label_non_mem=label_non_mem_

                    X_train_train=[]
                    X_train_test=[]
                    y_train_train=[]
                    y_train_test=[]
                    end=20
                    for idx in range(end):
                        mem_idx=mem[idx * index_mem:(idx + 1) * index_mem]
                        X_train_train.append([np.mean(mem_idx[:,0]),np.mean(mem_idx[:,1]),np.mean(mem_idx[:,2]),np.mean(mem_idx[:,3])])
                        y_train_train.append([1,1])
                    for idx in range(end,30):
                        mem_idx = mem[idx * index_mem:(idx + 1) * index_mem]
                        X_train_test.append([np.mean(mem_idx[:, 0]), np.mean(mem_idx[:, 1]), np.mean(mem_idx[:, 2]),
                                              np.mean(mem_idx[:, 3])])
                        y_train_test.append([1,1])

                    X_test_train=[]
                    X_test_test=[]
                    y_test_train=[]
                    y_test_test=[]
                    end=20
                    for idx in range(end):
                        non_mem_idx = non_mem[idx * index_non_mem:(idx + 1) * index_non_mem]
                        X_test_train.append([np.mean(non_mem_idx[:, 0]), np.mean(non_mem_idx[:, 1]), np.mean(non_mem_idx[:, 2]),
                                              np.mean(non_mem_idx[:, 3])])
                        y_test_train.append([0,0])
                    for idx in range(end, 30):
                        non_mem_idx = non_mem[idx * index_non_mem:(idx + 1) * index_non_mem]
                        X_test_test.append([np.mean(non_mem_idx[:, 0]), np.mean(non_mem_idx[:, 1]), np.mean(non_mem_idx[:, 2]),
                                             np.mean(non_mem_idx[:, 3])])

                        y_test_test.append([0,0])



                    # from sklearn.model_selection import train_test_split
                    #
                    # X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(mem,
                    #                                                                             label_mem,
                    #                                                                             test_size=0.3,
                    #                                                                             random_state=42)
                    #
                    # X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(non_mem, label_non_mem,
                    #                                                                         test_size=0.3,
                    #                                                                         random_state=42)

                    X_train = np.concatenate((X_train_train, X_test_train), axis=0)
                    X_test = np.concatenate((X_train_test, X_test_test), axis=0)
                    y_train = np.concatenate((y_train_train, y_test_train), axis=0)
                    y_test = np.concatenate((y_train_test, y_test_test), axis=0)

                    from sklearn import metrics
                    from sklearn.neural_network import MLPClassifier

                    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 8), random_state=1,
                                        max_iter=500)

                    mlp.fit(X_train, y_train[:, 1])

                    print("Training set score: %f" % mlp.score(X_train, y_train[:, 1]))
                    print("Test set score: %f" % mlp.score(X_test, y_test[:, 1]))


                    save_dir=f'./outputs/heteroAnalysis/{data_}/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    with open('./outputs/heteroAnalysis/{}/{}_{}_{}-mia1.pkl'.format(data_, ii, jj,kk), 'wb') as f:
                        pickle.dump(mlp, f)

                    y_score = mlp.predict(X_test)
                    proba = mlp.predict_proba(X_test)
                    # proba = np.amax(proba, axis=1)
                    proba = proba[:, 1]
                    print(metrics.f1_score(y_test[:, 1], y_score, average='micro'))
                    print(metrics.classification_report(y_test[:, 1], y_score, labels=range(3)))
                    report = metrics.classification_report(y_test[:, 1], y_score, labels=range(3), output_dict=True)

                    # out="{}{}-mlp_sim-report.csv".format(res_dir, F)
                    #
                    # df = pd.DataFrame(report).transpose()
                    # df.to_csv(out, index=True)

                    acc = accuracy_score(y_test[:, 1], y_score)
                    recall = recall_score(y_score, y_test[:, 1])
                    precision = precision_score(y_score, y_test[:, 1])
                    f1 = f1_score(y_score, y_test[:, 1])
                    auc = roc_auc_score(y_test[:, 1], proba)

                    print(ii, jj, kk,acc, recall, precision, f1, auc)

                    tsts = []
                    for i in range(len(y_score)):
                        tst = [y_score[i], proba[i], y_test[i][1]]
                        tsts.append(tst)
                    name = ['y_score', 'y_prob', 'y_test_grd']
                    result = pd.DataFrame(columns=name, data=tsts)
                    result.to_csv(
                        './outputs/heteroAnalysis/{}/{}_{}_{}-mia1.csv'.format(data_, ii, jj,kk))
                    result_all1.append([data_, ii, jj, kk,acc, recall, precision, f1, auc])




                    mem=mem2_
                    non_mem=non_mem2_
                    label_mem_ = np.ones((np.shape(mem2_)[0], np.shape(mem2_)[1]))
                    label_non_mem_ = np.zeros((np.shape(non_mem2_)[0], np.shape(non_mem2_)[1]))
                    label_mem=label_mem_
                    label_non_mem=label_non_mem_

                    X_train_train = []
                    X_train_test = []
                    y_train_train = []
                    y_train_test = []
                    end = 20
                    for idx in range(end):
                        mem_idx = mem[idx * index_mem:(idx + 1) * index_mem]
                        X_train_train.append([np.mean(mem_idx[:, 0]), np.mean(mem_idx[:, 1]), np.mean(mem_idx[:, 2]),
                                              np.mean(mem_idx[:, 3])])
                        y_train_train.append([1, 1])
                    for idx in range(end, 30):
                        mem_idx = mem[idx * index_mem:(idx + 1) * index_mem]
                        X_train_test.append([np.mean(mem_idx[:, 0]), np.mean(mem_idx[:, 1]), np.mean(mem_idx[:, 2]),
                                             np.mean(mem_idx[:, 3])])
                        y_train_test.append([1, 1])

                    X_test_train = []
                    X_test_test = []
                    y_test_train = []
                    y_test_test = []
                    end = 20
                    for idx in range(end):
                        non_mem_idx = non_mem[idx * index_non_mem:(idx + 1) * index_non_mem]
                        X_test_train.append(
                            [np.mean(non_mem_idx[:, 0]), np.mean(non_mem_idx[:, 1]), np.mean(non_mem_idx[:, 2]),
                             np.mean(non_mem_idx[:, 3])])
                        y_test_train.append([0, 0])
                    for idx in range(end, 30):
                        non_mem_idx = non_mem[idx * index_non_mem:(idx + 1) * index_non_mem]
                        X_test_test.append(
                            [np.mean(non_mem_idx[:, 0]), np.mean(non_mem_idx[:, 1]), np.mean(non_mem_idx[:, 2]),
                             np.mean(non_mem_idx[:, 3])])

                        y_test_test.append([0, 0])

                    # from sklearn.model_selection import train_test_split
                    #
                    # X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(mem,
                    #                                                                             label_mem,
                    #                                                                             test_size=0.3,
                    #                                                                             random_state=42)
                    #
                    # X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(non_mem, label_non_mem,
                    #                                                                         test_size=0.3,
                    #                                                                         random_state=42)

                    X_train = np.concatenate((X_train_train, X_test_train), axis=0)
                    X_test = np.concatenate((X_train_test, X_test_test), axis=0)
                    y_train = np.concatenate((y_train_train, y_test_train), axis=0)
                    y_test = np.concatenate((y_train_test, y_test_test), axis=0)

                    from sklearn import metrics
                    from sklearn.neural_network import MLPClassifier

                    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16, 8), random_state=1,
                                        max_iter=500)

                    mlp.fit(X_train, y_train[:, 1])

                    print("Training set score: %f" % mlp.score(X_train, y_train[:, 1]))
                    print("Test set score: %f" % mlp.score(X_test, y_test[:, 1]))

                    with open('./outputs/heteroAnalysis/{}/{}_{}_{}-mia2.pkl'.format(data_, ii, jj,kk), 'wb') as f:
                        pickle.dump(mlp, f)

                    y_score = mlp.predict(X_test)
                    proba = mlp.predict_proba(X_test)
                    # proba = np.amax(proba, axis=1)
                    proba = proba[:, 1]
                    print(metrics.f1_score(y_test[:, 1], y_score, average='micro'))
                    print(metrics.classification_report(y_test[:, 1], y_score, labels=range(3)))
                    report = metrics.classification_report(y_test[:, 1], y_score, labels=range(3), output_dict=True)

                    # out="{}{}-mlp_sim-report.csv".format(res_dir, F)
                    #
                    # df = pd.DataFrame(report).transpose()
                    # df.to_csv(out, index=True)

                    acc = accuracy_score(y_test[:, 1], y_score)
                    recall = recall_score(y_score, y_test[:, 1])
                    precision = precision_score(y_score, y_test[:, 1])
                    f1 = f1_score(y_score, y_test[:, 1])
                    auc = roc_auc_score(y_test[:, 1], proba)

                    print(ii, jj, kk,acc, recall, precision, f1, auc)

                    tsts = []
                    for i in range(len(y_score)):
                        tst = [y_score[i], proba[i], y_test[i][1]]
                        tsts.append(tst)
                    name = ['y_score', 'y_prob', 'y_test_grd']
                    result = pd.DataFrame(columns=name, data=tsts)
                    result.to_csv(
                        './outputs/heteroAnalysis/{}/{}_{}_{}-mia2.csv'.format(data_, ii, jj,kk))
                    result_all2.append([data_, ii, jj, kk,acc, recall, precision, f1, auc])


            name = ['data', 'client', 'class', 'client','acc', 'recall', 'precision', 'f1', 'auc']
            result = pd.DataFrame(columns=name, data=result_all2)
            result.to_csv('./outputs/heteroAnalysis/black-mia2-{}.csv'.format(data_))


            name = ['data', 'client', 'class', 'client','acc', 'recall', 'precision', 'f1', 'auc']
            result = pd.DataFrame(columns=name, data=result_all1)
            result.to_csv('./outputs/heteroAnalysis/black-mia1-{}.csv'.format(data_))


