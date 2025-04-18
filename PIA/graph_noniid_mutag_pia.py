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


def main_ratio_KStest_distDiff_nodeDegrees(ds_pair):
    ds1, ds2 = ds_pair
    print(ds_pair)
    dists_diff1 = dict_dist_diff_degrs[ds1]
    dists_diff2 = dict_dist_diff_degrs[ds2]
    # ks_pvalues = np.zeros((len(dists_diff1), len(dists_diff2)))
    count_noniid = 0
    for i in range(len(dists_diff1)):
        for j in range(len(dists_diff2)):
            # ks_pvalues[i, j] = stats.ks_2samp(dists_diff1[i], dists_diff2[j])[1]
            if stats.ks_2samp(dists_diff1[i], dists_diff2[j])[1] <= 0.01:
                count_noniid += 1
    # ratio = len(ks_pvalues[ks_pvalues <= 0.01]) * 1. / (len(dists_diff1) * len(dists_diff2))
    ratio = count_noniid * 1. / (len(dists_diff1) * len(dists_diff2))
    df = pd.DataFrame()
    df.loc[ds1, ds2] = ratio
    # df.to_csv(f"./outputs/remote/featureStats/tmps/tmp_ratio_noniid_nodeDegrees_{ds1}_{ds2}.csv")
    df.to_csv(f"./outputs/tmp_ratio_noniid_nodeDegrees_{ds1}_{ds2}.csv")


def aggregate_ratioFiles(ks, indir, suffix):
    # ks = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1",
    #       "ENZYMES", "DD", "PROTEINS",
    #       "IMDB-BINARY", "IMDB-MULTI", "COLLAB", "REDDIT-BINARY"]
    # ks = ["AIDS", "BZR", "COX2", "DHFR", "ENZYMES", "MUTAG", "NCI1", "PROTEINS", "PTC_MR"]
    df = pd.DataFrame(index=ks, columns=ks)
    for tmpfile in os.listdir(os.path.join(indir, 'tmps')):
        if tmpfile.startswith(f'tmp_ratio_noniid_{suffix}_'):
            df_tmp = pd.read_csv(os.path.join(indir, "tmps", tmpfile), header=0, index_col=0)
            df.loc[df_tmp.columns, df_tmp.index] = df_tmp.values
            df.loc[df_tmp.index, df_tmp.columns] = df_tmp.values
    pd.options.display.max_columns = None
    print(df)
    df.to_csv(os.path.join(indir, f'ratio_pvalues_KStest_{suffix}.csv'))



def _get_client_trainData_oneDS(tudataset, num_client, overlap, seed_dataSplit):
    # graphs = [(idx, x) for idx, x in enumerate(tudataset)]
    indices = list(range(len(tudataset)))

    client_trainIndices = []
    y = torch.cat([graph.y for graph in tudataset])
    indices_tv, _ = train_test_split(indices, test_size=0.1, stratify=y, shuffle=True, random_state=seed_dataSplit)
    # indices_tv, _ = split_data(indices, test=0.1, shuffle=True, seed=seed_dataSplit)
    indices_tv_chunks = _randChunk(indices_tv, num_client, overlap, seed=seed_dataSplit)
    for idx, chunks in enumerate(indices_tv_chunks):
        indices_train, _ = train_test_split(chunks, train_size=0.9, test_size=0.1, shuffle=True, random_state=seed_dataSplit)
        client_trainIndices.append(indices_train)

    return client_trainIndices #[[idx, ...], [...], ...]



def _get_avg_JSdist_awe_byClient(df_awe, client1, client2):
    jsDists = []
    for idx1 in client1:
        for idx2 in client2:
            jsDists.append(distance.jensenshannon(df_awe[str(idx1)], df_awe[str(idx2)]))
    return np.nanmean(jsDists), np.nanstd(jsDists)

def _get_avg_JSdist_awe_byClient_(df_awe_o, i, df_awe_g):
    jsDists = []
    for idx1 in range(len(df_awe_g.keys())):
        jsDists.append(distance.jensenshannon(df_awe_o[str(i)], df_awe_g[str(idx1)]))
    return np.nanmean(jsDists), np.nanstd(jsDists)

def _generate_distribution_nodeLabelSimilarity(tudataset):
    dict_sim = {}
    for idx, g in enumerate(tudataset):
        sims = []
        for idx_e in range(len(g.edge_index[0])):
            n1 = g.edge_index[0][idx_e]
            n2 = g.edge_index[1][idx_e]
            sim = 1 - distance.cosine(g.x[n1], g.x[n2])  # 0 or 1
            sims.append(sim)
        distribution_sim = []
        for v in sorted(set(sims)):
            distribution_sim.append(sims.count(v))
        dict_sim[idx] = distribution_sim
    return dict_sim


def _generate_distribution_nodeLabelSimilarity_(tudataset,list_tudatasets):
    # print(np.shape(list_tudatasets))
    # print('@@@',list_tudatasets[0][1])
    dict_sim = {}
    for idx, g in enumerate(tudataset):
        sims = []
        for idx_e in range(len(g.edge_index[0])):
            n1 = g.edge_index[0][idx_e]
            n2 = g.edge_index[1][idx_e]
            sim = 1 - distance.cosine(list_tudatasets[0][idx].x[n1], list_tudatasets[0][idx].x[n2])  # 0 or 1
            sims.append(sim)
        distribution_sim = []
        for v in sorted(set(sims)):
            distribution_sim.append(sims.count(v))
        dict_sim[idx] = distribution_sim
    return dict_sim

def _get_avg_JSdist_simDistribution_byClient(dict_sim, client1, client2):
    jsDist = []
    for idx1 in client1:
        for idx2 in client2:
            jsDist.append(distance.jensenshannon(dict_sim[idx1], dict_sim[idx2]))
    return np.nanmean(jsDist), np.nanstd(jsDist)


def _pca_analysis(client_names, df_awe, client_indices):
    df = pd.DataFrame(columns=client_names)
    for i in range(len(client_names)):
        df[client_names[i]] = list(df_awe[[str(x) for x in client_indices[i]]].mean(axis=1))
    # print(df)
    df = df.T
    x = df.values

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principal = pca.fit_transform(x)
    df_pca = pd.DataFrame(data=principal, index=client_names, columns=['pc1', 'pc2'])

    return df_pca

def _generate_graphAWE_proteins(ds, graphs, indices):
    outpath = './outputs/AWEs'
    gk = GraphKernel()
    for graph in graphs:
        g = to_networkx(graph, to_undirected=True)
        gk.read_graph_from_nx(g)

    klens = (3, 4, 5, 6, 7)
    df = pd.DataFrame()
    print("Dataset: {}; #graphs: {}".format(ds, len(gk.graphs)))
    for i in range(len(indices)):
        embs = generate_awe_v2(gk.graphs[i], klens)
        df[indices[i]] = embs
    # print(df)
    df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}.csv'), index=False)
    print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}.csv'))
    # df.to_csv(os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'), index=False)
    # print("Wrote to file:", os.path.join(outpath, f'AWEs_{ds}_{klens[0]}-{klens[-1]}_sampling.csv'))
    return df

def analyze_struct_feature_proteins():
    seed_dataSplit = 123
    datapath = "./data/"
    dataset = 'PROTEINS'
    num_client = 30
    overlap = False

    tudataset = TUDataset(f"{datapath}/TUDataset", dataset)
    client_trainIndices = _get_client_trainData_oneDS(tudataset, num_client, overlap, seed_dataSplit)

    client_names = [f'{i}-{dataset}' for i in range(num_client)]

    """ structral similarity """
    df_awe = pd.read_csv(f"./outputs/AWEs/AWEs_{dataset}_3-7.csv", index_col=None, header=0)

    jsDist = np.zeros((num_client, num_client))
    jsDist_std = np.zeros((num_client, num_client))
    for i in range(num_client):
        for j in range(i+1, num_client):
            diff, diff_std = _get_avg_JSdist_awe_byClient(df_awe, client_trainIndices[i], client_trainIndices[j])
            print((i, j), diff, diff_std)
            jsDist[i, j] = diff
            jsDist[j, i] = diff
            jsDist_std[i, j] = diff_std
            jsDist_std[j, i] = diff_std

    df = pd.DataFrame(jsDist, index=client_names, columns=client_names)
    # print(df)
    df.to_csv(f"./outputs//heteroAnalysis/{dataset}/jsDists_awes_{overlap}-{num_client}clients.csv", header=True, index=True)
    df = pd.DataFrame(jsDist_std, index=client_names, columns=client_names)
    df.to_csv(f"./outputs/heteroAnalysis/{dataset}/std_jsDists_awes_{overlap}-{num_client}clients.csv",
        header=True, index=True)

    """ feature similarity """
    dict_sim = _generate_distribution_nodeLabelSimilarity(tudataset)

    df_js = pd.DataFrame(0, index=client_names, columns=client_names)
    df_js_std = pd.DataFrame(0, index=client_names, columns=client_names)
    for i in range(num_client):
        for j in range(i + 1, num_client):
            diff, diff_std = _get_avg_JSdist_simDistribution_byClient(dict_sim, client_trainIndices[i], client_trainIndices[j])
            print((i, j), diff, diff_std)
            df_js.loc[client_names[i], client_names[j]] = diff
            df_js.loc[client_names[j], client_names[i]] = diff
            df_js_std.loc[client_names[i], client_names[j]] = diff_std
            df_js_std.loc[client_names[j], client_names[i]] = diff_std

    df_js.to_csv(f"./outputs/heteroAnalysis/{dataset}/jsDists_nodeLabelSim_{overlap}-{num_client}clients.csv",
                 header=True, index=True)
    df_js_std.to_csv(f"./outputs/heteroAnalysis/{dataset}/std_jsDists_nodeLabelSim_{overlap}-{num_client}clients.csv",
                     header=True, index=True)



def analyze_struct_feature_proteins_():
    num_clients = 3
    datasets = ["mutag", "enzymes", "imdb-binary", "imdb-multi"]
    # datasets = ["imdb-binary"]
    result_all = []
    for data_ in datasets:
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


        den_orig={}
        den_gen = {}
        dgr_orig={}
        dgr_gen = {}

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
                        # print(num)
                        sim=[]

                        den_orig[str(ii) + '-' + str(jj)+'-'+str(num)]=[]
                        den_gen[str(ii) + '-' + str(jj) + '-' + str(num)] = []
                        dgr_orig[str(ii) + '-' + str(jj) + '-' + str(num)] = []
                        dgr_gen[str(ii) + '-' + str(jj) + '-' + str(num)] = []
                        # df_awe_o = pd.read_csv(f"./outputs/AWEs_/AWEs_{data_}_{ii}_{jj}_3-5_sampling_{num}.csv",
                        #                        index_col=None, header=0)

                        g_orig=g_all[str(ii)+'-'+str(jj)]

                        # print(g_all_num_nodes[str(ii) + '-' + str(jj)])


                        idx_orig=np.array(np.where(np.array(g_orig_node_idx)==num)[0])
                        idx_gen=np.array(np.where(np.array(num_nodes)==num)[0])

                        g_idx_orig=np.array(g_orig)[idx_orig]
                        g_idx_gen=np.array(graph_s_gen)[idx_gen]

                        # print(np.shape(g_idx_gen))
                        # print(np.shape(g_idx_orig))
                        #
                        # print(g_idx_orig)



                        g_num_flag = 0
                        for g_i in g_idx_gen:
                            plt.figure()
                            # print(nx.adjacency_matrix(g_i))
                            dens=nx.number_of_edges(g_i)/((nx.number_of_nodes(g_i))*(nx.number_of_nodes(g_i)-1))
                            dgr=nx.number_of_edges(g_i)/nx.number_of_nodes(g_i)
                            den_gen[str(ii) + '-' + str(jj) + '-' + str(num)].append(dens)
                            dgr_gen[str(ii) + '-' + str(jj) + '-' + str(num)].append(dgr)


                        g_num_flag = 0
                        for g_i in g_idx_orig:
                            dens=nx.number_of_edges(g_i)/((nx.number_of_nodes(g_i))*(nx.number_of_nodes(g_i)-1))
                            dgr=nx.number_of_edges(g_i)/nx.number_of_nodes(g_i)
                            den_orig[str(ii) + '-' + str(jj) + '-' + str(num)].append(dens)
                            dgr_orig[str(ii) + '-' + str(jj) + '-' + str(num)].append(dgr)

                            # # print(g_i)
                            # plt.figure()
                            # nx.draw_networkx(g_i, with_labels=g_i.nodes,node_size=140, font_size=9)
                            # plt.savefig('./plot_%s/orig_%s_%s_%s_%s_%s_%s.png'%(data_,str(ii),str(jj),str(num),str(g_num_flag),str(dens),str(dgr)))
                            # g_num_flag+=1
                            # plt.show()
                        # print(g_num_flag)
                        # print(den_gen[str(ii) + '-' + str(jj) + '-' + str(num)])
                        # print(den_orig[str(ii) + '-' + str(jj) + '-' + str(num)])
                        # print(dgr_gen[str(ii) + '-' + str(jj) + '-' + str(num)])
                        # print(dgr_orig[str(ii) + '-' + str(jj) + '-' + str(num)])

                        print(ii,jj,num,np.mean(np.array(den_gen[str(ii) + '-' + str(jj) + '-' + str(num)])),np.mean(np.array(dgr_gen[str(ii) + '-' + str(jj) + '-' + str(num)])),np.mean(np.array(den_orig[str(ii) + '-' + str(jj) + '-' + str(num)])),np.mean(np.array(dgr_orig[str(ii) + '-' + str(jj) + '-' + str(num)])))
                #
                with open('./plot_%s/dens_gen.pkl' % (data_),
                          'wb') as f:
                    pkl.dump(den_gen, f)

                with open('./plot_%s/dens_orig.pkl' % (data_),
                          'wb') as f:
                    pkl.dump(den_orig, f)

                with open('./plot_%s/dgr_gen.pkl' % (data_),
                          'wb') as f:
                    pkl.dump(dgr_gen, f)

                with open('./plot_%s/dgr_orig.pkl' % (data_),
                          'wb') as f:
                    pkl.dump(dgr_orig, f)