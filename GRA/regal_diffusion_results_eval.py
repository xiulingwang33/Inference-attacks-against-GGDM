import copy

import numpy as np
import argparse
import networkx as nx
import time
import os
import sys
try: import cPickle as pickle 
except ImportError:
	import pickle
from scipy.sparse import csr_matrix

import xnetmf
from config import *
from alignments import *
import scipy.sparse as sp
from match_diffusion import *
import itertools
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Create a list of unique combinations (e.g. (alexa, actual)).

def parse_args():
	parser = argparse.ArgumentParser(description="Run REGAL.")

	parser.add_argument('--input', nargs='?', default='data/arenas_combined_edges.txt', help="Edgelist of combined input graph")

	parser.add_argument('--output', nargs='?', default='emb/arenas990-1.emb',
	                    help='Embeddings path')

	parser.add_argument('--attributes', nargs='?', default=None,
	                    help='File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate.  Default is 5 synthetic.')

	parser.add_argument('--attrvals', type=int, default=2,
	                    help='Number of attribute values. Only used if synthetic attributes are generated')


	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--k', type=int, default=10,
	                    help='Controls of landmarks to sample. Default is 10.')

	parser.add_argument('--untillayer', type=int, default=2,
                    	help='Calculation until the layer for xNetMF.')
	parser.add_argument('--alpha', type=float, default = 0.01, help = "Discount factor for further layers")
	parser.add_argument('--gammastruc', type=float, default = 1, help = "Weight on structural similarity")
	parser.add_argument('--gammaattr', type=float, default = 1, help = "Weight on attribute similarity")
	parser.add_argument('--numtop', type=int, default=10,help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
	parser.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")
	return parser.parse_args()

def main(args,g_i1,g_i2):
	# dataset_name = args.output.split("/")
	# if len(dataset_name) == 1:
	# 	dataset_name = dataset_name[-1].split(".")[0]
	# else:
	# 	dataset_name = dataset_name[-2]

	# #Get true alignments
	# true_alignments_fname = args.input.split("_")[0] + "_edges-mapping-permutation.txt" #can be changed if desired
	# print("true alignments file: ", true_alignments_fname)
	true_alignments = None
	# if os.path.exists(true_alignments_fname):
	# 	with open(true_alignments_fname, "rb") as true_alignments_file:
	# 		try:
	# 			true_alignments = pickle.load(true_alignments_file)
	# 		except:
	# 			true_alignments = pickle.load(true_alignments_file, encoding = "latin1")

	#Load in attributes if desired (assumes they are numpy array)
	if args.attributes is not None:
		args.attributes = np.load(args.attributes) #load vector of attributes in from file
		# print(args.attributes.shape)

	#Learn embeddings and save to output
	print("learning representations...")
	before_rep = time.time()
	embed = learn_representations(args,g_i1,g_i2)
	# print(np.shape(embed))
	# exit()
	after_rep = time.time()
	print("Learned representations in %f seconds" % (after_rep - before_rep))

	emb1, emb2 = get_embeddings(embed)
	before_align = time.time()
	if args.numtop == 0:
		args.numtop = None
	alignment_matrix = get_embedding_similarities(emb1, emb2, num_top = None)#args.numtop)

	print('***',np.shape(alignment_matrix))
	# print(alignment_matrix)

	if not sp.issparse(alignment_matrix):
		sorted_indices = np.argsort(alignment_matrix,axis=1)
		# print(alignment_matrix)
		# print(np.shape(alignment_matrix))

		# print('@@@',np.sort(alignment_matrix,axis=1))
		# print(sorted_indices[:,-1])

	# alignment_matrix_sorted=np.sort(alignment_matrix, axis=1)
	# match_diffusion(alignment_matrix_sorted,ii,jj,num,i1,i2,args.dataset_name)

	res1=match_diffusion(alignment_matrix)
	# res2=match_diffusion2(alignment_matrix)
	# exit()

	#Report scoring and timing
	after_align = time.time()
	total_time = after_align - before_align
	print("Align time: "), total_time
	return res1
	# if true_alignments is not None:
	# # if true_alignments is None:
	#
	# 	topk_scores = [1]#,5,10,20,50]
	# 	for k in topk_scores:
	# 		score, correct_nodes = score_alignment_matrix_diffusion(alignment_matrix, topk = k, true_alignments = true_alignments)
	# 		# score, correct_nodes = score_alignment_matrix_diffusion(alignment_matrix, true_alignments = true_alignments)
	#
	# 		print("score top%d: %f" % (k, score))

#Should take in a file with the input graph as edgelist (args.input)
#Should save representations to args.output
def learn_representations(args,g_i1,g_i2):
	# nx_graph = nx.read_edgelist(args.input, nodetype = int, comments="%")
	nx_graph1 =g_i1
	nx_graph2 = g_i2
	nx_graph= nx.disjoint_union(nx_graph1, nx_graph2)
	print("read in graph")
	adj = nx.adjacency_matrix(nx_graph, nodelist = range(nx_graph.number_of_nodes()) )
	print("got adj matrix")
	
	graph = Graph(adj, node_attributes = args.attributes)
	max_layer = args.untillayer
	if args.untillayer == 0:
		max_layer = None
	alpha = args.alpha
	num_buckets = args.buckets #BASE OF LOG FOR LOG SCALE
	if num_buckets == 1:
		num_buckets = None
	rep_method = RepMethod(max_layer = max_layer, 
							alpha = alpha, 
							k = args.k, 
							num_buckets = num_buckets, 
							normalize = True, 
							gammastruc = args.gammastruc, 
							gammaattr = args.gammaattr)
	if max_layer is None:
		max_layer = 1000
	print("Learning representations with max layer %d and alpha = %f" % (max_layer, alpha))
	representations = xnetmf.get_representations(graph, rep_method)
	np.save(args.output, representations)
	return representations		

if __name__ == "__main__":
	args = parse_args()

	num_clients = 3
	datasets = ["mutag", "enzymes", "imdb-binary", "imdb-multi"]
	# datasets = ["imdb-multi"]

	result_all = []
	for data_ in datasets:
		args.dataset_name =data_
		for data_ in datasets:
			args.dataset_name = data_
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

		num_label = set(label_list_train)
		print(num_label)

		g_all = {}

		g_all_num_nodes = {}
		for i in range(num_clients):
			for j in range(num_class):
				g_all[str(i) + '-' + str(j)] = []
				g_all_num_nodes[str(i) + '-' + str(j)] = []

			client_data_x = feats_list_train[startup:division + startup]
			# print(np.shape(client_data_x))
			# exit()
			# print(np.shape(x_gen[i]))
			client_data_y = label_list_train[startup:division + startup]
			client_data_edge = edge_list_train[startup:division + startup]

			for kk in range(len(client_data_y)):
				# print(client_data_edge[kk])

				g = nx.Graph()
				# print('***',np.shape(client_data_x[kk]))
				g.add_nodes_from(list(range(np.shape(client_data_x[kk])[0])))
				g.add_edges_from(list(client_data_edge[kk].T))

				kk_l = client_data_y[kk]

				g_all[str(i) + '-' + str(kk_l)].append(g)
				g_all_num_nodes[str(i) + '-' + str(kk_l)].append(np.shape(client_data_x[kk])[0])

		den_orig = {}
		den_gen = {}
		dgr_orig = {}
		dgr_gen = {}
		ress_eval = {}
		for ii in range(num_clients):
			for jj in range(num_class):

				graph_dir = graph_s_gen_dirs_s[jj][ii]

				graph_s_gen_dir = os.path.join(graph_dir, "sample/sample_data")
				files = os.listdir(graph_s_gen_dir)
				print(files)
				division = 500
				for file in files:
					if not "_sample.pkl" in file:
						continue
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

					node_num_inter = set(g_orig_node_idx) & set(num_nodes)

					g_orig = g_all[str(ii) + '-' + str(jj)]

					print(set(node_num_inter))

					for num in set(node_num_inter):
						print(num)

						eval_g = []

						path = './%s/pred_g_%s_%s_%s_%s' % (
							args.dataset_name, str(ii), str(jj), str(num), args.dataset_name)
						if not os.path.exists(path):
							continue

						ressfile = open('./%s/pred_g_%s_%s_%s_%s' % (
						args.dataset_name, str(ii), str(jj), str(num), args.dataset_name), 'rb')
						ress = pickle.load(ressfile)
						# print(ress)
						# for res in ress:
						# 	print(np.shape(res))
						# exit()


						g_idx = np.array(np.where(np.array(g_all_num_nodes[str(ii) + '-' + str(jj)]) == num))[0]

						for i2 in g_idx:
							g=g_all[str(ii) + '-' + str(jj)][i2]
							g_i2 =g
							m_eye=list(np.array(np.where(np.eye(num)!=0)).T)
							g_i2.add_edges_from(m_eye)
							for i1 in range(len(ress)):
								res = ress[i1]
								# print('gg0',np.shape(res))
								res = res + np.eye(num)
								g_i1 = nx.Graph()
								# print('***',np.shape(client_data_x[kk]))
								g_i1.add_nodes_from(list(range(np.shape(res)[0])))
								g_i1.add_edges_from(list(np.array(np.where(res != 0)).T))

								# exit()
								res0_eval = main(args, g_i1, g_i2)
								ress_eval[str(ii) + '_' +str(jj) + '_' +str(i1) + '_' + str(i2)] = res0_eval

								res_eval = copy.deepcopy(res0_eval)
								res_eval = np.array(res_eval)

								if (np.shape(res_eval)[0] != num - 1) and (np.shape(res_eval)[0] != num) :
									print(np.shape(res_eval)[0],res_eval,num)
									print('^^^')
									continue


								if np.shape(res_eval)[0]==num-1:
									print('***')
									n0_i1=list(set(res_eval[:,0]).difference(set(res_eval[:,1])))
									n0_i2=list(set(res_eval[:,1]).difference(set(res_eval[:,0])))

									# n0_i1 = list(set(res[:, 0]).difference(set(np.array(list(range(num)),np.float64))))
									# n0_i2 = list(set(res[:, 1]).difference(set(np.array(list(range(num)),np.float64))))

									# n0_i1 = list(set(np.array(res[:, 0],np.int64)).difference(set(list(range(num)))))
									# n0_i2 = list(set(np.array(res[:, 1],np.int64)).difference(set(list(range(num)))))

									if n0_i1==n0_i2==[]:
										print('###')
										n0=(0+num-1)*num/2-np.sum(res_eval[:, 0])
										res_eval.append([n0, n0, 0, res_eval[-1][-1]])


									# print(res)
									# print(n0_i1,n0_i2)
									else:

										res_eval.append([n0_i1[0],n0_i2[0],0,res_eval[-1][-1]])

									# print(np.shape(res0))

									res=np.array(res_eval)

								g1_edges=nx.edges(g_i1)
								g2_edges = nx.edges(g_i2)

								g2_adj=nx.adjacency_matrix(g_i2).todense()
								# print('kkk',g1_edges,nx.nodes(g_i1))
								# print(np.shape(g2_adj))

								if max(nx.nodes(g_i1))>num-1:
									print('%%%%%%%%%%5')
									continue

								g1_edges_map=[]
								g1_map_adj = np.zeros((num,num))
								# print(g1_map_adj)
								for n1, n2 in g1_edges:
									# print(n1, n2)
									# print(res)
									n1_idx=np.where(res[:,0]==n1)[0]
									n2_idx = np.where(res[:, 0] == n2)[0]
									# print(n1_idx,n2_idx)
									if n1_idx.size>0 and n2_idx.size>0:
										print('***!!!')
										g1_edges_map.append([int(res[n1_idx[0]][1]),int(res[n2_idx[0]][1])])
										g1_edges_map.append([int(res[n2_idx[0]][1]), int(res[n1_idx[0]][1])])


										g1_map_adj[int(res[n1][1])][int(res[n2][1])]=1
										g1_map_adj[int(res[n2][1])][int(res[n1][1])]=1


								g1_map_adj_=np.array(list(itertools.chain.from_iterable(list(g1_map_adj))))
								g2_adj_=np.array(list(itertools.chain.from_iterable(list(g2_adj))))

								acc = accuracy_score(g1_map_adj_, g2_adj_)
								recall = recall_score(g2_adj_, g1_map_adj_)
								precision = precision_score(g2_adj_, g1_map_adj_)
								f1 = f1_score(g2_adj_, g1_map_adj_)

								eval_g.append([acc,recall,precision,f1,i2,i1])

								# print('eval',eval_g)

								# exit()

								result_all.append([ii, jj, num, acc, recall, precision, f1, i2, i1])

							name = ["clients", "class", "num", "acc", "recall", "precision", "f1", "i2", "i1"]
							result = pd.DataFrame(columns=name, data=result_all)
							result.to_csv(
								'./%s/eval_%s_all-ratio.csv' % (
									args.dataset_name, args.dataset_name))



							# with open('./res1_729/%s/eval_%s_%s_%s_%s-ratio' % (args.dataset_name,str(ii), str(jj), str(num), args.dataset_name), 'wb') as f:
							# 	pickle.dump(eval_g, f)



							# exit()


