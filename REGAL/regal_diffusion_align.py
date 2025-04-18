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
		print(args.attributes.shape)

	#Learn embeddings and save to output
	print("learning representations...")
	before_rep = time.time()
	embed = learn_representations(args,g_i1,g_i2)
	# print(embed)
	# exit()
	after_rep = time.time()
	print("Learned representations in %f seconds" % (after_rep - before_rep))

	emb1, emb2 = get_embeddings(embed)
	before_align = time.time()
	if args.numtop == 0:
		args.numtop = None
	alignment_matrix = get_embedding_similarities(emb1, emb2, num_top = None)#args.numtop)

	# print(alignment_matrix)
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
		ress = {}
		for ii in range(num_clients):
			for jj in range(num_class):

				graph_dir = graph_s_gen_dirs_s[jj][ii]

				graph_s_gen_dir = os.path.join(graph_dir, "sample/sample_data")
				files = os.listdir(graph_s_gen_dir)
				print(files)
				division = 500

				file_ = []
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

					node_num_inter = set(g_orig_node_idx) & set(num_nodes)

					g_orig = g_all[str(ii) + '-' + str(jj)]

					print(set(node_num_inter))

					for num in set(node_num_inter):
						#
						print('num',num)
						# exit()

						pred_g=[]

						idx_orig = np.array(np.where(np.array(g_orig_node_idx) == num)[0])
						idx_gen = np.array(np.where(np.array(num_nodes) == num)[0])

						g_idx_orig = np.array(g_orig)[idx_orig]
						g_idx_gen = np.array(graph_s_gen)[idx_gen]

						for g_ii in g_idx_orig:
							print('_____',nx.density(g_ii)*2*num)

						g_num_flag = 0

						ressfile = open('./%s/match_%s_%s_%s_%s' % (args.dataset_name,str(ii), str(jj), str(num), args.dataset_name), 'rb')
						ress = pickle.load(ressfile)

						totals=np.zeros((len(g_idx_gen),len(g_idx_gen)))

						print(len(g_idx_gen))

						for i1 in range(len(g_idx_gen) - 1):
							g_i1 = g_idx_gen[i1]
							totals[i1][i1]=1
							for i2 in range(i1 + 1, len(g_idx_gen)):
								g_i2 = g_idx_gen[i2]

								if g_i1.number_of_edges()==g_i2.number_of_edges():

									# print(')))))')

									res=ress[str(i1) + '_' + str(i2)]

									totals[i1][i2]=res[-1][-1]
								# totals[i2][i1] = res[-1][-1]

						# totals[i1][i1] = 0
						print('***',np.shape(np.where(totals!=0)))

						min_total=np.min(totals)
						max_total = np.max(totals)

						if max_total==0:
							continue


						totals_=np.array(list(itertools.chain.from_iterable(totals)))
						totals_=np.sort(totals_)
						print(totals_)
						threshold=totals_[int(0.999*len(totals_))-1]
						print(threshold)
						threshold=totals_[-1]
						idx=np.array(np.where(totals>=threshold)).T
						# print(idx)

						# print(np.shape(idx))
						# exit()
						for i1,i2 in idx:
							if i1==i2:
								continue
							g_i1 = g_idx_gen[i1]
							g_i2 = g_idx_gen[i2]
							res0 = ress[str(i1) + '_' + str(i2)]

							res=copy.deepcopy(res0)
							res=np.array(res)

							# print(res)
							# exit()

							if (np.shape(res)[0] != num - 1) and (np.shape(res)[0] != num) :
								# print('^^^')
								continue


							if np.shape(res)[0]==num-1:
								print('***')
								n0_i1=list(set(res[:,0]).difference(set(res[:,1])))
								n0_i2=list(set(res[:,1]).difference(set(res[:,0])))

								if n0_i1==n0_i2==[]:
									print('###')
									n0=(0+num-1)*num/2-np.sum(res[:, 0])
									res0.append([n0, n0, 0, res0[-1][-1]])


								# print(res)
								# print(n0_i1,n0_i2)
								else:

									res0.append([n0_i1[0],n0_i2[0],0,res0[-1][-1]])

								# print(np.shape(res0))

								res=np.array(res0)

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

							# print(nx.adjacency_matrix(g_i1).todense(),g1_map_adj,g2_adj)

							pred_adj=np.logical_and(g1_map_adj,g2_adj).astype(np.int)
							print(pred_adj,np.shape(np.where(pred_adj==1)))
							# pred_adj=np.logical_or(g1_map_adj,g2_adj).astype(np.int)
							# print(pred_adj,np.shape(np.where(pred_adj==1)))

							pred_g.append(pred_adj)
							# exit()



						with open('./%s/pred_g_%s_%s_%s_%s' % (args.dataset_name,str(ii), str(jj), str(num), args.dataset_name), 'wb') as f:
							pickle.dump(pred_g, f)

