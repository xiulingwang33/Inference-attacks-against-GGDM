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
import dgl
from ged import graph_edit_distance

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
	g1_edges = nx.edges(g_i1)
	g2_edges = nx.edges(g_i2)
	g1_adj = nx.adjacency_matrix(g_i1).todense()
	g2_adj = nx.adjacency_matrix(g_i2).todense()

	g1_adj_ = np.triu(g1_adj,k=1)
	g2_adj_ = np.triu(g2_adj,k=1)

	g1_adj_idx=np.where(g1_adj_==1)
	g2_adj_idx=np.where(g2_adj_==1)

	src1 = g1_adj_idx[0]
	dst1 = g1_adj_idx[1]
	src2 = g2_adj_idx[0]
	dst2 = g2_adj_idx[1]

	G1 = dgl.DGLGraph((src1, dst1))
	G2 = dgl.DGLGraph((src2, dst2))
	distance, node_mapping, edge_mapping = graph_edit_distance(G1, G2, algorithm='astar')

	# distance, node_mapping, edge_mapping = graph_edit_distance(G1, G2, algorithm='astar')
	# print('***',distance)
	return (distance, node_mapping, edge_mapping)




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
	# datasets = ["mutag", "enzymes"]

	result_all = []
	for data_ in datasets:
		args.dataset_name =data_
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
		division = int(len(label_list_train) / 3)

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
				for file in files:
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

					for num in set(node_num_inter):
						idx_orig = np.array(np.where(np.array(g_orig_node_idx) == num)[0])
						idx_gen = np.array(np.where(np.array(num_nodes) == num)[0])

						g_idx_orig = np.array(g_orig)[idx_orig]
						g_idx_gen = np.array(graph_s_gen)[idx_gen]

						g_num_flag = 0

						ress= {}

						for i1 in range(len(g_idx_gen) - 1):
							g_i1 = g_idx_gen[i1]
							for i2 in range(i1 + 1, len(g_idx_gen)):
								g_i2 = g_idx_gen[i2]

								# print(np.shape(g_i1),np.shape(g_i2))

								if g_i1.number_of_edges() == g_i2.number_of_edges():
									res=main(args,g_i1,g_i2)
									ress[str(i1) + '_' + str(i2)] = res
								# ress2[str(i1) + '_' + str(i2)] = res2

						with open('./%s/match_%s_%s_%s_%s' % (args.dataset_name,str(ii), str(jj), str(num), args.dataset_name), 'wb') as f:
							pickle.dump(ress, f)

