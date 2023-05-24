import sys
sys.path.append('./')
import glim
import numpy as np
import os
import glim.utils as utils

relationship_file = 'relationship_table.txt'
embedding_save_file = './results/hmln_feature.npy'


def load_graph_network(adj_path, feature_path):
    X, A, Y = [], None, []
    n_node = 0

    # Acquire Edges
    edge_list = []
    node_list = []
    node_type = {}

    with open(adj_path, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f.readlines():
            node1, node2, *_ = line.strip().split('\t')
            edge_list.append((node1, node2))
            node_list.extend([node1, node2])

    # 构建节点的映射表（node_map），通过将节点列表中的名字映射成连续的整数索引来实现。并且统计映射表的大小，即节点的总数            
    node_map = {item:i for i, item in enumerate(sorted(list(set(node_list))))}
    n_node = len(node_map)
    # 构建邻接矩阵
    A = np.zeros((n_node, n_node))
    for node1, node2 in edge_list:
        A[node_map[node1], node_map[node2]] = 1
        A[node_map[node2], node_map[node1]] = 1
    A = np.float32(A)
    
    
    ####################################################
    #            Acquire Features                      #
    ####################################################

    if os.path.exists(feature_path):
        X = np.load(feature_path)
    else:
        # 使用Node2vec生成节点特征
        '''
            A:邻接矩阵
            512:隐层神经元个数
            4:控制游走深度/广度
            1:控制游走深度/广度
        '''
        X = np.float32(utils.N2V(A, 512, 4, 1))
        np.save(feature_path, X)
    
    return X, A


# features, adj, labels = load_multilayer()
filepath='./data/'
adj_path = filepath + relationship_file
features, adj = load_graph_network(adj_path, embedding_save_file)

adj = adj + np.eye(adj.shape[0]) # self-loop is needed

glim.train.fit_transform(features, adj, embedding_save_file, device='cuda')


filepath='./data/'
# Acquire Edges
edge_list = []
node_list = []
node_type = {}
adj_path = filepath + relationship_file
with open(adj_path, 'rt', encoding='utf-8') as f:
    next(f)
    for line in f.readlines():
        node1, node2, type1, type2,_ = line.strip().split('\t')
        edge_list.append((node1, node2))
        node_list.extend([node1, node2])
        if len(type1) != 1:
            type1 = 'c'
        if len(type2) != 1:
            type2 = 'c'
        node_type[node1] = type1
        node_type[node2] = type2

node_map = {item:i for i, item in enumerate(sorted(list(set(node_list))))}
n_node = len(node_map)


import json
with open('./results/multilayer_node_map.json', 'wt') as f:
    json.dump(node_map, f)


