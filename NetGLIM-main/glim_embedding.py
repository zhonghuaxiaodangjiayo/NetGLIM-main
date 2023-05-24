import glim
import numpy as np
import os
import glim.utils as utils
import argparse


parser = argparse.ArgumentParser(
  description='GLIM',
  formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--relationship-file', type=str, default='./data/total_relationship_table.txt')
parser.add_argument('--node-feature-file', type=str, default='./data/M2V_feature.npy')
parser.add_argument('--embedding-save-file', type=str, default='./results/M2V_total_feature.npy')

args = parser.parse_args()

'''
    图网络的加载功能，该函数接受两个参数：
    一个是含有图网络架构信息的.adj文件的路径（adj_path），
    另一个是矩阵表示的节点特征信息（feature_path）。
    该函数返回节点特征矩阵(X)，邻接矩阵(A)和标签列表(Y)
'''
def load_graph_network(adj_path, feature_path):
    GRN = utils.importGRN()
    HTCG = utils.importDatas()

    # 获取 GRN 数据集中的所有节点元素。
    grn_nodes = set(GRN['tfs'].keys()) | set(GRN['gene_tfs'].keys()) | set(GRN['res'].keys()) | set(GRN['tf_res'].keys())
    # 获取 HTCG 数据集中的所有节点元素。
    htcg_nodes = (set(HTCG['g_g'].keys()) | set(HTCG['g_t'].keys()) | set(HTCG['g_c'].keys())
    | set(HTCG['c_g'].keys()) | set(HTCG['c_t'].keys()) | set(HTCG['c_c'].keys())
    | set(HTCG['t_g'].keys()) | set(HTCG['t_t'].keys()) | set(HTCG['t_c'].keys()))

    total_node = grn_nodes.union(htcg_nodes)
    print(f"len(total_node) = {len(total_node)}")

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
            if node1 not in total_node or node2 not in total_node:
                continue
            edge_list.append((node1, node2))
            # 会包含重复的节点
            node_list.extend([node1, node2])


    # 构建节点的映射表（node_map），通过将节点列表中的名字映射成连续的整数索引来实现。并且统计映射表的大小，即节点的总数            
    node_map = {item:i for i, item in enumerate(sorted(list(set(node_list))))}
    # with open('/share/home/liangzhongming/code4Article/NetGLIM-main/results/node_map2.txt', 'w') as f:
    #     for node_name, node_index in node_map.items():
    #         f.write(f"{node_name}\t{node_index}\n")

    n_node = len(node_map)
    print(f"len(node_map) = {len(node_map)}")
    # 构建邻接矩阵
    A = np.zeros((n_node, n_node))
    for node1, node2 in edge_list:
        A[node_map[node1], node_map[node2]] = 1
        A[node_map[node2], node_map[node1]] = 1

    # # 将缺失的节点添加到邻接矩阵中
    # for node_name in node_map:
    #     if node_name not in node_list:
    #         A[node_map[node_name], node_map[node_name]] = 1
    A = np.float32(A)

    print(f"Adjacency matrix shape:{A.shape}")
    
    
    ####################################################
    #            Acquire Features                      #
    ####################################################

    if os.path.exists(feature_path):
        X = np.load(feature_path)
        print(f"os.path.exists(feature_path) is TRUE")
    else:
        # 使用Node2vec生成节点特征
        '''
            A:邻接矩阵
            512:隐层神经元个数
            4:控制游走深度/广度
            1:控制游走深度/广度
        '''
        # X = np.float32(utils.N2V(A, 512, 4, 1))
        X = np.float32(utils.M2V(walk_length=40, num_walks=40))
        np.save(feature_path, X)
    
    print(f"Feature matrix shape:{X.shape}")
    
    return X, A


if __name__ == '__main__': 
    features, adj = load_graph_network(args.relationship_file, args.node_feature_file)
    # 这一行代码添加了邻接矩阵的对角线，以避免节点自连接的问题。此外，通过添加对角线后，网络就变成了无向的
    adj = adj + np.eye(adj.shape[0])
    # 使用 glim.train.fit_transform 方法来训练嵌入向量
    glim.train.fit_transform(features, adj, args.embedding_save_file, device='cpu')

    # # 加载npy文件，并将npy文件写入一个txt文件
    # test=np.load('/share/home/liangzhongming/code/GLIM-main/results/hmln_feature.npy',encoding = "latin1")  #加载文件
    # doc = open('/share/home/liangzhongming/code/GLIM-main/results/hmln_feature.txt', 'a')  #打开一个存储文件，并依次写入
    # print(test, file=doc)  #将打印内容写入文件中
