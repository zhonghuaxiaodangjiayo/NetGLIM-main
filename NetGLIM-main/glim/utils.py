import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
import torch
import pickle
import json
from collections import defaultdict
from random import choice
from tqdm import tqdm
from gensim.models import Word2Vec

import networkx as nx
from node2vec import Node2Vec
from torch_geometric.nn import MetaPath2Vec

def data2tsne(data, n_pca=0):
    # 接收一个数据集并可选地接收一个参数作为主成分分析（PCA）所使用的主成分数量
    if n_pca > 0:
        pca = PCA(n_components=n_pca)
        embedding = pca.fit_transform(data)
    else:
        embedding = data
    # 使用t-SNE来进一步降低数据的维度，以便进行可视化
    tsne = TSNE()
    tsne.fit_transform(embedding)
    return tsne.embedding_

def data2umap(data, n_pca=0):
    if n_pca > 0:
        pca = PCA(n_components=n_pca)
        embedding = pca.fit_transform(data)
    else:
        embedding = data
    # 同tsne思路，替换为UMAP降维
    embedding_ = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric='cosine',
        n_components = 2,
        learning_rate = 1.0,
        spread = 1.0,
        set_op_mix_ratio = 1.0,
        local_connectivity = 1,
        repulsion_strength = 1,
        negative_sample_rate = 5,
        angular_rp_forest = False,
        verbose = False
    ).fit_transform(embedding)
    return embedding_

def umap_plot(data, save_path):
    import seaborn as sns
    plt.figure(figsize=(10,10))
    fig = sns.scatterplot(
        x = 'UMAP_1',
        y = 'UMAP_2',
        data = data,
        hue = 'hue',
        palette="deep"
    )
    fig = plt.gcf()
    fig.savefig(save_path)
    plt.close()
    
def gplot(embedding_, type_info, filename):
    test = pd.DataFrame(embedding_, columns=['UMAP_1', 'UMAP_2'])
    test['hue'] = type_info
    save_path = './pic/'+filename + '.png'
    umap_plot(test, save_path)

def create_plot(features, labels, save_path, style='tsne', n_pca=None):
    if style=='tsne':
        if not n_pca:
            n_pca = 0
        embedding_ = data2tsne(features, n_pca)
    elif style=='umap':
        if not n_pca:
            n_pca = 30
        embedding_ = data2umap(features, n_pca)
    else:
        print(f'No style:{style}!')
        return
    gplot(embedding_, labels, save_path)
    
    
def N2V(adj, hid_units, p=1, q=1, walk_length=20, num_walks=40):
    # 返回那些大于0的元素的位置坐标，这里作为边的两个顶点的编号
    edge_index = np.where(adj>0)
    # np.r_ 将行向量合并为一个二维的矩阵，最终生成一个二维数组，每一行是一条边的两个端点
    edge_index = np.r_[[edge_index[0]], [edge_index[1]]].T
    # 将边列表转换为网络图
    def create_net(elist):
        import networkx as nx
        # nx.Graph() 表示创建一个无向图
        g = nx.Graph()
        elist = np.array(elist)
        # 将边列表加入图中
        g.add_edges_from(elist)
        # 循环每一条边，将权重集成到图中
        for edge in g.edges():
            g[edge[0]][edge[1]]['weight'] = 1
        return g
    # 创建了一个图，并基于该图实例化了 Node2Vec 类
    graph = create_net(edge_index)
    node2vec = Node2Vec(graph, dimensions=hid_units, walk_length=walk_length, num_walks=num_walks, p=p,q=q)
    # 基于给定的图和超参数拟合 Node2Vec 模型，生成所有节点的嵌入向量，并将这些向量保存为 numpy 数组返回
    model = node2vec.fit()
    outputs = np.array([model.wv[str(item)] for item in range(len(adj))])
    return outputs


def importDatas():
    filename='/share/home/liangzhongming/code4Article/NetGLIM-main/data/relationship_table.txt'
    g_g = dict()
    g_c = dict()
    g_t = dict()

    c_g = dict()
    c_c = dict()
    c_t = dict()

    t_g = dict()
    t_t = dict()
    t_c = dict()

    with open(filename, 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            src, dst, edge_type1, edge_type2, source = line
            edge_type = edge_type1 + "-" + edge_type2

            if edge_type1 == 'g' and edge_type2 == 'g':
                gene1 = src
                gene2 = dst
                if gene1 not in g_g:
                    g_g[gene1] = set()
                if gene2 not in g_g:
                    g_g[gene2] = set()
                g_g[gene1].add(gene2)
                g_g[gene2].add(gene1)
            
            elif edge_type1 == 'g' and edge_type2 == 't':
                gene = src
                tissue = dst
                if gene not in g_t:
                    g_t[gene] = set()
                if tissue not in t_g:
                    t_g[tissue] = set()
                g_t[gene].add(tissue)
                t_g[tissue].add(gene)
            
            elif edge_type1 == 'g' and (edge_type2 == 'c' or edge_type2 == 't,c'):
                gene = src
                cell = dst
                if gene not in g_c:
                    g_c[gene] = set()
                if cell not in c_g:
                    c_g[cell] = set()
                g_c[gene].add(cell)
                c_g[cell].add(gene)

            elif (edge_type1 == 'c' or edge_type1 == 't,c') and edge_type2 == 't':
                cell = src
                tissue = dst
                if cell not in c_t:
                    c_t[cell] = set()
                if tissue not in t_c:
                    t_c[tissue] = set()
                c_t[cell].add(tissue)
                t_c[tissue].add(cell)

            elif (edge_type1 == 'c' or edge_type1 == 't,c') and (edge_type2 == 'c' or edge_type2 == 't,c'):
                cell1 = src
                cell2 = dst
                if cell1 not in c_c:
                    c_c[cell1] = set()
                if cell2 not in c_c:
                    c_c[cell2] = set()
                c_c[cell1].add(cell2)
                c_c[cell2].add(cell1)
            
            elif edge_type1 == 't' and edge_type2 == 't':
                tissue1 = src
                tissue2 = dst
                if tissue1 not in t_t:
                    t_t[tissue1] = set()
                if tissue2 not in t_t:
                    t_t[tissue2] = set()
                t_t[tissue1].add(tissue2)
                t_t[tissue2].add(tissue1)

    print(f"len g_g: {len(g_g)}, len g_t: {len(g_t)}, len g_c: {len(g_c)}")
    print(f"len c_g: {len(c_g)}, len c_t: {len(c_t)}, len c_c: {len(c_c)}")
    print(f"len t_g: {len(t_g)}, len t_t: {len(t_t)}, len t_c: {len(t_c)}")
               
                    
    return {'g_g': g_g, 'g_t': g_t, 'g_c':g_c, 'c_g': c_g, 'c_t': c_t, 'c_c': c_c, 't_g': t_g, 't_c':t_c, 't_t': t_t}

def importGRN():
    filename = '/share/home/liangzhongming/code4Article/NetGLIM-main/data/grn-human.txt'
    gene_tf = dict()
    gene_re = dict()
    res = dict()
    tfs = dict()
    tf_res = dict()
    count_tf = 0
    count_tg = 0
    count_re = 0
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            line = line.split()
            gene = line[0]
            re = line[1]
            tf = line[2]

            if gene not in gene_tf:
                count_tg += 1
                gene_tf[gene] = set()
            gene_tf[gene].add(tf)

            if gene not in gene_re:
                count_re += 1
                gene_re[gene] = set()
            gene_re[gene].add(re)

            if not re in res:
                res[re] = {'reg': set(), 'binded': set()}
            res[re]['reg'].add(gene)
            res[re]['binded'].add(tf)

            if tf not in tfs:
                count_tf += 1
                tfs[tf] = set()
            tfs[tf].add(gene)

            if tf not in tf_res:
                tf_res[tf] = set()
            tf_res[tf].add(re)
    
    # print(f"len gene_tf: {len(gene_tf)}, len gene_re: {len(gene_re)}")
    # print(f"len tf_gene: {len(tfs)}, len tf_res: {len(tf_res)}")
    # print(f"len re_gene: {len(res[re]['reg'])}, len re_tf: {len(res[re]['binded'])}")

    return {'tfs': tfs, 'gene_tfs': gene_tf, 'res': res, 'gene_res': gene_re, 'tf_res': tf_res}

def generate_metapath(walk_length=50, num_walks=40):
    GRN = importGRN()
    HTCG = importDatas()

    # 获取 GRN 数据集中的所有节点元素。
    grn_nodes = set(GRN['tfs'].keys()) | set(GRN['gene_tfs'].keys()) | set(GRN['res'].keys()) | set(GRN['tf_res'].keys())
    # 获取 HTCG 数据集中的所有节点元素。
    htcg_nodes = (set(HTCG['g_g'].keys()) | set(HTCG['g_t'].keys()) | set(HTCG['g_c'].keys())
    | set(HTCG['c_g'].keys()) | set(HTCG['c_t'].keys()) | set(HTCG['c_c'].keys())
    | set(HTCG['t_g'].keys()) | set(HTCG['t_t'].keys()) | set(HTCG['t_c'].keys()))

    total_node = grn_nodes.union(htcg_nodes)

    walks = []
    rest_walks = []
    '''1.TF-RE-TG(g)-t-c'''
    cnt_tf = 0
    L = len(GRN['tf_res'])
    for tf in tqdm(GRN['tf_res'], desc='1.TF-RE-TG-g-c:', total=L):
        cnt_tf += 1
        walk = []
        for i in range(num_walks):
            re = choice(list(GRN['tf_res'][tf]))
            gene = choice(list(GRN['res'][re]['reg']))
            if gene in HTCG['g_t']:
                tissue = choice(list(HTCG['g_t'][gene]))
                if tissue in HTCG['t_c']:
                    cell = choice(list(HTCG['t_c'][tissue]))
                else:
                    continue

                walk.append(tf)
                walk.append(re)
                walk.append(gene)
                walk.append(tissue)
                walk.append(cell)
            else:
                continue
    print(f"len of 1.TF-RE-TG(g)-t-c = {len(walk)}")
    walks.append(walk)


    walks = []
    '''1.1.T-G-RE-TF-RE-G-T'''
    cnt_tf = 0
    L = len(HTCG['t_g'])
    for tissueA in tqdm(HTCG['t_g'], desc='1.1.T-G-RE-TF-RE-G-T:', total=L):
        walk = []
        walk.append(tissueA)
        for i in range(num_walks):
            tissue = tissueA
            for j in range(walk_length):
                overlap = list(HTCG['t_g'][tissue] & GRN['gene_res'].keys())
                if not overlap:
                    continue
                geneA = choice(list(HTCG['t_g'][tissue] & GRN['gene_res'].keys()))
                reA = choice(list(GRN['gene_res'][geneA]))
                tf = choice(list(GRN['res'][reA]['binded']))
                reB = choice(list(GRN['tf_res'][tf]))
                overlap = list(GRN['res'][reB]['reg'] & HTCG['g_t'].keys())
                if not overlap:
                    continue
                geneB = choice(overlap)
                tissue = choice(list(HTCG['g_t'][geneB]))

                walk.append(geneA)
                walk.append(reA)
                walk.append(tf)
                walk.append(reB)
                walk.append(geneB)
                walk.append(tissue)

    print(f"len of 1.1.T-G-RE-TF-RE-G-T = {len(walk)}")
    walks.append(walk) 

    walks = []
    '''1.2.C-G-RE-TF-RE-G-C'''
    cnt_tf = 0
    L = len(HTCG['c_g'])
    for cellA in tqdm(HTCG['c_g'], desc='1.2.C-G-RE-TF-RE-G-C:', total=L):
        walk = []
        walk.append(cellA)
        for i in range(num_walks):
            cell = cellA
            for j in range(walk_length):
                overlap = list(HTCG['c_g'][cell] & GRN['gene_res'].keys())
                if not overlap:
                    continue
                geneA = choice(list(HTCG['c_g'][cell] & GRN['gene_res'].keys()))
                reA = choice(list(GRN['gene_res'][geneA]))
                tf = choice(list(GRN['res'][reA]['binded']))
                reB = choice(list(GRN['tf_res'][tf]))
                overlap = list(GRN['res'][reB]['reg'] & HTCG['g_c'].keys())
                if not overlap:
                    continue
                geneB = choice(overlap)
                cell = choice(list(HTCG['g_c'][geneB]))

                walk.append(geneA)
                walk.append(reA)
                walk.append(tf)
                walk.append(reB)
                walk.append(geneB)
                walk.append(cell)

    print(f"len of 1.2.C-G-RE-TF-RE-G-C = {len(walk)}")
    walks.append(walk)


    # For gwas tasks
    '''2.RE-C'''
    walk = []
    cnt_re = 0
    L = len(GRN['res'])
    for re in tqdm(GRN['res'], desc='2.RE-C:', total=L):
        cnt_re += 1
        for gene in GRN['res'][re]['reg'] & HTCG['g_c'].keys():
            for cell in HTCG['g_c'][gene]:
                walk.append(re)
                walk.append(cell)
    print(f"len of 2.RE-C = {len(walk)}")
    walks.append(walk)
    
    '''3.RE-T'''
    walk = []
    cnt_re = 0
    L = len(GRN['res'])
    for re in tqdm(GRN['res'], desc='3.RE-T:', total=L):
        cnt_re += 1
        for gene in GRN['res'][re]['reg'] & HTCG['g_t'].keys():
            for tissue in HTCG['g_t'][gene]:
                walk.append(re)
                walk.append(tissue)
    print(f"len of 3.RE-T = {len(walk)}")
    walks.append(walk)
    

    '''4.T-G-G-T'''
    L = len(HTCG['t_g'])
    walk = []
    for tissue in tqdm(HTCG['t_g'], desc='4.T-G-G-T:', total=L):
        for i in range(num_walks):
            walk.append(tissue)
            for j in range(walk_length):
                geneA = choice(list(HTCG['t_g'][tissue]))
                if geneA in HTCG['g_g'].keys():
                    overlap = list(HTCG['g_g'][geneA] & HTCG['g_t'].keys())
                    if not overlap:
                        geneB = geneA
                    else:
                        geneB = choice(overlap)
                    tissue = choice(list(HTCG['g_t'][geneB]))
                else:
                    continue

                walk.append(geneA)
                if geneB != geneA:
                    walk.append(geneB)
                walk.append(tissue)
    print(f"len of 4.T-G-G-T = {len(walk)}")
    walks.append(walk)

    '''5.C-G-G-C'''
    L = len(HTCG['c_g'])
    walk = []
    for cell in tqdm(HTCG['c_g'], desc='5.C-G-G-C:', total=L):
        for i in range(num_walks):
            walk.append(cell)
            for j in range(walk_length):
                geneA = choice(list(HTCG['c_g'][cell]))
                if geneA in HTCG['g_g'].keys():
                    overlap = list(HTCG['g_g'][geneA] & HTCG['g_c'].keys())
                    if not overlap:
                        geneB = geneA
                    else:
                        geneB = choice(overlap)
                    cell = choice(list(HTCG['g_c'][geneB]))
                else:
                    continue

                walk.append(geneA)
                if geneB != geneA:
                    walk.append(geneB)
                walk.append(cell)
    print(f"len of 5.C-G-G-C = {len(walk)}")
    walks.append(walk)

    # For g-g tasks
    '''6.G-G-G-G'''
    L = len(HTCG['g_g'])
    walk = []
    for gene in tqdm(HTCG['g_g'], desc='6.G-G-G-G:', total=L):
        for i in range(num_walks):
            walk.append(gene)
            for j in range(walk_length):
                gene = choice(list(HTCG['g_g'][gene]))
                walk.append(gene)
    print(f"len of 6.G-G-G-G = {len(walk)}")
    walks.append(walk)


    # 剩余节点进行广度优先游走
    walk = []
    for node in total_node:
        if node not in walks:
            cnt += 1
            # 如果剩余节点在GRN中的gene
            if node in GRN['gene_tfs'].keys():
                for re in GRN['gene_res'][node]:
                    walk.append(node)
                    walk.append(re)
                for tf in GRN['gene_tfs'][node]:
                    walk.append(node)
                    walk.append(tf)
            if node in GRN['res'].keys():
                for gene in GRN['res'][node]['reg']:
                    walk.append(node)
                    walk.append(gene)
                for tf in GRN['res'][node]['binded']:
                    walk.append(node)
                    walk.append(tf)        
            if node in GRN['tfs'].keys():
                for gene in GRN['tfs'][node]:
                    walk.append(node)
                    walk.append(gene)
            if node in GRN['tf_res'].keys():
                for re in GRN['tf_res'][node]:
                    walk.append(node)
                    walk.append(re)

            
            if node in HTCG['c_c'].keys():
                for cell in HTCG['c_c'][node]:
                    walk.append(node)
                    walk.append(cell)
            if node in HTCG['c_t'].keys():
                for tissue in HTCG['c_t'][node]:
                    walk.append(node)
                    walk.append(tissue)
            if node in HTCG['c_g'].keys():
                for gene in HTCG['c_g'][node]:
                    walk.append(node)
                    walk.append(gene)

            if node in HTCG['t_t'].keys():
                for tissue in HTCG['t_t'][node]:
                    walk.append(node)
                    walk.append(tissue)
            if node in HTCG['t_c'].keys():
                for cell in HTCG['t_c'][node]:
                    walk.append(node)
                    walk.append(cell)
            if node in HTCG['t_g'].keys():
                for gene in HTCG['t_g'][node]:
                    walk.append(node)
                    walk.append(gene)

            if node in HTCG['g_g'].keys():
                for gene in HTCG['g_g'][node]:
                    walk.append(node)
                    walk.append(gene)
            if node in HTCG['g_c'].keys():
                for cell in HTCG['g_c'][node]:
                    walk.append(node)
                    walk.append(cell)
            if node in HTCG['g_t'].keys():
                for tissue in HTCG['g_t'][node]:
                    walk.append(node)
                    walk.append(tissue)
            
    print(f"node not in walks = {cnt}") 
    print(f"len of rest walk = {len(walk)}")                
    walks.append(walk)

    return walks

def merge_file():
    w = open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/total_relationship_table.txt', 'w')
    cnt = 0
    with open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/relationship_table.txt', 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            src, dst, edge_type1, edge_type2, source = line
            w.write(src + '\t' + dst + '\n')
            cnt += 1
    
    with open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/grn-human.txt', 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            tg, re, tf, score1, score2 = line
            w.write(tf + '\t' + re + '\n')
            w.write(re + '\t' + tg + '\n')
            cnt += 1
    
    print(f"cnt : {cnt}")


def M2V(walk_length=20, num_walks=40):
    GRN = importGRN()
    HTCG = importDatas()

    # 获取 GRN 数据集中的所有节点元素。
    grn_nodes = set(GRN['tfs'].keys()) | set(GRN['gene_tfs'].keys()) | set(GRN['res'].keys())
    # 获取 HTCG 数据集中的所有节点元素。
    htcg_nodes = (set(HTCG['g_g'].keys()) | set(HTCG['g_t'].keys()) | set(HTCG['g_c'].keys())
    | set(HTCG['c_g'].keys()) | set(HTCG['c_t'].keys()) | set(HTCG['c_c'].keys())
    | set(HTCG['t_g'].keys()) | set(HTCG['t_t'].keys()) | set(HTCG['t_c'].keys()))

    total_node = grn_nodes.union(htcg_nodes)
    print(f"len of total_node ={len(total_node)}")

    node_map = {item:i for i, item in enumerate(sorted(list(set(total_node))))}

    # with open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/walks.pkl', 'rb') as f:
    #     walks = pickle.load(f)

    # model = Word2Vec(walks, vector_size=256, window=5, min_count=0, sg=1, workers=32, epochs=20)
    # model.save('/share/home/liangzhongming/code4Article/NetGLIM-main/results/m2v_vec')
    # model.wv.save_word2vec_format('/share/home/liangzhongming/code4Article/NetGLIM-main/results/m2v_vec.txt')

    model = Word2Vec.load('/share/home/liangzhongming/code4Article/NetGLIM-main/results/m2v_vec')
    # outputs = np.array([model.wv[node_map[item]] for item in range(len(total_node))])
    outputs = np.zeros((len(total_node), model.vector_size))
    for item in range(len(total_node)):
        if item in node_map.values():
            node_name = list(node_map.keys())[list(node_map.values()).index(item)]
            outputs[item] = model.wv[node_name]
    return outputs

def For_M2V_data():
    # 读取边文件，将边信息存储为字典和列表形式
    edges_dict = defaultdict(list)
    cnt = 0
    with open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/relationship_table.txt', 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            src, dst, edge_type1, edge_type2, source = line
            edge_type = edge_type1 + "-" + edge_type2
            edges_dict[edge_type].append((src, dst))
            cnt += 1

    with open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/grn-human.txt', 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            tg, re, tf, score1, score2 = line
            edge_type1 = "TF-RE"
            edge_type2 = "RE-TG"
            edges_dict[edge_type1].append((tf, re))
            edges_dict[edge_type2].append((re, tg))
            cnt += 1
    print(f"cnt : {cnt}")

    # 添加节点映射
    node_index = {}
    index_count = 0

    # 构建边索引字典
    edge_index_dict = {}
    for edge_type, edge_list in edges_dict.items():
        edge_list = list(set(edge_list))  # 去重，因为同一边可能存在多次出现
        # print(edge_list)
        # break
        src_nodes_list, dst_nodes_list = zip(*edge_list)
        all_nodes = list(set(list(src_nodes_list) + list(dst_nodes_list)))
        for node in all_nodes:
            if node not in node_index:
                node_index[node] = index_count
                index_count += 1

        # 更新节点ID并构建边张量
        src_nodes_int = [node_index[x] for x in src_nodes_list]
        dst_nodes_int = [node_index[x] for x in dst_nodes_list]
        edge_index_dict[edge_type] = torch.tensor([src_nodes_int, dst_nodes_int], dtype=torch.long)

        if "-" in edge_type:  # Saving the symmetric edge types to the edge index dictionary
            inverse_edge_type = "-".join(reversed(edge_type.split("-")))
            inverse_edges = torch.tensor([dst_nodes_int, src_nodes_int], dtype=torch.long)
            edge_index_dict[inverse_edge_type] = inverse_edges

    # 保存节点-索引映射到本地文件
    with open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/node_index.pkl', 'wb') as f:
        pickle.dump(node_index, f)
    
    with open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/node_index.json', 'w') as f:
        json.dump(node_index, f)

    # 保存边索引字典到本地文件
    with open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/edge_index_dict.pkl', 'wb') as f:
        pickle.dump(edge_index_dict, f)
    
    # 保存边索引字典到文本文件
    with open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/edge_index_dict.txt', 'w') as f:
        print(len(edge_index_dict))
        for edge_type, edge_index in edge_index_dict.items():
            edges = edge_index.t().tolist()  # 转置，并转换为列表形式
            edges_str = "\t".join(map(str, edges))  # 列表元素转换为字符串，并用空格连接
            f.write(f"{edge_type}\t{edges_str}\n")

if __name__ == '__main__':
    pass
    # For_M2V_data()
    # importDatas()
    # importGRN()
    # merge_file()



    # walks = generate_metapath()
    # with open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/walks.pkl','wb') as f:
    #     pickle.dump(walks, f)
    # with open('/share/home/liangzhongming/code4Article/NetGLIM-main/data/walks.txt','w') as f:
    #     for walk in walks:
    #         # The map function is used to convert each element of the walk list to a string
    #         f.write(" ".join(map(str, walk)) + "\n")
    # M2V()


