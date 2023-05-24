import os 
import numpy as np
import math
import random

import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv

from . import utils


class Encoder(nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super(Encoder, self).__init__()
            self.conv = GCNConv(in_channels, hidden_channels, cached=True)
            # PReLU 是一种带有参数的修正线性单元激活函数
            self.prelu = nn.PReLU(hidden_channels)

        def forward(self, x, edge_index):
            out = self.conv(x, edge_index)
            out = self.prelu(out)
            return out

# 继承了 MessagePassing 的类 Summary，它定义了一种图节点的汇聚方法
# 聚集周围节点信息，作为一个local域的表示
class Summary(MessagePassing):
    # aggregation type: 1.mean, 2.max, 3.sum
    def __init__(self, aggr='max'): # 以社区最值最大的节点作为社区值表示
        super().__init__(aggr=aggr)

    def forward(self, x, edge_index):
        # 将输入数据 x 和边索引 edge_index 作为输入，通过调用 propagate() 函数进行传递和汇聚
        return self.propagate(edge_index, x=x)

    # 提供一个模板方法，实际使用时可以重写 message() 函数来定义不同的汇聚方法
    def message(self, x_j):
        return x_j

# 混淆输入特征
def corruption(x, edge_index):
    # 将 x 的行进行随机重排
    return x[torch.randperm(x.size(0))], edge_index

# 用于初始化权重矩阵的值
def uniform(size, tensor):
    '''
        先计算一个较小的值 bound,
        根据该值作为下限和上限对 tensor 中的元素进行均匀分布的初始化。
        这样做的目的是为了使权重的初始值不太大不太小，更有利于模型的收敛
    '''
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)

# 用于重置神经网络中模型的参数值
def reset(nn):
    # 调用其 reset_parameters() 方法重置参数
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        # 如果 nn 具有 children() 方法，则说明 nn 是一个容器型模型，它的各个子模块需要逐一进行参数重置，因此先遍历 nn 的 children()，对每个子模块进行递归调用
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class GraphLocalInfomax(torch.nn.Module):
    def __init__(self, hidden_channels, encoder, summary, corruption):
        super(GraphLocalInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        self.weight = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels)) # n * n 矩阵
        # 调用 reset_parameters() 函数来初始化模型参数
        self.reset_parameters()

    # 对图卷积层和汇聚层进行参数重置，并对权重矩阵进行初始化
    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)

    def forward(self, x, edge_index):
        # 对输入数据进行图卷积操作得到正样本 pos_z
        pos_z = self.encoder(x, edge_index)
        # 通过 corruption 对输入数据进行损坏操作得到负样本 neg_z
        cor = self.corruption(x, edge_index)
        cor = cor if isinstance(cor, tuple) else (cor, )
        neg_z = self.encoder(*cor)
        # 汇聚正样本节点作为区域的表示
        summary = self.summary(pos_z, edge_index)
        return pos_z, neg_z, summary

    # 评估正负样本是否与summary的相似度
    def discriminate(self, z, summary, sigmoid=True):
        value = torch.sum(torch.mul(z, torch.matmul(summary, self.weight)), dim=1)
        return value

    # 通过 discriminate() 计算出正负样本对summary的相似度，然后使用一个交叉熵损失函数对正负样本进行分类
    def loss(self, pos_z, neg_z, summary):
        pos_loss = self.discriminate(pos_z, summary)
        neg_loss = self.discriminate(neg_z, summary)
        # 二元交叉熵
        return -torch.log(1/(1 + torch.exp(torch.clamp(neg_loss-pos_loss, max=10)))).mean()
