#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
@author: Guweiwei
@function: core code implements the deepwalk algorithm
'''

import random
import networkx as nx
from collections import Counter


class Graph(object):

    def __init__(self, nx_G):
        self.G = nx_G
        random.seed()  # random seed
        self.corpus_num = 0
        self.length = 0
        self.walks = []

    # deepwalk嵌入--返回由流网络构建的语料库
    def deepwalk2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = G.neighbors(cur)
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        walk = map(str, walk)
        walk = ",".join(walk) + "\n"
        return walk

    # 模拟游走
    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print 'DeepWalk2Vec Walk iteration:'
        for walk_iter in range(num_walks):
            print str(walk_iter+1), '/', str(num_walks)
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.deepwalk2vec_walk(walk_length=walk_length, start_node=node))
        self.walks = walks
        return walks

    # 通过corpus估计边权
    def set_weights(self, walks=None):
        g_edges = self.G.edges()
        cnt = Counter(g_edges)
        if not walks:
            walks = self.walks
        for walk in walks:
            forward = zip(walk[:-1], walk[1:])
            walk.reverse()
            backward = zip(walk[:-1], walk[1:])
            cnt.update(forward)
            cnt.update(backward)
        self.add_weights(dict(cnt))
        return dict(cnt)

    # 添加边权属性
    def add_weights(self, weight_dict):
        nx.set_edge_attributes(self.G, 'weight', weight_dict)

    # 获取边权属性
    def get_weights(self):
        return nx.get_edge_attributes(self.G, 'weight')

    # 轮盘赌模型-按概率选择指定区域
    def roulette(self, datas, ps):
        return np.random.choice(datas, p=ps)
