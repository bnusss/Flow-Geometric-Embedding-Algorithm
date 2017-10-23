#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
@author: Guweiwei, Gongli
@function: core code of flownetwork algorithm
'''

import numpy as np
import networkx as nx

def plus_one(item):
    return item + 1

# 构建流网络
def create(walks, node_num):
    mg = nx.MultiDiGraph()
    walks = [map(plus_one, walk) for walk in walks]
    # create
    for walk in walks:
        walk.insert(0, 0)
        walk.append(node_num+1)
        edges = zip(walk[:-1], walk[1:])
        for edge in edges:
            mg.add_edge(edge[0], edge[1], weight=1)

    # merge the same edge
    g = nx.DiGraph()
    for v, nbrs in mg.adjacency_iter():
        for nbr, edict in nbrs.items():
            sum_weight = sum([d['weight'] for d in edict.values()])
            g.add_edge(v, nbr, weight=sum_weight)

    print 'Node numbers is:{0} Edge numbers is:{1}\n'.format(g.number_of_nodes(), g.number_of_edges())

    return g

# 计算M矩阵
def markov_matrix(g):
    m_temp = np.zeros((g.number_of_nodes(), g.number_of_nodes()))
    for v, nbrs in g.adjacency_iter():
        for nbr, attrs in nbrs.items():
            m_temp[v][nbr] = attrs['weight']

    src_idx = 0
    sink_idx = g.number_of_nodes() - 1

    check_count = 0
    for i in xrange(g.number_of_nodes()):
        if i != src_idx and i != sink_idx:
            if np.sum(m_temp[i, :]) == np.sum(m_temp[:, i]):
                print 'node:%s check!\tsum is %d' % (i, np.sum(m_temp[i, :]))
                check_count += 1
            else:
                print 'node:%s uncheck!' % (i,)
    print '%d check!\t %d uncheck!' % (check_count, g.number_of_nodes() - 2 - check_count)

    row_sum_weight = np.sum(m_temp, axis=1)
    row_sum_weight[g.number_of_nodes() - 1] = 1  # avoid divide by zero
    # print row_sum_weight[-1], len(row_sum_weight), NODES.index(MAX_ID)
    row_sum_weight = row_sum_weight.reshape((g.number_of_nodes(), 1))
    return np.mat(np.divide(m_temp, row_sum_weight))

# 计算U矩阵
def fundamental_matrix(g, m_matrix):
    Iden = np.identity(g.number_of_nodes())
    return np.linalg.inv(Iden - m_matrix)

# 计算L矩阵
def average_first_time_matrix(g, m_matrix, u_matrix):
    m_u_square = m_matrix*u_matrix**2
    l_temp = np.mat(np.zeros((g.number_of_nodes(), g.number_of_nodes())))
    for row_node in g.nodes():
        for col_node in g.nodes():
            _i = row_node
            _j = col_node
            l_temp[_i, _j] = m_u_square[_i, _j]/u_matrix[_i, _j] - m_u_square[_j, _j]/u_matrix[_j, _j]
    return l_temp

# 计算C矩阵
def l2c_matrix(l_matrix):
#    this symmetric method is from gl, i change it to node_ij = node_i + ndoe_ j
    c = np.zeros(l_matrix.shape)
    for _i in range(l_matrix.shape[0]):
        for _j in range(l_matrix.shape[1]):
            c[_i, _j] = 2*l_matrix[_i, _j]*l_matrix[_j, _i] / (l_matrix[_i, _j] + l_matrix[_j, _i])
    return np.nan_to_num(c)

#def l2c_matrix(l_matrix):
##    this symmetric method is from gww, node_ij = node_i + ndoe_ j
#    c = np.zeros(l_matrix.shape)
#    for _i in range(l_matrix.shape[0]):
#        for _j in range(l_matrix.shape[1]):
#            c[_i, _j] = l_matrix[_i, _j] + l_matrix[_j, _i] 
#            c[_j, _i] = l_matrix[_i, _j] + l_matrix[_j, _i] 
#    return c

def filter_src_sink(g, c):
    MIN_ID = 0
    MAX_ID = g.number_of_nodes() - 1
    c = c[MIN_ID+1:MAX_ID, MIN_ID+1:MAX_ID]
    return c
