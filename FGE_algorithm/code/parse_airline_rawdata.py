#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
@author: Guweiwei, Gongli
@function: parse the raw airline data(airline_rawdata.rtf)
'''

import networkx as nx
from collections import Counter


source      = []
destination = []
with open('graph/raw_data.rtf','rb') as fr:
    for line in fr:
        line     = line.decode('Latin-1').encode('utf8')
        segments = line.strip().split(",")
        if len(segments) < 5:
            continue
        if segments[2] != '' and segments[4] != '':
            source.append(segments[2])
            destination.append(segments[4])

cnt = Counter()
cnt.update(source);cnt.update(destination)

# 以出现次数即流量从大到小排序
sorted_cnt = cnt.most_common()
nodes = [node for node, freq in sorted_cnt]

# 转为用节点序号存储
source2dest = zip([nodes.index(src) for src in source], [nodes.index(dest) for dest in destination])

cnt.clear()
cnt.update(source2dest)
weighted_edges = [(edge[0], edge[1], weight) for edge, weight in cnt.iteritems()]

# 创建有向图
g = nx.DiGraph()
g.add_weighted_edges_from(weighted_edges)

# 图信息
print '节点数：', g.number_of_nodes()
print '连边数：', g.number_of_edges()

# 存储图，格式： from to weight
nx.write_weighted_edgelist(g, 'graph/airline.edgelist')
