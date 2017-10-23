# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 23:32:04 2017
cal appro flow distance based on sequence, input is randomwalk and nodeNum, output 
flow distance between each node pair
@author: Guweiwei, Gongli
"""

import numpy as np
from collections import defaultdict


def create_dist(sequences, node_number):
    dist_list_dict = defaultdict(list)
    for seq in sequences: 
        walk_length = len(seq)
        for i in range(walk_length):
            source = seq[i]
            dist = 1
            for j in range(i+1, walk_length):
                dest = seq[j]
                if source+'\t'+dest not in dist_list_dict:
                    dist_list_dict[source+'\t'+dest].append(dist)
                    dist_list_dict[source+'\t'+dest].append(1)
                else:
                    dist_list_dict[source+'\t'+dest][0] += dist
                    dist_list_dict[source+'\t'+dest][1] += 1
                dist += 1
    return dist_list_dict


def cal_appro_flow_dist(dist_list_dict, node_number):
    flow_matrix = np.zeros((node_number,node_number), dtype=np.float32)
    for key, iterms in dist_list_dict.iteritems():
        source_dest_list = key.split('\t')
        source = int(source_dest_list[0])
        dest = int((source_dest_list[1]))
        if iterms[0] == 0 and iterms[1] == 0:
            aver_dist = np.inf
        else:
            aver_dist = float(iterms[0])/iterms[1]
        flow_matrix[source, dest] = aver_dist
    return flow_matrix


def cal_f2c_matrix(flow_matrix, node_number):
    l_matrix = flow_matrix
    c = np.zeros((node_number,node_number))
    for _i in range(node_number):
        for _j in range(_i+1, node_number):
            c[_i, _j] = l_matrix[_i, _j] + l_matrix[_j, _i] 
            c[_j, _i] = l_matrix[_i, _j] + l_matrix[_j, _i] 
    return c  
    

if __name__ == "__main__":
#    for test
    walk_seq = [[0,1,2,3],[3,2,1],[1,3,2],[1,3,2]]
    node_number = 4
    ha = create_dist(walk_seq, node_number)   
    flow_matrix = cal_appro_flow_dist(ha, node_number)
    final = cal_f2c_matrix(flow_matrix, node_number)
        
        
#    dist_matrix = np.zeros((node_number,node_number))        
