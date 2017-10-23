#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
@author: Guweiwei, Gongli
@function: the main file to run
'''
from sklearn.metrics.cluster import normalized_mutual_info_score
import node2vec
import deepwalk2vec
import align_algorithm as align_algo
import flow_network as fn
import approximately_flow_network as appro_fn
import cPickle as pickle
#print 'haha'
import random
import argparse
import networkx as nx
from gensim.models import Word2Vec
from collections import Counter

# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy.cluster.vq import kmeans, vq, whiten
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
#from sklearn.utils.extmath import _ravel
# Random state.
seed = np.random.RandomState(seed=3)
import time
# We'll use matplotlib for graphics.
#import matplotlib.pyplot as plt
#import matplotlib.patheffects as PathEffects
#import matplotlib
#import matplotlib.patches as mpatches

# We import seaborn to make nice plots.
#import seaborn as sns
#sns.set_style('white')
#sns.set_palette('muted')
#sns.set_context("notebook", font_scale=1.5,
#                rc={"lines.linewidth": 2.5})

# 参数解析
def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec & deepwalk2vec .")

    parser.add_argument('--input', nargs='?', default='../dataset/karate.edgelist',
                        help='Input edgelist-graph path')

    parser.add_argument('--output_n2v', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
#    parser.add_argument('--network-name', type=str, default='karate',
#                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=2048,
                        help='Number of walks per source. Default is 2048.')

    parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=10, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default= 1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default= 1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()

# 读取图像
def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
    	for edge in G.edges():
    	       G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G

# word2vec 计算嵌入向量
def learn_embeddings(g, walks, save_fpath, dim, window_size, workers, iters):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=dim, window=window_size, workers=workers, iter=iters, min_count=0, sg=1)
#    model.save_word2vec_format(save_fpath)
    embed_vec = []
    for node in g.nodes():
        embed_vec.append(model[str(node)].tolist())
    return embed_vec

# 计算向量间的欧式距离
def calc_euler_dis(array_2d):
    dis = np.zeros((array_2d.shape[0], array_2d.shape[0]))
    for row in xrange(array_2d.shape[0]):
        current_row = array_2d[row:row+1, :]
        dists = np.sqrt(np.sum(np.square(current_row - array_2d), axis=1))
        dis[row, :] = dists
    return dis
# 计算node2vec向量间的内积距离
def calc_innerDot_dis(array_2d):
    dis = np.zeros((array_2d.shape[0], array_2d.shape[0]))
    for i in xrange(array_2d.shape[0]):
        for j in xrange(i+1 ,array_2d.shape[0]):
            current_row = array_2d[i, :]
            next_row = array_2d[j, :]
            dists = np.dot(current_row, next_row)
            dis[i,j] = dists
            dis[j,i] = dists
    return dis
# k-means聚类
def cluster(data, clusters=1):
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(data)
    return kmeans.labels_
    # whitened = whiten(data)
    # centroid = kmeans(whitened, clusters)[0]
    # return vq(whitened, centroid)[0]

# w2c mds降维
def w2c_mds_dec(data, dim=2):
    mds = MDS(n_components=dim, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity='euclidean', n_jobs=1)
    return mds.fit(data).embedding_

# flow network mds降维
def flow_mds_dec(data, dim=2, pos=None):
    mds = MDS(n_components=dim, max_iter=3000, eps=1e-9, random_state=seed, dissimilarity='precomputed', n_jobs=1, n_init=1)
    return mds.fit_transform(data, init=pos)

# 画聚类图
def plot_cluster(data, clses, palette, alpha=1., size=40):
    # We choose a color palette with seaborn.
    ax = plt.subplot(aspect='equal')
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    sc = ax.scatter(data[:,0], data[:,1], lw=0, s=size, c=palette[clses], alpha=alpha)
    ax.axis('off')
    ax.axis('tight')

    for i, (x, y) in enumerate(zip(data[:, 0], data[:, 1])):
        ax.annotate(x, y, i, color=palette[clses[i]], fontsize=10)

# 存储权重
def save_weights(weights):
    fp = open('./Result/%s_weights_p%.3f_q%.3f' % (args.output_n2v[4:-13], args.p, args.q), 'w')
    fp.writelines(map(lambda item:str(item[1])+'\n', weights))
    fp.close()

# 计算参数敏感性
def correlation(): 
    ### read the graph ###
    nx_G = read_graph()

    ### deep walk ###
    G_node2vec = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G_node2vec.preprocess_transition_probs()
#    +1 is temp, 
    node_number = nx_G.number_of_nodes()
    print nx_G.nodes()
    last_nd_euler_dis = np.zeros((node_number, node_number))
    last_flow_euler_dis = np.zeros((node_number, node_number))

    # according to our experiments, we choose this word2vec params
    iters = 20
    window_size = 5
    dimensions = 2**7
    p = 1; q =1; network_name = 'karate'
    f_nmi = open('../result/{}.p={}.q={}.nmi.iters{}_wsize{}_dim{}.txt'
                            .format(network_name,p,q,iters, window_size, dimensions), 'w+')
    f_result = open('../result/{}.p={}.q={}.relations.iters{}_wsize{}_dim{}.txt'.format(network_name,p,q,iters, window_size, dimensions), 'w+')
    t_result = open('../result/{}.p={}.q={}.scalability.iters{}_wsize{}_dim{}.txt'.format(network_name,p,q,iters, window_size, dimensions), 'w+')
#    for i in range(10, 101, 10):   
    #gl change walk_length, we set walk_length=10
    # we embed into dim = 256 emb = 9
    for j in range(10,12):
#     num_walks = 2**j 
     num_walks = 512
        #we walk till j = 11
     for emb in range(7,8):
         for walk_length in range(30,46,10):
            dimensions = 2**emb
#            walk_length = 10
            #### simulate walks ####
            t1 = time.time()
            node2vec_walks = G_node2vec.simulate_walks(num_walks, walk_length)
            t2 = time.time()
            #### random-walk word2vec ####
            nd_vec = learn_embeddings(nx_G, node2vec_walks, args.output_n2v, dimensions, window_size, args.workers,iters)
            t3 = time.time()
            vector_file = '../result/vector/node2vec_%s_embedding_dim=%s_walktimes=%s.npy'%(network_name, dimensions, num_walks)
            np.save(vector_file, np.array(nd_vec))           
            print 'node2vec done'
            nd_euler_dis = calc_euler_dis(np.array(nd_vec))            
            ###save euler dis to file
            #Eucli_file = 'Eucli_dist/node2vec_%s_embedding_dim=%s_walktimes=%s.npy'%(network_name, dimensions, num_walks)
            #np.save(Eucli_file, nd_euler_dis)
            ###save inner dot dist            
           # nd_innerDot_dis = calc_innerDot_dis(np.array(nd_vec))            
            #InnerDot_file = 'InnerDot_dist/node2vec_%s_embedding_dim=%s_walktimes=%s.npy'%(network_name, dimensions, num_walks)
            #np.save(InnerDot_file, nd_innerDot_dis)
            
            ### approximately flow network
            appro_dict = appro_fn.create_dist(node2vec_walks, node_number)   
            appro_flow_matrix = appro_fn.cal_appro_flow_dist(appro_dict, node_number)
            print 'appro_flow_matrix done'
            t4 = time.time()
            appro_c_matrix = appro_fn.cal_f2c_matrix(appro_flow_matrix, node_number)
            t5 = time.time()
            appro_flow_euler_dis = np.array(appro_c_matrix)
            appro_flow_euler_dis = np.array(appro_c_matrix)
            Eucli_file = '../result/vector/Eucli_dist/appro_flow_%s_embedding_dim=%s_walktimes=%s.npy'%(network_name, dimensions, num_walks)
            np.save(Eucli_file, appro_flow_euler_dis)          
                     
            # cal vector embedding for appro_flow_dist vector calculated previous
#            flow_dim = 2
            t6 = time.time()
            appro_embed_pos = flow_mds_dec(appro_c_matrix, dim=2, pos=None)
            print 'embed done'
            t7 = time.time()
#            vector_file = 'vector/appro_flow_%s_embedding_dim=%s'%(network_name, flow_dim)
#                      # save time for scalability
#            np.save(vector_file, np.array(appro_embed_pos))    
            t_result.write('{},{},{},{},{},{},{},{}\n'.format(num_walks, walk_length, dimensions,
                window_size, t2-t1, t3-t2 ,t5-t4,t7-t6))
            t_result.flush()
            #### calculator pearsonr relations ####
            pearsonr_rate3 = pearsonr(nd_euler_dis.flatten(), appro_flow_euler_dis.flatten())[0]

            ##### save the result ####
            f_result.write('{},{},{},{},{}\n'.format(num_walks, walk_length, dimensions,
                window_size,  pearsonr_rate3))
            f_result.flush()

            #### pca and clustering nmi index ####
#            nd_pos = w2c_mds_dec(nd_vec, dim=2)
#            appro_fn_pos = flow_mds_dec(appro_c_matrix, dim=2, pos=nd_pos)
#            nd_labels  = cluster(nd_pos, 4)
#            appro_fn_labels  = cluster(appro_fn_pos, 4)            
#            nmi_3 = normalized_mutual_info_score(nd_labels, appro_fn_labels)
#            f_nmi.write('{},{},{},{},{}\n'.format(num_walks, walk_length, dimensions,
#                window_size, nmi_3))
#            f_nmi.flush()
    #### cluster ####
#    nd_labels  = cluster(nd_pos, 4)
#    fn_labels = cluster(fn_pos, 4)
#    f_result.close()

# 画word2vec和flownetwork嵌入图
def plot():
    ### read the graph ###
    nx_G = read_graph()

    ### deep walk ###
    G_node2vec = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G_node2vec.preprocess_transition_probs()

    # simulate walks
    node2vec_walks = G_node2vec.simulate_walks(args.num_walks, args.walk_length)
    G_node2vec.set_weights(node2vec_walks)
#    save_weights(G_node2vec.get_weights())

    #### word2vec embedding ####
    nd_vec = learn_embeddings(nx_G, node2vec_walks, args.output_n2v, args.dimensions, args.window_size, args.iter, args.workers)
    nd_pos = w2c_mds_dec(nd_vec, dim=2)

    #### flow network ####
    r_G_node2vec = fn.create(node2vec_walks, G_node2vec.G.number_of_nodes())
    m_matrix = fn.markov_matrix(r_G_node2vec)
    u_matrix = fn.fundamental_matrix(r_G_node2vec, m_matrix)
    l_matrix = fn.average_first_time_matrix(r_G_node2vec, m_matrix, u_matrix)
    c_matrix = fn.l2c_matrix(l_matrix)
    c_matrix = fn.filter_src_sink(r_G_node2vec, c_matrix)

    fn_pos = flow_mds_dec(c_matrix, dim=2, pos=nd_pos)

    #### cluster ####
    nd_labels  = cluster(nd_pos, 4)
    fn_labels = cluster(fn_pos, 4)

    #### eliminate the embedding scale and direction differences ####
    nd_pos, fn_pos = align_algo.Point_Align(nd_pos, fn_pos).align()

    #### plot ####
    plt.style.use('dark_background')
    fig = plt.figure()

    colors = [(1, 1, 0.0), (0, 1, 1), (1, 0.2, 1) , (0, 1, 0)]
    node_color = [colors[label] for label in nd_labels]

    nx.draw_networkx_nodes(nx_G, pos=nd_pos, node_color=node_color, alpha=0.9 , label='$node2vec\ embedding$', node_shape='o')
    nx.draw_networkx_nodes(nx_G, pos=fn_pos, node_color=node_color, alpha=0.3, label='$FGE\ embedding$', node_shape='s')

    nx.draw_networkx_labels(nx_G, pos=nd_pos)
    nx.draw_networkx_labels(nx_G, pos=fn_pos)

    plt.axis('off')
    plt.legend(loc='upper right', fontsize='x-small')
    plt.savefig('embedding.svg')
    plt.show()

# 主程序
def main(args):
     correlation()
#    plot()

if __name__ == "__main__":
    args = parse_args()
    main(args)
