# FGE_algorithm

   
FGE is an algorithmic framework for representational learning on graphs. Given any graph, it can learn continuous feature representations for the nodes, which can then be used for various downstream machine learning tasks. 

The code works under Windows , Linux and other Unix variants with python. 

This repository provides a reference implementation of *FGE_algorithm* as described in the paper:<br>
> The Hidden Flow Structure and Metric Space of Network Embedding Algorithms Based on Random Walks<br>
> Weiwei Gu, Li Gong, Xiaodan Lou, and Jiang Zhang<br>
> Scientific Reports.<br>
> <Insert paper link>

The *FGE* algorithm learns continuous representations for nodes in any (un)directed, (un)weighted graph. 
### Basic Usage

#### Example
To run *FGE* on Zachary's karate club network, execute the following command from the project home directory:<br/>
	``python main_appro_nd.py``

#### Options
You can check out the other options available to use with *FGE* using:<br/>
	``python main_appro_nd.py --help``

#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int <weight_float, optional>
		
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

#### Output
The vector representation of node in graph.


Parameters:\
``Input graph path (-input) `` \
``Output graph path (--output_n2v)``\
``Number of dimensions. Default is 128 (--dimensions)``\
``Length of walk per source. Default is 10 (--walk-length)``\
``Number of walks per source. Default is 2048 (--num-walks)``\
``Context size for optimization. Default is 10 (--window-size)``\
``Number of epochs in SGD. Default is 1 (-iter)``\
``Return hyperparameter. Default is 1 (--p)``\
``Inout hyperparameter. Default is 1 (--q)``\
``Graph is directed. (--directed)``\
``Graph is weighted. (--weighted )``


Usage:\
``python ./main_appro_nd.py --input ./karate.edgelist --output_n2v ./karate_embedding --dimensions 128`` 
