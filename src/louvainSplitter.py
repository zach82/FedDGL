import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx
import copy
import networkx as nx
import community as community_louvain

from baseSplitter import BaseSplitter


class LouvainSplitter(BaseTransform, BaseSplitter):
    """
    Split Data into small data via louvain algorithm.

    Args:
        client_num (int): Split data into ``client_num`` of pieces.
        delta (int): The gap between the number of nodes on each client.
    """
    def __init__(self, client_num, delta=20):
        self.delta = delta
        BaseSplitter.__init__(self, client_num)
    def __call__(self, data, dataset, **kwargs):
        data.index_orig = torch.arange(data.num_nodes)
        G = to_networkx(
            data,
            node_attrs=['x', 'y'],
            to_undirected=True)
        nx.set_node_attributes(G,
                               dict([(nid, nid)
                                     for nid in range(nx.number_of_nodes(G))]),
                               name="index_orig")
        partition = community_louvain.best_partition(G)
        cluster2node = {}
        for node in partition:
            cluster = partition[node]
            if cluster not in cluster2node:
                cluster2node[cluster] = [node]
            else:
                cluster2node[cluster].append(node)
        max_len = len(G) // self.client_num - self.delta
        max_len_client = len(G) // self.client_num
        tmp_cluster2node = {}
        for cluster in cluster2node:
            while len(cluster2node[cluster]) > max_len:
                tmp_cluster = cluster2node[cluster][:max_len]
                tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) +
                                 1] = tmp_cluster
                cluster2node[cluster] = cluster2node[cluster][max_len:]
        cluster2node.update(tmp_cluster2node)
        orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
        orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)
        client_node_idx = {idx: [] for idx in range(self.client_num)}
        idx = 0
        for (cluster, node_list) in orderedc2n:
            while len(node_list) + len(
                    client_node_idx[idx]) > max_len_client + self.delta:
                idx = (idx + 1) % self.client_num
            client_node_idx[idx] += node_list
            idx = (idx + 1) % self.client_num
        graphs = []
        x_shapes = []
        edge_shapes = []
        for owner in client_node_idx:
            nodes = client_node_idx[owner]
            graphs.append(from_networkx(nx.subgraph(G, nodes)))
            x_shapes.append(graphs[owner].x.shape[0])
            edge_shapes.append(graphs[owner].edge_index.shape[1])
        graphs = [copy.deepcopy(dataset) for i in range(self.client_num)]
        for i in range(self.client_num):
            graphs[i].x_shape = x_shapes[i]
            graphs[i].edge_shape = edge_shapes[i]
        return graphs
