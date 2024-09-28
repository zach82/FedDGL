from collections import defaultdict
import random
import numpy as np
import torch
from torch_geometric.data.data import Data
from utils import *
import copy
from dataset_loader.utils import negative_sampling
from louvainSplitter import LouvainSplitter

class DataCenter(object):
    """docstring for DataCenter"""
    def __init__(self, client_num):
        super(DataCenter, self).__init__()
        self.client_num = client_num
    def load_dataSet(self,
                     dataset_name,
                     dataset,
                     split_mode="label"):
        assert dataset_name in [
                "BitcoinAlpha", "WikiElec", "RedditBody", "Brain", "DBLP3", "DBLP5", "Reddit"
        ], "error dataset name!"
        test_sample_rate = 0.3
        sample_rate = 0.5
        major_rate = 0.8
        test_num = 200
        if dataset_name in ["BitcoinAlpha", "WikiElec", "RedditBody"]:
            major_label = 1
            datas = [copy.deepcopy(dataset) for i in range(self.client_num+1)]
            gid = 0
            _len = len(dataset.graphs)
            while gid < _len:
                x = [
                    np.array(graph.dstdata["X"].cpu())
                    for graph in dataset
                ]
                x = np.concatenate(x)
                edges = [
                    torch.stack(graph.edges(), dim=1).reshape(-1, 2)
                    for graph in dataset
                ]
                non_edges = [
                    negative_sampling(adj_list)
                    for adj_list in dataset.adj_lists
                ]
                non_edges = [
                    torch.tensor(non_edge, dtype=torch.long, device=dataset.device).reshape(-1, 2)
                    for non_edge in non_edges
                ]
                pairs = [torch.concat([edge, non_edge]) for edge, non_edge in zip(edges, non_edges)]
                pairs = torch.cat(pairs)
                pairs = np.transpose(np.array(pairs.cpu()))
                label_kwargs = {'dtype': torch.long}
                labels = [
                    torch.concat([
                        torch.ones(edge.shape[0], **label_kwargs),
                        torch.zeros(non_edge.shape[0], **label_kwargs)
                    ])
                    for edge, non_edge in zip(edges, non_edges)
                ]
                labels = torch.cat(labels)
                labels = np.array(labels.cpu())
                edge_index = pairs
                edge_cnt = edge_index.shape[1]
                y = np.array(labels)
                adj_lists = defaultdict(set)
                node_map = {}
                for i in range(edge_cnt):
                    paper1 = edge_index[0][i]
                    paper2 = edge_index[1][i]
                    if not paper1 in node_map:
                        node_map[paper1] = len(node_map)
                    if not paper2 in node_map:
                        node_map[paper2] = len(node_map)
                    adj_lists[node_map[paper1]].add(node_map[paper2])
                    adj_lists[node_map[paper2]].add(node_map[paper1])
                x = x[list(node_map)]
                np.random.shuffle(y)
                y = y[:len(x)]
                for i in range(edge_cnt):
                    edge_index[0][i] = node_map[edge_index[0][i]]
                    edge_index[1][i] = node_map[edge_index[1][i]]
                assert len(x) == len(y) == len(adj_lists)
                num_classes = 2
                node_num = x.shape[0]
                in_feats = x.shape[1]
                sampling_node_nums = int(node_num * test_sample_rate)
                test_index = list(np.random.permutation(
                    np.arange(node_num))[:sampling_node_nums])
                distibution = np.zeros(num_classes)
                for node in y[test_index]:
                    distibution[node] += 1
                pos = {}
                for i, node in enumerate(test_index):
                    pos[node] = i
                test_edge_index_u = []
                test_edge_index_v = []
                for u in test_index:
                    for v in test_index:
                        if (v in adj_lists[u]):
                            test_edge_index_u.append(pos[u])
                            test_edge_index_u.append(pos[v])
                            test_edge_index_v.append(pos[v])
                            test_edge_index_v.append(pos[u])
                assert len(test_edge_index_u) % 2 == 0 and len(
                    test_edge_index_v) % 2 == 0
                test_edge_index = torch.stack(
                    [torch.tensor(test_edge_index_u),
                    torch.tensor(test_edge_index_v)], 0)
                test_x = torch.tensor(x[test_index])
                test_y = torch.tensor(y[test_index])
                test_data = Data(x=test_x, edge_index=test_edge_index, y=test_y)
                test_data.index_orig = torch.tensor(test_index)
                delete_index = list(test_index)
                sampling_node_num = int(node_num * sample_rate)
                clients_data = []
                index_list = []
                for cid in range(self.client_num):
                    client_nodes = np.delete(np.arange(node_num), delete_index)
                    node_by_label = defaultdict(list)
                    for node in client_nodes:
                        node_by_label[y[node]].append(node)
                    holding_label = np.random.permutation(
                        np.arange(num_classes))[:major_label]
                    holding_label_index = []
                    for label in holding_label:
                        holding_label_index += node_by_label[label]
                    major_num = int(sampling_node_num * major_rate)
                    major_num = min(major_num, len(holding_label_index))
                    major_index = list(np.random.permutation(
                        holding_label_index)[:major_num])
                    major_pos = []
                    _len = 1
                    for pos, node in enumerate(client_nodes):
                        if node in major_index:
                            major_pos.append(pos)
                    major_pos = np.array(major_pos, dtype=np.int_)
                    rest_num = sampling_node_num - major_num
                    rest_index = np.delete(client_nodes, major_pos)
                    other_index = list(np.random.permutation(rest_index)[:rest_num])
                    index = major_index + other_index
                    connect = set()
                    for u in index:
                        for v in index:
                            if (u in adj_lists[v]):
                                connect.add(u)
                                connect.add(v)
                    index = list(connect)
                    random.shuffle(index)
                    delete_index += index[len(index) - test_num:]
                    pos = {}
                    for i, node in enumerate(index):
                        pos[node] = i
                    index_list += [index]
                    client_edge_index_u = []
                    client_edge_index_v = []
                    for u in index:
                        for v in index:
                            if (u in adj_lists[v]):
                                client_edge_index_u.append(pos[u])
                                client_edge_index_u.append(pos[v])
                                client_edge_index_v.append(pos[v])
                                client_edge_index_v.append(pos[u])
                    assert len(client_edge_index_u) % 2 == 0 and len(
                        client_edge_index_v) % 2 == 0
                    client_edge_index = torch.stack(
                        [torch.tensor(client_edge_index_u),
                        torch.tensor(client_edge_index_v)], 0)
                    client_x = torch.tensor(x[index])
                    client_y = torch.tensor(y[index])
                    client_data = Data(
                        x=client_x, edge_index=client_edge_index, y=client_y)
                    client_data.index_orig = torch.tensor(index)
                    clients_data.append(client_data)
                    distibution = np.zeros(num_classes)
                    for node in client_y:
                        distibution[node] += 1
                gid = gid + 1
        elif dataset_name in ["Brain", "DBLP3", "DBLP5", "Reddit"]:
            if split_mode == "label":
                major_label = 3
                datas = [copy.deepcopy(dataset) for i in range(self.client_num)]
                gid = 0
                _len = len(dataset.graphs)
                while gid < _len:
                    x = [
                        np.array(graph.dstdata["X"].cpu())
                        for graph in dataset
                    ]
                    x = np.concatenate(x)
                    edges = [
                        torch.stack(graph.edges(), dim=1).reshape(-1, 2)
                        for graph in dataset
                    ]
                    pairs = torch.cat(edges)
                    pairs = np.transpose(np.array(pairs.cpu()))
                    labels = dataset.ndata['label']
                    labels = np.array(labels.cpu())
                    edge_index = pairs
                    edge_cnt = edge_index.shape[1]
                    y = np.array(labels)
                    adj_lists = defaultdict(set)
                    node_map = {}
                    for i in range(edge_cnt):
                        paper1 = edge_index[0][i]
                        paper2 = edge_index[1][i]
                        if not paper1 in node_map:
                            node_map[paper1] = len(node_map)
                        if not paper2 in node_map:
                            node_map[paper2] = len(node_map)
                        adj_lists[node_map[paper1]].add(node_map[paper2])
                        adj_lists[node_map[paper2]].add(node_map[paper1])
                    x = x[list(node_map)]
                    np.random.shuffle(y)
                    y = y[:len(x)]
                    for i in range(edge_cnt):
                        edge_index[0][i] = node_map[edge_index[0][i]]
                        edge_index[1][i] = node_map[edge_index[1][i]]
                    assert len(x) == len(y) == len(adj_lists)
                    num_classes = len(set(y))
                    node_num = x.shape[0]
                    in_feats = x.shape[1]
                    sampling_node_nums = int(node_num * test_sample_rate)
                    test_index = list(np.random.permutation(
                        np.arange(node_num))[:sampling_node_nums])
                    _len = 1
                    distibution = np.zeros(num_classes)
                    for node in y[test_index]:
                        distibution[node] += 1
                    pos = {}
                    for i, node in enumerate(test_index):
                        pos[node] = i
                    test_edge_index_u = []
                    test_edge_index_v = []
                    for u in test_index:
                        for v in test_index:
                            if (v in adj_lists[u]):
                                test_edge_index_u.append(pos[u])
                                test_edge_index_u.append(pos[v])
                                test_edge_index_v.append(pos[v])
                                test_edge_index_v.append(pos[u])
                    assert len(test_edge_index_u) % 2 == 0 and len(
                        test_edge_index_v) % 2 == 0
                    test_edge_index = torch.stack(
                        [torch.tensor(test_edge_index_u),
                        torch.tensor(test_edge_index_v)], 0)
                    test_x = torch.tensor(x[test_index])
                    test_y = torch.tensor(y[test_index])
                    test_data = Data(x=test_x, edge_index=test_edge_index, y=test_y)
                    test_data.index_orig = torch.tensor(test_index)
                    delete_index = list(test_index)
                    sampling_node_num = int(node_num * sample_rate)
                    clients_data = []
                    index_list = []
                    for cid in range(self.client_num):
                        client_nodes = np.delete(np.arange(node_num), delete_index)
                        node_by_label = defaultdict(list)
                        for node in client_nodes:
                            node_by_label[y[node]].append(node)
                        holding_label = np.random.permutation(
                            np.arange(num_classes))[:major_label]
                        holding_label_index = []
                        graph = dataset.graphs[gid]
                        for label in holding_label:
                            holding_label_index += node_by_label[label]
                        major_num = int(sampling_node_num * major_rate)
                        major_num = min(major_num, len(holding_label_index))
                        major_index = list(np.random.permutation(
                            holding_label_index)[:major_num])
                        major_pos = []
                        for pos, node in enumerate(client_nodes):
                            if node in major_index:
                                major_pos.append(pos)
                        major_pos = np.array(major_pos, dtype=np.int_)
                        rest_num = sampling_node_num - major_num
                        rest_index = np.delete(client_nodes, major_pos)
                        other_index = list(np.random.permutation(rest_index)[:rest_num])
                        index = major_index + other_index
                        labels = datas[cid].ndata['label']
                        connect = set()
                        for u in index:
                            for v in index:
                                if (u in adj_lists[v]):
                                    connect.add(u)
                                    connect.add(v)
                        index = list(connect)
                        random.shuffle(index)
                        delete_index += index[len(index) - test_num:]
                        pos = {}
                        for i, node in enumerate(index):
                            pos[node] = i
                        index_list += [index]
                        client_edge_index_u = []
                        client_edge_index_v = []
                        for u in index:
                            for v in index:
                                if (u in adj_lists[v]):
                                    client_edge_index_u.append(pos[u])
                                    client_edge_index_u.append(pos[v])
                                    client_edge_index_v.append(pos[v])
                                    client_edge_index_v.append(pos[u])
                        assert len(client_edge_index_u) % 2 == 0 and len(
                            client_edge_index_v) % 2 == 0
                        client_edge_index = torch.stack(
                            [torch.tensor(client_edge_index_u),
                            torch.tensor(client_edge_index_v)], 0)
                        client_x = torch.tensor(x[index])
                        client_y = torch.tensor(y[index])
                        client_data = Data(
                            x=client_x, edge_index=client_edge_index, y=client_y)
                        client_data.index_orig = torch.tensor(index)
                        clients_data.append(client_data)
                        distibution = np.zeros(num_classes)
                        for node in client_y:
                            distibution[node] += 1
                        top_indices = np.argsort(distibution)[-len(holding_label):][::-1]
                        top_values = distibution[top_indices]
                        for i, hold_index in enumerate(holding_label):
                            original_value = distibution[hold_index]
                            top_value_index = top_indices[i]
                            top_value = distibution[top_value_index]
                            distibution[hold_index] = top_value
                            distibution[top_value_index] = original_value
                    gid = gid + 1
            elif split_mode == "louvain":
                louvainSplitter = LouvainSplitter(self.client_num + 1)
                x = [
                    np.array(graph.dstdata["X"].cpu())
                    for graph in dataset
                ]
                x = np.concatenate(x)
                edges = [
                    torch.stack(graph.edges(), dim=1).reshape(-1, 2)
                    for graph in dataset
                ]
                pairs = torch.cat(edges)
                pairs = np.transpose(np.array(pairs.cpu()))
                labels = dataset.ndata['label']
                labels = np.array(labels.cpu())
                edge_index = pairs
                edge_cnt = edge_index.shape[1]
                y = np.array(labels)
                x = torch.tensor(x, dtype=torch.float)[:len(y)]
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                y = torch.tensor(y, dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, y=y)
                datas = louvainSplitter(data, dataset)
        return datas
