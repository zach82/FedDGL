import torch
from loguru import logger
from copy import deepcopy
from itertools import chain
from dataset_loader.link import LinkDatasetTemplate
from dataset_loader.node import NodeDatasetTemplate
from dataset_loader.utils import negative_sampling
from collections import defaultdict
from utils import load_item, save_item
import numpy as np

class Trainer():
    def __init__(self, model, decoder, lossfn, dataset, evaluator, 
                 augment_method, lr, weight_decay, lr_decay, datasetG, 
                 save_folder_name, role, kwargs):
        """
        Parameters
        ----------
        model
            graph neural network model
        decoder
            decoder model
        lossfn
            loss function
        dataset
            dataset
        evaluator
            evaluator
        augment_method
            dataset augmentation method
        """
        self.model = model
        self.decoder = decoder
        self.lossfn = lossfn
        self.dataset = dataset
        self.evaluator = evaluator
        self.dataset.input_graphs = augment_method(dataset)
        self.datasetG = datasetG
        self.globalmodel = model
        self.datasetG.input_graphs = augment_method(datasetG)
        self.save_folder_name = save_folder_name
        self.role = role
        self.a = 0
        self.kwargs = kwargs

        # initialize before main loop

        self.optimizer = torch.optim.Adam(
            chain(
                self.model.parameters(),
                self.decoder.parameters()
            ),
            lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, lr_decay)
        self.b= 0.1
    def train(
        self,
        epochs,
        early_stopping
    ):
        """
        Parameters
        ----------
        epochs
            max epochs
        lr
            learning rate
        weight_decay
            weight decay
        lr_decay
            learning rate decay
        early_stopping
            early stopping patient

        Returns
        -------
        model
            trained model
        decoder
            trained decoder
        history
            traning history
        """
        mode = self.kwargs.get('mode', None)
        if mode in ["mine", "mine_kd", "mine_gp", "mine_pa"]:
            global_protos = load_item('Server', 'global_protos', self.save_folder_name)
            client_protos = load_item(self.role, 'protos', self.save_folder_name)
        for epoch in range(epochs):
            self.model.train()
            self.decoder.train()
            train_loss, _ = self.calc_loss_and_metrics('train', False)
            if mode in ["mine", "mine_gp", "mine_pa"]:
                cand, candG = self.nodefeature()
                distances = [np.linalg.norm(cand[key][0].cpu() - candG[key][0].cpu()) for key in candG.keys() if key in candG]
                average_distance = np.mean(distances)
                train_loss = train_loss + self.a*average_distance
            if mode in ["mine", "mine_kd", "mine_pa"] and global_protos is not None:
                distances = [np.linalg.norm(global_protos[key].cpu() - client_protos[key].cpu()) for key in global_protos.keys() if key in client_protos]
                average_distance = np.mean(distances)
                train_loss = train_loss + self.b*average_distance
            if mode in ["mine", "mine_kd", "mine_gp", "mine_pa"] and global_protos is not None:
                pass
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if mode in ["fedego", "fedproto", "fedprotodyn", "mine", "mine_kd", "mine_gp", "mine_pa"]:
            self.collect_protos()
        return train_loss.item()
    def collect_protos(self):
        protos = defaultdict(list)
        if isinstance(self.dataset, LinkDatasetTemplate):
            split = 'train'
            evaluate = False
        elif isinstance(self.dataset, NodeDatasetTemplate):
            split = 'train'
            embedding = self.model(
                self.dataset, len(self.dataset)-1, len(self.dataset)
            ).squeeze()
            label = self.dataset.ndata['label']
            mask = self.dataset.ndata[split]
            label = label[mask]
            embedding = embedding[mask]
            for i, yy in enumerate(label):
                y_c = yy.item()
                protos[y_c].append(embedding[i, :].detach())
        else:
            raise
        save_item(self.agg_func(protos), self.role, 'protos', self.save_folder_name)
    def nodefeature(self):
        cand = defaultdict(list)
        candG = defaultdict(list)
        split = 'train'
        embedding = self.model(
            self.dataset, len(self.dataset)-1, len(self.dataset)
        ).squeeze()
        label = self.dataset.ndata['label']
        mask = self.dataset.ndata[split]
        label = label[mask]
        embedding = embedding[mask]
        for i, yy in enumerate(label):
            y_c = yy.item()
            cand[y_c].append(embedding[i, :].detach())
        embedding = self.globalmodel(
            self.dataset, len(self.dataset)-1, len(self.dataset)
        ).squeeze()
        label = self.dataset.ndata['label']
        mask = self.dataset.ndata[split]
        label = label[mask]
        embedding = embedding[mask]
        for i, yy in enumerate(label):
            y_c = yy.item()
            candG[y_c].append(embedding[i, :].detach())
        return cand, candG
    def agg_func(self, protos):
        for [label, proto_list] in protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                protos[label] = proto / len(proto_list)
            else:
                protos[label] = proto_list[0]
        return protos
    def test(self):
        with torch.no_grad():
            test_loss, test_metric = self.calc_loss_and_metrics('test', True)
        return test_loss.item(), test_metric.item()
    def testG(self):
        with torch.no_grad():
            test_loss, test_metric = self.calc_loss_and_metricsG('test', True)
        return test_loss.item(), test_metric.item()
    def calc_loss_and_metrics(self, split, evaluate):
        """
        Calculate losses and metrics for given task

        Parameters
        ----------
        split
            dataset split
        grad
            whether need to calcudate gradient

        Returns
        -------
        loss and metrics
        """
        self.kwargs['test'] = 'local'
        if isinstance(self.dataset, LinkDatasetTemplate):
            return self.link_prediction(split, evaluate)
        elif isinstance(self.dataset, NodeDatasetTemplate):
            return self.node_prediction(split)
        else:
            raise
    def link_prediction(self, split, evaluate):
        """
        Calculate losses and metrics for link prediction task

        Be careful for the start and end points of the input and output of this task
        This task aims to predict links at time t + 1 given data at time t

        Parameters
        ----------
        split
            dataset split
        evalueate
            use stored negative edge sample

        Returns
        -------
        loss and metric
        """
        start, end = self.dataset.get_range_by_split(split)
        if split == 'train':
            input_start, input_end = start, end-1
            output_start, output_end = start+1, end
        else:
            input_start, input_end = start-1, end-1
            output_start, output_end = start, end
        edges = [
            torch.stack(graph.edges(), dim=1).reshape(-1, 2)
            for graph in self.dataset[output_start:output_end]
        ]
        if evaluate:
            non_edges = self.dataset.non_edges[output_start:output_end]
        else:
            non_edges = [
                negative_sampling(adj_list)
                for adj_list in self.dataset.adj_lists[output_start:output_end]
            ]
            non_edges = [
                torch.tensor(non_edge, dtype=torch.long, device=self.dataset.device).reshape(-1, 2)
                for non_edge in non_edges
            ]
        assert len(edges) == len(non_edges)
        for edge, non_edge in zip(edges, non_edges):
            assert edge.shape == non_edge.shape
        pairs = [torch.concat([edge, non_edge]) for edge, non_edge in zip(edges, non_edges)]
        label_kwargs = {'dtype': torch.long, 'device': self.dataset.device}
        labels = [
            torch.concat([
                torch.ones(edge.shape[0], **label_kwargs),
                torch.zeros(non_edge.shape[0], **label_kwargs)
            ])
            for edge, non_edge in zip(edges, non_edges)
        ]
        embedding = self.model(self.dataset, input_start, input_end)
        logits = self.decoder(embedding, pairs)
        loss = self.lossfn(logits, labels)
        metric = self.evaluator(logits, labels)
        return loss, metric
    def node_prediction(self, split):
        """
        Calculate losses and metrics for node prediction task
        Only last time step has used for this task

        Parameters
        ----------
        split
            dataset split

        Returns
        -------
        loss and metric
        """
        embedding = self.model(
            self.dataset, len(self.dataset)-1, len(self.dataset)
        ).squeeze()
        logit = self.decoder(embedding)
        label = self.dataset.ndata['label']
        mask = self.dataset.ndata[split]
        logit, label = logit[mask], label[mask]
        loss = self.lossfn(logit, label)
        metric = self.evaluator(logit, label)
        return loss, metric
    def calc_loss_and_metricsG(self, split, evaluate):
        """
        Calculate losses and metrics for given task

        Parameters
        ----------
        split
            dataset split
        grad
            whether need to calcudate gradient

        Returns
        -------
        loss and metrics
        """
        self.kwargs['test'] = 'global'
        if isinstance(self.datasetG, LinkDatasetTemplate):
            return self.link_predictionG(split, evaluate)
        elif isinstance(self.datasetG, NodeDatasetTemplate):
            return self.node_predictionG(split)
        else:
            raise
    def link_predictionG(self, split, evaluate):
        """
        Calculate losses and metrics for link prediction task

        Be careful for the start and end points of the input and output of this task
        This task aims to predict links at time t + 1 given data at time t

        Parameters
        ----------
        split
            dataset split
        evalueate
            use stored negative edge sample

        Returns
        -------
        loss and metric
        """
        start, end = self.datasetG.get_range_by_split(split)
        if split == 'train':
            input_start, input_end = start, end-1
            output_start, output_end = start+1, end
        else:
            input_start, input_end = start-1, end-1
            output_start, output_end = start, end
        edges = [
            torch.stack(graph.edges(), dim=1).reshape(-1, 2)
            for graph in self.datasetG[output_start:output_end]
        ]
        if evaluate:
            non_edges = self.datasetG.non_edges[output_start:output_end]
        else:
            non_edges = [
                negative_sampling(adj_list)
                for adj_list in self.datasetG.adj_lists[output_start:output_end]
            ]
            non_edges = [
                torch.tensor(non_edge, dtype=torch.long, device=self.datasetG.device).reshape(-1, 2)
                for non_edge in non_edges
            ]
        assert len(edges) == len(non_edges)
        for edge, non_edge in zip(edges, non_edges):
            assert edge.shape == non_edge.shape
        pairs = [torch.concat([edge, non_edge]) for edge, non_edge in zip(edges, non_edges)]
        label_kwargs = {'dtype': torch.long, 'device': self.datasetG.device}
        labels = [
            torch.concat([
                torch.ones(edge.shape[0], **label_kwargs),
                torch.zeros(non_edge.shape[0], **label_kwargs)
            ])
            for edge, non_edge in zip(edges, non_edges)
        ]
        embedding = self.model(self.datasetG, input_start, input_end)
        logits = self.decoder(embedding, pairs)
        loss = self.lossfn(logits, labels)
        metric = self.evaluator(logits, labels)
        return loss, metric
    def node_predictionG(self, split):
        """
        Calculate losses and metrics for node prediction task
        Only last time step has used for this task

        Parameters
        ----------
        split
            dataset split

        Returns
        -------
        loss and metric
        """
        embedding = self.model(
            self.datasetG, len(self.datasetG)-1, len(self.datasetG)
        ).squeeze()
        logit = self.decoder(embedding)
        label = self.datasetG.ndata['label']
        mask = self.datasetG.ndata[split]
        logit, label = logit[mask], label[mask]
        loss = self.lossfn(logit, label)
        metric = self.evaluator(logit, label)
        return loss, metric
