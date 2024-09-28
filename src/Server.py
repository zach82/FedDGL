import torch.nn as nn
import numpy as np
from torch.optim import *
import torch
from torch.utils.data import DataLoader
from utils import load_item, save_item
from collections import defaultdict
import module
from dataset_loader.link import LinkDatasetTemplate
from dataset_loader.node import NodeDatasetTemplate
from evaluator import AUCMetric, F1Metric
import torch.nn.functional as F
from itertools import chain

class Server(nn.Module):
    def __init__(self, output_dim, device, kwargs, dataset, decoder_dim, 
                 save_folder_name, lr, weight_decay, lr_decay,
                 batch_size, margin_threthold, server_epochs):
        super(Server, self).__init__()
        self.num_classes = dataset.num_label
        self.h_feats = output_dim
        self.device = device
        self.lr = lr
        self.dataset = dataset
        if isinstance(dataset, LinkDatasetTemplate):
            decoder = module.PairDecoder(output_dim, decoder_dim, device)
            lossfn = module.PairLoss()
            evaluator = AUCMetric("multiclass", kwargs)
        elif isinstance(dataset, NodeDatasetTemplate):
            decoder = module.NodeDecoder(output_dim, decoder_dim, dataset.num_label, device)
            lossfn = module.NodeLoss()
            evaluator = F1Metric(dataset.num_label, "multiclass", kwargs)
        else:
            raise
        self.decoder = decoder
        self.evaluator = evaluator
        self.lossfn = lossfn
        self.server_epochs = server_epochs
        self.margin_threthold = margin_threthold
        self.batch_size = batch_size
        self.role = 'Server'
        self.save_folder_name = save_folder_name
        PROTO = Trainable_prototypes(
            self.num_classes, 
            self.h_feats, 
            self.h_feats, 
            self.device
        ).to(self.device)
        save_item(PROTO, self.role, 'PROTO', self.save_folder_name)
        print(PROTO)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = None
        self.max_gap = None
        self.optimizer = torch.optim.Adam(
            chain(
                self.decoder.parameters()
            ),
            lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, lr_decay)
    def update_Gen(self):
        PROTO = load_item(self.role, 'PROTO', self.save_folder_name)
        Gen_opt = torch.optim.SGD(PROTO.parameters(), lr=self.lr)
        PROTO.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size, 
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                y = torch.Tensor(y).type(torch.int64).to(self.device)
                proto_gen = PROTO(list(range(self.num_classes)))
                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)
                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                gap2 = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * gap2
                loss = self.CEloss(-dist, y)
                Gen_opt.zero_grad()
                loss.backward()
                Gen_opt.step()
        save_item(PROTO, self.role, 'PROTO', self.save_folder_name)
        PROTO.eval()
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = PROTO(torch.tensor(class_id, device=self.device)).detach()
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)
    @torch.no_grad()
    def receive_protos(self, clients):
        assert (len(clients) > 0)
        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_protos_per_client = []
        for client in clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            for k in protos.keys():
                self.uploaded_protos.append((protos[k], k))
            self.uploaded_protos_per_client.append(protos)
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = self.proto_cluster(self.uploaded_protos_per_client)
        for k1 in avg_protos.keys():
            for k2 in avg_protos.keys():
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)
        self.min_gap = torch.min(self.gap)
        for i in range(len(self.gap)):
            if self.gap[i] > torch.tensor(1e8, device=self.device):
                self.gap[i] = self.min_gap
        self.max_gap = torch.max(self.gap)
    @torch.no_grad()
    def proto_cluster(self, protos_list):
        proto_clusters = defaultdict(list)
        for protos in protos_list:
            for k in protos.keys():
                proto_clusters[k].append(protos[k])
        for k in proto_clusters.keys():
            protos = torch.stack(proto_clusters[k])
            proto_clusters[k] = torch.mean(protos, dim=0).detach()
        return proto_clusters
    def finetuningClassifier(self):
        embedding = self.uploaded_protos
        tensors = [t for t, _ in embedding]
        labels = [label for _, label in embedding]
        tensors = torch.stack(tensors)
        labels = torch.tensor(labels)
        tensors.to(self.device)
        labels.to(self.device)
        self.decoder.train()
        split = "train"
        if isinstance(self.dataset, LinkDatasetTemplate):
            return self.link_prediction(split, False)
            pass
        elif isinstance(self.dataset, NodeDatasetTemplate):
            logit = self.decoder(tensors).cpu()
            train_loss = self.lossfn(logit, labels)
            metric = self.evaluator(logit, labels)
        else:
            raise
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
    def getGlobalProto(self, clients):
        for client in clients:
            self.clients_prototype.append(client.local_prototype)
            for i in range(len(self.classes_len)):
                self.classes_len[i] += client.classes_len[i]
        for client in clients:
            weight = []
            for i in range(len(self.classes_len)):
                if self.classes_len[i] != 0:
                    weight_i = client.classes_len[i] / self.classes_len[i]
                    proto_weight = [element * weight_i for element in client.local_prototype[i]]
                    self.prototype[i] = [self.prototype[i][j] + proto_weight[j] for j in range(len(proto_weight))]
    def calClassifierWeight(self):
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        global_protos = list(global_protos.values())
        global_protos = [global_proto.cpu() for global_proto in global_protos]
        cosine_similaritys = []
        _sum = 0.0
        for client_proto in self.uploaded_protos_per_client:
            client_proto = [v.cpu() for k, v in sorted(client_proto.items(), key=lambda item: item[0])]
            a = client_proto
            b = global_protos
            a_array = np.array(a)
            b_array = np.array(b)
            dot_product = np.dot(a_array.flatten(), b_array.flatten())
            norm_a = np.linalg.norm(a_array)
            norm_b = np.linalg.norm(b_array)
            cosine_similarity = dot_product / (norm_a * norm_b)
            cosine_similarity = np.exp(0.5*cosine_similarity)
            cosine_similaritys.append(cosine_similarity)
            _sum += cosine_similarity
        classifierWeights = []
        for cosine_similarity in cosine_similaritys:
            weight = cosine_similarity / _sum
            classifierWeights.append(weight)
        return classifierWeights
    def resetGlobalProto(self):
        self.clients_prototype = []
        self.prototype = [[0 for i in range(self.h_feats)] for i in range(self.num_classes)]
        self.classes_len = [0 for i in range(self.num_classes)]
class Trainable_prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()
        self.device = device
        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim), 
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)
    def forward(self, class_id):
        class_id = torch.as_tensor(class_id, device=self.device)
        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)
        return out
