import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import *
import torch.nn.functional as F
import time
import module
import augmenter
from dataset_loader.link import LinkDatasetTemplate
from dataset_loader.node import NodeDatasetTemplate
from evaluator import AUCMetric, F1Metric
from trainer import Trainer

class Client(nn.Module):
    def __init__(self, input_dim, output_dim, device, kwargs, symmetric_trick,
                  dataset, alpha, beta, eps, K, dense, verbose, decoder_dim, 
                  epochs, lr, weight_decay, lr_decay, early_stopping, 
                  augment_method, model, datasetG, save_folder_name, id):
        super(Client, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.input_dim = input_dim
        self.kwargs = kwargs
        self.symmetric_trick = symmetric_trick
        self.dataset = dataset
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.K = K
        self.dense = dense
        self.verbose = verbose
        self.decoder_dim = decoder_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.early_stopping = early_stopping
        self.datasetG = datasetG
        self.save_folder_name = save_folder_name
        self.id = id  # integer
        self.role = 'Client_' + str(self.id)
        model_arguments = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'device': device,
            'renorm_order': 'sym',
            **kwargs
        }
        if augment_method == 'tiara' and not symmetric_trick:
            model_arguments['renorm_order'] = 'row'
        if model == 'GCN':
            model = module.GCN(**model_arguments)
        elif model == 'GCRN':
            model = module.GCRN(num_nodes=dataset.num_nodes, **model_arguments)
        elif model == 'EvolveGCN':
            model = module.EGCN(**model_arguments)
        else:
            raise NotImplementedError("no such model {}".format(model))
        tiara_argments = (
            alpha, beta, eps, K,
            symmetric_trick,
            device,
            dense,
            verbose
        )
        if augment_method == 'tiara':
            augment_method = augmenter.Tiara(*tiara_argments)
        elif augment_method == 'merge':
            augment_method = augmenter.Merge(device)
        elif augment_method == 'none':
            augment_method = augmenter.GCNNorm(device)
        else:
            raise NotImplementedError('no such augmenter {}'.format(augment_method))
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
        trainer = Trainer(
            model, decoder, lossfn, dataset, evaluator, augment_method, self.lr,
            self.weight_decay, self.lr_decay, self.datasetG, self.save_folder_name, 
            self.role, self.kwargs
        )
        self.model = model
        self.augment_method = augment_method
        self.decoder = decoder
        self.evaluator = evaluator
        self.trainer = trainer
    def supervisedTrain(self):
        train_loss = self.trainer.train(
            self.epochs,
            self.early_stopping
        )
        return train_loss
    def test(self):
        test_loss, test_metric = self.trainer.test()
        return test_loss, test_metric
    def testG(self):
        test_loss, test_metric = self.trainer.testG()
        return test_loss, test_metric

