from __future__ import annotations

import copy
from collections import OrderedDict

import flwr as fl
import torch
from torch import nn
from torch.utils.data import DataLoader

from cifar_baseline.engine import build_optimizer, train_one_epoch, evaluate
from cifar_baseline.model import ResNet18CIFAR

class CIFARClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        
        optimizer = build_optimizer(self.model, self.config)
        
        for _ in range(int(self.config["epochs"])):
            train_one_epoch(self.model, self.train_loader, optimizer, self.criterion, self.device)
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        
        metrics = evaluate(self.model, self.val_loader, self.criterion, self.device)
        
        return float(metrics.loss), len(self.val_loader.dataset), {"accuracy": float(metrics.accuracy)}
