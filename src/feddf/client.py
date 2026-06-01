from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

from cifar_baseline.engine import EvalMetrics, build_optimizer, train_one_epoch
from cifar_baseline.model import ResNet18CIFAR


def local_train(
    server_state: dict,
    train_loader: DataLoader,
    *,
    config: dict,
    device: torch.device,
    num_classes: int = 10,
) -> tuple[dict, list[EvalMetrics]]:
    """Initialise a client model from server weights and fine-tune on local data.

    Returns the updated state dict and per-epoch train metrics.
    The server state is deep-copied so the original is never mutated.
    """
    model = ResNet18CIFAR(num_classes=num_classes)
    model.load_state_dict(copy.deepcopy(server_state))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    metrics: list[EvalMetrics] = []
    for _ in range(int(config["epochs"])):
        m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics.append(m)

    return model.state_dict(), metrics
