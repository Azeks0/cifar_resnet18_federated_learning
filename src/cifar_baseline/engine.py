from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from cifar_baseline.cifar10 import CIFAR10_CLASSES


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    epoch_time_sec: float


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    accuracy: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(cuda_device: int | None = None) -> torch.device:
    if torch.cuda.is_available():
        if cuda_device is not None:
            return torch.device(f"cuda:{cuda_device}")
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def limit_subset(dataset, max_samples: int | None, *, seed: int):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(np.arange(len(dataset)), size=max_samples, replace=False)
    return Subset(dataset, indices.tolist())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> EvalMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += int(labels.size(0))
    return EvalMetrics(total_loss / total_samples, total_correct / total_samples)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += int(labels.size(0))
    return EvalMetrics(total_loss / total_samples, total_correct / total_samples)


@torch.no_grad()
def confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int = 10) -> np.ndarray:
    model.eval()
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = labels.numpy()
        for target, pred in zip(targets, preds):
            matrix[target, pred] += 1
    return matrix


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    name = config.get("optimizer", "sgd").lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(config["learning_rate"]),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 5e-4)),
            nesterov=True,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 1e-4)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    scheduler_name = config.get("scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))
    if scheduler_name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> tuple[list[EpochMetrics], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    model.to(device)

    best_val_accuracy = -1.0
    best_epoch = 0
    best_path = checkpoint_dir / "best_resnet18_cifar10.pt"
    history: list[EpochMetrics] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        elapsed = time.perf_counter() - start

        row = EpochMetrics(
            epoch=epoch,
            train_loss=train_metrics.loss,
            train_accuracy=train_metrics.accuracy,
            val_loss=val_metrics.loss,
            val_accuracy=val_metrics.accuracy,
            epoch_time_sec=elapsed,
        )
        history.append(row)
        print(
            f"epoch {epoch:03d}: train_loss={row.train_loss:.4f}, "
            f"train_acc={row.train_accuracy:.4f}, val_loss={row.val_loss:.4f}, "
            f"val_acc={row.val_accuracy:.4f}, time={row.epoch_time_sec:.1f}s",
            flush=True,
        )

        if row.val_accuracy > best_val_accuracy:
            best_val_accuracy = row.val_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": CIFAR10_CLASSES,
                    "config": config,
                },
                best_path,
            )

    return history, {"best_epoch": best_epoch, "best_val_accuracy": best_val_accuracy, "best_checkpoint": str(best_path)}


def save_history(history: list[EpochMetrics], path: Path) -> pd.DataFrame:
    dataframe = pd.DataFrame([asdict(row) for row in history])
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return dataframe


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
