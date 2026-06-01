from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from cifar_baseline.engine import (
    EvalMetrics,
    build_optimizer,
    build_scheduler,
    evaluate,
    train_one_epoch,
)
from cifar_baseline.model import ResNet18CIFAR


def pretrain(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    config: dict,
    device: torch.device,
) -> list[tuple[EvalMetrics, EvalMetrics]]:
    """Train server model from scratch on labeled server data.

    config must contain: epochs, learning_rate, optimizer, scheduler,
    momentum, weight_decay (and optionally label_smoothing).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config.get("label_smoothing", 0.0))
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Optional linear LR warmup: ramp from 0 → peak LR over warmup_epochs epochs.
    # Prevents the sharp-landscape instability that occurs at high LR on small datasets.
    warmup_epochs = int(config.get("warmup_epochs", 0))
    peak_lr = float(config["learning_rate"])

    history: list[tuple[EvalMetrics, EvalMetrics]] = []
    n = int(config["epochs"])
    for epoch in range(1, n + 1):
        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = peak_lr * epoch / warmup_epochs
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        history.append((train_m, val_m))
        print(
            f"  pretrain {epoch:02d}/{n}: "
            f"train_loss={train_m.loss:.4f} train_acc={train_m.accuracy:.4f} "
            f"val_loss={val_m.loss:.4f} val_acc={val_m.accuracy:.4f}",
            flush=True,
        )
    return history


@torch.no_grad()
def make_soft_labels(
    client_states: list[dict],
    distill_loader: DataLoader,
    *,
    temperature: float,
    device: torch.device,
    num_classes: int = 10,
) -> list[torch.Tensor]:
    """Average softmax outputs of all client models over the distillation set.

    distill_loader must have shuffle=False so that the returned list of
    per-batch tensors matches the loader's iteration order in distill_step.
    Returns CPU tensors (one per batch) to free GPU memory between rounds.
    """
    clients: list[nn.Module] = []
    for state in client_states:
        m = ResNet18CIFAR(num_classes=num_classes)
        m.load_state_dict(state)
        m.eval().to(device)
        clients.append(m)

    soft_batches: list[torch.Tensor] = []
    for images in distill_loader:
        images = images.to(device, non_blocking=True)
        avg = torch.zeros(images.size(0), num_classes, device=device)
        for c in clients:
            avg += F.softmax(c(images) / temperature, dim=1)
        avg /= len(clients)
        soft_batches.append(avg.cpu())

    return soft_batches


def distill_step(
    model: nn.Module,
    distill_loader: DataLoader,
    soft_batches: list[torch.Tensor],
    *,
    config: dict,
    device: torch.device,
    server_loader: DataLoader | None = None,
) -> list[float]:
    """Fine-tune server model on ensemble soft labels via KL divergence.

    Loss = alpha * T² · KL( softmax(student/T) ‖ soft_labels )
         + (1 - alpha) * CE( student, server_labeled_batch )

    The CE term on the server's own labeled data prevents catastrophic forgetting:
    without it the KL objective overwrites previously learned knowledge each round.
    Set config['distill_alpha'] = 1.0 to disable (pure KL, original behaviour).
    """
    model.to(device).train()
    optimizer = build_optimizer(model, config)
    temperature = float(config["temperature"])
    alpha = float(config.get("distill_alpha", 1.0))
    epoch_losses: list[float] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        total_loss, n_batches = 0.0, 0
        server_iter = iter(server_loader) if server_loader is not None else None

        for images, soft_target in zip(distill_loader, soft_batches):
            images = images.to(device, non_blocking=True)
            soft_target = soft_target.to(device)
            optimizer.zero_grad(set_to_none=True)

            log_probs = F.log_softmax(model(images) / temperature, dim=1)
            kl_loss = F.kl_div(log_probs, soft_target, reduction="batchmean") * (temperature ** 2)
            loss = alpha * kl_loss

            if server_iter is not None:
                try:
                    srv_imgs, srv_labels = next(server_iter)
                except StopIteration:
                    server_iter = iter(server_loader)
                    srv_imgs, srv_labels = next(server_iter)
                srv_imgs = srv_imgs.to(device, non_blocking=True)
                srv_labels = srv_labels.to(device, non_blocking=True)
                ce_loss = F.cross_entropy(model(srv_imgs), srv_labels)
                loss = loss + (1.0 - alpha) * ce_loss

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        epoch_losses.append(total_loss / max(n_batches, 1))
        print(f"    distill epoch {epoch}: loss={epoch_losses[-1]:.4f}", flush=True)

    return epoch_losses
