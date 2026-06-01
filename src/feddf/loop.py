from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from cifar_baseline.cifar10 import CIFAR10_CLASSES
from cifar_baseline.engine import EvalMetrics, evaluate, make_loader
from cifar_baseline.model import ResNet18CIFAR
from feddf.client import local_train
from feddf.server import distill_step, make_soft_labels, pretrain


@dataclass(frozen=True)
class RoundMetrics:
    round: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    round_time_sec: float


def run_feddf(
    server_model: nn.Module,
    server_train_loader: DataLoader,
    server_eval_loader: DataLoader,
    val_loader: DataLoader,
    client_datasets: list,
    distill_loader: DataLoader,
    *,
    config: dict,
    output_dir: Path,
    device: torch.device,
    seed: int,
) -> tuple[list[RoundMetrics], dict]:
    """Full FedDF training loop: server pre-train → communication rounds.

    Each round:
      1. Broadcast server weights to all clients.
      2. Each client fine-tunes locally for config['client']['epochs'] epochs.
      3. Server generates ensemble soft labels on the distillation set.
      4. Server fine-tunes on those soft labels for config['server_distill']['epochs'] epochs.
      5. Checkpoint if val accuracy improved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_feddf.pt"

    client_cfg = config["client"]
    distill_cfg = config["server_distill"]
    num_classes = int(config["model"]["num_classes"])
    n_rounds = int(config["federated"]["rounds"])
    temperature = float(distill_cfg["temperature"])
    criterion = nn.CrossEntropyLoss()

    # ── server pre-training ───────────────────────────────────────────────────
    print("\n=== Server pre-training ===")
    pretrain(
        server_model,
        server_train_loader,
        val_loader,
        config=config["server_pretrain"],
        device=device,
    )

    best_val_acc = -1.0
    best_round = 0
    history: list[RoundMetrics] = []

    # ── communication rounds ──────────────────────────────────────────────────
    for rnd in range(1, n_rounds + 1):
        t0 = time.perf_counter()
        print(f"\n=== Round {rnd}/{n_rounds} ===")

        server_state = server_model.state_dict()

        # Step 1 + 2: broadcast and local training
        client_states: list[dict] = []
        for cid, client_ds in enumerate(client_datasets):
            loader = make_loader(
                client_ds,
                batch_size=int(client_cfg["batch_size"]),
                shuffle=True,
                num_workers=int(client_cfg["num_workers"]),
                seed=seed + rnd * 1000 + cid,
            )
            print(f"  [client {cid + 1}/{len(client_datasets)}] {len(client_ds)} samples", flush=True)
            state, _ = local_train(
                server_state,
                loader,
                config=client_cfg,
                device=device,
                num_classes=num_classes,
            )
            client_states.append(state)

        # Step 3: ensemble soft labels
        print("  [server] generating soft labels ...", flush=True)
        soft_batches = make_soft_labels(
            client_states,
            distill_loader,
            temperature=temperature,
            device=device,
            num_classes=num_classes,
        )

        # Step 4: distillation fine-tune (mixed KL + CE to prevent forgetting)
        print("  [server] distillation fine-tuning ...", flush=True)
        distill_step(
            server_model,
            distill_loader,
            soft_batches,
            config=distill_cfg,
            device=device,
            server_loader=server_train_loader,
        )

        # Evaluate
        train_m = evaluate(server_model, server_eval_loader, criterion, device)
        val_m = evaluate(server_model, val_loader, criterion, device)
        elapsed = time.perf_counter() - t0

        row = RoundMetrics(
            round=rnd,
            train_loss=train_m.loss,
            train_accuracy=train_m.accuracy,
            val_loss=val_m.loss,
            val_accuracy=val_m.accuracy,
            round_time_sec=elapsed,
        )
        history.append(row)
        print(
            f"  train_loss={train_m.loss:.4f} train_acc={train_m.accuracy:.4f} "
            f"val_loss={val_m.loss:.4f} val_acc={val_m.accuracy:.4f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if val_m.accuracy > best_val_acc:
            best_val_acc = val_m.accuracy
            best_round = rnd
            torch.save(
                {
                    "round": rnd,
                    "model_state_dict": server_model.state_dict(),
                    "class_names": CIFAR10_CLASSES,
                    "config": config,
                },
                best_path,
            )

    return history, {
        "best_round": best_round,
        "best_val_accuracy": best_val_acc,
        "best_checkpoint": str(best_path),
    }
