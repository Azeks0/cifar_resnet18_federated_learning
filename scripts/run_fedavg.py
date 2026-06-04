from __future__ import annotations

import argparse
import copy
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset

from cifar_baseline.cifar10 import CIFAR10Dataset, train_val_indices
from cifar_baseline.engine import (
    build_optimizer,
    choose_device,
    confusion_matrix,
    evaluate,
    make_loader,
    save_json,
    seed_everything,
    train_one_epoch,
)
from cifar_baseline.model import ResNet18CIFAR
from cifar_baseline.plotting import plot_confusion, plot_history
from cifar_baseline.transforms import CIFARTransform
from feddf.split import dirichlet_split, iid_split


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="FedAvg: custom federated averaging on CIFAR-10.")
    parser.add_argument("--config", type=Path, default=Path("configs/fedavg.json"))
    parser.add_argument("--iid", dest="iid", action="store_true", default=False,
                        help="Use IID client split (default: non-IID Dirichlet).")
    parser.add_argument("--non-iid", dest="iid", action="store_false",
                        help="Use non-IID Dirichlet split.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Dirichlet alpha for non-IID split (overrides config).")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override number of communication rounds.")
    parser.add_argument("--no-download", action="store_true",
                        help="Do not download datasets automatically.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = read_config(args.config)
    if args.rounds is not None:
        config["federated"]["rounds"] = args.rounds
    if args.alpha is not None:
        config["dataset"]["dirichlet_alpha"] = args.alpha

    alpha = float(config["dataset"]["dirichlet_alpha"])
    split_tag = "iid" if args.iid else f"noniid_a{alpha}"
    data_dir = resolve(project_root, config["data_dir"])
    output_dir = resolve(project_root, config["output_dir"]) / split_tag
    results_dir = output_dir / "results"
    plots_dir = output_dir / "plots"

    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device()
    download = bool(config.get("download", True)) and not args.no_download

    print(f"Device: {device}  |  Split: {split_tag}")

    # ── CIFAR-10 ──────────────────────────────────────────────────────────────
    train_transform = CIFARTransform(train=True, seed=seed)
    eval_transform = CIFARTransform(train=False, seed=seed)

    full_train_aug = CIFAR10Dataset(data_dir, train=True, transform=train_transform, download=download)
    full_train_eval = CIFAR10Dataset(data_dir, train=True, transform=eval_transform, download=False)
    test_set = CIFAR10Dataset(data_dir, train=False, transform=eval_transform, download=False)

    train_idx, val_idx = train_val_indices(
        full_train_eval.targets,
        train_size=45000,
        val_size=int(config["dataset"]["val_size"]),
        seed=seed,
    )
    val_set = Subset(full_train_eval, val_idx.tolist())
    train_eval_set = Subset(full_train_eval, train_idx.tolist())
    
    train_eval_loader = make_loader(
        train_eval_set,
        batch_size=int(config["client"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["client"]["num_workers"]),
        seed=seed,
    )
    val_loader = make_loader(
        val_set,
        batch_size=int(config["client"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["client"]["num_workers"]),
        seed=seed,
    )
    
    test_loader = make_loader(
        test_set,
        batch_size=int(config["client"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["client"]["num_workers"]),
        seed=seed,
    )

    # For fedavg, distribute all 45k training data to clients
    client_pool_aug = Subset(full_train_aug, train_idx.tolist())
    client_pool_targets = full_train_eval.targets[train_idx]
    n_clients = int(config["dataset"]["n_clients"])

    if args.iid:
        client_datasets = iid_split(client_pool_aug, client_pool_targets, n_clients=n_clients, seed=seed)
    else:
        client_datasets = dirichlet_split(
            client_pool_aug, client_pool_targets, n_clients=n_clients, alpha=alpha, seed=seed
        )

    print(f"\nData split ({split_tag}):")
    for i, ds in enumerate(client_datasets):
        print(f"  Client {i + 1}       : {len(ds):>6}")
    print(f"  Val            : {len(val_set):>6}")
    print(f"  Test           : {len(test_set):>6}")

    # ── Prepare client data loaders ────────────────────────────────────────────
    client_loaders = []
    for cid_int in range(n_clients):
        loader = make_loader(
            client_datasets[cid_int],
            batch_size=int(config["client"]["batch_size"]),
            shuffle=True,
            num_workers=int(config["client"]["num_workers"]),
            seed=seed + cid_int,
        )
        client_loaders.append(loader)

    # Create server (global) model
    server_model = ResNet18CIFAR(num_classes=int(config["model"]["num_classes"]))
    
    history_records = []
    t0 = time.perf_counter()
    best_val_accuracy = -1.0
    best_checkpoint = ""
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_fedavg.pt"
    
    criterion = nn.CrossEntropyLoss()
    local_epochs = int(config["client"]["epochs"])
    num_rounds = int(config["federated"]["rounds"])

    # ── Pure-PyTorch FedAvg loop ─────────────────────────────────────────────
    print("\n=== Starting FedAvg Training ===")

    # Round 0: evaluate the initial (random) model
    server_model.to(device)
    train_metrics = evaluate(server_model, train_eval_loader, criterion, device)
    val_metrics = evaluate(server_model, val_loader, criterion, device)
    elapsed = time.perf_counter() - t0
    print(f"  Round 0 train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} time={elapsed:.1f}s")
    history_records.append({
        "epoch": 0,
        "train_loss": train_metrics.loss, "train_accuracy": train_metrics.accuracy,
        "val_loss": val_metrics.loss, "val_accuracy": val_metrics.accuracy,
        "epoch_time_sec": elapsed,
    })

    global_state = copy.deepcopy(server_model.state_dict())

    for rnd in range(1, num_rounds + 1):
        client_states = []
        client_sizes = []

        for cid_int in range(n_clients):
            # Clone global model for this client
            client_model = ResNet18CIFAR(num_classes=int(config["model"]["num_classes"]))
            client_model.load_state_dict(global_state)
            client_model.to(device)

            optimizer = build_optimizer(client_model, config["client"])

            for _ in range(local_epochs):
                train_one_epoch(client_model, client_loaders[cid_int], optimizer, criterion, device)

            client_states.append({k: v.cpu() for k, v in client_model.state_dict().items()})
            client_sizes.append(len(client_datasets[cid_int]))

        # ── FedAvg aggregation ────────────────────────────────────────────
        total_examples = sum(client_sizes)
        avg_state = OrderedDict()
        for key in global_state.keys():
            avg_state[key] = sum(
                client_states[i][key] * (client_sizes[i] / total_examples)
                for i in range(n_clients)
            )
        global_state = avg_state

        # ── Centralized evaluation ────────────────────────────────────────
        server_model.load_state_dict(global_state)
        server_model.to(device)
        train_metrics = evaluate(server_model, train_eval_loader, criterion, device)
        val_metrics = evaluate(server_model, val_loader, criterion, device)
        elapsed = time.perf_counter() - t0
        print(f"  Round {rnd} train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f} time={elapsed:.1f}s")

        history_records.append({
            "epoch": rnd,
            "train_loss": train_metrics.loss, "train_accuracy": train_metrics.accuracy,
            "val_loss": val_metrics.loss, "val_accuracy": val_metrics.accuracy,
            "epoch_time_sec": elapsed,
        })

        if val_metrics.accuracy > best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            torch.save(
                {"round": rnd, "model_state_dict": server_model.state_dict(), "config": config},
                best_path,
            )
            best_checkpoint = str(best_path)

    # ── save history ──────────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(results_dir / "history.csv", index=False)

    # Avoid plotting issues if no history
    if not history_df.empty:
        plot_history(history_df, plots_dir)

    # ── final evaluation on best checkpoint ───────────────────────────────────
    if best_checkpoint:
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        server_model.load_state_dict(checkpoint["model_state_dict"])
    
    val_metrics = evaluate(server_model, val_loader, criterion, device)
    test_metrics = evaluate(server_model, test_loader, criterion, device)
    matrix = confusion_matrix(
        server_model, test_loader, device=device, num_classes=int(config["model"]["num_classes"])
    )

    summary = {
        "model": config["model"]["name"],
        "method": f"fedavg_{split_tag}",
        "dataset": "CIFAR-10",
        "device": str(device),
        "rounds": int(config["federated"]["rounds"]),
        "n_clients": n_clients,
        "total_client_samples": len(train_idx),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "dirichlet_alpha": None if args.iid else alpha,
        "val_loss": val_metrics.loss,
        "val_accuracy": val_metrics.accuracy,
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
        "best_checkpoint": best_checkpoint,
    }
    save_json(summary, results_dir / "summary.json")
    pd.DataFrame([summary]).to_csv(results_dir / "summary.csv", index=False)
    pd.DataFrame(matrix).to_csv(results_dir / "confusion_matrix.csv", index=False, header=False)
    plot_confusion(matrix, plots_dir)

    print("\nFinal FedAvg summary:")
    print(pd.DataFrame([summary]).to_string(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)))
    print(f"\nArtifacts written to {output_dir}")

if __name__ == "__main__":
    main()
