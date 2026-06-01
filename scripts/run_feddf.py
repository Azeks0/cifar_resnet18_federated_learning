from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset

from cifar_baseline.cifar10 import CIFAR10Dataset, train_val_indices
from cifar_baseline.engine import (
    choose_device,
    confusion_matrix,
    evaluate,
    make_loader,
    save_json,
    seed_everything,
)
from cifar_baseline.model import ResNet18CIFAR
from cifar_baseline.plotting import plot_confusion, plot_history
from cifar_baseline.transforms import CIFARTransform
from feddf.loop import run_feddf
from feddf.split import dirichlet_split, iid_split
from feddf.stl10 import STL10Unlabeled


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="FedDF: federated distillation on CIFAR-10.")
    parser.add_argument("--config", type=Path, default=Path("configs/feddf.json"))
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
    stl10_dir = resolve(project_root, config["stl10_dir"])
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

    # Identical val split to baseline (same function, same seed, same sizes)
    train_idx, val_idx = train_val_indices(
        full_train_eval.targets,
        train_size=45000,
        val_size=int(config["dataset"]["val_size"]),
        seed=seed,
    )
    val_set = Subset(full_train_eval, val_idx.tolist())

    # Partition the 45k training pool into server + client data
    rng = np.random.default_rng(seed)
    pool = train_idx.copy()
    rng.shuffle(pool)
    server_size = int(config["dataset"]["server_size"])
    server_idx = pool[:server_size]
    client_pool_idx = pool[server_size:]  # 40k indices for clients

    server_train_aug = Subset(full_train_aug, server_idx.tolist())
    server_train_eval = Subset(full_train_eval, server_idx.tolist())

    # Client datasets — nested Subsets: outer indexes into client_pool, inner into full dataset
    client_pool_aug = Subset(full_train_aug, client_pool_idx.tolist())
    client_pool_targets = full_train_eval.targets[client_pool_idx]
    n_clients = int(config["dataset"]["n_clients"])

    if args.iid:
        client_datasets = iid_split(client_pool_aug, client_pool_targets, n_clients=n_clients, seed=seed)
    else:
        client_datasets = dirichlet_split(
            client_pool_aug, client_pool_targets, n_clients=n_clients, alpha=alpha, seed=seed
        )

    print(f"\nData split ({split_tag}):")
    print(f"  Server labeled : {len(server_train_aug):>6}")
    for i, ds in enumerate(client_datasets):
        print(f"  Client {i + 1}       : {len(ds):>6}")
    print(f"  Val            : {len(val_set):>6}")
    print(f"  Test           : {len(test_set):>6}")

    # ── STL-10 distillation set ───────────────────────────────────────────────
    distill_size = int(config["dataset"]["distillation_size"])
    stl10_full = STL10Unlabeled(stl10_dir, download=download)
    distill_idx = np.random.default_rng(seed + 999).choice(len(stl10_full), size=distill_size, replace=False)
    distill_set = Subset(stl10_full, distill_idx.tolist())
    print(f"  Distillation   : {distill_size:>6}  (STL-10 unlabeled)\n")

    # ── data loaders ─────────────────────────────────────────────────────────
    server_pre_cfg = config["server_pretrain"]
    distill_cfg = config["server_distill"]

    server_train_loader = make_loader(
        server_train_aug,
        batch_size=int(server_pre_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(server_pre_cfg["num_workers"]),
        seed=seed,
    )
    server_eval_loader = make_loader(
        server_train_eval,
        batch_size=int(server_pre_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(server_pre_cfg["num_workers"]),
        seed=seed,
    )
    val_loader = make_loader(
        val_set,
        batch_size=int(server_pre_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(server_pre_cfg["num_workers"]),
        seed=seed,
    )
    test_loader = make_loader(
        test_set,
        batch_size=int(server_pre_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(server_pre_cfg["num_workers"]),
        seed=seed,
    )
    # num_workers=0 for distillation: avoids loading 2.6 GB STL-10 into each worker process
    distill_loader = make_loader(
        distill_set,
        batch_size=int(distill_cfg["batch_size"]),
        shuffle=False,  # must be False: soft_batches must align with loader order
        num_workers=0,
        seed=seed,
    )

    # ── model + training ──────────────────────────────────────────────────────
    model = ResNet18CIFAR(num_classes=int(config["model"]["num_classes"]))

    history, best = run_feddf(
        model,
        server_train_loader,
        server_eval_loader,
        val_loader,
        client_datasets,
        distill_loader,
        config=config,
        output_dir=output_dir,
        device=device,
        seed=seed,
    )

    # ── save history ──────────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame([asdict(row) for row in history])
    history_df.to_csv(results_dir / "history.csv", index=False)

    # Rename 'round' → 'epoch' so plot_history works without modification
    plot_df = history_df.rename(columns={"round": "epoch", "round_time_sec": "epoch_time_sec"})
    plot_history(plot_df, plots_dir)

    # ── final evaluation on best checkpoint ───────────────────────────────────
    checkpoint = torch.load(best["best_checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    server_metrics = evaluate(model, server_eval_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    matrix = confusion_matrix(
        model, test_loader, device=device, num_classes=int(config["model"]["num_classes"])
    )

    summary = {
        "model": config["model"]["name"],
        "method": f"feddf_{split_tag}",
        "dataset": "CIFAR-10",
        "distillation_dataset": "STL-10 (unlabeled)",
        "device": str(device),
        "rounds": int(config["federated"]["rounds"]),
        "best_round": int(best["best_round"]),
        "n_clients": n_clients,
        "server_samples": server_size,
        "total_client_samples": sum(len(ds) for ds in client_datasets),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "distillation_samples": distill_size,
        "temperature": float(distill_cfg["temperature"]),
        "dirichlet_alpha": None if args.iid else alpha,
        "server_loss": server_metrics.loss,
        "server_accuracy": server_metrics.accuracy,
        "val_loss": val_metrics.loss,
        "val_accuracy": val_metrics.accuracy,
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
        "best_checkpoint": best["best_checkpoint"],
    }
    save_json(summary, results_dir / "summary.json")
    pd.DataFrame([summary]).to_csv(results_dir / "summary.csv", index=False)
    pd.DataFrame(matrix).to_csv(results_dir / "confusion_matrix.csv", index=False, header=False)
    plot_confusion(matrix, plots_dir)

    print("\nFinal FedDF summary:")
    print(pd.DataFrame([summary]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nArtifacts written to {output_dir}")


if __name__ == "__main__":
    main()
