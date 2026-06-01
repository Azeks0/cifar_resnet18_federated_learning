from __future__ import annotations

import argparse
import json
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
    fit,
    limit_subset,
    make_loader,
    save_history,
    save_json,
    seed_everything,
)
from cifar_baseline.model import ResNet18CIFAR
from cifar_baseline.plotting import plot_confusion, plot_history
from cifar_baseline.transforms import CIFARTransform


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a centralized ResNet-18 baseline on CIFAR-10.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for quick experiments.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional subset limit for smoke tests.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional validation subset limit.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional test subset limit.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate for smoke tests.")
    parser.add_argument("--no-download", action="store_true", help="Do not download CIFAR-10 automatically.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = read_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate

    data_dir = resolve(project_root, config["data_dir"])
    output_dir = resolve(project_root, config["output_dir"])
    results_dir = output_dir / "results"
    plots_dir = output_dir / "plots"
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device()

    train_transform = CIFARTransform(train=True, seed=seed)
    eval_transform = CIFARTransform(train=False, seed=seed)

    download = bool(config.get("download", True)) and not args.no_download
    full_train_aug = CIFAR10Dataset(data_dir, train=True, transform=train_transform, download=download)
    full_train_eval = CIFAR10Dataset(data_dir, train=True, transform=eval_transform, download=False)
    test_set = CIFAR10Dataset(data_dir, train=False, transform=eval_transform, download=False)

    train_idx, val_idx = train_val_indices(
        full_train_eval.targets,
        train_size=int(config["dataset"]["train_size"]),
        val_size=int(config["dataset"]["val_size"]),
        seed=seed,
    )
    train_set = Subset(full_train_aug, train_idx.tolist())
    val_set = Subset(full_train_eval, val_idx.tolist())
    train_eval_set = Subset(full_train_eval, train_idx.tolist())

    train_set = limit_subset(train_set, args.max_train_samples, seed=seed)
    val_set = limit_subset(val_set, args.max_val_samples, seed=seed + 1)
    train_eval_set = limit_subset(train_eval_set, args.max_train_samples, seed=seed)
    test_set = limit_subset(test_set, args.max_test_samples, seed=seed + 2)

    training = config["training"]
    train_loader = make_loader(
        train_set,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        seed=seed,
    )
    val_loader = make_loader(
        val_set,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=seed,
    )
    train_eval_loader = make_loader(
        train_eval_set,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=seed,
    )
    test_loader = make_loader(
        test_set,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
        seed=seed,
    )

    model = ResNet18CIFAR(num_classes=int(config["model"]["num_classes"]))
    history, best = fit(model, train_loader, val_loader, config=training, output_dir=output_dir, device=device)
    history_df = save_history(history, results_dir / "history.csv")

    checkpoint = torch.load(best["best_checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    train_metrics = evaluate(model, train_eval_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    matrix = confusion_matrix(model, test_loader, device=device, num_classes=int(config["model"]["num_classes"]))

    summary = {
        "model": config["model"]["name"],
        "dataset": "CIFAR-10",
        "device": str(device),
        "epochs": int(training["epochs"]),
        "best_epoch": int(best["best_epoch"]),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "train_loss": train_metrics.loss,
        "train_accuracy": train_metrics.accuracy,
        "val_loss": val_metrics.loss,
        "val_accuracy": val_metrics.accuracy,
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
        "best_checkpoint": best["best_checkpoint"],
    }
    save_json(summary, results_dir / "summary.json")
    pd.DataFrame([summary]).to_csv(results_dir / "summary.csv", index=False)
    pd.DataFrame(matrix).to_csv(results_dir / "confusion_matrix.csv", index=False, header=False)

    plot_history(history_df, plots_dir)
    plot_confusion(matrix, plots_dir)

    print("\nFinal baseline summary:")
    print(pd.DataFrame([summary]).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nArtifacts written to {output_dir}")


if __name__ == "__main__":
    main()
