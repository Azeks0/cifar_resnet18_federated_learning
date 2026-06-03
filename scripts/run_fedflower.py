from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import torch
from torch import nn
from torch.utils.data import Subset

import ray
import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server.server_config import ServerConfig

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
from feddf.split import dirichlet_split, iid_split
from fedavg.client import CIFARClient


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower FedAdam strategy on CIFAR-10.")
    parser.add_argument("--config", type=Path, default=Path("configs/fedflower.json"))
    parser.add_argument("--iid", dest="iid", action="store_true", default=False,
                        help="Use IID client split (default: non-IID Dirichlet).")
    parser.add_argument("--non-iid", dest="iid", action="store_false",
                        help="Use non-IID Dirichlet split.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Dirichlet alpha for non-IID split (overrides config).")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override number of communication rounds.")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="CUDA device index to use for training.")
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
    device = choose_device(cuda_device=args.gpu_id)
    download = bool(config.get("download", True)) and not args.no_download
    use_gpu = device.type == "cuda"
    if args.gpu_id is not None and not use_gpu:
        print("WARNING: GPU requested but CUDA is not available in this environment. Falling back to CPU.")

    print(f"Device: {device}  |  Split: {split_tag}")

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

    server_model = ResNet18CIFAR(num_classes=int(config["model"]["num_classes"]))
    criterion = nn.CrossEntropyLoss()
    server_model.to(device)

    initial_metrics = evaluate(server_model, val_loader, criterion, device)
    history_records: list[dict[str, object]] = [
        {
            "round": 0,
            "val_loss": initial_metrics.loss,
            "val_accuracy": initial_metrics.accuracy,
        }
    ]

    initial_ndarrays = [val.cpu().numpy() for _, val in server_model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)
    final_parameters: dict[str, object] = {"parameters": initial_parameters}

    def evaluate_fn(server_round: int, parameters, config_str=None):
        keys = list(server_model.state_dict().keys())
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
        server_model.load_state_dict(state_dict)
        server_model.to(device)
        metrics = evaluate(server_model, val_loader, criterion, device)
        final_parameters["parameters"] = parameters
        history_records.append(
            {
                "round": server_round,
                "val_loss": metrics.loss,
                "val_accuracy": float(metrics.accuracy),
            }
        )
        return float(metrics.loss), {"accuracy": float(metrics.accuracy)}

    StrategyCls = getattr(fl.server.strategy, "FedAdam", None)
    if StrategyCls is None:
        raise RuntimeError("Flower does not expose a FedAdam strategy in this environment.")

    strategy = StrategyCls(
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
    )

    def client_fn(context: Context) -> fl.client.Client:
        idx = int(context.client_id)
        model = ResNet18CIFAR(num_classes=int(config["model"]["num_classes"]))
        numpy_client = CIFARClient(
            model=model,
            train_loader=client_loaders[idx],
            val_loader=val_loader,
            config=config["client"],
            device=device,
        )
        return numpy_client.to_client()

    print("\n=== Starting Flower simulation with FedAdam strategy ===")
    t0 = time.perf_counter()
    client_resources = {"num_cpus": 1, "num_gpus": 1.0} if use_gpu else {"num_cpus": 1, "num_gpus": 0.0}
    
    # Initialize Ray with runtime environment that includes project root in PYTHONPATH
    if not ray.is_initialized():
        import os
        current_python_path = os.environ.get("PYTHONPATH", "")
        new_python_path = f"{str(SRC_ROOT)}{os.pathsep}{current_python_path}".rstrip(os.pathsep)
        
        ray.init(
            runtime_env={
                "working_dir": str(PROJECT_ROOT),
                "env_vars": {"PYTHONPATH": new_python_path},
            },
            ignore_reinit_error=True,
        )
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=int(config["federated"]["rounds"])),
        strategy=strategy,
    )
    elapsed = time.perf_counter() - t0
    print(f"Flower simulation finished in {elapsed:.1f}s")
    ray.shutdown()

    if final_parameters["parameters"] is not None:
        keys = list(server_model.state_dict().keys())
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, final_parameters["parameters"])})
        server_model.load_state_dict(state_dict)

    server_model.to(device)
    val_metrics = evaluate(server_model, val_loader, criterion, device)
    test_metrics = evaluate(server_model, test_loader, criterion, device)
    matrix = confusion_matrix(server_model, test_loader, device=device, num_classes=int(config["model"]["num_classes"]))

    results_dir.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    history_df = pd.DataFrame(history_records)
    history_df.to_csv(results_dir / "history.csv", index=False)
    plot_history(history_df.rename(columns={"round": "epoch"}), plots_dir)

    summary = {
        "model": config["model"]["name"],
        "method": f"fedadam_{split_tag}",
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
        "elapsed_sec": elapsed,
    }

    save_json(summary, results_dir / "summary.json")
    pd.DataFrame([summary]).to_csv(results_dir / "summary.csv", index=False)
    pd.DataFrame(matrix).to_csv(results_dir / "confusion_matrix.csv", index=False, header=False)
    plot_confusion(matrix, plots_dir)

    print("\nFinal Flower summary:")
    print(pd.DataFrame([summary]).to_string(index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)))
    print(f"\nArtifacts written to {output_dir}")


if __name__ == "__main__":
    main()
