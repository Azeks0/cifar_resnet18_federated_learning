from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cifar_baseline.cifar10 import CIFAR10_CLASSES


def plot_history(history: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=150)
    ax.plot(history["epoch"], history["train_accuracy"], marker="o", label="Train")
    ax.plot(history["epoch"], history["val_accuracy"], marker="o", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("ResNet-18 CIFAR-10 Accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curve.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=150)
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Train")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("ResNet-18 CIFAR-10 Loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(CIFAR10_CLASSES)))
    ax.set_yticks(np.arange(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
