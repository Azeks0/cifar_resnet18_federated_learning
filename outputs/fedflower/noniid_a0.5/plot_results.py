from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"
 
def plot_history(history_file=RESULTS_DIR / "history.csv"):
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(history_file)

        has_train = "train_accuracy" in df.columns
        x_col = "epoch" if "epoch" in df.columns else "round"
        x_label = "Epoch" if has_train else "Round"

        fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=150)
        if has_train:
            ax.plot(df[x_col], df["train_accuracy"], marker="o", label="Train")
        ax.plot(df[x_col], df["val_accuracy"], marker="o", label="Validation")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Accuracy")
        ax.set_title("FedFlower CIFAR-10 Accuracy")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "accuracy_curve.png", bbox_inches="tight")
        print("Saved accuracy_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=150)
        if has_train:
            ax.plot(df[x_col], df["train_loss"], marker="o", label="Train")
        ax.plot(df[x_col], df["val_loss"], marker="o", label="Validation")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title("FedFlower CIFAR-10 Loss")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "loss_curve.png", bbox_inches="tight")
        print("Saved loss_curve.png")
        plt.close(fig)
        
    except FileNotFoundError:
        print(f"Could not find '{history_file}'.")
 
def plot_confusion_matrix(cm_file=RESULTS_DIR / "confusion_matrix.csv"):
    try:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        cm = pd.read_csv(cm_file, header=None).values

        classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=150)
        image = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Test Confusion Matrix")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "confusion_matrix.png", bbox_inches="tight")
        print("Saved confusion_matrix.png")
        plt.close(fig)
 
    except FileNotFoundError:
        print(f"Could not find '{cm_file}'.")
 
if __name__ == "__main__":
    print("Generăm graficele...")
    plot_history()
    plot_confusion_matrix()