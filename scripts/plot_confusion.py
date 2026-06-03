from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
from cifar_baseline.plotting import plot_confusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a confusion matrix CSV for CIFAR-10.")
    parser.add_argument("--confusion-csv", type=Path, required=True, help="Path to confusion_matrix.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "plots", help="Directory to save confusion_matrix.png")
    args = parser.parse_args()

    if not args.confusion_csv.exists():
        raise FileNotFoundError(f"Confusion CSV not found: {args.confusion_csv}")

    matrix = pd.read_csv(args.confusion_csv, header=None).to_numpy()
    plot_confusion(matrix, args.output_dir)
    print(f"Saved confusion matrix plot to {args.output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
