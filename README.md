# CIFAR-10 ResNet-18 Centralized Baseline

This folder contains the baseline-only part of the federated-learning project. The goal is to train a full-precision centralized classifier on CIFAR-10 so later federated strategies can be compared against a clean reference.

The implementation avoids `torchvision` because it is not installed in the current environment. It includes:

- official CIFAR-10 download and pickle parsing
- training and evaluation transforms
- CIFAR-style ResNet-18
- centralized training loop with validation checkpointing
- accuracy/loss curves, confusion matrix, CSV/JSON summaries
- a Word report builder

## Structure

```text
cifar_resnet18_baseline/
  configs/default.json
  scripts/run_baseline.py
  scripts/build_paper.py
  src/cifar_baseline/
  outputs/
    results/
    plots/
  paper/
```

## Run the Baseline

From this directory:

```bash
PYTHONPATH=src python3 scripts/run_baseline.py --config configs/default.json
```

The default config downloads CIFAR-10 into `data/`, trains ResNet-18 for 30 epochs, and writes:

- `outputs/results/history.csv`
- `outputs/results/summary.csv`
- `outputs/results/summary.json`
- `outputs/results/confusion_matrix.csv`
- `outputs/plots/accuracy_curve.png`
- `outputs/plots/loss_curve.png`
- `outputs/plots/confusion_matrix.png`
- `outputs/checkpoints/best_resnet18_cifar10.pt`

For a quick smoke test:

```bash
PYTHONPATH=src python3 scripts/run_baseline.py --config configs/default.json --epochs 1 --learning-rate 0.001 --max-train-samples 512 --max-val-samples 256 --max-test-samples 256
```

## Build the Paper

```bash
/Users/matei/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_paper.py
```

The paper is written to `paper/resnet18_cifar10_baseline_report.docx`.

## Notes

- The baseline is centralized by design, matching Step 3 of the project handout.
- CIFAR-10 data and model checkpoints are ignored by git to avoid committing large generated artifacts.
- If `torchvision` is available later, the project can still run as-is; it simply does not require it.
