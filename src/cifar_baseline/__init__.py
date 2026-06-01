"""Centralized CIFAR-10 ResNet-18 baseline for the federated-learning project."""

from cifar_baseline.cifar10 import CIFAR10_CLASSES, CIFAR10Dataset
from cifar_baseline.model import ResNet18CIFAR

__all__ = ["CIFAR10Dataset", "CIFAR10_CLASSES", "ResNet18CIFAR"]
