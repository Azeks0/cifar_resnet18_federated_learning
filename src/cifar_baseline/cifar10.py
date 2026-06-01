from __future__ import annotations

import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


CIFAR10_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-py"
CIFAR10_CLASSES = [
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


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_cifar10(root: str | Path, *, force: bool = False) -> Path:
    """Download and extract the official CIFAR-10 Python archive."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / CIFAR10_ARCHIVE
    extracted = root / CIFAR10_FOLDER

    if extracted.exists() and not force:
        return extracted

    if force or not archive.exists() or _md5(archive) != CIFAR10_MD5:
        print(f"Downloading CIFAR-10 from {CIFAR10_URL}")
        urllib.request.urlretrieve(CIFAR10_URL, archive)

    checksum = _md5(archive)
    if checksum != CIFAR10_MD5:
        raise RuntimeError(f"CIFAR-10 archive checksum mismatch: expected {CIFAR10_MD5}, got {checksum}")

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    return extracted


def cifar10_available(root: str | Path) -> bool:
    root = Path(root)
    return (root / CIFAR10_FOLDER / "data_batch_1").exists() and (root / CIFAR10_FOLDER / "test_batch").exists()


def _load_batch(path: Path) -> tuple[np.ndarray, list[int]]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    data = batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch["labels"]
    return data.astype(np.uint8), labels


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset reader that does not require torchvision."""

    def __init__(
        self,
        root: str | Path,
        *,
        train: bool,
        transform=None,
        download: bool = False,
    ) -> None:
        root = Path(root)
        if download:
            download_cifar10(root)
        if not cifar10_available(root):
            raise FileNotFoundError(
                f"CIFAR-10 was not found under {root}. Run with download enabled or place "
                f"{CIFAR10_ARCHIVE} there and extract it."
            )

        folder = root / CIFAR10_FOLDER
        if train:
            data_parts: list[np.ndarray] = []
            label_parts: list[int] = []
            for batch_id in range(1, 6):
                data, labels = _load_batch(folder / f"data_batch_{batch_id}")
                data_parts.append(data)
                label_parts.extend(labels)
            self.data = np.concatenate(data_parts, axis=0)
            self.targets = np.asarray(label_parts, dtype=np.int64)
        else:
            data, labels = _load_batch(folder / "test_batch")
            self.data = data
            self.targets = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int):
        image = self.data[index]
        label = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def train_val_indices(
    targets: np.ndarray,
    *,
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic stratified train/validation split."""

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    labels = np.asarray(targets)
    classes = sorted(np.unique(labels))
    per_class_val = val_size // len(classes)

    for class_id in classes:
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        val_indices.extend(class_indices[:per_class_val].tolist())
        train_indices.extend(class_indices[per_class_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train_indices = train_indices[:train_size]
    val_indices = val_indices[:val_size]
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)
