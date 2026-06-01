from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cifar_baseline.transforms import CIFAR10_MEAN, CIFAR10_STD

STL10_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
STL10_MD5 = "91f7769df0f17e558f3565bffb0c7dfb"
STL10_ARCHIVE = "stl10_binary.tar.gz"
STL10_FOLDER = "stl10_binary"
STL10_UNLABELED_FILE = "unlabeled_X.bin"
_RAW_C, _RAW_H, _RAW_W = 3, 96, 96
_OUT_SIZE = 32


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_stl10(root: str | Path, *, force: bool = False) -> Path:
    """Download and extract the STL-10 binary archive."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / STL10_ARCHIVE
    extracted = root / STL10_FOLDER

    if extracted.exists() and not force:
        return extracted

    if force or not archive.exists() or _md5(archive) != STL10_MD5:
        print(f"Downloading STL-10 from {STL10_URL}")
        urllib.request.urlretrieve(STL10_URL, archive)

    checksum = _md5(archive)
    if checksum != STL10_MD5:
        raise RuntimeError(
            f"STL-10 archive checksum mismatch: expected {STL10_MD5}, got {checksum}"
        )

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    return extracted


def stl10_available(root: str | Path) -> bool:
    return (Path(root) / STL10_FOLDER / STL10_UNLABELED_FILE).exists()


class STL10Unlabeled(Dataset):
    """STL-10 unlabeled split, resized to 32×32 with CIFAR-10 normalization.

    Images are stored as uint8 in memory and converted to float32 on access.
    Normalization uses CIFAR-10 statistics so client models (trained on CIFAR-10)
    operate in the same input space during distillation.
    """

    def __init__(self, root: str | Path, *, download: bool = False) -> None:
        root = Path(root)
        if download:
            download_stl10(root)
        if not stl10_available(root):
            raise FileNotFoundError(
                f"STL-10 unlabeled data not found under {root}. "
                "Pass download=True or place stl10_binary.tar.gz there and extract it."
            )
        path = root / STL10_FOLDER / STL10_UNLABELED_FILE
        # Binary format: N × (C × H × W), uint8, channel-first
        raw = np.fromfile(path, dtype=np.uint8)
        # Reshape to (N, C, H, W) then transpose to (N, H, W, C) for PIL compatibility
        self._data: np.ndarray = raw.reshape(-1, _RAW_C, _RAW_H, _RAW_W).transpose(0, 2, 3, 1)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.fromarray(self._data[index]).resize((_OUT_SIZE, _OUT_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - CIFAR10_MEAN) / CIFAR10_STD
        return torch.tensor(np.transpose(arr, (2, 0, 1)), dtype=torch.float32)
