from __future__ import annotations

import numpy as np
import torch


CIFAR10_MEAN = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.asarray([0.2470, 0.2435, 0.2616], dtype=np.float32)


class CIFARTransform:
    """Numpy/PyTorch transform stack used instead of torchvision."""

    def __init__(
        self,
        *,
        train: bool,
        crop_padding: int = 4,
        horizontal_flip: bool = True,
        seed: int | None = None,
    ) -> None:
        self.train = train
        self.crop_padding = crop_padding
        self.horizontal_flip = horizontal_flip
        self.rng = np.random.default_rng(seed)

    def _random_crop(self, image: np.ndarray) -> np.ndarray:
        if self.crop_padding <= 0:
            return image
        padded = np.pad(
            image,
            ((self.crop_padding, self.crop_padding), (self.crop_padding, self.crop_padding), (0, 0)),
            mode="reflect",
        )
        y = self.rng.integers(0, 2 * self.crop_padding + 1)
        x = self.rng.integers(0, 2 * self.crop_padding + 1)
        return padded[y : y + 32, x : x + 32, :]

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        if self.train:
            image = self._random_crop(image)
            if self.horizontal_flip and self.rng.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1, :])

        image = (image - CIFAR10_MEAN) / CIFAR10_STD
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image, dtype=torch.float32)
