from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.seed import seed_worker


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_transforms(use_augmentation: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Return train/test transforms for CIFAR-10.
    """
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, test_transform


def get_cifar10_datasets(
    data_root: str,
    use_augmentation: bool = True,
    download: bool = True,
):
    """
    Build CIFAR-10 train/test datasets.
    """
    data_root = Path(data_root).expanduser()

    train_transform, test_transform = get_cifar10_transforms(
        use_augmentation=use_augmentation
    )

    train_set = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=download,
        transform=train_transform,
    )

    test_set = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=download,
        transform=test_transform,
    )

    return train_set, test_set


def get_cifar10_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_augmentation: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Build CIFAR-10 train/test DataLoaders.
    """
    train_set, test_set = get_cifar10_datasets(
        data_root=data_root,
        use_augmentation=use_augmentation,
        download=True,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    class_to_idx = train_set.class_to_idx
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

    return train_loader, test_loader, idx_to_class
