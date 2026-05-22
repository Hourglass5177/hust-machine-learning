from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import yaml

from utils.data import CIFAR10_MEAN, CIFAR10_STD, get_cifar10_loaders
from utils.seed import seed_everything


def denormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img = img.cpu() * std + mean
    return img.clamp(0, 1)


def save_sample_grid(images, labels, idx_to_class, save_path: Path, n: int = 16) -> None:
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue

        img = denormalize(images[i])
        img = img.permute(1, 2, 0).numpy()

        label = labels[i].item()
        class_name = idx_to_class[label]

        ax.imshow(img)
        ax.set_title(class_name, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    config_path = Path("configs/cifar10.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["project"]["seed"])
    seed_everything(seed, deterministic=False)

    data_root = cfg["data"]["root"]
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])
    pin_memory = bool(cfg["data"]["pin_memory"])

    train_loader, test_loader, idx_to_class = get_cifar10_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_augmentation=True,
        seed=seed,
    )

    print("========== CIFAR-10 Data Pipeline Check ==========")
    print(f"Data root: {data_root}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Pin memory: {pin_memory}")
    print(f"Classes: {idx_to_class}")

    images, labels = next(iter(train_loader))

    print("Image batch shape:", tuple(images.shape))
    print("Label batch shape:", tuple(labels.shape))
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)
    print("Image min/max after normalization:", float(images.min()), float(images.max()))

    fig_dir = Path(cfg["outputs"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    save_path = fig_dir / "cifar10_samples.png"
    save_sample_grid(images, labels, idx_to_class, save_path)

    print(f"Saved sample grid to: {save_path.resolve()}")
    print("Data pipeline check passed.")


if __name__ == "__main__":
    main()
