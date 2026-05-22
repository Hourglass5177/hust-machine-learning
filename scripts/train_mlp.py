from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml

from models.mlp import build_mlp
from utils.data import get_cifar10_loaders
from utils.seed import seed_everything
from utils.trainer import evaluate, save_checkpoint, train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP baseline on CIFAR-10")

    parser.add_argument("--config", type=str, default="configs/cifar10.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def plot_curves(history, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(history)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "mlp_loss_curve.png", dpi=200)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy")
    plt.plot(df["epoch"], df["test_acc"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MLP Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "mlp_accuracy_curve.png", dpi=200)
    plt.close()


def main():
    args = parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["project"]["seed"])
    seed_everything(seed, deterministic=False)

    data_root = cfg["data"]["root"]
    batch_size = args.batch_size or int(cfg["data"]["batch_size"])
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["data"]["num_workers"])
    pin_memory = bool(cfg["data"]["pin_memory"])

    epochs = args.epochs or int(cfg["train"]["epochs"])
    lr = args.lr or float(cfg["train"]["learning_rate"])

    if args.device is not None:
        device_name = args.device
    else:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_name)

    print("========== Train MLP Baseline ==========")
    print(f"Config: {config_path}")
    print(f"Data root: {data_root}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Num workers: {num_workers}")
    print(f"Use augmentation: {not args.no_aug}")

    train_loader, test_loader, idx_to_class = get_cifar10_loaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_augmentation=not args.no_aug,
        seed=seed,
    )

    model = build_mlp(num_classes=len(idx_to_class)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    output_root = Path(cfg["outputs"]["root"])
    ckpt_dir = Path(cfg["outputs"]["checkpoints"])
    log_dir = Path(cfg["outputs"]["logs"])
    fig_dir = Path(cfg["outputs"]["figures"])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            split="Test",
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.4f} "
            f"test_loss={row['test_loss']:.4f} "
            f"test_acc={row['test_acc']:.4f}"
        )

        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            save_checkpoint(
                save_path=ckpt_dir / "mlp_best.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                history=history,
            )
            print(f"Saved best checkpoint: {ckpt_dir / 'mlp_best.pth'}")

        pd.DataFrame(history).to_csv(log_dir / "mlp_history.csv", index=False)
        plot_curves(history, fig_dir)

    print("========== Training Finished ==========")
    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Log saved to: {(log_dir / 'mlp_history.csv').resolve()}")
    print(f"Figures saved to: {fig_dir.resolve()}")
    print(f"Best checkpoint saved to: {(ckpt_dir / 'mlp_best.pth').resolve()}")


if __name__ == "__main__":
    main()