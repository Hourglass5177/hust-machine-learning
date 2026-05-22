from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.mlp import build_mlp
from models.simple_cnn import build_simple_cnn
from utils.data import CIFAR10_MEAN, CIFAR10_STD, get_cifar10_loaders
from utils.seed import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on CIFAR-10")
    parser.add_argument("--config", type=str, default="configs/cifar10.yaml")
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "cnn"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-error-samples", type=int, default=25)
    return parser.parse_args()


def build_model(model_name: str, num_classes: int):
    if model_name == "mlp":
        return build_mlp(num_classes=num_classes)
    if model_name == "cnn":
        return build_simple_cnn(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def load_checkpoint(model, checkpoint_path: Path, device: torch.device):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        best_acc = checkpoint.get("best_acc", None)
        epoch = checkpoint.get("epoch", None)
    else:
        model.load_state_dict(checkpoint)
        best_acc = None
        epoch = None

    return model, best_acc, epoch


def denormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img = img.cpu() * std + mean
    return img.clamp(0, 1)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []
    all_images = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)

        y_true.append(targets.cpu())
        y_pred.append(preds.cpu())
        all_images.append(images.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    all_images = torch.cat(all_images, dim=0)

    return y_true, y_pred, all_images


def save_confusion_matrix(cm, class_names, save_path: Path, normalize: bool = False):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_to_plot = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_to_plot = np.nan_to_num(cm_to_plot)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_to_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_to_plot)

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    threshold = cm_to_plot.max() / 2.0
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            value = cm_to_plot[i, j]
            text = format(value, fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_error_samples(images, y_true, y_pred, class_names, save_path: Path, max_samples: int = 25):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    wrong_indices = np.where(y_true != y_pred)[0]
    if len(wrong_indices) == 0:
        print("No wrong samples found.")
        return

    selected = wrong_indices[:max_samples]

    cols = 5
    rows = int(np.ceil(len(selected) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.array(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(selected):
            ax.axis("off")
            continue

        idx = selected[ax_idx]
        img = denormalize(images[idx]).permute(1, 2, 0).numpy()

        true_name = class_names[y_true[idx]]
        pred_name = class_names[y_pred[idx]]

        ax.imshow(img)
        ax.set_title(f"T: {true_name}\nP: {pred_name}", fontsize=8)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_top_confusions(cm, class_names, save_path: Path, top_k: int = 20):
    rows = []

    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i == j:
                continue
            rows.append({
                "true_class": true_name,
                "pred_class": pred_name,
                "count": int(cm[i, j]),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("count", ascending=False).head(top_k)
    df.to_csv(save_path, index=False)


def main():
    args = parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["project"]["seed"])
    seed_everything(seed, deterministic=False)

    if args.device is not None:
        device_name = args.device
    else:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    data_root = cfg["data"]["root"]
    num_workers = args.num_workers if args.num_workers is not None else int(cfg["data"]["num_workers"])
    pin_memory = bool(cfg["data"]["pin_memory"])

    _, test_loader, idx_to_class = get_cifar10_loaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_augmentation=False,
        seed=seed,
    )

    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = build_model(args.model, num_classes=len(class_names)).to(device)
    checkpoint_path = Path(args.checkpoint)

    model, best_acc_from_ckpt, epoch_from_ckpt = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model.eval()

    print("========== Evaluate Model ==========")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {epoch_from_ckpt}")
    print(f"Checkpoint best_acc: {best_acc_from_ckpt}")

    y_true, y_pred, images = collect_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )

    results_dir = Path(cfg["outputs"]["results"])
    fig_dir = Path(cfg["outputs"]["figures"])
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    report_csv_path = results_dir / f"{args.model}_classification_report.csv"
    report_txt_path = results_dir / f"{args.model}_classification_report.txt"
    cm_csv_path = results_dir / f"{args.model}_confusion_matrix.csv"
    top_confusions_path = results_dir / f"{args.model}_top_confusions.csv"

    pd.DataFrame(report_dict).transpose().to_csv(report_csv_path)
    report_txt_path.write_text(report_text, encoding="utf-8")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_csv_path)
    save_top_confusions(cm, class_names, top_confusions_path)

    save_confusion_matrix(
        cm,
        class_names,
        fig_dir / f"{args.model}_confusion_matrix.png",
        normalize=False,
    )

    save_confusion_matrix(
        cm,
        class_names,
        fig_dir / f"{args.model}_confusion_matrix_normalized.png",
        normalize=True,
    )

    save_error_samples(
        images,
        y_true,
        y_pred,
        class_names,
        fig_dir / f"{args.model}_error_samples.png",
        max_samples=args.max_error_samples,
    )

    summary = {
        "model": args.model,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": epoch_from_ckpt,
        "checkpoint_best_acc": best_acc_from_ckpt,
        "evaluated_accuracy": acc,
        "num_test_samples": len(y_true),
    }

    pd.DataFrame([summary]).to_csv(results_dir / f"{args.model}_eval_summary.csv", index=False)

    print(report_text)
    print(f"Evaluated accuracy: {acc:.4f}")
    print(f"Saved report CSV: {report_csv_path.resolve()}")
    print(f"Saved report TXT: {report_txt_path.resolve()}")
    print(f"Saved confusion matrix CSV: {cm_csv_path.resolve()}")
    print(f"Saved top confusions CSV: {top_confusions_path.resolve()}")
    print(f"Saved figures to: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
