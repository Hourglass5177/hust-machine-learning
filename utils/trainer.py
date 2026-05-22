from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.metrics import accuracy_from_logits


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        correct, total = accuracy_from_logits(logits, targets)

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += total

        progress.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "acc": f"{total_correct / total_samples:.4f}",
        })

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    epoch: int,
    split: str = "Test",
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=f"{split} Epoch {epoch}", leave=False)

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        correct, total = accuracy_from_logits(logits, targets)

        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += total

        progress.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "acc": f"{total_correct / total_samples:.4f}",
        })

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
    }


def save_checkpoint(
    save_path: str | Path,
    model: nn.Module,
    optimizer,
    epoch: int,
    best_acc: float,
    history: List[Dict],
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "history": history,
    }

    torch.save(checkpoint, save_path)
