from typing import Tuple

import torch


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    """
    Return number of correct predictions and total samples.
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total
