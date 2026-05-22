import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    Fix random seeds for Python, NumPy and PyTorch.

    deterministic=False is recommended for normal training speed.
    deterministic=True can improve reproducibility but may slow down CUDA training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    """
    Make DataLoader workers reproducible.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
