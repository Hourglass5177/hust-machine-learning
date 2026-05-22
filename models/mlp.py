import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    MLP baseline for CIFAR-10 classification.

    Input image shape: [B, 3, 32, 32]
    Flattened feature dimension: 3 * 32 * 32 = 3072
    Output: [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dims=(1024, 512, 256),
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        logits = self.net(x)
        return logits


def build_mlp(num_classes: int = 10) -> MLPClassifier:
    return MLPClassifier(num_classes=num_classes)
