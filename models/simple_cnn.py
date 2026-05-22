import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv-BN-ReLU block for CIFAR-10.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    CNN baseline for CIFAR-10 classification.

    Input:  [B, 3, 32, 32]
    Output: [B, 10]
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # 32x32 -> 32x32
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            nn.Dropout(dropout),

            # 16x16 -> 16x16
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            nn.Dropout(dropout),

            # 8x8 -> 8x8
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def build_simple_cnn(num_classes: int = 10) -> SimpleCNN:
    return SimpleCNN(num_classes=num_classes)
