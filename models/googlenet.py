import torch
from torch import nn


class Inception(nn.Module):
    # c2, c3: (số kênh 1x1 giảm chiều, số kênh conv 3x3 / 5x5)
    def __init__(self, in_channels: int, c1: int, c2: tuple[int, int], c3: tuple[int, int], c4: int):
        super().__init__()

        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)],
            dim=1,
        )


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 21, in_channels: int = 3, dropout: float = 0.4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # (64, 112, 112)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),           # (64, 56, 56)
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),                    # (192, 56, 56)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),           # (192, 28, 28)
        )

        self.inception_3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),            # 3a: (256, 28, 28)
            Inception(256, 128, (128, 192), (32, 96), 64),          # 3b: (480, 28, 28)
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # (480, 14, 14)
        )

        self.inception_4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),           # 4a: (512, 14, 14)
            Inception(512, 160, (112, 224), (24, 64), 64),          # 4b: (512, 14, 14)
            Inception(512, 128, (128, 256), (24, 64), 64),          # 4c: (512, 14, 14)
            Inception(512, 112, (144, 288), (32, 64), 64),          # 4d: (528, 14, 14)
            Inception(528, 256, (160, 320), (32, 128), 128),        # 4e: (832, 14, 14)
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),  # (832, 7, 7)
        )

        self.inception_5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),        # 5a: (832, 7, 7)
            Inception(832, 384, (192, 384), (48, 128), 128),        # 5b: (1024, 7, 7)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (1024, 1, 1)
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (bs, 3, 224, 224)
        features = self.stem(images)
        features = self.inception_3(features)
        features = self.inception_4(features)
        features = self.inception_5(features)
        return self.classifier(features)  # (bs, num_classes)
