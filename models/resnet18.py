import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        # dùng 1x1 conv ở nhánh tắt khi số kênh thay đổi
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn_1(self.conv_1(x)))
        y = self.bn_2(self.conv_2(y))
        return F.relu(y + self.shortcut(x))


class ResNet18(nn.Module):
    # Giữa các nhóm residual block có MaxPool(3, stride 2, padding 0) theo yêu cầu đề,
    # MaxPool này đã giảm kích thước nên các block giữ stride 1.
    def __init__(self, num_classes: int = 21, in_channels: int = 3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),  # (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                            # (64, 56, 56)
        )

        self.conv2_x = nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))        # (64, 56, 56)
        self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)                   # (64, 27, 27)
        self.conv3_x = nn.Sequential(ResidualBlock(64, 128), ResidualBlock(128, 128))     # (128, 27, 27)
        self.pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)                   # (128, 13, 13)
        self.conv4_x = nn.Sequential(ResidualBlock(128, 256), ResidualBlock(256, 256))    # (256, 13, 13)
        self.pool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)                   # (256, 6, 6)
        self.conv5_x = nn.Sequential(ResidualBlock(256, 512), ResidualBlock(512, 512))    # (512, 6, 6)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (512, 1, 1)
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (bs, 3, 224, 224)
        features = self.stem(images)
        features = self.pool_2(self.conv2_x(features))
        features = self.pool_3(self.conv3_x(features))
        features = self.pool_4(self.conv4_x(features))
        features = self.conv5_x(features)
        return self.classifier(features)  # (bs, num_classes)
