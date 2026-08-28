"""A small, readable implementation of ResNet-34.

The original tutorial mixes BasicBlock-specific code with the more general
Bottleneck/ResNeXt implementation.  This file keeps only the BasicBlock used
by the flower-classification example so that the data flow is easy to follow.
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """The two 3x3-convolution residual block used by ResNet-18/34."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        # When shape is unchanged, the skip path is just x. Otherwise a
        # 1x1 convolution performs the required projection and downsampling.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.activation(out + identity)
        return out


class ResNet(nn.Module):
    """ResNet classifier assembled from a sequence of residual stages."""

    def __init__(self, layers: tuple[int, int, int, int], num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channels = 64
        # Keep the conventional names (conv1/bn1/layer1...) so that an
        # official torchvision ResNet-34 state_dict can be reused directly.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.activation(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), start_dim=1)
        return self.fc(x)


def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet((2, 2, 2, 2), num_classes=num_classes)


def resnet34(num_classes: int = 1000) -> ResNet:
    return ResNet((3, 4, 6, 3), num_classes=num_classes)


if __name__ == "__main__":
    model = resnet34(num_classes=5)
    sample = torch.randn(2, 3, 224, 224)
    print(model(sample).shape)  # torch.Size([2, 5])
