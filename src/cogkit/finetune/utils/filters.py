import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanFilter(nn.Module):
    def __init__(self, kernel_size: int, in_channels: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.in_channels = in_channels

        self.mean_value = 1.0 / (kernel_size * kernel_size * in_channels)
        weight = torch.full((1, in_channels, kernel_size, kernel_size), 1.0)

        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}")

        if height % self.kernel_size != 0 or width % self.kernel_size != 0:
            raise ValueError(
                f"Input dimensions (height={height}, width={width}) must be divisible by kernel_size={self.kernel_size}"
            )

        filtered = F.conv2d(x, self.weight, stride=self.stride, padding=0, bias=None)
        filtered = filtered * self.mean_value

        return filtered
