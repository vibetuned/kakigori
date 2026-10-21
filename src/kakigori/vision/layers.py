# Standard library imports
import math

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise: Applies a single filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        # Pointwise: 1x1 convolution to mix the channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # SiLU (Swish) is standard for modern YOLO variants

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class PANetNeck(nn.Module):
    """Top-down (and optionally bottom-up) fusion over N backbone levels,
    finest first. Layer names follow the levels (``reduce_layer1``,
    ``fuse_td_p3_p2``...), so 3-level checkpoints load unchanged."""

    def __init__(self, in_channels_list, hidden_dim=128, use_bottom_up=False):
        super().__init__()
        self.use_bottom_up = use_bottom_up
        self.levels = len(in_channels_list)

        # 1x1 Convs to standardize the channel dimensions from the backbone
        for i, channels in enumerate(in_channels_list, 1):
            setattr(self, f"reduce_layer{i}", nn.Conv2d(channels, hidden_dim, 1))

        # Top-down fusion blocks (upsampling deep features to match shallow ones)
        for i in range(self.levels, 1, -1):
            setattr(self, f"fuse_td_p{i}_p{i - 1}", DepthwiseSeparableConv(hidden_dim, hidden_dim))

        # Bottom-up fusion blocks (downsampling shallow features back to deep ones)
        if use_bottom_up:
            for i in range(1, self.levels):
                setattr(self, f"fuse_bu_p{i}_p{i + 1}", DepthwiseSeparableConv(hidden_dim, hidden_dim))

    def forward(self, features):
        # Standardize channels; p[0] is the finest map (stride 4)
        p = [getattr(self, f"reduce_layer{i}")(c) for i, c in enumerate(features, 1)]

        # Top-down pathway
        for i in range(self.levels - 1, 0, -1):
            upsampled = F.interpolate(p[i], size=p[i - 1].shape[-2:], mode="nearest")
            p[i - 1] = getattr(self, f"fuse_td_p{i + 1}_p{i}")(p[i - 1] + upsampled)

        # Bottom-up pathway (optional, for full PANet)
        if self.use_bottom_up:
            # Safely downsample by taking the max features, preserving tiny objects
            for i in range(self.levels - 1):
                downsampled = F.adaptive_max_pool2d(p[i], output_size=p[i + 1].shape[-2:])
                p[i + 1] = getattr(self, f"fuse_bu_p{i + 1}_p{i + 2}")(p[i + 1] + downsampled)

        # Return the fused multi-scale features
        return p


class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_slots=1):
        super().__init__()
        self.num_slots = num_slots

        # The Classification Branch (What is it?)
        self.cls_branch = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels),
            DepthwiseSeparableConv(in_channels, in_channels),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

        # The Regression Branch (Where is it?)
        self.reg_branch = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels),
            DepthwiseSeparableConv(in_channels, in_channels),
            # 4 outputs per regression slot: cx, cy, w, h
            nn.Conv2d(in_channels, 4 * num_slots, kernel_size=1),
        )
        # Initialize the biases
        self._init_biases()

    def _init_biases(self):
        # 1. Initialize Classification Branch (Focal Loss Prior)
        # Tell the network that it should initially guess a ~1% probability for objects
        pi = 0.01
        focal_bias = -math.log((1.0 - pi) / pi)
        nn.init.constant_(self.cls_branch[-1].bias, focal_bias)

        # 2. Initialize Regression Branch (Width/Height Exp Prior)
        # Tell the network that objects are initially ~1% of the image size
        # Indices: 0=tx, 1=ty, 2=width, 3=height
        # Safely overwrite the 1D bias tensor all at once
        with torch.no_grad():
            self.reg_branch[-1].bias.copy_(
                torch.tensor([0.0, 0.0, math.log(0.01), math.log(0.01)]).repeat(self.num_slots)
            )

    def forward(self, x):
        cls_output = self.cls_branch(x)
        reg_output = self.reg_branch(x)
        return cls_output, reg_output
