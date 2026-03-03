# Standard library imports
import math

# Third party imports
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise: Applies a single filter per input channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        # Pointwise: 1x1 convolution to mix the channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() # SiLU (Swish) is standard for modern YOLO variants

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class PANetNeck(nn.Module):
    def __init__(self, in_channels_list, hidden_dim=128, use_bottom_up=False):
        super().__init__()
        self.use_bottom_up = use_bottom_up

        # 1x1 Convs to standardize the channel dimensions from the backbone
        self.reduce_layer1 = nn.Conv2d(in_channels_list[0], hidden_dim, 1)
        self.reduce_layer2 = nn.Conv2d(in_channels_list[1], hidden_dim, 1)
        self.reduce_layer3 = nn.Conv2d(in_channels_list[2], hidden_dim, 1)

        # Top-down fusion blocks (upsampling deep features to match shallow ones)
        self.fuse_td_p3_p2 = DepthwiseSeparableConv(hidden_dim, hidden_dim)
        self.fuse_td_p2_p1 = DepthwiseSeparableConv(hidden_dim, hidden_dim)

        # Bottom-up fusion blocks (downsampling shallow features back to deep ones)
        if use_bottom_up:
            self.fuse_bu_p1_p2 = DepthwiseSeparableConv(hidden_dim, hidden_dim)
            self.fuse_bu_p2_p3 = DepthwiseSeparableConv(hidden_dim, hidden_dim)

    def forward(self, features):
        c1, c2, c3 = features # e.g., sizes: 64x64, 32x32, 16x16
        
        # Standardize channels
        p1 = self.reduce_layer1(c1)
        p2 = self.reduce_layer2(c2)
        p3 = self.reduce_layer3(c3)

        # Top-down pathway
        p3_upsampled = F.interpolate(p3, size=p2.shape[-2:], mode='nearest')
        p2 = self.fuse_td_p3_p2(p2 + p3_upsampled)

        p2_upsampled = F.interpolate(p2, size=p1.shape[-2:], mode='nearest')
        p1 = self.fuse_td_p2_p1(p1 + p2_upsampled)

        # Bottom-up pathway (optional, for full PANet)
        if self.use_bottom_up:
            # Safely downsample by taking the max features, preserving tiny objects
            p1_downsampled = F.adaptive_max_pool2d(p1, output_size=p2.shape[-2:])
            p2 = self.fuse_bu_p1_p2(p2 + p1_downsampled)

            p2_downsampled = F.adaptive_max_pool2d(p2, output_size=p3.shape[-2:])
            p3 = self.fuse_bu_p2_p3(p3 + p2_downsampled)

        # Return the fused multi-scale features
        return [p1, p2, p3]

class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # The Classification Branch (What is it?)
        self.cls_branch = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels),
            DepthwiseSeparableConv(in_channels, in_channels),
            nn.Conv2d(in_channels, num_classes, kernel_size=1) 
        )
        
        # The Regression Branch (Where is it?)
        self.reg_branch = nn.Sequential(
            DepthwiseSeparableConv(in_channels, in_channels),
            DepthwiseSeparableConv(in_channels, in_channels),
            nn.Conv2d(in_channels, 4, kernel_size=1) # 4 outputs: cx, cy, w, h
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
                torch.tensor([0.0, 0.0, math.log(0.01), math.log(0.01)])
            )

    def forward(self, x):
        cls_output = self.cls_branch(x)
        reg_output = self.reg_branch(x)
        return cls_output, reg_output