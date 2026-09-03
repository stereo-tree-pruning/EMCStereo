# @Author  : EMCStereo
# Attention modules selected from paper_moduals/ for EMCStereo
# (E: EMA, M: MSFblock, C: CoordAtt).
#   - EMA       : 2_3_EMAttention.py  (Efficient Multi-Scale Attention with Cross-Spatial Learning)
#   - CoordAtt  : 1_5_CoordAttention.py (Coordinate Attention for Efficient Mobile Network Design)
#   - MSFblock  : 6_2_MSFblock.py       (SHISRCNet multi-scale fusion block)
# These are ported verbatim (with minor cleanup) so the selection can be traced back
# to the original modules under paper_moduals/.
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# EMA: Efficient Multi-Scale Attention (2_3_EMAttention.py)
# ---------------------------------------------------------------------------
class EMA(nn.Module):
    def __init__(self, channels, factor=16):
        super().__init__()
        # factor must divide channels
        assert channels % factor == 0, f"EMA: channels({channels}) must be divisible by factor({factor})"
        self.groups = factor
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups,
                               channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                 kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups,
                                 kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() *
                     x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(
            b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        y1 = torch.matmul(x11, x12)

        x21 = self.softmax(self.agp(x2).reshape(
            b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        y2 = torch.matmul(x21, x22)

        weights = (y1 + y2).reshape(b * self.groups, 1, h, w).sigmoid()
        out = (group_x * weights).reshape(b, c, h, w)
        return out


# ---------------------------------------------------------------------------
# CoordAtt: Coordinate Attention (1_5_CoordAttention.py)
# ---------------------------------------------------------------------------
class _HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class _HSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = _HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = _HSwish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_w * a_h


# ---------------------------------------------------------------------------
# MSFblock: Multi-Scale Fusion block (6_2_MSFblock.py)
# ---------------------------------------------------------------------------
class _OneConv(nn.Module):
    def __init__(self, in_channels, out_channels, k, p, d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              padding=p, dilation=d, bias=False)

    def forward(self, x):
        return self.conv(x)


class MSFblock(nn.Module):
    """Fuse 4 same-shape feature maps via channel-wise softmax weights."""

    def __init__(self, in_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.se1 = _OneConv(in_channels, in_channels, 1, 0, 1)
        self.se2 = _OneConv(in_channels, in_channels, 1, 0, 1)
        self.se3 = _OneConv(in_channels, in_channels, 1, 0, 1)
        self.se4 = _OneConv(in_channels, in_channels, 1, 0, 1)

    def forward(self, x0, x1, x2, x3):
        y0_w = self.se1(self.gap(x0))
        y1_w = self.se2(self.gap(x1))
        y2_w = self.se3(self.gap(x2))
        y3_w = self.se4(self.gap(x3))

        weight = torch.cat([y0_w, y1_w, y2_w, y3_w], dim=2)  # (B,C,4,1)
        weight = self.softmax(self.sigmoid(weight))

        y0_w = weight[:, :, 0:1, :]
        y1_w = weight[:, :, 1:2, :]
        y2_w = weight[:, :, 2:3, :]
        y3_w = weight[:, :, 3:4, :]

        fused = y0_w * x0 + y1_w * x1 + y2_w * x2 + y3_w * x3
        return self.project(fused)
