# @Author  : EMCStereo
# EMCStereo backbone: PSMNet backbone with EMA + MSFblock + CoordAtt injection.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodule import conv_bn, conv_bn_relu, BasicBlock
from .attention_modules import EMA, CoordAtt, MSFblock


class EMCStereoBackbone(nn.Module):
    """EMCStereo backbone: PSMNet backbone with three injected paper modules.

    Flow:
        firstconv -> layer1 -> layer2 (out_4_0, 64ch)
                            -> layer3 (128ch)
                            -> layer4 (out_8, 128ch)
        out_8_attn = EMA(out_8)
        4 SPP branches (32ch each, upsampled to H/8 x W/8)
        fused_spp = MSFblock(b1, b2, b3, b4)          # 32 ch
        concat_feat = cat(out_4_0(64), out_8_attn(128), fused_spp(32)) = 224 ch
        lastconv: 224 -> 128 -> 32
        out_feat = CoordAtt(out_feat)
    """

    def __init__(self, in_planes=3, batch_norm=True):
        super().__init__()
        self.in_planes_orig = in_planes
        self.batch_norm = batch_norm

        self.firstconv = nn.Sequential(
            conv_bn_relu(batch_norm, in_planes, 32, 3, 2, 1, 1, bias=False),
            conv_bn_relu(batch_norm, 32, 32, 3, 1, 1, 1, bias=False),
            conv_bn_relu(batch_norm, 32, 32, 3, 1, 1, 1, bias=False),
        )

        self.in_planes = 32
        self.layer1 = self._make_layer(batch_norm, BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(batch_norm, BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(batch_norm, BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(batch_norm, BasicBlock, 128, 3, 1, 2, 2)

        # === Injected module #1: EMA on 128-channel feature after layer4 ===
        self.ema = EMA(channels=128, factor=16)

        # SPP-style 4 branches (same as PSMNet)
        self.branch1 = nn.Sequential(
            nn.AvgPool2d((64, 64), stride=(64, 64)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d((32, 32), stride=(32, 32)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d((16, 16), stride=(16, 16)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d((8, 8), stride=(8, 8)),
            conv_bn_relu(batch_norm, 128, 32, 1, 1, 0, 1, bias=False),
        )

        # === Injected module #2: MSFblock replaces concat-fusion of 4 branches ===
        self.msf = MSFblock(in_channels=32)

        # lastconv: 64 (out_4_0) + 128 (out_8_ema) + 32 (msf) = 224 -> 128 -> 32
        self.lastconv = nn.Sequential(
            conv_bn_relu(batch_norm, 224, 128, 3, 1, 1, 1, bias=False),
            nn.Conv2d(128, 32, kernel_size=1, padding=0,
                      stride=1, dilation=1, bias=False),
        )

        # === Injected module #3: CoordAtt on final 32-channel feature ===
        self.coord_att = CoordAtt(inp=32, oup=32, reduction=8)

    def _make_layer(self, batch_norm, block, out_planes, blocks, stride, padding, dilation):
        downsample = None
        if stride != 1 or self.in_planes != out_planes * block.expansion:
            downsample = conv_bn(
                batch_norm, self.in_planes, out_planes * block.expansion,
                kernel_size=1, stride=stride, padding=0, dilation=1,
            )

        layers = [block(batch_norm, self.in_planes, out_planes, stride,
                        downsample, padding, dilation)]
        self.in_planes = out_planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(batch_norm, self.in_planes,
                          out_planes, 1, None, padding, dilation))
        return nn.Sequential(*layers)

    def _forward(self, x):
        out_2_0 = self.firstconv(x)
        out_2_1 = self.layer1(out_2_0)
        out_4_0 = self.layer2(out_2_1)
        out_4_1 = self.layer3(out_4_0)
        out_8 = self.layer4(out_4_1)

        # EMA on deep 128-ch feature
        out_8 = self.ema(out_8)

        h, w = out_8.size(2), out_8.size(3)
        b1 = F.interpolate(self.branch1(out_8), (h, w),
                           mode='bilinear', align_corners=True)
        b2 = F.interpolate(self.branch2(out_8), (h, w),
                           mode='bilinear', align_corners=True)
        b3 = F.interpolate(self.branch3(out_8), (h, w),
                           mode='bilinear', align_corners=True)
        b4 = F.interpolate(self.branch4(out_8), (h, w),
                           mode='bilinear', align_corners=True)

        # MSFblock adaptive fusion (replaces concat of 4 branches)
        fused_spp = self.msf(b4, b3, b2, b1)  # 32 ch

        feat = torch.cat((out_4_0, out_8, fused_spp),
                         dim=1)  # 64 + 128 + 32 = 224
        feat = self.lastconv(feat)

        # CoordAtt on final feature
        feat = self.coord_att(feat)
        return feat

    def forward(self, inputs):
        ref_img = inputs["left"]
        tgt_img = inputs["right"]
        l_fms = self._forward(ref_img)
        r_fms = self._forward(tgt_img)
        return {"ref_feature": l_fms, "tgt_feature": r_fms}
