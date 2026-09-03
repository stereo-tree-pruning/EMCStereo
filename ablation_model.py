# @Author  : EMCStereo ablation
# Configurable EMCStereo variant used for the module ablation study.
#
# The full EMCStereo backbone injects three paper modules into a PSMNet
# backbone:
#     E -- EMA        (Efficient Multi-Scale Attention) on the 128-ch feature
#     M -- MSFblock   (Multi-Scale Fusion block)        on the 4 SPP branches
#     C -- CoordAtt   (Coordinate Attention)            on the final 32-ch feat
#
# This file mirrors src/emc_stereo_backbone.py exactly, but makes each of the
# three modules individually toggleable so we can measure their contribution.
# Everything else (firstconv, layer1-4, SPP branches, cost/disp processors,
# loss) is identical to the original so the ablation is a fair comparison.
#
# Nothing in the original src/ is modified — this only *reuses* its primitives.
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the repo root importable so `src` / `data` resolve wherever this module
# is imported from.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.submodule import conv_bn, conv_bn_relu, BasicBlock       # noqa: E402
from src.attention_modules import EMA, CoordAtt, MSFblock          # noqa: E402
from src.cost_processor import PSMCostProcessor                    # noqa: E402
from src.disp_processor import PSMDispProcessor                    # noqa: E402


class EMCStereoBackboneAblation(nn.Module):
    """PSMNet backbone with individually toggleable EMA / MSFblock / CoordAtt.

    Fallback behaviour when a module is disabled (reverts to plain PSMNet):
        use_ema=False   -> the 128-ch feature is passed through unchanged.
        use_msf=False   -> the 4 SPP branches are concatenated (PSMNet style),
                           giving 4*32=128 ch instead of the fused 32 ch. The
                           lastconv input width adapts accordingly (320 vs 224).
        use_coord=False -> the final 32-ch feature is passed through unchanged.
    """

    def __init__(self, in_planes=3, batch_norm=True,
                 use_ema=True, use_msf=True, use_coord=True):
        super().__init__()
        self.use_ema = use_ema
        self.use_msf = use_msf
        self.use_coord = use_coord
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

        # === Module #1: EMA on 128-channel feature after layer4 ===
        if self.use_ema:
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

        # === Module #2: MSFblock adaptive fusion of the 4 branches ===
        if self.use_msf:
            self.msf = MSFblock(in_channels=32)
            spp_out_ch = 32
        else:
            # PSMNet fallback: plain concat of the 4 branches.
            spp_out_ch = 4 * 32

        # lastconv: out_4_0(64) + out_8(128) + spp_out_ch -> 128 -> 32
        concat_ch = 64 + 128 + spp_out_ch
        self.lastconv = nn.Sequential(
            conv_bn_relu(batch_norm, concat_ch, 128, 3, 1, 1, 1, bias=False),
            nn.Conv2d(128, 32, kernel_size=1, padding=0,
                      stride=1, dilation=1, bias=False),
        )

        # === Module #3: CoordAtt on final 32-channel feature ===
        if self.use_coord:
            self.coord_att = CoordAtt(inp=32, oup=32, reduction=8)

    def _make_layer(self, batch_norm, block, out_planes, blocks, stride,
                    padding, dilation):
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

        # EMA on deep 128-ch feature (identity when disabled).
        if self.use_ema:
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

        if self.use_msf:
            # MSFblock adaptive fusion (replaces concat of 4 branches).
            fused_spp = self.msf(b4, b3, b2, b1)          # 32 ch
        else:
            # PSMNet fallback: concat the 4 branches (branch4..branch1 order).
            fused_spp = torch.cat((b4, b3, b2, b1), dim=1)  # 128 ch

        feat = torch.cat((out_4_0, out_8, fused_spp), dim=1)
        feat = self.lastconv(feat)

        # CoordAtt on final feature (identity when disabled).
        if self.use_coord:
            feat = self.coord_att(feat)
        return feat

    def forward(self, inputs):
        ref_img = inputs["left"]
        tgt_img = inputs["right"]
        l_fms = self._forward(ref_img)
        r_fms = self._forward(tgt_img)
        return {"ref_feature": l_fms, "tgt_feature": r_fms}


class EMCStereoAblation(nn.Module):
    """EMCStereo with individually toggleable EMA / MSFblock / CoordAtt.

    Drop-in replacement for src.emc_stereo.EMCStereo: identical forward / loss
    interface, only the backbone modules are configurable. `cfgs` needs a
    `MAX_DISP` attribute (exactly like the original).
    """

    def __init__(self, cfgs, use_ema=True, use_msf=True, use_coord=True):
        super().__init__()
        self.maxdisp = cfgs.MAX_DISP
        self.use_ema = use_ema
        self.use_msf = use_msf
        self.use_coord = use_coord
        self.Backbone = EMCStereoBackboneAblation(
            use_ema=use_ema, use_msf=use_msf, use_coord=use_coord)
        self.CostProcessor = PSMCostProcessor(max_disp=self.maxdisp)
        self.DispProcessor = PSMDispProcessor(max_disp=self.maxdisp)

    def forward(self, inputs):
        backbone_out = self.Backbone(inputs)
        inputs.update(backbone_out)
        cost_out = self.CostProcessor(inputs)
        inputs.update(cost_out)
        disp_out = self.DispProcessor(inputs)
        return {'disp_pred': disp_out[-1], 'train_preds': disp_out}


def make_emc_factory(use_ema=True, use_msf=True, use_coord=True):
    """Return a callable ``factory(cfgs) -> EMCStereoAblation`` with the given
    module flags baked in. Used to monkey-patch ``train.EMCStereo`` so the whole
    train.py pipeline runs unchanged on the ablation variant."""
    def _factory(cfgs):
        return EMCStereoAblation(cfgs, use_ema=use_ema,
                                 use_msf=use_msf, use_coord=use_coord)
    return _factory


# Registry of the ablation experiments: the plain-PSMNet baseline (no modules),
# the 3 single modules, the 3 pairwise combinations, and the full 3-module
# model (E=EMA, M=MSFblock, C=CoordAtt).
EXPERIMENTS = {
    'none':          dict(use_ema=False, use_msf=False, use_coord=False),
    'ema':           dict(use_ema=True,  use_msf=False, use_coord=False),
    'msf':           dict(use_ema=False, use_msf=True,  use_coord=False),
    'coord':         dict(use_ema=False, use_msf=False, use_coord=True),
    'ema_msf':       dict(use_ema=True,  use_msf=True,  use_coord=False),
    'ema_coord':     dict(use_ema=True,  use_msf=False, use_coord=True),
    'msf_coord':     dict(use_ema=False, use_msf=True,  use_coord=True),
    'ema_msf_coord': dict(use_ema=True,  use_msf=True,  use_coord=True),
}
