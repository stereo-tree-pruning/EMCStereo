# @Author  : EMCStereo
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cost_processor import PSMCostProcessor
from .disp_processor import PSMDispProcessor
from .emc_stereo_backbone import EMCStereoBackbone


class EMCStereo(nn.Module):
    """EMCStereo = PSMNet backbone + EMA + MSFblock + CoordAtt.

    Name decomposition:
        E -- EMA        (Efficient Multi-scale Attention)
        M -- MSFblock   (Multi-Scale Fusion block)
        C -- CoordAtt   (Coordinate Attention)
    """

    def __init__(self, cfgs):
        super().__init__()
        self.maxdisp = cfgs.MAX_DISP
        self.Backbone = EMCStereoBackbone()
        self.CostProcessor = PSMCostProcessor(max_disp=self.maxdisp)
        self.DispProcessor = PSMDispProcessor(max_disp=self.maxdisp)

    def forward(self, inputs):
        backbone_out = self.Backbone(inputs)
        inputs.update(backbone_out)
        cost_out = self.CostProcessor(inputs)
        inputs.update(cost_out)
        disp_out = self.DispProcessor(inputs)
        return {'disp_pred': disp_out[-1], 'train_preds': disp_out}

    def get_loss(self, model_preds, input_data):
        disp_gt = input_data["disp"]  # [B, H, W]
        # Valid: finite, positive, and below max_disp.
        mask = torch.isfinite(disp_gt) & (
            disp_gt > 0) & (disp_gt < self.maxdisp)
        preds = model_preds['train_preds']
        weights = [0.5, 0.7, 1.0]

        # Whole-batch empty mask → zero loss that still carries grad.
        if not mask.any():
            loss = sum((p.float() * 0.0).sum() for p in preds)
            return loss, {'scalar/train/loss_disp': 0.0,
                          'scalar/train/empty_mask': 1.0}

        # Disparity values should live in [0, maxdisp). Early training can
        # produce exploding predictions (1e4+); clamping here keeps the
        # smooth-L1 loss finite and prevents a single bad batch from
        # producing gigantic gradients that corrupt the weights.
        clamp_hi = float(self.maxdisp) * 2.0
        clamp_lo = -clamp_hi

        loss = None
        n_terms = 0
        for pred, w in zip(preds, weights):
            pred = pred.float()
            # Drop NaN/Inf pred pixels.
            term_mask = mask & torch.isfinite(pred)
            if not term_mask.any():
                continue
            pred_c = pred[term_mask].clamp(clamp_lo, clamp_hi)
            term = F.smooth_l1_loss(pred_c, disp_gt[term_mask], reduction='mean')
            if not torch.isfinite(term):
                continue
            loss = w * term if loss is None else loss + w * term
            n_terms += 1

        if loss is None or n_terms == 0:
            loss = sum((p.float() * 0.0).sum() for p in preds)
            return loss, {'scalar/train/loss_disp': 0.0,
                          'scalar/train/all_nan': 1.0}

        # Final safety: hard-cap loss so one pathological batch cannot
        # explode gradients. Preserves grad direction via detach-scaling.
        LOSS_CAP = 1.0e3
        if loss.detach() > LOSS_CAP:
            scale = (LOSS_CAP / loss.detach()).clamp(min=1e-6)
            loss = loss * scale

        return loss, {'scalar/train/loss_disp': loss.item()}
