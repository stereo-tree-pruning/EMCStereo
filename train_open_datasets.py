"""EMCStereo — train + eval on each of the Open datasets.

Datasets (trained separately, same EMCStereo pipeline):
    ETH3, KITTI2012, KITTI2015, Middlebury, SceneFlow

Outputs for each dataset go to
    EMCStereo/Others_output/<dataset>/
        train/ckpt/{best.pth,last.pth}
        tensorboard/
        eval/disparity/disp_0000..0009.png
        eval/eval_results.txt
        logs/*.log

Usage:
    python train_others.py                      # train+eval Scene Flow (multi-GPU)
    python train_others.py --datasets ETH3
    python train_others.py --datasets "Scene Flow" --gpus 2,3
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.emc_stereo import EMCStereo                                  # noqa: E402
from data.virtualtree import build_transform                          # noqa: E402
from data.open_datasets import (                                      # noqa: E402
    OpenStereoDataset, build_dataset_items, split_train_val,
)

# ───────────────────────── defaults ─────────────────────────
OPEN_ROOT = Path(os.environ.get('EMCSTEREO_OPEN_ROOT', str(HERE / 'datasets')))

DATASET_ROOTS = {
    'ETH3':       OPEN_ROOT / 'ETH3',
    'KITTI2012':  OPEN_ROOT / 'KITTI2012',
    'KITTI2015':  OPEN_ROOT / 'KITTI2015',
    'Middlebury': OPEN_ROOT / 'Middlebury',
    'Scene Flow': OPEN_ROOT / 'Scene Flow',
}

ALL_DATASETS = list(DATASET_ROOTS.keys())

DEFAULTS = dict(
    output_dir=str(HERE / 'Others_output'),

    max_disp=192,
    crop_size=(256, 512),

    epochs=300,
    early_stop_patience=50,
    batch_per_gpu=4,          # global batch = batch_per_gpu * #GPUs
    workers=8,

    optimizer='adamw',
    lr=5e-4,
    weight_decay=1e-4,
    warmup_iters=500,
    scheduler='cosine',
    multistep_milestones=(15, 20),
    multistep_gamma=0.1,

    amp=True,
    grad_clip=1.0,

    disp_pred_clamp_factor=2.0,
    loss_cap=1.0e3,
    loss_weights=(0.5, 0.7, 1.0),
    color_aug=True,

    log_interval=10,
    eval_vis_count=10,
    val_ratio=0.1,
    seed=0,
    pretrained=str(HERE / 'pretrain' / 'sceneflow.pth'),

    # memory safety: cap peak GPU memory per process, then keep the tuned
    # effective batch via gradient accumulation (both handled in run_train).
    mem_ceiling_gib=20.0,     # hard per-GPU reserved-memory cap (GiB)
    accum_steps=None,         # None/0 = auto: recover the effective batch

    ema=False,                # weight EMA off by default (on for Scene Flow)
    ema_decay=0.9998,
    tta=False,                # test-time vertical-flip averaging (final eval)
)

# ── per-dataset accuracy-tuned overrides ──
# Scene Flow is the pre-training source, so it is trained from scratch with a
# PSMNet-style high-accuracy recipe (AdamW + cosine, longer schedule, larger
# effective batch, mild colour aug). Everything else warm-starts from
# pretrain/sceneflow.pth (the DEFAULTS above).
DATASET_HP = {
    'Scene Flow': dict(
        # Recipe tuned to beat the 1.09 EPE baseline WITHOUT touching the
        # architecture / blocks / flowchart. Only training-side levers:
        # larger effective batch, larger crop, a full cosine anneal that is
        # NOT early-stopped, bf16 AMP and weight-EMA (both in run_train).
        epochs=36,
        early_stop_patience=0,    # 0 = disabled: let cosine fully anneal to ~0
        batch_per_gpu=8,          # target effective per-GPU; physically reduced
                                  # to fit the mem cap, recovered via grad-accum
        workers=12,
        optimizer='adamw',
        lr=8.0e-4,
        weight_decay=1.0e-4,
        scheduler='cosine',
        warmup_iters=2000,
        crop_size=(288, 576),     # more context than 256x512, fits 46GB
        color_aug=True,
        val_ratio=0.1,            # matches the existing best.pth val split
        pretrained='',            # train from scratch
        ema=True,                 # eval + save EMA weights (free EPE gain)
        ema_decay=0.9995,
        loss_weights=(0.25, 0.5, 1.0),  # emphasise final head (used at test)
        tta=True,                 # also report vertical-flip TTA at final eval
    ),
}

# ── fine-tune overrides applied when --resume is used ──
# Continuing from a checkpoint: anneal a moderate LR to ~0 over a short horizon
# (cosine restart). This is what actually lowers EPE past the earlier plateau
# (the original 300-epoch cosine never decayed). Explicit CLI flags still win.
RESUME_HP = dict(
    epochs=30,                # ADDITIONAL epochs
    lr=4.0e-4,                # fine-tune peak → cosine → 0
    warmup_iters=200,
    scheduler='cosine',
    early_stop_patience=10,
)


METRIC_DESCRIPTIONS = {
    'epe': 'End Point Error (Mean Absolute Error, lower is better)',
    'd1_all': 'D1-all: % pixels with error >3px and >5% (lower is better)',
    'thres_05': 'Bad 0.5: % pixels with error >0.5px (lower is better)',
    'thres_1': 'Bad 1.0: % pixels with error >1px (lower is better)',
    'thres_2': 'Bad 2.0: % pixels with error >2px (lower is better)',
    'thres_3': 'Bad 3.0: % pixels with error >3px (lower is better)',
    'thres_4': 'Bad 4.0: % pixels with error >4px (lower is better)',
    'thres_5': 'Bad 5.0: % pixels with error >5px (lower is better)',
    'rmse': 'Root Mean Square Error (lower is better)',
    'absrel': 'Absolute Relative Error |pred-gt|/gt (lower is better)',
    'mae': 'Mean Absolute Error (lower is better)',
    'sqrel': 'Squared Relative Error (pred-gt)^2/gt (lower is better)',
    'rmse_log': 'RMSE of log disparity (lower is better)',
    'delta1': 'Accuracy δ<1.25 (higher is better)',
    'delta2': 'Accuracy δ<1.25^2 (higher is better)',
    'delta3': 'Accuracy δ<1.25^3 (higher is better)',
}


# ───────────────────────── misc ─────────────────────────
def _set_seed(seed: int) -> None:
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup_logger(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_dir / f'{name.lower().replace(" ", "_")}.log')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _disp_to_color(disp: np.ndarray, max_disp: float = 192.0) -> np.ndarray:
    import cv2
    d = np.clip(disp, 0, max_disp) / max_disp * 255.0
    return cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_JET)


# ───────────────────────── loss ─────────────────────────
def compute_loss(preds, disp_gt, max_disp, weights=(0.5, 0.7, 1.0),
                 clamp_factor=2.0, loss_cap=1.0e3):
    mask = torch.isfinite(disp_gt) & (disp_gt > 0) & (disp_gt < max_disp)
    if not mask.any():
        return (sum((p.float() * 0.0).sum() for p in preds),
                {'loss_disp': 0.0, 'empty_mask': 1.0})
    lo = -clamp_factor * max_disp
    hi = clamp_factor * max_disp
    loss = None
    n_terms = 0
    for pred, w in zip(preds, weights):
        pred = pred.float()
        tm = mask & torch.isfinite(pred)
        if not tm.any():
            continue
        p = pred[tm].clamp(lo, hi)
        term = F.smooth_l1_loss(p, disp_gt[tm], reduction='mean')
        if not torch.isfinite(term):
            continue
        loss = w * term if loss is None else loss + w * term
        n_terms += 1
    if loss is None or n_terms == 0:
        return (sum((p.float() * 0.0).sum() for p in preds),
                {'loss_disp': 0.0, 'all_nan': 1.0})
    d = loss.detach()
    if float(d) > loss_cap:
        loss = loss * (loss_cap / d).clamp(min=1e-6)
    return loss, {'loss_disp': float(loss.item())}


# ───────────────────────── metrics ─────────────────────────
def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, max_disp: int) -> dict:
    mask = torch.isfinite(gt) & (gt > 0) & (gt < max_disp)
    zero = {k: 0.0 for k in METRIC_DESCRIPTIONS}
    zero['n_valid'] = 0
    if not mask.any():
        return zero
    p = pred[mask].float()
    g = gt[mask].float()
    err = (p - g).abs()
    n = int(mask.sum().item())
    d1 = (((err > 3) & (err / g.clamp(min=1e-6) > 0.05)).float().mean().item())
    p_safe = p.clamp(min=1e-6)
    g_safe = g.clamp(min=1e-6)
    log_diff = torch.log(p_safe) - torch.log(g_safe)
    ratio = torch.max(p_safe / g_safe, g_safe / p_safe)
    return {
        'epe': err.mean().item(),
        'd1_all': d1,
        'thres_05': (err > 0.5).float().mean().item(),
        'thres_1': (err > 1).float().mean().item(),
        'thres_2': (err > 2).float().mean().item(),
        'thres_3': (err > 3).float().mean().item(),
        'thres_4': (err > 4).float().mean().item(),
        'thres_5': (err > 5).float().mean().item(),
        'rmse': torch.sqrt((err ** 2).mean()).item(),
        'absrel': (err / g_safe).mean().item(),
        'mae': err.mean().item(),
        'sqrel': ((err ** 2) / g_safe).mean().item(),
        'rmse_log': torch.sqrt((log_diff ** 2).mean()).item(),
        'delta1': (ratio < 1.25).float().mean().item(),
        'delta2': (ratio < 1.25 ** 2).float().mean().item(),
        'delta3': (ratio < 1.25 ** 3).float().mean().item(),
        'n_valid': n,
    }


# ───────────────────────── optimizer ─────────────────────────
def build_optimizer(model, cfg):
    no_decay, decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{'params': decay, 'weight_decay': cfg.weight_decay},
              {'params': no_decay, 'weight_decay': 0.0}]
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.999))
    if cfg.optimizer == 'rmsprop':
        return torch.optim.RMSprop([p for g in groups for p in g['params']],
                                   lr=cfg.lr)
    raise ValueError(cfg.optimizer)


def compute_lr_scale(step, total_steps, warmup, scheduler, milestones, gamma):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if scheduler == 'cosine':
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    return 1.0


# ───────────────────────── weight EMA ─────────────────────────
class ModelEMA:
    """Exponential moving average of weights (params + float buffers).

    Architecture-preserving: the forward graph is untouched; we just keep a
    smoothed copy of the weights and evaluate / save those. Typically lowers
    stereo EPE by a few hundredths for free.
    """

    def __init__(self, model, decay=0.9998):
        core = (model.module if isinstance(model, torch.nn.DataParallel)
                else model)
        self.module = copy.deepcopy(core).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model, step=None):
        d = self.decay
        if step is not None:                    # warm up: track fast early on
            d = min(d, (1.0 + step) / (10.0 + step))
        src = (model.module if isinstance(model, torch.nn.DataParallel)
               else model)
        msd = src.state_dict()
        for k, v in self.module.state_dict().items():
            mv = msd[k].detach().to(v.device)
            if v.dtype.is_floating_point:
                v.mul_(d).add_(mv, alpha=1.0 - d)
            else:
                v.copy_(mv)


# ───────────────────────── train / eval ─────────────────────────
def _fit_global_batch(model, optimizer, scaler, cfg, device, desired,
                      n_gpus, mem_ceiling_gib, logger):
    """Largest global batch (a multiple of #GPUs) whose peak GPU memory stays
    under ``mem_ceiling_gib`` on every visible device and survives a full
    train step.

    Unlike a plain OOM-or-not probe, this runs a *real* fwd+bwd+optimizer.step
    (so AdamW state, gradients and the DataParallel gather are all counted),
    reads back ``max_memory_reserved`` and keeps a safety margin under the hard
    cap — so run-to-run cuDNN workspace variance can't push training over the
    wall mid-run. Weights / optimizer / scaler state are rolled back so training
    starts exactly as if the probe never ran (crop is fixed, so if one step
    fits, every train step fits).
    """
    if device.type != 'cuda':
        return max(1, int(desired))

    h, w = cfg.crop_size
    visible = list(range(torch.cuda.device_count()))
    step = max(1, n_gpus)
    # headroom under the hard cap for the CUDA context + cuDNN workspace jitter
    target_gib = max(1.0, mem_ceiling_gib - 1.5)

    model_snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt_snap = copy.deepcopy(optimizer.state_dict())
    scaler_snap = copy.deepcopy(scaler.state_dict())

    def _one_step(b):
        for i in visible:
            torch.cuda.reset_peak_memory_stats(i)
        left = torch.randn(b, 3, h, w, device=device)
        right = torch.randn(b, 3, h, w, device=device)
        disp = torch.rand(b, h, w, device=device) * cfg.max_disp
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=cfg.amp):
            out = model({'left': left, 'right': right, 'disp': disp})
            loss, _ = compute_loss(
                out['train_preds'], disp, cfg.max_disp,
                weights=cfg.loss_weights,
                clamp_factor=cfg.disp_pred_clamp_factor,
                loss_cap=cfg.loss_cap)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        peak = max((torch.cuda.max_memory_reserved(i) for i in visible),
                   default=0) / (1024 ** 3)
        del left, right, disp, out, loss
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return peak

    b = max(step, (int(desired) // step) * step)
    chosen = step
    try:
        while b >= step:
            try:
                peak = _one_step(b)
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                logger.warning(f'[{cfg.dataset}] probe batch {b} OOM '
                               f'→ try {b - step}')
                b -= step
                continue
            if peak <= target_gib:
                chosen = b
                if b < desired:
                    logger.warning(
                        f'[{cfg.dataset}] mem-fit global_batch={b} '
                        f'(peak {peak:.1f} GiB ≤ {target_gib:.1f} GiB, '
                        f'reduced from desired {desired})')
                else:
                    logger.info(
                        f'[{cfg.dataset}] global_batch={b} fits '
                        f'(peak {peak:.1f} GiB ≤ {target_gib:.1f} GiB)')
                break
            logger.warning(f'[{cfg.dataset}] probe batch {b} peak '
                           f'{peak:.1f} GiB > {target_gib:.1f} GiB '
                           f'→ try {b - step}')
            b -= step
    finally:
        model.load_state_dict(model_snap, strict=True)
        optimizer.load_state_dict(opt_snap)
        scaler.load_state_dict(scaler_snap)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return chosen


def run_train(cfg, dataset_name, train_ds, val_ds, run_dir, logger) -> Path:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'[{dataset_name}] train={len(train_ds)}  val={len(val_ds)}')

    n_gpus = max(1, torch.cuda.device_count())
    desired_batch = (cfg.batch_size if getattr(cfg, 'batch_size', None)
                     else cfg.batch_per_gpu * n_gpus)
    desired_batch = max(1, min(desired_batch, len(train_ds)))
    logger.info(f'[{dataset_name}] GPUs={n_gpus} '
                f'batch/gpu={cfg.batch_per_gpu} desired_batch={desired_batch}')

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=max(1, cfg.workers // 2),
                            pin_memory=True)

    class _Cfg:
        MAX_DISP = cfg.max_disp
    model = EMCStereo(_Cfg()).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f'[{dataset_name}] EMCStereo params: {n_params:.3f} M')

    # ── hard per-GPU memory cap: the caching allocator may not reserve more
    #    than mem_ceiling_gib on any device, so peak stays ≤ cap (shared-box
    #    friendly). The batch probe below fits the batch under this wall. ──
    mem_ceiling_gib = float(getattr(cfg, 'mem_ceiling_gib', 20.0) or 20.0)
    if device.type == 'cuda':
        total_gib = (torch.cuda.get_device_properties(0).total_memory
                     / (1024 ** 3))
        frac = max(0.05, min(1.0, mem_ceiling_gib / total_gib))
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_per_process_memory_fraction(frac, i)
            except Exception as e:                               # noqa: BLE001
                logger.warning(f'[{dataset_name}] mem-cap set failed on '
                               f'cuda:{i}: {e}')
        logger.info(f'[{dataset_name}] GPU memory cap {mem_ceiling_gib:.1f} '
                    f'GiB/GPU (fraction {frac:.3f} of {total_gib:.1f} GiB)')

    # ── resume: continue training from a checkpoint ──
    resume_path = getattr(cfg, 'resume', None)
    resume_ckpt = None
    resumed = False
    start_epoch = 0
    if resume_path:
        rp = Path(resume_path)
        if rp.exists():
            resume_ckpt = torch.load(rp, map_location=device,
                                     weights_only=False)
            state = (resume_ckpt['model'] if 'model' in resume_ckpt
                     else resume_ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            start_epoch = int(resume_ckpt.get('epoch', -1)) + 1
            prev_epe = (resume_ckpt.get('val_metrics') or {}).get('epe')
            resumed = True
            logger.info(
                f'[{dataset_name}] RESUME from {rp} '
                f'(prev epoch={resume_ckpt.get("epoch")}, '
                f'gstep={resume_ckpt.get("global_step")}, '
                f'prev val epe={prev_epe})')
            if missing:
                logger.warning(f'  missing keys ({len(missing)}): '
                               f'{missing[:5]}...')
            if unexpected:
                logger.warning(f'  unexpected keys ({len(unexpected)}): '
                               f'{unexpected[:5]}...')
        else:
            logger.warning(
                f'[{dataset_name}] --resume path not found: {rp}')

    # ── pretrained warm-start (skipped when resuming) ──
    pretrained = '' if resumed else getattr(cfg, 'pretrained', '')
    if pretrained:
        pt_path = Path(pretrained)
        if pt_path.exists():
            pt = torch.load(pt_path, map_location=device, weights_only=False)
            state = pt['model'] if 'model' in pt else (
                pt.get('state_dict') or pt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info(
                f'[{dataset_name}] Loaded pretrained weights from {pt_path}')
            if missing:
                logger.warning(
                    f'  missing keys ({len(missing)}): {missing[:5]}...')
            if unexpected:
                logger.warning(
                    f'  unexpected keys ({len(unexpected)}): {unexpected[:5]}...')
        else:
            logger.warning(
                f'[{dataset_name}] Pretrained path not found, skipping: {pt_path}')

    # ── weight EMA (architecture-preserving: we eval & save EMA weights) ──
    ema = (ModelEMA(model, decay=getattr(cfg, 'ema_decay', 0.9998))
           if getattr(cfg, 'ema', False) else None)
    if ema is not None:
        logger.info(f'[{dataset_name}] weight EMA on (decay={cfg.ema_decay})')

    # ── multi-GPU ──
    if torch.cuda.device_count() > 1:
        logger.info(
            f'[{dataset_name}] DataParallel on {torch.cuda.device_count()} GPUs')
        model = torch.nn.DataParallel(model)

    optimizer = build_optimizer(model, cfg)
    if resumed and resume_ckpt is not None and 'optimizer' in resume_ckpt:
        try:
            optimizer.load_state_dict(resume_ckpt['optimizer'])
            logger.info(f'[{dataset_name}] optimizer state restored')
        except Exception as e:                                   # noqa: BLE001
            logger.warning(f'[{dataset_name}] optimizer restore failed: {e}')

    # bf16 AMP when supported → more stable than fp16 (cleaner convergence);
    # GradScaler is only needed for fp16.
    use_bf16 = (bool(cfg.amp) and device.type == 'cuda'
                and torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda',
                                  enabled=bool(cfg.amp) and not use_bf16)
    if cfg.amp:
        logger.info(f'[{dataset_name}] AMP dtype='
                    f'{"bfloat16" if use_bf16 else "float16"}')

    # Memory-safe batch: fit the largest global batch (multiple of #GPUs) whose
    # peak reserved memory stays under the cap, measured with a real train step.
    global_batch = _fit_global_batch(
        model, optimizer, scaler, cfg, device, desired_batch,
        n_gpus, mem_ceiling_gib, logger)
    cfg.global_batch = global_batch

    # Preserve the tuned effective batch via gradient accumulation: if the mem
    # cap forced a smaller physical batch, accumulate to recover the intended
    # (accuracy-critical) effective batch of `desired_batch`.
    if getattr(cfg, 'accum_steps', None):
        accum_steps = max(1, int(cfg.accum_steps))
    else:
        accum_steps = max(1, round(desired_batch / max(1, global_batch)))
    effective_batch = global_batch * accum_steps
    cfg.accum_steps = accum_steps
    cfg.effective_batch = effective_batch
    logger.info(f'[{dataset_name}] physical global_batch={global_batch} '
                f'x accum={accum_steps} -> effective batch={effective_batch}')

    train_loader = DataLoader(train_ds, batch_size=global_batch,
                              shuffle=True, num_workers=cfg.workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=cfg.workers > 0)

    micro_per_epoch = max(1, len(train_loader))
    updates_per_epoch = max(1, micro_per_epoch // accum_steps)
    total_steps = cfg.epochs * updates_per_epoch      # in optimizer updates

    if cfg.scheduler == 'multistep':
        step_milestones = [int(m) * updates_per_epoch
                           for m in cfg.multistep_milestones]

        def lr_scale(step):
            if step < cfg.warmup_iters:
                return (step + 1) / max(1, cfg.warmup_iters)
            k = sum(1 for m in step_milestones if step >= m)
            return cfg.multistep_gamma ** k
    else:
        def lr_scale(step):
            return compute_lr_scale(step, total_steps, cfg.warmup_iters,
                                    cfg.scheduler,
                                    cfg.multistep_milestones,
                                    cfg.multistep_gamma)

    ckpt_dir = run_dir / 'train' / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / 'tensorboard'))

    (run_dir / 'config.json').write_text(json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v)
         for k, v in vars(cfg).items()},
        indent=2, default=str))

    best_epe = float('inf')
    best_path = ckpt_dir / 'best.pth'
    last_path = ckpt_dir / 'last.pth'
    epochs_since_improve = 0

    # When resuming, measure the baseline on THIS val split so "best" tracking
    # is consistent with the loaded weights (existing best.pth kept unless
    # actually beaten).
    if resumed:
        base = run_eval(model, val_loader, device, cfg.max_disp,
                        save_dir=None, logger=logger,
                        tag=f'{dataset_name}/resume-baseline', save_limit=0)
        best_epe = base.get('epe', float('inf'))
        for k, v in base.items():
            writer.add_scalar(f'val/{k}', v, max(0, start_epoch - 1))
        # Guarantee best/last exist in this run_dir (baseline == current best),
        # so the final eval step has a checkpoint even if we never beat it.
        base_ckpt = {
            'model': (model.module
                      if isinstance(model, torch.nn.DataParallel)
                      else model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': start_epoch - 1,
            'val_metrics': base,
            'global_step': 0,
            'dataset': dataset_name,
        }
        torch.save(base_ckpt, best_path)
        torch.save(base_ckpt, last_path)
        logger.info(f'[{dataset_name}] resume baseline val epe={best_epe:.4f} '
                    f'(continuing {cfg.epochs} epochs, '
                    f'cosine lr {cfg.lr:.2e} → 0)')

    global_step = 0                       # counts optimizer updates
    t0 = time.time()
    for local_epoch in range(cfg.epochs):
        epoch = start_epoch + local_epoch
        model.train()
        running = []
        optimizer.zero_grad(set_to_none=True)
        micro_in_window = 0
        for it, data in enumerate(train_loader):
            left = data['left'].to(device, non_blocking=True)
            right = data['right'].to(device, non_blocking=True)
            disp_gt = data['disp'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=cfg.amp, dtype=amp_dtype):
                out = model({'left': left, 'right': right, 'disp': disp_gt})
                loss, _ = compute_loss(
                    out['train_preds'], disp_gt, cfg.max_disp,
                    weights=cfg.loss_weights,
                    clamp_factor=cfg.disp_pred_clamp_factor,
                    loss_cap=cfg.loss_cap)

            if torch.isfinite(loss):
                # divide by accum so summed grads match one effective batch
                scaled = loss / accum_steps
                if scaler.is_enabled():
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                running.append(float(loss.item()))
                micro_in_window += 1
            else:
                logger.warning(
                    f'[{dataset_name} ep {epoch} it {it}] '
                    f'non-finite loss, skipping micro-step')

            # step only on accumulation boundaries (or the epoch's tail)
            is_boundary = ((it + 1) % accum_steps == 0
                           or (it + 1) == micro_per_epoch)
            if not is_boundary:
                continue
            if micro_in_window == 0:
                optimizer.zero_grad(set_to_none=True)
                continue

            scale = lr_scale(global_step)
            for g in optimizer.param_groups:
                g['lr'] = cfg.lr * scale

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model, step=global_step)

            if global_step % cfg.log_interval == 0:
                cur = running[-1] if running else float('nan')
                writer.add_scalar('train/loss', cur, global_step)
                writer.add_scalar('train/lr',
                                  optimizer.param_groups[0]['lr'], global_step)
                avg = sum(running[-50:]) / max(1, len(running[-50:]))
                logger.info(
                    f'[{dataset_name}] Ep {epoch}/{start_epoch + cfg.epochs} '
                    f'It {it + 1}/{micro_per_epoch} upd {global_step} '
                    f'loss={cur:.4f}(avg50={avg:.4f}) '
                    f'lr={optimizer.param_groups[0]["lr"]:.2e} '
                    f'accum={accum_steps} '
                    f'elapsed={(time.time() - t0) / 60:.1f}m')
            global_step += 1
            micro_in_window = 0

        # When EMA is on, the EMA weights are what we keep, so validate and
        # best-track on them (single eval either way).
        eval_target = ema.module if ema is not None else model
        val_metrics = run_eval(eval_target, val_loader, device, cfg.max_disp,
                               save_dir=None, logger=logger,
                               tag=f'{dataset_name}/ep{epoch}',
                               save_limit=cfg.eval_vis_count)
        save_state = (ema.module.state_dict() if ema is not None
                      else (model.module
                            if isinstance(model, torch.nn.DataParallel)
                            else model).state_dict())
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        logger.info(f'[{dataset_name}] Ep {epoch} '
                    f'VAL{"(ema)" if ema is not None else ""}: ' +
                    ' '.join(f'{k}={v:.4f}' for k, v in val_metrics.items()))

        ckpt = {'model': save_state,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics,
                'global_step': global_step,
                'dataset': dataset_name,
                'ema': ema is not None}
        torch.save(ckpt, last_path)
        if val_metrics['epe'] < best_epe:
            best_epe = val_metrics['epe']
            torch.save(ckpt, best_path)
            logger.info(f'[{dataset_name}] => new best EPE={best_epe:.4f}, '
                        f'saved {best_path}')
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            logger.info(f'[{dataset_name}] => no improvement for '
                        f'{epochs_since_improve} epoch(s) '
                        f'(best EPE={best_epe:.4f})')
            patience = getattr(cfg, 'early_stop_patience', 0)
            if patience and epochs_since_improve >= patience:
                logger.info(f'[{dataset_name}] Early stopping: '
                            f'no improvement for {epochs_since_improve} '
                            f'epochs (patience={patience}).')
                break

    writer.close()
    logger.info(f'[{dataset_name}] TRAIN DONE. best EPE={best_epe:.4f}, '
                f'best ckpt: {best_path}')
    return best_path


@torch.no_grad()
def run_eval(model, loader, device, max_disp, save_dir, logger,
             tag='eval', save_limit=10, tta=False):
    import cv2
    model.eval()
    agg = {k: 0.0 for k in METRIC_DESCRIPTIONS}
    n = 0
    saved = 0
    if save_dir is not None:
        save_dir = Path(save_dir)
        (save_dir / 'disparity').mkdir(parents=True, exist_ok=True)

    for data in loader:
        left = data['left'].to(device, non_blocking=True)
        right = data['right'].to(device, non_blocking=True)
        disp_gt = data['disp'].to(device, non_blocking=True)

        out = model({'left': left, 'right': right, 'disp': disp_gt})
        pred = out['disp_pred']
        if pred.dim() == 4:
            pred = pred.squeeze(1)

        if tta:
            # Vertical flip preserves the horizontal (disparity) geometry, so
            # the flipped-back estimate is directly averageable. Pure inference
            # trick — network / flowchart unchanged.
            lf = torch.flip(left, dims=[-2])
            rf = torch.flip(right, dims=[-2])
            pf = model({'left': lf, 'right': rf, 'disp': disp_gt})['disp_pred']
            if pf.dim() == 4:
                pf = pf.squeeze(1)
            pred = 0.5 * (pred + torch.flip(pf, dims=[-2]))

        if 'pad' in data:
            pad = data['pad'][0].tolist()
            pt, pr, _, _ = pad
            if pt:
                pred = pred[:, pt:, :]
                disp_gt = disp_gt[:, pt:, :]
            if pr:
                pred = pred[:, :, :-pr]
                disp_gt = disp_gt[:, :, :-pr]

        m = compute_metrics(pred, disp_gt, max_disp)
        for k in agg:
            agg[k] += m[k]
        n += 1

        if save_dir is not None and saved < save_limit:
            img = _disp_to_color(pred[0].detach().float().cpu().numpy(),
                                 max_disp=max_disp)
            gt_img = _disp_to_color(disp_gt[0].detach().float().cpu().numpy(),
                                    max_disp=max_disp)
            combo = np.concatenate([img, gt_img], axis=0)
            cv2.imwrite(str(save_dir / 'disparity' /
                            f'disp_{saved:04d}.png'), combo)
            saved += 1

    agg = {k: v / max(1, n) for k, v in agg.items()}
    if save_dir is not None:
        lines = [f'EMCStereo [{tag}] eval over {n} samples', '']
        for k, v in agg.items():
            desc = METRIC_DESCRIPTIONS.get(k, '')
            lines.append(f'{k}: {v:.6f}    # {desc}' if desc
                         else f'{k}: {v:.6f}')
        txt = '\n'.join(lines) + '\n'
        (save_dir / 'eval_results.txt').write_text(txt)
        if logger:
            logger.info(f'[{tag}] Saved eval results: '
                        f'{save_dir / "eval_results.txt"}')
            logger.info(f'[{tag}] Saved {saved} disparity vis: '
                        f'{save_dir / "disparity"}')
    return agg


def run_final_eval(cfg, dataset_name, val_ds, run_dir, ckpt_path, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class _Cfg:
        MAX_DISP = cfg.max_disp
    model = EMCStereo(_Cfg()).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    logger.info(f'[{dataset_name}] Loaded ckpt {ckpt_path}')

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=max(1, cfg.workers // 2),
                            pin_memory=True)

    eval_dir = run_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    agg = run_eval(model, val_loader, device, cfg.max_disp,
                   save_dir=eval_dir, logger=logger,
                   tag=f'{dataset_name}/final',
                   save_limit=cfg.eval_vis_count)
    logger.info(f'[{dataset_name}] FINAL EVAL :: ' +
                ' '.join(f'{k}={v:.6f}' for k, v in agg.items()))
    if getattr(cfg, 'tta', False):
        tta_dir = run_dir / 'eval_tta'
        tta_dir.mkdir(parents=True, exist_ok=True)
        agg_tta = run_eval(model, val_loader, device, cfg.max_disp,
                           save_dir=tta_dir, logger=logger,
                           tag=f'{dataset_name}/final-tta',
                           save_limit=cfg.eval_vis_count, tta=True)
        better = 'TTA better' if agg_tta['epe'] < agg['epe'] else 'plain better'
        logger.info(f'[{dataset_name}] FINAL EVAL (TTA vflip) :: ' +
                    ' '.join(f'{k}={v:.6f}' for k, v in agg_tta.items()))
        logger.info(f'[{dataset_name}] EPE plain={agg["epe"]:.6f} '
                    f'vs TTA={agg_tta["epe"]:.6f} -> {better}')
    return eval_dir


# ───────────────────────── dataset prep ─────────────────────────
def prepare_datasets(cfg, dataset_name, logger):
    root = DATASET_ROOTS[dataset_name]
    items = build_dataset_items(dataset_name, root)
    if not items:
        logger.warning(f'[{dataset_name}] No samples found under {root}')
        return None, None
    logger.info(f'[{dataset_name}] {len(items)} samples at {root}')
    train_items, val_items = split_train_val(items,
                                             val_ratio=cfg.val_ratio,
                                             seed=cfg.seed)
    logger.info(f'[{dataset_name}] split: train={len(train_items)}  '
                f'val={len(val_items)}')
    train_tf = build_transform('train',
                               crop_size=tuple(cfg.crop_size),
                               max_disp=cfg.max_disp,
                               color_aug=cfg.color_aug)
    val_tf = build_transform('val', max_disp=cfg.max_disp)
    return (OpenStereoDataset(train_items, transform=train_tf),
            OpenStereoDataset(val_items, transform=val_tf))


# ───────────────────────── GPU / config resolution ─────────────────────────
def _pick_gpus(min_free_mib: int = 20000):
    """Auto-select GPUs with at least ``min_free_mib`` free (shared box safe).

    Returns a comma-separated index string for CUDA_VISIBLE_DEVICES, or None
    if nvidia-smi is unavailable.
    """
    try:
        import subprocess
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.free',
             '--format=csv,noheader,nounits'], text=True)
    except Exception:                                            # noqa: BLE001
        return None
    free = []
    for line in out.strip().splitlines():
        idx, mem = (x.strip() for x in line.split(','))
        free.append((int(idx), int(mem)))
    good = [str(i) for i, m in free if m >= min_free_mib]
    if good:
        return ','.join(good)
    if free:
        return str(max(free, key=lambda t: t[1])[0])
    return None


def _check_cuda_ok(logger) -> bool:
    """Fail fast (with guidance) if PyTorch can't launch kernels on the GPUs.

    Catches the common 'no kernel image is available' error on new GPUs
    (e.g. Blackwell / RTX PRO 5000, sm_120) when the installed torch build is
    too old, instead of crashing deep inside training.
    """
    if not torch.cuda.is_available():
        logger.warning('CUDA not available — running on CPU (very slow).')
        return True
    try:
        for i in range(torch.cuda.device_count()):
            dev = f'cuda:{i}'
            x = torch.randn(2, 3, 16, 16, device=dev)
            w = torch.randn(4, 3, 3, 3, device=dev)
            y = torch.nn.functional.conv2d(x, w, padding=1)
            _ = (y * 2).relu().sum().item()      # aten kernel (per-arch SASS)
            torch.cuda.synchronize(i)
        names = sorted({torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())})
        logger.info(f'CUDA OK on {names} (torch {torch.__version__}, '
                    f'archs {torch.cuda.get_arch_list()})')
        return True
    except Exception as e:                                       # noqa: BLE001
        names = [torch.cuda.get_device_name(i)
                 for i in range(torch.cuda.device_count())]
        logger.error(
            'CUDA kernel launch FAILED — the installed PyTorch likely does '
            'not support this GPU architecture.\n'
            f'  GPUs : {names}\n'
            f'  torch: {torch.__version__} (CUDA {torch.version.cuda}), '
            f'built archs: {torch.cuda.get_arch_list()}\n'
            '  Blackwell (RTX PRO 5000, sm_120) needs a CUDA 12.8+ wheel:\n'
            '    pip install --index-url '
            'https://download.pytorch.org/whl/cu128 torch torchvision\n'
            f'  error: {e}')
        return False


# Keys resolvable per-dataset: CLI (non-None) > DATASET_HP > DEFAULTS.
_RESOLVE_KEYS = (
    'max_disp', 'crop_size', 'epochs', 'early_stop_patience',
    'batch_per_gpu', 'batch_size', 'workers', 'optimizer', 'lr',
    'weight_decay', 'warmup_iters', 'scheduler', 'multistep_milestones',
    'multistep_gamma', 'amp', 'grad_clip', 'disp_pred_clamp_factor',
    'loss_cap', 'color_aug', 'log_interval', 'eval_vis_count',
    'val_ratio', 'pretrained', 'ema', 'ema_decay', 'loss_weights', 'tta',
    'mem_ceiling_gib', 'accum_steps',
)


def resolve_cfg(args, dataset_name):
    """Build the effective config for one dataset."""
    eff = dict(DEFAULTS)
    eff['batch_size'] = None                       # explicit global override
    eff.update(DATASET_HP.get(dataset_name, {}))
    resuming = getattr(args, 'resume', None) is not None
    if resuming:
        eff.update(RESUME_HP)
    for k in _RESOLVE_KEYS:
        v = getattr(args, k, None)
        if v is not None:
            eff[k] = v
    eff['amp'] = bool(eff['amp'])
    eff['color_aug'] = bool(eff['color_aug'])
    eff['ema'] = bool(eff['ema'])
    eff['tta'] = bool(eff['tta'])
    eff['crop_size'] = tuple(eff['crop_size'])
    eff['loss_weights'] = tuple(eff['loss_weights'])
    eff['multistep_milestones'] = tuple(eff['multistep_milestones'])
    eff['seed'] = args.seed
    eff['output_dir'] = args.output_dir
    eff['dataset'] = dataset_name
    eff['resume'] = getattr(args, 'resume', None)
    return SimpleNamespace(**eff)


# ───────────────────────── CLI ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description='EMCStereo train/eval on Open datasets (SceneFlow-tuned)')
    p.add_argument('--datasets', nargs='+', default=['Scene Flow'],
                   choices=ALL_DATASETS,
                   help='datasets to run (default: Scene Flow)')
    p.add_argument('--mode', default='all',
                   choices=['all', 'train', 'eval'])
    p.add_argument('--skip-train', action='store_true')
    p.add_argument('--skip-eval', action='store_true')

    p.add_argument('--output-dir', default=DEFAULTS['output_dir'])

    # Tunable hyper-params: default None → resolved per-dataset (DATASET_HP).
    p.add_argument('--max-disp', type=int)
    p.add_argument('--crop-size', type=int, nargs=2)

    p.add_argument('--epochs', type=int)
    p.add_argument('--early-stop-patience', type=int)
    p.add_argument('--batch-per-gpu', type=int,
                   help='per-GPU batch; global batch = this * #GPUs')
    p.add_argument('--batch-size', type=int,
                   help='explicit GLOBAL batch (overrides --batch-per-gpu)')
    p.add_argument('--accum-steps', type=int,
                   help='gradient-accumulation steps; default auto to keep the '
                        'tuned effective batch under the memory cap')
    p.add_argument('--mem-ceiling-gib', type=float,
                   help='hard per-GPU reserved-memory cap in GiB (default 20)')
    p.add_argument('--workers', type=int)

    p.add_argument('--optimizer', choices=['adamw', 'rmsprop'])
    p.add_argument('--lr', type=float)
    p.add_argument('--weight-decay', type=float)
    p.add_argument('--warmup-iters', type=int)
    p.add_argument('--scheduler', choices=['cosine', 'multistep'])
    p.add_argument('--multistep-milestones', type=int, nargs='+')
    p.add_argument('--multistep-gamma', type=float)

    p.add_argument('--amp', type=int)
    p.add_argument('--grad-clip', type=float)
    p.add_argument('--disp-pred-clamp-factor', type=float)
    p.add_argument('--loss-cap', type=float)
    p.add_argument('--color-aug', type=int)
    p.add_argument('--ema', type=int, help='1=enable weight EMA, 0=disable')
    p.add_argument('--ema-decay', type=float)
    p.add_argument('--loss-weights', type=float, nargs=3,
                   help='multi-scale loss weights: disp1 disp2 disp3')
    p.add_argument('--tta', type=int,
                   help='1=also report vertical-flip TTA at final eval')

    p.add_argument('--log-interval', type=int)
    p.add_argument('--eval-vis-count', type=int)
    p.add_argument('--val-ratio', type=float)
    p.add_argument('--seed', type=int, default=DEFAULTS['seed'])
    p.add_argument('--pretrained',
                   help="warm-start weights; '' to disable "
                        '(Scene Flow trains from scratch by default)')
    p.add_argument('--resume', nargs='?', const='auto', default=None,
                   help='continue training from a checkpoint; bare --resume '
                        'uses <output>/<dataset>/train/ckpt/best.pth. Applies '
                        'the fine-tune recipe (cosine LR → 0) for higher '
                        'accuracy.')
    p.add_argument('--ckpt', default=None,
                   help='eval: ckpt path override (only valid for 1 dataset)')
    p.add_argument('--gpus', default=None,
                   help='comma list e.g. "2,3"; default auto-picks free GPUs')
    return p.parse_args()


def _sanitize_tag(name: str) -> str:
    return name.replace(' ', '_')


def main() -> int:
    args = parse_args()

    # GPU selection must precede any CUDA context creation (_set_seed inits it).
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    elif not os.environ.get('CUDA_VISIBLE_DEVICES'):
        picked = _pick_gpus()
        if picked:
            os.environ['CUDA_VISIBLE_DEVICES'] = picked

    # Reduce allocator fragmentation (the 'reserved but unallocated' MiB seen in
    # the OOM report) so the per-process memory cap is easier to honour. Must be
    # set before any CUDA context is created (below).
    if not (os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
            or os.environ.get('PYTORCH_ALLOC_CONF')):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    if args.skip_train:
        args.mode = 'eval'
    elif args.skip_eval:
        args.mode = 'train'

    _set_seed(args.seed)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    top_logger = _setup_logger(out_root / 'logs', 'EMCStereo_Others')
    top_logger.info('== EMCStereo multi-dataset pipeline ==')
    top_logger.info(f'datasets: {args.datasets}')
    top_logger.info(
        f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "<all>")}')

    if not _check_cuda_ok(top_logger):
        top_logger.error('Aborting: GPU / PyTorch compatibility check failed.')
        return 2

    summary = {}
    for dataset_name in args.datasets:
        cfg = resolve_cfg(args, dataset_name)
        tag = _sanitize_tag(dataset_name)
        run_dir = out_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        if cfg.resume == 'auto':
            cfg.resume = str(run_dir / 'train' / 'ckpt' / 'best.pth')
        logger = _setup_logger(run_dir / 'logs', f'EMCStereo_{tag}')
        logger.info(f'=============== {dataset_name} ===============')
        logger.info('resolved cfg: ' + json.dumps(
            {k: str(v) for k, v in vars(cfg).items()}, indent=2))

        try:
            train_ds, val_ds = prepare_datasets(cfg, dataset_name, logger)
        except Exception as e:
            logger.exception(f'[{dataset_name}] dataset prep failed: {e}')
            continue
        if train_ds is None or val_ds is None:
            logger.error(f'[{dataset_name}] skipped — no data.')
            continue

        best_ckpt = (Path(args.ckpt) if args.ckpt
                     else run_dir / 'train' / 'ckpt' / 'best.pth')

        try:
            if args.mode in ('all', 'train'):
                best_ckpt = run_train(cfg, dataset_name, train_ds, val_ds,
                                      run_dir, logger)
            if args.mode in ('all', 'eval'):
                if not best_ckpt.exists():
                    logger.error(
                        f'[{dataset_name}] ckpt not found: {best_ckpt}')
                    continue
                eval_dir = run_final_eval(cfg, dataset_name, val_ds, run_dir,
                                          best_ckpt, logger)
                results = eval_dir / 'eval_results.txt'
                if results.exists():
                    body = results.read_text()
                    top_logger.info(
                        f'\n===== {dataset_name} RESULTS =====\n' + body)
                    summary[dataset_name] = body
        except Exception as e:
            logger.exception(f'[{dataset_name}] failed: {e}')
            continue

    if summary:
        combined = out_root / 'SUMMARY.txt'
        with open(combined, 'w') as f:
            for k, v in summary.items():
                f.write(f'\n========== {k} ==========\n{v}\n')
        top_logger.info(f'Summary written to {combined}')

    top_logger.info('ALL DONE.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
