"""EMCStereo — standalone train + eval on VirtualTree.

No dependency on OpenStereo-2.

The VirtualTree root is taken from $EMCSTEREO_DATA_ROOT, falling back to
<repo>/datasets/Virtual_branches_data; --data-root overrides both.

Usage:
    python train.py                 # train + eval with defaults
    python train.py --skip-train    # eval existing best.pth only
    python train.py --epochs 30
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.emc_stereo import EMCStereo                      # noqa: E402
from data.virtualtree import VirtualTreeDataset, build_transform  # noqa: E402

# ───────────────────────── defaults ─────────────────────────
DEFAULTS = dict(
    data_root=os.environ.get(
        'EMCSTEREO_DATA_ROOT',
        str(HERE / 'datasets' / 'Virtual_branches_data')),
    train_split=str(HERE / 'data' / 'virtualtree_train.txt'),
    val_split=str(HERE / 'data' / 'virtualtree_val.txt'),

    output_dir=str(HERE / 'output'),

    max_disp=192,
    crop_size=(256, 512),

    epochs=300,
    early_stop_patience=50,
    batch_size=4,
    workers=8,

    # AdamW tends to train attention-heavy backbones more stably than RMSprop.
    optimizer='adamw',
    lr=5e-4,
    weight_decay=1e-4,
    warmup_iters=500,
    scheduler='cosine',           # 'cosine' or 'multistep'
    multistep_milestones=(15, 20),
    multistep_gamma=0.1,

    amp=True,
    grad_clip=1.0,

    # Loss-safety knobs (matches the clamp/cap in EMCStereo.get_loss).
    disp_pred_clamp_factor=2.0,   # clamp preds to ±factor*max_disp
    loss_cap=1.0e3,
    color_aug=True,

    log_interval=10,
    eval_vis_count=10,
    seed=0,
)


# ───────────────────────── misc ─────────────────────────
def _set_seed(seed: int) -> None:
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup_logger(log_dir: Path, name: str = 'EMCStereo') -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_dir / f'{name.lower()}.log')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _disp_to_color(disp: np.ndarray, max_disp: float = 192.0) -> np.ndarray:
    """Colorize a disparity map to an uint8 BGR image for PNG saving."""
    import cv2
    d = np.clip(disp, 0, max_disp) / max_disp * 255.0
    return cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_JET)


# ───────────────────────── loss ─────────────────────────
def compute_loss(preds, disp_gt, max_disp: int, weights=(0.5, 0.7, 1.0),
                 clamp_factor: float = 2.0, loss_cap: float = 1.0e3):
    """NaN/Inf-safe multi-scale smooth_l1.

    1) ignores pixels where disp_gt is not finite, not positive, or ≥ max_disp.
    2) clamps predictions to ±clamp_factor*max_disp — prevents runaway batches.
    3) caps the final loss to `loss_cap` while preserving grad direction.
    """
    mask = torch.isfinite(disp_gt) & (disp_gt > 0) & (disp_gt < max_disp)
    if not mask.any():
        # zero loss that still carries grad wrt every pred
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

    # Hard cap — keep grad direction but shrink magnitude.
    d = loss.detach()
    if float(d) > loss_cap:
        loss = loss * (loss_cap / d).clamp(min=1e-6)

    return loss, {'loss_disp': float(loss.item())}


# ───────────────────────── metrics ─────────────────────────
def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, max_disp: int) -> dict:
    """Comprehensive stereo metrics on valid pixels.

    Returns EPE, D1-all, multiple Bad-N (>Npx) error rates, RMSE, AbsRel,
    MAE, SqRel, log-RMSE, and relative accuracy thresholds (δ<1.25, 1.25², 1.25³).
    """
    mask = torch.isfinite(gt) & (gt > 0) & (gt < max_disp)
    zero = {
        'epe': 0.0, 'd1_all': 0.0,
        'thres_05': 0.0, 'thres_1': 0.0, 'thres_2': 0.0,
        'thres_3': 0.0, 'thres_4': 0.0, 'thres_5': 0.0,
        'rmse': 0.0, 'absrel': 0.0, 'mae': 0.0, 'sqrel': 0.0,
        'rmse_log': 0.0, 'delta1': 0.0, 'delta2': 0.0, 'delta3': 0.0,
        'n_valid': 0,
    }
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


# ───────────────────────── optimizer ─────────────────────────
def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
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


def compute_lr_scale(step: int, total_steps: int, warmup: int,
                     scheduler: str, milestones, gamma: float) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if scheduler == 'cosine':
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
    if scheduler == 'multistep':
        epoch_milestones = [int(m) for m in milestones]
        # Convert epoch milestones -> step milestones externally if needed.
        return 1.0
    return 1.0


# ───────────────────────── train / eval ─────────────────────────
def run_train(cfg, logger) -> Path:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = VirtualTreeDataset(cfg.data_root, cfg.train_split,
                                  transform=build_transform(
                                      'train',
                                      crop_size=tuple(cfg.crop_size),
                                      max_disp=cfg.max_disp,
                                      color_aug=cfg.color_aug))
    val_ds = VirtualTreeDataset(cfg.data_root, cfg.val_split,
                                transform=build_transform(
                                    'val', max_disp=cfg.max_disp))
    logger.info(f'train samples: {len(train_ds)}  val samples: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=cfg.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=max(1, cfg.workers // 2),
                            pin_memory=True)

    # Model
    class _Cfg:
        MAX_DISP = cfg.max_disp
    model = EMCStereo(_Cfg()).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f'EMCStereo params: {n_params:.3f} M')

    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)

    iters_per_epoch = len(train_loader)
    total_steps = cfg.epochs * iters_per_epoch

    # Multi-step milestones → convert epoch → iter.
    if cfg.scheduler == 'multistep':
        step_milestones = [int(m) * iters_per_epoch
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

    run_dir = Path(cfg.output_dir) / 'train' / cfg.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / 'ckpt'
    ckpt_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(str(run_dir / 'tensorboard'))

    (run_dir / 'config.json').write_text(json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v)
         for k, v in vars(cfg).items()},
        indent=2, default=str))

    best_epe = float('inf')
    best_path = ckpt_dir / 'best.pth'
    last_path = ckpt_dir / 'last.pth'
    epochs_since_improve = 0

    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        running = []
        for it, data in enumerate(train_loader):
            scale = lr_scale(global_step)
            for g in optimizer.param_groups:
                g['lr'] = cfg.lr * scale

            left = data['left'].to(device, non_blocking=True)
            right = data['right'].to(device, non_blocking=True)
            disp_gt = data['disp'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg.amp):
                out = model({'left': left, 'right': right, 'disp': disp_gt})
                loss, info = compute_loss(
                    out['train_preds'], disp_gt, cfg.max_disp,
                    clamp_factor=cfg.disp_pred_clamp_factor,
                    loss_cap=cfg.loss_cap)

            if not torch.isfinite(loss):
                logger.warning(
                    f'[ep {epoch} it {it}] non-finite loss, skipping step')
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running.append(float(loss.item()))
            if global_step % cfg.log_interval == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'],
                                  global_step)
                avg = sum(running[-50:]) / max(1, len(running[-50:]))
                logger.info(
                    f'Ep {epoch}/{cfg.epochs} It {it}/{iters_per_epoch} '
                    f'loss={loss.item():.4f}(avg50={avg:.4f}) '
                    f'lr={optimizer.param_groups[0]["lr"]:.2e} '
                    f'elapsed={(time.time() - t0) / 60:.1f}m')
            global_step += 1

        # ── validation ──
        val_metrics = run_eval(model, val_loader, device, cfg.max_disp,
                               save_dir=None, logger=logger, tag=f'ep{epoch}')
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        logger.info(f'Ep {epoch} VAL: ' +
                    ' '.join(f'{k}={v:.4f}' for k, v in val_metrics.items()))

        # save
        ckpt = {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics,
                'global_step': global_step}
        torch.save(ckpt, last_path)
        if val_metrics['epe'] < best_epe:
            best_epe = val_metrics['epe']
            torch.save(ckpt, best_path)
            logger.info(f'=> new best EPE={best_epe:.4f}, saved {best_path}')
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            logger.info(f'=> no improvement for {epochs_since_improve} epoch(s) '
                        f'(best EPE={best_epe:.4f})')
            patience = getattr(cfg, 'early_stop_patience', 0)
            if patience and epochs_since_improve >= patience:
                logger.info(f'Early stopping: no improvement for '
                            f'{epochs_since_improve} epochs (patience={patience}).')
                break

    writer.close()
    logger.info(f'TRAIN DONE. best EPE={best_epe:.4f}, best ckpt: {best_path}')
    return best_path


@torch.no_grad()
def run_eval(model, loader, device, max_disp, save_dir, logger, tag='eval'):
    """Validation / test loop. If `save_dir` is a Path, save the first N
    disparity visualizations (cfg.eval_vis_count) and an eval_results.txt."""
    import cv2
    model.eval()
    agg = {k: 0.0 for k in METRIC_DESCRIPTIONS.keys()}
    n = 0
    saved = 0
    save_limit = 10
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

        # Undo DivisiblePad so metrics / vis use original H, W.
        if 'pad' in data:
            pad = data['pad'][0].tolist()  # [top, right, bottom, left]
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
            logger.info(f'Saved eval results: {save_dir / "eval_results.txt"}')
            logger.info(
                f'Saved {saved} disparity vis: {save_dir / "disparity"}')
    return agg


def run_final_eval(cfg, ckpt_path: Path, logger) -> Path:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class _Cfg:
        MAX_DISP = cfg.max_disp
    model = EMCStereo(_Cfg()).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    logger.info(f'Loaded ckpt {ckpt_path}')

    val_ds = VirtualTreeDataset(cfg.data_root, cfg.val_split,
                                transform=build_transform(
                                    'val', max_disp=cfg.max_disp))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=max(1, cfg.workers // 2),
                            pin_memory=True)

    eval_dir = Path(cfg.output_dir) / 'eval' / cfg.tag
    eval_dir.mkdir(parents=True, exist_ok=True)
    agg = run_eval(model, val_loader, device, cfg.max_disp,
                   save_dir=eval_dir, logger=logger, tag='final')
    logger.info('FINAL EVAL :: ' +
                ' '.join(f'{k}={v:.6f}' for k, v in agg.items()))
    return eval_dir


# ───────────────────────── CLI ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='EMCStereo standalone train/eval')
    p.add_argument('--mode', default='all',
                   choices=['all', 'train', 'eval'])
    p.add_argument('--skip-train', action='store_true',
                   help='alias for --mode eval (requires --ckpt or existing best.pth)')
    p.add_argument('--skip-eval', action='store_true',
                   help='alias for --mode train')

    p.add_argument('--tag', default='default')
    p.add_argument('--data-root', default=DEFAULTS['data_root'])
    p.add_argument('--train-split', default=DEFAULTS['train_split'])
    p.add_argument('--val-split', default=DEFAULTS['val_split'])
    p.add_argument('--output-dir', default=DEFAULTS['output_dir'])

    p.add_argument('--max-disp', type=int, default=DEFAULTS['max_disp'])
    p.add_argument('--crop-size', type=int, nargs=2,
                   default=list(DEFAULTS['crop_size']))

    p.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    p.add_argument('--early-stop-patience', type=int,
                   default=DEFAULTS['early_stop_patience'],
                   help='stop if val EPE does not improve for this many epochs (0=disable)')
    p.add_argument('--batch-size', type=int, default=DEFAULTS['batch_size'])
    p.add_argument('--workers', type=int, default=DEFAULTS['workers'])

    p.add_argument('--optimizer', default=DEFAULTS['optimizer'],
                   choices=['adamw', 'rmsprop'])
    p.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    p.add_argument('--weight-decay', type=float,
                   default=DEFAULTS['weight_decay'])
    p.add_argument('--warmup-iters', type=int,
                   default=DEFAULTS['warmup_iters'])
    p.add_argument('--scheduler', default=DEFAULTS['scheduler'],
                   choices=['cosine', 'multistep'])
    p.add_argument('--multistep-milestones', type=int, nargs='+',
                   default=list(DEFAULTS['multistep_milestones']))
    p.add_argument('--multistep-gamma', type=float,
                   default=DEFAULTS['multistep_gamma'])

    p.add_argument('--amp', type=int, default=int(DEFAULTS['amp']),
                   help='1=enable float16 AMP, 0=fp32')
    p.add_argument('--grad-clip', type=float, default=DEFAULTS['grad_clip'])
    p.add_argument('--disp-pred-clamp-factor', type=float,
                   default=DEFAULTS['disp_pred_clamp_factor'])
    p.add_argument('--loss-cap', type=float, default=DEFAULTS['loss_cap'])
    p.add_argument('--color-aug', type=int,
                   default=int(DEFAULTS['color_aug']))

    p.add_argument('--log-interval', type=int,
                   default=DEFAULTS['log_interval'])
    p.add_argument('--eval-vis-count', type=int,
                   default=DEFAULTS['eval_vis_count'])
    p.add_argument('--seed', type=int, default=DEFAULTS['seed'])
    p.add_argument('--ckpt', default=None, help='eval: ckpt path override')
    p.add_argument('--gpus', default=None,
                   help='sets CUDA_VISIBLE_DEVICES before torch loads CUDA')
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if args.skip_train:
        args.mode = 'eval'
    elif args.skip_eval:
        args.mode = 'train'

    args.amp = bool(args.amp)
    args.color_aug = bool(args.color_aug)
    args.crop_size = tuple(args.crop_size)

    _set_seed(args.seed)
    out_dir = Path(args.output_dir)
    log_dir = out_dir / 'logs'
    logger = _setup_logger(log_dir, name=f'EMCStereo_{args.tag}')
    logger.info('== EMCStereo standalone pipeline ==')
    logger.info(json.dumps({k: str(v)
                for k, v in vars(args).items()}, indent=2))

    best_ckpt = (Path(args.ckpt) if args.ckpt
                 else out_dir / 'train' / args.tag / 'ckpt' / 'best.pth')

    if args.mode in ('all', 'train'):
        best_ckpt = run_train(args, logger)
    if args.mode in ('all', 'eval'):
        if not best_ckpt.exists():
            raise SystemExit(f'Checkpoint not found: {best_ckpt}')
        eval_dir = run_final_eval(args, best_ckpt, logger)
        results = eval_dir / 'eval_results.txt'
        if results.exists():
            print('\n' + '=' * 70)
            print('[EMCStereo] RESULTS')
            print('=' * 70)
            print(results.read_text())

    logger.info('DONE.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
