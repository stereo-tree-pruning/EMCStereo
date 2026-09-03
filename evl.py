#!/usr/bin/env python3
"""EMCStereo standalone evaluation script.

Usage:
    python evl.py --ckpt checkpoints/emcstereo_virtualtree_best.pth
    python evl.py --ckpt output/train/default/ckpt/best.pth --tag default
    python evl.py --split test        # evaluate on test split instead of val
    python evl.py --gpus 0            # pick a specific GPU
    python evl.py --save-count 20     # save 20 disparity PNGs instead of 10

Runs validation on VirtualTree using a trained checkpoint and:
    1. Computes EPE / d1_all / thres_1 / thres_2 / thres_3 on the whole split
    2. Saves the FIRST N disparity visualizations (default N=10) as
       output/eval/{tag}/disparity/disp_0000.png ... disp_{N-1:04d}.png
       (each PNG is top=prediction, bottom=ground truth, both JET-colored)
    3. Writes output/eval/{tag}/eval_results.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.emc_stereo import EMCStereo                          # noqa: E402
from data.virtualtree import VirtualTreeDataset, build_transform  # noqa: E402


# ───────────────────────── helpers ─────────────────────────
def disp_to_color(disp: np.ndarray, max_disp: float = 192.0) -> np.ndarray:
    """Colorize a disparity map to uint8 BGR for saving as PNG."""
    d = np.clip(disp, 0, max_disp) / max_disp * 255.0
    return cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_JET)


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, max_disp: int) -> dict:
    mask = torch.isfinite(gt) & (gt > 0) & (gt < max_disp)
    if not mask.any():
        return {'epe': 0.0, 'd1_all': 0.0, 'thres_1': 0.0,
                'thres_2': 0.0, 'thres_3': 0.0, 'n_valid': 0}
    p, g = pred[mask], gt[mask]
    err = (p - g).abs()
    return {
        'epe': err.mean().item(),
        'd1_all': (((err > 3) & (err / g.clamp(min=1e-6) > 0.05)).float().mean().item()),
        'thres_1': (err > 1).float().mean().item(),
        'thres_2': (err > 2).float().mean().item(),
        'thres_3': (err > 3).float().mean().item(),
        'n_valid': int(mask.sum().item()),
    }


def unpad(pred: torch.Tensor, gt: torch.Tensor, pad):
    """Undo DivisiblePad so metrics / visualization use original H, W."""
    pt, pr, _, _ = [int(x) for x in pad]
    if pt:
        pred = pred[:, pt:, :]
        gt = gt[:, pt:, :]
    if pr:
        pred = pred[:, :, :-pr]
        gt = gt[:, :, :-pr]
    return pred, gt


# ───────────────────────── main ─────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='EMCStereo standalone evaluation')
    p.add_argument('--tag', default='default',
                   help='experiment tag; ckpt/output paths are derived from this')
    p.add_argument('--ckpt', default=None,
                   help='explicit checkpoint path (default: output/train/{tag}/ckpt/best.pth)')
    p.add_argument('--split', default='val', choices=['val', 'test'],
                   help='which split file under data/ to evaluate on')
    p.add_argument('--split-file', default=None,
                   help='explicit split .txt path (overrides --split)')
    p.add_argument('--data-root',
                   default=os.environ.get(
                       'EMCSTEREO_DATA_ROOT',
                       str(HERE / 'datasets' / 'Virtual_branches_data')))
    p.add_argument('--max-disp', type=int, default=192)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--save-count', type=int, default=10,
                   help='number of disparity visualizations to save')
    p.add_argument('--output-dir', default=str(HERE / 'output'))
    p.add_argument('--gpus', default=None,
                   help='comma-separated GPU id(s); sets CUDA_VISIBLE_DEVICES')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)

    # Resolve ckpt path.
    ckpt_path = (Path(args.ckpt) if args.ckpt
                 else out_dir / 'train' / args.tag / 'ckpt' / 'best.pth')
    if not ckpt_path.exists():
        raise SystemExit(f'[EMCStereo eval] checkpoint not found: {ckpt_path}')

    # Resolve split file.
    if args.split_file:
        split_file = Path(args.split_file)
    else:
        split_file = HERE / 'data' / f'virtualtree_{args.split}.txt'
    if not split_file.exists():
        raise SystemExit(
            f'[EMCStereo eval] split file not found: {split_file}')

    # Dataset / loader.
    ds = VirtualTreeDataset(args.data_root, str(split_file),
                            transform=build_transform('val', max_disp=args.max_disp))
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == 'cuda'))

    # Model.
    class _Cfg:
        MAX_DISP = args.max_disp
    model = EMCStereo(_Cfg()).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model'] if isinstance(
        ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    # Output dirs.
    eval_dir = out_dir / 'eval' / args.tag
    disp_dir = eval_dir / 'disparity'
    disp_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'[EMCStereo eval]')
    print(f'  ckpt        : {ckpt_path}')
    print(f'  split       : {args.split}  ({len(ds)} samples)')
    print(f'  device      : {device}')
    print(f'  params      : {n_params:.3f} M')
    print(f'  max_disp    : {args.max_disp}')
    print(f'  save_count  : {args.save_count}  -> {disp_dir}')
    print('=' * 70, flush=True)

    agg = {'epe': 0.0, 'd1_all': 0.0, 'thres_1': 0.0,
           'thres_2': 0.0, 'thres_3': 0.0}
    n = 0
    saved = 0
    t0 = time.time()

    with torch.no_grad():
        for idx, data in enumerate(loader):
            left = data['left'].to(device, non_blocking=True)
            right = data['right'].to(device, non_blocking=True)
            gt = data['disp'].to(device, non_blocking=True)

            out = model({'left': left, 'right': right, 'disp': gt})
            pred = out['disp_pred']
            if pred.dim() == 4:
                pred = pred.squeeze(1)

            if 'pad' in data:
                pred, gt = unpad(pred, gt, data['pad'][0].tolist())

            m = compute_metrics(pred, gt, args.max_disp)
            for k in agg:
                agg[k] += m[k]
            n += 1

            if saved < args.save_count:
                pred_png = disp_to_color(
                    pred[0].float().cpu().numpy(), max_disp=args.max_disp)
                gt_png = disp_to_color(
                    gt[0].float().cpu().numpy(), max_disp=args.max_disp)
                # Separator bar between pred and GT for readability.
                sep = np.full((4, pred_png.shape[1], 3), 255, dtype=np.uint8)
                combo = np.concatenate([pred_png, sep, gt_png], axis=0)
                cv2.imwrite(str(disp_dir / f'disp_{saved:04d}.png'), combo)
                saved += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == len(loader):
                avg_epe = agg['epe'] / n
                print(f'  [{idx + 1:4d}/{len(loader)}] running EPE={avg_epe:.4f}',
                      flush=True)

    agg = {k: v / max(1, n) for k, v in agg.items()}
    elapsed = time.time() - t0

    # Write eval_results.txt.
    lines = [f'EMCStereo evaluation on VirtualTree [{args.split}]',
             f'checkpoint: {ckpt_path}',
             f'samples   : {n}',
             f'elapsed   : {elapsed:.1f}s  ({elapsed / max(1, n) * 1000:.1f} ms/sample)',
             '-' * 60]
    lines += [f'{k:10s}: {v:.6f}' for k, v in agg.items()]
    lines.append('-' * 60)
    lines.append(f'saved {saved} disparity visualizations at {disp_dir}')
    results_txt = '\n'.join(lines) + '\n'
    (eval_dir / 'eval_results.txt').write_text(results_txt)

    print()
    print(results_txt)
    print(f'[EMCStereo eval] results.txt : {eval_dir / "eval_results.txt"}')
    print(f'[EMCStereo eval] disparity   : {disp_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
