#!/usr/bin/env python3
"""Run a trained EMCStereo checkpoint on a single rectified stereo pair.

Used to produce the real-world qualitative figure (`EMCStereo_3305.png`) from
the physical ZED Mini capture. Mirrors the val/test transform of
`data/virtualtree.py`: DivisiblePad(32) + ImageNet normalisation, then
soft-argmin disparity, JET-colourised over [0, max_disp].

    python infer_real_pair.py --ckpt <best.pth> --left left_image_3305.png \
        --right right_image_3305.png --out EMCStereo_3305.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.emc_stereo import EMCStereo  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--raw-out', default=None,
                   help='optional .npy path for the raw float disparity')
    p.add_argument('--max-disp', type=int, default=192)
    p.add_argument('--scale', type=float, default=1.0,
                   help='resize factor applied to both views before inference')
    p.add_argument('--threads', type=int, default=0,
                   help='torch CPU threads (0 = leave default)')
    return p.parse_args()


def load_rgb(path: Path, scale: float) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    if scale != 1.0:
        w, h = img.size
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return np.asarray(img)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    x = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    t = torch.from_numpy(np.ascontiguousarray(x))
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)


def jet(disp: np.ndarray, max_disp: float) -> np.ndarray:
    """COLORMAP_JET equivalent without OpenCV, on [0, max_disp] -> uint8 RGB."""
    v = np.clip(disp, 0.0, max_disp) / max_disp  # 0..1
    four = 4.0 * v
    r = np.clip(np.minimum(four - 1.5, -four + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(four - 0.5, -four + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(four + 0.5, -four + 2.5), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def main() -> int:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    left = load_rgb(Path(args.left), args.scale)
    right = load_rgb(Path(args.right), args.scale)
    if left.shape != right.shape:
        raise SystemExit(f'shape mismatch: {left.shape} vs {right.shape}')
    H, W = left.shape[:2]

    # DivisiblePad(32): pad top and right by edge replication, as in eval.
    pad_top = (32 - H % 32) % 32
    pad_right = (32 - W % 32) % 32
    if pad_top or pad_right:
        pad = ((pad_top, 0), (0, pad_right), (0, 0))
        left = np.pad(left, pad, 'edge')
        right = np.pad(right, pad, 'edge')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class _Cfg:
        MAX_DISP = args.max_disp

    model = EMCStereo(_Cfg()).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    n_par = sum(p.numel() for p in model.parameters())
    print(f'[infer] ckpt      : {args.ckpt}', flush=True)
    print(f'[infer] params    : {n_par:,} ({n_par / 1e6:.4f} M)', flush=True)
    print(f'[infer] device    : {device}   threads={torch.get_num_threads()}',
          flush=True)
    print(f'[infer] input     : {W}x{H} -> padded {left.shape[1]}x{left.shape[0]}',
          flush=True)

    t0 = time.time()
    with torch.no_grad():
        out = model({'left': to_tensor(left).to(device),
                     'right': to_tensor(right).to(device)})
        pred = out['disp_pred']
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    disp = pred[0].float().cpu().numpy()
    dt = time.time() - t0
    print(f'[infer] forward   : {dt:.1f}s', flush=True)

    # Un-pad back to the original frame.
    if pad_top:
        disp = disp[pad_top:, :]
    if pad_right:
        disp = disp[:, :-pad_right]

    print(f'[infer] disparity : min={disp.min():.2f} max={disp.max():.2f} '
          f'mean={disp.mean():.2f}', flush=True)

    Image.fromarray(jet(disp, args.max_disp)).save(args.out)
    print(f'[infer] wrote     : {args.out}', flush=True)
    if args.raw_out:
        np.save(args.raw_out, disp.astype(np.float32))
        print(f'[infer] wrote     : {args.raw_out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
