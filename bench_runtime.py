#!/usr/bin/env python3
"""Measure EMCStereo forward-pass cost, for the runtime numbers in the paper.

    python bench_runtime.py --ckpt <best.pth> --height 1080 --width 1920 --runs 5
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from src.emc_stereo import EMCStereo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--max-disp', type=int, default=192)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--modules', action='store_true',
                    help='time the three injected modules instead of the net')
    a = ap.parse_args()

    if a.modules:
        bench_modules(a.height + (32 - a.height % 32) % 32,
                      a.width + (32 - a.width % 32) % 32)
        return 0

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cuda':
        print(f'[bench] gpu    : {torch.cuda.get_device_name(0)}')
        print(f'[bench] vram   : '
              f'{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB')
    print(f'[bench] torch  : {torch.__version__}')

    class _Cfg:
        MAX_DISP = a.max_disp

    model = EMCStereo(_Cfg()).to(dev).eval()
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'] if 'model' in ck else ck)
    n = sum(p.numel() for p in model.parameters())
    print(f'[bench] params : {n:,} ({n / 1e6:.4f} M)')

    H = a.height + (32 - a.height % 32) % 32
    W = a.width + (32 - a.width % 32) % 32
    left = torch.randn(1, 3, H, W, device=dev)
    right = torch.randn(1, 3, H, W, device=dev)
    print(f'[bench] input  : {a.width}x{a.height} -> padded {W}x{H}')

    with torch.no_grad():
        for _ in range(a.warmup):
            model({'left': left, 'right': right})
        if dev.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(a.runs):
            t0 = time.perf_counter()
            model({'left': left, 'right': right})
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)

    ts = sorted(ts)
    mean = sum(ts) / len(ts)
    print(f'[bench] latency: mean {mean * 1000:.0f} ms   median '
          f'{ts[len(ts) // 2] * 1000:.0f} ms   min {ts[0] * 1000:.0f} ms   '
          f'({1 / mean:.2f} FPS)')
    if dev.type == 'cuda':
        print(f'[bench] peak vram: '
              f'{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')
    return 0


# ── module-level cost, invoked as: python bench_runtime.py --modules ──────────
def bench_modules(H: int, W: int, runs: int = 20) -> None:
    """Time the three injected modules at the 1/4-resolution feature size."""
    import torch.nn as nn
    from src.attention_modules import EMA, CoordAtt, MSFblock

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h, w = H // 4, W // 4
    print(f'[modules] feature map: {h}x{w}  (1/4 of {W}x{H})')

    cases = {
        'EMA(128)': (EMA(channels=128, factor=16).to(dev).eval(),
                     (torch.randn(1, 128, h, w, device=dev),)),
        'MSFblock(32)': (MSFblock(in_channels=32).to(dev).eval(),
                         tuple(torch.randn(1, 32, h, w, device=dev)
                               for _ in range(4))),
        'CoordAtt(32)': (CoordAtt(inp=32, oup=32, reduction=8).to(dev).eval(),
                         (torch.randn(1, 32, h, w, device=dev),)),
    }
    total = 0.0
    with torch.no_grad():
        for name, (mod, args) in cases.items():
            for _ in range(3):
                mod(*args)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(runs):
                mod(*args)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / runs * 1000
            total += ms
            n = sum(p.numel() for p in mod.parameters())
            print(f'[modules] {name:<14} {n:>6,} params   {ms:7.2f} ms')
    print(f'[modules] {"total":<14} {"":>6}          {total:7.2f} ms')


if __name__ == "__main__":
    raise SystemExit(main())
