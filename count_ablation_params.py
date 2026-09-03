#!/usr/bin/env python3
"""Count the exact trainable parameters of all eight ablation variants.

Builds the real EMCStereo model and swaps the injected modules the way the
ablation does: EMA/CoordAtt become Identity when off, and turning MSFblock off
restores PSMNet's four-branch concatenation, which widens the first `lastconv`
layer from 224 to 320 input channels.

Cross-checked against the trained checkpoints, which must agree exactly:
    ablation/none      -> 5,225,152
    ablation/ema       -> 5,225,824
    virtualtree (full) -> 5,121,272

Only the full model's checkpoint ships with this repository, so by default
just that row is cross-checked; --ckpt-dir points the other two at a run
tree that holds them.

    python count_ablation_params.py [--ckpt-dir RUN_TREE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.emc_stereo import EMCStereo            # noqa: E402
from src.submodule import conv_bn_relu          # noqa: E402


class _Cfg:
    MAX_DISP = 192


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build(use_ema: bool, use_msf: bool, use_coord: bool) -> nn.Module:
    """The full model with the requested modules removed."""
    model = EMCStereo(_Cfg())
    bb = model.Backbone
    if not use_ema:
        bb.ema = nn.Identity()
    if not use_coord:
        bb.coord_att = nn.Identity()
    if not use_msf:
        # PSMNet concatenates the four 32-channel branches instead of fusing
        # them, so lastconv sees 64 + 128 + 4*32 = 320 channels.
        del bb.msf
        bb.lastconv[0] = conv_bn_relu(True, 320, 128, 3, 1, 1, 1, bias=False)
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt-dir', default='Others_output_new2')
    ap.add_argument('--full-ckpt',
                    default='checkpoints/emcstereo_virtualtree_best.pth',
                    help='checkpoint for the full (1,1,1) model')
    a = ap.parse_args()

    rows = [(e, m, c) for e in (0, 1) for m in (0, 1) for c in (0, 1)]
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    counts = {}
    print(f"{'EMA':>4}{'MSF':>5}{'Coord':>7}{'params':>12}{'M (3dp)':>10}"
          f"{'vs base':>10}")
    print('-' * 48)
    base = n_params(build(False, False, False))
    for e, m, c in rows:
        p = n_params(build(bool(e), bool(m), bool(c)))
        counts[(e, m, c)] = p
        print(f"{e:>4}{m:>5}{c:>7}{p:>12,}{p / 1e6:>10.3f}"
              f"{p - base:>+10,}")

    print('-' * 48)
    print(f"saving of the full model: {base - counts[(1, 1, 1)]:,} "
          f"({100 * (base - counts[(1, 1, 1)]) / base:.3f}%)")

    # Cross-check against the trained checkpoints.
    BUF = ('running_mean', 'running_var', 'num_batches_tracked')
    ck = {
        (0, 0, 0): f'{a.ckpt_dir}/ablation/none/train/default/ckpt/best.pth',
        (1, 0, 0): f'{a.ckpt_dir}/ablation/ema/train/default/ckpt/best.pth',
        (1, 1, 1): (a.full_ckpt
                    if Path(a.full_ckpt).is_file()
                    else f'{a.ckpt_dir}/virtualtree/train/default/ckpt/best.pth'),
    }
    print('\ncheckpoint cross-check')
    ok = True
    for key, rel in ck.items():
        path = HERE / rel
        if not path.exists():
            print(f"  {key}  SKIP (not found: {rel})")
            continue
        sd = torch.load(path, map_location='cpu', weights_only=False)['model']
        got = sum(v.numel() for k, v in sd.items() if not k.endswith(BUF))
        mark = 'OK ' if got == counts[key] else 'MISMATCH'
        ok &= got == counts[key]
        print(f"  {key}  built {counts[key]:,}  ckpt {got:,}   {mark}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
