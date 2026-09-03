#!/usr/bin/env python3
"""Check that the headline eval dump and the panel dump enumerate the same
VirtualTree samples in the same order, by comparing their ground-truth images.

  headline (val) : EMCStereo/output/eval/default/disparity/disp_XXXX.png
                   (1920x2160, prediction on top of ground truth)
  panels   (val) : Others_output_new2/virtualtree/eval/default/panels/gt_XXXX.png
"""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
HEAD = HERE.parent / "EMCStereo" / "output" / "eval" / "default" / "disparity"
PANE = HERE / "Others_output_new2" / "virtualtree" / "eval" / "default" / "panels"


def arr(im):
    return np.asarray(im.convert("RGB"), dtype=np.int16)


print(f"{'i':>3}  {'same index':>12}  {'best match':>12}")
ok = True
for i in range(10):
    im = Image.open(HEAD / f"disp_{i:04d}.png")
    gt_head = arr(im.crop((0, im.height // 2, im.width, im.height)))

    diffs = {}
    for j in range(10):
        p = PANE / f"gt_{j:04d}.png"
        if not p.exists():
            continue
        g = arr(Image.open(p))
        diffs[j] = float(np.abs(gt_head - g).mean()) if g.shape == gt_head.shape else float("inf")

    best = min(diffs, key=diffs.get)
    print(f"{i:>3}  {diffs.get(i, float('nan')):12.4f}  {best:>7} {diffs[best]:8.4f}")
    if best != i or diffs[best] > 0.5:
        ok = False

print("\nALIGNED — panel left_XXXX.png pairs with the headline prediction"
      if ok else "\nNOT ALIGNED — do not pair these dumps by index")
