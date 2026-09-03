#!/usr/bin/env python3
"""Build qualitative figures for main.tex.

    NOTE: qual_synth.png is now built by make_virtualtree_figure.py instead --
    the test-split dumps this script needs are not in this checkout. Only
    --real is still usable here.

    qual_synth.png      VirtualTree test scenes: left RGB | ground truth |
                        EMCStereo prediction. Ground truth and prediction come
                        from the headline run's saved visualisations
                        (new_results/output/eval/test/disparity/disp_XXXX.png,
                        prediction on top of ground truth in one image); the
                        left RGB frames come from the panel dump of the same
                        split, whose sample order is identical (verified
                        pixel-exact on the shared ground-truth halves).
    EMCStereo_3305.png  Real ZED Mini pair, re-colourised from the raw
                        disparity produced by infer_real_pair.py so that it is
                        directly comparable to the DEFOM-Stereo reference.

    python make_paper_figures.py --all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DISP = HERE / "new_results" / "output" / "eval" / "test" / "disparity"
LEFT = HERE / "Others_output_new2" / "virtualtree" / "eval" / "test" / "panels"

# Two of the ten saved test scenes: a typical thin-branch canopy view and one
# with a near branch spanning most of the disparity range.
SCENES = [4, 1]
HEADERS = ["Left image", "Ground truth", "EMCStereo"]

TILE_W = 560          # per-column width in the assembled sheet
GAP = 6               # gutter between tiles
LABEL_H = 30          # header strip height


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "seguisb.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _scene(idx: int) -> list[Image.Image]:
    """left RGB, ground truth, prediction for one test scene."""
    dp = DISP / f"disp_{idx:04d}.png"
    lp = LEFT / f"left_{idx:04d}.png"
    for p in (dp, lp):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")
    disp = Image.open(dp).convert("RGB")
    h = disp.height // 2                      # top = prediction, bottom = GT
    pred = disp.crop((0, 0, disp.width, h))
    gt = disp.crop((0, h, disp.width, disp.height))
    left = Image.open(lp).convert("RGB")
    if left.size != gt.size:
        raise SystemExit(f"size mismatch for scene {idx}: {left.size} vs {gt.size}")
    return [left, gt, pred]


def build_qual_synth(out: Path) -> None:
    tiles = []
    for s in SCENES:
        row = []
        for im in _scene(s):
            h = round(im.height * TILE_W / im.width)
            row.append(im.resize((TILE_W, h), Image.LANCZOS))
        tiles.append(row)

    tile_h = tiles[0][0].height
    ncol, nrow = len(HEADERS), len(SCENES)
    W = ncol * TILE_W + (ncol - 1) * GAP
    H = LABEL_H + nrow * tile_h + (nrow - 1) * GAP
    sheet = Image.new("RGB", (W, H), "white")

    draw = ImageDraw.Draw(sheet)
    font = _font(22)
    for j, head in enumerate(HEADERS):
        x = j * (TILE_W + GAP)
        tw = draw.textlength(head, font=font)
        draw.text((x + (TILE_W - tw) / 2, 3), head, fill="black", font=font)

    for i, row in enumerate(tiles):
        y = LABEL_H + i * (tile_h + GAP)
        for j, im in enumerate(row):
            sheet.paste(im, (j * (TILE_W + GAP), y))

    sheet.save(out)
    print(f"[fig] wrote {out}  {sheet.size}  (test scenes {SCENES})")


def build_real(raw: Path, out: Path, lo: float, hi: float) -> None:
    from matplotlib import colormaps

    d = np.load(raw)
    v = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    rgb = (colormaps["magma"](v)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(out)
    print(f"[fig] wrote {out}  colour-mapped over [{lo:g}, {hi:g}] px  "
          f"(actual range {d.min():.2f}-{d.max():.2f}, median {np.median(d):.2f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--raw", default=str(HERE / "EMCStereo_3305_disp.npy"))
    ap.add_argument("--lo", type=float, default=0.0,
                    help="low end of the disparity colour range, in px")
    ap.add_argument("--hi", type=float, default=50.0,
                    help="high end of the disparity colour range, in px")
    a = ap.parse_args()
    if a.all:
        a.synth = a.real = True
    if a.synth:
        build_qual_synth(HERE / "qual_synth.png")
    if a.real:
        build_real(Path(a.raw), HERE / "EMCStereo_3305.png", a.lo, a.hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
