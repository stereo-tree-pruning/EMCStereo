#!/usr/bin/env python3
"""Build qual_synth.png: VirtualTree scenes as left RGB | ground truth | EMCStereo.

Sources (both enumerate the VirtualTree validation split in the same order --
``check_panel_alignment.py`` confirms their ground-truth images are identical
pixel for pixel at every index):

  ../EMCStereo/output/eval/default/disparity/disp_XXXX.png
      the headline checkpoint's saved visualisation, 1920x2160 with the
      prediction on top of the ground truth, JET-coloured over [0, 192].  Its
      eval_results.txt reports epe 1.314849 / d1_all 0.058010 / rmse 4.825643,
      which identifies it as the deployed model of Table II.
  Others_output_new2/virtualtree/eval/default/panels/left_XXXX.png
      the matching left input frame.

Four scenes are laid out two per row, so the figure shows twice as many
examples as the old two-scene version in less vertical space.

    python make_virtualtree_figure.py             # write qual_synth.png
    python make_virtualtree_figure.py --contact    # preview all ten scenes
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DISP = HERE.parent / "EMCStereo" / "output" / "eval" / "default" / "disparity"
LEFT = HERE / "Others_output_new2" / "virtualtree" / "eval" / "default" / "panels"

# Two views with a near branch spanning much of the search range (7, 5) and two
# canopy views dominated by thin twigs (6, 8), out of the ten saved samples.
SCENES = [7, 5, 6, 8]
HEADERS = ["Left image", "Ground truth", "EMCStereo"]
PANEL_W = 400
GAP = 7
LABEL_H = 26


def _font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def scene(idx, width=PANEL_W):
    """left RGB, ground truth, prediction for one saved validation sample."""
    dp, lp = DISP / f"disp_{idx:04d}.png", LEFT / f"left_{idx:04d}.png"
    for p in (dp, lp):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")
    im = Image.open(dp).convert("RGB")
    h = im.height // 2                       # top = prediction, bottom = GT
    pred = im.crop((0, 0, im.width, h))
    gt = im.crop((0, h, im.width, im.height))
    left = Image.open(lp).convert("RGB")
    if left.size != gt.size:
        raise SystemExit(f"size mismatch for scene {idx}: {left.size} vs {gt.size}")
    out = []
    for p in (left, gt, pred):
        out.append(p.resize((width, round(p.height * width / p.width)), Image.LANCZOS))
    return out


def contact(out):
    tiles = [scene(i, 260) for i in range(10)]
    tw, th = tiles[0][0].size
    sheet = Image.new("RGB", (3 * tw * 2 + 20, 5 * th), "white")
    draw = ImageDraw.Draw(sheet)
    for k, row in enumerate(tiles):
        x0 = (k % 2) * (3 * tw + 20)
        y = (k // 2) * th
        for j, im in enumerate(row):
            sheet.paste(im, (x0 + j * tw, y))
        draw.text((x0 + 4, y + 4), str(k), fill="white", font=_font(22))
    sheet.save(out)
    print("wrote", out, sheet.size)


def build(out):
    rows = [scene(i) for i in SCENES]
    tile_h = rows[0][0].height
    ncol = 2 * len(HEADERS)
    W = ncol * PANEL_W + (ncol - 1) * GAP
    H = LABEL_H + 2 * tile_h + GAP
    sheet = Image.new("RGB", (W, H), "white")

    for k, row in enumerate(rows):
        r, c = divmod(k, 2)                  # two scenes per row
        y = LABEL_H + r * (tile_h + GAP)
        for j, im in enumerate(row):
            sheet.paste(im, ((3 * c + j) * (PANEL_W + GAP), y))

    draw = ImageDraw.Draw(sheet)
    font = _font(21)
    for j, head in enumerate(HEADERS * 2):
        x = j * (PANEL_W + GAP)
        tw = draw.textlength(head, font=font)
        draw.text((x + (PANEL_W - tw) / 2, 2), head, fill="black", font=font)

    sheet.save(out)
    print(f"[fig] wrote {out}  {sheet.size}  (validation scenes {SCENES})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--contact"]
    target = Path(args[0]) if args else HERE / "qual_synth.png"
    (contact if "--contact" in sys.argv else build)(target)
