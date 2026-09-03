#!/usr/bin/env python3
"""Build qual_sceneflow.png for the paper.

Each eval visualisation written by evl.py is a 960x1080 PNG holding the
prediction on top and the ground truth underneath, both JET-coloured over
[0, 192].  This script splits two of them and lays the panels out as

    GT(scene A) | EMCStereo(scene A)
    GT(scene B) | EMCStereo(scene B)

so the figure can be dropped into a single paper column.
"""
import os

from PIL import Image, ImageDraw, ImageFont

SRC = os.path.join("Others_output_v4", "Scene_Flow", "eval", "disparity")
SCENES = ["disp_0007.png", "disp_0006.png"]
PANEL_W, PANEL_H = 720, 405
GAP = 10
OUT = "qual_sceneflow.png"


def split(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    pred = im.crop((0, 0, w, h // 2)).resize((PANEL_W, PANEL_H), Image.LANCZOS)
    gt = im.crop((0, h // 2, w, h)).resize((PANEL_W, PANEL_H), Image.LANCZOS)
    return gt, pred


rows = [split(os.path.join(SRC, name)) for name in SCENES]

W = 2 * PANEL_W + GAP
H = len(rows) * PANEL_H + (len(rows) - 1) * GAP
sheet = Image.new("RGB", (W, H), "white")
for r, (gt, pred) in enumerate(rows):
    y = r * (PANEL_H + GAP)
    sheet.paste(gt, (0, y))
    sheet.paste(pred, (PANEL_W + GAP, y))

# Column labels are drawn inside the top row so the figure costs no extra
# vertical space in the paper.
try:
    font = ImageFont.truetype("arialbd.ttf", 34)
except OSError:
    font = ImageFont.load_default()

draw = ImageDraw.Draw(sheet)
for x, text in ((0, "Ground truth"), (PANEL_W + GAP, "EMCStereo (ours)")):
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((x + 10, 10, x + 24 + box[2], 22 + box[3]), fill=(0, 0, 0))
    draw.text((x + 17, 16), text, fill=(255, 255, 255), font=font)

sheet.save(OUT)
print("wrote", OUT, sheet.size)
