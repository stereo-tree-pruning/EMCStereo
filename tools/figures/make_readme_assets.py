#!/usr/bin/env python3
"""Build every image in ../../assets/ for the README.

Sources, all outside this repository (they are the raw evaluation dumps and the
paper's own figure inputs), resolved relative to the *project* root -- the
directory that holds both this repository and the EMCStereo run tree:

  <project>/EMCStereo/output/eval/default/disparity/disp_XXXX.png
        the deployed checkpoint's VirtualTree validation visualisations,
        1920x2160 with the prediction on top of the ground truth, JET over
        [0, 192].  Its eval_results.txt reports epe 1.314849, which identifies
        it as the checkpoint released in checkpoints/.
  <project>/icra_2027_paper3/Others_output_new2/virtualtree/eval/default/panels/left_XXXX.png
        the matching left input frames.  These are model-independent, and
        check_panel_alignment.py proves the two dumps enumerate the split in
        the same order.
  <project>/trees/{left,right}_2988.png, depth_2988.exr
        one raw VirtualTree *test*-split sample (line 1 of
        data/virtualtree_test.txt), used for the dataset figure.
  <project>/icra_2027_paper3/{left_image_3305,EMCStereo_3305,DEFOM-Stereo_3305}.png
        the real ZED Mini pair and the two disparity maps of the paper figure.
  <project>/icra_2027_paper3/{flow_chart,qual_synth,qual_sceneflow}.png
        the paper's architecture and qualitative figures.
  <project>/icra_2027_paper3/final_results/virtualtree/emcstereo_headline/logs/
        the deployed run's training log, for the validation curve.

    python tools/figures/make_readme_assets.py            # build everything
    python tools/figures/make_readme_assets.py --only curve ablation
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent                      # .../icra_2027_paper3
WORKSPACE = PROJECT.parent                 # holds EMCStereo/, trees/, ...
ASSETS = REPO / 'assets'

DISP = WORKSPACE / 'EMCStereo' / 'output' / 'eval' / 'default' / 'disparity'
LEFT = (PROJECT / 'Others_output_new2' / 'virtualtree' / 'eval' / 'default'
        / 'panels')
TREES = WORKSPACE / 'trees'
LOG = (PROJECT / 'final_results' / 'virtualtree' / 'emcstereo_headline'
       / 'logs' / 'emcstereo_default.log')

# VirtualTree rig (see data/virtualtree.py).
BASELINE_CM, FOCAL_PX, MAX_DISP = 6.3, 960.0, 192

INK = (26, 29, 33)
GAP = 8
LABEL_H = 30


# ───────────────────────────── helpers ─────────────────────────────
def font(size, bold=True):
    names = (('arialbd.ttf', 'DejaVuSans-Bold.ttf') if bold
             else ('arial.ttf', 'DejaVuSans.ttf'))
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def need(*paths):
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit('missing input(s):\n  ' + '\n  '.join(map(str, missing)))


def fit(im, width):
    return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)


def on_white(path, width=None):
    """Load a PNG, flattening any alpha onto white (transparent diagrams are
    unreadable on GitHub's dark theme)."""
    im = Image.open(path)
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert('RGB')
    return fit(im, width) if width else im


def scene(idx, width):
    """left RGB, ground truth, prediction for one saved validation sample."""
    dp, lp = DISP / f'disp_{idx:04d}.png', LEFT / f'left_{idx:04d}.png'
    need(dp, lp)
    im = Image.open(dp).convert('RGB')
    h = im.height // 2                       # top = prediction, bottom = GT
    pred, gt = im.crop((0, 0, im.width, h)), im.crop((0, h, im.width, im.height))
    left = Image.open(lp).convert('RGB')
    if left.size != gt.size:
        raise SystemExit(f'size mismatch for scene {idx}: {left.size} vs {gt.size}')
    return [fit(p, width) for p in (left, gt, pred)]


def sheet(rows, headers, panel_w, per_row=2, title_size=21):
    """Lay tiles out as `per_row` scenes across, each scene one row of tiles."""
    tile_h = rows[0][0].height
    ncol = per_row * len(headers)
    nrow = -(-len(rows) // per_row)
    W = ncol * panel_w + (ncol - 1) * GAP
    H = LABEL_H + nrow * tile_h + (nrow - 1) * GAP
    out = Image.new('RGB', (W, H), 'white')
    for k, row in enumerate(rows):
        r, c = divmod(k, per_row)
        y = LABEL_H + r * (tile_h + GAP)
        for j, im in enumerate(row):
            out.paste(im, ((len(headers) * c + j) * (panel_w + GAP), y))
    draw = ImageDraw.Draw(out)
    f = font(title_size)
    for j, head in enumerate(headers * per_row):
        x = j * (panel_w + GAP)
        w = draw.textlength(head, font=f)
        draw.text((x + (panel_w - w) / 2, 4), head, fill=INK, font=f)
    return out


def jet(disp, max_disp=MAX_DISP):
    d = np.clip(disp, 0, max_disp) / max_disp * 255.0
    bgr = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def save(im, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    p = ASSETS / name
    im.save(p, optimize=True)
    print(f'wrote {p.relative_to(REPO)}  {im.size[0]}x{im.size[1]}  '
          f'{p.stat().st_size / 1e6:.2f} MB')


# ───────────────────────────── builders ─────────────────────────────
def build_architecture():
    """The EMCStereo block diagram, flattened onto white."""
    src = PROJECT / 'flow_chart.png'
    need(src)
    save(on_white(src, 2200), 'architecture.png')


def build_qual_virtualtree():
    """The paper's Fig. 2 (validation scenes 7, 5, 6, 8)."""
    src = PROJECT / 'qual_synth.png'
    need(src)
    save(on_white(src), 'qual_virtualtree.png')


def build_qual_gallery():
    """All ten saved validation scenes — the complete dump, nothing selected."""
    rows = [scene(i, 300) for i in range(10)]
    save(sheet(rows, ['Left image', 'Ground truth', 'EMCStereo'], 300,
               title_size=17),
         'qual_virtualtree_gallery.png')


def build_qual_sceneflow():
    src = PROJECT / 'qual_sceneflow.png'
    need(src)
    save(on_white(src), 'qual_sceneflow.png')


def build_qual_real():
    """The real ZED Mini figure, composed into one labelled image."""
    paths = [PROJECT / n for n in ('left_image_3305.png', 'EMCStereo_3305.png',
                                   'DEFOM-Stereo_3305.png')]
    need(*paths)
    w = 620
    tiles = [on_white(p, w) for p in paths]
    heads = ['Left input (physical ZED Mini)', 'EMCStereo (ours)',
             'DEFOM-Stereo']
    save(sheet([tiles], heads, w, per_row=1), 'qual_real.png')


def build_virtualtree_sample():
    """One raw VirtualTree test sample: left, right and ground-truth disparity."""
    lp, rp, dp = (TREES / 'left_2988.png', TREES / 'right_2988.png',
                  TREES / 'depth_2988.exr')
    need(lp, rp, dp)
    exr = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
    if exr is None:
        raise SystemExit(f'OpenCV could not read {dp} (OpenEXR codec missing)')
    depth = (exr[:, :, 2] if exr.ndim == 3 else exr).astype(np.float32)
    disp = np.zeros_like(depth)
    valid = np.isfinite(depth) & (depth > 0)
    disp[valid] = FOCAL_PX * BASELINE_CM / depth[valid]

    w = 620
    tiles = [on_white(lp, w), on_white(rp, w), fit(jet(disp), w)]
    heads = ['Left view', 'Right view',
             'Ground-truth disparity (from the UE5 EXR depth)']
    save(sheet([tiles], heads, w, per_row=1), 'virtualtree_sample.png')


def build_detail(idx=7, box=(520, 120, 1480, 660)):
    """A zoomed crop of one branch fork: the silhouette and the twigs behind it."""
    dpath, lpath = DISP / f'disp_{idx:04d}.png', LEFT / f'left_{idx:04d}.png'
    need(dpath, lpath)
    full = Image.open(dpath).convert('RGB')
    h = full.height // 2
    pred, gt = full.crop((0, 0, full.width, h)), full.crop((0, h, full.width,
                                                            full.height))
    left = Image.open(lpath).convert('RGB')
    w = 620
    tiles = [fit(p.crop(box), w) for p in (left, gt, pred)]
    heads = ['Left image (detail)', 'Ground truth', 'EMCStereo']
    save(sheet([tiles], heads, w, per_row=1), 'thin_structure_detail.png')


def _curve_points():
    """Validation EPE per epoch of the *deployed* run.

    The log file holds two pipeline invocations end to end: an earlier
    182-epoch run that bottoms out at 1.5207 px and produced the warm-start
    weights, then the 300-epoch run that reaches 1.3148 px at epoch 252 and
    wrote the released checkpoint. Only the last segment is the deployed run,
    so the curve is cut at the final '== EMCStereo standalone pipeline ==='
    banner rather than merged across both.
    """
    need(LOG)
    segment = []
    for line in LOG.read_text(encoding='utf-8', errors='replace').splitlines():
        if 'EMCStereo standalone pipeline' in line:
            segment = []
        m = re.search(r'Ep (\d+) VAL: epe=([0-9.]+)', line)
        if m:
            segment.append((int(m.group(1)), float(m.group(2))))
    if not segment:
        raise SystemExit(f'no "Ep N VAL:" lines in {LOG}')
    return [e for e, _ in segment], [v for _, v in segment]


def build_curve():
    """Validation EPE against epoch for the deployed 300-epoch run."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ep, epe = _curve_points()
    best = min(range(len(ep)), key=lambda i: epe[i])

    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=170)
    ax.plot(ep, epe, lw=1.5, color='#2a6fdb', label='Validation EPE')
    ax.scatter([ep[best]], [epe[best]], s=46, zorder=5, color='#d1495b',
               edgecolor='white', linewidth=1.2)
    ax.annotate(f'best  {epe[best]:.4f} px\nepoch {ep[best]}  (released checkpoint)',
                xy=(ep[best], epe[best]), xytext=(-14, 34),
                textcoords='offset points', ha='right', fontsize=9,
                color='#d1495b',
                arrowprops=dict(arrowstyle='-', color='#d1495b', lw=1))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('VirtualTree validation EPE (px)')
    ax.set_title('EMCStereo — deployed run (300 epochs, warm-started)',
                 fontsize=11, color=  '#1a1d21')
    ax.set_ylim(1.2, 3.0)
    ax.set_xlim(0, max(ep))
    ax.grid(alpha=.25, lw=.7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / 'training_curve.png'
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    print(f'wrote {out.relative_to(REPO)}  '
          f'{out.stat().st_size / 1e6:.2f} MB  ({len(ep)} epochs)')


# EPE of the eight grid runs, plus the independent repeat of the baseline that
# fixes the noise floor. Values from results/expected_metrics.json.
ABLATION = [
    ('none',           1.748995),
    ('EMA',            1.739770),
    ('MSF',            1.776870),
    ('Coord',          1.785270),
    ('EMA+MSF',        1.737644),
    ('EMA+Coord',      1.735543),
    ('MSF+Coord',      1.791517),
    ('EMA+MSF+Coord',  1.752222),
]
REPEAT = 1.740423


def build_ablation():
    """The 8-way grid against the run-to-run noise floor."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [n for n, _ in ABLATION]
    vals = [v for _, v in ABLATION]
    base = ABLATION[0][1]
    floor = abs(base - REPEAT)

    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=170)
    ax.axhspan(min(base, REPEAT) - floor, max(base, REPEAT) + floor,
               color='#9aa5b1', alpha=.22, lw=0,
               label=f'baseline ± noise floor ({floor:.3f} px)')
    ax.axhline(base, color='#6b7280', lw=1, ls='--')
    # inside the baseline's noise band -> blue, outside -> grey.
    lo, hi = min(base, REPEAT) - floor, max(base, REPEAT) + floor
    colors = ['#2a6fdb' if lo <= v <= hi else '#9aa5b1' for v in vals]
    colors[0] = '#4b5563'                 # the no-attention baseline
    colors[-1] = '#d1495b'                # the full model
    ax.bar(names, vals, color=colors, width=.62)
    for i, v in enumerate(vals):
        ax.text(i, v + .004, f'{v:.3f}', ha='center', fontsize=8.5,
                color='#1a1d21')
    ax.set_ylim(1.70, 1.81)
    ax.set_ylabel('VirtualTree validation EPE (px)')
    ax.set_title('Module ablation at a matched 100-epoch budget\n'
                 'only the EMA-bearing variants stay inside the noise floor',
                 fontsize=10.5, color='#1a1d21', linespacing=1.5)
    ax.tick_params(axis='x', labelrotation=18, labelsize=8.5)
    ax.grid(axis='y', alpha=.25, lw=.7)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc='upper left')
    fig.tight_layout()
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / 'ablation_chart.png'
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    print(f'wrote {out.relative_to(REPO)}  {out.stat().st_size / 1e6:.2f} MB')


BUILDERS = {
    'architecture': build_architecture,
    'virtualtree': build_qual_virtualtree,
    'gallery': build_qual_gallery,
    'sceneflow': build_qual_sceneflow,
    'real': build_qual_real,
    'sample': build_virtualtree_sample,
    'detail': build_detail,
    'curve': build_curve,
    'ablation': build_ablation,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', nargs='*', choices=sorted(BUILDERS),
                    help='build a subset (default: everything)')
    a = ap.parse_args()
    for name in (a.only or BUILDERS):
        print(f'-- {name}')
        BUILDERS[name]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
