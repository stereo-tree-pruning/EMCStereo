"""Stereo loaders for ETH3D / KITTI2012 / KITTI2015 / Middlebury / Scene Flow.

Uses the same `build_transform` from `data.virtualtree` so the training
pipeline is drop-in compatible with EMCStereo.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')


# ───────────────────── disparity readers ─────────────────────
def _read_pfm(path: str) -> np.ndarray:
    """Read a PFM disparity file. Middlebury/SceneFlow/ETH3D format."""
    with open(path, 'rb') as f:
        header = f.readline().decode('latin-1').rstrip()
        if header not in ('PF', 'Pf'):
            raise ValueError(f'Not a PFM file: {path}')
        color = header == 'PF'
        dim_line = f.readline().decode('latin-1').strip()
        while dim_line.startswith('#'):
            dim_line = f.readline().decode('latin-1').strip()
        w, h = (int(x) for x in re.split(r'\s+', dim_line))
        scale = float(f.readline().decode('latin-1').rstrip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)
        data = np.frombuffer(f.read(), endian + 'f')
        if color:
            data = data.reshape(h, w, 3)
        else:
            data = data.reshape(h, w)
        data = np.flipud(data)
        return np.ascontiguousarray(data, dtype=np.float32)


def _read_kitti_disp(path: str) -> np.ndarray:
    """KITTI stores disparity as uint16 PNG, disp = value / 256, 0 = invalid."""
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    disp = arr.astype(np.float32) / 256.0
    disp[arr == 0] = 0
    return disp


def _sanitize(disp: np.ndarray) -> np.ndarray:
    disp = np.asarray(disp, dtype=np.float32)
    disp[~np.isfinite(disp)] = 0.0
    disp[disp < 0] = 0.0
    return disp


# ───────────────────── dataset ─────────────────────
class OpenStereoDataset(Dataset):
    """Generic (left, right, disp) dataset.

    `items` is a list of (left_path, right_path, disp_path, disp_kind) where
    `disp_kind` ∈ {'pfm', 'kitti'}.
    All paths are absolute.
    """

    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        lp, rp, dp, kind = self.items[i]
        left = np.array(Image.open(lp).convert('RGB'), dtype=np.float32)
        right = np.array(Image.open(rp).convert('RGB'), dtype=np.float32)
        if kind == 'kitti':
            disp = _read_kitti_disp(dp)
        else:
            disp = _read_pfm(dp)
            if disp.ndim == 3:
                disp = disp[:, :, 0]
        disp = _sanitize(disp)
        # Make sure shapes match (some Middlebury scenes may differ by 1 px).
        Hl, Wl = left.shape[:2]
        Hr, Wr = right.shape[:2]
        Hd, Wd = disp.shape[:2]
        H = min(Hl, Hr, Hd)
        W = min(Wl, Wr, Wd)
        left = left[:H, :W]
        right = right[:H, :W]
        disp = disp[:H, :W]
        s = {'left': left, 'right': right, 'disp': disp}
        if self.transform is not None:
            s = self.transform(s)
        s['index'] = i
        s['name'] = str(lp)
        return s


# ───────────────────── index builders ─────────────────────
def _index_eth3d(root: Path) -> List[Tuple[str, str, str, str]]:
    img_root = root / 'two_view_training'
    disp_root = root / 'two_view_training_disparity'
    items = []
    if not img_root.exists() or not disp_root.exists():
        return items
    for scene in sorted(p for p in img_root.iterdir() if p.is_dir()):
        l = scene / 'im0.png'
        r = scene / 'im1.png'
        d = disp_root / scene.name / 'disp0GT.pfm'
        if l.exists() and r.exists() and d.exists():
            items.append((str(l), str(r), str(d), 'pfm'))
    return items


def _index_kitti(root: Path) -> List[Tuple[str, str, str, str]]:
    left = root / 'left'
    right = root / 'right'
    gt = root / 'ground_truth'
    items = []
    if not (left.exists() and right.exists() and gt.exists()):
        return items
    for lp in sorted(left.glob('*.png')):
        rp = right / lp.name
        dp = gt / lp.name
        if rp.exists() and dp.exists():
            items.append((str(lp), str(rp), str(dp), 'kitti'))
    return items


def _index_middlebury(root: Path) -> List[Tuple[str, str, str, str]]:
    items = []
    if not root.exists():
        return items
    for scene in sorted(p for p in root.iterdir() if p.is_dir()):
        l = scene / 'im0.png'
        r = scene / 'im1.png'
        d = None
        for cand in ('disp0.pfm', 'disp0GT.pfm'):
            if (scene / cand).exists():
                d = scene / cand
                break
        if l.exists() and r.exists() and d is not None:
            items.append((str(l), str(r), str(d), 'pfm'))
    return items


def _index_sceneflow(root: Path) -> List[Tuple[str, str, str, str]]:
    items: List[Tuple[str, str, str, str]] = []
    if not root.exists():
        return items

    # flyingthings3d : frames A/B/C/NNNN/left/*.png  disp in flyingthings3d_disparity/...
    ft_img = root / 'flyingthings3d__frames'
    ft_dsp = root / 'flyingthings3d_disparity'
    if ft_img.exists() and ft_dsp.exists():
        for sub in sorted(p for p in ft_img.iterdir() if p.is_dir()):
            for seq in sorted(p for p in sub.iterdir() if p.is_dir()):
                lft = seq / 'left'
                rgt = seq / 'right'
                ddir = ft_dsp / sub.name / seq.name / 'left'
                if not (lft.exists() and rgt.exists() and ddir.exists()):
                    continue
                for lp in sorted(lft.glob('*.png')):
                    rp = rgt / lp.name
                    dp = ddir / (lp.stem + '.pfm')
                    if rp.exists() and dp.exists():
                        items.append((str(lp), str(rp), str(dp), 'pfm'))

    # monkaa : monkaa__frames/<scene>/left/NNNN.png
    mk_img = root / 'monkaa__frames'
    mk_dsp = root / 'monkaa_disparity'
    if mk_img.exists() and mk_dsp.exists():
        for scene in sorted(p for p in mk_img.iterdir() if p.is_dir()):
            lft = scene / 'left'
            rgt = scene / 'right'
            ddir = mk_dsp / scene.name / 'left'
            if not (lft.exists() and rgt.exists() and ddir.exists()):
                continue
            for lp in sorted(lft.glob('*.png')):
                rp = rgt / lp.name
                dp = ddir / (lp.stem + '.pfm')
                if rp.exists() and dp.exists():
                    items.append((str(lp), str(rp), str(dp), 'pfm'))

    # driving : driving__frames/<focal>/<direction>/<speed>/left/*.png
    dr_img = root / 'driving__frames'
    dr_dsp = root / 'driving__disparity'
    if dr_img.exists() and dr_dsp.exists():
        for focal in sorted(p for p in dr_img.iterdir() if p.is_dir()):
            for direction in sorted(p for p in focal.iterdir() if p.is_dir()):
                for speed in sorted(p for p in direction.iterdir() if p.is_dir()):
                    lft = speed / 'left'
                    rgt = speed / 'right'
                    ddir = dr_dsp / focal.name / direction.name / speed.name / 'left'
                    if not (lft.exists() and rgt.exists() and ddir.exists()):
                        continue
                    for lp in sorted(lft.glob('*.png')):
                        rp = rgt / lp.name
                        dp = ddir / (lp.stem + '.pfm')
                        if rp.exists() and dp.exists():
                            items.append((str(lp), str(rp), str(dp), 'pfm'))

    return items


DATASET_BUILDERS = {
    'ETH3': _index_eth3d,
    'KITTI2012': _index_kitti,
    'KITTI2015': _index_kitti,
    'Middlebury': _index_middlebury,
    'Scene Flow': _index_sceneflow,
}


def build_dataset_items(name: str, root: Path) -> List[Tuple[str, str, str, str]]:
    builder = DATASET_BUILDERS[name]
    return builder(Path(root))


def split_train_val(items, val_ratio=0.1, seed=0):
    """Deterministic train/val split."""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(items) * val_ratio)))
    val_idx = set(idx[:n_val])
    train = [x for i, x in enumerate(items) if i not in val_idx]
    val = [x for i, x in enumerate(items) if i in val_idx]
    return train, val
