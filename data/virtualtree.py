"""Self-contained VirtualTree stereo dataset + transforms for EMCStereo.

No dependency on OpenStereo-2 whatsoever.
"""
from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms.functional import normalize

# Enable OpenEXR codec (required for .exr depth files from UE5).
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# ZED Mini stereo camera config used to render Virtual_branches_data (UE5).
BASELINE_CM = 6.3          # 63mm
FOCAL_LENGTH_PX = 960.0    # HFoV=90° at 1920 px width
DEPTH_CM_MAX = 65000.0     # UE5 fp16 sky pixels ≈ 65504

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ───────────────────────── transforms ─────────────────────────

class Compose:
    def __init__(self, ts):
        self.ts = ts

    def __call__(self, s):
        for t in self.ts:
            s = t(s)
        return s


class RandomCrop:
    """Crop to (H, W). If require_valid_disp, keep the crop with the most
    valid disparity pixels over `max_tries` attempts."""

    def __init__(self, size, require_valid_disp=True,
                 min_valid_pixels=1024, max_tries=20, max_disp=192):
        self.h, self.w = size
        self.require = require_valid_disp
        self.min_valid = int(min_valid_pixels)
        self.max_tries = int(max_tries)
        self.max_disp = max_disp

    def _valid(self, d):
        v = np.isfinite(d) & (d > 0)
        if self.max_disp is not None:
            v &= d < self.max_disp
        return int(v.sum())

    def __call__(self, s):
        H, W = s['left'].shape[:2]
        if self.h > H or self.w > W:
            return s
        best = None
        best_n = -1
        tries = self.max_tries if self.require and 'disp' in s else 1
        for _ in range(max(1, tries)):
            y1 = random.randint(0, H - self.h)
            x1 = random.randint(0, W - self.w)
            if self.require and 'disp' in s:
                n = self._valid(s['disp'][y1:y1 + self.h, x1:x1 + self.w])
                if n > best_n:
                    best_n = n
                    best = (y1, x1)
                if n >= self.min_valid:
                    break
            else:
                best = (y1, x1)
                break
        y1, x1 = best
        for k in ('left', 'right', 'disp'):
            if k in s:
                s[k] = s[k][y1:y1 + self.h, x1:x1 + self.w]
        return s


class DivisiblePad:
    """Right/top pad so H, W are divisible by `by` (eval)."""

    def __init__(self, by=32):
        self.by = by

    def __call__(self, s):
        H, W = s['left'].shape[:2]
        pad_h = (self.by - H % self.by) % self.by
        pad_w = (self.by - W % self.by) % self.by
        pad_top, pad_right = pad_h, pad_w
        for k in ('left', 'right'):
            if k in s:
                s[k] = np.pad(
                    s[k], ((pad_top, 0), (0, pad_right), (0, 0)), 'edge')
        if 'disp' in s:
            s['disp'] = np.pad(s['disp'], ((pad_top, 0), (0, pad_right)),
                               'constant', constant_values=0)
        s['pad'] = np.array([pad_top, pad_right, 0, 0], dtype=np.int32)
        return s


class AsymmetricColorJitter:
    """Mild stereo-appropriate color jitter. Asymmetric with small prob so the
    two views usually agree; symmetric otherwise. Helps generalization."""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4,
                 hue=0.5 / 3.14, asymmetric_prob=0.2):
        self.jitter = ColorJitter(brightness=brightness, contrast=contrast,
                                  saturation=saturation, hue=hue)
        self.asym = asymmetric_prob

    def __call__(self, s):
        l = s['left'].astype(np.uint8)
        r = s['right'].astype(np.uint8)
        if np.random.rand() < self.asym:
            l = np.array(self.jitter(Image.fromarray(l)), dtype=np.float32)
            r = np.array(self.jitter(Image.fromarray(r)), dtype=np.float32)
        else:
            stack = np.concatenate([l, r], axis=0)
            stack = np.array(self.jitter(
                Image.fromarray(stack)), dtype=np.float32)
            l, r = np.split(stack, 2, axis=0)
        s['left'] = l
        s['right'] = r
        return s


class ToTensorNormalize:
    """CHW + /255 + ImageNet normalize for left/right; tensor for disp."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean, self.std = mean, std

    def __call__(self, s):
        for k in ('left', 'right'):
            img = s[k].transpose(2, 0, 1).astype(np.float32) / 255.0
            t = torch.from_numpy(np.ascontiguousarray(img))
            s[k] = normalize(t, self.mean, self.std)
        if 'disp' in s:
            s['disp'] = torch.from_numpy(
                np.ascontiguousarray(s['disp'])).float()
        return s


def build_transform(mode: str, crop_size=(256, 512), max_disp=192,
                    color_aug=True):
    if mode == 'train':
        ts = [RandomCrop(crop_size, require_valid_disp=True,
                         min_valid_pixels=1024, max_tries=20,
                         max_disp=max_disp)]
        if color_aug:
            ts.append(AsymmetricColorJitter())
        ts.append(ToTensorNormalize())
        return Compose(ts)
    else:  # val / test
        return Compose([DivisiblePad(by=32), ToTensorNormalize()])


# ───────────────────────── dataset ─────────────────────────

class VirtualTreeDataset(Dataset):
    """VirtualTree (Virtual_branches_data) stereo dataset.

    Split-file format (whitespace-separated, dir names contain spaces):
        left image/left_X.png  left depth/depth_X.exr  right image/right_X.png
    """

    def __init__(self, data_root: str, split_file: str, transform=None):
        self.root = data_root
        self.transform = transform
        self.items = []
        with open(split_file, 'r') as f:
            for line in f:
                toks = line.strip().split()
                if not toks:
                    continue
                # Rejoin tokens because directory names like "left image"
                # contain spaces → 6 tokens expected.
                if len(toks) == 6:
                    self.items.append((toks[0] + ' ' + toks[1],
                                       toks[4] + ' ' + toks[5],
                                       toks[2] + ' ' + toks[3]))
                elif len(toks) == 3:
                    self.items.append((toks[0], toks[2], toks[1]))
                else:
                    raise ValueError(f'Unexpected split line: {line!r}')

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _read_exr_depth(path: str) -> np.ndarray:
        """UE5 depth (cm) from an EXR. Uses OpenCV when its build ships the
        OpenEXR codec, otherwise falls back to the OpenEXR package. Depth lives
        in the R channel (the other channels are NaN)."""
        exr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if exr is not None:
            return (exr[:, :, 2] if exr.ndim == 3 else exr).astype(np.float32)
        # importlib keeps this robust to import-sorting tools hoisting the dep.
        import importlib
        oexr = importlib.import_module('OpenEXR')
        imath = importlib.import_module('Imath')
        f = oexr.InputFile(path)
        hdr = f.header()
        dw = hdr['dataWindow']
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        chans = hdr['channels']
        cname = 'R' if 'R' in chans else next(iter(chans))
        raw = f.channel(cname, imath.PixelType(imath.PixelType.FLOAT))
        return np.frombuffer(raw, dtype=np.float32).reshape(h, w).astype(
            np.float32, copy=True)

    @staticmethod
    def _load_disp_from_exr(path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Cannot read EXR: {path}')
        depth = VirtualTreeDataset._read_exr_depth(path)
        disp = np.zeros_like(depth, dtype=np.float32)
        valid = (depth > 0) & (depth < DEPTH_CM_MAX) & np.isfinite(depth)
        disp[valid] = FOCAL_LENGTH_PX * BASELINE_CM / depth[valid]
        disp[~np.isfinite(disp)] = 0
        disp[disp < 0] = 0
        return disp

    def __getitem__(self, i):
        lp, rp, dp = self.items[i]
        left = np.array(Image.open(os.path.join(self.root, lp)).convert('RGB'),
                        dtype=np.float32)
        right = np.array(Image.open(os.path.join(self.root, rp)).convert('RGB'),
                         dtype=np.float32)
        disp = self._load_disp_from_exr(os.path.join(self.root, dp))
        s = {'left': left, 'right': right, 'disp': disp}
        if self.transform is not None:
            s = self.transform(s)
        s['index'] = i
        s['name'] = lp
        return s
