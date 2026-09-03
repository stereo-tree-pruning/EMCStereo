#!/usr/bin/env python3
"""Re-run every validation reported in main.tex from the checkpoints stored in
final_results/ and compare against expected_metrics.json.

Just run it (everything is written under final_results/val/):
    CUDA_VISIBLE_DEVICES=0 nohup python val_all.py 2>&1 &

Optional flags: --only NAME ... | --skip-sceneflow | --out-dir DIR |
    --data-root $EMCSTEREO_OPEN_ROOT | --tol 2e-3

Each entry writes val/<entry>/eval_<split>/eval_results.txt (re-evaluated), plus
val/val_all_summary.txt, val/val_all_summary.json and val/val_all.log.
Evaluation mirrors train.py / train_others.py run_eval exactly:
fp32, batch 1, DivisiblePad(32) removed before metrics, mask 0<gt<max_disp,
metrics averaged per image, no TTA.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

HERE = Path(__file__).resolve().parent
REPO = HERE.parent            # repository root (model + data packages live here)
CODE = REPO
sys.path.insert(0, str(CODE))

# Directory holding <group>/<run>/best.pth for every run in the paper. Only
# virtualtree/emcstereo_headline ships with this repository; point
# --artefact-root at the full final_results/ tree to re-validate the rest.
DEFAULT_ARTEFACT_ROOT = REPO / 'final_results'

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.open_datasets import (OpenStereoDataset, build_dataset_items,  # noqa: E402
                                split_train_val)
from data.virtualtree import VirtualTreeDataset, build_transform  # noqa: E402
from src.emc_stereo import EMCStereo  # noqa: E402
import ablation_model  # noqa: E402

MAX_DISP = 192
METRIC_DESCRIPTIONS = {
    'epe': 'End Point Error (Mean Absolute Error, lower is better)',
    'd1_all': 'D1-all: % pixels with error >3px and >5% (lower is better)',
    'thres_05': 'Bad 0.5: % pixels with error >0.5px (lower is better)',
    'thres_1': 'Bad 1.0: % pixels with error >1px (lower is better)',
    'thres_2': 'Bad 2.0: % pixels with error >2px (lower is better)',
    'thres_3': 'Bad 3.0: % pixels with error >3px (lower is better)',
    'thres_4': 'Bad 4.0: % pixels with error >4px (lower is better)',
    'thres_5': 'Bad 5.0: % pixels with error >5px (lower is better)',
    'rmse': 'Root Mean Square Error (lower is better)',
    'absrel': 'Absolute Relative Error |pred-gt|/gt (lower is better)',
    'mae': 'Mean Absolute Error (lower is better)',
    'sqrel': 'Squared Relative Error (pred-gt)^2/gt (lower is better)',
    'rmse_log': 'RMSE of log disparity (lower is better)',
    'delta1': 'Accuracy δ<1.25 (higher is better)',
    'delta2': 'Accuracy δ<1.25^2 (higher is better)',
    'delta3': 'Accuracy δ<1.25^3 (higher is better)',
}

# entry (relative to final_results) -> (model kind, list of (split, dataset spec))
# model kind: 'full' -> EMCStereo ; ('abl', ema, msf, coord) -> EMCStereoAblation
VT = 'virtualtree'
ENTRIES = {
    'virtualtree/emcstereo_headline': ('full', [('val', VT), ('test', VT)]),
    'virtualtree/emcstereo_scratch_snapshot': ('full', [('val', VT)]),
    'virtualtree/emcstereo_scratch_2ep_restart': ('full', [('val', VT), ('test', VT)]),
    'ablation/none': (('abl', 0, 0, 0), [('val', VT)]),
    'ablation/none_repeat': (('abl', 0, 0, 0), [('val', VT)]),
    'ablation/ema': (('abl', 1, 0, 0), [('val', VT)]),
    'ablation/msf': (('abl', 0, 1, 0), [('val', VT)]),
    'ablation/coord': (('abl', 0, 0, 1), [('val', VT)]),
    'ablation/ema_msf': (('abl', 1, 1, 0), [('val', VT)]),
    'ablation/ema_coord': (('abl', 1, 0, 1), [('val', VT)]),
    'ablation/msf_coord': (('abl', 0, 1, 1), [('val', VT)]),
    'ablation/ema_msf_coord': (('abl', 1, 1, 1), [('val', VT)]),
    'cross_dataset/KITTI2012': ('full', [('val', 'KITTI2012')]),
    'cross_dataset/KITTI2015': ('full', [('val', 'KITTI2015')]),
    'cross_dataset/Middlebury': ('full', [('val', 'Middlebury')]),
    'cross_dataset/ETH3D': ('full', [('val', 'ETH3')]),
    'sceneflow/baseline_ep31': ('full', [('val', 'Scene Flow')]),
    'sceneflow/v4_best': ('full', [('val', 'Scene Flow')]),
}
SCENEFLOW_ENTRIES = {k for k in ENTRIES if k.startswith('sceneflow/')}


def compute_metrics(pred, gt, max_disp):
    mask = torch.isfinite(gt) & (gt > 0) & (gt < max_disp)
    if not mask.any():
        return {k: 0.0 for k in METRIC_DESCRIPTIONS}
    p = pred[mask].float()
    g = gt[mask].float()
    err = (p - g).abs()
    d1 = (((err > 3) & (err / g.clamp(min=1e-6) > 0.05)).float().mean().item())
    p_safe = p.clamp(min=1e-6)
    g_safe = g.clamp(min=1e-6)
    log_diff = torch.log(p_safe) - torch.log(g_safe)
    ratio = torch.max(p_safe / g_safe, g_safe / p_safe)
    return {
        'epe': err.mean().item(),
        'd1_all': d1,
        'thres_05': (err > 0.5).float().mean().item(),
        'thres_1': (err > 1).float().mean().item(),
        'thres_2': (err > 2).float().mean().item(),
        'thres_3': (err > 3).float().mean().item(),
        'thres_4': (err > 4).float().mean().item(),
        'thres_5': (err > 5).float().mean().item(),
        'rmse': torch.sqrt((err ** 2).mean()).item(),
        'absrel': (err / g_safe).mean().item(),
        'mae': err.mean().item(),
        'sqrel': ((err ** 2) / g_safe).mean().item(),
        'rmse_log': torch.sqrt((log_diff ** 2).mean()).item(),
        'delta1': (ratio < 1.25).float().mean().item(),
        'delta2': (ratio < 1.25 ** 2).float().mean().item(),
        'delta3': (ratio < 1.25 ** 3).float().mean().item(),
    }


@torch.no_grad()
def run_eval(model, loader, device, save_dir: Path, tag: str, log):
    model.eval()
    agg = {k: 0.0 for k in METRIC_DESCRIPTIONS}
    n = 0
    t0 = time.time()
    for data in loader:
        left = data['left'].to(device, non_blocking=True)
        right = data['right'].to(device, non_blocking=True)
        disp_gt = data['disp'].to(device, non_blocking=True)
        pred = model({'left': left, 'right': right, 'disp': disp_gt})[
            'disp_pred']
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if 'pad' in data:
            pt, pr, _, _ = data['pad'][0].tolist()
            if pt:
                pred, disp_gt = pred[:, pt:, :], disp_gt[:, pt:, :]
            if pr:
                pred, disp_gt = pred[:, :, :-pr], disp_gt[:, :, :-pr]
        m = compute_metrics(pred, disp_gt, MAX_DISP)
        for k in agg:
            agg[k] += m[k]
        n += 1
        if n % 200 == 0:
            log(f'    {n}/{len(loader)}  running epe={agg["epe"] / n:.4f}  '
                f'({time.time() - t0:.0f}s)')
    agg = {k: v / max(1, n) for k, v in agg.items()}
    save_dir.mkdir(parents=True, exist_ok=True)
    lines = [f'EMCStereo [{tag}] eval over {n} samples', '']
    lines += [f'{k}: {v:.6f}    # {METRIC_DESCRIPTIONS[k]}' for k,
              v in agg.items()]
    (save_dir / 'eval_results.txt').write_text('\n'.join(lines) + '\n')
    agg['n'] = n
    return agg


def build_model(kind, ckpt_path: Path, device):
    cfgs = SimpleNamespace(MAX_DISP=MAX_DISP)
    if kind == 'full':
        model = EMCStereo(cfgs)
    else:
        _, e, m, c = kind
        model = ablation_model.EMCStereoAblation(cfgs, use_ema=bool(e),
                                                 use_msf=bool(m), use_coord=bool(c))
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=True), None
    info = {'epoch': ck.get('epoch') if isinstance(ck, dict) else None,
            'ckpt_val_epe': (ck.get('val_metrics') or {}).get('epe')
            if isinstance(ck, dict) else None,
            'params': sum(p.numel() for p in model.parameters())}
    return model.to(device), info


_DS_CACHE = {}


def build_dataset(spec: str, split: str, data_root: Path):
    key = (spec, split)
    if key in _DS_CACHE:
        return _DS_CACHE[key]
    tf = build_transform('val', max_disp=MAX_DISP)
    if spec == VT:
        ds = VirtualTreeDataset(str(data_root / 'Virtual_branches_data'),
                                str(CODE / 'data' /
                                    f'virtualtree_{split}.txt'),
                                transform=tf)
    else:
        items = build_dataset_items(spec, data_root / spec)
        if not items:
            raise FileNotFoundError(
                f'no samples for {spec} under {data_root / spec}')
        # identical deterministic 90/10 split (val_ratio=0.1, seed=0) used in training
        _, val_items = split_train_val(items, val_ratio=0.1, seed=0)
        ds = OpenStereoDataset(val_items, transform=tf)
    _DS_CACHE[key] = ds
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root',
                    default=os.environ.get('EMCSTEREO_OPEN_ROOT',
                                           str(HERE.parent / 'datasets')))
    ap.add_argument('--only', nargs='*', default=None,
                    help='subset of entry names (substring match)')
    ap.add_argument('--skip-sceneflow', action='store_true')
    ap.add_argument('--tol', type=float, default=2e-3,
                    help='abs tolerance on each expected metric')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out-dir', default=str(REPO / 'val'),
                    help='where to write eval results, summary and log')
    ap.add_argument('--artefact-root', default=str(DEFAULT_ARTEFACT_ROOT),
                    help='directory containing <group>/<run>/best.pth')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    artefacts = Path(args.artefact_root)
    expected = json.loads(
        (REPO / 'results' / 'expected_metrics.json').read_text())
    logf = open(out / 'val_all.log', 'a')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    log(f'==== val_all.py start {time.strftime("%Y-%m-%d %H:%M:%S")} device={device} '
        f'torch={torch.__version__}')
    names = [n for n in ENTRIES
             if (args.only is None or any(s in n for s in args.only))
             and not (args.skip_sceneflow and n in SCENEFLOW_ENTRIES)]

    rows, all_ok = [], True
    for name in names:
        kind, evals = ENTRIES[name]
        ckpt = artefacts / name / 'best.pth'
        if not ckpt.exists():
            log(f'[{name}] MISSING {ckpt}')
            all_ok = False
            continue
        model, info = build_model(kind, ckpt, device)
        log(f'[{name}] loaded {ckpt.name} epoch={info["epoch"]} '
            f'ckpt_val_epe={info["ckpt_val_epe"]} params={info["params"]:,}')
        for split, spec in evals:
            ds = build_dataset(spec, split, Path(args.data_root))
            loader = DataLoader(ds, batch_size=1, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
            t0 = time.time()
            agg = run_eval(model, loader, device, out / name / f'eval_{split}',
                           f'{name}:{split}', log)
            exp = expected.get(f'{name}:{split}', {})
            diffs = {k: agg[k] - v for k, v in exp.items()}
            ok = all(abs(d) <= args.tol for d in diffs.values())
            all_ok &= ok
            worst = max(diffs.items(), key=lambda kv: abs(
                kv[1]), default=('-', 0.0))
            log(f'[{name}:{split}] n={agg["n"]} epe={agg["epe"]:.6f} '
                f'd1={agg["d1_all"]:.6f} bad1={agg["thres_1"]:.6f} '
                f'rmse={agg["rmse"]:.6f} | expected epe={exp.get("epe")} '
                f'| {"OK" if ok else "MISMATCH"} (worst {worst[0]} {worst[1]:+.6f}) '
                f'[{time.time() - t0:.0f}s]')
            rows.append({'entry': name, 'split': split, 'n': agg['n'],
                         'epoch': info['epoch'], 'params': info['params'],
                         'metrics': {k: agg[k] for k in METRIC_DESCRIPTIONS},
                         'expected': exp, 'ok': ok})
        del model
        torch.cuda.empty_cache()

    # summary
    hdr = f'{"entry":45s} {"split":5s} {"n":>5s} {"EPE":>9s} {"D1":>8s} {"Bad1":>8s} {"Bad3":>8s} {"RMSE":>8s} {"exp.EPE":>9s} status'
    out_lines = [hdr, '-' * len(hdr)]
    for r in rows:
        m = r['metrics']
        out_lines.append(f'{r["entry"]:45s} {r["split"]:5s} {r["n"]:5d} {m["epe"]:9.4f} '
                         f'{m["d1_all"]:8.4f} {m["thres_1"]:8.4f} {m["thres_3"]:8.4f} '
                         f'{m["rmse"]:8.4f} {str(r["expected"].get("epe", "-")):>9s} '
                         f'{"OK" if r["ok"] else "MISMATCH"}')
    out_lines.append('')
    out_lines.append('ALL OK' if all_ok else 'SOME MISMATCHES')
    txt = '\n'.join(out_lines)
    log('\n' + txt)
    suffix = '' if args.only is None and not args.skip_sceneflow else '_partial'
    (out / f'val_all_summary{suffix}.txt').write_text(txt + '\n')
    (out /
     f'val_all_summary{suffix}.json').write_text(json.dumps(rows, indent=1))
    log(f'==== done {time.strftime("%Y-%m-%d %H:%M:%S")}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
