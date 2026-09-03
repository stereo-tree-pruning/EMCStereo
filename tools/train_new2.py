#!/usr/bin/env python3
"""train_new2.py — 2-epoch warm-start finish of the interrupted `run.py` tree.

The original reproduction (run.py) was stopped before it finished. Whatever it
DID finish already lives, with a trained ``best.pth`` + all its parameters, in

    /vol/bennylin-solar/EMCStereo/Others_output_new/
        virtualtree/train/default/ckpt/best.pth       (full EMCStereo)
        ablation/none/train/default/ckpt/best.pth      (PSMNet baseline)
        ablation/ema/train/default/ckpt/best.pth       (EMA-only variant)
        ... any further ablation/<v> that later finishes is auto-picked up ...

This script does NOT touch that tree or any existing code. For every finished
model it:

    1. rebuilds the exact architecture and loads its best.pth,
    2. fine-tunes 2 more epochs (gentle cosine LR, warm start),
    3. re-evaluates and saves every paper artefact — metrics + disparity
       colourmaps + left/pred/gt/error panels,

writing everything into a fresh, parallel tree:

    /vol/bennylin-solar/EMCStereo/Others_output_new2/
        <exp>/train/default/ckpt/{best,last}.pth
        <exp>/eval/<split>/eval_results.txt
        <exp>/eval/<split>/disparity/disp_0000..0009.png     (pred over gt)
        <exp>/eval/<split>/panels/panel_0000..0009.png       (left|pred|gt|err)
        results/<exp>_<split>.txt          (copied metrics)
        SUMMARY.txt                        (one-line-per-model table)
        logs/<exp>.log

The metric protocol is byte-for-byte the original one: it reuses train.py's own
``run_eval`` / ``compute_loss`` / ``build_optimizer`` unchanged, so the new
numbers are directly comparable to the paper's.

Run (from the EMCStereo folder), using both GPUs:

    CUDA_VISIBLE_DEVICES=0,1 nohup python train_new2.py 2>&1 &

Each model is fine-tuned in its own subprocess pinned to a single GPU, two at a
time, so both cards stay busy. The run is resumable: a model whose eval already
exists is skipped (set FRESH=1 to redo everything).

Optional environment switches:
    EXTRA_EPOCHS=2   number of additional fine-tune epochs (default 2)
    FT_LR=2e-4       fine-tune peak learning rate (cosine → 0)
    N_VIS=10         how many paper panels/disparity images to save per split
    FRESH=1          ignore existing results and redo every model
    ONLY=a,b         only these experiment keys (e.g. ONLY=virtualtree)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ───────────────────────── paths / knobs ─────────────────────────
REPO = Path(__file__).resolve().parent
BLOCKS = REPO / 'blocks_test'
SRC_ROOT = REPO / 'Others_output_new'          # finished models (read-only)
DST_ROOT = REPO / 'Others_output_new2'         # everything new goes here
LOGS = DST_ROOT / 'logs'
RESULTS = DST_ROOT / 'results'

DATA_VT = os.environ.get('EMCSTEREO_DATA_ROOT',
                         str(REPO / 'datasets' / 'Virtual_branches_data'))
VT_TRAIN = str(REPO / 'data' / 'virtualtree_train.txt')
VT_VAL = str(REPO / 'data' / 'virtualtree_val.txt')
VT_TEST = str(REPO / 'data' / 'virtualtree_test.txt')

EXTRA_EPOCHS = int(os.environ.get('EXTRA_EPOCHS', 2))
FT_LR = float(os.environ.get('FT_LR', 2e-4))
FT_WARMUP = int(os.environ.get('FT_WARMUP', 50))
N_VIS = int(os.environ.get('N_VIS', 10))
FRESH = os.environ.get('FRESH', '0') == '1'
ONLY = {s for s in os.environ.get('ONLY', '').split(',') if s}

# Ablation module flags, copied verbatim from blocks_test/ablation_model.py so
# the stdlib-only parent process never has to import torch to plan the run.
ABLATION_FLAGS = {
    'none':          dict(use_ema=False, use_msf=False, use_coord=False),
    'ema':           dict(use_ema=True,  use_msf=False, use_coord=False),
    'msf':           dict(use_ema=False, use_msf=True,  use_coord=False),
    'coord':         dict(use_ema=False, use_msf=False, use_coord=True),
    'ema_msf':       dict(use_ema=True,  use_msf=True,  use_coord=False),
    'ema_coord':     dict(use_ema=True,  use_msf=False, use_coord=True),
    'msf_coord':     dict(use_ema=False, use_msf=True,  use_coord=True),
    'ema_msf_coord': dict(use_ema=True,  use_msf=True,  use_coord=True),
}

# Prefer the repo's cu128 venv (works on both A40 and Blackwell hosts) for the
# heavy workers, exactly like run.py, even if the parent was launched with a
# different interpreter.
_VENV_PY = REPO / '.venv-cu128' / 'bin' / 'python'
WORKER_PY = str(_VENV_PY) if _VENV_PY.exists() else sys.executable


def log(msg: str) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[train_new2 {ts}] {msg}', flush=True)


# ───────────────────────── experiment registry ─────────────────────────
def build_registry() -> dict:
    """Discover every finished model under Others_output_new (torch-free).

    A model counts as finished when its train/default/ckpt/best.pth exists.
    Returns key -> metadata dict usable by both the parent and the worker.
    """
    reg: dict[str, dict] = {}

    vt = SRC_ROOT / 'virtualtree' / 'train' / 'default' / 'ckpt' / 'best.pth'
    if vt.exists():
        reg['virtualtree'] = dict(
            key='virtualtree', kind='headline', flags=None,
            src_ckpt=str(vt), out_sub='virtualtree', do_test=True)

    abl_root = SRC_ROOT / 'ablation'
    if abl_root.is_dir():
        for d in sorted(p for p in abl_root.iterdir() if p.is_dir()):
            ck = d / 'train' / 'default' / 'ckpt' / 'best.pth'
            if ck.exists() and d.name in ABLATION_FLAGS:
                reg[f'ablation_{d.name}'] = dict(
                    key=f'ablation_{d.name}', kind='ablation',
                    flags=ABLATION_FLAGS[d.name], src_ckpt=str(ck),
                    out_sub=f'ablation/{d.name}', do_test=False)
    return reg


def _eval_done(meta: dict) -> bool:
    """True when this experiment's val metrics already exist in the new tree."""
    return (DST_ROOT / meta['out_sub'] / 'eval' / 'default' /
            'eval_results.txt').exists()


# ══════════════════════════════════════════════════════════════════════
#  WORKER  — fine-tune + evaluate ONE model on the single visible GPU
# ══════════════════════════════════════════════════════════════════════
def run_worker(key: str) -> int:
    # Heavy imports live inside the worker so the parent stays torch-free and
    # so an import-sorter can't hoist them above the sys.path setup below.
    for p in (str(REPO), str(BLOCKS)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

    import copy
    from argparse import Namespace

    import torch
    import train as T

    meta = build_registry().get(key)
    if meta is None:
        log(f'[worker {key}] no such finished model under {SRC_ROOT}')
        return 2

    out_dir = DST_ROOT / meta['out_sub']
    logger = T._setup_logger(out_dir / 'logs', name=f'ft2_{key}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('=' * 72)
    logger.info(f'== fine-tune+eval [{key}] kind={meta["kind"]} '
                f'device={device} epochs=+{EXTRA_EPOCHS} lr={FT_LR} ==')
    logger.info(f'source ckpt : {meta["src_ckpt"]}')
    logger.info(f'output dir  : {out_dir}')

    # ── config: start from train.py DEFAULTS, override only the ft levers ──
    cfg = Namespace(**copy.deepcopy(T.DEFAULTS))
    cfg.tag = 'default'
    cfg.data_root = DATA_VT
    cfg.output_dir = str(out_dir)
    cfg.train_split = VT_TRAIN
    cfg.val_split = VT_VAL
    cfg.crop_size = tuple(cfg.crop_size)
    cfg.epochs = EXTRA_EPOCHS
    cfg.lr = FT_LR
    cfg.warmup_iters = FT_WARMUP
    cfg.scheduler = 'cosine'
    cfg.early_stop_patience = 0
    if meta['flags'] is not None:
        cfg.experiment = key
        cfg.use_ema = meta['flags']['use_ema']
        cfg.use_msf = meta['flags']['use_msf']
        cfg.use_coord = meta['flags']['use_coord']

    # ── build the exact architecture and warm-start it ──
    class _Cfg:
        MAX_DISP = cfg.max_disp

    if meta['kind'] == 'headline':
        model = T.EMCStereo(_Cfg())
    else:
        import ablation_model as AM
        model = AM.make_emc_factory(**meta['flags'])(_Cfg())
    model = model.to(device)

    ckpt = torch.load(meta['src_ckpt'], map_location=device,
                      weights_only=False)
    state = ckpt['model'] if isinstance(
        ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        logger.info(f'load_state_dict: missing={len(missing)} '
                    f'unexpected={len(unexpected)} (expected 0/0)')
    logger.info('warm-started from source best.pth')

    best_path = _fine_tune(T, cfg, model, device, logger)

    # Reload the best-of{baseline, fine-tuned} weights before the final eval so
    # a noisy 2-epoch update can never make the reported numbers worse.
    best_state = torch.load(best_path, map_location=device,
                            weights_only=False)['model']
    model.load_state_dict(best_state)

    _eval_and_visualize(T, cfg, model, device, 'default', VT_VAL, logger)
    if meta['do_test']:
        _eval_and_visualize(T, cfg, model, device, 'test', VT_TEST, logger)

    logger.info(f'[{key}] DONE')
    return 0


def _fine_tune(T, cfg, model, device, logger) -> Path:
    """2-epoch warm-start fine-tune reusing train.py's exact step + optimiser.

    Saves last.pth every epoch and keeps best.pth = lowest val EPE seen,
    seeded with the warm-started weights so the result never regresses.
    """
    from torch.utils.data import DataLoader

    train_ds = T.VirtualTreeDataset(
        cfg.data_root, cfg.train_split,
        transform=T.build_transform('train', crop_size=tuple(cfg.crop_size),
                                    max_disp=cfg.max_disp,
                                    color_aug=cfg.color_aug))
    val_ds = T.VirtualTreeDataset(
        cfg.data_root, cfg.val_split,
        transform=T.build_transform('val', max_disp=cfg.max_disp))
    logger.info(f'train samples: {len(train_ds)}  val samples: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=cfg.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=max(1, cfg.workers // 2),
                            pin_memory=True)

    import torch
    optimizer = T.build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)
    iters_per_epoch = len(train_loader)
    total_steps = max(1, cfg.epochs * iters_per_epoch)

    run_dir = Path(cfg.output_dir) / 'train' / cfg.tag
    ckpt_dir = run_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = ckpt_dir / 'best.pth', ckpt_dir / 'last.pth'
    import json
    (run_dir / 'config.json').write_text(json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v)
         for k, v in vars(cfg).items()}, indent=2, default=str))
    (run_dir / 'NOTE.txt').write_text(
        f'2-epoch warm-start fine-tune of {SRC_ROOT}/... best.pth\n'
        f'created by train_new2.py at {datetime.now():%Y-%m-%d %H:%M}\n')

    # Baseline: evaluate the warm-started model and seed best.pth with it.
    base = T.run_eval(model, val_loader, device, cfg.max_disp,
                      save_dir=None, logger=logger, tag='baseline')
    best_epe = base['epe']
    logger.info(f'baseline VAL EPE={best_epe:.4f}')
    torch.save({'model': model.state_dict(), 'epoch': -1,
                'val_metrics': base, 'global_step': 0}, best_path)
    torch.save({'model': model.state_dict(), 'epoch': -1,
                'val_metrics': base, 'global_step': 0}, last_path)

    global_step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        for it, data in enumerate(train_loader):
            scale = T.compute_lr_scale(global_step, total_steps,
                                       cfg.warmup_iters, cfg.scheduler,
                                       cfg.multistep_milestones,
                                       cfg.multistep_gamma)
            for g in optimizer.param_groups:
                g['lr'] = cfg.lr * scale

            left = data['left'].to(device, non_blocking=True)
            right = data['right'].to(device, non_blocking=True)
            disp_gt = data['disp'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg.amp):
                out = model({'left': left, 'right': right, 'disp': disp_gt})
                loss, _ = T.compute_loss(
                    out['train_preds'], disp_gt, cfg.max_disp,
                    clamp_factor=cfg.disp_pred_clamp_factor,
                    loss_cap=cfg.loss_cap)
            if not torch.isfinite(loss):
                logger.warning(f'[ep {epoch} it {it}] non-finite loss, skip')
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if global_step % cfg.log_interval == 0:
                logger.info(f'Ep {epoch}/{cfg.epochs} It {it}/{iters_per_epoch} '
                            f'loss={loss.item():.4f} '
                            f'lr={optimizer.param_groups[0]["lr"]:.2e} '
                            f'elapsed={(time.time() - t0) / 60:.1f}m')
            global_step += 1

        val_metrics = T.run_eval(model, val_loader, device, cfg.max_disp,
                                 save_dir=None, logger=logger, tag=f'ep{epoch}')
        logger.info(f'Ep {epoch} VAL: ' +
                    ' '.join(f'{k}={v:.4f}' for k, v in val_metrics.items()))
        payload = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(), 'epoch': epoch,
                   'val_metrics': val_metrics, 'global_step': global_step}
        torch.save(payload, last_path)
        if val_metrics['epe'] < best_epe:
            best_epe = val_metrics['epe']
            torch.save(payload, best_path)
            logger.info(f'=> new best EPE={best_epe:.4f}, saved {best_path}')

    logger.info(f'FINE-TUNE DONE. best VAL EPE={best_epe:.4f}')
    return best_path


def _eval_and_visualize(T, cfg, model, device, split, split_file, logger):
    """Final eval on a split: metrics + train.py disparity vis + paper panels."""
    from torch.utils.data import DataLoader

    ds = T.VirtualTreeDataset(cfg.data_root, split_file,
                              transform=T.build_transform('val',
                                                          max_disp=cfg.max_disp))
    eval_dir = Path(cfg.output_dir) / 'eval' / split
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1) metrics + the original pred-over-gt colourmaps + eval_results.txt.
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=max(1, cfg.workers // 2), pin_memory=True)
    agg = T.run_eval(model, loader, device, cfg.max_disp,
                     save_dir=eval_dir, logger=logger, tag=split)
    logger.info(f'FINAL EVAL [{split}] :: ' +
                ' '.join(f'{k}={v:.6f}' for k, v in agg.items()))

    # 2) richer paper panels (left | pred | gt | error) for the first N samples.
    panel_loader = DataLoader(ds, batch_size=1, shuffle=False,
                              num_workers=max(1, cfg.workers // 2),
                              pin_memory=True)
    _save_paper_panels(T, model, panel_loader, device, cfg.max_disp,
                       eval_dir, logger, n=N_VIS)


def _save_paper_panels(T, model, loader, device, max_disp, eval_dir, logger,
                       n=10):
    import cv2
    import numpy as np
    import torch
    from data.virtualtree import IMAGENET_MEAN, IMAGENET_STD

    panels_dir = eval_dir / 'panels'
    panels_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    model.eval()
    saved = 0
    with torch.no_grad():
        for data in loader:
            if saved >= n:
                break
            left = data['left'].to(device, non_blocking=True)
            right = data['right'].to(device, non_blocking=True)
            disp_gt = data['disp'].to(device, non_blocking=True)

            out = model({'left': left, 'right': right, 'disp': disp_gt})
            pred = out['disp_pred']
            if pred.dim() == 4:
                pred = pred.squeeze(1)

            # Undo DivisiblePad so every panel matches the original image size.
            pt = pr = 0
            if 'pad' in data:
                pt, pr, _, _ = data['pad'][0].tolist()
            left_vis = (left.detach().cpu() * std + mean).clamp(0, 1)
            if pt:
                pred = pred[:, pt:, :]
                disp_gt = disp_gt[:, pt:, :]
                left_vis = left_vis[:, :, pt:, :]
            if pr:
                pred = pred[:, :, :-pr]
                disp_gt = disp_gt[:, :, :-pr]
                left_vis = left_vis[:, :, :, :-pr]

            pred_np = pred[0].float().cpu().numpy()
            gt_np = disp_gt[0].float().cpu().numpy()
            left_bgr = (left_vis[0].permute(1, 2, 0).numpy()[:, :, ::-1]
                        * 255.0).astype(np.uint8)
            pred_col = T._disp_to_color(pred_np, max_disp=max_disp)
            gt_col = T._disp_to_color(gt_np, max_disp=max_disp)

            err = np.abs(pred_np - gt_np)
            err[~np.isfinite(gt_np) | (gt_np <= 0)] = 0.0
            err_col = cv2.applyColorMap(
                np.clip(err / 5.0 * 255.0, 0, 255).astype(np.uint8),
                cv2.COLORMAP_MAGMA)

            panel = np.concatenate(
                [left_bgr, pred_col, gt_col, err_col], axis=1)
            cv2.imwrite(str(panels_dir / f'panel_{saved:04d}.png'), panel)
            cv2.imwrite(str(panels_dir / f'left_{saved:04d}.png'), left_bgr)
            cv2.imwrite(str(panels_dir / f'pred_{saved:04d}.png'), pred_col)
            cv2.imwrite(str(panels_dir / f'gt_{saved:04d}.png'), gt_col)
            cv2.imwrite(str(panels_dir / f'error_{saved:04d}.png'), err_col)
            saved += 1

    if logger:
        logger.info(f'Saved {saved} paper panels: {panels_dir}')


# ══════════════════════════════════════════════════════════════════════
#  PARENT  — schedule one worker per finished model across the visible GPUs
# ══════════════════════════════════════════════════════════════════════
def parse_gpu_pool() -> list[str]:
    raw = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    ids = [x for x in raw.split(',') if x != '']
    if not ids:
        log('CUDA_VISIBLE_DEVICES empty; defaulting to GPU "0". Launch with '
            'CUDA_VISIBLE_DEVICES=0,1 to use both cards.')
        ids = ['0']
    return ids


def _child_env(gpu: str) -> dict:
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = gpu
    env['PYTHONPATH'] = (str(REPO) + os.pathsep + str(BLOCKS) + os.pathsep
                         + env.get('PYTHONPATH', ''))
    env['PYTHONUNBUFFERED'] = '1'
    env['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    return env


def orchestrate() -> int:
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    registry = build_registry()
    if ONLY:
        registry = {k: v for k, v in registry.items() if k in ONLY}
    if not registry:
        log(f'FATAL: no finished models (best.pth) found under {SRC_ROOT}')
        return 2
    if not Path(DATA_VT).is_dir():
        log(f'FATAL: VirtualTree data root not found: {DATA_VT}')
        return 2

    gpu_pool = parse_gpu_pool()
    log('== EMCStereo 2-epoch warm-start finish ==')
    log(f'worker python : {WORKER_PY}')
    log(f'source tree   : {SRC_ROOT}')
    log(f'output tree   : {DST_ROOT}')
    log(f'GPU pool      : {gpu_pool}')
    log(f'extra epochs  : {EXTRA_EPOCHS}   ft lr: {FT_LR}   panels/split: {N_VIS}')
    log('models to finish:')
    for k, m in registry.items():
        log(f'    {k:<18} kind={m["kind"]:<8} -> {DST_ROOT / m["out_sub"]}')

    pending = list(registry.keys())
    for k in list(pending):                       # resumability
        if not FRESH and _eval_done(registry[k]):
            log(f'SKIP  {k} (eval already present)')
            pending.remove(k)

    free = list(gpu_pool)
    running: dict[str, tuple] = {}
    failed: list[str] = []
    while pending or running:
        while pending and free:
            k = pending.pop(0)
            gpu = free.pop(0)
            logf = LOGS / f'{k}.log'
            fh = open(logf, 'a', buffering=1)
            fh.write(
                f'\n===== launch {k} @ {datetime.now()} GPU={gpu} =====\n')
            proc = subprocess.Popen(
                [WORKER_PY, str(Path(__file__).resolve()), '--worker', k],
                cwd=str(REPO), env=_child_env(gpu), stdout=fh,
                stderr=subprocess.STDOUT)
            running[k] = (proc, gpu, fh, time.time())
            log(f'START {k}  GPU={gpu}  pid={proc.pid}  -> {logf.name}')

        if not running:
            break
        time.sleep(10)
        for k in list(running):
            proc, gpu, fh, t0 = running[k]
            rc = proc.poll()
            if rc is None:
                continue
            fh.flush()
            fh.close()
            free.append(gpu)
            del running[k]
            dt = (time.time() - t0) / 60.0
            ok = (rc == 0) and _eval_done(registry[k])
            if ok:
                log(f'DONE  {k}  rc={rc}  {dt:.1f} min')
            else:
                failed.append(k)
                why = f'rc={rc}' if rc != 0 else 'missing eval_results.txt'
                log(f'FAIL  {k}  ({why})  {dt:.1f} min  -- see {LOGS / (k + ".log")}')

    consolidate(registry)
    if failed:
        log(f'FAILED: {sorted(set(failed))}')
    log('== all done; see Others_output_new2/SUMMARY.txt ==')
    return 1 if failed else 0


def _parse_eval(path: Path) -> dict:
    import re
    d: dict[str, float] = {}
    if not path.exists():
        return d
    for line in path.read_text().splitlines():
        m = re.match(r'\s*([A-Za-z0-9_]+):\s*([-\d.eE+]+)', line)
        if m:
            try:
                d[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return d


def consolidate(registry: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = ['EMCStereo — 2-epoch warm-start finish  '
            f'{datetime.now():%Y-%m-%d %H:%M}',
            f'source: {SRC_ROOT}   output: {DST_ROOT}', '',
            f'{"model":<20}{"split":<8}{"EPE":>8}{"D1-all%":>9}'
            f'{"Bad1%":>8}{"Bad2%":>8}{"RMSE":>8}', '-' * 69]
    for k, m in registry.items():
        splits = ['default', 'test'] if m['do_test'] else ['default']
        for sp in splits:
            src = (DST_ROOT / m['out_sub'] / 'eval' / sp / 'eval_results.txt')
            if not src.exists():
                rows.append(f'{k:<20}{sp:<8}{"MISSING":>8}')
                continue
            shutil.copyfile(src, RESULTS / f'{k}_{sp}.txt')
            e = _parse_eval(src)
            rows.append(
                f'{k:<20}{sp:<8}{e.get("epe", 0):>8.3f}'
                f'{e.get("d1_all", 0) * 100:>9.2f}{e.get("thres_1", 0) * 100:>8.2f}'
                f'{e.get("thres_2", 0) * 100:>8.2f}{e.get("rmse", 0):>8.3f}')
    (DST_ROOT / 'SUMMARY.txt').write_text('\n'.join(rows) + '\n')
    log(f'wrote {DST_ROOT / "SUMMARY.txt"}')


# ───────────────────────── entrypoint ─────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', default=None,
                    help='internal: fine-tune+eval a single model by key')
    args = ap.parse_args()
    if args.worker:
        return run_worker(args.worker)
    return orchestrate()


if __name__ == '__main__':
    raise SystemExit(main())
