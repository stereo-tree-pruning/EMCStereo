"""EMCStereo module ablation on VirtualTree — train + eval, saved to blocks_test/.

Run it from the repository root; outputs default to <repo>/blocks_test/.

Ablation of the three injected modules (E=EMA, M=MSFblock, C=CoordAtt).
Six experiments: the 3 single modules + the 3 pairwise combinations.

    ema         E only
    msf         M only
    coord       C only
    ema_msf     E + M
    ema_coord   E + C
    msf_coord   M + C

Each experiment is a self-contained folder under blocks_test/<exp>/:
    train/default/ckpt/{best.pth,last.pth}
    train/default/config.json
    train/default/tensorboard/
    eval/default/eval_results.txt
    eval/default/disparity/disp_0000..0009.png
    logs/*.log

Design: this driver does NOT reimplement training / evaluation. It reuses the
original train.py verbatim (run_train / run_final_eval / run_eval /
compute_loss / compute_metrics) and only swaps the model by monkey-patching
`train.EMCStereo` with an ablation factory. That guarantees the metrics and the
whole protocol are identical to the original full-model pipeline.

Usage:
    # one experiment (recommended: pin a GPU per process)
    CUDA_VISIBLE_DEVICES=1 python train_ablation.py --experiment ema

    # all six, sequentially, in one process
    python train_ablation.py --experiment all

    # eval only (reuse an already-trained best.pth)
    python train_ablation.py --experiment ema --skip-train
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from argparse import Namespace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent       # repository root
_HERE = _REPO_ROOT / 'blocks_test'                 # default output tree

# VirtualTree root: $EMCSTEREO_DATA_ROOT, else <repo>/datasets/Virtual_branches_data.
# Overridable with --data-root.
_DEFAULT_DATA_ROOT = os.environ.get(
    'EMCSTEREO_DATA_ROOT',
    str(_REPO_ROOT / 'datasets' / 'Virtual_branches_data'))


def parse_args():
    p = argparse.ArgumentParser(
        description='EMCStereo EMA/MSFblock/CoordAtt ablation on VirtualTree')
    p.add_argument('--experiment', default='all',
                   choices=['all', 'none', 'ema', 'msf', 'coord',
                            'ema_msf', 'ema_coord', 'msf_coord',
                            'ema_msf_coord'],
                   help="which ablation to run ('all' = every entry "
                        'sequentially)')

    p.add_argument('--output-root', default=str(_HERE),
                   help='parent dir for the per-experiment folders '
                        '(default: the blocks_test folder)')
    p.add_argument('--data-root', default=None,
                   help='VirtualTree dataset root (default: auto-detect)')
    p.add_argument('--train-split', default=None,
                   help='override train split file (default: train.py DEFAULTS)')
    p.add_argument('--val-split', default=None,
                   help='override val split file (default: train.py DEFAULTS)')

    p.add_argument('--skip-train', action='store_true',
                   help='eval only (requires an existing best.pth)')
    p.add_argument('--skip-eval', action='store_true',
                   help='train only')

    # Hyper-parameters — default to the original train.py recipe so the
    # ablation matches the full-model run. Override for a faster sweep.
    p.add_argument('--epochs', type=int, default=None,
                   help='override epochs (default: train.py DEFAULTS = 300)')
    p.add_argument('--early-stop-patience', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--amp', type=int, default=None, choices=[0, 1])
    p.add_argument('--seed', type=int, default=None)

    p.add_argument('--gpus', default=None,
                   help='sets CUDA_VISIBLE_DEVICES before torch loads')
    return p.parse_args()


def _resolve_data_root(explicit, train_defaults):
    if explicit:
        return explicit
    for c in (_DEFAULT_DATA_ROOT, train_defaults.get('data_root', '')):
        if c and Path(c).is_dir():
            return c
    return _DEFAULT_DATA_ROOT


def _build_cfg(train_mod, exp_name, flags, args, data_root):
    """Build the argparse-style cfg that train.run_train / run_final_eval
    expect, starting from train.py's own DEFAULTS so nothing drifts."""
    d = copy.deepcopy(train_mod.DEFAULTS)
    ns = Namespace(**d)

    ns.tag = 'default'
    ns.data_root = data_root
    ns.output_dir = str(Path(args.output_root) / exp_name)
    ns.crop_size = tuple(d['crop_size'])
    if args.train_split is not None:
        ns.train_split = args.train_split
    if args.val_split is not None:
        ns.val_split = args.val_split

    # optional overrides (fall back to the original recipe)
    if args.epochs is not None:
        ns.epochs = args.epochs
    if args.early_stop_patience is not None:
        ns.early_stop_patience = args.early_stop_patience
    if args.batch_size is not None:
        ns.batch_size = args.batch_size
    if args.workers is not None:
        ns.workers = args.workers
    if args.lr is not None:
        ns.lr = args.lr
    if args.amp is not None:
        ns.amp = bool(args.amp)
    if args.seed is not None:
        ns.seed = args.seed

    # record the module flags in config.json for traceability
    ns.experiment = exp_name
    ns.use_ema = flags['use_ema']
    ns.use_msf = flags['use_msf']
    ns.use_coord = flags['use_coord']
    return ns


def _run_one(train_mod, ablation_mod, exp_name, args, data_root):
    flags = ablation_mod.EXPERIMENTS[exp_name]
    cfg = _build_cfg(train_mod, exp_name, flags, args, data_root)

    # Swap ONLY the model: run_train / run_final_eval resolve `EMCStereo` from
    # the train module namespace at call time, so this makes the whole original
    # pipeline run on the ablation variant.
    train_mod.EMCStereo = ablation_mod.make_emc_factory(**flags)

    out_dir = Path(cfg.output_dir)
    log_dir = out_dir / 'logs'
    logger = train_mod._setup_logger(log_dir, name=f'EMCStereo_{exp_name}')

    tags = ''.join(k for k, on in
                   (('E', flags['use_ema']), ('M', flags['use_msf']),
                    ('C', flags['use_coord'])) if on)
    logger.info('=' * 70)
    logger.info(f'== ABLATION [{exp_name}] modules={tags} '
                f'(EMA={flags["use_ema"]} MSF={flags["use_msf"]} '
                f'Coord={flags["use_coord"]}) ==')
    logger.info(f'output_dir={out_dir}')
    logger.info('=' * 70)

    train_mod._set_seed(cfg.seed)

    best_ckpt = out_dir / 'train' / cfg.tag / 'ckpt' / 'best.pth'
    if not args.skip_train:
        best_ckpt = train_mod.run_train(cfg, logger)
    if not args.skip_eval:
        if not Path(best_ckpt).exists():
            raise SystemExit(f'[{exp_name}] checkpoint not found: {best_ckpt} '
                             '(train first, or drop --skip-train)')
        eval_dir = train_mod.run_final_eval(cfg, Path(best_ckpt), logger)
        results = eval_dir / 'eval_results.txt'
        if results.exists():
            print('\n' + '=' * 70)
            print(f'[ablation:{exp_name}] RESULTS')
            print('=' * 70)
            print(results.read_text())
    logger.info(f'[{exp_name}] DONE.')


def main() -> int:
    args = parse_args()

    # Must set device visibility BEFORE torch is imported (via `import train`).
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))

    import train                       # original pipeline (imports torch)
    import ablation_model              # our configurable model

    data_root = _resolve_data_root(args.data_root, train.DEFAULTS)
    if not Path(data_root).is_dir():
        raise SystemExit(f'Dataset root not found: {data_root} '
                         '(pass --data-root)')

    exps = (list(ablation_model.EXPERIMENTS.keys())
            if args.experiment == 'all' else [args.experiment])

    print(f'[ablation] data_root = {data_root}')
    print(f'[ablation] experiments = {exps}')
    print(f'[ablation] output_root = {args.output_root}')

    t0 = time.time()
    for exp in exps:
        _run_one(train, ablation_model, exp, args, data_root)
        # free VRAM between sequential runs
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f'\n[ablation] ALL DONE ({(time.time() - t0) / 60:.1f} min): {exps}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
