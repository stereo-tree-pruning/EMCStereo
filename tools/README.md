# tools/

Reproducibility scripts. Unlike the scripts in the repository root, these expect
artefacts that are **not** shipped here — the full set of run directories, or the
saved evaluation image dumps. They are included so the path from checkpoint to
published number and figure is fully documented.

## `val_all.py` — re-validate every run in the paper

Re-runs the validation of every checkpoint and compares each metric against
`../results/expected_metrics.json`, exiting 0 with `ALL OK` when all of them
match to `--tol` (default 2 × 10⁻³).

```bash
CUDA_VISIBLE_DEVICES=0 python tools/val_all.py --artefact-root /path/to/final_results
# options: --only ablation cross_dataset | --skip-sceneflow | --out-dir DIR
#          --data-root $EMCSTEREO_OPEN_ROOT | --tol 2e-3
```

`--artefact-root` must contain `<group>/<run>/best.pth` for each run
(`virtualtree/emcstereo_headline`, `ablation/ema`, `cross_dataset/KITTI2015`, …).
Only the headline checkpoint ships with this repository; runs whose `best.pth` is
absent are reported as `MISSING` and skipped. To validate just the released one:

```bash
mkdir -p final_results/virtualtree/emcstereo_headline
cp checkpoints/emcstereo_virtualtree_best.pth \
   final_results/virtualtree/emcstereo_headline/best.pth
python tools/val_all.py --only emcstereo_headline
```

Needs ~10 GiB of free GPU memory for full-resolution VirtualTree. Roughly 9 min
per VirtualTree split (552 pairs at 1920 × 1088) on an A40; the whole sweep is a
few hours.

## `train_new2.py` — the 2-epoch warm restart

The script that produced the `emcstereo_scratch_2ep_restart` run quoted in the
paper's training-budget section (1.644 px val / 1.606 px test). It rebuilds each
finished model from an existing run tree, fine-tunes two more epochs with a
gentle cosine schedule, and re-exports every artefact — metrics, disparity
colourmaps and `left | pred | gt | error` panels. It reuses `train.py`'s own
`run_eval` unchanged, so its numbers are directly comparable. The absolute paths
in its docstring describe the original cluster layout and are kept as
provenance.

## `figures/` — the README images and the paper's qualitative figures

These rebuild everything in `../assets/` from evaluation image dumps produced by
`evl.py` / `train_new2.py`. Regenerate the dumps first (`evl.py` writes
`output/eval/<tag>/disparity/disp_XXXX.png`, prediction on top of ground truth,
JET over `[0, 192]`) and adjust the input paths at the top of each script.

| Script | Builds |
|---|---|
| **`make_readme_assets.py`** | **every image in `../assets/`** — architecture, the VirtualTree figure and the ten-scene gallery, the detail crop, SceneFlow, the real-world comparison, the dataset sample, the training curve and the ablation chart. Its docstring lists the source dump behind each one. |
| `make_virtualtree_figure.py` | `qual_synth.png` — four VirtualTree scenes: left RGB \| ground truth \| EMCStereo |
| `make_sceneflow_figure.py` | `qual_sceneflow.png` — the SceneFlow qualitative panel |
| `make_paper_figures.py` | the real ZED Mini figure, re-colourised from the raw disparity of `infer_real_pair.py` (`--real`) |
| `check_panel_alignment.py` | proves two image dumps enumerate a split in the same order, by comparing their ground-truth halves pixel for pixel |
| `make_blur_layers.py` | the blur layers used in the flow-chart figure |

```bash
python tools/figures/make_readme_assets.py
python tools/figures/make_readme_assets.py --only curve ablation   # no dumps needed
```
