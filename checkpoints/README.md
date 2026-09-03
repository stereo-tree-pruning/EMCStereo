# Checkpoints

## `emcstereo_virtualtree_best.pth`

The **deployed model** — every VirtualTree number in the paper (Table
"EMCStereo on the two VirtualTree splits", the last row of the ablation table,
and the abstract) comes from this file.

| | |
|---|---|
| Architecture | full EMCStereo (EMA + MSFblock + CoordAtt), `MAX_DISP = 192` |
| Parameters | 5,121,272 |
| Epoch | 252 of a 300-epoch schedule (early-stopping patience 50) |
| Initialisation | warm-started from a VirtualTree checkpoint |
| VirtualTree **val** EPE | 1.314849 px (D1-all 5.80 %, RMSE 4.826, δ₁ 96.20 %) |
| VirtualTree **test** EPE | 1.307740 px (D1-all 5.96 %, RMSE 4.708, δ₁ 96.11 %) |
| Size | 61,866,765 bytes |
| SHA-256 | `bc2e591d9138ab09af2023fbf313c1b54bac0dcd955fe2264fd6290a93d68747` |

Verify the download:

```bash
sha256sum checkpoints/emcstereo_virtualtree_best.pth
```

### File contents

A plain `torch.save` dict — the training state, not just the weights:

| Key | Contents |
|---|---|
| `model` | `state_dict` of `EMCStereo`, 544 tensors |
| `optimizer` | AdamW state (two moment buffers; this is ~2/3 of the file size) |
| `epoch` | `252` |
| `global_step` | `279312` |
| `val_metrics` | the full metric dict measured at that epoch |

```python
import torch
from src.emc_stereo import EMCStereo

class Cfg:
    MAX_DISP = 192

ckpt = torch.load('checkpoints/emcstereo_virtualtree_best.pth',
                  map_location='cpu', weights_only=False)
model = EMCStereo(Cfg())
model.load_state_dict(ckpt['model'])     # -> <All keys matched successfully>
```

`evl.py`, `bench_runtime.py` and `infer_real_pair.py` all accept either this
dict or a bare `state_dict`, so a stripped weights-only file works too:

```python
torch.save({k: ckpt[k] for k in ('model', 'epoch', 'val_metrics')},
           'emcstereo_weights_only.pth')       # ~1/3 the size
```

### Training configuration

The exact config this run used is in
[`../results/headline_train_config.json`](../results/headline_train_config.json)
(its `data_root` / `output_dir` / `pretrained` paths are the original cluster
paths and are not needed to load the weights).

## The other runs in the paper

The paper also reports checkpoints for the eight ablation variants, the four
cross-dataset benchmarks and two SceneFlow runs. Together they are ~1.2 GB, too
large to commit here; their reference metrics are in
[`../results/expected_metrics.json`](../results/expected_metrics.json). Publish
them as GitHub **Release** assets (or on Zenodo / OneDrive) and link them from
the main README rather than committing them to the repository.

| Run | Split | EPE (px) |
|---|---|---|
| `virtualtree/emcstereo_headline` *(this file)* | val / test | 1.3148 / 1.3078 |
| `virtualtree/emcstereo_scratch_snapshot` | val | 1.6736 |
| `virtualtree/emcstereo_scratch_2ep_restart` | val / test | 1.6437 / 1.6058 |
| `ablation/none` | val | 1.7490 |
| `ablation/none_repeat` *(independent repeat, fixes the 0.009 px noise floor)* | val | 1.7404 |
| `ablation/ema`, `msf`, `coord` | val | 1.7398 / 1.7769 / 1.7853 |
| `ablation/ema_msf`, `ema_coord`, `msf_coord` | val | 1.7376 / 1.7355 / 1.7915 |
| `ablation/ema_msf_coord` | val | 1.7522 |
| `cross_dataset/KITTI2012` / `KITTI2015` | val | 0.8039 / 0.7298 |
| `cross_dataset/ETH3D` / `Middlebury` | val | 0.6163 / 3.1914 |
| `sceneflow/v4_best` | val | 1.0050 |
| `sceneflow/baseline_ep31` | val | 1.1998 |

## A note on committing a 59 MB file

GitHub accepts files up to 100 MB but warns above 50 MB. This checkpoint is
59 MB, so a normal `git push` works. If you prefer Git LFS:

```bash
git lfs install
git lfs track "checkpoints/*.pth"
git add .gitattributes checkpoints/emcstereo_virtualtree_best.pth
```

Do this **before** the first commit of the file — converting an
already-committed binary to LFS means rewriting history.
