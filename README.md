# EMCStereo

**Attention-Enhanced Stereo Matching for Thin-Structure Depth Estimation with a Synthetic Tree-Branch Benchmark**

Yida Lin, Bing Xue, Mengjie Zhang (Victoria University of Wellington) ·
Sam Schofield, Richard Green (University of Canterbury)

> 📄 Paper: *(add the link once the paper is online)*

Official code and trained weights for **EMCStereo**, a PSMNet-style cost-volume
network with three lightweight attention modules injected at the points where
thin-structure detail is normally lost:

| Letter | Module | Where it is injected |
|---|---|---|
| **E** | **EMA** — Efficient Multi-scale Attention | on the deep 128-channel semantic feature |
| **M** | **MSFblock** — Multi-Scale Fusion block | over the four SPP pyramid branches, replacing concatenation |
| **C** | **CoordAtt** — Coordinate Attention | on the final 32-channel matching feature |

Because MSFblock collapses the four pyramid branches into one, the full model is
**2.0 % smaller** than the same backbone without attention (5,121,272 vs
5,225,152 parameters) and the three modules cost **1.7 %** of inference time.

This repository ships the model, the training/evaluation pipeline, the
VirtualTree data loader with the exact train/val/test splits, and the
**deployed checkpoint** behind every VirtualTree number in the paper.

---

## Results

### VirtualTree (552 pairs per split, 1920×1088)

| Split | EPE ↓ (px) | D1-all ↓ (%) | Bad 1.0 ↓ (%) | Bad 3.0 ↓ (%) | RMSE ↓ (px) | δ₁ ↑ (%) |
|---|---|---|---|---|---|---|
| Validation (selects the checkpoint) | 1.315 | 5.80 | 15.33 | 7.33 | 4.83 | 96.20 |
| **Test (held out)** | **1.308** | **5.96** | **15.16** | **7.31** | **4.71** | **96.11** |

Reproduced by `checkpoints/emcstereo_virtualtree_best.pth` — the exact numbers
are in [`results/`](results/).

### Cross-dataset (trained on each benchmark's public training set)

| Dataset | Pairs | EPE ↓ (px) | D1-all ↓ (%) | RMSE ↓ (px) | AbsRel ↓ | δ₁ ↑ (%) |
|---|---|---|---|---|---|---|
| SceneFlow | 3,545 | 1.00 | 3.59 | 5.13 | 0.097 | 97.67 |
| KITTI 2012 | 19 | 0.80 | 3.25 | 2.79 | 0.022 | 98.74 |
| KITTI 2015 | 20 | 0.73 | 2.20 | 1.94 | 0.027 | 98.63 |
| ETH3D | 3 | 0.62 | 3.86 | 1.50 | 0.078 | 92.60 |
| Middlebury | 5 | 3.19 | 12.65 | 9.43 | 0.050 | 96.17 |

"Pairs" is the size of the 10 % held-out split, which also supplies the
early-stopping signal, so these are selection-biased upper bounds, not
leaderboard entries. VirtualTree is the only benchmark whose evaluation split is
disjoint from checkpoint selection.

### Ablation (8-way grid, matched 100-epoch budget, VirtualTree validation)

| EMA | MSF | Coord | Params (M) | EPE ↓ (px) | D1-all ↓ (%) | Bad 1.0 ↓ (%) | RMSE ↓ (px) |
|:---:|:---:|:---:|---|---|---|---|---|
| | | | 5.225 | 1.749 | 8.01 | 19.73 | 5.346 |
| ✓ | | | 5.226 | 1.740 | 8.07 | 19.99 | 5.300 |
| | ✓ | | 5.120 | 1.777 | 8.10 | 19.87 | 5.401 |
| | | ✓ | 5.226 | 1.785 | 8.15 | 19.97 | 5.407 |
| ✓ | ✓ | | 5.120 | 1.738 | 8.00 | 19.79 | 5.318 |
| ✓ | | ✓ | 5.227 | 1.736 | 8.07 | 19.73 | 5.288 |
| | ✓ | ✓ | 5.121 | 1.792 | 8.20 | 20.05 | 5.435 |
| ✓ | ✓ | ✓ | 5.121 | 1.752 | 8.08 | 19.90 | 5.365 |
| *Deployed model (300 ep.)* | | | 5.121 | **1.315** | **5.80** | **15.33** | **4.826** |

An independent repeat of the no-attention baseline puts the run-to-run noise
floor at **0.009 px EPE**, so at matched budget the attention stack is
accuracy-neutral. The deployed model's margin is a *budget* effect, not an
architectural one — see the paper.

---

## Installation

```bash
git clone https://github.com/<your-account>/EMCStereo.git
cd EMCStereo
pip install -r requirements.txt
```

Tested with Python 3.9–3.12 and PyTorch 2.x (CUDA or CPU). Install the PyTorch
build that matches your CUDA version from [pytorch.org](https://pytorch.org)
first if you need GPU support.

**OpenEXR note.** VirtualTree ground truth is stored as UE5 `.exr` depth. The
loader sets `OPENCV_IO_ENABLE_OPENEXR=1` and uses OpenCV when its build ships the
OpenEXR codec, otherwise it falls back to the `OpenEXR` + `Imath` packages. If
you hit EXR read errors, `pip install OpenEXR Imath`.

---

## Quick start — run the trained model on one stereo pair

```bash
python infer_real_pair.py \
  --ckpt checkpoints/emcstereo_virtualtree_best.pth \
  --left  left.png --right right.png \
  --out disparity.png --raw-out disparity.npy
```

Writes a JET-colourised disparity map over `[0, 192]` and, with `--raw-out`, the
raw float disparity. Inputs must be **rectified**; the script applies the same
`DivisiblePad(32)` + ImageNet normalisation as the val/test transform.

Loading the model in your own code:

```python
import torch
from src.emc_stereo import EMCStereo

class Cfg:
    MAX_DISP = 192

model = EMCStereo(Cfg()).eval()
ckpt = torch.load('checkpoints/emcstereo_virtualtree_best.pth',
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])          # 5,121,272 parameters

out = model({'left': left, 'right': right})   # both [B, 3, H, W], H and W divisible by 32
disp = out['disp_pred']                       # [B, H, W] float disparity in pixels
```

Convert disparity to metric depth with the ZED Mini rig used to render
VirtualTree — baseline `B = 6.3 cm`, focal length `f = 960 px` (HFoV 90° at
1920 px width): `Z_cm = f · B / d`. This is exactly the relation
`data/virtualtree.py` inverts to turn the UE5 EXR depth buffer into ground-truth
disparity.

---

## Dataset layout

Point `EMCSTEREO_DATA_ROOT` (VirtualTree) and `EMCSTEREO_OPEN_ROOT` (the public
benchmarks) at your data, or pass `--data-root` on the command line. The default
is `<repo>/datasets/`:

```
datasets/
  Virtual_branches_data/     VirtualTree — splits in data/virtualtree_{train,val,test}.txt
  KITTI2012/  KITTI2015/  ETH3/  Middlebury/  Scene Flow/
```

VirtualTree holds **5,520 stereo pairs** rendered in Unreal Engine 5 from 115
distinct scenes with a simulated ZED Mini rig; the EXR depth buffer gives dense,
exact disparity for thin-branch geometry. The split files in [`data/`](data/)
hold 4,416 / 552 / 552 pairs (train / val / test), one pair per line, three
space-separated paths relative to `Virtual_branches_data/`:

```
left image/left_9.png    left depth/depth_9.exr    right image/right_9.png
```

```bash
export EMCSTEREO_DATA_ROOT=/path/to/Virtual_branches_data
export EMCSTEREO_OPEN_ROOT=/path/to/open_datasets
```

---

## Evaluation

```bash
python evl.py --ckpt checkpoints/emcstereo_virtualtree_best.pth --split test
python evl.py --ckpt checkpoints/emcstereo_virtualtree_best.pth --split val
```

Writes `output/eval/<tag>/eval_results.txt` plus the first N disparity
visualisations (`--save-count`, default 10) as
`output/eval/<tag>/disparity/disp_XXXX.png` — **prediction on top, ground truth
underneath**, JET over `[0, 192]`.

Evaluation protocol, identical everywhere in the paper: fp32, batch 1,
`DivisiblePad(32)` removed *before* scoring, valid mask `0 < gt < 192`, metrics
averaged per image, **no test-time augmentation**.

---

## Training

```bash
# VirtualTree, the deployed recipe
python train.py --epochs 300 --batch-size 4 --crop-size 256 512

# eval only, reusing an existing best.pth
python train.py --skip-train --ckpt checkpoints/emcstereo_virtualtree_best.pth

# the 8-way module ablation (one experiment per process; pin a GPU each)
CUDA_VISIBLE_DEVICES=0 python train_ablation.py --experiment ema
python train_ablation.py --experiment all

# the public benchmarks
python train_open_datasets.py --datasets "Scene Flow" --gpus 0,1
python train_open_datasets.py --datasets KITTI2015
```

| Setting | Value |
|---|---|
| Optimizer | AdamW (β = 0.9/0.999, wd 10⁻⁴) |
| Learning rate | 5 × 10⁻⁴, cosine, 500-iter warm-up |
| Precision / clipping | AMP; gradient clip 1.0; seed 0 |
| Max disparity / crop / batch | 192; 256 × 512; 4 |
| Colour augmentation | Jitter 0.4/0.4/0.4/0.16, asymmetric p = 0.2 |
| Epochs (VirtualTree) | 300, warm-started from a VirtualTree checkpoint |
| Epochs (ablation) | fixed 100 from scratch, patience 50 |
| KITTI / Middlebury | from scratch, up to 300, patience 50 |
| ETH3D | 1-epoch fine-tune of the SceneFlow checkpoint |
| SceneFlow | 288 × 576, effective batch 24, weight EMA 0.9995, epoch 108 |

One 100-epoch VirtualTree run takes about 79 h on a single GPU of a shared
cluster. Training loss is the three-scale smooth-L1 of PSMNet with weights
(0.5, 0.7, 1.0), plus prediction clamping and a loss cap that keep a single
pathological batch from exploding the gradients.

### Measured cost

fp32, batch 1, on an **NVIDIA RTX 3060 Laptop GPU**:

| | |
|---|---|
| Parameters | 5,121,272 (5,225,152 without attention) |
| Latency @ 1248 × 384 | 393 ms |
| Peak memory @ 1248 × 384 | 2.37 GiB |
| Peak memory @ 1920 × 1088 | 10.27 GiB |
| The three attention modules | 3.31 ms per view → 6.6 ms, i.e. 1.7 % |

```bash
python bench_runtime.py --ckpt checkpoints/emcstereo_virtualtree_best.pth \
                        --height 384 --width 1248 --runs 5
python bench_runtime.py --modules          # time the three modules alone
python count_ablation_params.py            # exact parameter count of all 8 variants
```

---

## Repository layout

```
src/                        the model
  emc_stereo.py             EMCStereo: forward + three-scale smooth-L1 loss
  emc_stereo_backbone.py    PSMNet backbone with EMA / MSFblock / CoordAtt injected
  attention_modules.py      EMA, MSFblock, CoordAtt as used by the backbone
  cost_processor.py         PSM cost volume + stacked hourglass aggregation
  disp_processor.py         soft-argmin disparity regression
  submodule.py              conv/BN/ReLU and BasicBlock primitives
  trainer.py                optional adapter for the OpenStereo-2 framework;
                            not used by train.py / evl.py, needs the external
                            `stereo` package

data/                       datasets
  virtualtree.py            VirtualTree loader + transforms (EXR depth → disparity)
  open_datasets.py          ETH3D / KITTI 2012 / KITTI 2015 / Middlebury / SceneFlow
  virtualtree_{train,val,test}.txt   the exact splits (4416 / 552 / 552 pairs)

modules/                    reference implementations of the three published
                            attention modules, as cited in the paper

ablation_model.py           EMCStereoAblation — the same backbone with EMA / MSF /
                            CoordAtt individually toggleable
train.py                    VirtualTree train + eval
train_ablation.py           the 8-way module grid (reuses train.py verbatim)
train_open_datasets.py      train + eval on the five public benchmarks
evl.py                      standalone evaluation + disparity visualisations
infer_real_pair.py          run a checkpoint on one rectified stereo pair
bench_runtime.py            latency / memory / parameter measurement
count_ablation_params.py    exact parameter count of all eight ablation variants

checkpoints/                the deployed VirtualTree checkpoint (see its README)
results/                    the eval files and reference metrics behind the paper
assets/                     qualitative figures from the paper
tools/                      reproducibility scripts (see tools/README.md)
```

`train_open_datasets.py` is the script referred to internally as
`train_others.py`; only the filename changed.

---

## Citation

```bibtex
@inproceedings{lin2027emcstereo,
  title     = {{EMCStereo}: Attention-Enhanced Stereo Matching for Thin-Structure
               Depth Estimation with a Synthetic Tree-Branch Benchmark},
  author    = {Lin, Yida and Xue, Bing and Zhang, Mengjie and
               Schofield, Sam and Green, Richard},
  booktitle = {Proc. IEEE Int. Conf. Robotics and Automation (ICRA)},
  year      = {2027}
}
```

## Acknowledgements

EMCStereo builds on the cost-volume backbone of
[PSMNet](https://github.com/JiaRenChang/PSMNet) and injects three published
attention modules:

- **EMA** — Ouyang *et al.*, "Efficient Multi-Scale Attention Module with
  Cross-Spatial Learning", ICASSP 2023
- **CoordAtt** — Hou *et al.*, "Coordinate Attention for Efficient Mobile Network
  Design", CVPR 2021
- **MSFblock** — Xie *et al.*, "SHISRCNet: Super-resolution and Classification
  Network for Low-resolution Breast Cancer Histopathology Image", MICCAI 2023

## License

Released under the MIT License — see [LICENSE](LICENSE). The reference module
implementations in `modules/` and the PSMNet-derived backbone remain subject to
their original authors' terms.
