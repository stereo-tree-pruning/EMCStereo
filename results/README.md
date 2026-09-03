# Results

The raw evaluation output behind the paper's VirtualTree numbers, plus the
reference metrics for every other run.

| File | Contents |
|---|---|
| `virtualtree_val_eval.txt` | eval written by the original run on the **validation** split (552 pairs) — EPE 1.314849 |
| `virtualtree_test_eval.txt` | same on the held-out **test** split (552 pairs) — EPE 1.307740 |
| `headline_train_config.json` | the training config the deployed run actually used |
| `expected_metrics.json` | reference metrics for **all** runs in the paper (ablation grid, cross-dataset, SceneFlow), used by `tools/val_all.py` as the ground truth to compare against |

## Protocol

Every number in these files, and everywhere in the paper, was produced the same
way:

- fp32, batch size 1, **no test-time augmentation**
- `DivisiblePad(32)` removed *before* metrics, so no padded pixel is scored
- valid mask `isfinite(gt) & (0 < gt < 192)`
- metrics averaged **per image**, then over the split
- val splits for the public benchmarks are the deterministic
  `split_train_val(items, val_ratio=0.1, seed=0)` of `data/open_datasets.py`

## Reproducing them

```bash
export EMCSTEREO_DATA_ROOT=/path/to/Virtual_branches_data
python evl.py --ckpt checkpoints/emcstereo_virtualtree_best.pth --split test
python evl.py --ckpt checkpoints/emcstereo_virtualtree_best.pth --split val
```

Re-evaluating one checkpoint on different hardware agrees to 4 × 10⁻⁶ px, so any
disagreement beyond the fourth decimal points at a protocol difference, not at
numerical noise. Note that `evl.py` reports the core metric set (EPE, D1-all,
Bad 1/2/3); the fuller set in these files (Bad 0.5/4/5, RMSE, AbsRel, SqRel,
log-RMSE, δ₁₋₃) comes from `train.py --skip-train` and `tools/val_all.py`, which
share one `compute_metrics`.
