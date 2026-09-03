# assets/

Every image on the main README. All of them are rebuilt by
[`tools/figures/make_readme_assets.py`](../tools/figures/make_readme_assets.py),
which documents the raw dump each one comes from.

| File | What it shows | Source |
|---|---|---|
| `architecture.png` | The EMCStereo block diagram: PSMNet backbone with EMA on the deep semantic feature, MSFblock fusing the four SPP branches, CoordAtt on the final matching feature. | the paper's `flow_chart.png`, flattened onto white |
| `qual_virtualtree.png` | Four VirtualTree validation scenes: left RGB, ground truth, EMCStereo. Two views with a near branch spanning much of the search range, two canopy views of thin twigs. | the paper's Fig. 2 |
| `qual_virtualtree_gallery.png` | **All ten** saved validation scenes, same three columns. Nothing selected — including the two scenes where a leaf at the lens makes the prediction speckle. | the deployed checkpoint's `disparity/` dump (epe 1.314849) + the matching `panels/left_XXXX.png` |
| `thin_structure_detail.png` | A zoomed crop of one branch fork — silhouette, crossing limb, twigs behind. | scene 7 of the same dump |
| `qual_sceneflow.png` | Two scenes from the held-out SceneFlow split: ground truth against EMCStereo. | the paper's Fig. 4 |
| `qual_real.png` | A physical ZED Mini capture: left input, EMCStereo, DEFOM-Stereo. Our map is coded over 0–50 px and the reference uses its own normalization, so only structure is comparable. | the paper's real-world figure, composed into one labelled image |
| `virtualtree_sample.png` | One raw VirtualTree **test**-split sample: left view, right view, and the ground-truth disparity obtained from the UE5 EXR depth buffer. | `left_2988.png` / `right_2988.png` / `depth_2988.exr`, line 1 of `data/virtualtree_test.txt` |
| `training_curve.png` | Validation EPE against epoch for the deployed 300-epoch run, with epoch 252 (the released checkpoint) marked. | that run's training log |
| `ablation_chart.png` | The eight ablation variants against the ±0.009 px run-to-run noise floor. | `results/expected_metrics.json` |

Every disparity panel taken from an `evl.py` dump is JET-coloured over
`[0, 192]`, and in the raw `disp_XXXX.png` files the **prediction is on top and
the ground truth underneath** — the builder splits them apart.

## Rebuilding

```bash
python tools/figures/make_readme_assets.py             # everything
python tools/figures/make_readme_assets.py --only curve ablation
```

`curve` and `ablation` need only files inside this repository plus the training
log. The rest read the evaluation dumps, which are not committed here; the
script says exactly which path is missing if one is.
