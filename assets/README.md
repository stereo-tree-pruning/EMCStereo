# assets/

Qualitative figures from the paper.

| File | Figure |
|---|---|
| `qual_synth.png` | Four VirtualTree validation scenes, two per row: left RGB, ground-truth disparity from the UE5 geometry buffer, and the EMCStereo prediction. Both disparity panels share the JET map over `[0, 192]`. Built by `tools/figures/make_virtualtree_figure.py`. |
| `qual_sceneflow.png` | The SceneFlow qualitative panel. Built by `tools/figures/make_sceneflow_figure.py`. |
| `flow_chart.png` | The EMCStereo architecture diagram: PSMNet backbone with EMA on the deep semantic feature, MSFblock fusing the four SPP branches, and CoordAtt on the final matching feature. |
