# Figures Directory — NeuroSelect Paper

No figure files exist in this directory and none exist anywhere in the repository, because
no experiment has been run. Figures will be generated only by the real pipeline runs and no
synthetic figures were, or will be, produced.

Expected figure outputs once Phases 4 and 6 are executed (as produced by the code):

| Planned figure | Source | Path |
|---|---|---|
| Electrode importance ranking | `scripts/select_electrodes.py` -> `plot_channel_importance` | `results/plots/channel_importance.png` |
| Confusion matrix per configuration | `scripts/run_experiment.py` -> `evaluate.plot_confusion_matrix` | `results/plots/confusion_matrix_22.png` / `_5.png` / `_4.png` |
| Training curves per configuration | `scripts/run_experiment.py` | `results/plots/training_curve_*.png` |
| Baseline comparison (if run) | `scripts/run_baseline.py` | `results/plots/baseline_comparison.png` |

Current directory content: none (empty until figures are generated). Captions will be added
to the manuscript when the figures exist.