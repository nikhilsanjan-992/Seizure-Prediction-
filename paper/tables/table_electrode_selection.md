# Table: Electrode Selection — Method Description and Current Status

Status: **selection procedure not yet executed** (no dataset). `results/metrics/
selected_channels.json` does not exist, so `top_5` and `top_4` are NOT AVAILABLE.

## Procedure (as implemented in `src/electrode_selection.py`)

| Element | Value |
|---|---|
| Input windows | labeled 10 s training windows (seizure vs background) |
| Per-channel features (11) | mean, std, variance, RMS, line length, Hjorth mobility, Hjorth complexity, relative band power (delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz) |
| Power spectrum | Welch, nperseg = 256, 50% overlap |
| Importance statistic | mean ANOVA F-statistic across features per channel, contrasting seizure vs background training windows |
| Normalization | channel score / largest channel's mean F -> [0, 1] |
| Data scope | training split only (validation/test never used) |
| Top-5 | `top_5` = highest 5 ranked channels |
| Top-4 | `top_4` = highest 4 ranked channels (subset of top-5) |
| Determinism | no random sampling; seed 42 |
| Output | `results/metrics/selected_channels.json` |

## Selected channels (to be populated after Phase 4)

| Rank list | Channels | Status |
|---|---|---|
| top_5 | — | NOT AVAILABLE (selection not run) |
| top_4 | — | NOT AVAILABLE (selection not run) |

The selected channels, once measured, are the top-ranked channels under this procedure for
the training data of the study; they are not claimed to be universally optimal.