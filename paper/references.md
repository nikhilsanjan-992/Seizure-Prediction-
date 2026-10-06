# References — NeuroSelect Paper

Reference policy: this project does not store a bibliography. No papers, citations, or
dataset references have been used or verified within the project. To comply with the
no-fabrication rule, no DOI, title, author, year, journal, or URL is asserted as verified
below.

## References requiring verification

The following topics need real, verifiable references before publication. Each must be
confirmed against the authoritative source before inclusion:

1. Seizure prediction from EEG — survey or landmark papers establishing the problem and
   outcome measures.
2. Electrode subset / montage reduction for seizure detection or prediction — prior art on
   minimizing channel counts.
3. CNN-BiLSTM applied to EEG (seizure-related) — representative architecture references.
4. EDF file format specification.
5. Dataset source citation(s) for whichever real dataset is eventually used (e.g., if
   CHB-MIT, the PhysioNet contribution; if used, its license terms and subject/acquisition
   details). No dataset has been installed, so no dataset is cited as used.
6. Statistical definitions used here — ANOVA F-statistic, balanced class weighting, and the
   evaluation metrics in Section 9 (standard definitions; authoritative textbooks/tutorial
   sources required for citation).

## Candidate standard software references (NOT verified in this project)

The software packages below are used directly by the code. The following entries are the
well-known canonical citations, but they are listed as *candidates only*: verify the exact
authors, year, title, venue, volume, pages, and DOI against the publisher before use.

- MNE-Python: Gramfort et al., "MEG and EEG data analysis with MNE-Python", Frontiers in
  Neuroscience, 2013. (used in `src/eeg_pipeline.py`, `src/preprocessing.py`)
- TensorFlow: Abadi et al., "TensorFlow: A system for large-scale machine learning", OSDI,
  2016. (used via `src/model.py`, `src/train.py`)
- scikit-learn: Pedregosa et al., Journal of Machine Learning Research, 2011. (used in
  `src/train.py`, `scripts/run_baseline.py`)
- NumPy: Harris et al., Nature, 2020.
- SciPy: Virtanen et al., Nature Methods, 2020.
- pandas: McKinney, Proceedings of the 9th Python in Science Conference, 2010.
- Matplotlib: Hunter, Computing in Science & Engineering, 2007.

These candidate entries must be checked against a citation manager or the publisher's site;
the project itself does not attest to their completeness or exactness.