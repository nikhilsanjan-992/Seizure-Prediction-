# Paper Readiness

Date: Phase 11 documentation pass. This report records the true readiness of the NeuroSelect
research artifacts.

## Completed

- Manuscript skeleton with all 18 sections, written to a scientific standard and grounded in
  the actual implementation (`paper/manuscript.md`).
- Methodology sections derived from source code and recorded configuration (preprocessing,
  segmentation, split, electrode-selection method, architecture, training parameters,
  evaluation protocol, leakage prevention) — no values invented.
- Paper tables: `paper/tables/table_main_results.md` (empty N/A scaffold), 
  `paper/tables/table_electrode_selection.md` (method + status).
- Figures: no files exist; an honest `paper/figures/README.md` documents the planned outputs.
- `paper/reproducibility.md` with environment, configuration, and real commands; mirrors
  `results/metrics/reproducibility.json`.
- `paper/references.md` with a candid "References requiring verification" section and
  clearly-labeled candidate software citations.
- Claim audit of README and manuscript: PASS — no promotional or unsupported claims
  (remaining "best" mentions are Keras `restore_best_weights` terminology or explicit
  negations).
- Consistency audit: results/metrics contains only `leakage_audit.json`,
  `reproducibility.json`, `research_summary.md`; README/manuscript/FINAL_STATUS all agree on
  the blocked, no-results state. No cross-file value conflicts exist because no result values
  exist.

## Experimental Evidence

None. No experiment has been run: no dataset, no electrode selection, no trained models,
no metrics. All audited items are recorded as NOT_VERIFIABLE / BLOCKED
(`results/metrics/leakage_audit.json`, `results/metrics/research_summary.md`,
`results/FINAL_STATUS.md`).

## Missing Evidence

- Real EDF dataset (name, subjects, recordings, sample rate, annotations, class
  distributions) — required for Sections 3, 6, 10.
- Phase 4 output `selected_channels.json` — required for the electrode table and Sections 6/10.
- Phase 6 outputs: `comparison.csv`, `validated_metrics.csv`, `final_research_table.csv`,
  `experiment_config.json`, `error_analysis*`, trained `models/*.keras` — required for
  Section 10 (Results) and Section 14 (Error Analysis).
- Baseline results (Section 11) — not run.
- Ablation runs (Section 12) — not run.
- Repeated runs / robustness statistics (Section 13) — not run; only seed 42 configured.
- Per-sample test predictions — not persisted by Phase 6, so ROC-AUC and paired statistics
  cannot be re-derived from artifacts.
- Verified references and dataset citation (Section references).

## Reproducibility

Partial. Full configuration and commands are documented and a machine-readable record
exists, but nothing has been executed, so there are no results or runtimes to reproduce.
Software imports and syntax verified (compileall passes; `scripts/check_setup.py` reports
setup success).

## Figures

None exist. Only the planned figure list is documented. No synthetic figures were generated.

## References

Unverified. The project stores no bibliography; `paper/references.md` lists topics and
candidate standard software references that all require verification before submission.

## Remaining Work

1. Add a real EEG dataset to `data/raw/` and run `scripts/select_electrodes.py`.
2. Run `scripts/run_experiment.py` to produce 22/5/4 models, metrics, and plots.
3. (Optional) run `scripts/run_baseline.py` and any repeated seeds / ablations.
4. Run `scripts/validate_experiment.py` to confirm leakage-free, reproducible results.
5. Fill the Results table, electrode table, and figures from validated artifacts only.
6. Add verified references and the real dataset citation.
7. Verify the backend `/predict` path end-to-end on the real dataset (log the live demo)
   if the paper will include it.

## Status Classification

- SOFTWARE COMPLETE: the research scaffold (pipeline, validation, baseline, backend,
  frontend) is implemented, imports cleanly, and builds; the backend's `/predict` path
  awaits real data/models for end-to-end verification.
- RESEARCH EXPERIMENT INCOMPLETE: no experiment has been executed; there is no dataset, no
  trained model, and no results.
- PAPER DRAFT READY FOR REVIEW: NOT at this time — the manuscript exists, but its critical
  sections (Dataset, Results, Error Analysis, Discussion) contain no data, and the
  references are unverified.

This project is not publication-ready and will not become so solely because a manuscript
file exists; a reviewed draft requires the missing experimental evidence described above.