# Hidden Bias and Residual Repair for Low-Response AIPW

This repository contains the submission-facing replication package for the
paper *Hidden Bias and Residual Repair for Low-Response AIPW*.

The package includes:

- the core fixed-expert repair implementation;
- paired replication rows for the 24 reported benchmark settings;
- the paper-facing tables and diagnostic summaries;
- deterministic aggregation scripts and integrity checks.

## Verify the release

The verification command uses only the Python standard library:

```bash
python3 scripts/verify_release.py
```

Run the unit tests with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

The primary table gives every benchmark setting equal weight. Its average MSE
gains are 8.95% for AIPW, 5.19% for selective ML, 7.66% for Ma DR-BC, and
4.98% for C-TMLE.

## Layout

- `scripts/weighted_residual_selection.py`: reference implementation of the
  candidate-selection calculation used by the reported replay.
- `scripts/dml_render_weighted_residual_gamma_tables.py`: primary-table renderer.
- `scripts/dml_summarize_section4_same_sample_penalty.py`: equal-setting
  same-sample audit.
- `artifacts/`: compact paired rows, summaries, generated tables, provenance,
  and checksums used by the paper.
- `tests/`: unit tests for the estimator and aggregation logic.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact scope of the public
artifacts and commands for regenerating the same-sample audit.
