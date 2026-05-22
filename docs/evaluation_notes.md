# Evaluation Notes

Use one evaluation protocol when comparing LLM-GRU essay scoring runs.

## Metrics

Track at least:

- Quadratic weighted kappa.
- Mean absolute error.
- Root mean squared error.
- Pearson correlation when reporting trend agreement.
- Per-rubric metrics if multiple score columns are predicted.

## Result Record

Each result summary should include:

- Git commit and command line.
- Dataset split revision.
- Model configuration from `models/`.
- Preprocessing settings from `data_provider/`.
- Random seed and hardware.
- Output path for predictions and metrics.

Avoid comparing runs across different preprocessing revisions unless the report clearly marks the difference.
