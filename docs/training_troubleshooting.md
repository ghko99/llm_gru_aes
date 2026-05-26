# Training Troubleshooting

Use these checks when LLM-GRU essay scoring training fails or behaves unexpectedly.

## Data Loader

- Inspect one batch before launching a full run.
- Confirm tokenized shapes match model expectations.
- Check that labels are numeric and scaled correctly.
- Verify split paths point to the intended dataset version.

## Model Training

- For CUDA memory errors, reduce batch size or sequence length.
- For unstable loss, inspect learning rate, gradient clipping, and label scale.
- For poor validation metrics, compare preprocessing settings against the previous known run.

## Debug Record

Save the failing command, config values, batch shape summary, and stack trace with the run folder.
