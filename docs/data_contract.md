# Data Contract

The training entry point expects dataset paths and score columns to be configured before running `train.py`. Keep those assumptions explicit so experiments can be reproduced on a new machine.

## Required Inputs

- Essay text column used by the dataset provider.
- Score or rubric label columns consumed by the model head.
- Train, validation, and test split files.
- Any prompt, topic, or metadata fields used during preprocessing.

## Pre-Run Checks

- Confirm all input files are UTF-8 encoded.
- Check that score ranges match the model output scaling.
- Verify split counts before starting training.
- Keep raw datasets outside Git unless a separate data policy allows tracking them.

## Path Handling

Prefer documented relative paths, symlinks, or environment variables over hard-coded local absolute paths. Record the data revision with each reported result.
