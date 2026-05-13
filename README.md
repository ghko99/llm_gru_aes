# Automated Essay Scoring with LLM-GRU Architecture

This repository contains an automated essay scoring experiment that combines a pretrained Korean language model representation with a GRU-based scoring head.

## Files

- `train.py`: main training entry point.
- `data_provider/`: dataset loading and preprocessing code.
- `models/`: model definitions for essay scoring.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Prepare the dataset paths expected by `train.py`, then start training:

```bash
python train.py
```

Training configuration and data paths should be checked in the script before running on a new machine.
