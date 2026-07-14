# Colleague Static Card Module

This directory contains the colleague's implementation of the static card representation pipeline.
All static logic (including CSV reading, detail feature aggregation, static card pretraining, evaluation, and static artifact export) is self-contained in this directory.

## Directory Structure
- `configs/`: Static training configs.
- `data/`: CSV loading, parsing, and Dataset construction.
- `models/`: Static CardEncoder model structure.
- `training/`: Pretraining, exporting, and evaluation pipelines.
- `scripts/`: Static utilities.
- `tests/`: Module unit tests.
- `kaggle/`: Kaggle kernel build and training runner scripts.
