# Experiments

This directory contains isolated research that is not part of the current dynamic-model mainline. Each experiment owns its assumptions, tests, and retained conclusions; production modules must not import from this directory.

Current experiment:

- [`companion_csv_ppo/`](companion_csv_ppo/README.md): deterministic flat CSV card vectors and a candidate-scoring PPO prototype.

Run experiment tests explicitly rather than through the root pytest configuration:

```bash
python -m pytest experiments/companion_csv_ppo/tests -q
```
