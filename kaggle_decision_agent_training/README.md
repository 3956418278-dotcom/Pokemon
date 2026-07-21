# Kaggle Decision Agent V1 training kernel

This is a thin offline GPU entry point. It discovers the packaged `decision_agent_v1/` code Dataset, uses `data.online_replay_importer.prepare_mounted_daily_replays` to enumerate the mounted daily Replay datasets, writes an explicit dated index/config, and invokes the same cache audit and training modules used locally. It does not contain a second parser, model, or training implementation.

All Kaggle artifacts are written to `/kaggle/working/decision_agent_v1_outputs/`. A local wiring check is available with:

```bash
python kaggle_decision_agent_training/run_training.py --code-root . --dry-run
```

The code Dataset is mechanically staged from the authoritative local sources, not maintained by hand:

```bash
python kaggle_decision_agent_training/package_code.py
```
