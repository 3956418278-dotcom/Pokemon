# Decision Agent V1 — Policy–Value foundation

This package learns a complete visible-board representation shared by a legal-option Policy head and a LOSS/DRAW/WIN Value head. `CardInstanceEncoder` is only the board input layer: its random Card ID embedding is trained end to end from Policy and Value losses. There is no independent card pretraining or card checkpoint.

The implementation runs beside the existing static/dynamic card route and does not import its model modules. It reuses `data.replay_dataset.ReplayDecisionDataset`, `data.observation_parser.parse_observation`, `data.game_memory.GameMemoryState`, and the audited functions in `data.legal_options`.

Visibility is limited to `observation.current`, public `observation.logs`, and `observation.select`. `DecisionSampleV1` retains no raw observation, `visualize`, opponent hidden hand/deck identities, or hidden prize identities. Card rule identity (`card_id`) and per-game instance identity (`serial`) remain separate; serial is used only for tracking and contextual references.

Runtime artifacts are written under `outputs/decision_agent_v1/`. The four stage-01 commands are:

```bash
/home/feng/miniforge3/envs/ml/bin/python -m decision_agent_v1.scripts.audit_interfaces
/home/feng/miniforge3/envs/ml/bin/python -m decision_agent_v1.scripts.smoke_pipeline
/home/feng/miniforge3/envs/ml/bin/python -m decision_agent_v1.scripts.tiny_overfit
/home/feng/miniforge3/envs/ml/bin/python -m decision_agent_v1.scripts.run_legal_selfplay
```

The source tree is partitioned into contracts, existing-interface adapters, the deterministic legal baseline, dataset/collation, shared models, joint losses/metrics, minimal inference, four formal scripts, and three concentrated test files. Generated audits, metrics, checkpoints, caches, and self-play records stay in the output partition.

Stage 01 acceptance is **PASS**. Stage 02 full-corpus training is also **PASS**: the reusable dated cache is under `outputs/decision_agent_v1/cache/`, the four formal checkpoint variants are under `outputs/decision_agent_v1/checkpoints/`, and the machine-readable temporal validation/test results are in `outputs/decision_agent_v1/metrics/full_training/final_evaluation.json`. The focused suite passes 9 tests and the repository suite passes 105 tests. Belief, PPO, search, and submission-agent work remain outside the completed Policy–Value training stage.
