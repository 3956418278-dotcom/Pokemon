# Kaggle 工作流

本项目在本地维护唯一正式源码，在 Kaggle 完成公开 replay 提取、静态 CardEncoder 训练以及后续动态/策略训练。

## 1. 当前组件

| 目录 | 用途 |
|---|---|
| `kaggle_extract/` | 提取公开 replay 与热门牌组 |
| `kaggle_card_pretrain/` | 训练并导出静态 CardEncoder |
| `kaggle_cg_runtime/` | 构建自包含 `cg` runtime Dataset |
| `kaggle_cg_runtime_dataset/` | `cg` runtime Dataset metadata；实际 `cg/` 本地生成 |
| `kaggle_dynamic_code_dataset/` | 动态代码 Dataset metadata；代码从根源码生成 |
| `kaggle_dynamic_training/` | 唯一正式动态单卡训练 Kernel |

旧共享 PPO、旧 submission 和临时动态 smoke kernel 已从主分支移除。正式动态训练统一由 `kaggle_dynamic_training/` 进入。

## 2. 提取热门牌组

```bash
kaggle kernels push -p kaggle_extract
kaggle kernels status f7e6n5g4/ptcg-popular-deck-extract
kaggle kernels output f7e6n5g4/ptcg-popular-deck-extract -p outputs/popular_decks -o
```

主要输出：

```text
popular_deck_outputs/popular_test_decks.json
popular_deck_outputs/popular_test_decks.md
popular_deck_outputs/popular_test_deck.csv
popular_deck_outputs/popular_deck_summary.csv
```

## 3. 静态 CardEncoder

发布或更新 `cg` runtime Dataset：

```bash
kaggle kernels push -p kaggle_cg_runtime
kaggle kernels output f7e6n5g4/ptcg-cg-runtime -p outputs/cg_runtime -o

kaggle datasets version \
  -p kaggle_cg_runtime_dataset \
  --dir-mode zip \
  -m "update cg runtime"
```

运行静态训练：

```bash
python scripts/build_kaggle_card_pretrain_kernel.py
python scripts/build_kaggle_card_pretrain_kernel.py --check

kaggle kernels push -p kaggle_card_pretrain
kaggle kernels status f7e6n5g4/ptcg-card-pretrain
kaggle kernels output f7e6n5g4/ptcg-card-pretrain -p outputs/card_pretrain -o
```

正式输出：

```text
outputs/card_pretrain/artifacts/card_embeddings.pt
outputs/card_pretrain/artifacts/card_detail_tokens.pt
outputs/card_pretrain/artifacts/card_detail_masks.pt
outputs/card_pretrain/artifacts/card_detail_type_ids.pt
outputs/card_pretrain/artifacts/card_id_to_index.json
outputs/card_pretrain/artifacts/card_embedding_metadata.json
outputs/card_pretrain/checkpoints/card_encoder_best.pt
outputs/card_pretrain/checkpoints/card_encoder_last.pt
outputs/card_pretrain/logs/card_pretrain_metrics.jsonl
```

## 4. Replay 导入与审计

从已挂载的每日 replay Dataset 生成 decision index：

```bash
python scripts/import_online_replay_decisions.py \
  --episodes-index-dir /kaggle/input/pokemon-tcg-ai-battle-episodes-index \
  --use-daily-manifest \
  --daily-dataset-mount-root /kaggle/input \
  --reserve-recent-days 3 \
  --import-split train \
  --max-days 1 \
  --max-samples 4096 \
  --output-dir outputs/replay_decisions
```

审计模型实际会读取的决策点：

```bash
python scripts/audit_replay_features.py \
  /kaggle/input/pokemon-tcg-ai-battle-episodes-2026-07-07 \
  --max-samples 4096 \
  --static-id-map outputs/card_pretrain/artifacts/card_id_to_index.json \
  --output outputs/replay_feature_audit.json
```

最近日期作为 reserved split。首次运行保持少量日期与样本，先确认 parser error、Card ID 覆盖率、实例数、事件数和 option 数分布。

## 5. 生成动态代码 Dataset

Kaggle Dataset 中的 `data/`、`models/` 和 `scripts/` 不在 Git 中维护第二份副本。发布前生成：

```bash
python scripts/sync_kaggle_dynamic_code_dataset.py
kaggle datasets version \
  -p kaggle_dynamic_code_dataset \
  --dir-mode zip \
  -m "sync dynamic source"
```

检查现有生成内容是否与根源码一致：

```bash
python scripts/sync_kaggle_dynamic_code_dataset.py --check
```

`source_manifest.json` 除 Dataset 内 canonical 源码外，还记录动态 Kernel runner、
Kernel metadata 和代码 Dataset metadata 的 SHA-256。修改入口配置或挂载 metadata 后，
必须重新同步、检查并发布 Dataset；Kernel 启动时要求 v2 publication lineage 完整，
并将 manifest 中的源码 hash 与 Kaggle 实际运行脚本的 runtime hash 一并写入输出 metadata。

## 6. 动态单卡融合训练

正式配置位于 `configs/dynamic_card_fusion.json`，首次发布使用同一入口的 `configs/dynamic_card_fusion_smoke.json`；smoke 全链路通过后，将 Kernel 入口切换到正式配置并发布下一版本。当前时间划分为：

```text
train:      2026-07-08, 2026-07-09
validation: 2026-07-10
test:       2026-07-11
```

发布动态源码 Dataset：

```bash
conda run -n kaggle python scripts/sync_kaggle_dynamic_code_dataset.py
conda run -n kaggle python scripts/sync_kaggle_dynamic_code_dataset.py --check
conda run -n kaggle kaggle datasets version \
  -p kaggle_dynamic_code_dataset \
  --dir-mode zip \
  -m "sync structured dynamic card training"
```

提交并监控唯一训练 Kernel：

```bash
conda run -n kaggle kaggle kernels push -p kaggle_dynamic_training
conda run -n kaggle kaggle kernels status f7e6n5g4/ptcg-dynamic-card-instance-train
conda run -n kaggle kaggle kernels output \
  f7e6n5g4/ptcg-dynamic-card-instance-train \
  -p outputs/dynamic_card_instance \
  -o
```

Kernel 会依次运行自动测试、replay 审计、单 batch/梯度检查、tiny overfit、正式训练、时间保留集评估、checkpoint 回载和 CPU benchmark。正式结果树为：

```text
outputs/
├── audit/replay_feature_audit.json
├── checkpoints/dynamic_card_fusion_best.pt
├── checkpoints/dynamic_card_fusion_last.pt
├── logs/training_metrics.jsonl
├── evaluation/validation_metrics.json
├── evaluation/diagnostic_examples.json
├── benchmark/benchmark.json
├── metadata/run_config.json
├── metadata/replay_split.json
├── metadata/artifact_versions.json
├── metadata/kernel_wrapper_lineage.json
└── run_summary.json
```

训练子进程失败时，以子进程写出的精确 `completed_stage` 为准；Kernel wrapper 只在同一
`run_summary.json` 中追加 `kernel_wrapper_error`，不会把具体阶段覆盖成笼统的
`dynamic_training`。

## 7. 辅助命令

重新生成 baseline decks：

```bash
python scripts/make_baseline_decks.py
```

动态单卡融合 CPU benchmark 由正式 Kernel 自动执行。下列命令仅用于 Kaggle 或其他已安装
PyTorch 的环境中手工复现；不要在只安装 `kaggle` CLI 的本地 Conda 环境中运行：

```bash
python scripts/benchmark_dynamic_card_fusion.py \
  /path/to/replay-or-replay-directory \
  --checkpoint outputs/dynamic_card_instance/outputs/checkpoints/dynamic_card_fusion_best.pt \
  --config configs/dynamic_card_fusion.json \
  --static-artifact-dir outputs/card_pretrain/artifacts \
  --card-records outputs/card_pretrain/artifacts/card_data/card_records.json \
  --detail-metadata outputs/card_pretrain/artifacts/card_detail_metadata.json \
  --output outputs/dynamic_card_instance/outputs/benchmark/benchmark.json
```

`scripts/benchmark_dynamic_state.py` 仅保留为弃用兼容入口，并会转调上述正式 benchmark；
它不再构造随机静态向量或绕过 detail Cross-Attention。

静态 energy-type probe：

```bash
python scripts/linear_probe_energy_type.py
```

## 8. 未来正式提交

ActionEncoder、行为克隆、Value 和 self-play 完成后，再建立唯一 submission kernel。提交包顶层需要包含：

```text
main.py
deck.csv
模型权重与运行所需文件
```

运行路径为 `/kaggle_simulations/agent/`，提交包大小上限为 197.7 MiB。打包前验证 self-play validation、相对导入路径、CPU 延迟和内存占用。
