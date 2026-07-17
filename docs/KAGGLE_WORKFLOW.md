# Kaggle 工作流

所有命令从仓库根目录执行。动态训练仍因静态 artifact contract 未接入而暂停；下面的目录、同步、发布和状态查询命令对应当前真实结构。

## Replay 与热门牌组提取

```bash
kaggle kernels push -p kaggle/kernels/replay_extract
kaggle kernels status f7e6n5g4/ptcg-popular-deck-extract
kaggle kernels output f7e6n5g4/ptcg-popular-deck-extract \
  -p outputs/replay_extract -o
```

Kernel 内输出位于 `/kaggle/working/outputs/replay_extract/`。

同一次公开 replay 遍历会写出热门牌组文件，以及：

- `replays/replays.zip` 与 `replays/replay_index.csv`：每日期稳定选取最多 500 局，合并为一个归档；
- `decks/YYYY-MM-DD/`：每个 episode/seat 至多一条完整牌组观察及当日热门牌组；
- `statistics/YYYY-MM-DD/`：使用当日全部有效完整牌组计算的单卡与 pair 频率；
- `reports/`：全局 extraction summary、errors、audit 与 import manifest。

决策点导出继续使用现有入口，并统一写入相同目录：

```bash
python scripts/import_online_replay_decisions.py \
  --daily-replay-dir /path/to/mounted/replays \
  --output-dir outputs/replay_extract
```

该入口只导出真实 `select` 决策点（无效的 `include_no_select` 接口已删除），逐 replay 写出紧凑的 `decisions/decision_references.jsonl.gz` 和 `reports/replay_feature_audit.json`。完整 observation、日志、卡牌实例和 Memory 不重复写入；训练时按含 `replay_key` 的 decision key 回读原始 Replay，并校验 source content hash、observation fingerprint、parser/schema version 后重建八类输入，任一不一致都会明确失败。

无 EpisodeId 和 replay id 时，`replay_key` 使用单个 Replay JSON 对象的
canonical SHA-256，不使用路径、JSONL 行号或 ZIP member。定位字段仍单独保留，
同内容的无 ID Replay 会被去重。

完整决策合同审计统一运行：

```bash
python scripts/audit_replay_decision_contract.py \
  outputs/replay_extract/replays/replays.zip \
  --output-dir outputs/replay_decision_contract_audit_v2
```

归档应包含 7,500 场；审计目录只包含 `audit.json`、
`action_semantics.csv`、`equivalence_resolution.csv`、
`policy_mask_reasons.csv`、`turn_owner_audit.csv` 和 `errors.jsonl`。
`agent_index != current.yourIndex`、显式 turn owner 与奇偶回合公式冲突、非法
action index、DecisionKey 碰撞或 masking invariant 失败都会令命令非零退出，
对应样本不会生成训练标签。

热门牌组的完整 variant 使用 Card ID multiset，因此洗牌顺序不产生新 variant；Pokémon+Energy archetype 分组仍是独立逻辑。主流程依赖 `EN_Card_Data.csv` 的 Card Type，找不到时会列出搜索路径并停止，不会生成空 signature 结果。

归一化统计必须明确训练时间窗，例如：

```bash
python scripts/normalize_replay_statistics.py \
  --input-dir outputs/replay_extract/statistics \
  --output-dir outputs/replay_extract/statistics_normalized \
  --start-date 2026-07-01 \
  --end-date 2026-07-14
```

也可重复传入 `--date YYYY-MM-DD`。`normalization_summary.json` 会保存 requested window、included/excluded dates 和实际 deck 总数；reserved/test 日期必须位于窗口外。

## cg Runtime

发布 builder 并下载构建结果：

```bash
kaggle kernels push -p kaggle/builders/cg_runtime
kaggle kernels status f7e6n5g4/ptcg-cg-runtime
kaggle kernels output f7e6n5g4/ptcg-cg-runtime \
  -p outputs/cg_runtime -o
```

本地直接构建时默认写入 `outputs/cg_runtime/`：

```bash
python kaggle/builders/cg_runtime/build_cg_runtime.py
```

准备发布 Dataset 时，可显式写入被 Git 忽略的 Dataset 目录，再创建版本：

```bash
python kaggle/builders/cg_runtime/build_cg_runtime.py \
  --output-dir kaggle/datasets/cg_runtime
kaggle datasets version -p kaggle/datasets/cg_runtime \
  --dir-mode zip -m "update cg runtime"
```

## 动态代码 Dataset

`kaggle/datasets/dynamic_code/` 长期只维护 `dataset-metadata.json`。以下内容由同步脚本生成并被 Git 忽略：`data/`、`models/`、`training/`、`scripts/`、`configs/`、`tests/` 和 `source_manifest.json`。

```bash
python scripts/sync_kaggle_dynamic_code_dataset.py
python scripts/sync_kaggle_dynamic_code_dataset.py --check
kaggle datasets version -p kaggle/datasets/dynamic_code \
  --dir-mode zip -m "sync dynamic source"
```

同步报告写入 `outputs/dynamic_code_dataset/last_sync.json`。manifest 记录根源码路径、新 Dataset 目标路径、文件 hash 和 publication lineage。

## 动态训练 Kernel

```bash
kaggle kernels push -p kaggle/kernels/dynamic_training
kaggle kernels status f7e6n5g4/ptcg-dynamic-card-instance-train
kaggle kernels output f7e6n5g4/ptcg-dynamic-card-instance-train \
  -p outputs/dynamic_card_training -o
```

当前 Kernel 只挂载动态代码 Dataset 和指定 replay 日期 Dataset，不再引用已退出主线的静态 Kernel。启动后会在 `/kaggle/working/outputs/dynamic_card_training/run_summary.json` 写入 `StaticArtifactContractNotConfigured`，随后非零退出。

收到 colleague artifact contract 并完成 `StaticCardAdapter` 接入后，才恢复正式训练命令、静态 Dataset source 和结果产物说明。
