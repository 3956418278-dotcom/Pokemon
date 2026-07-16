# Kaggle 工作流

## 当前状态

Kaggle 仍用于公开 replay 提取、未来静态训练和后续动态训练，但当前训练边界处于暂停状态。

| 阶段 | 状态 |
|---|---|
| Replay 与热门牌组提取 | 保留现有 `kaggle_extract/` 流程 |
| 静态训练 | 等待 colleague 模块导入后，由其目录和说明负责 |
| 动态训练 | 等待 `StaticCardAdapter` 完成真实接入后恢复 |

旧静态训练路线、旧产物文件名和旧模型训练步骤已退出主线，不在本文提供运行命令。当前也不提供动态正式训练命令，避免在 contract 未确定时误启动。

## Replay 提取

`kaggle_extract/` 继续负责公开 replay 与热门牌组提取。根仓库的 replay parser、decision sample 和审计工具可在不依赖静态产物时继续开发与测试。

## 动态代码 Dataset

`kaggle_dynamic_code_dataset/` 只保留 `dataset-metadata.json`。`data/`、`models/`、`scripts/`、`training/`、`configs/`、`tests/` 和 `source_manifest.json` 都是生成内容，不在工作树中维护副本。

未来发布前由以下脚本从根仓库正式源码生成：

```bash
python scripts/sync_kaggle_dynamic_code_dataset.py
python scripts/sync_kaggle_dynamic_code_dataset.py --check
```

## 暂停行为

- `training/train_dynamic_card_fusion.py` 在训练开始前检查 adapter `ready`。
- `kaggle_dynamic_training/run_dynamic_card_training.py` 启动时写出明确暂停原因并以非零状态退出。
- 两个动态 benchmark 入口执行同一检查。
- 所有入口缺少 contract 时抛出 `StaticArtifactContractNotConfigured`，不会搜索旧产物、猜测目录或生成替代静态向量。

## 恢复条件

恢复静态和动态 Kaggle 流程前，必须由 colleague 提供完整模块说明及 artifact contract。完成 adapter 加载、manifest 版本记录、Card ID/unknown 语义、detail 对齐和端到端验证后，再补充新的 Kernel 命令与产物树。
