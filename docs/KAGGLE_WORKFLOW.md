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
