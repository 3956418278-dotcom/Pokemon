# Codex 执行说明：在 Kaggle 云端完成实验

## 使用方式

将本文档与 `NEXT_GOAL_DYNAMIC_CARD_INSTANCE.md` 一起交给 Codex。

- `NEXT_GOAL_DYNAMIC_CARD_INSTANCE.md` 规定要实现和训练的模型目标。
- 本文档规定实验环境、Kaggle 发布方式、任务监控方式和结果回收方式。

---

## 给 Codex 的执行 Prompt

你需要继续完成 Pokémon TCG AI Agent 项目的下一阶段工作。功能目标以 `NEXT_GOAL_DYNAMIC_CARD_INSTANCE.md` 为准，所有需要 PyTorch、GPU、真实 replay 或正式训练的实验均在 Kaggle 云端执行。

### 1. 本地环境边界

本地从名为 `kaggle` 的 Conda 环境启动：

```bash
conda activate kaggle
```

该环境只保证安装了：

```text
kaggle
```

本地环境没有 PyTorch，也不作为正式训练环境。执行过程中应满足：

- 本地只进行源码编辑、纯 Python 静态检查、Git 操作、Kaggle Dataset/Kernel 发布、状态查询和输出下载。
- 需要 `torch` 的测试、forward/backward、tiny-batch overfit、GPU 训练和 benchmark 均放入 Kaggle Kernel。
- 本地验证命令应与现有环境能力匹配。
- 不以“本地没有 torch”为任务阻塞原因，直接构建对应 Kaggle 云端验证流程。

### 2. Kaggle 云端职责

Kaggle Kernel 负责：

- 安装或使用 Kaggle 镜像中的 PyTorch 运行时。
- 挂载静态 CardEncoder artifacts。
- 挂载 replay 数据集。
- 运行完整自动测试。
- 执行真实 replay 数据审计。
- 执行 forward/backward 和 tiny-batch overfit。
- 训练动态单卡融合模型。
- 运行验证集评估与 CPU/GPU benchmark。
- 保存 checkpoint、指标、审计报告和日志。

正式源码继续在仓库根目录维护。Kaggle 动态代码 Dataset 由同步脚本生成：

```bash
conda run -n kaggle python scripts/sync_kaggle_dynamic_code_dataset.py
conda run -n kaggle python scripts/sync_kaggle_dynamic_code_dataset.py --check
```

新增的动态训练 Kernel 应形成唯一正式入口，避免复制出多套功能相同的训练目录。Kernel metadata、Dataset 引用、GPU 设置、代码入口和输出目录一并纳入版本管理。

### 3. 每轮实验流程

每轮云端实验按照以下顺序执行：

```text
检查当前代码与 Git 状态
→ 完成一个可验证的实现单元
→ 生成并检查 Kaggle 动态代码 Dataset
→ 发布 Dataset 新版本
→ 发布或更新 Kaggle Kernel
→ 查询 Kernel 状态
→ 持续等待运行完成
→ 下载输出文件
→ 读取日志、指标和错误
→ 根据结果继续修改
→ 再次发布与验证
```

一次实验只围绕当前阶段目标推进。每次发布前记录：

- Git commit 或当前源码版本。
- Dataset 版本。
- Kernel 版本。
- 训练配置。
- replay 日期和 split。
- 静态 artifacts 版本。

### 4. Kernel 状态监控规则

提交 Kernel 后需要持续跟踪运行状态。使用类似命令：

```bash
conda run -n kaggle kaggle kernels push -p <kernel-directory>
conda run -n kaggle kaggle kernels status <owner>/<kernel-slug>
```

状态监控采用约 5 分钟的轮询间隔：

```text
提交 Kernel
→ 查询状态
→ 如果仍在 QUEUED 或 RUNNING，等待约 5 分钟
→ 再次查询
→ 持续循环，直到 COMPLETE、ERROR 或 CANCELLED
```

运行暂时没有输出、状态仍为 `QUEUED`/`RUNNING`，或者 Kaggle 尚未生成可下载结果时，任务仍在进行中。此时继续等待并轮询，不将“暂时没有结果”视为完成，也不立即向用户返回最终答复。

单次等待控制在约 5 分钟。每次等待结束后重新查询状态，并根据新状态继续处理。

### 5. 终态处理

#### COMPLETE

Kernel 状态变为 `COMPLETE` 后下载完整输出：

```bash
conda run -n kaggle kaggle kernels output <owner>/<kernel-slug> -p <local-output-directory> -o
```

下载后必须实际读取并检查：

- 主运行日志。
- replay 审计报告。
- parser error。
- 训练与验证指标。
- unresolved mask 比例。
- tiny-batch overfit 结果。
- best/last checkpoint 是否存在。
- benchmark 结果。
- 配置和数据版本 metadata。

只有完成结果下载和内容检查，才能判断本轮实验成功。

#### ERROR

Kernel 状态变为 `ERROR` 时：

1. 下载可用的 Kernel 输出和日志。
2. 定位最早出现的有效错误。
3. 区分源码错误、路径错误、Dataset 挂载错误、依赖错误、显存错误和 Kaggle 平台错误。
4. 在正式源码中修复问题。
5. 重新同步 Dataset、发布 Kernel 并持续监控。

一次报错不代表任务结束。只要修复仍属于当前目标范围，就继续迭代。

#### CANCELLED

确认取消原因。如果属于超时、资源或人为取消，调整单轮工作量或恢复执行，并重新发布。

### 6. 输出文件要求

每次 Kaggle 运行至少输出：

```text
outputs/
├── audit/
│   └── replay_feature_audit.json
├── checkpoints/
│   ├── dynamic_card_fusion_best.pt
│   └── dynamic_card_fusion_last.pt
├── logs/
│   └── training_metrics.jsonl
├── evaluation/
│   ├── validation_metrics.json
│   └── diagnostic_examples.json
├── benchmark/
│   └── benchmark.json
├── metadata/
│   ├── run_config.json
│   ├── replay_split.json
│   └── artifact_versions.json
└── run_summary.json
```

`run_summary.json` 应集中记录：

- 运行是否成功。
- 实际执行到哪个阶段。
- 数据量与 split。
- 四项辅助任务指标。
- parser error 数量。
- Card ID/detail 对齐率。
- unresolved 比例。
- best checkpoint 路径。
- benchmark 摘要。
- 需要继续处理的问题。

### 7. 实验推进规则

按照由小到大的顺序推进：

1. Kernel 环境与 Dataset 挂载 smoke test。
2. 少量 replay 数据审计。
3. 单 batch forward/backward。
4. tiny-batch overfit。
5. 多日期训练集与时间保留集验证。
6. 正式动态单卡融合训练。
7. checkpoint 回载与 benchmark。

前一项通过后再扩大数据和训练规模。遇到错误时保留已经验证有效的部分，修复当前阻塞点并重新运行。

### 8. 结果汇报条件

以下任一情况出现时，可以向用户汇报：

- 本轮 Kernel 已经结束，输出已下载并完成分析。
- 当前阶段全部验收标准已经满足。
- Kaggle 返回了明确且短期无法自动恢复的平台阻塞。
- 继续执行需要用户提供新的权限、Dataset、凭据或会显著改变方案的决定。

汇报内容保持简洁并包含：

- 本轮完成了什么。
- Kaggle Kernel 最终状态。
- 关键指标和产物。
- 当前发现的问题。
- 下一步准备执行什么。

Kernel 仍在正常排队或运行时，保持监控流程。暂时没有输出时继续等待约 5 分钟再查询。

### 9. 当前任务完成条件

本次任务以 `NEXT_GOAL_DYNAMIC_CARD_INSTANCE.md` 的验收标准为最终条件。目标产物是经过真实 replay 训练和验证的动态单卡融合 checkpoint，而不只是能够运行的模型结构。

完成后，仓库应具备：

- 唯一正式的动态单卡训练入口。
- 可复现的 Kaggle Kernel 和 Dataset 配置。
- 真实 replay 审计结果。
- 结构化动态字段。
- 动态条件四头 Cross-Attention。
- 四项辅助任务。
- tiny-batch overfit 证据。
- 时间保留集指标。
- best/last checkpoint。
- checkpoint 回载测试和 benchmark。

持续推进到这些结果经过 Kaggle 云端实际验证。
