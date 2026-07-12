> 历史快照：本清单记录 2026-07-11 整理前的文件状态，不再作为当前仓库导航维护。

# Pokemon TCG 项目文件功能清单

本清单记录当前工作区中每个项目文件的用途。`.git/`、`.agents/`、`.codex/`、`__pycache__/` 和测试缓存属于工具内部状态，不逐文件展开。

标记说明：

- **正式源码**：后续应从这里修改。
- **部署镜像**：为 Kaggle 运行复制的源码，不是独立实现。
- **生成产物**：可由源码或训练重新生成。
- **运行快照**：某次 Kaggle 运行时保存的源码或日志。
- **样例数据**：只用于结构验证或测试。
- **历史代码**：旧 agent、旧训练或旧提交链路，不属于当前特征构造主线。

## 根目录

- `.gitignore`：定义缓存、模型产物、下载数据和本地工具状态的 Git 忽略规则。
- `PROJECT_FILE_MANIFEST.md`：本文件，记录项目全部文件的职责和来源关系。
- `CARD_FEATURE_EXTRACTION_IDEAS.md`：早期卡牌特征分层、动作特征和实施阶段构想。
- `README_DYNAMIC_TEMPORAL_STATE_HANDOFF.md`：当前动态与时序特征工作的简要交接记录。
- `README_KAGGLE_RUN.md`：Kaggle 数据提取、训练、提交和静态预训练操作说明。
- `TRAINING_IDEAS.md`：旧 PPO、对手采样、先后手偏差和训练失败记录。
- `codex_dynamic_temporal_state_from_static_baseline(1).md`：当前动态实例、Ledger、Recent Events、Board Transformer 的正式实现规范。
- `state_feature_audit.md`：静态产物、cabt Observation 字段、可见性边界和当前实现状态审计。
- `baseline_decks.json`：基线牌组的结构化 Card ID 数据。
- `baseline_decks.md`：基线牌组卡池匹配和替换结果的人类可读报告。
- `decks.md`：原始牌组文本资料。
- `cri_rulebook_en.pdf`：Pokemon TCG Pocket 英文规则资料。
- `pokemon-tcg-ai-battle.zip`：下载的 Kaggle 竞赛资源压缩包。

## 配置

- `configs/card_pretrain.yaml`：当前正式静态 CardEncoder 预训练配置，包含 attention detail pooling、400 epoch 上限、早停、masking 和损失权重。

## data：正式数据代码

- `data/__init__.py`：数据包标记文件。
- `data/card_preprocessing.py`：读取 CSV/cg API，按 Card ID 聚合卡牌，构造 Attack/Ability/Effect/CardRecord，并输出静态卡牌缓存。
- `data/card_dataset.py`：构造静态特征 schema、类别与数值编码、逐 detail batch、训练目标及按 Card ID 划分数据。
- `data/state_schema.py`：当前动态状态 dataclass、区域和能量词表、固定维度以及动态 tensor collate 定义。
- `data/observation_parser.py`：把 cabt Observation 转成当前全局快照、卡牌实例、日志事件和合法选项。
- `data/game_memory.py`：维护 serial 级单局记忆、近期事件和当前简化 Ledger 特征。
- `data/replay_dataset.py`：读取变长 replay JSON，在有效决策点生成 ReplayDecisionSample，并按时间顺序重建 memory。
- `data/online_replay_importer.py`：读取每日数据集 manifest、选择日期 split、挂载或下载 replay，并构造 ReplayDecisionDataset。
- `data/replay_training_features.py`：把 replay decision samples 逐样本送入当前状态编码器并收集 board embedding 与目标。

## models：正式模型代码

- `models/__init__.py`：模型包标记文件。
- `models/card_encoder.py`：静态 CardEncoder；编码类别、数值、整卡文本和逐攻击/特性/特殊效果 detail，输出 128 维 summary 与 detail tokens。
- `models/card_pretrain_heads.py`：静态预训练的 masked field、关系、攻击归属、detail 恢复和能量辅助任务头及损失。
- `models/card_instance_encoder.py`：早期通用单卡静态、动态和 appearance 预留向量拼接接口；不是当前动态时序规范的最终融合器。
- `models/static_card_adapter.py`：按 Card ID 从静态导出产物读取 summary、detail、mask 和类型。
- `models/static_detail_aggregator.py`：以静态 card summary 为 query 对 detail token 做 attention 聚合；不是规范要求的动态条件 Cross-Attention。
- `models/dynamic_instance_encoder.py`：当前初版 32 维动态字段与 32 维临时 appearance 字段编码器，输出 64 维表示；仍需按正式结构化 schema 重做。
- `models/card_instance_fusion.py`：当前初版静态与动态向量浅层拼接融合；尚未实现规范要求的 4-head 动态条件 Cross-Attention。
- `models/board_tokenizer.py`：把当前全局、卡牌、Decision、Match、Ledger 和 Event 特征投影为 Board token；当前顺序仍不是最终规范顺序。
- `models/board_transformer.py`：两层 TransformerEncoder 初版，输出上下文化 token 和 pooled board embedding。
- `models/dynamic_state_encoder.py`：串联 Observation parser、memory、静态适配、动态编码、实例融合、tokenizer 和 Board Transformer 的初版总入口。

## training：正式静态训练代码

- `training/__init__.py`：训练包标记文件。
- `training/pretrain_card_encoder.py`：静态 CardEncoder 的数据 masking、关系任务、训练/验证、早停和 checkpoint 保存入口。
- `training/export_card_embeddings.py`：从静态 checkpoint 导出 summary、detail tensors、Card ID 映射和 metadata。
- `training/evaluate_card_embeddings.py`：计算 embedding 邻居、类别 purity、PCA 和分析报告。

## scripts：命令行工具

- `scripts/make_baseline_decks.py`：把文本牌组匹配到引擎 Card ID，并处理缺卡替换。
- `scripts/linear_probe_energy_type.py`：用导出的静态 embedding 训练线性 probe，评估 Basic Energy 等类型可分类性。
- `scripts/smoke_energy_separation.py`：快速检查未训练和训练后静态 embedding 的能量类型分离程度。
- `scripts/analyze_replay_observations.py`：分析单局 replay 的实例、选项、事件、静态 detail 对齐和 token 规模。
- `scripts/audit_episode_index_dataset.py`：只读检查 Kaggle episode index/manifest 的字段和文件结构。
- `scripts/build_replay_decision_dataset.py`：把本地 replay 转成 decision index 和汇总文件。
- `scripts/import_online_replay_decisions.py`：从挂载的每日数据集或 Kaggle API 导入 replay decision samples。
- `scripts/benchmark_dynamic_state.py`：用构造 observation 测量当前动态状态解析和编码耗时。
- `scripts/build_kaggle_dynamic_training_kernel.py`：把动态模块拼成单文件 Kaggle kernel 的旧打包工具；代码数据集方案建立后主要作为备用。
- `scripts/train_dynamic_replay_features.py`：临时 replay-to-board smoke 训练，预测 select type/context、option count 和 reward；不是 CardInstanceFusion 正式辅助预训练。

## tests：测试代码

- `tests/conftest.py`：把项目根目录加入测试导入路径。
- `tests/test_card_preprocessing.py`：测试静态缺失值、伤害、能量费用和卡牌类型预处理 helper。
- `tests/test_card_dataset.py`：测试静态 batch shape、攻击绑定、detail schema 和 Card ID split 隔离。
- `tests/test_card_encoder.py`：测试静态 summary/detail shape、padding 和旧 CardInstanceEncoder 接口。
- `tests/test_pretrain_tasks.py`：测试静态辅助任务损失可前向和反向传播。
- `tests/test_observation_parser.py`：测试可见性、隐藏卡、附着卡、异常状态、事件和动态 batch。
- `tests/test_dynamic_instance_encoder.py`：测试当前动态编码和浅层融合的 shape 与 backward。
- `tests/test_board_encoder.py`：测试静态适配、detail 聚合、Board token、Transformer 和端到端初版状态编码。
- `tests/test_replay_dataset.py`：测试真实 replay 决策点读取、agent 过滤和普通 JSON 文件名支持。
- `tests/test_online_replay_importer.py`：测试 API 导入、日期保留集、manifest 选择和挂载每日数据集读取。
- `tests/test_train_dynamic_replay_features.py`：测试临时动态 smoke 训练入口的日期目录选择和无 torch 的 help 命令。

## notes：协作记录

- `notes/feature_architecture.md`：早期四类特征边界记录；其中 appearance/热门度描述已经被最新决定取代。
- `notes/kaggle_replay_structure_probe_v13.json`：Kaggle version 13 的最小线上 replay 结构探测结果，避免重复运行。

## artifacts：本地静态预处理缓存

- `artifacts/card_data/card_records.json`：本地聚合后的 CardRecord 数据。
- `artifacts/card_data/card_id_to_index.json`：本地 Card ID 到静态 dataset index 的映射。
- `artifacts/card_data/card_preprocess_summary.json`：本地预处理卡牌数量、detail 数量和数据来源汇总。

## data_from_submission：样例与派生数据

- `data_from_submission/ptcg-replay-data-miner.ipynb`：原有 replay 数据挖掘 Notebook。
- `data_from_submission/replays/episode-84817357-replay.json`：单局本地 replay fixture，仅用于接口和结构测试。
- `data_from_submission/replay_audit/episode-84817357-replay-feature-audit.json`：该样例 replay 的机器可读字段统计。
- `data_from_submission/replay_audit/episode-84817357-replay-feature-decision.md`：基于该样例作出的动态特征初步判断。
- `data_from_submission/replay_dataset/decision_index.csv`：从样例 replay 生成的决策点索引。
- `data_from_submission/replay_dataset/summary.json`：样例 decision dataset 的规模和最大长度汇总。

## kaggle_card_pretrain：静态预训练 Kaggle kernel

该目录是静态预训练的自包含部署目录，不是当前正式源码的第二份实现。

- `kaggle_card_pretrain/kernel-metadata.json`：静态预训练 kernel 的 Kaggle ID、入口、CPU 和数据源配置。
- `kaggle_card_pretrain/run_card_pretrain.py`：在 Kaggle working 目录重建内嵌源码、运行静态训练、导出和评估的自包含入口。
- `kaggle_card_pretrain/configs/card_pretrain.yaml`：该部署快照中的旧 1-epoch 配置，不是当前正式 400-epoch 配置。
- `kaggle_card_pretrain/data/__init__.py`：部署数据包标记。
- `kaggle_card_pretrain/data/card_preprocessing.py`：静态预处理的部署快照。
- `kaggle_card_pretrain/data/card_dataset.py`：静态 dataset/collate 的部署快照。
- `kaggle_card_pretrain/models/__init__.py`：部署模型包标记。
- `kaggle_card_pretrain/models/card_encoder.py`：静态 CardEncoder 的部署快照。
- `kaggle_card_pretrain/models/card_instance_encoder.py`：旧 CardInstanceEncoder 的部署快照。
- `kaggle_card_pretrain/models/card_pretrain_heads.py`：静态辅助头的部署快照。
- `kaggle_card_pretrain/training/__init__.py`：部署训练包标记。
- `kaggle_card_pretrain/training/pretrain_card_encoder.py`：静态训练入口的部署快照。
- `kaggle_card_pretrain/training/export_card_embeddings.py`：静态导出入口的部署快照。
- `kaggle_card_pretrain/training/evaluate_card_embeddings.py`：静态 embedding 评估的部署快照。

## kaggle_cg_runtime：构建引擎运行时的 Kaggle kernel

- `kaggle_cg_runtime/kernel-metadata.json`：cg runtime 构建 kernel 的 Kaggle 配置。
- `kaggle_cg_runtime/build_cg_runtime.py`：包含并释放 cg Python wrapper 与多平台二进制的自包含构建脚本。

## kaggle_cg_runtime_dataset：cg 引擎 Dataset

- `kaggle_cg_runtime_dataset/dataset-metadata.json`：私有 cg runtime Dataset 的 Kaggle 元数据。
- `kaggle_cg_runtime_dataset/cg/__init__.py`：cg Python 包入口。
- `kaggle_cg_runtime_dataset/cg/api.py`：cabt/cg 的枚举、Observation、State、Card、Attack 和日志 API dataclass。
- `kaggle_cg_runtime_dataset/cg/game.py`：对局环境和游戏推进 wrapper。
- `kaggle_cg_runtime_dataset/cg/sim.py`：模拟器 wrapper。
- `kaggle_cg_runtime_dataset/cg/utils.py`：cg runtime 辅助函数。
- `kaggle_cg_runtime_dataset/cg/cg.dll`：Windows cg 原生运行库。
- `kaggle_cg_runtime_dataset/cg/libcg.so`：Linux x86_64 cg 原生运行库。
- `kaggle_cg_runtime_dataset/cg/libcg-arm64.so`：Linux ARM64 cg 原生运行库。
- `kaggle_cg_runtime_dataset/cg/libcg.dylib`：macOS cg 原生运行库。

## kaggle_dynamic_code_dataset：动态代码 Kaggle Dataset

该目录是动态代码的部署镜像。Notebook 挂载它后由薄入口导入，不能在这里独立维护逻辑。

- `kaggle_dynamic_code_dataset/dataset-metadata.json`：私有动态代码 Dataset 的 Kaggle 元数据。
- `kaggle_dynamic_code_dataset/data/__init__.py`：部署数据包标记。
- `kaggle_dynamic_code_dataset/data/state_schema.py`：`data/state_schema.py` 的部署镜像。
- `kaggle_dynamic_code_dataset/data/observation_parser.py`：`data/observation_parser.py` 的部署镜像。
- `kaggle_dynamic_code_dataset/data/game_memory.py`：`data/game_memory.py` 的部署镜像。
- `kaggle_dynamic_code_dataset/data/replay_dataset.py`：`data/replay_dataset.py` 的部署镜像。
- `kaggle_dynamic_code_dataset/data/online_replay_importer.py`：`data/online_replay_importer.py` 的部署镜像。
- `kaggle_dynamic_code_dataset/data/replay_training_features.py`：`data/replay_training_features.py` 的部署镜像。
- `kaggle_dynamic_code_dataset/models/__init__.py`：部署模型包标记。
- `kaggle_dynamic_code_dataset/models/static_card_adapter.py`：静态适配器的部署镜像。
- `kaggle_dynamic_code_dataset/models/static_detail_aggregator.py`：静态 detail 聚合器的部署镜像。
- `kaggle_dynamic_code_dataset/models/dynamic_instance_encoder.py`：动态实例编码器的部署镜像。
- `kaggle_dynamic_code_dataset/models/card_instance_fusion.py`：单卡融合器的部署镜像。
- `kaggle_dynamic_code_dataset/models/board_tokenizer.py`：Board tokenizer 的部署镜像。
- `kaggle_dynamic_code_dataset/models/board_transformer.py`：Board Transformer 的部署镜像。
- `kaggle_dynamic_code_dataset/models/dynamic_state_encoder.py`：动态状态总入口的部署镜像。
- `kaggle_dynamic_code_dataset/scripts/analyze_replay_observations.py`：replay 分析脚本的部署镜像。
- `kaggle_dynamic_code_dataset/scripts/benchmark_dynamic_state.py`：动态 benchmark 的部署镜像。
- `kaggle_dynamic_code_dataset/scripts/build_replay_decision_dataset.py`：decision dataset 构造脚本的部署镜像。
- `kaggle_dynamic_code_dataset/scripts/import_online_replay_decisions.py`：线上 replay 导入脚本的部署镜像。
- `kaggle_dynamic_code_dataset/scripts/train_dynamic_replay_features.py`：临时动态 smoke 训练脚本的部署镜像。

## kaggle_dynamic_state_tests：动态代码薄 kernel

- `kaggle_dynamic_state_tests/kernel-metadata.json`：挂载动态代码 Dataset、episode index 和两个每日 replay Dataset 的 kernel 配置。
- `kaggle_dynamic_state_tests/run_dynamic_code_dataset_entry.py`：把挂载的动态代码 Dataset 加入 import path，并调用临时 replay smoke 训练入口。

## kaggle_extract：热门牌组提取 kernel

- `kaggle_extract/kernel-metadata.json`：热门牌组提取 kernel 的 Kaggle 配置。
- `kaggle_extract/extract_popular_decks.py`：读取公开 episode/replay、提取牌组、统计使用率并写出热门牌组报告。

## kaggle_kernel：旧训练与提交混合目录

- `kaggle_kernel/kernel-metadata.json`：旧 agent 训练 kernel 配置。
- `kaggle_kernel/train_agent.py`：旧规则特征、候选特征、轻量 PPO、牌组赛程和 submission 生成的一体化脚本。
- `kaggle_kernel/submit_agent.py`：旧 submission 打包入口。
- `kaggle_kernel/NOTEBOOK_TRAINING_STRUCTURE.md`：旧训练 Notebook 拆分方案说明。
- `kaggle_kernel/baseline_deck.csv`：旧提交选择的一副 20 卡牌组。
- `kaggle_kernel/baseline_decks.json`：旧 kernel 使用的基线牌组集合副本。

## kaggle_training：旧 agent 训练 kernel

- `kaggle_training/kernel-metadata.json`：旧 agent 训练 kernel 配置。
- `kaggle_training/train_agent.py`：支持基线与热门测试牌组的旧规则特征/PPO 训练脚本。
- `kaggle_training/NOTEBOOK_TRAINING_STRUCTURE.md`：旧训练 Notebook 结构说明，与 `kaggle_kernel` 中版本重复。
- `kaggle_training/baseline_deck.csv`：旧训练选择的单副牌组。
- `kaggle_training/baseline_decks.json`：旧训练使用的基线牌组集合副本。

## kaggle_submission：旧 agent 提交 kernel

- `kaggle_submission/kernel-metadata.json`：依赖旧训练 kernel 输出的提交 kernel 配置。
- `kaggle_submission/submit_agent.py`：查找旧训练产物并构建 `submission.tar.gz`。
- `kaggle_submission/train_agent.py`：旧训练脚本副本，用作提交构建兼容代码。
- `kaggle_submission/baseline_deck.csv`：旧提交使用的单副牌组。
- `kaggle_submission/baseline_decks.json`：旧提交使用的基线牌组集合副本。

## outputs/card_pretrain：成功静态训练运行

### 正式静态输出

- `outputs/card_pretrain/artifacts/card_embeddings.pt`：1267 张卡的 128 维 summary tensor，PyTorch 格式。
- `outputs/card_pretrain/artifacts/card_embeddings.npy`：同一 summary table 的 NumPy 格式。
- `outputs/card_pretrain/artifacts/card_detail_tokens.pt`：每张卡 padding 后的 128 维 detail token table。
- `outputs/card_pretrain/artifacts/card_detail_masks.pt`：detail 有效位置 mask。
- `outputs/card_pretrain/artifacts/card_detail_type_ids.pt`：padding/attack/ability/special-effect 类型 ID。
- `outputs/card_pretrain/artifacts/card_detail_metadata.json`：detail index、类型和当前名称级对齐信息。
- `outputs/card_pretrain/artifacts/card_id_to_index.json`：Card ID 到导出 tensor 行号的映射。
- `outputs/card_pretrain/artifacts/card_embedding_metadata.json`：embedding 维度、数据版本、词表、normalization 和文本编码器信息。

### 静态预处理数据

- `outputs/card_pretrain/artifacts/card_data/card_feature_schema.json`：本次训练使用的正式静态 schema、词表和 normalization。
- `outputs/card_pretrain/artifacts/card_data/card_id_to_index.json`：预处理阶段的 Card ID 映射。
- `outputs/card_pretrain/artifacts/card_data/card_preprocess_summary.json`：预处理数据规模与来源摘要。
- `outputs/card_pretrain/artifacts/card_data/card_records.json`：本次训练使用的聚合 CardRecord 完整数据。

### Embedding 分析

- `outputs/card_pretrain/artifacts/card_embedding_analysis/metrics.json`：邻居 purity 和数值相关性等分析指标。
- `outputs/card_pretrain/artifacts/card_embedding_analysis/nearest_neighbors.json`：每张卡的最近邻列表。
- `outputs/card_pretrain/artifacts/card_embedding_analysis/pca_2d.npy`：二维 PCA 坐标。
- `outputs/card_pretrain/artifacts/card_embedding_analysis/report.md`：代表卡牌邻居与指标解释报告。

### Checkpoint 与日志

- `outputs/card_pretrain/checkpoints/card_encoder_best.pt`：早停选择的最佳静态 encoder checkpoint。
- `outputs/card_pretrain/checkpoints/card_encoder_last.pt`：最后一个 epoch 的静态 encoder checkpoint。
- `outputs/card_pretrain/logs/card_pretrain_metrics.jsonl`：逐次训练/验证指标。
- `outputs/card_pretrain/ptcg-card-pretrain.log`：Kaggle 静态训练完整标准输出日志。

### 成功运行的源码快照

- `outputs/card_pretrain/card_pretrain_src/configs/card_pretrain.yaml`：该成功运行实际使用的配置快照。
- `outputs/card_pretrain/card_pretrain_src/data/__init__.py`：运行快照数据包标记。
- `outputs/card_pretrain/card_pretrain_src/data/card_preprocessing.py`：运行时静态预处理源码快照。
- `outputs/card_pretrain/card_pretrain_src/data/card_dataset.py`：运行时静态 dataset 源码快照。
- `outputs/card_pretrain/card_pretrain_src/models/__init__.py`：运行快照模型包标记。
- `outputs/card_pretrain/card_pretrain_src/models/card_encoder.py`：运行时 CardEncoder 源码快照。
- `outputs/card_pretrain/card_pretrain_src/models/card_instance_encoder.py`：运行时旧实例 encoder 源码快照。
- `outputs/card_pretrain/card_pretrain_src/models/card_pretrain_heads.py`：运行时静态辅助头源码快照。
- `outputs/card_pretrain/card_pretrain_src/training/__init__.py`：运行快照训练包标记。
- `outputs/card_pretrain/card_pretrain_src/training/pretrain_card_encoder.py`：运行时训练入口快照。
- `outputs/card_pretrain/card_pretrain_src/training/export_card_embeddings.py`：运行时导出入口快照。
- `outputs/card_pretrain/card_pretrain_src/training/evaluate_card_embeddings.py`：运行时评估入口快照。
- `outputs/card_pretrain/card_pretrain_src/artifacts/card_data/card_feature_schema.json`：运行源包内的预处理 schema 副本。
- `outputs/card_pretrain/card_pretrain_src/artifacts/card_data/card_id_to_index.json`：运行源包内的 Card ID 映射副本。
- `outputs/card_pretrain/card_pretrain_src/artifacts/card_data/card_preprocess_summary.json`：运行源包内的预处理摘要副本。
- `outputs/card_pretrain/card_pretrain_src/artifacts/card_data/card_records.json`：运行源包内的 CardRecord 副本。

## outputs/card_pretrain_error：失败静态运行快照

- `outputs/card_pretrain_error/ptcg-card-pretrain.log`：失败运行日志，用于定位当时错误，不是有效训练结果。
- `outputs/card_pretrain_error/card_pretrain_src/configs/card_pretrain.yaml`：失败运行的配置快照。
- `outputs/card_pretrain_error/card_pretrain_src/data/__init__.py`：失败运行数据包标记。
- `outputs/card_pretrain_error/card_pretrain_src/data/card_preprocessing.py`：失败运行的预处理源码快照。
- `outputs/card_pretrain_error/card_pretrain_src/data/card_dataset.py`：失败运行的 dataset 源码快照。
- `outputs/card_pretrain_error/card_pretrain_src/models/__init__.py`：失败运行模型包标记。
- `outputs/card_pretrain_error/card_pretrain_src/models/card_encoder.py`：失败运行 CardEncoder 快照。
- `outputs/card_pretrain_error/card_pretrain_src/models/card_instance_encoder.py`：失败运行旧实例 encoder 快照。
- `outputs/card_pretrain_error/card_pretrain_src/models/card_pretrain_heads.py`：失败运行静态辅助头快照。
- `outputs/card_pretrain_error/card_pretrain_src/training/__init__.py`：失败运行训练包标记。
- `outputs/card_pretrain_error/card_pretrain_src/training/pretrain_card_encoder.py`：失败运行训练入口快照。
- `outputs/card_pretrain_error/card_pretrain_src/training/export_card_embeddings.py`：失败运行导出入口快照。
- `outputs/card_pretrain_error/card_pretrain_src/training/evaluate_card_embeddings.py`：失败运行评估入口快照。
- `outputs/card_pretrain_error/artifacts/card_data/card_feature_schema.json`：失败运行生成的 schema。
- `outputs/card_pretrain_error/artifacts/card_data/card_id_to_index.json`：失败运行生成的 Card ID 映射。
- `outputs/card_pretrain_error/artifacts/card_data/card_preprocess_summary.json`：失败运行生成的预处理摘要。
- `outputs/card_pretrain_error/artifacts/card_data/card_records.json`：失败运行生成的 CardRecord 数据。

## outputs/cg：解出的 cg runtime

- `outputs/cg/__init__.py`：运行时 Python 包入口。
- `outputs/cg/api.py`：当前本地审计使用的 cg API 定义副本。
- `outputs/cg/game.py`：本地 cg game wrapper。
- `outputs/cg/sim.py`：本地 cg simulator wrapper。
- `outputs/cg/utils.py`：本地 cg 辅助函数。
- `outputs/cg/cg.dll`：Windows 原生库副本。
- `outputs/cg/libcg.so`：Linux x86_64 原生库副本。
- `outputs/cg/libcg-arm64.so`：Linux ARM64 原生库副本。
- `outputs/cg/libcg.dylib`：macOS 原生库副本。

## outputs/dynamic_state_tests：旧动态 smoke 输出

- `outputs/dynamic_state_tests/ptcg-dynamic-state-tests.log`：旧动态状态 smoke kernel 日志，不是正式训练指标。
- `outputs/dynamic_state_tests/data/__init__.py`：smoke 运行的数据包快照标记。
- `outputs/dynamic_state_tests/data/state_schema.py`：smoke 运行的 schema 快照。
- `outputs/dynamic_state_tests/data/observation_parser.py`：smoke 运行的 parser 快照。
- `outputs/dynamic_state_tests/data/game_memory.py`：smoke 运行的 memory 快照。
- `outputs/dynamic_state_tests/models/__init__.py`：smoke 运行的模型包快照标记。
- `outputs/dynamic_state_tests/models/static_card_adapter.py`：smoke 运行的静态适配器快照。
- `outputs/dynamic_state_tests/models/static_detail_aggregator.py`：smoke 运行的 detail 聚合器快照。
- `outputs/dynamic_state_tests/models/dynamic_instance_encoder.py`：smoke 运行的动态实例编码器快照。
- `outputs/dynamic_state_tests/models/card_instance_fusion.py`：smoke 运行的融合器快照。
- `outputs/dynamic_state_tests/models/board_tokenizer.py`：smoke 运行的 tokenizer 快照。
- `outputs/dynamic_state_tests/models/board_transformer.py`：smoke 运行的 Board Transformer 快照。
- `outputs/dynamic_state_tests/models/dynamic_state_encoder.py`：smoke 运行的总入口快照。

## outputs：旧 agent 训练与提交产物

- `outputs/deck.csv`：旧 agent 最终使用的提交牌组。
- `outputs/main.py`：旧 submission 中的可执行 agent 入口。
- `outputs/model.json`：旧轻量策略模型参数。
- `outputs/policy_state.pt`：旧 PyTorch policy state。
- `outputs/ppo_weights.json`：旧 PPO 权重导出。
- `outputs/weights.json`：旧 agent 兼容权重文件。
- `outputs/training_summary.json`：旧 agent 训练摘要。
- `outputs/ptcg-agent-training.log`：旧 agent Kaggle 训练日志。
- `outputs/submission.tar.gz`：旧 agent 提交包。

## 当前主线与非主线

当前静态特征主线的唯一正式输出是 `outputs/card_pretrain/artifacts/`。动态时序代码仍处于初版接口阶段，没有正式动态 checkpoint。`kaggle_dynamic_code_dataset/` 是部署镜像；`kaggle_kernel/`、`kaggle_training/`、`kaggle_submission/` 和根 `outputs/` 中的 agent 文件属于旧策略训练链路。
