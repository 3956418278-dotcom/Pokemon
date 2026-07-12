# Kaggle 工作流

本项目在本地维护代码，在 Kaggle 完成公开 replay 提取、静态训练、动态 smoke 与最终提交构建。

## 1. 当前有效的 Kaggle 组件

| 目录 | 用途 | 状态 |
|---|---|---|
| `kaggle_extract/` | 提取公开 replay 与热门牌组 | 有效 |
| `kaggle_card_pretrain/` | 训练与导出静态 CardEncoder | 已成功运行 |
| `kaggle_cg_runtime_dataset/` | 向 kernel 提供 `cg` runtime | 有效，本地内容被 ignore |
| `kaggle_dynamic_code_dataset/` | 发布动态数据与模型代码 | 有效部署镜像 |
| `kaggle_dynamic_state_tests/` | 挂载 replay 进行动态 smoke | 原型验证入口 |
| `kaggle_kernel/` | 旧一体化训练/提交 | 历史 baseline |
| `kaggle_training/` | 旧规则特征与轻量 PPO | 历史 baseline |
| `kaggle_submission/` | 旧 agent 提交构建 | 历史 baseline |

当前模型主线从静态 artifacts 进入动态状态训练。旧 PPO 目录继续用于对照，不作为新模型入口。

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

可用环境变量：

```text
PTCG_RECENT_SUBMISSIONS_TO_USE=8
PTCG_MAX_REPLAYS_TO_DOWNLOAD=200
PTCG_MIN_POPULAR_DECK_GAMES=2
PTCG_MAX_POPULAR_DECKS=24
PTCG_SUBMISSION_IDS="54449821 54449681"
```

## 3. 静态 CardEncoder

### 发布 `cg` runtime

```bash
kaggle datasets create -p kaggle_cg_runtime_dataset --dir-mode zip
```

已有 Dataset 更新：

```bash
kaggle datasets version -p kaggle_cg_runtime_dataset --dir-mode zip -m "update cg runtime"
```

### 运行静态训练

```bash
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

后续模块按 Card ID 读取 summary/detail，不重新解析静态 CSV 行数作为卡牌副本数。

## 4. 动态 replay smoke

动态 kernel 使用三类输入：

1. `ptcg-dynamic-code-dataset`
2. Kaggle episode index
3. 少量按日期挂载的 replay Dataset

更新代码 Dataset：

```bash
kaggle datasets version \
  -p kaggle_dynamic_code_dataset \
  --dir-mode zip \
  -m "update dynamic state code"
```

运行：

```bash
kaggle kernels push -p kaggle_dynamic_state_tests
kaggle kernels status f7e6n5g4/ptcg-dynamic-state-tests
kaggle kernels output f7e6n5g4/ptcg-dynamic-state-tests -p outputs/dynamic_state_tests -o
```

`kaggle_dynamic_state_tests/run_dynamic_code_dataset_entry.py` 当前只读取一个早期训练日期并保留最近日期作为时间验证。首次检查保持有限数据规模，确认字段分布后再增加日期和样本数。

## 5. 本地辅助命令

重新生成 baseline decks：

```bash
python scripts/make_baseline_decks.py
```

状态 benchmark：

```bash
python scripts/benchmark_dynamic_state.py --iterations 100
```

本地 replay decision dataset：

```bash
python scripts/build_replay_decision_dataset.py \
  --replay-path data_from_submission/replays \
  --output-dir data_from_submission/replay_dataset
```

## 6. 旧 Agent 与提交

旧链路仍可生成 baseline submission：

```bash
kaggle kernels push -p kaggle_training
kaggle kernels output f7e6n5g4/ptcg-agent-training -p outputs -o

kaggle kernels push -p kaggle_submission
kaggle kernels output f7e6n5g4/ptcg-agent-submission -p outputs/submission -o
```

该链路输出 `model.json`、`ppo_weights.json`、`policy_state.pt` 和 `submission.tar.gz`，与当前静态/动态主线 checkpoint 不兼容。

未来正式提交入口将在 ActionEncoder、行为克隆和 self-play 完成后重新建立。

## 7. 比赛提交要求

提交包顶层必须包含：

```text
main.py
deck.csv
模型权重与运行所需文件
```

构建后提交：

```bash
kaggle competitions submit \
  -c pokemon-tcg-ai-battle \
  -f outputs/submission/submission.tar.gz \
  -m "message"
```

运行路径为 `/kaggle_simulations/agent/`，提交包大小上限为 197.7 MiB。正式打包前需要验证 self-play validation、相对导入路径、CPU 延迟和内存占用。
