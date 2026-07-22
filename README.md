# Pokémon TCG fixed-deck self-play

`decks` 分支只维护竞赛导向的固定套牌路线。它以一套已经成功运行的
Raging Bolt Ogerpon 牌组为边界，围绕机械策略、非对称自对战、策略晋升与回放人工抽检推进。

本分支刻意不包含两条旧路线：

- 最初的全局卡牌特征构造、动态状态编码和 replay 模仿方案；
- 中间的 8-prompt / `decision_agent_v1` 决策代理方案。

## 固定套牌契约

目标牌组位于 `decks/baseline_decks.json`，索引为 6，名称为
`Raging Bolt Ogerpon`。加载时会校验牌组名称、来源索引、替换状态、完整的 60 张顺序哈希、
Card ID、副本序号和总副本数。

卡牌 embedding 索引由这 60 张牌的 Card ID 在本地确定性生成；`0`、`1` 分别预留给
padding 和 unknown。该路线不依赖旧的全局静态卡牌特征产物。

## 分支内容

- `competition_selfplay/`：因果事务级 PPO、固定语义概念/校准门、league、机械 agent、打包与回放播放器；
- `decks/`：目标牌组和基础牌组数据；
- `kaggle/builders/cg_runtime/`：生成模拟器运行时的 Kaggle builder；
- `scripts/make_baseline_decks.py`：从模拟器环境生成基础牌组；
- `tests/`：固定套牌与机械 agent 的测试；
- `records/competition_selfplay/`：协作状态和当前训练门槛。

生成的 replay、checkpoint、rollout 和指标属于本地数据，不提交到 Git。

## 当前门槛

正式训练路径已替换为因果事务级 PPO，schema v3 固定为十维事务语义接口。五个 prize 坐标
分别覆盖互不重叠的 I0–I4，可精确重构 H1/H3/H6；攻击标签只认真实 type-15 结算事件。
Phase A 前 20,000 个完整对局严格使用 `+1/-1/0` 终局奖励，同时训练语义概念、校准语义势、
残差值与标量 full critic。固定 holdout 的 Brier/ECE 与 `semantic_value` 自身的座位反对称/排序
门通过后才进入 Phase B；`full_value` 指标只监控 critic，不能替语义势过门。之后在 50,000 局
内把语义势差系数升至 0.15。每批只更新 learner；frozen opponent 仅随 league 晋升，完整
target semantic 路径仅在批次更新结束后做 EMA。实现、测试和两局真实 `cg` smoke 已具备开始
Phase A 的条件，但尚未声称 20,000 局训练或 Phase B 校准已经完成。

提交 `54825132` 及其 `audit-006` 对局已被明确排除为训练材料。当前机械候选必须先通过 replay
人工抽检，详细状态见 `records/competition_selfplay/CURRENT.md`。

## 验证

```bash
python -m competition_selfplay.cli --dry-run
python -m pytest -q
```

构建或运行机械 agent 需要先生成本地模拟器目录
`kaggle/datasets/cg_runtime/cg/`：

```bash
python -m competition_selfplay.build_mechanical_submission
python -m competition_selfplay.run_mechanical_selfplay --episodes 20
```

这些命令只生成本地文件，不会上传 Kaggle。回放可通过
`competition_selfplay/replay_viewer/index.html` 本地查看。

项目采用 [MIT License](LICENSE)。
