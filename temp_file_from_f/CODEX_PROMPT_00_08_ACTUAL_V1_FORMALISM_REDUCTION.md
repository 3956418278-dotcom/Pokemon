# Codex 修订指令：按真实 Decision Agent V1 路线精简 Prompt 00–08

## 0. 本文件的作用

请修改当前这组文件：

```text
00_README_EXECUTION_ORDER.md
01_policy_value_foundation.md
02_replay_corpus_and_full_training.md
03_state_history_and_belief_upgrade.md
04_inference_and_hybrid_agent.md
05_simulator_evaluation_and_shallow_search.md
06_conservative_selfplay_finetune.md
07_deck_specialization_and_submission.md
08_final_champion_selection.md
```

本次修改只删除形式主义工作和重复验证，使现有 Decision Agent V1 路线更快落地。

不得修改当前真实模型方向，不得把旧静态卡牌主线重新接入本项目。

当前正在执行的 Prompt 02 不停止、不重启、不删除缓存和 checkpoint。若 Prompt 02 已经进入正式训练，继续让本轮训练自然完成。

---

# 1. 必须保持不变的真实模型路线

Decision Agent V1 是与旧静态卡牌预训练、单卡融合路线并行的新实现。

当前模型输入层固定为：

\[
c_i=
E_{\mathrm{card\_id}}(card\_index_i)
+E_{\mathrm{owner}}
+E_{\mathrm{zone}}
+E_{\mathrm{position}}
+W_{\mathrm{dyn}}d_i
\]

其中 `d_i` 是当前可见卡牌实例的动态字段。

随后：

```text
[GLOBAL]
[HISTORY_SUMMARY]
[VISIBLE CARD INSTANCE TOKENS]
        ↓
两层 Shared Board Transformer
        ↓
board_embedding
contextual_card_tokens
```

Policy 和 Value 共享 Board Encoder：

```text
Shared Board Encoder
├── Policy Head：对当前合法 options 评分
└── Value Head：预测 LOSS / DRAW / WIN
```

OptionEncoder 读取当前合法 option 的：

```text
option type
select type
select context
relative player
area
position
card_id
serial reference
energy/damage/count fields
field masks
```

若 option 引用当前可见 serial，则关联对应 `contextual_card_token`。

必须保持：

```text
Card ID embedding 随机初始化
Card ID embedding 由 Policy/Value 损失端到端训练
Card ID 与 serial 分离
Policy 与 Value 共用 Board Encoder
Value 只读取当前局面，不读取 selected action 或后继状态
MultiSelectDecoder 保持 autoregressive without replacement
equivalence-aware loss 保持
Actor 和 Value 只读取 agent 当时可见信息
```

本路线当前明确不使用：

```text
card_summary
detail_tokens
CardEncoder checkpoint
静态 embedding
静态 summary/detail 融合
动态条件 Cross-Attention
卡牌文字编码
旧 CardInstanceFusion
旧 DynamicStateEncoder
独立单卡预训练
独立单卡 checkpoint
```

不要在任何 Prompt 中新增或暗示上述内容。

---

# 2. 统一精简原则

## 保留

下列内容直接决定训练正确性或 Agent 强度，必须保留：

```text
DecisionSampleV1
action semantics contract
legal option target
episode/date split
visibility boundary
Card ID / serial 分离
Board Encoder
Option Encoder
Policy Head
Value Head
multi-select 解码
equivalence-aware loss
Replay 正式训练
Public Ledger
Recent Events
Self Deck Context
Opponent Belief
simulator 推理
Value-guided shallow search
保守 self-play / PPO
牌组专项训练
submission 独立运行验证
```

## 删除或压缩

下列内容统一精简：

```text
重复扫描已经审计完成的 7500 场 Replay
为每个阶段生成大型 Markdown 审计报告
为同一公共代码对每个 checkpoint 重复测试
完整指标分桶和装饰性图表
A/B/C/D 等多次完整重训式消融
通用 tournament/registry/release 管理平台
没有实际需要时预先做量化与压缩比较
多份 README / STATUS / HANDOFF 同步更新
只为记录 hash 而建立复杂候选注册系统
```

验证应直接进入现有测试、缓存统计、训练日志和最终运行结果。

---

# 3. Prompt 00：执行顺序

## 保留

保留原执行顺序：

```text
01 Policy–Value foundation
02 7500 场 Replay 完整训练
03 History / Self Deck / Opponent Belief
04 Simulator inference / hybrid agent
05 Fixed evaluation / shallow search
06 Conservative self-play PPO
07 Deck specialization / submission
08 Final candidate selection
```

这些阶段不是要删除的形式工作。

## 修改

把：

```text
Codex 必须先读取上一阶段报告
```

改成：

```text
Codex 必须先读取上一阶段真实源码、配置、checkpoint、缓存 manifest 和机器可读指标。
存在上一阶段 Markdown 报告时可以参考，但报告缺失不得阻塞下一阶段。
```

删除：

- 每阶段必须补齐大型人工报告后才能继续；
- 为了文档完整反向重跑已经完成的训练；
- 重复更新多个项目说明文件。

阶段门禁只看可执行产物：

```text
01：真实 Replay smoke/tiny-overfit 通过
02：正式 checkpoint 可加载
03：V2 增强模型可训练并完成最小对比
04：真实 simulator 对局合法稳定
05：搜索在固定评估中有收益且满足时间预算
06：PPO 不低于 BC champion
07：submission 可独立运行
08：最终候选胜率、稳定性和资源合规
```

---

# 4. Prompt 01：Policy–Value Foundation

## 模型保持原样

必须保留：

```text
Card ID embedding + owner/zone/position embedding + dynamic projection
Shared Board Encoder
Option Encoder
Policy Head
Value Head
MultiSelectDecoder
DecisionSampleV1
Replay/Observation/Simulator adapters
deterministic legal baseline
smoke
tiny-overfit
```

不得接入旧静态 summary/detail 或 Cross-Attention。

## 删除

删除以下独立产物：

```text
decision_agent_v1/scripts/audit_interfaces.py
outputs/decision_agent_v1/audits/interface_audit.json
outputs/decision_agent_v1/audits/interface_audit.md
```

删除再次统计：

```text
selection mode 全量分布
select.type/context 全量分布
option 数量全量分布
serial 引用总体匹配率
equivalence group 总体覆盖率
UNKNOWN 组合再次扫描
Replay 重复规律再次分析
```

这些已经由现有 7500 场 action contract 审计确认。

## 保留的硬验证

并入现有测试和 smoke：

```text
DecisionSampleV1 可由真实 Replay 生成
Actor 不读取 visualize.current
target 属于当前 legal options
original option index 可恢复
card_id 与 serial 不混用
无 select 的 observation 不生成样本
Policy/Value/Board/Card ID embedding 均有梯度
32–128 个真实样本可 tiny-overfit
deterministic baseline 能完成少量合法对局
```

Prompt 01 已完成时，不返工，不为缺少审计报告重新运行。

---

# 5. Prompt 02：7500 场 Replay 正式训练

## 当前运行处理

若 Prompt 02 full training 正在运行：

```text
继续当前训练
不停止
不重新构建已完成缓存
不改当前模型结构
不删除 checkpoint
不从 epoch 0 重启
```

## 保留

```text
全部约 7500 场 Replay
按完整 episode/date split
正式可复用缓存
Policy–Value 联合训练
episode-balanced Value loss
equivalence-aware Policy loss
multi-select sequence loss
resume
best_policy / best_value / best_joint / last checkpoint
validation
CPU checkpoint load
```

## 删除独立语料审计

删除：

```text
outputs/decision_agent_v1/audits/replay_corpus_audit.json
outputs/decision_agent_v1/audits/replay_corpus_audit.md
```

必要信息统一进入：

```text
cache/.../statistics.json
cache/.../manifest.json
build.log
failed_episodes.jsonl  # 只有失败时生成
training metrics
```

不再重复统计已经确认的：

```text
action semantics 全量分布
selection mode 全量分布
context 全量分布
equivalence 覆盖率
serial 总体匹配率
重复 observation/decision 规律
```

## 缩减评估

保留：

```text
policy loss
equivalence-aware accuracy
single-select accuracy
sequence/set exact match
value loss
WIN/DRAW/LOSS accuracy
macro F1
random legal baseline
untrained model baseline
```

删除或改为可选：

```text
Brier score
ECE
按所有 agent/牌组/turn/context/option count 的完整分桶
context 内最高频 baseline
deterministic baseline 的离线逐样本大比较
大量图表与分析报告
```

这些不得阻塞 checkpoint 产出。

Prompt 02 完成后只需要确认：

```text
训练结束或可恢复
无 NaN/Inf
invalid target 为 0
split 交集为 0
best checkpoint 可加载
核心 validation 指标已保存
```

---

# 6. Prompt 03：History、Self Deck 与 Opponent Belief

Prompt 03 是真实模型增强阶段，不能删除。

## 必须保留

```text
[SELF_DECK_CONTEXT]
[OPPONENT_BELIEF]
[LEDGER_SUMMARY]
[RECENT_EVENT_1 ... K]
[VISIBLE CARD INSTANCE TOKENS]
```

保留：

- Public Ledger；
- Recent Events；
- Self Deck Context；
- 基于训练 split 牌组模板的 Opponent Belief；
- opponent archetype prediction；
- next public card/group prediction；
- V2 schema 与 V2 cache；
- `best_joint_v2.pt`。

## 删除重复形式工作

删除：

```text
belief_audit.md
state_upgrade_ablation.md
```

机器可读结果保留一个：

```text
state_upgrade_results.json
```

不再执行四个完整训练：

```text
A：基础
B：A + Ledger/Recent Events
C：B + Self Deck
D：C + Belief
```

改为两次直接比较：

```text
V1：Prompt 02 best_joint
V2：Ledger + Recent Events + Self Deck Context + Opponent Belief
```

V2 先运行：

```text
真实 batch forward/backward
tiny-overfit
短程 validation run
```

确认无泄漏、loss 正常、辅助头可学习后，再执行一次正式 V2 增量训练。

不为每个子模块单独完整重训。

若 V2 整体不改善，保留 V1 为默认，但实现和结果不得伪造。

---

# 7. Prompt 04：Simulator Inference / Hybrid Agent

## 必须保留

```text
统一 inference adapter
checkpoint 只加载一次
训练与推理使用同一 observation adapter
GameMemory 跨 decision 保留、跨 episode reset
legal option scoring
MultiSelectDecoder
original option index 输出
UNKNOWN → deterministic legal fallback
异常/NaN/shape mismatch → deterministic legal fallback
完整 simulator 对局
invalid/crash/timeout/latency 统计
```

## 精简低置信度回退

保留置信度字段的计算接口，但不在本阶段建立复杂调参工程。

第一版只允许：

```text
模型正常且语义已知 → 模型
UNKNOWN → deterministic fallback
模型异常 → deterministic fallback
```

低置信度回退保持配置关闭，除非 validation 数据已经明确证明一个简单全局阈值有收益。

删除：

```text
按 context 单独 calibration
多组 top1/margin/entropy/value threshold 联合搜索
safe_heuristics.py 中大量策略规则
每条 heuristic 单独消融
model-only/fallback-only/hybrid 大型模式矩阵
inference_readiness_report.md
```

延迟只记录：

```text
mean action latency
P95 action latency
peak RAM
checkpoint size
```

不强制拆分 parse/encode/score/decode 四套报告。

先运行 20–30 局集成测试；100 局稳定性放到 Prompt 07/08 最终包验证。

---

# 8. Prompt 05：固定评估与浅层搜索

Prompt 05 的 Value-guided search 属于真实策略增强，必须保留。

## 精简评估代码

不建立五个通用框架文件：

```text
opponent_registry.py
deck_registry.py
tournament.py
statistics.py
复杂 registry
```

保留一个集中入口即可：

```text
evaluation/match_runner.py
evaluation/evaluate_candidates.py
```

评估集合固定为：

```text
deterministic legal baseline
仓库现有一个可运行 baseline
当前 Policy/Hybrid
```

使用：

```text
当前计划牌组
一个不同 archetype
一个镜像 matchup
paired seeds
交换先后手
```

输出一个机器可读文件：

```text
evaluation_results.json
```

删除：

```text
evaluation_report.md
matchup_matrix.csv
Wilson interval 专用模块
大型牌组矩阵
```

## 状态复制检查

保留一个最小自动测试：

```text
状态可复制
副本与原状态隔离
随机状态可恢复
pending selection 可继续
rollout 时间可接受
```

删除：

```text
search_capability_audit.md
```

## 搜索保持原设计

继续实现：

```text
Policy top-k 候选
clone 当前 simulator state
应用候选 option
由 Policy 完成同一 pending effect
到 effect/turn/depth/terminal 边界
终局 reward 或 Value(s')
```

评分保持：

\[
S(a)=\alpha\log\pi(a\mid s)+(1-\alpha)\bar V(s'_a)
\]

第一版只比较：

```text
Policy/Hybrid without search
Policy/Hybrid with one组默认 search config
```

删除：

```text
top-2、top-3、多 depth、多 rollout 的全网格消融
search_ablation.md
champion.json 指针管理
```

搜索无收益或超时则关闭，不影响后续 PPO 和提交。

---

# 9. Prompt 06：Conservative Self-play PPO

Prompt 06 是突破 Replay 模仿上限的训练阶段，不能删除。

## 保留

```text
从 BC/V2 champion 初始化
selection decision 作为 PPO timestep
multi-select sequence log probability
GAE
PPO clipped objective
clipped Value loss
entropy
BC/KL anchor
冻结历史 opponent pool
终局 WIN/DRAW/LOSS reward
定期固定评估
回滚
ppo_last / ppo_best / history
```

## 精简

对手池第一版只保留：

```text
BC champion
V2 champion（存在时）
deterministic baseline
最近两个冻结 PPO snapshot
```

删除过大的牌组/对手排列。

冻结计划缩成两阶段：

```text
Phase 1：
冻结 Card ID embedding 与 Board Encoder 前层
训练 Policy Head、Value Head、Board Encoder 最后一层

Phase 2：
仅在固定评估有改善时，以更小学习率解冻整个 Board Encoder
```

删除 Stage A/B/C 的重复报告和大量超参数试验。

先在现有环境做最小 self-play run，确认轨迹、logprob、GAE 和 update 正确后，再创建/提交 Kaggle self-play kernel。不要先搭完整云端平台。

每轮只保存机器可读：

```text
ppo_metrics.jsonl
evaluation_snapshots.json
```

不生成大型训练 Markdown 报告。

---

# 10. Prompt 07：Deck Specialization 与 Submission

## 保留

```text
Self Deck Context
候选牌组
deck-specific Policy–Value fine-tune
可选少量 self-play
最终 deck + checkpoint 选择
main.py
deck.csv
model/config/vocab/action contract/belief assets
submission.tar.gz
空目录导入
/kaggle_simulations/agent 路径模拟
fallback 测试
```

## 精简牌组专项

候选牌组最多 2–3 套。

先根据：

```text
Replay 支持量
历史使用率
基线对局结果
```

筛选后，只对最有希望的 1–2 套执行专项训练。

不为每套牌从头训练，也不对所有牌组执行完整 round-robin。

## 精简压缩

默认保存稳定的 state_dict。

只有满足以下任一条件时才测试压缩：

```text
submission 超过大小限制
首次加载太慢
CPU 推理超时
RAM 超限
```

不得预先同时比较：

```text
FP16
BF16
dynamic quantization
TorchScript
```

## 精简提交验证

删除：

```text
submission_audit.md
```

保留：

```text
submission_validation.json
```

Prompt 07 运行 30–50 局包内 Agent 测试。

最终 100 局稳定性只对 Prompt 08 选出的最终包运行一次。

---

# 11. Prompt 08：最终候选选择

Prompt 08 不再建立发布管理系统，只完成真实候选选择和最终回归。

## 候选只包含实际存在版本

例如：

```text
V1 BC
V2 history/belief
PPO best
最终 deck-specific checkpoint
search on/off
```

不存在的候选不创建占位项。

## 删除

```text
candidates.json 注册表
每个文件的复杂 SHA256 依赖链
RELEASE_MANIFEST.json
final_champion_report.md
final_matchup_matrix.csv
known_limitations.md
重复更新 README / PROJECT_FILE_MANIFEST / README_KAGGLE_RUN
对每个 checkpoint 重复 hidden-information audit
对每个 checkpoint 重复 action contract 全覆盖测试
对每个 checkpoint 重复 multi-select contract 测试
```

公共推理代码的：

```text
visibility
action contract
MultiSelectDecoder
episode reset
fallback
```

只在公共代码变化时测试一次。

每个候选只比较：

```text
win/loss/draw
invalid
crash
timeout
mean/P95 latency
fallback rate
checkpoint cold load
```

## 最终回归

只对最终 `submission.tar.gz` 执行一次：

```text
顶层 main.py
顶层 deck.csv
空目录导入
模拟 /kaggle_simulations/agent/
无网络和训练数据依赖
连续至少 100 局
invalid/crash/timeout 为 0
包大小、RAM、CPU 合规
```

最终保留：

```text
submission.tar.gz
final_results.json
```

---

# 12. Codex 最终输出要求

修改完 Prompt 00–08 后，只汇报：

```text
1. 修改了哪些 Prompt 文件。
2. 每个 Prompt 删除了哪些形式工作。
3. 每个 Prompt 保留了哪些真实模型、训练或推理任务。
4. 是否修改了正式模型源码。
5. 是否停止或重启了当前 Prompt 02。
```

第 4 项预期为：

```text
没有。本次只修改 Prompt 文档。
```

第 5 项预期为：

```text
没有。当前 Prompt 02 保持原运行状态。
```

不要启动新的全量扫描，不要重新训练，不要修改当前 checkpoint，不要接入静态 summary/detail，也不要实现动态单卡 Cross-Attention。
