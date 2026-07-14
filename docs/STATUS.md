# 项目状态

更新时间：2026-07-12

## 一句话进度

静态 CardEncoder 已完成并产出可靠的 summary/detail artifacts；动态单卡的结构化 schema、四头 Cross-Attention、辅助任务和唯一 Kaggle 训练入口已进入云端验证阶段。只有多日真实 replay 训练产物下载并通过验收后，动态阶段才会标记完成。

## 已确认并保留的成果

### 静态卡牌表示

- CSV 中同一 Card ID 的多行已经聚合为一个 `CardRecord`。
- 多个攻击、特性和特殊效果保留为独立 detail，费用、伤害和文本绑定不丢失。
- 当前卡池共 1267 张卡。
- 成功产物包含：
  - 128 维 `card_summary`
  - 128 维 `detail_tokens`
  - `detail_mask`
  - `detail_type_ids`
  - Card ID 映射与 metadata
- Basic Energy 的类型信息通过显式静态字段保留。
- 静态辅助训练、best/last checkpoint 和 embedding 分析已经完成。

静态产物继续作为后续模型的固定输入。本阶段保持其 schema 与 checkpoint 不变。

### Replay 与 observation 基础设施

- `data/observation_parser.py` 可以解析当前局面、可见卡牌、日志和合法选项。
- `data/replay_dataset.py` 按每局真实 `steps` 读取变长 replay。
- 训练样本来自 `observation.select` 非空的决策点。
- `data/online_replay_importer.py` 支持每日 replay Dataset、日期保留集和有限样本导入。
- 已验证的两个在线日期样例均无 parser error。
- 已确认单样本可出现约 22 个卡牌实例、6 个合法选项，当前 token 规模约 43。

## 当前原型的真实边界

### 动态单卡表示

当前已在正式源码实现：

- 类别、数值、计数、布尔和 validity mask 分离的 `CardDynamicBatch`。
- 保留 serial 的 HP、区域、12 类能量、异常状态、Tool、进化与本回合出场解析。
- 分组编码并输出 64 维表示的 `DynamicInstanceEncoder`。
- 由动态 query 查询独立 detail token 的四头 `CardInstanceFusion`。
- 攻击可支付、分类型剩余能量、HP/伤害、zone/role 四项辅助任务。
- 保守特殊能量 resolver、detail-level 标签、真实 replay collator 与扩展审计。
- 唯一 `training.train_dynamic_card_fusion` 入口和 `kaggle_dynamic_training/` Kernel。

仍待真实 Kaggle 结果证明：

- 多日期 replay 字段覆盖率和 unresolved 比例。
- 真实 batch forward/backward、四任务梯度和 tiny-batch overfit。
- 时间保留集指标、能量反事实诊断和 checkpoint 回载一致性。
- best/last checkpoint 与 CPU benchmark 的实际产物。

### 时序与全局状态

当前已有：

- `GameMemoryState`、serial 记录、Recent Events 和两侧 Ledger 的最小接口。
- `[STATE]`、`[DECISION]`、`[MATCH]`、Ledger/Event 投影和两层 Board Transformer 原型。

当前缺口：

- Ledger 目前主要是 serial 统计汇总，还不是按 `(owner, card_id)` 维护的长期认知表。
- memory 缺少正式的 reset、clone、序列化、幂等更新和 shuffle 知识降级。
- Board token 顺序与正式架构尚未完全一致。
- `state_embedding` 目前来自通用池化，尚未固定为 `[STATE]` 的上下文化输出。
- 尚未使用真实静态 artifacts 和在线 replay batch 形成正式动态 checkpoint。

### 策略学习

当前主线尚未实现：

- ActionEncoder
- 合法动作逐项评分
- 行为克隆
- Value Head
- Oracle teacher / student distillation
- Self-play PPO

旧规则特征与共享 PPO 代码已经从当前仓库移除。失败原因和仍有价值的结论保存在 [实验结论记录](EXPERIMENT_HISTORY.md)，完整旧实现可从 Git 历史读取。

## 接下来的完整顺序

1. 从少量多日 replay 生成真实 decision-point 数据，固定字段覆盖率、Card ID/detail 对齐率、事件长度和特殊能量 unresolved 比例。
2. 完成结构化 `CardDynamicBatch`、动态条件 detail Cross-Attention 和单卡辅助任务。
3. 完成按 Card ID 聚合的双方 Ledger、Recent Events、幂等 memory 和正式 Board token 顺序。
4. 用真实静态 artifacts 完成端到端 forward/backward、tiny-batch overfit、CPU benchmark 和融合 checkpoint。
5. 增加 ActionEncoder，对引擎提供的变长合法选项逐项编码和评分。
6. 先进行高质量 replay 行为克隆与 Value 学习，再进入 oracle 蒸馏和 self-play PPO。
7. 在固定牌组、先后手、随机种子和历史 checkpoint 对手池上评估，最后构建提交包。

## 数据划分规则

- 静态卡牌任务：按 Card ID 划分，确保同一 Card ID 只属于一个 split。
- Replay 任务：按完整 episode 和日期划分，确保同一局不会跨 split。
- 最近若干日期作为时间保留集。
- Oracle 或完整隐藏信息只作为未来 teacher 标签，正式 Agent 输入保持公开可见边界。

## 当前唯一主线

```text
静态 artifacts
→ 真实 replay 决策点
→ 动态 CardInstanceFusion
→ Ledger + Recent Events + Board Transformer
→ ActionEncoder
→ 行为克隆 / Value
→ self-play
→ submission
```

## 仓库清理结果

- 当前仓库只保留静态训练、动态状态、replay 数据和未来策略需要的正式源码。
- Kaggle 动态代码副本改为由 `scripts/sync_kaggle_dynamic_code_dataset.py` 生成。
- 旧 PPO 三套重复目录、临时动态 smoke 训练、旧 CardInstanceEncoder 和 replay notebook 已移除。
- 牌组资料统一放在 `decks/`。
