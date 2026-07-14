# 模型与数据架构

## 1. 输入与输出

每个决策点的正式输入由四部分组成：

1. 静态卡牌 artifacts。
2. 当前 `observation.current` 中可见的局面。
3. `observation.logs` 和上一时刻 `GameMemoryState`。
4. `observation.select` 中的选择上下文与合法选项。

状态编码器输出：

```text
state_embedding:          [B, 128]
contextualized_tokens:    [B, N, 128]
board_mask:               [B, N]
entity metadata
updated_memory
```

未来策略层为每个合法选项生成 action token，并根据 `state_embedding` 与 action token 的匹配分数选择动作。

## 2. 卡牌身份

项目始终区分：

```text
card_id = 卡牌种类、规则与静态身份
serial  = 一局对战中的具体卡牌副本
```

同一 Card ID 的多行静态 CSV 表示多个攻击或技能条目。它们先聚合成一张卡，再生成多个 detail token。场上出现两张相同 Card ID 时，它们拥有不同 serial 和不同动态状态。

## 3. 静态卡牌模块边界

静态卡牌部分已交由 colleague 脚本完整实现（置于 `static_card/` 目录），包含 CSV 定位与读取、同一 Card ID 聚合、静态特征构造、预训练与静态产物导出。根目录不再维护重复的静态 pre-preprocessing、schema 或 CardEncoder。

根目录的 `StaticCardAdapter` 为唯一接入边界，它只负责读取 colleague 正式导出的产物并对外暴露以下接口：

- `card_summary`
- `detail_tokens` 或等价的细节表示
- `detail_mask`
- `detail_type_ids` 或等价的类型信息
- `known_mask`

后续动态模型（如 `CardInstanceFusion`）将通过该适配器透明接入静态特征，静态与动态在物理和架构上均完成解耦。

## 4. 动态卡牌实例

每个当前可见实例包含：

- owner、zone、field role、visibility 和 knowledge state。
- 当前 HP、最大 HP、伤害比例。
- 12 类附着能量计数。
- 附着 Energy Card、Tool 和进化链摘要。
- 异常状态。
- `appearThisTurn` 等本回合标记。
- 手牌与弃牌区按 Card ID 聚合后的 `copy_count`。

SELF 手牌按精确 Card ID 聚合，同时保留原始 serial 与索引映射。场上的 Active/Bench 按 serial 分开。对手隐藏手牌只进入数量和 UNKNOWN Ledger，不产生虚构 Card ID。

## 5. 单卡融合

推荐的正式融合只有一种：动态条件 Cross-Attention。

```text
dynamic state → DynamicInstanceEncoder → [B_cards, 64]

query = LayerNorm(card_summary + project(dynamic_repr))
key/value = detail_tokens

query + Cross-Attention + FFN
→ card_instance_token [B_cards, 128]
```

这样，同一张卡在不同能量、HP 和区域下会关注不同的攻击或效果 detail。无有效 detail 时使用零 context，并保证前向与反向无 NaN。

## 6. 当前局面与时序记忆

全局层固定为：

```text
[STATE]       当前快速变化的公开资源
[DECISION]    当前选择类型和限制
[MATCH]       先后手、绝对回合与环境版本
[SELF_LEDGER] 己方牌组组成、已知区域和累计使用
[OPP_LEDGER]  对手已公开卡、匿名隐藏数量和长期证据
[EVENT]*      最近 16 个重要公开事件
```

### Ledger

Ledger 按 `(relative_owner, card_id)` 聚合长期事实：当前已知区域数量、未解析隐藏数量、出现/使用/附着/进化/攻击/弃牌/回收/搜索计数、最近变化和知识状态。

UNKNOWN Card ID 表示匿名隐藏资源。当前区域数量由 snapshot 重建；累计原因由 logs 更新，从而避免重复计数。

### Recent Events

Recent Events 保留顺序，回答“刚才发生了什么”。主要事件包括抽牌、打出、附着、进化、弃牌、回收、洗牌、换位、攻击、HP 变化和异常状态变化。

反向日志只记录匿名事件。洗牌会将已知隐藏位置降级为模糊隐藏状态。

## 7. Board Token

正式顺序：

```text
[STATE]
[DECISION]
[MATCH]
[SELF_LEDGER]
[OPP_LEDGER]
[RECENT_EVENT] * N
[SELF_ACTIVE]?
[SELF_BENCH] * N
[OPP_ACTIVE]?
[OPP_BENCH] * N
[SELF_HAND_UNIQUE] * N
[STADIUM]?
[VISIBLE_PRIZE] * N
[VISIBLE_SELECTION_CARD] * N
[SELF_DISCARD_SUMMARY]
[OPP_DISCARD_SUMMARY]
```

Board Transformer 使用两层、四头、128 维 pre-norm Transformer。`state_embedding` 取 `[STATE]` 位置的上下文化输出。

## 8. ActionEncoder

动作层在状态编码完成后接入。每个合法选项编码：

- option type 与 select context。
- source card/serial。
- 对应 attack/ability detail。
- target card/serial。
- 数值、区域和其他 payload。

策略只对引擎提供的合法选项计算 logits。交换选项顺序时，输出 logits 应同步交换；padding 不应影响有效选项。

## 9. 隐藏信息边界

正式 Agent 只使用：

- 当前 observation。
- 当前 logs。
- 己方已知牌组总量。
- 之前公开且仍可合理追踪的信息。

对手隐藏手牌、牌库和奖赏卡身份使用 UNKNOWN、数量或由公开证据得到的概率表示。训练期 oracle 信息与正式输入物理分离。

## 10. 训练顺序

```text
静态 CardEncoder
→ CardInstanceFusion 辅助预训练
→ Board / temporal encoder 预训练
→ ActionEncoder + Behavior Cloning
→ Value 学习与 oracle 蒸馏
→ self-play PPO
→ deck-policy 评估与提交
```

静态任务按 Card ID 划分；动态 replay 任务按完整 episode 与日期划分。所有阶段先完成 tiny-batch overfit、mask 测试和 CPU benchmark，再扩大训练规模。
