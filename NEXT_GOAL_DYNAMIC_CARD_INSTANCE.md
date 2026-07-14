# 下一阶段目标：完成动态单卡表示

## 1. 阶段目标

下一阶段集中完成可训练、可验证、可复用的动态单卡表示。

当前仓库已经具备可靠的静态卡牌表示、observation/replay 解析和动态状态编码原型。本阶段要把现有的“静态向量与临时动态向量拼接”升级为正式的 `CardInstanceFusion`：模型能够根据具体卡牌实例当前的 HP、能量、区域、异常状态和进化情况，动态选择与当前局面相关的攻击、特性和其他 detail，最终生成 128 维 `card_instance_token`。

本阶段完成后，每个可见卡牌实例都应拥有同时包含以下信息的表示：

- 卡牌本身是什么。
- 它当前处于什么状态。
- 它当前能够使用哪些攻击或效果。
- 哪些静态 detail 与当前状态最相关。

## 2. 当前基础

### 2.1 已完成内容

- 静态 CardEncoder 已完成训练。
- 当前卡池共有 1267 张卡。
- 已导出 128 维 `card_summary`。
- 已导出逐卡变长的 `detail_tokens`、`detail_mask` 和 `detail_type_ids`。
- 同一 Card ID 的多个攻击、特性和特殊效果保持为独立 detail。
- observation 与 replay 已能解析当前可见卡牌实例、日志和合法选项。
- `DynamicInstanceEncoder`、`StaticCardEmbeddingAdapter` 和 `CardInstanceFusion` 已有可运行的前向与反向原型。

### 2.2 当前核心缺口

目前动态字段主要被压入固定的 32 维临时向量，随后与静态向量通过 MLP 拼接：

```text
static_card_embedding + dynamic_embedding
→ MLP
→ card_instance_token
```

这种结构没有根据当前动态状态选择相关 detail。例如，一张 Pokémon 卡有两个攻击，而当前附着能量只能支付第一个攻击时，模型还没有明确机制将注意力集中到第一个攻击上。

因此，本阶段先完成动态单卡表示，再将其交给 Ledger、Recent Events 和 Board Transformer。

## 3. 阶段边界

本阶段只处理“单个具体卡牌实例当前是什么状态、能够做什么”。

本阶段包括：

- 真实 replay 动态字段审计。
- 结构化 `CardDynamicBatch`。
- `DynamicInstanceEncoder` 正式实现。
- 动态条件四头 Cross-Attention。
- 单卡辅助训练任务。
- 动态融合 checkpoint、测试和 benchmark。

后续阶段负责：

- 双方 Ledger。
- Recent Events。
- 完整 GameMemory。
- Board Transformer 正式训练。
- ActionEncoder。
- 行为克隆、Value、oracle 蒸馏和 self-play PPO。

## 4. 正式数据流

```text
Card ID
├── card_summary [128]
└── detail_tokens [M, 128]
        ↑
动态卡牌实例
├── owner / visibility
├── zone / field role
├── 当前 HP / 最大 HP / damage ratio
├── 12 类附着能量
├── 附着 Energy Card / Tool
├── 进化链
├── 异常状态
├── appear_this_turn
└── 字段有效性与 unresolved mask
        ↓
DynamicInstanceEncoder
        ↓
dynamic_repr [64]
        ↓
动态条件四头 Cross-Attention
        ↓
card_instance_token [128]
```

`card_id` 表示卡牌静态身份，`serial` 表示一局中的具体卡牌副本。场上出现两张相同 Card ID 时，它们必须按不同 serial 建立实例，并根据各自动态状态得到不同表示。

## 5. 实施任务

### 5.1 建立真实 replay 数据基线

从多个日期导入真实 replay decision points，并生成统一审计报告。报告至少覆盖：

- replay 数量和有效决策点数量。
- parser error 数量及错误类型。
- 可见 Card ID 与静态 artifacts 的匹配率。
- HP、最大 HP、damage ratio 的覆盖率与分布。
- zone、field role、owner 和 visibility 的分布。
- 附着能量类型与数量的分布。
- Tool、进化链和异常状态的覆盖率。
- `appear_this_turn` 的覆盖率。
- Card ID 与 detail token 的对齐率。
- 特殊能量无法解析的比例。
- 空 detail、未知 Card ID 和匿名隐藏实例的数量。

Replay 数据按完整 episode 和日期划分。最近若干日期作为时间保留集，同一局只属于一个 split。

### 5.2 重构 `CardDynamicBatch`

将目前统一的 `dynamic_features [N, 32]` 改为具有明确语义的结构化字段。

建议字段分组：

#### 类别字段

- relative owner
- zone
- field role
- attachment kind
- knowledge / visibility state

#### 数值字段

- current HP
- max HP
- damage
- HP ratio
- damage ratio
- copy count
- energy card count
- tool count
- pre-evolution count

#### 多标签或计数字段

- 12 类附着能量计数
- 5 类异常状态

#### 布尔字段

- is visible
- is face down
- is Pokémon
- appear this turn
- has tool
- has pre-evolution
- is attachment

#### Mask 字段

- HP 是否有效
- Card ID 是否已知
- 当前实例是否可见
- 能量支付是否成功解析
- detail 是否存在
- 各动态字段是否可用于监督

缺失值、未知值和真实零值需要保持可区分。

### 5.3 完成 `DynamicInstanceEncoder`

各类字段分别编码，再融合为 64 维动态表示：

```text
categorical fields → Embedding
numerical fields   → normalization + MLP
count vectors      → projection
boolean fields     → projection
validity masks     → mask embedding
                         ↓
                  fusion MLP
                         ↓
               dynamic_repr [64]
```

编码器需要支持：

- 正常可见实例。
- 未知 Card ID。
- 匿名隐藏实例。
- 空 batch。
- 部分字段缺失。
- 多张相同 Card ID、不同 serial 的实例。

### 5.4 实现动态条件 detail Cross-Attention

正式融合结构固定为：

```text
query = LayerNorm(card_summary + project(dynamic_repr))
key   = detail_tokens + detail_type_embedding
value = detail_tokens + detail_type_embedding

query
→ 4-head Cross-Attention
→ residual connection
→ feed-forward network
→ LayerNorm
→ card_instance_token [128]
```

实现要求：

- `detail_mask` 正确屏蔽 padding。
- attack、ability 和其他 detail 保持独立。
- 动态状态参与 query 构造。
- 无有效 detail 时使用零 context，并保留稳定 residual 路径。
- 全 mask、空 batch 和未知卡牌前向与反向均无 NaN。
- 可选输出 attention weights，用于训练诊断和解释。

### 5.5 建立单卡辅助训练任务

本阶段优先完成四项与动态单卡状态直接相关的任务。

#### 任务一：攻击可支付预测

对每个 attack detail 预测当前能量是否足以支付。

```text
输入：card_instance_token + attack_detail_token
输出：payable / not payable
```

#### 任务二：剩余能量需求预测

对每个 attack detail 预测当前距离满足攻击费用还缺多少能量，必要时按能量类型输出缺口。

#### 任务三：HP 与伤害状态重建

从动态表示重建 HP ratio 或 damage ratio，确保实例表示确实保留当前生存状态。

#### 任务四：区域与场上角色识别

预测实例当前所在区域及角色，例如 Active、Bench、Hand、Discard、Attachment。

攻击任务以单个 attack detail 为监督单位。特殊能量支付关系无法可靠判断时，设置 `unresolved_mask`，该样本不进入对应损失。

总损失形式：

```text
L = λ1 L_payable
  + λ2 L_energy_remaining
  + λ3 L_hp_state
  + λ4 L_zone_role
```

各项损失分别记录训练集和验证集指标，权重通过配置文件维护。

### 5.6 建立训练入口

训练脚本需要完成：

- 读取静态 summary/detail artifacts。
- 冻结静态 CardEncoder 及其导出表示。
- 读取按 episode/date 划分的 replay decision dataset。
- 构造动态实例 batch 和 detail-level 标签。
- 训练 DynamicInstanceEncoder、Cross-Attention Fusion 和辅助 heads。
- 支持断点恢复。
- 保存 best/last checkpoint。
- 输出分任务指标、mask 比例和 unresolved 比例。
- 保存训练配置、数据版本和静态 artifacts metadata。

本阶段 checkpoint 至少包含：

- `dynamic_instance_encoder`
- `card_instance_fusion`
- auxiliary heads
- optimizer state
- training config
- static artifact version
- replay split metadata

## 6. 测试要求

### 6.1 数据测试

- 同一 Card ID 的不同 serial 保持为不同实例。
- 对手隐藏手牌不会生成虚构 Card ID。
- 缺失字段与真实零值可区分。
- attack detail 与能量费用正确对齐。
- replay split 不发生 episode 泄漏。

### 6.2 模型测试

- 正常 batch 前向输出形状正确。
- 空 batch 可以安全执行。
- 无 detail 卡牌可以安全执行。
- 全 detail mask 情况无 NaN。
- padding 的变化不影响有效 detail 输出。
- 同一 Card ID 在不同 HP 下生成不同 instance token。
- 同一 Card ID 在不同能量状态下生成不同 instance token。
- 改变能量后，对应攻击的可支付预测发生合理变化。
- loss 可以反向传播到动态编码器、Cross-Attention 和辅助 heads。
- 静态 artifacts 保持冻结。

### 6.3 训练测试

- 小批量样本可以明显过拟合。
- 四项辅助任务均产生有效梯度。
- unresolved 样本不会进入错误监督。
- checkpoint 保存后重新加载结果一致。
- 固定随机种子时，关键指标可复现。

## 7. 验收标准

本阶段只有同时满足以下条件才算完成：

1. 多日期 replay 审计报告已经生成，主要字段覆盖率和 unresolved 比例已明确。
2. `CardDynamicBatch` 已完成结构化重构。
3. 动态条件四头 Cross-Attention 已替换浅层静态/动态拼接。
4. 四项辅助训练任务已经实现。
5. 真实 replay batch 可以完成 forward 和 backward。
6. tiny batch 可以明显过拟合。
7. 同一卡牌在不同 HP、能量和区域下产生不同实例表示。
8. detail mask、空 detail、未知 Card ID 和隐藏实例均无 NaN。
9. 按日期划分的验证集已经输出正式指标。
10. 已保存 best/last 动态单卡融合 checkpoint。
11. CPU benchmark 已记录实例数、token 数、耗时和内存占用。
12. 自动测试覆盖数据对齐、mask、特殊能量和同 Card ID 多实例。

## 8. 本阶段交付物

代码交付物：

- 结构化动态状态 schema。
- 重构后的 `DynamicInstanceEncoder`。
- 动态条件 `CardInstanceFusion`。
- 四个辅助任务 heads。
- replay 动态训练 dataset/collator。
- 正式训练入口。
- 评估与 benchmark 脚本。
- 完整自动测试。

训练交付物：

- replay 字段审计报告。
- train/validation/test split metadata。
- best/last checkpoint。
- 分任务训练与验证指标。
- attention 与能量可支付诊断样例。
- CPU benchmark 结果。

## 9. 完成后的项目状态

```text
静态 CardEncoder                 完成
动态 CardInstanceFusion         完成
Ledger / Recent Events          下一阶段
Board Transformer 正式训练      下一阶段
ActionEncoder / 行为克隆        后续阶段
Value / oracle / self-play PPO  后续阶段
```

下一阶段的完整执行顺序为：

```text
真实 replay 数据审计
→ 结构化动态字段
→ DynamicInstanceEncoder
→ 动态条件 Cross-Attention
→ 单卡辅助训练
→ tiny-batch overfit
→ 时间保留集验证
→ 动态单卡融合 checkpoint
```

该 checkpoint 完成并通过验收后，再进入 Ledger、Recent Events 和完整局面表示阶段。
