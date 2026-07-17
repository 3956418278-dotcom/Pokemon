# 原 12 个自定义问题的合并处理要求

以下 12 项是此前已经确认需要重新审查的 Codex 自定义设计。本轮不能遗漏。

它们与前述 15 个问题放在同一轮工作中处理，但执行顺序如下：

```text
第一阶段：修复 Replay 事实、状态、动作标签和可复现性
第二阶段：冻结训练目标接口
第三阶段：以后再实现可选的复杂模块
```

## 阶段划分

### 本轮必须完成

```text
1. multi-select unordered 语义
2. policy_loss_mask
3. option equivalence class
5. is_turn_owner
6. Temporal Event 时间语义
7. Anonymous Hidden Pools
9. Card ID Memory
10. Compact Dataset
11. Missingness / OptionalField
```

### 本轮只整理接口和文档，暂不实现训练功能

```text
4. group-aware subset loss
8. Residual IPF
12. used_detail
```

原因是第 4、8、12 项依赖前面的 Replay 语义、equivalence、memory 和 missingness 正确。先实现会把尚未确认的假设固化进训练代码。

---

# 1. Multi-select context 的动作语义

涉及文件：

```text
data/legal_options.py
scripts/audit_replay_decision_contract.py
docs/replay_decision_data_contract.md
tests/test_hidden_belief_and_legal_options.py
```

需要审查的 context：

```text
2, 5, 7, 8, 9, 15, 21, 26, 27, 34
```

另保留 context 22 的顺序语义审查。

对每个 context 单独选择：

```text
反驳当前质疑
或
执行修复
```

### 允许设为 UNORDERED_UNIQUE_SUBSET 的条件

必须证明：

1. 选择结果仅取决于成员集合或各等价类选择数量；
2. 改变 action 中的顺序不会改变：

   * 卡牌和目标的配对；
   * 伤害或能量的分配；
   * 后续 select 的顺序；
   * 日志顺序中具有游戏意义的部分；
   * 最终状态；
3. 规则文本支持集合语义；
4. Replay 中存在实际样本支持该解释。

### 证据不足时

使用：

```text
ORDERED_INDEX_SEQUENCE
```

这是保守标签，不会把两个可能不同的动作错误合并。

### 特别要求

context 5、7、8、9 是此前未向用户明确说明的额外自定义定义，必须单独报告，不能与其他 context 合并说明。

审计输出增加：

```text
semantic_source:
  RULE_CONFIRMED
  REPLAY_SUPPORTED
  IMPLEMENTATION_ASSUMPTION
  UNRESOLVED
```

`IMPLEMENTATION_ASSUMPTION` 和 `UNRESOLVED` 不能自动进入 unordered 白名单。

---

# 2. policy_loss_mask 的语义

涉及文件：

```text
data/legal_options.py
data/replay_dataset.py
data/decision_schema.py
docs/replay_decision_data_contract.md
tests/test_hidden_belief_and_legal_options.py
tests/test_replay_dataset.py
```

当前原则是：

```text
只有一个语义结果时，不计算 policy loss。
```

该原则本身可以保留，但必须建立在可靠的 equivalence class 和 action semantics 上。

### 执行顺序

先完成：

```text
动作语义修复
option equivalence 修复
```

再修复 `policy_loss_mask`。

### 必须满足

`policy_loss_mask=False` 只允许用于：

1. 引擎只提供一个合法动作；
2. 多个 option 已被可靠证明为同一语义结果；
3. 固定数量的 unordered action 只有一个可行等价类计数组合；
4. 强制选择全部选项且顺序无语义；
5. COUNT_VALUE 只有一个不同数值。

### 不能屏蔽

以下情况必须保留 policy loss：

* option 不同但 equivalence 无法证明；
* 只有一个等价类，但可选择不同数量；
* ordered sequence 存在不同顺序；
* context 尚处于 `UNRESOLVED`；
* equivalence resolution 失败；
* option 的实体、目标或效果引用不完整。

增加审计字段：

```text
policy_mask_reason
```

例如：

```text
SINGLE_LEGAL_OPTION
ONE_COUNT_VALUE
ONE_FEASIBLE_CLASS_TARGET
FORCED_FULL_SUBSET
UNRESOLVED_EQUIVALENCE
REAL_POLICY_CHOICE
```

---

# 3. Option Equivalence Class

涉及文件：

```text
data/legal_options.py
data/observation_parser.py
data/state_schema.py
data/decision_schema.py
scripts/audit_replay_decision_contract.py
docs/replay_decision_data_contract.md
tests/test_hidden_belief_and_legal_options.py
```

这是本轮最先修复的核心之一。

### 两个 option 只有在以下信息全部等价时才能合并

```text
select type
select context
option type
source entity
target entity
source zone
target zone
source owner
target owner
detail / attack / ability reference
effect reference
contextCard reference
所有实际参与结算的 option 字段
字段 missingness 状态
```

### Serial 处理

Serial 只能在满足以下条件时被移除：

1. 两个 Serial 均成功解析到当前可见实例；
2. Card ID 相同；
3. 所有可见动态状态相同；
4. 当前区域和 owner 相同；
5. option 对结算的其他字段相同；
6. 引擎允许选择任一副本得到相同结果。

### 当前需要修复的边界

* Card ID 缺失时不能只因删除 `name` 而合并；
* unresolved index 必须保留；
* unresolved direct identity 必须保留；
* `select.deck` 的 zone 要和 Card Instance 一致；
* Tool/Energy 子项的 target zone 要指向实际父实体区域；
* 显式空列表和字段缺失不能通过 `or` 混合；
* class ID 仅是样本内编号，文档不能暗示它跨样本稳定。

### 审计要求

输出：

```text
equivalence_resolution_status:
  FULLY_RESOLVED
  PARTIALLY_RESOLVED
  UNRESOLVED
```

只有 `FULLY_RESOLVED` 允许自动合并。

---

# 4. Group-aware Subset Loss

涉及文件：

```text
data/decision_schema.py
docs/replay_decision_data_contract.md
tests/test_hidden_belief_and_legal_options.py
```

本轮不实现正式训练 loss。

### 本轮只完成

1. 保留 `LegalActionTarget` 中：

   * equivalence class capacities；
   * chosen class counts；
   * selected count；
2. 确认数据结构足够支持未来 subset loss；
3. 文档明确标注：

```text
TARGET CONTRACT IMPLEMENTED
TRAINING LOSS NOT IMPLEMENTED
```

4. 删除任何暗示该 loss 已经进入训练的表述；
5. 不新增训练模块和单独 loss 脚本。

### 以后实现的前置条件

```text
unordered context 全部确认
equivalence class 全部确认
policy_loss_mask 确认
至少一次真实 Replay target audit 通过
```

---

# 5. is_turn_owner

涉及文件：

```text
data/replay_dataset.py
data/decision_schema.py
data/observation_parser.py
scripts/audit_replay_decision_contract.py
docs/replay_decision_data_contract.md
tests/test_replay_dataset.py
```

当前统一设置为：

```text
UNKNOWN
```

不能直接保留而不进行审计。

### 必须完成的工作

检查 Replay 中以下信息能否共同确定 turn owner：

```text
current.turn
current.yourIndex
current.firstPlayer
select 类型和来源
当前 agent seat
前后 observation
turnActionCount
日志 actor
回合边界变化
```

### 输出审计

统计：

```text
explicit_turn_owner_count
deterministically_inferred_count
ambiguous_count
conflicting_count
```

### 可确定时

字段应为：

```text
value
state = PRESENT 或 INFERRED
inference_source
```

建议增加状态来源：

```text
EXPLICIT
INFERRED_AUDITED
UNKNOWN
```

### 无法确定时

只对无法确定的样本保留 UNKNOWN。

有效反驳“可以确定”的条件是：

* 给出反例；
* 说明相同字段组合为何可能对应不同 turn owner；
* 说明由强制效果、攻击结算或对手选择造成的歧义；
* 给出 Replay 示例。

不能因为没有单一显式字段，就把整列全部丢弃。

---

# 6. Temporal Event 时间语义

涉及文件：

```text
data/state_schema.py
data/observation_parser.py
data/game_memory.py
docs/replay_decision_data_contract.md
tests/test_dynamic_features.py
```

当前字段混合了：

```text
事件发生时间
事件被观察的时间
日志 batch 内位置
当前决策的位置
```

### 执行要求

统一为可证明的语义：

```text
observed_at_turn
observed_at_turn_action_count
batch_position
observation_age
turn_age
```

只有 Replay 能明确提供事件真实发生时间时，才添加：

```text
occurred_at_turn
occurred_at_action_count
```

### 模型输入

保留的时间字段必须真正进入 Event Feature，至少包括：

```text
event type
actor relative
source/target
batch position
observation age
turn age
```

不能只在 dataclass 中保存却永远不进入模型。

### 日志 batch 顺序

日志 batch 的原始顺序必须保留。

反向日志或撤销日志使用独立标记，不能通过重新排序破坏事件序列。

---

# 7. Anonymous Hidden Pools

涉及文件：

```text
data/game_memory.py
data/decision_schema.py
data/replay_dataset.py
docs/replay_decision_data_contract.md
tests/test_dynamic_features.py
```

本项与“匿名事件清空整个区域 Serial”问题同时修复。

### 正确边界

Anonymous Hidden Pools 只记录：

```text
当前匿名隐藏数量
匿名区域流量
无法绑定到具体 Serial 的移动
```

它不负责：

```text
猜测具体 Card ID
修改所有可能 Serial
产生 presence probability
恢复具体卡牌当前位置
```

### 当前数量与累计流量分开

必须明确分为：

```text
current_unknown_count
cumulative_anonymous_in_count
cumulative_anonymous_out_count
```

不能把累计事件流量当成当前池大小。

### 当前未知数量

优先根据当前 snapshot 计算：

```text
公开的区域总数量
减去当前可以精确识别的实例数量
```

要求检查自方 deck/prize 是否应合并。若两个区域在当前视角下可区分，应分别保留：

```text
self_unknown_deck_count
self_unknown_prize_count
```

不要无必要合并为一个字段。

---

# 8. Residual IPF

涉及文件：

```text
data/hidden_belief.py
data/decision_schema.py
docs/replay_decision_data_contract.md
tests/test_hidden_belief_and_legal_options.py
```

本轮不删除数学工具，但保持 inactive。

### 本轮执行

1. 保留独立纯函数；
2. 保留数值约束测试；
3. 明确输入必须来自未来可靠的：

   * Card ID expected count；
   * 当前隐藏区容量；
   * exact known hidden copies；
4. 当前 Replay dataset 默认不运行 IPF；
5. `hidden_belief_state` 明确为：

   * `NOT_APPLICABLE`：模块未启用；
   * `PRESENT`：真实计算完成；
   * `UNKNOWN`：模块启用但输入不足。

### 不允许

* 用伪造或默认 expected count 运行 IPF；
* 将输出称为 calibrated posterior；
* 将 expected zone count 称为 presence probability；
* 为了保留接口而给模型输入全零但标记有效。

### 后续启用条件

```text
Hidden Belief 的 presence/count 训练目标已定义
牌组统计时间窗无泄漏
匿名池当前数量可靠
Card ID Memory 语义可靠
```

---

# 9. Card ID Memory

涉及文件：

```text
data/game_memory.py
data/decision_schema.py
data/replay_dataset.py
docs/replay_decision_data_contract.md
tests/test_dynamic_features.py
```

当前需要重新检查：

```text
revealed_unique_copy_count
historical_seen_count
historical_move_count
first_seen_turn
last_seen_turn
```

### 原则

字段名必须对应真实计算式。

### 建议处理

将现有字段重构为：

```text
known_serial_count
currently_exact_zone_counts
ambiguous_serial_count
visible_observation_count
movement_event_count
first_known_turn
last_known_turn
```

只有存在明确 reveal 事件时才增加：

```text
revealed_copy_count
```

### 去重

以下信息如可从 Serial Registry 无损推导，并且当前模型没有单独使用，可不重复存储：

```text
first/last known turn
总移动次数
总观察次数
```

Card ID Memory 应优先表达聚合后的当前知识，而不是复制全部历史统计。

### 特别检查

同一 Card ID 的多个 Serial：

* 必须保持副本数量；
* 不能把同一副本跨多个 observation 误算成多个副本；
* 不能把 attachment 和 parent 误聚合成同一区域副本。

---

# 10. Compact Dataset

涉及文件：

```text
data/replay_dataset.py
data/decision_schema.py
scripts/audit_replay_decision_contract.py
scripts/import_online_replay_decisions.py
docs/replay_decision_data_contract.md
docs/KAGGLE_WORKFLOW.md
tests/test_replay_dataset.py
```

紧凑存储原则可以保留，但必须保证可复现。

### Reference row 必须包含

```text
replay_key
episode_id
source_kind
source_path 或 archive path
archive member
JSONL line
source content hash
decision step
action step
player index
observation fingerprint
parser version
schema version
action target
supervision
```

按来源类型，未适用字段可以为空。

### Loader 必须校验

```text
source hash
replay identity
observation fingerprint
decision key
parser version
schema version
```

任何不一致都明确报错。

### 不能依赖

* 可变的绝对临时路径；
* 文件遍历顺序；
* 仅用 EpisodeId；
* 当前最新 parser 自动重新解释旧 reference。

### 版本变化

本轮修复后提升：

```text
decision reference schema version
replay decision contract version
```

旧 reference 不得被无提示地当作新版读取。

---

# 11. Missingness / OptionalField

涉及文件：

```text
data/decision_schema.py
data/state_schema.py
data/observation_parser.py
data/legal_options.py
data/replay_dataset.py
docs/replay_decision_data_contract.md
tests/test_dynamic_features.py
tests/test_replay_dataset.py
tests/test_hidden_belief_and_legal_options.py
```

保留以下状态是合理的：

```text
PRESENT
MISSING
UNKNOWN
NOT_APPLICABLE
EXPLICIT_NULL
```

但必须统一实现。

### 字段级解析

每个字段定义自己的规则：

```text
字段是否适用
字段是否存在
是否显式 null
是否使用特定 sentinel
值是否合法
```

不能使用全局：

```python
所有负整数 -> UNKNOWN
```

### OptionalField 的使用

模型读取任何 placeholder 时必须同时读取 state 或 validity mask。

不允许：

```text
保存了 state
但模型只读取 placeholder value
```

### 合并多个字段状态

`_combined_state()` 不能简单依靠固定优先级掩盖语义。

例如：

```text
benchMax PRESENT + bench MISSING
```

应表示该派生字段无法计算，而不是随意继承某个状态。

为每个派生字段定义明确的 applicability 和 validity。

### effect 与 contextCard

继续分别保存：

```text
effect_reference + state
context_card_reference + state
```

两者不能互相覆盖。

---

# 12. used_detail

涉及文件：

```text
data/state_schema.py
data/observation_parser.py
data/replay_dataset.py
docs/replay_decision_data_contract.md
tests/test_dynamic_features.py
```

本轮不继续扩展完整 used_detail 系统。

当前审计显示 Ability 行为没有稳定的 detail ID 日志映射，因此不能可靠产生 TRUE/FALSE 标签。

### 本轮执行

保留最小状态：

```text
UNKNOWN
NOT_APPLICABLE
```

仅在有显式、稳定且已审计映射时允许：

```text
TRUE
FALSE
```

同时记录：

```text
inference_source
```

### 默认规则

```text
非 Pokémon 或没有 detail -> NOT_APPLICABLE
存在 detail 但无法确定使用情况 -> UNKNOWN
有显式 attackId 映射 -> 可针对对应 attack detail 标记
Ability 无可靠映射 -> UNKNOWN
```

### 暂时删除或停用

如果以下字段没有任何当前消费者，可以先不进入模型 batch：

```text
used_detail_states
used_detail_inference_sources
```

可保留 schema 接口，但明确标注 inactive。

### 后续启用条件

```text
建立稳定 detail resolver
Replay 日志能绑定到具体 detail
至少通过全量映射审计
定义实际辅助训练目标
```

---

# 与前述 15 个问题的重叠关系

以下问题合并处理，不重复实现：

```text
原问题 1  <-> 新问题：context 5/7/8/9 未披露
原问题 3  <-> 新问题：equivalence class 边界
原问题 5  <-> 新问题：is_turn_owner 全 UNKNOWN
原问题 6  <-> 新问题：事件时间字段不准确
原问题 7  <-> 新问题：匿名事件破坏 Serial Registry
原问题 9  <-> 新问题：Card ID Memory 命名错误
原问题 10 <-> 新问题：DecisionKey 与 compact reference 不可靠
原问题 11 <-> 新问题：全局负整数 UNKNOWN
```

修复时形成一个统一实现，不建立两套并行逻辑。

---

# 本轮推荐执行顺序

```text
1. 修复 DecisionKey 和 reference identity
2. 修复字段级 Missingness
3. 统一 select.deck 和实例区域
4. 修复 Anonymous Event / Serial Registry
5. 修正 Card ID Memory
6. 修正 Temporal Event
7. 审核 option equivalence
8. 审核 multi-select context
9. 修正 policy_loss_mask
10. 审核 is_turn_owner
11. 修复热门牌组与时间窗问题
12. 更新合同、schema 和测试
13. 将 subset loss、IPF、used_detail 标记为未启用
```

---

# 最终报告追加要求

最终报告除前述 15 个问题外，必须再输出原 12 项专表：

```text
原问题编号
当前结论
反驳或执行
证据
最终语义
修改文件
测试
是否立即启用
```

其中第 4、8、12 项的“是否立即启用”默认应为：

```text
否
```

除非有新的、明确的 Replay 证据证明它们已具备可靠输入和训练用途。
