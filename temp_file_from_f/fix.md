请基于当前最新提交：

```text
d73b58e765e25a4e055c09183e4605dcbf4dc4e7
data_processing_from_replay(improve needed)
```

修复 Replay 数据处理实现。

本轮目标是校正数据语义和训练标签，不扩展模型结构，不开始动态训练，不新增额外实验框架。

## 一、执行方式

对下面每个问题，必须明确选择以下两种处理之一：

### A. 反驳

当你认为当前实现正确时，必须同时提供：

1. 对应游戏规则或引擎语义；
2. 对应 Replay 字段或全量审计统计；
3. 当前代码为何不会产生错误样本；
4. 至少一个能验证该结论的现有 Replay 示例；
5. 对应回归测试。

只写“为了保守”“根据经验”“测试通过”“审计过”不构成有效反驳。

### B. 执行

当问题成立时：

1. 直接修改实现；
2. 修改对应数据合同或字段名称；
3. 在现有测试文件中增加最小回归测试；
4. 保持现有仓库结构；
5. 不单独创建临时测试脚本；
6. 不生成新的散乱输出目录；
7. 对 schema 语义发生变化的地方提升 schema version。

本轮不需要逐项询问我。证据不足时采用语义最保守、不会伪造信息的实现。

---

# 二、允许修改的文件

核心实现文件：

```text
data/decision_schema.py
data/state_schema.py
data/observation_parser.py
data/game_memory.py
data/legal_options.py
data/replay_dataset.py
data/hidden_belief.py
data/online_replay_importer.py
```

Replay 脚本：

```text
scripts/audit_replay_decision_contract.py
scripts/import_online_replay_decisions.py
scripts/normalize_replay_statistics.py
kaggle/kernels/replay_extract/extract_popular_decks.py
```

数据合同和说明：

```text
docs/replay_decision_data_contract.md
docs/KAGGLE_WORKFLOW.md
```

现有测试文件：

```text
tests/test_dynamic_features.py
tests/test_replay_dataset.py
tests/test_hidden_belief_and_legal_options.py
tests/test_online_replay_importer.py
tests/test_normalize_replay_statistics.py
tests/test_card_dataset.py
```

静态卡文件仅在处理第 12 项时修改：

```text
static_card/data/card_dataset.py
static_card/README.md
```

保持修改集中在以上文件。只有出现明确依赖时，才允许修改其他文件，并在最终报告中说明原因。

---

# 三、逐项反驳或执行

## 1. 匿名事件不能清空同一区域全部 Serial

涉及文件：

```text
data/game_memory.py
tests/test_dynamic_features.py
```

当前问题：

当事件没有 `serial`，但存在 `fromArea` 时，当前实现会遍历该玩家所有位于 `fromArea` 的 Serial，并将它们全部改成：

```text
current_area = None
possible_hidden_zone_mask |= ...
```

一个匿名移动事件通常只表示一张或指定数量的匿名卡移动，不能让整个区域内所有已知卡失去精确位置。

### 执行要求

实现以下语义：

* 没有 Serial 的事件只更新匿名区域流量；
* 不修改任何具体 Serial 的精确位置，除非 Replay 中有足够信息唯一确定该 Serial；
* `quantity` 只影响匿名池计数；
* 当前 snapshot 仍是 Serial 精确位置的最高优先级来源；
* 无法确定移动的是哪张已知 Serial 时，保留这些 Serial 的最后已知状态，并单独记录“不确定匿名流”。

增加测试：

* 一个区域内有三张已知 Serial；
* 日志只有一张匿名卡从该区域移出；
* 更新后不能把三张 Serial 全部设为未知。

---

## 2. 审核 context 5、7、8、9 的 unordered 定义

涉及文件：

```text
data/legal_options.py
scripts/audit_replay_decision_contract.py
docs/replay_decision_data_contract.md
tests/test_hidden_belief_and_legal_options.py
```

当前代码将以下 context 写入 unordered 集合：

```text
2, 5, 7, 8, 9, 15, 21, 26, 27, 34
```

但当前合同只充分说明了：

```text
2, 15, 21, 26, 27, 34
```

### 处理要求

对 context 5、7、8、9 分别反驳或执行。

有效反驳需要给出：

* select.type；
* select.context；
* option.type；
* 对应卡牌或游戏效果；
* 选择顺序是否会改变结算；
* Replay 样本数量；
* 至少一个实际 Replay 示例；
* 规则文本或结算日志证据。

没有足够证据时，将它们从：

```python
AUDITED_UNORDERED_SELECT_CONTEXTS
```

中移除，让它们回到：

```text
ORDERED_INDEX_SEQUENCE
```

测试不能只断言硬编码结果，必须体现该 context 的实际结算语义。

审计脚本中要区分：

```text
规则人工确认
Replay 支持证据
Replay 无法证明
```

不能把人工填写的 `CONTEXT_SEMANTICS` 当作审计自动得出的结论。

---

## 3. 修复 DecisionKey 在 EpisodeId 缺失时的碰撞

涉及文件：

```text
data/decision_schema.py
data/replay_dataset.py
docs/replay_decision_data_contract.md
tests/test_replay_dataset.py
```

当前 `DecisionKey` 只有：

```text
episode_id
decision_step_index
action_step_index
player_index
```

其中 `episode_id` 可以为 `None`。

不同 Replay 缺少 EpisodeId 时会产生相同主键。

### 执行要求

为正式决策主键加入稳定 Replay 身份。

推荐结构：

```text
replay_key
episode_id
decision_step_index
action_step_index
player_index
```

其中 `replay_key` 按顺序来源于：

```text
EpisodeId
replay.id
archive member / JSONL record identity
规范化 source path
```

要求：

* `replay_key` 必须非空；
* `episode_id` 保留为可空元数据；
* JSONL 同一文件的不同记录必须可区分；
* ZIP 中不同 member 必须可区分；
* schema version 提升；
* 文档主键同步更新。

增加测试：

* 两个 EpisodeId 均缺失的 Replay；
* step 和 player 完全相同；
* 生成的 DecisionKey 仍必须不同。

---

## 4. 处理实际上无效的 include_no_select

涉及文件：

```text
data/replay_dataset.py
scripts/import_online_replay_decisions.py
data/online_replay_importer.py
tests/test_replay_dataset.py
tests/test_online_replay_importer.py
```

当前接口提供：

```text
include_no_select
```

但遇到 `select is None` 时仍然直接跳过。

### 处理要求

选择以下一种：

#### 方案 A：正式支持

定义 no-select 样本的明确用途和 schema，例如仅作为状态/value 辅助样本：

```text
policy_loss_mask = false
legal_options = empty
legal_action target = not applicable
```

同时避免把它伪装成行为克隆决策。

#### 方案 B：删除接口

当当前训练流程只需要决策点时：

* 删除 `include_no_select` 参数；
* 删除命令行参数；
* 删除 importer 透传；
* 更新文档；
* 避免留下“看似支持但没有效果”的接口。

优先选择结构更简单且符合当前训练目标的方案。

---

## 5. 非法 Action 内容不能静默变成 0

涉及文件：

```text
data/replay_dataset.py
scripts/audit_replay_decision_contract.py
tests/test_replay_dataset.py
```

当前 `_action_list()` 会将无法转成整数的 action 元素变成 `0`。

### 执行要求

实现严格解析：

* 合法整数保留；
* 可明确解析为整数的数值字符串可以接受；
* `None` 按真实空 action 处理；
* 无法解析的元素记录为 action alignment error；
* 该样本不进入正式训练数据；
* 不得自动替换为 option 0。

增加测试：

```text
action = ["bad"]
action = [None]
action = [object-like invalid value]
```

确认不会生成选择第 0 项的伪标签。

---

## 6. 统一 select.deck 的区域语义

涉及文件：

```text
data/observation_parser.py
data/legal_options.py
data/state_schema.py
docs/replay_decision_data_contract.md
tests/test_dynamic_features.py
tests/test_hidden_belief_and_legal_options.py
```

当前：

* Observation parser 将 `select.deck` 标为 `LOOKING=12`；
* Legal option resolver 将其当作 `DECK=1`。

### 执行要求

使用统一表示：

```text
当前可见选择列表区域：LOOKING
来源信息：source = "select.deck"
```

即：

* Card Instance zone 使用 `LOOKING`；
* Legal option 中被解析的可见实体也使用 `LOOKING`；
* 如果模型需要知道它来源于牌库，使用独立 source/origin 字段；
* 不用一个 zone 同时表达“当前可见位置”和“历史来源”。

增加测试确保：

```text
Card Instance area
option equivalence source_zone
instance-option reference
```

三者一致。

若你认为 Legal Option 必须使用 `DECK`，需要反驳并说明为什么同一实体在 Card Instance 中可以安全使用 `LOOKING`。

---

## 7. FieldState 使用字段级 sentinel，而不是所有负整数统一 UNKNOWN

涉及文件：

```text
data/observation_parser.py
data/legal_options.py
data/replay_dataset.py
data/decision_schema.py
tests/test_dynamic_features.py
tests/test_hidden_belief_and_legal_options.py
```

当前通用逻辑：

```python
isinstance(value, int) and value < 0
```

会把所有负整数视为 `UNKNOWN`。

### 执行要求

建立字段级 sentinel 规则。

例如：

```text
firstPlayer = -1       -> UNKNOWN
result = -1            -> UNKNOWN
缺失字段                -> MISSING
显式 null              -> EXPLICIT_NULL
合法负数数值            -> PRESENT
```

不要使用全局“负数即未知”。

为以下情况增加测试：

* 引擎明确使用 `-1` 作为未知 sentinel；
* 合法负值字段仍保持 `PRESENT`；
* 缺失、null、unknown 三种状态互相区分。

---

## 8. 修正 Card ID Memory 字段名称和统计含义

涉及文件：

```text
data/decision_schema.py
data/game_memory.py
data/replay_dataset.py
docs/replay_decision_data_contract.md
tests/test_dynamic_features.py
tests/test_replay_dataset.py
```

当前问题：

```text
historical_seen_count
```

实际统计的是 Serial 出现在多少次 observation 中，不是被揭示多少次。

```text
revealed_unique_copy_count
```

实际接近“已建立 SerialMemory 的数量”，不保证每个都经过公开揭示。

### 执行要求

按真实含义改名。

推荐：

```text
visible_observation_count
known_serial_count
movement_event_count
first_known_turn
last_known_turn
```

只有存在明确 reveal 事件或公开可见证据时，才使用：

```text
revealed_copy_count
```

要求：

* 字段名称必须直接反映计算公式；
* 不把持续可见多个 observation 计算成多次 reveal；
* 不重复保存可由 Serial Registry 无损推导且模型暂时不使用的字段；
* 对明显冗余字段可以删除；
* schema version 同步提升。

---

## 9. 修正 Temporal Event 的时间语义

涉及文件：

```text
data/state_schema.py
data/observation_parser.py
data/game_memory.py
docs/replay_decision_data_contract.md
tests/test_dynamic_features.py
```

当前 `observed_turn` 和 `position_in_turn` 取自看到日志时的当前 snapshot，不一定是事件真实发生时刻。

同时：

```text
turn_delta
position_in_turn
```

虽然被保存，但没有真正进入 `recent_event_features()`。

### 处理要求

区分：

```text
observed_at_turn
observation_batch_position
current_turn_action_count_at_observation
event_age_in_observations
```

不要把无法证明的值命名为真实事件发生时间。

如果 Replay 无法恢复真实发生时刻：

* 明确使用 observation-relative 命名；
* 保留日志 batch 内顺序；
* 将事件新旧程度和 batch 顺序真正编码进 Event Feature；
* 删除未使用且容易误解的时间字段，或让它们进入模型输入；
* 更新 `EVENT_FEATURE_DIM` 和相应测试。

---

## 10. is_turn_owner 必须反驳或实现

涉及文件：

```text
data/replay_dataset.py
data/decision_schema.py
docs/replay_decision_data_contract.md
scripts/audit_replay_decision_contract.py
tests/test_replay_dataset.py
```

当前所有样本都将：

```text
is_turn_owner = UNKNOWN
turn_owner_relative = UNKNOWN
```

这会永久丢失决定动作权限和信息遮掩的重要变量。

### 处理要求

先审计 Replay 和引擎字段。

#### 能可靠推断时

实现：

```text
value
state
inference_source
```

例如：

```text
EXPLICIT_ENGINE_FIELD
AUDITED_SELECT_OWNERSHIP_RULE
AUDITED_TURN_RULE
```

#### 不能可靠推断时

保留 UNKNOWN，但必须给出完整反驳：

* Replay 为什么无法判断；
* 哪些 select 可能不是当前玩家正常回合；
* 哪些强制效果会破坏简单推断；
* 为什么根据 turn parity 或 select presence 推断会制造错误；
* 后续需要哪个引擎字段才能解决。

不能仅写“为了保守，所以 UNKNOWN”。

---

## 11. Compact reference 必须可稳定重建

涉及文件：

```text
data/replay_dataset.py
data/decision_schema.py
docs/replay_decision_data_contract.md
docs/KAGGLE_WORKFLOW.md
tests/test_replay_dataset.py
```

当前 reference 依赖 source path 和 decision key 重建，但 source path 会变化，parser/schema 也会变化。

### 执行要求

reference 至少保存：

```text
replay_key
source kind
source path / archive path
archive member 或 JSONL line
source content hash
observation fingerprint
parser contract version
decision schema version
```

重建时校验：

* 原始 Replay 内容 hash；
* observation fingerprint；
* decision key；
* parser/schema version。

校验失败时明确报错，不静默重建出不同样本。

不要重新复制完整 observation。

---

## 12. 静态 Card Dataset 的 split 语义

涉及文件：

```text
static_card/data/card_dataset.py
static_card/README.md
tests/test_card_dataset.py
```

当前 feature schema 和 normalization 由全量卡牌生成，然后再划分 train/validation/test。

对这一项反驳或执行。

### 可接受的反驳

如果当前 split 只用于固定卡池内的优化监控，不宣称未见卡泛化：

* 在 README 中明确说明；
* 将该 split 命名和用途写清；
* 不把结果解释成 unseen-card generalization。

### 需要执行的情况

如果 validation/test 用于评估未见 Card ID：

* vocab 和 normalization 只能由 train partition 构造；
* validation/test 的未知值进入 UNK；
* split manifest 保存 seed 和 Card ID 列表；
* 测试验证 validation/test 不参与 schema 构造。

优先保持当前项目真实需要，避免为不存在的泛化目标增加复杂度。

---

## 13. 修复热门牌组 Variant 对卡牌排列顺序敏感

涉及文件：

```text
kaggle/kernels/replay_extract/extract_popular_decks.py
tests/test_replay_dataset.py
docs/KAGGLE_WORKFLOW.md
```

当前使用：

```python
Counter(tuple(row["deck"]))
```

同一套 60 卡只要排列顺序不同，就会被统计为不同 variant。

### 执行要求

完整牌组身份使用 Card ID multiset：

```text
sorted(card_id -> copy_count)
```

或者已有的：

```text
deck_fingerprint
```

要求：

* 相同 Card ID count、不同列表顺序属于同一 variant；
* 不同 Trainer 配置仍可属于不同完整 variant；
* Pokémon+Energy signature 的 archetype 分组逻辑保持独立；
* 增加顺序打乱后的回归测试。

---

## 14. 找不到 Card CSV 时停止，而不是合并全部牌组

涉及文件：

```text
kaggle/kernels/replay_extract/extract_popular_decks.py
tests/test_replay_dataset.py
docs/KAGGLE_WORKFLOW.md
```

当前找不到 `EN_Card_Data.csv` 时，所有 `card_kind` 为空，最终所有 Pokémon+Energy signature 可能退化成空字符串。

### 执行要求

热门牌组分组需要 Card Type 时，Card Metadata 是必需输入。

实现：

* 找不到 Card CSV 时直接明确失败；
* 错误信息列出搜索过的路径；
* 不继续输出伪造的热门牌组；
* 如果只执行不依赖类型的 deck frequency，可以明确拆分模式，但默认主流程必须严格。

增加测试：

* 缺少 Card Metadata 时不会生成单一空 signature 热门牌组。

---

## 15. 归一化统计必须显式限制训练时间窗口

涉及文件：

```text
scripts/normalize_replay_statistics.py
tests/test_normalize_replay_statistics.py
docs/KAGGLE_WORKFLOW.md
```

当前脚本自动聚合输入目录中的全部日期。

### 执行要求

增加明确时间过滤接口，推荐：

```text
--start-date
--end-date
```

或：

```text
--date
```

可重复指定。

要求：

* 只聚合指定训练窗口；
* 默认行为必须在输出中明确记录全部实际使用日期；
* reserved/test 日期不会被训练统计读取；
* normalization summary 保存：

  * requested window；
  * included dates；
  * excluded dates；
  * total decks；
* 增加 train 和 reserved 日期混合目录的回归测试。

---

# 四、Policy Loss 和 Hidden Belief 的处理

本轮不要实现完整训练模型，但要校正接口状态。

## Group-aware subset loss

当前提交主要实现了 target 和合同数学式，没有在本轮代码中实现训练 loss。

处理方式：

* 文档明确标注“target contract 已实现，训练 loss 尚未接入”；
* 不宣称 group-aware subset loss 已完成；
* 不在本轮额外创建训练模块。

## Residual IPF

保留数学工具可以接受，但：

* 不将其描述为已完成 Hidden Belief 系统；
* 没有可靠 presence/count 输入时保持 inactive；
* `hidden_belief_state` 应表达未启用，而不是暗示已推断；
* 测试只验证数学约束，不把 fixture 测试当成真实 belief 有效性证明。

---

# 五、测试要求

只在现有测试文件中增加回归测试。

至少覆盖：

```text
匿名单卡移动不会清空整区 Serial
context 5/7/8/9 的最终处理
缺失 EpisodeId 时主键不碰撞
非法 action 不转成 0
select.deck zone 全链路一致
字段级 negative sentinel
Card ID Memory 字段真实含义
牌组顺序不影响 variant
缺少 Card CSV 时明确失败
训练统计排除 reserved 日期
```

运行以下相关测试即可：

```bash
pytest -q \
  tests/test_dynamic_features.py \
  tests/test_replay_dataset.py \
  tests/test_hidden_belief_and_legal_options.py \
  tests/test_online_replay_importer.py \
  tests/test_normalize_replay_statistics.py \
  tests/test_card_dataset.py
```

保持测试集中，不为每个问题创建独立脚本。

---

# 六、最终输出格式

修改完成后，输出一张表：

```text
问题编号
结论：反驳 / 执行
证据或修改理由
修改文件
新增测试
是否改变 schema
```

随后列出：

1. 实际修改的文件；
2. 未修改但完成反驳的项目；
3. 无法从当前 Replay 证明的项目；
4. schema version 变化；
5. 测试命令和结果；
6. 仍需真实 Replay 人工检查的最少项目。

不提交训练结果，不创建新实验目录，不开始动态训练。
