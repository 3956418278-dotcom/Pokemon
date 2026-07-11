# Codex 指令：从静态预处理基线实现动态实例与时序全局状态

## 0. 当前真实进度

当前仓库已经完成的只有：

```text
静态卡牌数据预处理
```

本轮以现有静态预处理产物为唯一基线。以下模块均视为待新建，只有仓库审计明确证明已经存在时才复用：

```text
StaticCardEncoder / StaticFeatureAdapter
Observation 状态解析
DynamicInstanceEncoder
CardInstanceFusion
GlobalSnapshotExtractor / Encoder
DecisionContextEncoder
MatchContextEncoder
GameMemoryState
VisibleSerialRegistry / CardLedger
RecentEventExtractor / EventEncoder
BoardTokenizer / BoardTransformer
```

本轮完成“特征与状态编码系统”。动作评分、行为克隆、价值学习和 PPO 留给后续训练阶段。

---

# 1. 本轮最终目标

以现有静态预处理结果为输入，完成：

```text
静态卡牌记录
    +
当前 observation
    +
自上次选择以来的 logs
    +
上一时刻 GameMemoryState
        ↓
当前卡牌实例动态特征
        +
当前公开局面快照
        +
当前选择上下文
        +
精简全局上下文
            ├── match_state
            ├── self/opponent card_ledger
            └── recent_events
        ↓
动态条件 Cross-Attention
        +
Board Transformer
        ↓
state_embedding
contextualized_tokens
entity/action-reference metadata
updated_memory
```

全局层固定采用三部分：

```text
1. match_state：先后手、回合与版本环境
2. card_ledger：双方当前已知位置、未知池与长期累计使用情况
3. recent_events：最近 16 个重要事件及其顺序
```

事实与推断的职责固定为：

```text
recent_events 保存“刚才发生了什么”
card_ledger 保存“这些事件更新后，现在知道什么”
```

正式 Board Token 序列固定为：

```text
[STATE]
[DECISION]
[MATCH]
[SELF_LEDGER]
[OPP_LEDGER]
[RECENT_EVENT] * N
[current visible card instance tokens]
[hand unique-card tokens]
[discard summary tokens]
```

本轮不再单独生成：

```text
[MEMORY]
[TRANSITION]
[UNKNOWN_SELF_POOL]
[UNKNOWN_OPPONENT_POOL]
[CARD_MEMORY] * N
[HISTORY_SUMMARY]
```

上述信息分别并入 Ledger 或 Recent Event。版本环境字段在 `[MATCH]` 中建立正式接口；当前没有版本数据时使用 UNKNOWN/default 值和有效性 mask。

# 2. 第一步：审计仓库与真实 cabt 数据

先阅读当前仓库并生成：

```text
state_feature_audit.md
```

审计报告必须写清：

```text
1. 静态预处理入口、输出文件、schema、样例和测试
2. 每个 Card ID 是否只有一个 CardRecord
3. 多攻击/多特性是否保存为多个 detail record
4. 静态输出是原始结构字段、数值特征，还是已经是 128 维 embedding
5. 当前是否存在 CardEncoder；如存在，实际输出 shape 是什么
6. observation / replay parser 当前状态
7. cabt Observation、State、PlayerState、Card、Pokemon、SelectData、Option、Log 的实际字段
8. 每个字段的可见性、缺失规则、card_id/serial 情况
9. 当前测试与可运行命令
```

重点核对正式 API 字段：

```text
Observation.current
Observation.select
Observation.logs

State.turn
State.turnActionCount
State.yourIndex
State.firstPlayer
State.supporterPlayed
State.stadiumPlayed
State.energyAttached
State.retreated
State.result
State.stadium
State.looking
State.players

PlayerState.active
PlayerState.bench
PlayerState.benchMax
PlayerState.deckCount
PlayerState.discard
PlayerState.prize
PlayerState.handCount
PlayerState.hand
PlayerState.poisoned / burned / asleep / paralyzed / confused

Pokemon.id / serial / hp / maxHp / appearThisTurn
Pokemon.energies / energyCards / tools / preEvolution

SelectData.type / context / minCount / maxCount
SelectData.remainDamageCounter / remainEnergyCost
SelectData.option / deck / contextCard / effect

全部 LogType 及每种日志实际提供的字段
```

后续实现以审计报告和运行时对象为准。

---

# 3. 静态预处理对接层

## 3.1 保持现有静态预处理不变

现有静态预处理继续负责：

```text
原始多行卡表
→ 按 Card ID 聚合
→ 一个 CardRecord
→ 多个 attack / ability / effect detail records
```

静态表中的同 Card ID 多行表示多个技能条目，不表示多个对局副本。

## 3.2 增加统一读取接口

实现：

```python
@dataclass
class StaticCardEncoding:
    card_summary: Tensor        # [B, 128]
    detail_tokens: Tensor       # [B, M, 128]
    detail_mask: Tensor         # [B, M], True = valid
    detail_type_ids: Tensor     # [B, M]
    card_ids: Tensor            # [B]


class StaticCardFeatureAdapter(nn.Module):
    def forward(self, card_ids: Tensor) -> StaticCardEncoding:
        ...
```

对接规则：

```text
若现有静态产物已经是 128 维 card_summary/detail_tokens：直接读取并校验。
若现有静态产物是结构化数值与文本特征：实现轻量 StaticCardEncoder 映射到 128 维。
若仓库已有 CardEncoder：复用其正式接口，并增加适配层统一输出。
```

本轮只建立可训练的模型接口和稳定前向，不启动静态预训练。

## 3.3 固定静态语义

```text
card_id = 卡牌种类与规则身份
serial  = 单局内具体卡牌实例身份
```

两者始终分离。

---

# 4. 固定模型配置

增加或确认：

```yaml
model:
  card_summary_dim: 128
  detail_token_dim: 128
  dynamic_hidden_dim: 64
  instance_token_dim: 128
  detail_attention_heads: 4
  instance_ffn_dim: 256
  instance_dropout: 0.1

  state_token_dim: 128
  decision_token_dim: 128
  match_token_dim: 128
  ledger_token_dim: 128
  event_token_dim: 128

  board_model_dim: 128
  board_num_layers: 2
  board_num_heads: 4
  board_ffn_dim: 256
  board_dropout: 0.1
  board_norm: pre_norm

  max_recent_events: 16
  ledger_pooling: attention
  hand_group_by_card_id: true
  discard_group_by_card_id: true
```

配置校验：

```text
card_summary_dim == detail_token_dim == instance_token_dim
instance_token_dim == board_model_dim
board_model_dim % board_num_heads == 0
detail_token_dim % detail_attention_heads == 0
```

全局层每个局面固定产生：

```text
1 个 [MATCH]
1 个 [SELF_LEDGER]
1 个 [OPP_LEDGER]
0–16 个 [RECENT_EVENT]
```

Ledger 内部可包含任意数量的唯一 Card ID 条目，但经过 owner-wise attention pooling 后只向 Board Transformer 输出两个 Ledger Token。

# 5. 状态数据结构

实现清晰的普通 dataclass。字段名可按仓库风格调整，但职责与 shape 保持一致。

```python
@dataclass
class PlacementContext:
    owner_id: Tensor
    zone_id: Tensor
    field_role_id: Tensor
    visibility_id: Tensor
    knowledge_state_id: Tensor


@dataclass
class CardDynamicBatch:
    hp_features: Tensor
    hp_mask: Tensor
    attached_energy_counts: Tensor
    attached_energy_mask: Tensor
    condition_flags: Tensor
    condition_mask: Tensor
    local_turn_flags: Tensor
    placement: PlacementContext
    copy_count: Tensor
    field_masks: dict[str, Tensor]


@dataclass
class GlobalSnapshotBatch:
    numeric_features: Tensor
    numeric_mask: Tensor
    categorical_features: dict[str, Tensor]
    boolean_features: Tensor
    provenance: dict[str, str]


@dataclass
class MatchContextBatch:
    self_is_first_player: Tensor
    turn_number: Tensor
    format_version_id: Tensor
    ruleset_version_id: Tensor
    environment_version_id: Tensor
    masks: dict[str, Tensor]


@dataclass
class DecisionContextBatch:
    select_type: Tensor
    select_context: Tensor
    min_count: Tensor
    max_count: Tensor
    remain_damage_counter: Tensor
    remain_energy_cost: Tensor
    context_card_id: Tensor
    effect_card_id: Tensor
    masks: dict[str, Tensor]


@dataclass
class GameEvent:
    event_type: int
    actor: int
    card_id: int | None
    serial: int | None
    from_zone: int | None
    to_zone: int | None
    source_card_id: int | None
    target_card_id: int | None
    target_serial: int | None
    attack_id: int | None
    value: int | None
    turn: int
    turn_action_count: int
    identity_visible: bool


@dataclass
class CardLedgerEntry:
    relative_owner: int
    card_id: int                 # UNKNOWN 使用统一特殊 ID
    initial_count: int | None
    current_known_zone_counts: dict[int, int]
    unresolved_hidden_count: int
    seen_count: int
    used_count: int
    attach_count: int
    evolve_count: int
    attack_count: int
    discarded_count: int
    recovered_count: int
    searched_count: int
    last_changed_turn: int
    last_event_type: int
    knowledge_state: int
    confidence: float


@dataclass
class GameMemoryState:
    match_id: str | None
    perspective_player_index: int
    visible_serial_registry: dict
    self_card_ledger: dict
    opponent_card_ledger: dict
    recent_events: list[GameEvent]
    processed_observation_key: str | None
    schema_version: int
```

`GameMemoryState` 支持：

```text
reset
clone
to_dict
from_dict
```

memory 作为显式输入和输出传递，便于在线对局、replay 重建、批训练和搜索分支复制。

隐藏区域的 UNKNOWN 数量作为双方 Ledger 中的特殊条目保存，不再维护独立的模型 Token。

# 6. 当前卡牌实例动态特征

## 6.1 可见 Pokémon

从当前 observation 提取：

```text
hp
max_hp
hp_ratio
damage_count
damage_ratio
appear_this_turn
attached_energy_counts_by_type[12]
attached_energy_card_count
attached_tool_count
pre_evolution_depth
special_condition_flags
owner
zone
field_role
visibility
knowledge_state
```

特殊状态来自对应玩家的 active 状态字段，并映射到 active Pokémon。Bench Pokémon 的这组 active condition mask 为无效。

附着卡处理：

```text
Pokemon.energies → 12 类能量计数
Pokemon.energyCards → 可见附着能量 Card ID 集合摘要
Pokemon.tools → Tool Card ID 集合摘要
Pokemon.preEvolution → 进化链 Card ID 集合摘要
```

第一版可以将附着卡静态摘要做 masked mean/attention pooling，再作为动态分支输入。附着卡自身不在 Board 序列中重复展开。

## 6.2 非 Pokémon 可见卡牌

Trainer、Energy、Stadium、手牌和弃牌中的普通 Card 使用：

```text
owner
zone
field_role
visibility
knowledge_state
copy_count
```

HP、异常状态等不适用字段通过 mask 屏蔽。

## 6.3 手牌聚合

SELF 手牌按精确 Card ID 聚合：

```text
one token per unique Card ID
+ copy_count
+ serial_list
+ original_hand_indices
```

保留原始映射 metadata，后续 ActionEncoder 可映射回具体 legal option。

对手手牌只保存：

```text
hand_count
公开后仍可确认在手的 Card ID 记录
其余匿名数量写入 OPPONENT 的 UNKNOWN Ledger Entry
```

---

# 7. DynamicInstanceEncoder

实现：

```python
class DynamicInstanceEncoder(nn.Module):
    def forward(self, dynamic_batch: CardDynamicBatch) -> Tensor:
        # [B_cards, 64]
        ...
```

编码规则：

```text
HP / damage / count → 标准化数值 + validity mask
attached energy → 原始 12 维计数 + energy-type embedding 加权表示
condition / local turn flags → bool 分支
owner / zone / role / visibility / knowledge → 独立 embedding
copy_count → count embedding
附着卡集合摘要 → 独立 projection
各分支 concat → MLP → LayerNorm → [B_cards,64]
```

同一 Card ID 在不同 HP、能量、区域和公开状态下应产生不同 dynamic representation。

---

# 8. CardInstanceFusion

实现固定的动态条件 Cross-Attention：

```python
class CardInstanceFusion(nn.Module):
    def forward(
        self,
        card_summary: Tensor,      # [B_cards,128]
        detail_tokens: Tensor,     # [B_cards,M,128]
        detail_mask: Tensor,       # [B_cards,M]
        dynamic_repr: Tensor,      # [B_cards,64]
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        ...
```

数学结构固定为：

\[
\begin{aligned}
d_i &= E_{\mathrm{dynamic}}(x_i^{dyn})\\
q_i &= \operatorname{LN}(s_i+W_dd_i)\\
r_i &= \operatorname{MHA}(q_i,T_i,T_i)\\
u_i &= \operatorname{LN}(q_i+r_i)\\
h_i &= \operatorname{LN}(u_i+\operatorname{FFN}(u_i))
\end{aligned}
\]

实现细节：

```text
embed_dim = 128
heads = 4
FFN = 128 → 256 → 128
batch_first = true
每张卡只查询自己的 detail tokens
```

无有效 detail token 时：

```text
detail_context = 0
输出无 NaN
前向和反向稳定
```

输出：

```text
card_instance_token: [B_cards,128]
```

---

# 9. 当前公开局面快照 `[STATE]`

`[STATE]` 只描述当前 observation 中快速变化的公开局面，不承担长期历史与版本环境。

## 9.1 视角统一

```text
SELF = current.yourIndex
OPPONENT = 1 - current.yourIndex
```

`yourIndex` 表示当前正在做选择的玩家。

## 9.2 当前行动状态

直接读取：

```text
current.turn
current.turnActionCount
current.firstPlayer
current.yourIndex
current.supporterPlayed
current.stadiumPlayed
current.energyAttached
current.retreated
```

推导：

```text
pre_game
current_turn_owner_is_self
turn_action_count
supporter_used_this_turn
stadium_used_this_turn
manual_energy_attachment_used
retreat_used_this_turn
```

当 `turn > 0` 且 `firstPlayer` 有效时：

```text
current_turn_player = firstPlayer           if turn is odd
current_turn_player = 1 - firstPlayer       if turn is even
current_turn_owner_is_self = current_turn_player == yourIndex
```

先后手身份与绝对回合数进入 `[MATCH]`，这里仅保留当前行动相关派生值，避免重复输入。

## 9.3 双方当前资源

每方提取：

```text
hand_count
deck_count
prizes_remaining = len(prize)
discard_count = len(discard)
bench_count = len(bench)
bench_max
has_active
visible_prize_count
```

全局提取：

```text
stadium_present
looking_visible_count
looking_hidden_count
```

`current.result` 只保存在样本 metadata 中用于终局标签与终止判断，不进入模型输入。

## 9.4 编码器

实现：

```python
class GlobalSnapshotEncoder(nn.Module):
    def forward(self, batch: GlobalSnapshotBatch) -> Tensor:
        # [B,128]
        ...
```

输出 `[STATE]` token。

# 10. 当前选择上下文 `[DECISION]`

实现：

```python
class DecisionContextEncoder(nn.Module):
    def forward(self, batch: DecisionContextBatch) -> Tensor:
        # [B,128]
        ...
```

读取：

```text
select.type
select.context
select.minCount
select.maxCount
select.remainDamageCounter
select.remainEnergyCost
select.contextCard
select.effect
```

`select.option` 原样保留给后续 ActionEncoder。

`current.looking` 与 `select.deck` 中非 None 的 Card 生成当前可见实体；None 项只贡献隐藏数量与 mask。

---

# 11. 精简全局上下文

全局上下文只保留：

```text
match_state
card_ledger
recent_events
```

原先的 deck context、opponent belief、history summary、transition summary 和 UNKNOWN pool 均通过这三部分表达。

---

# 12. Match Context `[MATCH]`

实现：

```python
class MatchContextEncoder(nn.Module):
    def forward(self, batch: MatchContextBatch) -> Tensor:
        # [B,128]
        ...
```

输入固定为：

```text
self_is_first_player
turn_number
format_version_id
ruleset_version_id
environment_version_id
```

说明：

```text
self_is_first_player：从当前代理视角编码
turn_number：绝对对局回合编号
format/ruleset/environment：当前无数据时使用 UNKNOWN ID + mask
```

版本环境数据后续更新时只替换对应 ID 或 embedding 表，不修改静态卡牌编码结构。

输出 `[MATCH]` token。

---

# 13. Card Ledger：长期事实与当前认知

## 13.1 身份层次

```text
card_id：卡牌种类
serial：单局具体副本，单局内唯一
ledger_key：(relative_owner, card_id)
```

当前公开实例按 serial 对齐；长期状态按 `(owner, card_id)` 聚合；身份未知的隐藏牌统一写入该 owner 的 `UNKNOWN_CARD_ID` Ledger Entry。

## 13.2 Visible Serial Registry

为当前 observation 或公开日志中暴露的 serial 维护：

```text
serial
card_id
relative_owner
current_exact_zone
previous_exact_zone
last_seen_turn
last_seen_action_count
last_event_type
is_currently_visible
```

Registry 仅用于：

```text
当前实例对齐
生成位置变化事件
更新 Card Ledger
```

Registry 本身不直接生成 Board Token。

## 13.3 Ledger Entry

每个 `(owner, card_id)` 只保存当前长期认知：

```text
initial_count / UNKNOWN
current_known_zone_counts
unresolved_hidden_count
seen_count
used_count
attach_count
evolve_count
attack_count
discarded_count
recovered_count
searched_count
last_changed_turn
last_event_type
knowledge_state
confidence
```

这里不保存完整位置变化序列。位置变化顺序统一由 `recent_events` 表达，避免重复。

知识状态：

```text
EXACT_VISIBLE
EXACT_KNOWN_HIDDEN
AMBIGUOUS_HIDDEN
HISTORICAL_ONLY
UNKNOWN
```

初始化：

```text
SELF：从当前 deck.csv 按 Card ID 初始化总量；尚未定位的副本计入 UNKNOWN/DECK|PRIZE unresolved entry。
OPPONENT：先初始化匿名 hand/deck/prize 数量；Card ID 首次公开后建立具体 Ledger Entry。
```

己方未知副本表示“身份总量已知、具体隐藏位置未解析”；对手未知副本表示“身份与具体位置均未知”。

## 13.4 Ledger 更新

实现：

```python
new_memory = GameStateManager.update_ledgers(
    previous_memory,
    observation,
    parsed_events,
)
```

更新顺序：

```text
1. 判断新比赛并 reset
2. 扫描 logs，生成 GameEvent
3. 扫描当前 observation 的全部可见区域
4. 以 serial 对齐公开实例
5. 重建每个 Card ID 的 current_known_zone_counts
6. 更新 UNKNOWN 隐藏数量
7. 根据事件累计 used/discarded/recovered/searched 等计数
8. 处理 shuffle 导致的知识状态降级
9. 写入 processed_observation_key
```

计数单一来源：

```text
当前区域数量由 snapshot 重建
累计使用原因由 logs 更新
```

同一个 observation 重复传入时保持幂等。

## 13.5 Ledger Encoder

实现：

```python
class CardLedgerEncoder(nn.Module):
    def forward(self, ledger_batch) -> tuple[Tensor, Tensor]:
        # self_ledger_token:     [B,128]
        # opponent_ledger_token: [B,128]
        ...
```

每个 Ledger Entry 的内部表示由：

```text
card_summary / UNKNOWN embedding
current_known_zone_counts
unresolved_hidden_count
cumulative usage counts
last event + recency
knowledge state
confidence
```

组成。然后分别对 SELF 和 OPPONENT 条目做 masked attention pooling：

```text
SELF entries     → [SELF_LEDGER]
OPPONENT entries → [OPP_LEDGER]
```

`[SELF_LEDGER]` 同时表达：

```text
完整 decklist 组成
当前已知区域分布
DECK|PRIZE 未解析资源
累计使用与剩余资源情况
```

`[OPP_LEDGER]` 同时表达：

```text
已公开 Card ID
当前仍确认的位置
搜索/弃牌/回收等累计证据
匿名隐藏数量
基于已见牌形成的 deck belief
belief confidence
```

本轮不向 Board Transformer展开逐 Card ID Ledger Token。原始 Ledger 保留在输出 metadata 中，供后续 ActionEncoder、调试和可解释性工具读取。

---

# 14. Recent Events：近期顺序信息

`Observation.logs` 表示自上次选择以来发生的事件。按原始顺序标准化为 `GameEvent`，并与 memory 中已有事件拼接，只保留最近 `max_recent_events=16` 个重要事件。

优先保留：

```text
SEARCH
REVEAL
DRAW
PLAY / USE_CARD
ATTACH
EVOLVE / DEVOLVE
DISCARD
RECOVER
SHUFFLE_BACK / SHUFFLE
SWITCH / MOVE_CARD
ATTACK
KNOCK_OUT
HP_CHANGE
SPECIAL_CONDITION_CHANGE
```

公开日志可记录：

```text
card_id / serial
from_zone / to_zone
source_card
target_card
turn / turn_action_count
```

匿名日志例如：

```text
DRAW_REVERSE
MOVE_CARD_REVERSE
```

只生成 UNKNOWN identity 事件并更新 Ledger 中的匿名数量，不推断 Card ID 或 serial。

`SHUFFLE` 触发对应已知隐藏位置从：

```text
EXACT_KNOWN_HIDDEN → AMBIGUOUS_HIDDEN
```

实现：

```python
class EventEncoder(nn.Module):
    def forward(self, events, mask) -> Tensor:
        # [B,E,128]
        ...
```

每个事件 Token 由：

```text
event_type
actor
card_id / UNKNOWN
source_card_id / UNKNOWN
from_zone / to_zone
turn_delta
action_delta
visibility / knowledge state
value
```

组成。

输出：

```text
[RECENT_EVENT_1] ... [RECENT_EVENT_N]
```

空历史时不增加事件 Token，由 Board mask 表示长度为零；无需额外 empty-transition token。

# 15. 当前区域 Token

独立 card instance token：

```text
SELF active
SELF bench
OPPONENT active
OPPONENT bench
SELF hand unique Card IDs
Stadium
visible prize cards
visible looking/select cards
```

双方弃牌区：

```text
按 Card ID 聚合
→ count-aware card token
→ masked attention pooling
→ SELF_DISCARD_SUMMARY / OPP_DISCARD_SUMMARY
```

弃牌区单卡候选信息仍保存在 metadata 中，供后续 ActionEncoder 精确读取。

备战区作为集合处理。模型不使用 bench slot 的普通绝对位置编码表达策略语义；保留原始索引仅用于动作映射。

---

# 16. Board Tokenizer 与 Board Transformer

固定顺序：

```text
[STATE]
[DECISION]
[MATCH]
[SELF_LEDGER]
[OPP_LEDGER]
[RECENT_EVENT]*
[SELF_ACTIVE]?
[SELF_BENCH]*
[OPP_ACTIVE]?
[OPP_BENCH]*
[SELF_HAND_UNIQUE]*
[STADIUM]?
[VISIBLE_PRIZE]*
[VISIBLE_SELECTION_CARD]*
[SELF_DISCARD_SUMMARY]
[OPP_DISCARD_SUMMARY]
```

每个 token 加入适用的：

```text
token_kind embedding
owner embedding
zone embedding
field_role embedding
visibility embedding
knowledge_state embedding
recency embedding（事件 token）
```

Ledger 的 owner、knowledge 与 confidence 已在 Ledger Encoder 内编码；Board 层只接收两个最终 Ledger Token。

实现：

```python
class BoardTransformer(nn.Module):
    ...
```

参数：

```text
d_model = 128
layers = 2
heads = 4
ffn = 256
pre-norm
dropout = 0.1
```

输出：

```python
@dataclass
class StateEncoderOutput:
    state_embedding: Tensor                 # [B,128]
    contextualized_tokens: Tensor           # [B,N,128]
    board_mask: Tensor                      # [B,N]
    token_metadata: list
    updated_memory: GameMemoryState | list[GameMemoryState]
    recent_event_tokens: Tensor | None
    self_ledger_token: Tensor               # [B,128]
    opponent_ledger_token: Tensor           # [B,128]
```

`state_embedding` 取 `[STATE]` 位置的上下文化输出。

# 17. 对外统一接口

实现或整理：

```python
class PokemonTCGStateEncoder(nn.Module):
    def forward(
        self,
        observation_batch,
        previous_memories,
        match_contexts,
        return_attention: bool = False,
    ) -> StateEncoderOutput:
        ...
```

内部流程：

```text
observation + previous memory
→ 解析 current logs 为 recent events
→ 更新 visible serial registry 与双方 card ledger
→ 构建 GlobalSnapshot / Decision / Match batches
→ 构建 visible card dynamic batches
→ StaticCardFeatureAdapter
→ DynamicInstanceEncoder
→ CardInstanceFusion
→ Match / Ledger / Event encoders
→ BoardTokenizer
→ BoardTransformer
→ StateEncoderOutput
```

replay 离线处理额外提供：

```python
build_state_samples_from_episode(...)
```

按时间顺序重建 memory，并可缓存每个决策点的状态样本。

# 18. 隐藏信息边界

正式状态编码只使用：

```text
observation
Observation.logs
当前己方 deck.csv 的 Card ID 总量
此前公开且仍可合理追踪的信息
```

对手隐藏手牌、牌库和奖赏卡身份统一表示为 UNKNOWN 或概率/区域汇总。

需要建立测试证明：

```text
DRAW_REVERSE 不产生 Card ID
MOVE_CARD_REVERSE 不产生 serial
对手 hand 为 None 时不会读取内部真实手牌
隐藏 prize 的 None 不会被静态查表恢复
shuffle 后已知隐藏位置正确降级
```

Oracle/coach 数据可在后续训练阶段作为独立 teacher 输入，正式 agent 路径保持公开信息边界。

---

# 19. 必须完成的测试

## 19.1 静态对接

```text
1. 同 Card ID 多攻击只返回一个 card_summary
2. 多个攻击/特性返回多个 detail token
3. 静态行数不进入 copy_count
4. 未知 Card ID 使用稳定 UNKNOWN 静态表示
```

## 19.2 单卡动态与融合

```text
1. DynamicInstanceEncoder 输出 [B,64]
2. CardInstanceFusion 输出 [B,128]
3. detail padding 不参与 attention
4. 无 detail 时无 NaN
5. 同卡不同 HP 得到不同 token
6. 同卡不同能量得到不同 token
7. 同卡 ACTIVE/BENCH 得到不同 token
8. 前向、反向、保存、加载正常
```

## 19.3 `[STATE]`、`[DECISION]` 与 `[MATCH]`

```text
1. turn=1/2/3 时 current_turn_player 推导正确
2. firstPlayer=-1 时 mask 正确
3. SELF/OPPONENT 视角互换正确
4. hand/deck/prize/discard/bench 数量正确
5. result 不进入模型输入
6. select 为空时 [DECISION] 稳定
7. [MATCH] 正确编码 self_is_first_player 与 turn_number
8. 缺少版本信息时 UNKNOWN ID 与 mask 正确
```

## 19.4 Ledger 与 Recent Events

```text
1. visible serial 在 ACTIVE→BENCH 后 registry 正确更新
2. 已知 Card ID HAND→DISCARD 后当前 zone count 正确
3. ATTACH / EVOLVE / ATTACK 更新对应累计计数
4. SEARCH / REVEAL 更新 opponent ledger 的证据与 confidence
5. DRAW_REVERSE 只更新 UNKNOWN entry 并生成匿名事件
6. MOVE_CARD_REVERSE 不创建虚构身份
7. SHUFFLE 触发知识状态降级
8. 己方未解析副本保持 DECK|PRIZE unresolved
9. 同 observation 重复更新保持幂等
10. 新比赛 reset 后无旧局残留
11. replay 顺序重建与在线逐步更新一致
12. snapshot 与 logs 不产生双重计数
13. recent events 最多 16 个且保持时间顺序
14. Ledger 不保存重复的完整事件序列
15. 两侧 Ledger Encoder 各只输出一个 [B,128] token
```

## 19.5 Board 编码

```text
1. token 顺序固定为 STATE/DECISION/MATCH/SELF_LEDGER/OPP_LEDGER/EVENTS/CARDS
2. 不再出现 MEMORY/TRANSITION/UNKNOWN_POOL/CARD_MEMORY token
3. 空手牌/空弃牌区/无 Stadium/无事件稳定
4. batch 内不同 token 数可 padding
5. state_embedding 为 [B,128]
6. contextualized_tokens 为 [B,N,128]
7. CPU 前向稳定
```

# 20. CPU Benchmark

在 CPU 上记录：

```text
单 observation 状态构建时间
memory 增量更新时间
单次模型前向时间
典型 token 数 / 最大 token 数
峰值内存
参数量
序列化后权重大小
```

给出典型局面和高 token 局面两组结果。

---

# 21. 本轮验收产物

完成后提交：

```text
state_feature_audit.md
实现代码
配置文件
单元测试与 replay fixture
state_encoder_implementation_report.md
CPU benchmark 结果
```

实现报告必须列出：

```text
1. 复用的静态预处理文件和接口
2. 新增模块与文件
3. 各阶段实际 tensor shape
4. GameMemoryState schema
5. Card ID / serial / UNKNOWN 的处理方式
6. [STATE]、[MATCH]、Ledger 与 Event 的职责边界
7. SELF/OPPONENT Ledger 的聚合方式与输入字段
8. recent event 的筛选、排序与裁剪规则
9. 隐藏信息审计结果
10. 测试与 benchmark 结果
11. 后续 ActionEncoder 与训练阶段的接入接口
```

# 22. 执行顺序

按以下依赖顺序完成：

```text
A. 审计静态预处理与 cabt observation
B. 建立静态读取适配层
C. 建立 observation schema 与当前动态字段提取
D. 实现 DynamicInstanceEncoder + CardInstanceFusion
E. 实现 GlobalSnapshot + DecisionContext + MatchContext
F. 实现 GameMemoryState + visible serial registry
G. 实现双方 Card Ledger 与 UNKNOWN 特殊条目
H. 实现 Recent Event 提取与编码
I. 实现 Ledger attention pooling，只输出 SELF/OPPONENT 两个 token
J. 实现当前区域 Token、BoardTokenizer 与 BoardTransformer
K. 完成测试、replay 重建和 CPU benchmark
L. 输出实现报告
```

本轮结束时，仓库应从“仅有静态预处理”推进到：

```text
任意合法 observation
+ 显式 previous_memory
→ 当前动态卡牌表示
+ 精简全局上下文
→ 稳定的 128 维局面表示
+ updated_memory
```

全局层最终固定为：

```text
[MATCH]
[SELF_LEDGER]
[OPP_LEDGER]
[RECENT_EVENT] * N
```

保持功能完整、结构紧凑，并为后续 ActionEncoder、行为克隆、价值学习和 PPO 提供稳定接口。
