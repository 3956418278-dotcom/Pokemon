# cabt 状态字段审计

本文件只记录已经从源码或 replay 验证的字段事实。模型设计见 `docs/ARCHITECTURE.md`，项目进度见 `docs/STATUS.md`。

## 1. 已退出主线的历史静态 baseline

以下路径是历史记录，均已退出主线并从当前工作树删除：

- `data/card_preprocessing.py`：读取 CSV 与 `cg.api`，按 Card ID 聚合 `CardRecord`。
- `data/card_dataset.py`：构造 schema、detail batch 和 Card ID split。
- `models/card_encoder.py`：输出静态 summary 与 detail tokens。
- `training/pretrain_card_encoder.py`：静态辅助训练。

同一 Card ID 的 CSV 多行表示同一张卡的多个技能条目。聚合后：

- 每个 Card ID 只有一个 `CardRecord`。
- 每个攻击保留 `attack_id/name/effect/damage/energy_costs` 绑定。
- Ability 和特殊效果分别保留。

成功 artifacts：

```text
card_summary:     [1267, 128]
detail_tokens:    [1267, 7, 128]
detail_mask:      [1267, 7]
detail_type_ids:  [1267, 7]
```

Detail type：`0=padding`、`1=attack`、`2=ability`、`3=special_effect`。

## 2. cabt 枚举

### AreaType

```text
DECK=1, HAND=2, DISCARD=3, ACTIVE=4, BENCH=5, PRIZE=6,
STADIUM=7, ENERGY=8, TOOL=9, PRE_EVOLUTION=10,
PLAYER=11, LOOKING=12
```

### EnergyType

```text
COLORLESS=0, GRASS=1, FIRE=2, WATER=3, LIGHTNING=4,
PSYCHIC=5, FIGHTING=6, DARKNESS=7, METAL=8,
DRAGON=9, RAINBOW=10, TEAM_ROCKET=11
```

### SelectType

```text
MAIN=0, CARD=1, ATTACHED_CARD=2, CARD_OR_ATTACHED_CARD=3,
ENERGY=4, SKILL=5, ATTACK=6, EVOLVE=7, COUNT=8,
YES_NO=9, SPECIAL_CONDITION=10
```

### OptionType

```text
NUMBER=0, YES=1, NO=2, CARD=3, TOOL_CARD=4,
ENERGY_CARD=5, ENERGY=6, PLAY=7, ATTACH=8, EVOLVE=9,
ABILITY=10, DISCARD=11, RETREAT=12, ATTACK=13,
END=14, SKILL=15, SPECIAL_CONDITION=16
```

### LogType

```text
SHUFFLE=0, HAS_BASIC_POKEMON=1, TURN_START=2, TURN_END=3,
DRAW=4, DRAW_REVERSE=5, MOVE_CARD=6, MOVE_CARD_REVERSE=7,
SWITCH=8, CHANGE=9, PLAY=10, ATTACH=11, EVOLVE=12,
DEVOLVE=13, MOVE_ATTACHED=14, ATTACK=15, HP_CHANGE=16,
POISONED=17, BURNED=18, ASLEEP=19, PARALYZED=20,
CONFUSED=21, COIN=22, RESULT=23
```

## 3. Observation 对象

```text
Observation.select
Observation.logs
Observation.current
Observation.search_begin_input
```

### State

```text
turn, turnActionCount, yourIndex, firstPlayer,
supporterPlayed, stadiumPlayed, energyAttached, retreated,
result, stadium, looking, players
```

### PlayerState

```text
active, bench, benchMax, deckCount, discard, prize,
handCount, hand,
poisoned, burned, asleep, paralyzed, confused
```

### Card / Pokemon

```text
Card: id, serial, playerIndex

Pokemon: id, serial, hp, maxHp, appearThisTurn,
energies, energyCards, tools, preEvolution
```

### SelectData / Option

```text
SelectData: type, context, minCount, maxCount,
remainDamageCounter, remainEnergyCost,
option, deck, contextCard, effect

Option: type, number, area, index, playerIndex,
toolIndex, energyIndex, count,
inPlayArea, inPlayIndex, attackId,
cardId, serial, specialConditionType
```

### Log

```text
type, playerIndex, hasBasicPokemon,
cardId, serial, fromArea, toArea,
cardIdActive, serialActive,
cardIdBench, serialBench,
cardIdBefore, serialBefore,
cardIdAfter, serialAfter,
cardIdTarget, serialTarget,
attackId, value, putDamageCounter,
isRecover, head, result, reason
```

## 4. 可见性边界

- `current` 和 `select` 在初始 deck selection 时可能为 `None`。
- 对手 `hand` 为 `None`，只可使用 `handCount`。
- `prize`、`active` 和 `looking` 内部可能包含 `None`，表示未知身份。
- `select.deck` 只在查看牌库选择时出现。
- `DRAW_REVERSE` 与 `MOVE_CARD_REVERSE` 不提供可用 Card ID/serial。

状态结构需要区分：

```text
exact visible card
known hidden card
anonymous hidden slot
public count only
not applicable / padding
```

## 5. Replay 事实

- episode index 提供每日 Dataset manifest，不直接包含 replay。
- 每日 Dataset 内为普通 JSON replay 文件。
- 每局 `steps` 长度不同；每步包含双方 agent entry。
- 决策训练样本来自 `observation.select` 非空的 entry。
- 训练/验证按完整 episode 和日期划分。

已保存的最小探测结果见 `docs/reference/replay_structure_probe_v13.json`。正式覆盖率使用 `scripts/audit_replay_features.py` 在多局 decision samples 上重新统计。
