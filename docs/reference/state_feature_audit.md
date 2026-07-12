# State Feature Audit

本文件记录静态 baseline、cabt observation 字段、可见性边界和动态状态实现入口。它是字段事实参考；当前架构与进度分别见 `docs/ARCHITECTURE.md` 和 `docs/STATUS.md`。

## 1. 静态卡牌 baseline

### 数据入口

- CSV 入口：`data/card_preprocessing.py::load_csv_rows()`，优先找本地 `EN_Card_Data.csv`，否则从 `pokemon-tcg-ai-battle.zip::EN_Card_Data.csv` 读取。
- 结构化 API 入口：`data/card_preprocessing.py::load_cg_data()`，从 `outputs/cg/api.py` 或 Kaggle input 中导入 `cg.api`，读取 `all_card_data()` 和 `all_attack()`。
- 聚合入口：`data/card_preprocessing.py::build_records()`。
- 训练 dataset 入口：`data/card_dataset.py::CardDataset.from_cache()`。

### Card ID 聚合状态

当前实现已经按 `Card ID` 分组：CSV 中一张卡多行攻击会聚成一个 `CardRecord`。

聚合规则：

- `grouped[str(row.get("Card ID", "")).strip()].append(row)` 先按 Card ID 分组。
- 每个 Card ID 输出一个 `CardRecord`。
- CSV 攻击行先生成 `AttackRecord`。
- 如果 cg 结构化数据存在且 `cg_card.attacks` 非空，则使用 cg attack 列表重建攻击，并按 `attack_id` 去重。
- 每个攻击保留绑定：`attack_id/name/effect_text/damage_raw/damage_value/damage_mode/energy_costs`。
- 每个技能生成 `AbilityRecord`。
- Trainer / Tool / Stadium / Special Energy / rule flags 生成 `EffectRecord`。

结论：同一 ID 的多行 CSV 不会作为多张卡进入静态数据；后续动态状态应以 `card_id` 查询同一个静态 summary/detail。

### 静态数据结构

`CardRecord` 当前包含：

- 基础字段：`card_id`, `name`, `card_type`, `subtype`, `pokemon_type`, `stage`, `hp`, `retreat_cost`, `weakness_type`, `weakness_value`, `resistance_type`, `resistance_value`, `evolves_from`, `rule_flags`, `trainer_type`, `provided_energy_types`, `full_effect_text`。
- 旧兼容字段：`attack_names`, `attack_texts`, `attack_damage`, `attack_energy_costs`, `ability_texts`, `attack_ids`。
- 新细粒度字段：`attacks`, `abilities`, `special_effects`。

`outputs/card_pretrain/artifacts/card_data/card_feature_schema.json`：

- `schema_version = static_detail_v1`
- `numeric_fields = ['hp', 'retreat_cost', 'weakness_value', 'resistance_value', 'attack_count', 'ability_count', 'special_effect_count']`

已从主要静态输入移除：

- `mean_attack_damage`
- `mean_attack_energy_cost`
- `text_length`
- 整卡级 `energy_cost_*`

### 静态 encoder 输出

`models/card_encoder.py` 当前提供：

- 旧接口：`card_encoder(batch)` -> `[B, 128]`
- 新接口：`card_encoder(batch, return_details=True)` -> `CardEncoderOutput`

`CardEncoderOutput`：

- `card_summary`: `[B, 128]`
- `detail_tokens`: `[B, max_detail_count, 128]`
- `detail_mask`: `[B, max_detail_count]`
- `detail_type_ids`: `[B, max_detail_count]`

`detail_type_ids`：

- `0 = padding`
- `1 = attack`
- `2 = ability`
- `3 = special_effect`

最新导出 metadata：

- `card_count = 1267`
- `embedding_dim = 128`
- `detail_token_dim = 128`
- `max_detail_count = 7`

Detail 数量统计：

- 攻击数：平均 `1.232`，最大 `2`，非空卡 `1062`
- 特性数：平均 `0.336`，最大 `2`，非空卡 `421`
- 特殊效果数：平均 `0.448`，最大 `3`，非空卡 `333`

样例：

- `card_id=21`, `Scrafty`
  - attack 1: `attack_id=1`, `Nab ’n’ Dash`, cost `{'C': 1}`, damage `0`, text 已绑定。
  - attack 2: `attack_id=2`, `High Jump Kick`, cost `{'C': 2, 'D': 1}`, damage `100`, text 已绑定。
- `card_id=1`, `Basic {G} Energy`
  - `card_type=BASIC_ENERGY`, `provided_energy_types=['G']`, 无 detail token。
- `card_id=9`, `Boomerang Energy`
  - `card_type=SPECIAL_ENERGY`, `special_effect.source_type=special_energy`。

## 2. 当前旧 agent 状态特征

旧训练/提交代码位置：

- `kaggle_training/train_agent.py::state_features()`
- `kaggle_training/train_agent.py::_state_features()`
- `kaggle_kernel/train_agent.py`
- `kaggle_submission/train_agent.py`

当前做法是 24 维全局状态 + 24 维候选动作，最终截断到 `FEATURE_DIM`。主要包括：

- turn / turnActionCount
- supporterPlayed / stadiumPlayed / energyAttached / retreated
- yourIndex / firstPlayer
- 双方 active 数、bench 数、deckCount、prize 数、handCount、discard 数
- select type/context/minCount/maxCount
- option type/number/index/inPlayIndex/attackId/cardId

问题：

- 没有 card instance token。
- active/bench 的 HP、能量、工具、进化链、特殊状态没有进入卡牌实例表达。
- hand/discard/prize/looking 没有统一 visibility mask。
- logs 只被规则选择隐式使用，没有转成可学习 temporal memory。
- 不能表达“同一卡牌静态信息 + 当前场上状态 + 出场/记忆特征”的分层结构。

结论：旧 `state_features` 可以保留为兼容 baseline，但不应作为新动态模块的主结构。

## 3. cabt Observation 实际字段

来源：`outputs/cg/api.py`。

### Enums

`AreaType`：

- `DECK=1`, `HAND=2`, `DISCARD=3`, `ACTIVE=4`, `BENCH=5`, `PRIZE=6`, `STADIUM=7`, `ENERGY=8`, `TOOL=9`, `PRE_EVOLUTION=10`, `PLAYER=11`, `LOOKING=12`

`EnergyType`：

- `COLORLESS=0`, `GRASS=1`, `FIRE=2`, `WATER=3`, `LIGHTNING=4`, `PSYCHIC=5`, `FIGHTING=6`, `DARKNESS=7`, `METAL=8`, `DRAGON=9`, `RAINBOW=10`, `TEAM_ROCKET=11`

`CardType`：

- `POKEMON=0`, `ITEM=1`, `TOOL=2`, `SUPPORTER=3`, `STADIUM=4`, `BASIC_ENERGY=5`, `SPECIAL_ENERGY=6`

`SpecialConditionType`：

- `POISON=0`, `BURN=1`, `SLEEP=2`, `PARALYZE=3`, `CONFUSE=4`

`SelectType`：

- `MAIN=0`, `CARD=1`, `ATTACHED_CARD=2`, `CARD_OR_ATTACHED_CARD=3`, `ENERGY=4`, `SKILL=5`, `ATTACK=6`, `EVOLVE=7`, `COUNT=8`, `YES_NO=9`, `SPECIAL_CONDITION=10`

`OptionType`：

- `NUMBER=0`, `YES=1`, `NO=2`, `CARD=3`, `TOOL_CARD=4`, `ENERGY_CARD=5`, `ENERGY=6`, `PLAY=7`, `ATTACH=8`, `EVOLVE=9`, `ABILITY=10`, `DISCARD=11`, `RETREAT=12`, `ATTACK=13`, `END=14`, `SKILL=15`, `SPECIAL_CONDITION=16`

`LogType`：

- `SHUFFLE=0`, `HAS_BASIC_POKEMON=1`, `TURN_START=2`, `TURN_END=3`, `DRAW=4`, `DRAW_REVERSE=5`, `MOVE_CARD=6`, `MOVE_CARD_REVERSE=7`, `SWITCH=8`, `CHANGE=9`, `PLAY=10`, `ATTACH=11`, `EVOLVE=12`, `DEVOLVE=13`, `MOVE_ATTACHED=14`, `ATTACK=15`, `HP_CHANGE=16`, `POISONED=17`, `BURNED=18`, `ASLEEP=19`, `PARALYZED=20`, `CONFUSED=21`, `COIN=22`, `RESULT=23`

### Dataclasses

`Card`：

- `id`, `serial`, `playerIndex`

`Pokemon`：

- `id`, `serial`, `hp`, `maxHp`, `appearThisTurn`, `energies`, `energyCards`, `tools`, `preEvolution`

`PlayerState`：

- `active`, `bench`, `benchMax`, `deckCount`, `discard`, `prize`, `handCount`, `hand`, `poisoned`, `burned`, `asleep`, `paralyzed`, `confused`

`State`：

- `turn`, `turnActionCount`, `yourIndex`, `firstPlayer`, `supporterPlayed`, `stadiumPlayed`, `energyAttached`, `retreated`, `result`, `stadium`, `looking`, `players`

`Option`：

- `type`, `number`, `area`, `index`, `playerIndex`, `toolIndex`, `energyIndex`, `count`, `inPlayArea`, `inPlayIndex`, `attackId`, `cardId`, `serial`, `specialConditionType`

`SelectData`：

- `type`, `context`, `minCount`, `maxCount`, `remainDamageCounter`, `remainEnergyCost`, `option`, `deck`, `contextCard`, `effect`

`Log`：

- `type`, `playerIndex`, `hasBasicPokemon`, `cardId`, `serial`, `fromArea`, `toArea`, `cardIdActive`, `serialActive`, `cardIdBench`, `serialBench`, `cardIdBefore`, `serialBefore`, `cardIdAfter`, `serialAfter`, `cardIdTarget`, `serialTarget`, `attackId`, `value`, `putDamageCounter`, `isRecover`, `head`, `result`, `reason`

`Observation`：

- `select`, `logs`, `current`, `search_begin_input`

## 4. 可见性与 None 边界

必须保留以下 mask，不允许用默认 ID 填充成“真实可见卡”：

- `Observation.current`：初始 deck selection 时可能为 `None`。
- `Observation.select`：初始 deck selection 时可能为 `None`。
- `PlayerState.hand`：对手手牌为 `None`，只能使用 `handCount` 和 memory 推断。
- `PlayerState.prize`：元素可能为 `None`，代表面朝下奖赏卡。
- `PlayerState.active`：长度 0 或 1，元素可能为 `None`，代表 face-down。
- `State.looking`：可能为 `None`；内部元素也可能为 `None`。
- `SelectData.deck`：只有选 deck 中卡时非空。
- reverse logs：如 `DRAW_REVERSE`、`MOVE_CARD_REVERSE` 不包含 card identity/serial，只能更新数量或 unknown-card memory。

后续动态 schema 需要显式区分：

- visible card id/serial
- known hidden card id/serial
- unknown hidden slot
- public count only
- not applicable / padding

## 5. 动态状态模块建议边界

后续实现应按四类特征组织：

1. 静态卡牌特征
   - 来源：`card_id -> card_summary/detail_tokens`。
   - 不包含局面状态。

2. 场上状态动态特征
   - 绑定到具体 card instance / Pokemon instance。
   - 例：area、owner、controller、active/bench/stadium/discard/hand/attached、slot、hp/maxHp/damage、special conditions、energies、energyCards、tools、preEvolution、appearThisTurn。

3. 出场/记忆特征
   - 绑定 serial 或 known card identity。
   - 例：是否已出现、首次可见 turn、最近可见 turn、从哪里到哪里、是否已打出/进 discard/作为 energy attached、对手已暴露卡组组成、热门度/先验占位。
   - 热门度和组合先验先预留维度，不在本轮从外部数据训练。

4. 全局状态特征
   - 绑定到 snapshot，而不是单卡。
   - 例：turn、action count、first player、your index、supporter/stadium/energy/retreat flags、双方 deck/hand/prize/bench counts、select type/context/count、remaining damage/energy cost、result。

## 6. 实现入口建议

新增文件建议：

- `data/state_schema.py`
  - enum 映射、维度常量、mask 约定。
- `data/observation_parser.py`
  - `Observation -> GlobalSnapshot + CardDynamicBatch + zones + events`。
- `data/game_memory.py`
  - `GameMemoryState`、serial registry、visible/hidden ledger、recent events。
- `models/dynamic_instance_encoder.py`
  - `DynamicInstanceEncoder`。
- `models/card_instance_fusion.py`
  - `CardInstanceFusion`，融合 static summary/detail context + dynamic + appearance。
- `models/board_tokenizer.py`
  - active/bench/hand/discard/prize/stadium/option tokens。
- `models/board_transformer.py`
  - Board-level token encoder。

兼容策略：

- 保留 `models/card_instance_encoder.py` 现有 generic wrapper，避免旧测试断裂。
- 新模块不要改动静态 card pretrain/export 默认行为。
- 先实现纯 Python parser + tensor collate，再接模型。

## 7. 测试重点

第一批测试应覆盖：

- 同一 Card ID 多攻击仍映射到一个静态 `CardRecord`。
- observation 中 visible Pokemon 转成 card instance 时保留 `id/serial/playerIndex/hp/maxHp/energies/tools/preEvolution/appearThisTurn`。
- opponent hand 为 `None` 时只产生 count，不泄露 card id。
- face-down active/prize/looking 产生 unknown hidden token 和 mask。
- reverse logs 不制造可见 card id。
- MOVE_CARD / DRAW / PLAY / ATTACH / EVOLVE / ATTACK / HP_CHANGE / special condition logs 更新 memory。
- `GameMemoryState` 在连续 observations 中保留 serial registry 和 recent events。
- dynamic encoder 输出维度稳定，padding 不参与 pooling/attention。
- 旧 `CardInstanceEncoder` 测试继续通过。

## 8. 当前结论

静态 baseline 已满足动态模块输入条件：一张 Card ID 一个静态 record，128 维 summary 可用，攻击/特性/特殊效果 detail token 可用，攻击费用-伤害-文字绑定已保留。

动态模块尚未实现。下一步应先写 observation parser 和 state schema，把 cabt observation 可靠转成可见性明确的中间结构，再接 `DynamicInstanceEncoder` 和 board tokenizer。
