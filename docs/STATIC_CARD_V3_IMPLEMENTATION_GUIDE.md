# Static Card Encoder v3 实施指南

## 1. 文档用途

本指南定义固定卡池的单卡静态特征预处理、预训练、导出和验收流程。它是当前 static-v3 工作的统一实施依据，面向没有参与前期讨论的开发者。

目标产物是：

```text
每张卡一个 128 维 base_card_summary
每条 Attack / Ability / CardEffect 一个 128 维 independent_detail_token
完整、可审计的卡牌、detail、文本引用和跨卡关系 metadata
```

当前工作只覆盖静态卡牌编码。动态局面、当前 HP、zone、board role、合法动作和规则生效条件由后续动态状态与规则模块处理。

---

## 2. 总体数据流

```text
固定 CSV 卡池
+ configs/text_reference_overrides.jsonl
+ CG 数据（严格 attack/card 对齐）
        │
        ▼
现有 v3 parser
        │
        ▼
artifacts/card_data_v3/
  cards.json
  details.json
  detail_offsets.json
  card_id_to_index.json
  preprocess_manifest.json
        │
        ├── 完整卡池 SentencePiece 拟合并冻结
        ├── connected-component split
        └── DataLoader batch 内临时 padding
                │
                ▼
CardEncoder
  15 个 card field slots
  Attack / Ability / CardEffect encoders
  Card Transformer
                │
                ▼
五项静态预训练目标
                │
                ▼
selection → test once → production refit
                │
                ▼
正式 embedding、metadata、关系文件和 manifest
```

不得通过搜索同名 JSON 自动选择输入。训练只读取显式配置的 canonical v3 cache。

---

## 3. 文件与职责

### 3.1 Canonical 预处理缓存

唯一 canonical cache：

```text
artifacts/card_data_v3/
```

只包含现有五个核心文件：

```text
cards.json
details.json
detail_offsets.json
card_id_to_index.json
preprocess_manifest.json
```

`artifacts/card_data_v2/`、旧 outputs 和 Kaggle 下载目录均为只读历史产物，不能作为 v3 训练输入，也不能被 v3 覆盖。

### 3.2 人工引用修正规则

```text
configs/text_reference_overrides.jsonl
```

这是 parser 的源代码管理输入，只记录已经人工确认的文本歧义。它不能从 `details.json` 反向生成。

### 3.3 现有代码入口

| 阶段 | 文件 | 主要职责 |
| --- | --- | --- |
| 预处理 | `data/card_preprocessing.py` | CSV 读取、CardRecord/detail 构造、CG 对齐、引用解析、canonical cache 写入与验证 |
| Tensor 化 | `data/card_dataset.py` | vocab、SentencePiece 绑定、Dataset、batch 临时 padding、关系矩阵 |
| 模型 | `models/card_encoder.py` | card fields、三个 DetailEncoder、Card Transformer、128 维输出 |
| 预训练 heads | `models/card_pretrain_heads.py` | 五项任务的预测 heads、loss 和指标 |
| 训练 | `training/pretrain_card_encoder.py` | masking、split、tiny overfit、selection、test、production refit |
| 导出 | `training/export_card_embeddings.py` | 正式 embedding、metadata、mapping 和 manifest |
| 验收 | `training/validate_card_artifacts.py` | canonical/正式产物完整性验证 |
| 诊断 | `training/evaluate_card_embeddings.py` | probes、baselines 和只读诊断 |
| 云端打包 | `scripts/build_kaggle_card_pretrain_formal_kernel.py` | 将上述现有模块打包进单个 formal kernel |

不得再建立平行 parser、平行 Dataset、`CardEncoderV3` 或另一套训练/导出脚本。

---

## 4. CSV 到 canonical 数据

### 4.1 聚合原则

```text
一张 Card ID → 一个 CardRecord
一条 CSV Move 数据记录 → 一个 detail
```

永久数据使用扁平 detail 表：

```text
cards
details
detail_offsets
```

不在磁盘缓存中预先 padding。

### 4.2 17 列的最终去向

| CSV 列 | Canonical 处理 | 模型用途 |
| --- | --- | --- |
| `Card ID` | 卡主键、聚合、split、mapping | 不作为语义特征 |
| `Card Name` | `card_name_raw/normalized/id` | `card_name_id` 是卡牌身份输入；不做 exact-name recovery |
| `Expansion` | 原样 metadata | 不输入模型 |
| `Collection No.` | 原样 metadata | 不输入模型 |
| `Stage (Pokémon)/Type (Energy and Trainer)` | 结合 Category 解析为 `printed_class` | 类别 embedding |
| `Rule` | `rule`、`rule_family` | 两个正式字段 token |
| `Category` | `card_kind` 与 `printed_class` 的解析依据 | 决定 Pokémon/Trainer/Energy 分支 |
| `Previous stage` | Pokémon 的 `evolves_from_card_name`；Fossil 的显式 evolve-to 目标 | 卡名关系 token、split 和 evolution edges |
| `HP` | `printed_hp` | 归一化数值 + MLP；Fossil 通过 applicability 保留 |
| `Type` | Pokémon 属性或 Energy 的印刷供能符号数量 | Pokémon type embedding 或 13-symbol Energy profile |
| `Weakness` | `weakness_type` | Pokémon 字段 embedding |
| `Resistance (Type)` | `resistance_type` | Pokémon 字段 embedding |
| `Retreat` | 印刷撤退费用 | 归一化数值 + MLP |
| `Move Name` | detail 的 raw/normalized/id | 仅对齐、引用、查询和审计；不输入模型 |
| `Cost` | 12-symbol attack cost counts + `cost_mode` | AttackEncoder 离散 count embeddings |
| `Damage` | `damage_value + damage_mode` | AttackEncoder 离散 embeddings |
| `Effect Explanation` | raw、canonical text、结构引用与模型 token 序列 | 规则文本编码 |

### 4.3 CardRecord 正式字段

模型使用的 15 个 card field slots 顺序固定为：

```text
card_name
card_kind
printed_class
rule
rule_family
category_family
category_qualifier
evolves_from
evolves_to
hp
pokemon_type
energy_printed_type
weakness
resistance
retreat
```

以下兼容字段可以继续存在于 JSON，但不能生成模型 token、恢复目标或关系边：

```text
rule_flags
card_tags
hp_applicability
provided_energy_types
previous_species
```

当前版本不生成 current-card `species` 或任何 `species_id`。

### 4.4 字段适用性

不能把不适用字段伪造为默认值后输入统一 MLP。每个 card field 必须有 applicability mask。

典型规则：

```text
Pokémon
  HP / Type / Weakness / Resistance / Retreat 可适用

Trainer
  不自动生成 Pokémon 字段
  但若卡面确有 HP 等字段，按实际 applicability 保留

Energy
  使用印刷 Energy profile
  不生成 Pokémon HP/Type/Weakness 等伪字段
```

### 4.5 Fossil

五张 Fossil 保持：

```text
printed_class = ITEM
category_family = FOSSIL
printed_hp = 60
pokemon_type = null
weakness_type = null
resistance_type = null
retreat = null
```

它们的 `[Ability]` 行正式分类为 `ABILITY`。

“作为 Basic Colorless Pokémon 使用”“没有 Weakness”“不能撤退”等规则由同卡 CardEffect 文本表达，不伪造卡级 `C / NONE / FORBIDDEN` 字段。

静态编码通过以下信息共同表达 Fossil：

```text
Fossil card fields
+ Fossil CardEffectDetail
+ Fossil AbilityDetail
```

实际对局中是否作为 Pokémon 在场、Ability 是否生效和能否撤退，由动态状态与规则模块判断。

### 4.6 Core Memory / Geobuster

detail 分类优先级固定为：

```text
[Ability] 前缀有效
→ ABILITY

否则 Cost 或 Damage 有效
→ ATTACK

否则 Effect Explanation 有效
→ CARD_EFFECT
```

非 Pokémon 来源卡出现 Attack 时，父卡必须属于 `TECHNICAL_MACHINE`，否则预处理失败并报告 Card ID、source row 和原始字段。

Core Memory 的 Geobuster 是合法 `AttackDetail`。

### 4.7 ACE SPEC

```text
rule = ACE_SPEC
rule_family = ACE_SPEC
```

兼容 `rule_flags` 可以包含 `ACE_SPEC`，但不重复输入模型。

---

## 5. Detail canonical 结构

### 5.1 Detail 类型

只存在三种正式数据类型：

```text
ATTACK
ABILITY
CARD_EFFECT
```

不保存或训练 `detail_subtype`。下列展示标签只能根据 detail 类型和父卡字段即时派生：

```text
Technical Machine Attack
Fossil Ability
Item Effect
Tool Effect
Stadium Effect
Special Energy Effect
Tera Rule
```

### 5.2 Detail name

永久保存：

```text
name_raw
name_normalized
name_id
```

规范化包括：

```text
Unicode 统一
连续空白统一
撇号统一
[Ability] 类型前缀移除
[Tera] 与裸 Tera 在语义一致时统一为 Tera
```

Detail name 用于 CG 对齐、同名攻击查询、名称引用、跨卡关系和人工审计。

禁止：

```text
送入 SentencePiece
创建 detail-name embedding
直接加入 Detail Transformer 序列
```

### 5.3 AttackDetail

固定 12-symbol Cost 顺序：

```text
C,G,R,W,L,P,F,D,M,N,Y,A
```

每个符号保存非负整数 count。

Cost 模式：

```text
No cost       → EXPLICIT_ZERO + 全零 count vector
字段缺失      → NOT_APPLICABLE + 全零 count vector
正常费用      → COUNTS
```

Attack 不允许 `NOT_APPLICABLE` Cost。

Damage：

```text
30    → value=30,  mode=FIXED
30+   → value=30,  mode=PLUS
20×   → value=20,  mode=MULTIPLY
-120  → value=120, mode=MINUS
空值  → value=null, mode=NONE
```

未知 Damage 格式必须失败，不能猜测。

### 5.4 AbilityDetail

模型输入：

```text
ABILITY typed start
+ canonical rule text tokens
+ structure-reference tokens
```

### 5.5 CardEffectDetail

Trainer、Tool、Stadium、Energy 和 Pokémon 卡牌级规则均使用 `CARD_EFFECT`。

模型输入：

```text
CARD_EFFECT typed start
+ canonical rule text tokens
+ structure-reference tokens
```

来源卡类型由父卡 summary 提供，不在 independent detail 内重复编码 subtype。

### 5.6 Raw、canonical 和 fingerprint

必须分开保存：

```text
effect_text_raw
  CSV 正确解码后的原始字段，仅审计

effect_text
  经显式、可追溯修正后的 canonical 模型文本

model_text_tokens
  effect_text 中普通文本与结构引用替换后的模型序列
```

任何 CSV→canonical 文本修正必须保存显式 override 信息，不能静默替换。

`detail_fingerprint` 基于模型实际可见内容：

```text
detail_type
cost_counts
damage_value
damage_mode
model_text_tokens
模型可见的结构引用类型/字段值
```

不包含：

```text
Card ID
source row
detail name
reference_id
source span
精确目标 Card ID / Detail ID
```

使用固定 canonical JSON serialization 后计算 SHA-256。

---

## 6. 文本结构引用

### 6.1 解析顺序

固定卡池使用一次性全量扫描：

1. 先识别明确的卡牌字段短语。
2. 建立完整 card-name/detail-name 目录。
3. 只在实际 card-detail 所属关系得到证明时生成名称关系引用。
4. 应用 `configs/text_reference_overrides.jsonl` 中的人工确认。
5. 无法可靠确认的候选保留普通文本，并写入 unresolved audit；不自动猜测。

### 6.2 引用表示

metadata 中每个引用必须同时具有：

```text
reference_type
reference_id
model_source_span
raw_source_span（可准确对应时）
payload
```

`reference_id` 只用于 metadata 精确查询：

```text
不建立 embedding
不进入 token sequence
不进入 loss
不作为 attention 数值
```

模型读取：

```text
引用关系类型
字段 selector 的 canonical field/value
```

跨卡引用的精确目标只保存在 metadata 中。

### 6.3 Same-card 引用

DataLoader 执行：

```text
reference_id
→ 查询 metadata
→ source_global_detail_index → target_global_detail_index
→ 构造 SAME_CARD_DETAIL_REF 布尔边
```

Card Transformer 只使用一个 `SAME_CARD_DETAIL_REF` relation bias。

### 6.4 Cross-card 引用

跨卡 metadata 至少保存：

```text
reference_id
source_card_id
source_global_detail_index
model_source_span
reference_type
target_card_name_id
target_detail_name_id
target_detail_type
matching_target_card_ids
matching_target_global_detail_indices
resolution_status
```

基础 CardEncoder 不做全卡池图传播，也不编码具体 target ID。

未来 `CrossCardRelationEncoder` 使用：

```text
source detail token
target detail token
target card summary
relation type
```

生成独立的 `card_relation_context`。

### 6.5 Evolution 关系

`cross_card_references.jsonl` 只保存 Detail 文本产生的跨卡引用。

Evolution edges 只来自：

```text
evolves_from_card_name
evolves_to_card_names
```

`previous_species` 不参与 edge、split 或模型。

---

## 7. Canonical 索引和保存规则

### 7.1 Detail 全局索引

所有正式 detail index 都是 `details.json` 的全局扁平行号，并与以下产物严格对齐：

```text
details.json
independent_detail_tokens.npy
```

跨卡字段名固定为：

```text
matching_target_global_detail_indices
```

卡内位置只用于调试：

```text
detail_position_within_card
```

### 7.2 detail_offsets

```text
detail_offsets.shape = [card_count + 1]
detail_offsets[0] = 0
detail_offsets[-1] = total_detail_count
```

每张卡的 details：

```text
details[detail_offsets[i]:detail_offsets[i+1]]
```

### 7.3 两种 card_id_to_index

预处理 mapping：

```text
mapping_role = PREPROCESS_CARD_RECORD_ROW
Card ID → canonical cards.json 行号
```

正式导出 mapping：

```text
mapping_role = EMBEDDING_TENSOR_ROW
Card ID → base_card_summary.npy 行号
source_preprocess_mapping_sha256 = canonical mapping hash
```

两份内容当前可以字节相同，但职责和 lineage 必须分开记录。

### 7.4 Cache 写入

写 canonical cache 时：

1. 在同一父目录创建 staging directory。
2. 写入并验证全部五个文件。
3. 原子替换目标文件。
4. manifest 最后生效。
5. 拒绝混入额外同名/未知文件。

---

## 8. SentencePiece

使用完整固定卡池的 canonical 普通文本拟合一次，之后冻结：

```yaml
model_type: unigram
vocab_size: 1024
hard_vocab_limit: false
character_coverage: 1.0
byte_fallback: false
pad_id: 0
unk_id: 1
bos_id: -1
eos_id: -1
user_defined_symbols:
  - "[MASK_TEXT]"
input_sentence_size: 0
shuffle_input_sentence: false
max_text_subword_tokens: 256
truncation: false
```

结构引用 token 不进入 SentencePiece vocab。

训练后必须扫描完整语料。任何 detail 超过 256 个混合文本/引用位置时，输出 Card ID、global detail index 和实际长度，然后提高配置；禁止静默截断。

Selection、test 和 production refit 必须使用完全相同的 tokenizer bytes 和 hash。

---

## 9. Batch Tensor 合同

永久缓存无 padding。`collate_cards()` 仅在当前 batch 内补齐：

```text
card_field_value_ids                [B, 15]
card_field_applicability_mask       [B, 15]
card_numeric_values                 [B, 2]      # HP, Retreat normalized
evolves_to_name_ids                 [B, E]
evolves_to_name_mask                [B, E]
provided_energy_count_ids           [B, 13]

detail_mask                         [B, D]
detail_type_ids                     [B, D]
attack_mask                         [B, D]
attack_energy_count_ids             [B, D, 12]
attack_damage_value_ids             [B, D]
attack_damage_mode_ids              [B, D]

detail_text_ids                     [B, D, T]
detail_text_token_kind_ids          [B, D, T]
detail_text_token_mask              [B, D, T]
detail_plain_text_mask              [B, D, T]
detail_structure_reference_mask     [B, D, T]
detail_structure_reference_type_ids [B, D, T]
detail_structure_reference_field_ids[B, D, T]
detail_structure_reference_value_ids[B, D, T]

same_card_detail_reference_matrix   [B, D, D]
detail_fingerprint_ids              [B, D]
detail_global_indices               [B, D]
```

数字 `reference_id`、detail name、detail subtype 和 species 不是模型 tensor。

---

## 10. CardEncoder 模型

### 10.1 卡级字段编码

类别字段使用 embedding。

`HP` 和 `Retreat`：

```text
印刷数值
→ 使用完整卡池固定 scale 归一化
→ 各自的数值 MLP
→ 128 维 field token
```

恢复任务中的离散 CE target 与输入的数值 MLP 不冲突；前者是预测目标，后者是静态输入编码。

`evolves_from` 和 `evolves_to` 使用 card-name vocab。多个 evolves-to 目标先分别 lookup，再 masked-pool 成一个 slot token。

### 10.2 Energy printed profile

固定符号顺序：

```text
C,G,R,W,L,P,F,D,M,N,Y,A,TEAM_ROCKET
```

每个符号拥有独立 count embedding table：

```text
energy_count_embedding[symbol][count]
```

13 个 lookup 结果聚合、LayerNorm，再加 `ENERGY_PRINTED_PROFILE` slot embedding，得到一个 128 维 token。

具体条件下实际提供什么 Energy 继续由 CardEffect 文本表达。

### 10.3 三个卡类别分支

根据原始、未被 masking 改写的 card kind 路由：

```text
Pokémon branch
Trainer branch
Energy branch
```

每个分支只聚合 applicability 为真的字段。

输出：

```text
base_card_token [B, 128]
```

### 10.4 RuleTextEncoder

每个 detail 的模型文本序列只包含：

```text
普通 SentencePiece token
结构引用 token
```

普通文本位置使用 text embedding；结构引用位置使用 reference type、field 和 canonical value embedding。numeric reference ID 不参与。

输出：

```text
pooled_rule_text       [B, D, 128]
text_token_states      [B, D, T, 128]
```

### 10.5 AttackEncoder

输入：

```text
ATTACK type embedding
pooled_rule_text
12 个 symbol-specific cost count embeddings
damage_value embedding
damage_mode embedding
```

拼接并投影为：

```text
independent_attack_token [128]
```

### 10.6 AbilityEncoder

输入：

```text
ABILITY type embedding
pooled_rule_text
```

输出 128 维 independent token。

### 10.7 CardEffectEncoder

输入：

```text
CARD_EFFECT type embedding
pooled_rule_text
```

输出 128 维 independent token。

不加入来源 subtype；父卡上下文由 Card Transformer 提供。

### 10.8 Card Transformer

每张卡形成：

```text
[base CARD token, independent detail 0, ..., independent detail M-1]
```

固定配置：

```yaml
hidden_dim: 128
attention_heads: 4
layers: 2
ffn_dim: 256
```

只有解析完成的 `SAME_CARD_DETAIL_REF` 布尔边产生 relation bias。

输出：

```text
card_summary                    [B, 128]
contextualized_detail_tokens    [B, D, 128]
```

正式 detail 导出使用融合前的 `independent_detail_tokens`，确保它不依赖父卡上下文；使用 detail 时再组合 parent card summary、independent token 和动态状态。

---

## 11. 预训练目标

### 11.1 成组 card-field recovery

mask groups：

```text
printed_class + card_kind
rule + rule_family
category_family + category_qualifier
evolves_from + evolves_to
hp
pokemon_type
energy_printed_type
weakness
resistance
retreat
```

mask 某组时，同一目标的直接结构文本引用必须同时遮蔽，避免别名泄漏。

不做 exact card-name recovery。

预测：

```text
类别字段 → 各自 CE
HP / Retreat → 印刷值离散 target 的 CE
Energy profile → 13 个 symbol count CE
```

### 11.2 Detail 属性恢复

从完整的 `independent_detail_token` 预测：

```text
12 个 energy cost counts
damage value
damage mode
```

全部使用离散 CE。Cost head 只可能输出合法的非负整数类别。

不做 detail type recovery，因为 typed detail encoder 已经给出类型。

### 11.3 普通文本 MLM

仅对普通 SentencePiece token 做 15% masking。

结构引用位置不进入 SentencePiece softmax。

### 11.4 结构引用恢复

按完整结构引用单元做 15% masking。mask 时不保留 reference type、field value 或 numeric ID。

预测：

```text
reference type
适用时的 canonical selector field/value
```

不同 reference field 使用独立 head。

### 11.5 Leave-one-detail-out ownership

每张候选卡移除一条 detail，用剩余 card context 生成 summary；held-out independent detail 与原父卡构成唯一 positive。

InfoNCE negatives：

```text
其他卡拥有相同 detail_fingerprint
→ 从 negative mask 排除

其余卡
→ negative
```

不把相同 fingerprint 的其他卡改成额外 positive。

### 11.6 Loss

```text
L =
1.0 * L_field_recovery
+ 1.0 * L_detail_attributes
+ 0.5 * L_text_mlm
+ 0.5 * L_structure_reference
+ 1.0 * L_card_detail_matching
```

每一项先按自身有效预测数平均，再乘权重。

明确不包含：

```text
species / same-species task
全卡池 Card ID 分类
detail type recovery
detail subtype recovery
额外 card-relation classification loss
```

---

## 12. 指标和 baseline

### 12.1 Card fields

```text
各字段 accuracy
各字段 macro accuracy
Energy 各符号 count accuracy
```

类别 baseline 使用 selection-train 多数类；Energy 每个符号使用 selection-train 最常见 count。

### 12.2 Detail attributes

Cost：

```text
每个符号 count accuracy
每个符号 macro accuracy
non-zero count accuracy
完整 12 维 exact-vector accuracy
```

Damage：

```text
damage value accuracy
damage mode accuracy
damage mode macro F1
```

MAE 只作辅助指标，不是主门槛。

### 12.3 文本与引用

```text
普通 token MLM accuracy
普通 token perplexity
reference type accuracy
各 selector field/value accuracy
```

### 12.4 Ownership

```text
Recall@1
Recall@5
MRR
```

baseline 使用有效候选集合中的随机排序期望值。

Baseline 用于报告和诊断，不设置人为最小提升百分点，也不参与事后重选 test checkpoint。

---

## 13. Split、tiny、selection 和 refit

### 13.1 Split

```text
train / validation / test = 80% / 10% / 10%
seed = 20260713
```

相同 canonical card name 和直接 evolution 关系组成 connected component。同一 component 只能进入一个 split。

SentencePiece 和 field/name vocab 可以使用完整固定卡池建立；这是固定卡池的 transductive schema。训练样本和 checkpoint selection 仍严格遵守 split。

### 13.2 Tiny overfit

全量覆盖扫描后固定 16 张具体 Card ID，覆盖：

```text
普通 Pokémon
Pokémon ex
Tera
Attack
Ability
Trainer
Tool
Technical Machine
Fossil
Basic Energy
Special Energy
多 detail 卡
```

配置：

```yaml
max_steps: 500
batch_size_cards: 16
learning_rate: 0.001
weight_decay: 0.0
gradient_clip_norm: 1.0
```

固定 masks 和 negative candidates。

通过条件：

```text
total loss ≤ initial total loss 的 30%
每个激活分任务 loss ≤ 自身 initial loss 的 50%
loss 和 gradient 全部有限
checkpoint 保存并重新加载成功
```

固定 Card ID 缺失或覆盖发生变化时，preflight 直接失败。

### 13.3 Formal selection

```yaml
optimizer: AdamW
learning_rate: 0.0003
betas: [0.9, 0.95]
weight_decay: 0.01
effective_batch_size_cards: 32
gradient_clip_norm: 1.0
scheduler: cosine
warmup_ratio: 0.05
max_epochs: 100
early_stop_metric: validation_weighted_total_loss
early_stop_patience: 12
early_stop_min_delta: 0.0001
seed: 20260713
```

每个 epoch 结束执行 validation。Scheduler 在每次 `optimizer.step()` 后更新；100 epoch 是固定 cosine horizon。

Checkpoint selection：

1. 所有 loss、gradient 和 embedding 有限。
2. 每个启用 head 获得有效样本。
3. checkpoint 可保存并重新加载。
4. 数据 split、索引和 tokenizer 一致。
5. 在满足完整性条件的 checkpoint 中，选择 validation weighted total loss 最低者。

### 13.4 Test

冻结 checkpoint 和阈值后只运行一次 test。

普通指标较差不触发 test 后重选 checkpoint。若发现 NaN、索引错位、数据泄漏或产物不可加载，则停止 production refit。

### 13.5 Production refit

使用：

```text
同一 tokenizer bytes
同一 vocab
同一模型和训练配置
selection-best epoch 数
完整固定卡池
```

即使只训练 `selection_best_epoch` 个 epoch，cosine scheduler horizon 仍保持 100，不重新压缩曲线。

---

## 14. 正式导出

至少包含：

```text
base_card_summary.npy
independent_detail_tokens.npy
detail_offsets.npy

card_metadata.jsonl
detail_metadata.jsonl
text_references.jsonl
text_reference_overrides.jsonl
cross_card_references.jsonl

card_id_to_index.json
field_vocabs.json
sentencepiece.model
encoder_config.json
card_feature_schema.json
artifact_manifest.json
```

Shape：

```text
base_card_summary          [card_count, 128]
independent_detail_tokens  [total_detail_count, 128]
detail_offsets             [card_count + 1]
```

`cards.json` 行号、正式 card summary 行号和正式 embedding mapping 必须一致。

派生 JSONL 只能从 canonical `details.json` 确定性展开；基础训练不能通过文件名搜索并反向读取这些导出文件。

---

## 15. Manifest 和实际执行代码证明

`artifact_manifest.json` 至少保存：

```text
CSV sha256
config sha256
SentencePiece sha256
checkpoint sha256
git_commit
git_dirty
actual_source_tree_sha256
actual_source_files[]
```

Kaggle 实际执行源码逐文件记录：

```text
relative_path
sha256
size_bytes
```

覆盖：

```text
Kaggle entrypoint
data/*.py
models/*.py
training/*.py
configs/*.yaml
schema / vocab 配置
```

每个正式输出文件记录：

```text
path
dtype
shape
sha256
```

实际 Kaggle 运行的 source tree hash 是最终依据；git commit 仅作辅助来源信息。

---

## 16. 验收标准

### 16.1 预处理结构

```text
17 列全部有明确去向
每个 Card ID 只有一个 CardRecord
每个 Move 数据记录只生成一个 detail
所有 detail 类型已确定
detail_offsets[-1] == total_detail_count
保存加载后 detail 顺序、类型和全局行号完全一致
raw multiline、引号、NBSP 等内容无损
```

固定全卡池语义不变量：

```text
cards = 1267
details = 2014
ATTACK = 1556
ABILITY = 223
CARD_EFFECT = 235
Fossil abilities = 5
Core Memory / Geobuster attack_id = 1556
```

### 16.2 Tensor 和模型

```text
永久数据无 padding
batch padding mask 正确
13-symbol Energy profile 未退化成单值
12-symbol Attack cost 全为非负整数类别
数字 reference_id 未进入模型
detail name/subtype/species 未进入模型
same-card bias 只来自已解析布尔边
card summary/detail token 均为 128 维
```

### 16.3 训练完整性

```text
五个启用任务均获得有效样本
loss、gradient、checkpoint、embedding 全部有限
tiny-overfit 达标
validation 每 epoch 执行
test 只执行一次
production refit 使用 selection-best epoch 数
```

### 16.4 产物

```text
mapping、offsets 和 tensor 行号严格一致
checkpoint 保存加载后输出可重建
全部输出 hash、dtype、shape 写入 manifest
canonical validator 在下载后的正式产物上通过
```

---

## 17. 运行边界

### 本地

只运行纯解析和 metadata 测试：

```text
CardRecord 聚合
字段规范化
Detail 分类
Cost / Damage 解析
名称规范化
引用表生成
raw 字段回归
metadata / relation 导出
canonical cache validator
```

### Kaggle

只提交一个 formal kernel，自动执行：

```text
source/config/data guards
→ model preflight
→ fixed-mask tiny overfit
→ formal selection
→ test once
→ production refit
→ formal export
→ canonical artifact validation
```

不单独提交 smoke kernel。任何 preflight 或 tiny failure 都终止后续阶段并输出具体 Card ID、source row、原始字段或 tensor shape。

发布或启动 Kaggle 属于外部高风险操作，必须在实际提交前单独获得明确批准。

---

## 18. 当前实现状态

本节只描述当前代码是否达到本指南，不是设计变更记录。

### 18.1 已有且经过纯解析验证

`data/card_preprocessing.py` 当前能够在内存中完成固定卡池解析：

```text
cards = 1267
details = 2014
ATTACK = 1556
ABILITY = 223
CARD_EFFECT = 235
detail_offsets[-1] = 2014
```

已包含的主要逻辑：

```text
multiline CSV 读取
Fossil/Core Memory detail 分类
Cost/Damage canonical 化
CG attack/card 对齐
raw/canonical 文本分离
结构引用与 fingerprint
v2 cache 写保护
v3 staging 写入逻辑
```

当前解析有 4 个未可靠确认的名称候选；它们保留普通文本并进入 unresolved audit，符合非猜测规则。

### 18.2 已写成草稿、尚未模型运行验证

`data/card_dataset.py`：

```text
v3 feature schema
SentencePiece 训练/绑定
15-field tensor 化
13-symbol Energy profile
12-symbol Attack cost
mixed text/reference sequence
same-card relation matrix
```

`models/card_encoder.py`：

```text
唯一 CardEncoder
card field encoder
HP/Retreat numeric MLP
Energy profile encoder
三个 card branches
AttackEncoder
AbilityEncoder
CardEffectEncoder
RuleTextEncoder
Card Transformer
```

`models/card_pretrain_heads.py`：

```text
五项任务 heads 和 loss 草稿
```

这些模块尚未完成 Kaggle Torch forward/backward preflight，不能视为已验收模型。

### 18.3 未完成或当前不一致

```text
configs/text_reference_overrides.jsonl 尚未建立
artifacts/card_data_v3/ 尚未生成
parser raw-field 回归测试尚未完成
固定 16 张 tiny Card IDs 尚未选择
SentencePiece 尚未正式训练
Dataset/CardEncoder 未运行 Torch preflight
training/pretrain_card_encoder.py 仍含旧 relation/same-species 逻辑
训练模块与新 heads 当前无法端到端运行
export/evaluate/validator 尚未与最终 v3 tensor contract 对齐
formal Kaggle runner 仍含旧 v2 假设
没有 v3 checkpoint
没有 v3 embedding
没有发布或启动 Kaggle 训练
```

因此当前仓库状态是：

```text
canonical parser 可内存解析
模型与 tensor 层为未接通草稿
训练、导出和云端执行尚未完成
```

---

## 19. 后续严格执行顺序

1. 完成 `text_reference_overrides.jsonl` 审计并补齐 raw-field/parser tests。
2. 写入唯一 `artifacts/card_data_v3/`，立即用 canonical validator 重载验证。
3. 用完整 canonical cache 构建并冻结 field vocab、name vocab 和 SentencePiece。
4. 完成全卡池覆盖扫描，固定 16 张 tiny Card ID 与 coverage hash。
5. 在单个 Kaggle formal kernel 中运行 Dataset/CardEncoder forward、mask、loss、backward、save/reload preflight。
6. 通过后执行 fixed-mask tiny overfit。
7. tiny 通过后执行 selection training 和每 epoch validation。
8. 加载 selection-best checkpoint，执行一次 test。
9. 完整性通过后，按 best epoch 数执行 full-pool production refit。
10. 导出正式 embedding/metadata/relations/manifest，并运行 canonical artifact validator。

任何一步失败都停在当前步骤，不跳到后续训练或导出，也不通过修改报告绕过失败。
