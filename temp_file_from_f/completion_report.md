# Replay 数据处理修复完成报告

基线提交：`d73b58e765e25a4e055c09183e4605dcbf4dc4e7`

本报告对应 `fix.md` 与 `fix2.md`。本轮只修复 Replay 事实、状态、动作标签、可复现性及接口合同；没有扩展模型、开始动态训练、增加训练 loss 模块或创建实验目录。

## 验证结论摘要

- 全仓测试：93 项通过。
- Python 语法检查：通过。
- 本轮修改文件的 `git diff --check`：通过。
- 真实 Replay `episode-84817357-replay.json`：108 个对齐决策，0 parser error，0 非法 action index。
- 真实 Replay 动作语义：99 `SINGLE_INDEX`，9 `ORDERED_INDEX_SEQUENCE`。
- equivalence resolution：103 `FULLY_RESOLVED`，5 `UNRESOLVED`；未解析样本没有被静默 mask。
- turn owner：105 个 `PRESENT + INFERRED_AUDITED_TURN_RULE`，3 个 setup/信息不足样本为 `UNKNOWN`。

## 前述 15 个问题

| 问题 | 结论 | 证据或修改理由 | 修改文件 | 新增/更新测试 | schema |
|---:|---|---|---|---|---|
| 1 | 执行 | 匿名事件只累计匿名流量，不再遍历并清空 `fromArea` 内已知 Serial；snapshot 继续覆盖精确位置。 | game memory/schema | 三张已知手牌 + 一张匿名移出后仍全部保持 HAND。 | 是 |
| 2 | 执行 | context 5/7/8/9 从 unordered 白名单删除。引擎 numeric API 分别对应 `ToBench/ToHand/Discard/ToDeck`，且按 action 顺序建立 `targetList`。 | legal options/audit/docs | 四个 context 均为 `ORDERED_INDEX_SEQUENCE`。 | 是 |
| 3 | 执行 | `DecisionKey` 加入非空 `replay_key`；EpisodeId、replay id、JSONL line、ZIP member、规范化路径依序形成身份。 | schema/replay dataset | 两条无 ID 的 JSONL 不碰撞；ZIP member 身份不同。 | 是 |
| 4 | 执行（方案 B） | 删除无效 `include_no_select` dataset/importer/CLI 接口；只导出真实决策点。 | dataset/importer/CLI/docs | 真实决策测试覆盖。 | 否 |
| 5 | 执行 | action 只接受 int 或整数字符串；顶层 `None` 是空 action；列表内 `None`、`bad`、object-like 值产生 alignment error，不变成 0。 | dataset/audit | 三类非法 action 均不产生训练样本。 | 是 |
| 6 | 执行 | `select.deck` instance 与 option source zone 统一为 `LOOKING=12`；origin 单独保留；显式空 deck 不 fallback。 | parser/legal/docs | instance、option、空列表测试。 | 是 |
| 7 | 执行 | 删除通用“负整数即 UNKNOWN”；仅字段级 `-1` sentinel。 | parser/legal | sentinel、合法负值、missing/null/unknown 测试。 | 是 |
| 8 | 执行 | Memory 改为 exact zone、ambiguous serial、known serial、visible observation、movement event、first/last known 的真实公式名。 | schema/memory/replay/docs | count 公式测试。 | 是 |
| 9 | 执行 | 时间字段改为 observation-relative；保留字段全部进入 19 维 Event Feature。 | state/parser/memory/docs | batch 顺序、age、维度测试。 | 是 |
| 10 | 执行，并反驳“只能全 UNKNOWN” | 引擎 `activePlayerIndex()` 明确定义 `((turn + 1) ^ firstPlayer) & 1`。显式字段优先，setup/缺失仍 UNKNOWN。 | replay/audit/docs | inference source 与 fallback；真实 Replay 105/3。 | 是 |
| 11 | 执行 | reference v2 保存 source/member/line、原始 hash、fingerprint、parser/schema version；loader 全校验并 fail closed，支持同内容换路径。 | replay/docs | 重建、移动、hash 篡改测试。 | 是 |
| 12 | **反驳** | 当前 split 是固定卡池内 optimization/regression monitoring，不是 unseen-card generalization；保留全卡池 schema/normalization 并明确用途。 | static README；核心 dataset 未改 | 三个 partition 共用 frozen schema。 | 否 |
| 13 | 执行 | 完整 variant 改用 Card ID multiset；洗牌顺序不产生新 variant，Trainer copy-count 差异仍区分。 | replay kernel/docs | 反转 deck 后 variant_count=1。 | 否 |
| 14 | 执行 | 缺少 Card CSV 时列出搜索路径并停止，不生成空 signature。 | replay kernel/docs | 缺失 metadata 明确失败。 | 否 |
| 15 | 执行 | normalization 增加 start/end/repeated date；summary 保存 requested/included/excluded dates 和 deck 总数。 | script/docs | train + reserved 混合目录过滤。 | 否 |

## 原 12 个自定义问题专表

| 原问题 | 当前结论 | 反驳或执行 | 证据/最终语义 | 修改与测试 | 立即启用 |
|---:|---|---|---|---|---|
| 1 Multi-select | 5/7/8/9 原定义不成立 | 执行 | 引擎 enum/JSON/targetList；5/7/8/9 ordered，2/15/21/26/27/34 保留确认的 unordered，22 ordered。 | legal/audit/docs/tests | 是 |
| 2 policy mask | 原原则保留、边界收紧 | 执行 | resolution 不完整、ordered、可变数量均不 mask，并记录 reason。 | legal/schema/replay/tests | 是 |
| 3 Equivalence | 原 key 过度合并 | 执行 | Serial 默认保留；direct identity 唯一解析；不完整 ref 降级；仅 FULLY_RESOLVED 合并。 | legal/parser/tests | 是 |
| 4 Subset loss | 只冻结 target | 按要求整理接口 | `TARGET CONTRACT IMPLEMENTED / TRAINING LOSS NOT IMPLEMENTED`。 | schema/docs/tests | **否** |
| 5 turn owner | 全 UNKNOWN 不正确 | 执行 | 引擎公式；真实 Replay 105 个确定、3 个 UNKNOWN。 | replay/audit/docs/tests | 是 |
| 6 Temporal | 原字段暗示真实发生时间 | 执行 | observed-at/batch/age 语义，全部进入 19 维 feature。 | state/parser/memory/tests | 是 |
| 7 Anonymous pools | 原实现破坏 Serial | 执行 | 匿名流不猜 Card ID、不改 Serial，current count 与 cumulative flow 分离。 | memory/schema/tests | 是 |
| 8 Residual IPF | 工具保留、系统未就绪 | 按要求整理接口 | 默认 NOT_APPLICABLE；启用但输入不足 UNKNOWN；不称 posterior。 | replay/docs/tests | **否** |
| 9 Card ID Memory | 名称与公式不符 | 执行 | known serial/current zone/visible observation/movement event。 | schema/memory/tests | 是 |
| 10 Compact data | v1 不可稳定校验 | 执行 | raw hash + fingerprint + key + versions，reference v2 fail closed。 | replay/docs/tests | 是 |
| 11 Missingness | 通用负数规则错误 | 执行 | 字段级 sentinel、required-input derived state、五态统一。 | parser/legal/replay/tests | 是 |
| 12 used_detail | 无稳定 Ability 映射 | 按要求整理接口 | 保留 UNKNOWN/NOT_APPLICABLE 与 inference source；未伪造标签。 | state/parser/docs | **否** |

## context 5、7、8、9 的单独证据

引擎证据来自仓库已有 `pokemon-tcg-ai-battle.zip` 内 `ApiType.h`、JSON builder 与 `setSelectedCardTarget()`：numeric API 使用 `SelectContext enum - 1`，被选 option 按 action 顺序写入 `targetList`。

| context | select.type | option.type | 引擎 context | Replay 例子 | 对齐样本 | 结论 |
|---:|---:|---:|---|---|---:|---|
| 5 | 1 | 3 | `ToBench` | episode 84817357 step 23，Buddy-Buddy Poffin (1086)，action `[3,0]`，日志依次移到 Bench | 1 | ordered；Bench/list/log 顺序未证明可消除 |
| 7 | 1 | 3 | `ToHand` | step 5，Poké Pad (1152)，从 `select.deck` 选卡进手牌；同局还含 Prize-to-hand | 23 | ordered；context-wide 手牌/日志不变量未证明 |
| 8 | 1 | 3 | `Discard` | step 30，Ultra Ball (1121)，action `[5,2]`，日志按所选顺序丢两张手牌 | 1 | ordered；discard/list/log 顺序被保留 |
| 9 | 1 | 3 | `ToDeck` | step 106，Sacred Ash (1129)，action `[0,1,2]`，依次入 deck 后 shuffle | 1 | ordered；插入顺序进入引擎状态 |

Replay 接受某个顺序只证明 action 合法，不证明交换顺序后结算相同。因此四项选择执行修复，不是直接把意见当结论。

## 实际修改文件

```text
data/decision_schema.py
data/state_schema.py
data/observation_parser.py
data/game_memory.py
data/legal_options.py
data/replay_dataset.py
data/online_replay_importer.py
scripts/audit_replay_decision_contract.py
scripts/import_online_replay_decisions.py
scripts/normalize_replay_statistics.py
kaggle/kernels/replay_extract/extract_popular_decks.py
docs/reference/replay_decision_data_contract.md
docs/KAGGLE_WORKFLOW.md
static_card/README.md
tests/test_decision_schema.py
tests/test_observation_parser.py
tests/test_replay_dataset.py
tests/test_normalize_replay_statistics.py
static_card/tests/test_card_data.py
temp_file_from_f/completion_report.md
```

## 特别说明：另外做出的修改

任务文档列出的若干路径在当前提交中不存在，因此做了以下必要适配；这些是本轮另外做出的修改：

1. `docs/replay_decision_data_contract.md` 实际位于 `docs/reference/replay_decision_data_contract.md`，修改真实文件，没有新建重复合同。
2. `tests/test_dynamic_features.py` 与 `tests/test_hidden_belief_and_legal_options.py` 不存在；回归放入真实的 `tests/test_observation_parser.py` 与 `tests/test_decision_schema.py`。
3. `tests/test_card_dataset.py` 不存在；split 反驳测试更新在 `static_card/tests/test_card_data.py`。
4. 为满足 compact reference 可重建且 fail closed，新增 `rebuild_replay_decision_from_reference()`；这不是训练功能。
5. 只读审计了仓库已有引擎源码 ZIP 来验证 turn owner/context 数字语义；没有修改或解压产物到项目目录。

开始本轮前已经存在、且本轮没有修改其内容的用户工作树变化：

```text
.gitignore
data/__init__.py
temp_file_from_f/card_dataset.py
data_from_submission/
temp_file_from_f/fix.md
temp_file_from_f/fix2.md
```

其中前三项在开始时已是 modified；`card_dataset.py` 主要表现为已有换行符变化。本轮未覆盖或清理这些改动。

## 未修改核心实现但完成反驳

- 静态 Card Dataset split：`static_card/data/card_dataset.py` 保持不变；通过 README 用途限定和 split 测试补充完成反驳。

## 无法从当前 Replay 单独证明

- context 5/7/8/9 的“交换任意顺序总是等价”无法证明，故 ordered。
- 当前单局不能替代完整 7,500 Replay audit；2/15/21/26/27/34 的既有规则确认保留，但更换引擎版本后应重跑全量 settlement audit。
- Ability 日志没有稳定 detail ID 映射，`used_detail` 不启用。
- Hidden Belief 缺少可靠 presence/count 输入，Residual IPF 不启用。

## Schema/version 变化

```text
replay_decision_contract_v1 -> replay_decision_contract_v2
replay_decision_reference_v1 -> replay_decision_reference_v2
replay_observation_parser_v2（新增显式 parser contract）
replay_decision_contract_audit_v1 -> v2
legal_option_equivalence_v1 -> v2
EVENT_FEATURE_DIM: 16 -> 19
```

旧 reference 不会被新版 loader 静默接受。

## 测试命令和结果

```bash
python -m py_compile data/*.py scripts/*.py \
  kaggle/kernels/replay_extract/extract_popular_decks.py

TMPDIR=/tmp conda run -n ml python -m pytest -q
```

结果：`93 passed`。本轮修改文件的 `git diff --check` 通过。

全工作树直接执行 `git diff --check` 会报告开始前已有的 `data/__init__.py` 与 `temp_file_from_f/card_dataset.py` CRLF/whitespace 差异；没有擅自格式化这些用户改动。

## 仍需真实 Replay 人工检查的最少项目

1. 在完整 Replay ZIP 上重跑新版 audit，确认 5/7/8/9 在所有卡牌/effect 下保持 ordered，或识别可由规则证明的更窄 semantic key。
2. 启用 Ability `used_detail` 前完成日志到 detail 的全量 resolver audit。
3. 启用 Residual IPF/group-aware subset loss 前分别完成 hidden count/presence 输入审计和真实 target audit；本轮不宣称训练功能完成。
