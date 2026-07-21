# Codex 指令：实现锁定版因果事务级 PPO 与校准语义势

## 任务目标

请在仓库 `3956418278-dotcom/Pokemon` 的 `decks` 分支上，实现锁定版训练奖励与信用分配方案：

> 因果事务级 PPO + 固定语义概念向量 + 校准后的语义势差塑形 + 冻结对手的非对称自博弈。

本次任务直接替换当前 `competition_selfplay` 中的旧三维奖励原型。完成后，该分支应具备可开始 Phase A 训练的完整代码路径，并已经具备 Phase B 的启用逻辑。

本任务不修改牌组内容，不依赖机械 agent，不恢复 replay imitation 路线，不引入按具体 Card ID 编写的奖励规则。

本任务优先保证端到端训练数据流真实接通。每完成一个阶段，先检查生成的数据是否被下一个阶段实际消费，再继续。不要仅创建未接入训练循环的类、接口或占位实现。

---

## 一、先读取现有实现

先阅读并理解：

- `competition_selfplay/reward.py`
- `competition_selfplay/config.py`
- `competition_selfplay/configs/`
- `competition_selfplay/features.py`
- `competition_selfplay/league.py`
- 当前模型、rollout、训练和测试文件
- `records/competition_selfplay/CURRENT.md`

定位以下旧结构并替换：

- `RewardVector(outcome, prize_progress, setup_tempo)`
- `VectorReward`
- `_setup_potential`
- `actor_weights`
- `own_active_weight`
- `own_bench_weight`
- `own_energy_weight`
- `opponent_active_weight`
- `opponent_bench_weight`
- `setup_delta_clip`
- 强制 `model.value_dimensions == 3` 的校验
- 围绕上述旧结构编写的测试

保留现有牌组读取、特征编码、checkpoint、league 与输出目录约定。根据实际仓库结构做最小必要修改，不新建一批一次性审计脚本。

---

## 二、固定语义概念向量

### 2.1 固定位置

实现固定长度的语义概念向量：

```python
SEMANTIC_CONCEPT_NAMES = (
    "attack_available_next_own_turn",
    "active_survival_to_next_own_turn",
    "net_prize_gain_h1",
    "net_prize_gain_h3",
    "net_prize_gain_h6",
    "self_deckout_risk_h6",
    "rule_lock_persists_to_opponent_turn",
    "recovery_path_realized_next_own_turn",
    "armed_delayed_trigger_realized",
)
```

固定维度：

```python
NUM_SEMANTIC_CONCEPTS = 9
```

每个下标在所有牌组、所有局面和两个座位中含义保持不变。语义始终从当前 observation 所属玩家的视角定义。

这些维度不是卡槽位置，也不绑定某张卡。可变长度的手牌、场面、合法动作和卡牌实例仍由现有状态编码器处理；共享状态表示再输出这 9 个概念预测。

### 2.2 每个概念的精确定义

1. `attack_available_next_own_turn`

   当前玩家在下一次自己的完整行动事务中，实际执行至少一次攻击的概率。

2. `active_survival_to_next_own_turn`

   当前 Active Pokémon 仍以同一物理实例存活到下一次自己的根决策的概率。

3. `net_prize_gain_h1`

4. `net_prize_gain_h3`

5. `net_prize_gain_h6`

   在未来 1、3、6 个“本方事务”窗口内：

   ```text
   自己拿走的对方奖赏数 - 对方拿走的自己奖赏数
   ```

   目标除以 6，裁剪到 `[-1, 1]`。

6. `self_deckout_risk_h6`

   当前玩家在未来 6 个本方事务内因无法抽牌而失败的概率。

7. `rule_lock_persists_to_opponent_turn`

   当前已经存在的工具、场地、能力或状态型规则限制，持续影响到对方下一次根决策的概率。

8. `recovery_path_realized_next_own_turn`

   当前玩家在下一次自己的根决策前，实际形成一个可继续行动的恢复路径的概率。恢复路径包括重新建立 Active、准备后备攻击者、解除阻断状态或恢复关键资源链。

9. `armed_delayed_trigger_realized`

   当前已布置但尚未触发的延迟效果，在下一次自己的根决策前实际触发的概率。

### 2.3 applicability mask

`applicable` 由当前状态和规则事实确定，不作为自由预测量。例如当前没有已生效规则锁时，对应 mask 直接为 0；当前没有 armed delayed effect 时，对应 mask 直接为 0。不要让网络通过预测 mask 来逃避概念监督。

为每个概念同时输出：

```python
@dataclass
class SemanticConceptOutput:
    values: Tensor       # [batch, 9]
    applicable: Tensor   # bool/[0,1], [batch, 9]
    confidence: Tensor   # [0,1], [batch, 9]
```

规则：

- 概念有明确适用对象时，`applicable=1`。
- 当前没有规则锁时，`rule_lock_persists_to_opponent_turn` 的 `applicable=0`。
- 当前没有 armed delayed effect 时，`armed_delayed_trigger_realized` 的 `applicable=0`。
- 不适用概念不参与该 head 的监督损失。
- 不要把“不适用”直接训练成数值 0，因为 0 表示低概率，不等于没有适用对象。
- 进入语义势时，将 `value`、`applicable` 和 `confidence` 一并使用。
- `confidence` 不使用一个无约束的自由 sigmoid head。优先由小型 ensemble disagreement 计算；若第一版暂不实现 ensemble，则使用经过 holdout 校准的预测熵映射，并在进入势函数前 `detach`，防止模型通过把 confidence 压到 0 来逃避价值学习。

### 2.4 自动标签

概念标签全部从完成后的自博弈 trajectory 自动回看生成，不需要人工局面评分。

实现一个 trajectory label builder，输入完整对局的事务序列和事件记录，输出每个事务起点的：

```python
SemanticConceptTargets(
    values=[...],
    applicable=[...],
)
```

使用物理实例 serial 区分 Pokémon，避免只按 Card ID 判断存活或移动。

---

## 三、事务边界

### 3.1 事务定义

一个事务从某玩家的根选择开始，包含该动作产生的全部嵌套选择，直到满足任一条件：

- 返回同一玩家新的 `MAIN` 根决策；
- 控制权交给另一玩家；
- 回合结束；
- 游戏结束。

示例：

```text
打出 Crispin
→ 选择第一张能量
→ 选择第二张能量
→ 选择附着目标
→ 完成结算
```

以上是一个事务。

### 3.2 强制选择

合法选项只有一个时：

- 可以自动执行；
- 记录在事务事件中；
- 不计入 actor log probability；
- 不单独获得 advantage。

### 3.3 事务记录

实现或扩展事务数据结构，至少包含：

```python
@dataclass
class Transaction:
    transaction_id: int
    seat: int
    start_state: ...
    end_state: ...
    non_forced_log_probs: list[Tensor]
    old_log_prob_sum: float
    terminal: bool
    outcome: int
    event_records: list[...]
    cause_transaction_ids: list[int]
```

联合 log probability：

```python
transaction_log_prob = sum(non_forced_log_probs)
```

---

## 四、延迟因果边

延迟效果在实际触发时创建事件，并链接到布置该效果的早期事务：

```python
@dataclass
class CausalEventLink:
    cause_transaction_id: int
    trigger_transaction_id: int
    cause_kind: str
    source_card_serial: int | None
```

例如：

```text
装备 Handheld Fan 的事务
→ 若干事务后对手攻击
→ Handheld Fan 触发
```

触发事件只在实际发生时计入 trajectory 标签。第一版不额外向原因事务添加直接 bonus，避免同一效果重复奖励。

因果边用于：

- `armed_delayed_trigger_realized` 标签；
- full critic 的长程学习；
- 调试和解释输出。

---

## 五、模型输出

共享状态编码器后建立：

```python
policy_head
full_value_head
semantic_concept_heads
semantic_potential_head
residual_value_head
```

价值关系：

```python
semantic_value = semantic_potential(concepts, masks, confidence)
full_value = semantic_value + residual_value
```

要求：

- `full_value` 是标量；
- `semantic_value` 是标量；
- `residual_value` 是标量；
- 删除旧的 `value_dimensions=3` 语义；
- 配置中可保留 `value_dimensions` 时应设为 1，或直接移除该字段并由代码固定为标量；
- actor 与 full critic 可以使用高容量网络；
- residual value 永远不直接进入 shaping reward。

### 5.1 明确训练目标与梯度边界

对每个事务起点，从该玩家视角构造折扣后的终局回报目标：

\[
G_k = \gamma^{K-1-k} z
\]

其中 `z` 为整局最终胜负效用，`K` 为该玩家事务总数。

使用以下损失：

\[
L_{\mathrm{concept}}
=
\sum_i m_i\,\ell_i(\hat c_i,c_i)
\]

二分类概念使用 BCE 或 Brier loss，连续净奖赏概念使用 Huber loss。

\[
L_{\mathrm{semantic}}
=
\operatorname{Huber}(V_{\mathrm{semantic}},G_k)
\]

\[
L_{\mathrm{residual}}
=
\operatorname{Huber}
\left(
V_{\mathrm{residual}},
\operatorname{stopgrad}(G_k-V_{\mathrm{semantic}})
ight)
\]

\[
L_{\mathrm{full}}
=
\operatorname{Huber}(V_{\mathrm{full}},G_k)
\]

实现要求：

- 概念 head 主要由 `L_concept` 学习，保持每个维度的既定语义；
- `semantic_potential_head` 使用 `concept_values.detach()`、固定 applicability 和 `confidence.detach()` 训练，避免价值损失把概念维度改造成不可解释的隐藏通道；
- residual target 必须对 `G_k - V_semantic` 使用 `stopgrad`，防止 semantic 与 residual 互相抵消形成不可识别分解；
- `V_full = V_semantic + V_residual`，`L_full` 用于保证最终 critic 精度；
- 在 loss metrics 中分别记录四项损失，确认每个输出都实际被训练循环消费。

---

## 六、语义势

### 6.1 可解释结构

语义势使用固定概念位上的一维非线性与少量具名交互：

```text
attack_available × net_prize_gain_h1
active_survival × net_prize_gain_h3
recovery_path × active_survival
rule_lock_persists × net_prize_gain_h3
delayed_trigger_realized × net_prize_gain_h3
self_deckout_risk × net_prize_gain_h6
```

实现形式：

\[
L_{\mathrm{sem}}(S)
=
b+
\sum_i f_i(c_i,m_i,u_i)
+
\sum_{(i,j)\in E}
w_{ij} f_i f_j
\]

\[
\Psi(S)=q(S)\tanh(L_{\mathrm{sem}}(S))
\]

其中：

- `c_i` 是概念值；
- `m_i` 是 applicability；
- `u_i` 是该概念 confidence；
- `f_i` 使用可学习分段线性函数或单调/非单调一维 spline；
- `q(S)` 来自概念置信度汇总和/或小型 ensemble disagreement；
- `Psi` 裁剪在 `[-0.8, 0.8]`。

不要给某个概念手工指定固定正负奖励。所有 spline 和交互权重通过训练学习。

### 6.2 解释输出

提供结构化解释接口：

```python
SemanticPotentialExplanation(
    bias=...,
    unary_contributions={concept_name: value},
    interaction_contributions={interaction_name: value},
    confidence_gate=...,
    pre_tanh_logit=...,
    potential=...,
)
```

要求贡献项精确重构 `pre_tanh_logit`。

---

## 七、奖励函数

从某个玩家视角，对连续两个本方事务：

\[
r_k
=
z_k+
\alpha_t
\left[
\gamma\bar\Psi(S_{k+1})-\bar\Psi(S_k)
\right]
\]

其中：

```python
z_k = +1.0  # 该玩家获胜且当前事务终局
z_k = -1.0  # 该玩家失败且当前事务终局
z_k =  0.0  # 其他情况，包括平局
```

规定：

```python
Psi(terminal_state) = 0.0
```

固定默认参数：

```yaml
gamma: 0.997
gae_lambda: 0.95
terminal_win: 1.0
terminal_loss: -1.0
terminal_draw: 0.0
max_shaping_alpha: 0.15
potential_clip: 0.8
target_ema_tau: 0.01
```

没有以下直接奖励项：

- 奖赏进度 bonus；
- 附能 bonus；
- 攻击 bonus；
- 进化 bonus；
- 手牌 bonus；
- 保留场上 Pokémon bonus；
- 自爆 penalty；
- 送奖赏 penalty；
- 某张具体卡的专用 bonus。

---

## 八、事务级 GAE 与 PPO

事务 TD error：

\[
\delta_k
=
r_k+
\gamma(1-d_k)V_{\mathrm{full}}(S_{k+1})
-
V_{\mathrm{full}}(S_k)
\]

事务级 GAE：

\[
A_k
=
\delta_k+
\gamma\lambda(1-d_k)A_{k+1}
\]

一个事务内所有非强制选择共享同一个 `A_k`。

PPO ratio 使用事务联合概率：

\[
\rho_k
=
\exp
\left(
\sum_j\log\pi_\theta(a_{k,j}|o_{k,j})
-
\sum_j\log\pi_{\mathrm{old}}(a_{k,j}|o_{k,j})
\right)
\]

默认 PPO 参数：

```yaml
clip_epsilon: 0.2
value_coefficient: 0.5
entropy_coefficient: 0.01
max_grad_norm: 0.5
normalize_advantage: true
```

---

## 九、Phase A

前 20,000 个完整自博弈对局：

```python
shaping_alpha = 0.0
```

同时训练：

- learner actor；
- full critic；
- semantic concept heads；
- semantic potential online network；
- residual value head。

正式奖励只有终局胜负。

Phase A 仍使用事务级 PPO，不退回逐 `select` PPO。

---

## 十、校准门

在固定 holdout trajectory 上计算：

- Brier score；
- Expected Calibration Error；
- 座位交换反对称误差；
- value ranking accuracy。

启用 Phase B 的条件：

```text
Brier score 相比常数先验改善 >= 15%
ECE <= 0.10
座位交换反对称误差 <= 0.08
ranking accuracy >= 0.60
```

未通过时继续 Phase A，保持 `alpha=0`，不更换奖励框架。

校准指标和是否开启 Phase B 写入 metrics 与 checkpoint metadata。

---

## 十一、Phase B 的自博弈更新方式

### 11.1 不同时在线更新两套策略

实现一个在线 learner 和一个冻结 opponent snapshot：

```text
learner policy: πθ
frozen opponent: πopp
frozen target semantic reward module: Ψbar
```

这里的 `Ψbar` 指完整冻结的语义奖励路径，而不只是最后一层 potential head。它必须包括产生 shaping 所需的语义输入编码、concept heads、confidence 计算和 semantic potential head。若这些模块共享 actor/full critic 的 encoder，则 rollout 开始时必须复制一份独立 target semantic module，或者在采样时直接存储 `phi_before/phi_after`，训练更新阶段不得重新用已变化的在线 encoder 计算该批奖励。

每个 rollout batch 内：

- `πopp` 保持冻结；
- 完整的 `Ψbar` 保持冻结；
- 每个事务采样时存储 `target_phi_before` 和 `target_phi_after`；PPO update 直接使用存储值，不在更新期间重新计算当批 shaping；
- learner 在不同对局中交替坐 P0/P1；
- actor loss 只使用 learner 控制座位的事务；
- opponent 的动作不进入 learner PPO ratio；
- 完整对局可以从双方视角生成 semantic concept 标签；
- full critic/value loss 优先使用 learner 座位的 on-policy 事务；
- semantic concept supervised loss 可以使用双方视角样本。

rollout batch 完成后：

1. 更新 learner actor；
2. 更新 full critic；
3. 更新 online semantic concept/potential heads；
4. 更新 residual value；
5. 在 rollout batch 完成且 learner 更新结束后，更新完整 target semantic reward module：

   ```python
   target_semantic = (1 - tau) * target_semantic + tau * online_semantic
   ```

   默认 `tau=0.01`。EMA 覆盖语义奖励路径中的 encoder、concept heads、confidence 模块和 potential head，不覆盖 actor 与 residual critic。

6. frozen opponent 保持不变，直到既有 league/promotion 逻辑决定加入或替换 snapshot。

因此，Phase B 的“两方都变强”通过后续 snapshot/pool 迭代实现，不在同一 rollout batch 内让两边同时漂移。

### 11.2 shaping alpha

首次通过校准门后，在随后 50,000 个完整对局内：

\[
\alpha_t
=
0.15
\min
\left(
1,
\frac{N-N_{\mathrm{gate}}}{50000}
\right)
\]

每个 rollout batch 开始时计算并冻结当批 `alpha_t`。

---

## 十二、配置修改

重构 `RewardConfig`，建议字段：

```python
@dataclass(frozen=True)
class RewardConfig:
    terminal_win: float
    terminal_loss: float
    terminal_draw: float
    max_shaping_alpha: float
    potential_clip: float
    target_ema_tau: float
    phase_a_games: int
    phase_b_ramp_games: int
    calibration_brier_improvement: float
    calibration_ece_max: float
    calibration_antisymmetry_max: float
    calibration_ranking_min: float
```

删除旧 `actor_weights` 和 setup 相关权重。

配置 schema version 升级为清晰的新版本，例如：

```yaml
schema_version: transactional_semantic_selfplay_v2
```

旧配置读取时给出明确错误，说明旧三维奖励配置已废弃；不静默套用旧权重。

---

## 十三、测试

直接更新现有测试并补充少量集中测试。不要为每个小函数创建独立测试脚本。

至少覆盖：

1. 一个多层卡牌结算只生成一个事务；
2. 唯一合法选项不进入联合 log probability；
3. 两个非强制选择的 log probability 正确求和；
4. 终局胜负奖励为 `+1/-1/0`；
5. `alpha=0` 时严格退化为纯终局奖励；
6. terminal state 的势为 0；
7. target semantic potential 在 rollout batch 内无梯度且不变化；
8. Phase B actor loss 只包含 learner 座位；
9. frozen opponent 在一次 batch update 后参数不变化；
10. learner 交换 P0/P1；
11. applicability=0 的概念不参与 concept loss；
12. 语义势解释贡献精确重构 logit；
13. 座位交换后概念与 value 方向正确变换；
14. 事务级 GAE 使用事务步长，而不是原始 select 数；
15. 延迟触发事件能链接到早期 cause transaction；
16. 当前旧 `setup_tempo`、三维 `RewardVector` 和 `value_dimensions==3` 不再存在；
17. `applicable` 来自确定性状态事实，网络不能通过预测 mask 关闭监督；
18. semantic value、residual value 和 full value 的 target/stop-gradient 边界符合本指令；
19. 更新在线共享 encoder 后，当批已存储的 shaping reward 不发生变化；
20. confidence 不能通过梯度自行塌缩到 0。

运行仓库原有相关测试和新增测试。只保留能长期维护的测试文件。

---

## 十四、输出与记录

训练 metrics 至少增加：

```text
phase
shaping_alpha
semantic_brier
semantic_ece
semantic_antisymmetry_error
semantic_ranking_accuracy
semantic_potential_mean
semantic_potential_std
semantic_confidence_mean
transaction_count
non_forced_select_count
forced_select_count
```

checkpoint 保存：

- learner；
- optimizer；
- full critic；
- online semantic heads；
- target semantic potential；
- residual value；
- frozen opponent reference；
- current phase；
- completed games；
- shaping alpha；
- calibration metrics；
- config snapshot。

更新 `records/competition_selfplay/CURRENT.md`，准确写明：

- 旧三维奖励已替换；
- Phase A 使用纯终局事务 PPO；
- Phase B 由校准门启用；
- rollout 中 learner 更新、opponent 与 target potential 冻结；
- 当前是否已经具备正式训练条件。

---

## 十五、实施顺序

按以下顺序完成：

1. 重构 config 与旧奖励接口；
2. 实现事务组装；
3. 让 rollout 输出事务；
4. 实现终局事务奖励和事务级 GAE；
5. 接通 Phase A 的 actor/full critic 训练；
6. 实现固定语义概念向量及自动标签；
7. 实现 semantic potential、residual value 和解释接口；
8. 实现校准门；
9. 实现 Phase B alpha ramp、EMA target 和 frozen opponent 更新边界；
10. 更新测试、配置和 CURRENT 文档；
11. 跑相关测试并报告修改文件、测试结果和尚未完成的真实问题。

先完成一条可运行的端到端路径，再处理局部代码整理。保持实现集中、接口清楚，不增加与训练无关的审计框架。
