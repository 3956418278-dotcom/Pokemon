# Codex 修改指令：精简 Prompt 集中的重复审计

## 任务目标

修改当前用于逐步实现 Pokémon TCG Agent 的 Prompt 集，删除已经由 7,500 局 Replay 全量审计确认过的重复审计任务。

本次修改只调整 Prompt 的任务内容，不修改正式模型结构、数据结构、训练流程和现有源码。

修改后的原则是：

> 已确认的数据语义不再重复审计；实现过程中只保留能够直接阻止错误代码进入训练或提交包的硬验证。

---

## 一、Prompt 01：删除独立接口审计任务

### 删除内容

从 Prompt 01 中删除以下产物和对应任务：

```text
audit_interfaces.py
interface_audit.json
interface_audit.md
```

同时删除要求重新统计以下内容的任务：

```text
selection mode 分布
select.type / select.context 分布
option 数量分布
serial 匹配率
equivalence group 覆盖率
UNKNOWN 语义组合
Replay 决策重复规律
```

这些内容已经在 7,500 局 Replay 全量审计中确认，不需要再次扫描和生成报告。

### 保留内容

Prompt 01 只保留真实 Replay 驱动的接口测试：

```text
1. DecisionSampleV1 可以从真实 Replay 决策点生成。
2. Actor 侧数据不得读取 visualize.current。
3. expert target 必须属于当前合法 options。
4. 原始 option index 必须能够无损恢复。
5. card_id 与 serial 必须保持不同语义，不得混用。
6. 变长 episode 必须按真实 steps 遍历。
7. 无 select 的 observation 不得生成决策样本。
```

这些检查应放入现有测试文件或最小测试入口，不再创建独立审计脚本和审计报告。

---

## 二、Prompt 02：将“语料审计”改为“缓存构建验证”

### 删除内容

删除独立产物：

```text
replay_corpus_audit.md
```

如果 Prompt 中要求生成独立的 `replay_corpus_audit.json`，也一并删除；必要统计统一写入缓存目录下的 `statistics.json`。

删除以下重复统计任务：

```text
selection mode 全量分布
select.type / select.context 全量分布
option 数量全量分布
equivalence group 覆盖率
重复 observation / decision 规律
action 与 observation 的配对逻辑
serial 引用总体统计
UNKNOWN 语义组合重新统计
```

### 保留内容

缓存构建阶段必须保留以下硬正确性检查：

```text
1. Replay 成功转换数量、失败数量和失败原因。
2. 每个 DecisionSample 的 target 都属于当前 legal options。
3. train / validation / test 按完整 episode 或日期划分，三者不得有 episode 交集。
4. terminal outcome 必须转换到当前决策玩家视角。
5. 同一决策点不得重复写入缓存。
6. Actor 输入不得包含对手隐藏手牌、隐藏牌库内容或 visualize.current。
7. 未知 Card ID 数量和对应样本数量必须记录。
8. 解析失败、字段缺失和静态卡牌对齐失败必须有明确计数。
```

缓存目录只需要保留：

```text
statistics.json
build.log
failed_episodes.jsonl    # 仅在存在失败时生成
```

不要额外生成面向人类阅读的大型审计报告。

---

## 三、Prompt 05：删除搜索能力审计报告

### 删除内容

删除以下产物：

```text
search_capability_audit.md
```

不要为 simulator 状态复制能力生成正式审计文档。

### 调整执行顺序

浅层搜索属于可选增强，不是第一版中等效果 Agent 的必要组成部分。

因此：

```text
若当前版本不实现 Value-guided rollout 或浅层搜索，Prompt 05 中所有 simulator 状态复制工作暂时跳过。
```

只有明确进入搜索实现时，才增加一个最小可执行测试，检查：

```text
1. simulator 状态可以复制。
2. 原状态和副本推进后互不影响。
3. 随机状态或随机种子能够一致恢复。
4. 单步和短 rollout 耗时满足提交环境预算。
5. rollout 失败时能够安全回退到策略网络动作。
```

测试结果直接由测试命令的通过或失败表示，不生成 Markdown 审计报告。

---

## 四、Prompt 08：公共推理代码只审查一次

### 删除内容

删除对每个候选 checkpoint 重复执行以下工作的要求：

```text
hidden-information audit
action semantics 全覆盖审计
multi-select contract 全覆盖审计
DecisionSample contract 全覆盖审计
```

这些性质由共同的推理代码决定，而不是由模型权重决定。

### 修改后的测试分层

#### A. 公共代码测试一次

对所有候选共同使用的以下模块执行一次完整测试：

```text
observation adapter
DecisionSample schema
action encoder
MultiSelectDecoder
action contract
hidden-information boundary
episode reset
fallback logic
```

只有公共推理代码发生修改时，才重新运行这组测试。

#### B. 每个候选 checkpoint 只比较运行表现

每个候选模型只需要统计：

```text
1. 对固定对手集合的胜率。
2. 非法动作次数。
3. fallback 次数和 fallback rate。
4. 崩溃、异常和超时次数。
5. 平均、P95 和最大推理耗时。
6. 内存占用和提交包大小。
7. checkpoint 冷启动是否成功。
```

#### C. Champion 最终回归一次

选出 Champion 后，对最终 `submission.tar.gz` 运行一次完整回归：

```text
1. main.py 位于压缩包顶层。
2. deck.csv 位于压缩包顶层。
3. 所有模型和配置路径使用 /kaggle_simulations/agent/ 下的相对路径。
4. 无网络依赖。
5. 无训练数据和无关产物进入提交包。
6. 自对战 Validation Episode 能够完成。
7. 连续运行至少 100 局无崩溃、无非法动作。
8. 资源消耗满足 2 vCPU、12.2 GiB RAM 和提交大小限制。
```

---

## 五、其他 Prompt 的处理

以下内容不属于重复审计，继续保留：

```text
Prompt 03：模型消融和基线比较。
Prompt 04：完整推理闭环和动作输出验证。
Prompt 06：PPO 或后续强化学习前后的回归比较。
Prompt 07：提交包构建、导入、路径和运行测试。
```

不要因为本次精简而删除正常的单元测试、集成测试、tiny-batch overfit、训练指标或提交环境 smoke test。

---

## 六、统一措辞修改

在整个 Prompt 集中，将泛化的“审计”措辞替换成更准确的任务名称：

```text
数据结构审计       -> 数据读取测试
接口审计           -> 接口契约测试
语料审计           -> 缓存构建验证
隐藏信息审计       -> 可见性边界测试
搜索能力审计       -> simulator 状态复制测试
提交审计           -> submission 回归测试
```

只有下列情况允许继续使用“审计”一词：

```text
1. 发现 Actor 输入可能包含不可见信息。
2. 发现同一 episode 跨越不同 split。
3. 发现 action 标签与 observation 错位。
4. 发现训练标签的语义需要重新定义。
5. 发现新版本 simulator 改变了 Replay 或 action contract。
```

---

## 七、输出要求

完成修改后，只输出：

```text
1. 修改后的 Prompt 文件列表。
2. 每个文件删除了哪些重复审计任务。
3. 每个文件保留了哪些硬验证。
4. 是否新增了脚本或报告文件。
```

预期答案中第 4 项应为：

```text
没有新增独立审计脚本或审计报告。
```

不要运行全量 Replay 扫描，不要重新生成 7,500 局审计结果，不要修改正式模型源码，不要启动训练。

---

## 八、验收标准

修改完成后应满足：

```text
[ ] Prompt 01 不再创建 audit_interfaces.py、interface_audit.json、interface_audit.md。
[ ] Prompt 02 不再要求重复统计已确认的 action semantics 和 Replay 分布。
[ ] Prompt 02 的正确性信息统一进入 statistics.json 和 build.log。
[ ] Prompt 05 不再生成 search_capability_audit.md。
[ ] 未实现浅层搜索时，Prompt 05 不阻塞主流程。
[ ] Prompt 08 不再对每个 checkpoint 重复执行公共代码的可见性和 action contract 检查。
[ ] Champion 最终提交包仍会执行一次完整回归测试。
[ ] Prompt 03、04、06、07 的核心任务保持不变。
[ ] 没有新增独立审计脚本和大型审计报告。
[ ] 修改后的 Prompt 集可以直接推动实现、训练和提交，而不是再次停留在数据调查阶段。
```
