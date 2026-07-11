# Pokemon TCG 动态与时序特征交接

## 目标

当前阶段只完成特征构造与单卡实例融合预训练，不进入动作评分、行为克隆、Value Head 或 PPO。

卡牌表示固定分为四类：

1. 静态卡牌特征：卡牌身份、规则、逐攻击/特性/特殊效果 detail。
2. 单卡场上动态特征：HP、区域、附着能量、异常状态、进化与本回合出场状态。
3. 出场与长期记忆特征：单局出现记录、组合信息和热门度预留。
4. 全局状态特征：回合、双方公开区域计数、选择上下文、Ledger 和 Recent Events。

## 已确认事实

- 同一 Card ID 的 CSV 多行已经聚合为一张卡；多个攻击保持为多个独立 detail，费用、伤害和效果文字绑定不丢失。
- 静态预训练产物已经包含 128 维 `card_summary`、128 维 `detail_tokens`、mask、detail type 和 Card ID 映射，后续直接使用，不重新做静态基线。
- 当前静态数据共 1267 张卡，单卡 detail 最大数量为 7。攻击平均 1.232、最大 2；特性平均 0.336、最大 2；特殊效果平均 0.448、最大 3。
- Basic Energy 类型通过显式静态字段和 residual 保真；整卡平均攻击伤害、平均攻击费用、文本长度和整卡聚合攻击费用不再作为主要输入。
- detail 聚合只使用 attention，不再扩展 mean pooling 模式。
- `card_id` 表示卡牌规则身份，`serial` 表示一局中的具体实例，两者必须始终分开。
- 对局长度不同，训练数据必须按每局真实 `steps` 遍历，不能使用固定步长或跨局对齐。
- 静态或单卡辅助任务按 Card ID 划分，禁止同一 Card ID 进入不同 split；Replay 数据按完整 episode/日期划分，禁止同一局跨 split。

## 当前实现状态

已经完成：

- 静态 summary/detail 数据结构、编码、导出和兼容旧接口。
- Observation 解析、动态实例 schema、GameMemory、Replay decision sample 构造。
- Static adapter、以 card summary 为 query 的 detail attention、DynamicInstanceEncoder、浅层 CardInstanceFusion 初版前向接口。
- Ledger、Recent Events、Board tokenization 和轻量 Board Transformer 的初版接口。
- Kaggle 代码数据集与薄 kernel 入口；Notebook 直接挂载代码和 replay 数据集。
- 线上 replay 最小结构探测，结果已经单独保留，不需要重复跑。

尚未完成：

- 尚未用多日线上 replay 正式提取完整训练特征并统计字段覆盖率。
- 尚未用真实静态产物和线上 batch 完成单卡融合端到端验证。
- 当前 detail attention 由 card summary 查询，动态状态条件化 detail 的 cross-attention 尚未实现。
- 32 维 appearance 接口已预留，但当前只是单局 serial 记忆的临时实现；组合特征与热门度各占一半的正式定义和数据统计尚未完成。
- CardInstanceFusion 四项辅助任务尚未正式实现和训练。
- 与 simulator 一致的攻击能量支付 resolver 尚未接入，特殊能量 unresolved mask 也未完成。
- Board Transformer、ActionEncoder 和后续策略网络尚未进入正式训练。

## 线上数据结论

episode index 只提供每日数据集 manifest，不直接保存 replay。每日数据集内是普通 JSON 文件；每个 replay 的 `steps` 是变长序列，每步包含两个 agent entry。有效训练样本从 `observation.select` 非空的决策点生成。

已验证的两个日期样例均可正常解析：训练日首局 111 步，保留日首局 181 步；各抽取 10 个决策样本，parser error 均为 0。观察到单样本最大 22 个卡牌实例、6 个选项，当前最大 token 估计为 43。

最近若干天应作为时间保留集；首次正式提取只读取少量已挂载日期确认分布，不扫描完整 20GB 数据。

## 模型接入

按 Card ID 读取静态 summary 和 detail，按 serial 构造当前单卡动态状态。当前代码先聚合静态 detail，再与动态表示浅层融合；下一步要改成由动态状态条件化 detail attention，输出 128 维 `card_instance_token`。单卡辅助预训练放在该融合之后、Board Transformer 之前。

## 下一步工作

1. 直接读取已挂载的早期每日 replay，生成有限规模的真实 decision-point 特征集；记录字段缺失率、实例数、detail 对齐率、事件数和 token 长度分布。
2. 将单局 serial 记忆从 appearance 组中拆出；appearance 正式保留为组合特征与热门度两半，并先通过 replay 数据统计其可用标签和时间窗口。
3. 将 detail attention 改为受动态状态条件化，再用真实 batch 接入现有静态产物，验证 summary、detail 和动态实例确实共同进入 `card_instance_token`，并完成前向、反向和 mask 测试。
4. 复用或补齐 simulator 的正式能量支付逻辑，统一 canonical energy vocabulary；无法可靠解析的特殊能量样本进入 unresolved 清单并屏蔽监督。
5. 实现 CardInstanceFusion 辅助数据集、模型、损失和指标：Attack Affordability、Energy Deficit、Dynamic Retention、Static Retention。
6. 先用 32-128 个样本做 tiny-batch overfit；达标后再按 Card ID 的 80/10/10 split 做完整辅助预训练，并另外报告 instance validation。
7. 辅助预训练通过后保存融合 checkpoint，再接 Board Transformer 和 ActionEncoder；不要提前进入策略训练。

## 执行约束

- 不再重复探测已经确认的 replay JSON 结构，也不为路径问题反复提交 kernel。
- 长实验每 3-5 分钟读取一次状态，只报告变化、失败或完成。
- 小规模读取和 smoke test 优先 CPU；只有确认数据量和计算量值得时才启用 GPU。
- 发现非原则性问题继续修复推进；只有数据语义、可见性泄漏或 split 污染等原则问题才暂停。
