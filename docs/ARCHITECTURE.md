# 模型与数据架构

## 模块边界

colleague 静态模块完整负责 CSV 读取、聚合、特征、模型、训练、评估和导出。`static_card/` 当前只是其未来落位目录。根仓库不维护替代静态实现，只通过 `StaticCardAdapter` 消费正式产物。

正式依赖只允许 `scripts/` 和 `training/` 调用 `data/`、`models/`；正式模块不得反向导入 CLI。可复用 replay 审计位于 `data/replay_feature_audit.py`，`scripts/audit_replay_features.py` 只是命令行入口。

正式依赖方向固定为：

```text
colleague static module
→ StaticCardAdapter
→ DynamicInstanceEncoder
→ CardInstanceFusion
→ BoardTokenizer
→ BoardTransformer
```

真实 artifact contract 尚未确定。当前 `StaticCardAdapter` 只定义输出协议并 fail-fast；它不猜测文件名、兼容旧产物或构造虚假特征。

## 静态输出协议

动态侧预期 `StaticCardFeatureOutput` 提供：

- `card_summary`
- `known_mask`
- 可选 `detail_tokens`
- 可选 `detail_mask`
- 可选 `detail_type_ids`

具体 dtype、shape、Card ID 映射、unknown/padding 语义、detail 顺序和 manifest 字段由 colleague contract 决定后固化。

## 动态卡牌实例

`DynamicInstanceEncoder` 编码当前公开的 owner、zone、field role、HP、伤害、能量、状态、Tool、进化、可见性和出现时序。`CardInstanceFusion` 使用动态表示查询静态 detail，形成 card instance token。同一 Card ID 的不同 serial 保持独立动态状态。

单元测试通过 `tests/fakes/static_card_adapter.py` 提供可控静态张量；该 fake 不进入生产代码。

## Memory 与 Board

memory 维护双方 Ledger 与 Recent Events，但目前仍是接口原型。正式目标是只保存当前 observation、公开 logs、己方已知牌组及公开证据，不引入对手隐藏身份。

Board token 顺序固定为：

```text
[STATE]
[DECISION]
[MATCH]
[SELF_LEDGER]
[OPP_LEDGER]
[RECENT_EVENT] * N
card instance tokens
```

`BoardTransformer` 输出 `tokens`、`mask` 和 `state_embedding`；`state_embedding = encoded[:, 0]`。Board 与 memory 尚未达到策略训练所需的完整实现。

## 后续策略层

真实静态接入和动态训练恢复后，再增加合法动作编码、行为克隆、Value、teacher/student 蒸馏和 self-play。正式 Agent 只对引擎提供的合法选项评分，并保持公开信息边界。
