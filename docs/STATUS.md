# 项目状态

更新时间：2026-07-16

## 当前结论

静态模块将完整采用 colleague 的脚本，包括 CSV 读取、同 Card ID 聚合、特征、模型、训练、评估和导出。`static_card/` 当前只是正式模块预留位置，仓库内没有可替代 colleague 实现的静态模型。

根仓库负责 replay、动态卡牌实例、memory、Board、动作和策略。`models/static_card_adapter.py` 是唯一跨模块边界；真实 artifact contract 尚未确定，正式动态训练和 benchmark 当前暂停。

## 已保留能力

- observation 与变长 replay 解析、决策点样本和公开信息边界。
- 结构化动态卡牌字段与 `DynamicInstanceEncoder`。
- `CardInstanceFusion`、动态辅助任务和训练框架原型。
- `GameMemoryState`、双方 Ledger 和 Recent Events 接口原型。
- Board token 化与 Transformer 接口原型。

Board 的固定前缀顺序为：

```text
[STATE]
[DECISION]
[MATCH]
[SELF_LEDGER]
[OPP_LEDGER]
[RECENT_EVENT]*
card instance tokens
```

`state_embedding` 取 Transformer 编码后的 `[STATE]`，即 `encoded[:, 0]`。

## 明确暂停项

- `StaticCardAdapter.from_artifacts()` 在 contract 未配置时抛出 `StaticArtifactContractNotConfigured`。
- 正式 adapter 不生成零 summary、假 detail 或自动 known 标记。
- 动态训练 Kernel、训练入口和两个 benchmark 入口在运行前检查 `ready`，未接入即非零退出。
- 动态 Kernel 正式位置为 `kaggle/kernels/dynamic_training/`；暂停摘要写入 `/kaggle/working/outputs/dynamic_card_training/`。
- 旧静态路线已退出主线；历史实现只通过 Git 历史或明确标记的实验记录查阅。

## 原型边界

Board 和 memory 目前是接口原型。Ledger 尚未形成完整的长期认知表；memory 的 reset、clone、序列化、幂等更新和 shuffle 后知识降级仍需补齐。ActionEncoder、行为克隆、Value、蒸馏和 self-play 尚未进入正式主线。

## 下一接入点

收到 colleague 模块后，先确认 artifact contract 和 manifest，再实现 `StaticCardAdapter.from_artifacts()` 与 `forward_features()`。完成真实接入、对齐验证和端到端测试后，才恢复动态正式训练。

恢复时使用 `configs/dynamic_card_fusion/formal_20260712.json` 和 `configs/dynamic_card_fusion/smoke_20260712.json`，并根据新 contract 替换训练侧仍待重构的静态 catalog 参数。
