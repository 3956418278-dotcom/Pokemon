# 实验结论记录

本文件只保留会影响后续方案选择的实验结论。完整旧代码与原始讨论可从 Git 历史读取。

## 旧共享 PPO round-robin

旧实验使用 8 套 baseline decks，由同一个 PPO policy 同时控制主客双方，只记录 player 0 的 transition。训练运行约 50 个以上更新 batch 后，loss 在约 `0.1–0.5` 间震荡，没有形成可信的性能提升证据，因此停止继续扩展。

主要问题：

- 对手与训练策略同步变化，环境高度非平稳。
- 单一策略同时覆盖多套牌组，却没有 deck identity 输入。
- 终局胜负奖励稀疏。
- loss 无法替代固定对手池上的胜率评估。

后续 self-play 使用冻结 checkpoint 或历史策略池，并报告按牌组、先后手和对手版本拆分的胜率。

## 先后手偏差

早期线上样本显示先手方可能具有约 10 个百分点的全局胜率优势。该结果仍需按牌组 archetype、样本日期和代理强度分层验证。模型输入保留先后手信息，评估分别报告 first/second player 指标。

## Oracle teacher

训练期可以建立读取完整隐藏状态的 teacher，正式 student 只读取比赛允许的 observation。Teacher 用于：

- 生成 action logits 与 value target。
- 标记公开信息策略与全信息策略的严重分歧点。
- 帮助 student 学习对手隐藏资源的概率影响。

Teacher 标签只使用当前隐藏状态。未来随机结果通过多次 rollout 估计期望值，不作为确定答案直接泄露给 student。

## 保留的训练方向

1. 静态卡牌理解。
2. 动态单卡与局面状态编码。
3. 高质量 replay 行为克隆和 Value 学习。
4. Oracle distillation。
5. 冻结快照或 league self-play。
