# AGENTS 精简计划

## 背景

根目录 `AGENTS.md` 当前承担了长期项目记忆职责，内容已经过长。用户要求将关键信息提取到 `docs/ai/context/`，并让根 `AGENTS.md` 只保留极简描述。

## 必须保留

- 后续会话需要知道详细上下文在 `docs/ai/context/`。
- 当前研究主线是 InterHuman two-person motion forecasting 与 SoMoFormer/JRT/T2P 等结构化预测基线迁入。
- 论文结论边界必须保留：不能声称 multi-person forecasting、interaction-aware 或 explicit relation 首创。
- P5/P6/P7/P8 的主结果和边界需要可追溯到既有上下文文件。

## 迁移方案

- 新建 `docs/ai/context/20260614-185753-regennet-key-context-extracted-from-agents.md`，作为从旧 `AGENTS.md` 提取出的关键上下文索引。
- 将根 `AGENTS.md` 改为极简入口：只说明语言偏好、上下文入口、当前主线和近期工作边界。
- 不修改、重命名或删除任何历史 `docs/ai/context/` 文件。

## 验收

- `AGENTS.md` 大幅缩短，适合作为每次会话入口。
- 关键事实仍可从新增上下文提取文档和原有阶段文档追溯。
- 不改变代码、不启动训练、不改变实验结果。
