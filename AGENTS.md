# ReGenNet AI 入口

- 默认使用中文；文档、说明、计划、总结和代码注释都保持中文，除非用户明确要求英文。
- 每次进入实现前先做 design / plan，并把新增上下文、决策、取舍和结果记录到 `docs/ai/context/`；新文件使用 `YYYYMMDD-HHMMSS-文件名.md`，不要覆写历史上下文。
- 长期项目记忆已迁移到 `docs/ai/context/20260614-185753-regennet-key-context-extracted-from-agents.md`；更细的实验和设计以该目录下各阶段时间戳文档为准。
- 当前主线是 InterHuman 150/30/120 two-person motion forecasting，以及 SoMoFormer / T2P / JRT 等结构化预测 baseline 在 ReGenNet 内的同口径迁入。
- 论文结论必须保守：不能声称 multi-person forecasting、interaction-aware 或 explicit relation 首创；relation-aware 只可表述为相对 concat / parameter-matched concat 有稳定收益，不能写成全面优于 independent。
- 近期重点：P8 official SoMoFormer 已完成 3-seed，后续若继续扩展，优先在 ReGenNet 内做 `t2p_interhuman_xyz` 或 `jrt_xyz`，不要直接改外部官方仓库作为主实现。
