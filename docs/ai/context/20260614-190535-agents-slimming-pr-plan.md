# AGENTS 精简 PR 计划

## 目标

将根目录 `AGENTS.md` 的长上下文迁移到 `docs/ai/context/`，并通过新分支、PR、review、merge 的方式同步到 `main`。

## 分支策略

- 当前基线分支：`main`
- 新工作分支：`chore/slim-agents-context`
- 推送远端：`fork`
- PR 目标：`origin/main`

## 提交范围

只提交本次 AGENTS 精简相关文件：

- `AGENTS.md`
- `docs/ai/context/20260614-185753-agents-slimming-plan.md`
- `docs/ai/context/20260614-185753-regennet-key-context-extracted-from-agents.md`
- `docs/ai/context/20260614-190535-agents-slimming-pr-plan.md`

不提交当前工作区中已有的其他未跟踪文档。

## 验收

- 新分支包含上述文件变更。
- `git diff --check` 通过。
- PR review 客观检查上下文迁移是否丢失关键约束。
- PR 合并到 `main` 后，本地 `main` 与远端同步。
