# 工作区说明

这个目录用于多人协作时按方向拆分工作区，避免大家把阅读笔记、实验计划和共享代码混在一起。

## 当前组织说明

这里的历史文档中可能仍然出现旧 owner 名称，例如 `research_leader`。

这些名称只代表历史记录，不代表当前活跃配置。

当前活跃组织只认：

- `Leader`
- `Researcher`
- `Developer`

## 子目录

- `black-box/`：黑盒方向
- `white-box/`：白盒方向
- `gray-box/`：灰盒方向
- `implementation/`：共享工程实现

## 原则

- 阅读笔记、任务认领、阶段总结放工作区
- 共享代码放 `src/diffaudit/`
- 配置放 `configs/`
- 实验输出放 `experiments/`

## 当前入口

- 黑盒计划：`workspaces/black-box/plan.md`
- 白盒计划：`workspaces/white-box/plan.md`
- 灰盒计划：`workspaces/gray-box/plan.md`
- 工程计划：`workspaces/implementation/plan.md`
- 总体复现状态：`docs/reproduction-status.md`

## 状态面板约定

每个方向的 `plan.md` 建议至少维护这些字段：

- `owner`
- `scope`
- `status`
- `blocked by`
- `next command`
- `last updated`

这样团队成员进入任一工作区时，能先看状态，再看背景。
