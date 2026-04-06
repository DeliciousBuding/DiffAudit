# Project 仓库协作指南

这是 `D:\Code\DiffAudit\Project` 的仓库级协作文件。

先明确边界：

- `Project` 是纯研究仓库
- 它承载研究代码、实验、论文索引、研究文档、结果产物
- 它可以保存作为飞书内容源的 Markdown
- 它不承载本机飞书运维规则
- 它不承载服务器部署细节

对应角色：

- 主要负责：本地模型验证实验 Agent
- 可协同：总管理 Agent

以下原有研究协作规则继续有效。

---

# DiffAudit 仓库协作指南

这是一个研究仓库，不是普通业务系统。

## 核心目标

优先级按以下顺序执行：

1. 可复现
2. 实验边界清晰
3. 最短路径集成
4. 文档和证据完整
5. 最后才是界面和流程美化

## 默认研究主线

当前默认主线是：

`扩散模型上的 black-box membership inference`

白盒和灰盒可以推进，但不得破坏黑盒主线的代码结构。三条线都应复用：

- config 结构
- adapter 结构
- dry-run 机制
- 结果记录方式

## 工作区规则

多人协作统一按工作区拆分：

- `workspaces/black-box/`
- `workspaces/white-box/`
- `workspaces/gray-box/`
- `workspaces/implementation/`

这些目录只放：

- 阅读笔记
- 复现计划
- 任务归属
- 阻塞记录
- 阶段总结

共享可执行代码统一放在 `src/diffaudit/`。

## 编码优先级

优先补：

- 配置 schema
- planner / adapter
- dry-run 验证
- 结果记录
- import 和执行 smoke 测试

不要先做：

- 复杂前端
- 平台化用户系统
- 数据库重系统设计
- 没有真实攻击路径支撑的大而全框架

## 第三方代码规范

`third_party/` 只允许放最小必要 vendored 子集。

要求：

- 体量尽可能小
- 必须保留来源说明
- 只做必要补丁，不随意大改 upstream 逻辑
- `external/` 只用于本地探索 clone，禁止提交

## 实验资产规范

以下都属于实验资产：

- checkpoint
- `flagfile.txt`
- dataset root
- member split 文件

如果资产缺失：

- 优先实现 `dry-run` 或 `blocked` 状态
- 不要伪造实验完成
- 不要产出没有运行证据的结论

必须区分：

- `code-ready`
- `asset-ready`
- `experiment-ready`

## 研究诚信规则

- 没有真实运行证据，不得声称“复现成功”
- smoke 输出不能写成 benchmark 结果
- 黑盒结论不能直接外推到白盒或灰盒
- 单篇论文的假设不能表述成普遍事实

## 测试规则

新增仓库行为，默认先写测试再写实现。

至少维持以下测试能力：

- config loading
- attack planning
- adapter preparation
- dry-run validation
- vendored 模块 import smoke test

## 文档链接规则

所有 Markdown 文件中的链接必须使用**相对路径**，禁止使用绝对路径。

示例：

- 从 `docs/README.md` 引用 `docs/reproduction-status.md` → `[reproduction-status.md](reproduction-status.md)`
- 从 `docs/repo-map.md` 引用根目录 `README.md` → `[README.md](../README.md)`
- 从 `src/diffaudit/README.md` 引用根目录 → `[README.md](../../README.md)`

理由：仓库会被推送到 GitHub，绝对路径（如 `D:/Code/...`）在 GitHub 上无法解析，导致链接失效。

## 文档同步规则

新增攻击线或重大能力时，至少同步更新：

- `README.md`
- `configs/`
- `references/`
- `docs/`

仓库始终要能回答：

- 现在做到哪了
- 差什么
- 缺什么资产
- 用什么命令验证

## 多人协作纪律

- 一次尽量只负责一个方向
- 分支应聚焦于 `black-box` / `white-box` / `gray-box` / `implementation`
- 提交要小步快跑
- 修改共享接口前先补文档
- 被资产阻塞时必须显式记录

## 公开仓库约束

提交大文件前必须检查：

- 总体量
- 最大单文件大小
- 是否已更新资料索引

## 输出风格

研究结论优先按这四项组织：

- 假设
- 方法
- 证据
- 阻塞项

不要使用“应该可以”“差不多能跑”这类模糊表述。
