# GitHub Settings Baseline

这份文档定义 `Project` 研究仓库的 GitHub 后台设置基线。

## 一、仓库定位

- 仓库：`Project`
- 类型：研究与实验主仓库
- 当前可见性：`public`
- 目标：保证研究资产、实验代码、论文复现与状态文档可协作，同时避免把研究仓库按产品仓库方式管理

## 二、仓库级设置

- Visibility: `public`
- Issues: `on`
- Projects: `on`
- Wiki: `off`
- Discussions:
  - 推荐值：`off`
  - 备注：需要继续在网页端核对是否真正关闭
- Merge methods:
  - `squash`: `on`
  - `merge commit`: `off`
  - `rebase`: `off`
- Auto-merge: `on`
- Automatically delete head branches: `on`
- Always suggest updating pull request branches: `on`
- Web commit signoff required: `off`
- Release immutability: `off`

## 三、`main` 保护基线

- Require pull request: `on`
- Required status checks:
  - `unit-tests`
- Require branch up to date: `on`
- Require 1 approval: `on`
- Require CODEOWNERS review: `on`
- Dismiss stale reviews: `on`
- Require conversation resolution: `on`
- Enforce for admins: `on`
- Allow force pushes: `off`
- Allow deletions: `off`
- Require last push approval:
  - 推荐值：`on`
  - 备注：会提高 owner 自己快速合并小型文档/治理 PR 的摩擦

## 四、Copilot Review 基线

统一要求：

- `Use custom instructions when reviewing pull requests`: `on`
- 仓库级指令文件：
  - `.github/copilot-instructions.md`

研究仓库的 Copilot review 重点应该放在：

- `src/diffaudit/`
- `tests/`
- `scripts/`
- `configs/`
- 复现风险、路径依赖、provenance、状态口径、CLI/manifest 回归

研究仓库的 Copilot review 应主动降噪：

- `references/`
- PDF
- 长篇论文笔记
- 纯实验产物

## 五、安全设置基线

- Dependency graph: `on`
- Dependabot alerts: `on`
- Dependabot security updates: `on`
- Secret scanning: `on`
- Push protection: `on`

补充建议：

- version updates 可以开，但频率不宜太高
- ML 依赖不应被频繁自动漂移

## 六、自动合并

- `Allow auto-merge`: `on`

前提：

- 通过 `main` 的 required checks
- 满足 review 规则
- 合并方式统一走 `squash`

## 七、当前需要人工继续核对的项

- `Discussions` 是否已真正关闭
- Copilot 自动 review 是否按你的预期启用

## 八、关联文档

- `CONTRIBUTING.md`
- `docs/github-collaboration.md`
- `.github/copilot-instructions.md`
