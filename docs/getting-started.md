# 入手指南

这份文档面向第一次进入 DiffAudit 仓库的组员。

## 先做什么

建议按这个顺序入手：

1. 阅读根目录 [README.md](/D:/Code/DiffAudit/Project/README.md)
2. 阅读 [docs/environment.md](/D:/Code/DiffAudit/Project/docs/environment.md)
3. 阅读 [docs/github-collaboration.md](/D:/Code/DiffAudit/Project/docs/github-collaboration.md)
4. 进入自己负责的工作区
5. 跑一次环境验证和 `dry-run`

## 你应该先知道的几件事

- 这是研究仓库，不是产品仓库
- 当前主线优先是 `black-box`
- 代码、资产、实验三者必须分开表述
- 没有真实运行证据，不要说“复现成功”

## 第一次本地验证

```powershell
conda activate diffaudit-research
$env:PYTHONPATH='src;.'
python scripts/verify_env.py
python -m unittest
python -m diffaudit dry-run-secmi --config configs/attacks/secmi_plan.yaml --repo-root third_party/secmi
```

## 你应该去哪个工作区

- 负责黑盒：`workspaces/black-box/`
- 负责白盒：`workspaces/white-box/`
- 负责灰盒：`workspaces/gray-box/`
- 负责共享工程：`workspaces/implementation/`

## 做事方式

1. 先写计划
2. 再写配置
3. 再写 adapter / probe
4. 先跑 dry-run
5. 最后才跑真实实验
