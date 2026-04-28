# 入手指南

这份文档面向第一次进入 DiffAudit 仓库的组员。

## 先做什么

建议按这个顺序入手：

1. 阅读根目录 [README.md](../README.md)
2. 阅读 [docs/teammate-setup.md](teammate-setup.md)
3. 按 [docs/data-and-assets-handoff.md](data-and-assets-handoff.md) 补齐 `Download\` 资产并绑定本机路径
4. 按 [docs/command-reference.md](command-reference.md) 跑一次环境验证、资产探针和 `dry-run`
5. 阅读 [docs/environment.md](environment.md)
6. 阅读 [docs/github-collaboration.md](github-collaboration.md)
7. 进入自己负责的工作区

## 你应该先知道的几件事

- 这是研究仓库，不是产品仓库
- 当前不是“只做黑盒”，而是三线并行、分层推进
- 当前最成熟的“攻击 + 防御”主讲闭环是 `PIA`
- 代码、资产、实验三者必须分开表述
- 没有真实运行证据，不要说“复现成功”

## 第一次本地验证

```powershell
conda activate diffaudit-research
python scripts/bootstrap_research_env.py --install
python scripts/verify_env.py
python -m diffaudit --help
python -m pytest tests/test_cli_module_entrypoint.py tests/test_render_team_local_configs.py -q
python -m diffaudit plan-variation --config configs/attacks/variation_plan.yaml
python -m diffaudit run-variation-synth-smoke --workspace experiments/variation-synth-smoke-local
python -m diffaudit plan-pia --config configs/attacks/pia_plan.yaml
python -m diffaudit probe-gsa-assets --repo-root workspaces/white-box/external/GSA --assets-root workspaces/white-box/assets/gsa
```

第一次使用前，先复制 `configs/assets/team.local.template.yaml` 为本地 `configs/assets/team.local.yaml`，再填写自己的真实路径。大数据集、权重和 supplementary 包默认放在仓库外的 `<DIFFAUDIT_ROOT>/Download/`，不要放进 `Research/external/`。

如果当前 shell 还没激活 conda，也可以直接写成：

```powershell
conda run -n diffaudit-research python scripts/verify_env.py
conda run -n diffaudit-research python -m diffaudit --help
conda run -n diffaudit-research python -m pytest tests/test_cli_module_entrypoint.py tests/test_render_team_local_configs.py -q
conda run -n diffaudit-research python -m diffaudit probe-secmi-assets --config configs/attacks/secmi_plan.yaml
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
4. 先跑对应方向的 `probe-*` 命令
5. 最后才跑真实实验
