# 配置说明

这个目录用于放可声明、可版本化的实验配置。

## 当前目录

- `attacks/`：攻击配置与超参数
- `benchmarks/`：基准或 smoke 场景配置
- `assets/`：本地资产映射模板和占位示例，不提交个人真实路径

## 使用原则

- 共享配置里不要写个人私有资产的真实路径
- 需要本地路径时，优先参考 `assets/team.local.template.yaml`
- `attacks/*.yaml` 负责描述攻击设定，不负责泄露个人机器目录结构
- 原始数据集、权重和 supplementary 包默认放在仓库外的 `<DIFFAUDIT_ROOT>/Download/`，再通过本地 `assets/team.local.yaml` 绑定到各攻击线

## 当前建议

- `attacks/secmi_plan.yaml`：作为 `SecMI` 真实资产探针与 dry-run 的模板
- `attacks/pia_plan.yaml`：作为 `PIA` 探针、dry-run 与 smoke 的模板
- `attacks/clid_plan.yaml`：作为 `CLiD` 条件场景探针与 dry-run 的模板
- `attacks/recon_plan.yaml`：作为纯黑盒 reconstruction 攻击探针与 smoke 的模板
- `attacks/variation_plan.yaml`：作为 API-only 黑盒 variation 攻击探针与 smoke 的模板
- `benchmarks/secmi_smoke.yaml`：作为最小 smoke 配置
- `assets/example.local.yaml`：无真实机器绑定的占位示例
- `assets/team.local.template.yaml`：作为队友接仓时统一填写的三线本地资产模板

## 不要提交什么

- `assets/team.local.yaml`
- `assets/*.local.yaml` 中的个人机器路径
- 指向 `<DIFFAUDIT_ROOT>/Download/` 内大文件的本机绝对路径快照

如果脚本需要当前机器的真实资产位置，先复制 `assets/team.local.template.yaml` 到被忽略的 `assets/team.local.yaml`，再把该文件路径传给脚本或使用脚本默认值。
