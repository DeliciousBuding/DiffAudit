# 配置说明

这个目录用于放可声明、可版本化的实验配置。

## 当前目录

- `attacks/`：攻击配置与超参数
- `benchmarks/`：基准或 smoke 场景配置
- `assets/`：本地资产映射示例，不直接作为共享实验配置提交

## 使用原则

- 共享配置里不要写个人私有资产的真实路径
- 需要本地路径时，先参考 `assets/example.local.yaml`
- `attacks/*.yaml` 负责描述攻击设定，不负责泄露个人机器目录结构

## 当前建议

- `attacks/secmi_plan.yaml`：作为 `SecMI` 真实资产探针与 dry-run 的模板
- `attacks/pia_plan.yaml`：作为 `PIA` 探针、dry-run 与 smoke 的模板
- `attacks/clid_plan.yaml`：作为 `CLiD` 条件场景探针与 dry-run 的模板
- `benchmarks/secmi_smoke.yaml`：作为最小 smoke 配置
- `assets/example.local.yaml`：作为个人本地资产映射参考
