# 脚本目录

这里放可重复执行的小工具脚本。

## 原则

- 一个脚本只做一件事
- 优先服务环境验证、资产检查、实验整理
- 不要把一次性的本地试验命令直接塞进这里

## 便携规则

- `LocalOps/paper-resource-scheduler` 只是本地共享机器治理工具，不是研究仓硬依赖
- 直接运行这些脚本时，不需要本地 scheduler；scheduler 只是可选能力
- 推荐通过环境变量覆盖本机路径：
  - `DIFFAUDIT_WORKSPACE_ROOT`
  - `DIFFAUDIT_RESEARCH_PYTHON`
- 如果不显式传参，DPDM 相关脚本会优先用环境变量，再退回到当前 `Research/scripts` 所在仓库位置推导默认路径

## 当前补充

- `run_local_checks.py`
  - 运行研究仓本地质量门禁
  - 支持 `--python` 或环境变量 `DIFFAUDIT_RESEARCH_PYTHON`
- `prepare_clid_local_bridge.py`
  - 使用本机 `configs/assets/team.local.yaml` 准备 CLiD 本地桥接运行
  - 不再依赖仓库里提交的作者机器路径；需要其他路径时显式传 `--asset-config`
- `validate_attack_defense_table.py`
  - 校验 admitted 统一 attack-defense 总表的最小机器合同
  - 强制 `source` 使用 repo-relative 路径，并检查灰盒 `quality / cost / adaptive_check / provenance_status`
- `validate_intake_index.py`
  - 校验 intake index 中的 manifest、assets root 和 admitted 合同
- `validate_local_api_registry_alignment.py`
  - 校验研究仓 intake 与相邻 `Runtime-Server` registry seed 的 promoted 入口是否一致
- `monitor_gsa_sequence.py`
  - 汇总当前 `GSA` 训练链的 `phase / active split / latest checkpoint / latest epoch-step`
  - 适合低频监控训练资产树，不直接代表 admitted runtime 摘要是否已写回系统读链
- `launch_dpdm_training.ps1`
  - 启动单个 DPDM 训练任务
  - 支持通过 `DIFFAUDIT_WORKSPACE_ROOT` 和 `DIFFAUDIT_RESEARCH_PYTHON` 做便携覆盖
- `launch_dpdm_target_and_shadows.ps1`
  - 串起 target + shadows 的顺序训练
  - 默认相对 `Research/scripts` 解析子脚本，不依赖作者本机绝对路径
- `launch_dpdm_shadow_sequence.ps1`
  - 按顺序启动 shadow 训练
  - 不要求本地 scheduler；直接在协作者自己的 CPU/GPU 环境即可运行
