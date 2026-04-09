# 脚本目录

这里放可重复执行的小工具脚本。

## 原则

- 一个脚本只做一件事
- 优先服务环境验证、资产检查、实验整理
- 不要把一次性的本地试验命令直接塞进这里

## 当前补充

- `run_local_checks.py`
  - 运行研究仓本地质量门禁
  - 现在支持 `--python` 或环境变量 `DIFFAUDIT_RESEARCH_PYTHON`
  - 适合在不同 conda / venv / portable Python 解释器下复用同一套检查入口
- `init_variation_query_set.py`
  - 为 `variation/Towards` 生成最小真实 query-image 目录骨架
  - 只解决目录结构和接手模板，不伪装成真实资产恢复
- `monitor_gsa_sequence.py`
  - 汇总当前 `GSA` 训练链的 phase、active split、latest checkpoint、latest epoch/step
  - 用于持续跟踪白盒串行训练与 runtime 是否已经接上
