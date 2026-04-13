# 2026-04-09 White-Box Same-Protocol Bridge

## 目的

这份文档定义当前白盒唯一 active GPU 问题。

目标不是继续强化 `GSA` 攻击结果，而是把已经 admitted 的 `GSA rerun1` 与当前 defended 主 rung `W-1 strong-v3 full-scale` 放到同一个更可解释、可复现、可退出的比较协议面上。

## Decision Review（强制三选一）

决策面材料见：`workspaces/white-box/2026-04-09-whitebox-bridge-decision-review.md`。

硬约束保持不变：当前 `batch32 diagnostic` 仍是 `runtime-smoke`，不得升 admitted，也不得用于改写 admitted 合同口径。

截至 `2026-04-10` 的当前推荐分支为：

- `保持冻结`

含义是：

- bridge 当前仍是 active 主问题
- 但当前没有新的 decisive evidence 支撑继续扩大
- 也没有足够的结构性 blocker 支撑失败收口
- 因此在下一轮决策节点前，不新增新的 bridge 长 GPU 任务，不改 admitted 合同

## 当前主合同

- owner: `research_leader`
- status: `active`
- GPU rule:
  - GPU 允许使用，但同一时段只允许一个白盒主线问题占用主注意力
  - 当前问题就是 `GSA rerun1` 与 `W-1 strong-v3 full-scale` 的 same-protocol bridge
- admitted attack baseline:
  - `workspaces/white-box/runs/gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1/summary.json`
- defended comparator baseline:
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-full-rerun8-20260408/summary.json`
- reference rung:
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv2-3shadow-full-20260408/summary.json`

## 协议不变量

- dataset family 固定为 `CIFAR-10`
- attack family 固定为 `DDPM / GSA 1k-3shadow`
- defended comparator 固定为 `DPDM / W-1`
- machine-readable 结果仍只认现有 `summary.json / manifest / intake / unified table`
- admitted 口径继续区分：
  - `attack main evidence`
  - `defended main rung`
  - `reference rung`
- 不新增新的系统合同名词；继续只用 `catalog / evidence / job`

## 成功条件

满足下面任一条，bridge 即可收口：

1. 产出一份新的 white-box bridge runtime summary，且可明确解释与 `GSA rerun1` / `W-1 strong-v3 full-scale` 的协议关系
2. 产出一份稳定、可复述的失败模式，能够说明为什么当前 `DPDM/W-1` 还不能进入与 `GSA rerun1` 同协议的 benchmark 面

## 失败模式也算结果

下面几类结果都允许作为正式 bridge 输出：

- 架构不兼容，导致现有 `GSA` 提取器无法直接消费 `DPDM` checkpoint
- 数据切分、训练目标或评估规模仍无法收敛到同一个协议面
- 脚本入口只能在作者本机路径下工作，未达到 portable 要求

关键要求是：失败模式必须可复现、可复述、可指向具体 root cause。

## 明确不做

- 不重跑已冻结的 `PIA GPU128 / GPU256 / GPU512`
- 不把 admitted `GSA rerun1` 再当成 attack-strengthening 任务
- 不把 `LocalOps/paper-resource-scheduler` 作为研究仓硬依赖
- 不在 `SecMI` 真实 `flagfile + checkpoint root` 未到位时抢 GPU
- 不并行开第二条白盒长任务

## 启动前检查清单

- `GSA rerun1` admitted attack baseline 路径存在
- `W-1 strong-v3 full-scale` defended baseline 路径存在
- `Research/scripts` 下的 DPDM 启动脚本已支持 portable 路径覆盖
- `Research/scripts/README.md` 已说明 scheduler 只是可选本地治理工具
- `validate_attack_defense_table.py` 已通过
- `validate_intake_index.py` 已通过
- `validate_local_api_registry_alignment.py` 已通过

## 当前建议启动路径

建议先走现有入口，不发明新的 scheduler 依赖：

1. 用 `launch_dpdm_training.ps1` 启动 target-member 训练
2. 优先用 `launch_dpdm_training.ps1` 人工串行启动 `shadow-01/02/03`
3. `launch_dpdm_shadow_sequence.ps1` 目前只适合作为辅助入口，不应再假设它能无人工干预地完成整条序列

当前固定启动命令：

```powershell
$env:DIFFAUDIT_WORKSPACE_ROOT = "D:/Code/DiffAudit"
$env:DIFFAUDIT_RESEARCH_PYTHON = "C:/Users/Ding/miniforge3/envs/diffaudit-research/python.exe"
powershell -ExecutionPolicy Bypass -File D:/Code/DiffAudit/Research/scripts/launch_dpdm_training.ps1 `
  -DataPath D:/Code/DiffAudit/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/target-member `
  -Workdir runs/dpdm-cifar10-targetmember-eps10-gpu-sameproto-20260409 `
  -Epochs 3 `
  -SigmaNoiseSamples 2

powershell -ExecutionPolicy Bypass -File D:/Code/DiffAudit/Research/scripts/launch_dpdm_training.ps1 `
  -DataPath D:/Code/DiffAudit/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/shadow-01-member `
  -Workdir runs/dpdm-cifar10-shadow01-eps10-gpu-sameproto3shadow-r2-20260409 `
  -Epochs 3 `
  -SigmaNoiseSamples 2 `
  -MasterPort 6132

powershell -ExecutionPolicy Bypass -File D:/Code/DiffAudit/Research/scripts/launch_dpdm_training.ps1 `
  -DataPath D:/Code/DiffAudit/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/shadow-02-member `
  -Workdir runs/dpdm-cifar10-shadow02-eps10-gpu-sameproto3shadow-batch32-r1-20260409 `
  -BatchSize 32 `
  -Epochs 3 `
  -SigmaNoiseSamples 2 `
  -MasterPort 6333

powershell -ExecutionPolicy Bypass -File D:/Code/DiffAudit/Research/scripts/launch_dpdm_training.ps1 `
  -DataPath D:/Code/DiffAudit/Research/workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/shadow-03-member `
  -Workdir runs/dpdm-cifar10-shadow03-eps10-gpu-sameproto3shadow-batch32-r1-20260409 `
  -BatchSize 32 `
  -Epochs 3 `
  -SigmaNoiseSamples 2 `
  -MasterPort 6334

conda run -n diffaudit-research python -m diffaudit run-dpdm-w1-multi-shadow-comparator `
  --workspace workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-sameproto3shadow-batch32-diagnostic-20260409 `
  --target-checkpoint-path external/DPDM/runs/dpdm-cifar10-targetmember-eps10-gpu-sameproto-20260409/checkpoints/final_checkpoint.pth `
  --shadow-checkpoint-paths `
    external/DPDM/runs/dpdm-cifar10-shadow01-eps10-gpu-sameproto3shadow-r2-20260409/checkpoints/final_checkpoint.pth `
    external/DPDM/runs/dpdm-cifar10-shadow02-eps10-gpu-sameproto3shadow-batch32-r1-20260409/checkpoints/final_checkpoint.pth `
    external/DPDM/runs/dpdm-cifar10-shadow03-eps10-gpu-sameproto3shadow-batch32-r1-20260409/checkpoints/final_checkpoint.pth `
  --target-member-dataset-dir workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/target-member `
  --target-nonmember-dataset-dir workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/target-nonmember `
  --shadow-member-dataset-dirs `
    workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/shadow-01-member `
    workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/shadow-02-member `
    workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/shadow-03-member `
  --shadow-nonmember-dataset-dirs `
    workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/shadow-01-nonmember `
    workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/shadow-02-nonmember `
    workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1/datasets/shadow-03-nonmember `
  --dpdm-root external/DPDM `
  --config-path external/DPDM/configs/cifar10_32/train_eps_10.0.yaml `
  --device cuda `
  --sigma-points 8 `
  --max-samples 128 `
  --provenance-status workspace-verified
```

预期首批产物：

- target stdout/stderr:
  - `Research/external/DPDM/runs/dpdm-cifar10-targetmember-eps10-gpu-sameproto-20260409.stdout.log`
  - `Research/external/DPDM/runs/dpdm-cifar10-targetmember-eps10-gpu-sameproto-20260409.stderr.log`
- shadow workdirs:
  - `Research/external/DPDM/runs/dpdm-cifar10-shadow01-eps10-gpu-sameproto3shadow-r2-20260409`
  - `Research/external/DPDM/runs/dpdm-cifar10-shadow02-eps10-gpu-sameproto3shadow-batch32-r1-20260409`
  - `Research/external/DPDM/runs/dpdm-cifar10-shadow03-eps10-gpu-sameproto3shadow-batch32-r1-20260409`
- batch32 bridge diagnostic workspace:
  - `Research/workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-sameproto3shadow-batch32-diagnostic-20260409`

当前停止条件：

- 成功：target + shadows 训练链完成，并留下后续 bridge 评估所需 checkpoint
- 失败：入口脚本、checkpoint 结构或数据切分暴露稳定 blocker，可复述并写回本文档

## 当前运行态

- target-member 训练已经完成：
  - `Research/external/DPDM/runs/dpdm-cifar10-targetmember-eps10-gpu-sameproto-20260409/checkpoints/final_checkpoint.pth`
- 第一轮影子序列 `sameproto3shadow-20260409` 暴露了一个稳定 blocker：
  - `shadow-01` 在旧端口段 `6032` 上 bind 失败
  - root cause 是前一轮错误影子序列留下的端口占用
- 第二轮影子序列 `sameproto3shadow-r2-20260409` 进一步暴露了自动化 blocker：
  - `shadow-01` 成功训练并写出 final checkpoint
  - 但 `launch_dpdm_shadow_sequence.ps1` 依赖子进程自然退出，当前 DPDM 训练在 final checkpoint 后不会自动退出
  - 结果是 wrapper 不能自动推进到下一条 shadow
- 当前已切回人工串行执行：
  - `shadow-01` 已完成：
    - `Research/external/DPDM/runs/dpdm-cifar10-shadow01-eps10-gpu-sameproto3shadow-r2-20260409/checkpoints/final_checkpoint.pth`
  - `shadow-02` 第一次人工串行尝试：
    - `Research/external/DPDM/runs/dpdm-cifar10-shadow02-eps10-gpu-sameproto3shadow-r2-20260409`
    - 日志推进到 `step 40` 后停止更新
    - `stdout.txt` 最后更新时间停在 `2026-04-09 19:32:34 +08:00`
    - 未写出 `checkpoints/final_checkpoint.pth`
  - `shadow-02` 第二次人工串行重试：
    - `Research/external/DPDM/runs/dpdm-cifar10-shadow02-eps10-gpu-sameproto3shadow-r3-20260409`
    - 仅打印 `Starting training at step 0`
    - `stdout.txt` 最后更新时间停在 `2026-04-09 19:49:03 +08:00`
    - GPU 仍维持高占用，但日志不再推进
    - 同样未写出 `checkpoints/final_checkpoint.pth`
  - 后续 root-cause 线索：
    - 旧失败 run 在父训练进程退出后，残留了多个 `multiprocessing-fork` 子进程持续占用 GPU
    - 清理这些 orphan 子进程后，GPU 占用从高位回落，新的单变量诊断才能在干净状态下重启
  - `shadow-02` 第三次单变量诊断：
    - `Research/external/DPDM/runs/dpdm-cifar10-shadow02-eps10-gpu-sameproto3shadow-batch32-r1-20260409`
    - 唯一改动：`batch_size 64 -> 32`
    - 训练正常推进到 `step 90`
    - 写出 `checkpoints/final_checkpoint.pth`
    - 当前结论：`shadow-02` 的主阻塞更接近训练规模 / GPU 压力，而不是数据集资产损坏
  - `shadow-03` 同配置验证：
    - `Research/external/DPDM/runs/dpdm-cifar10-shadow03-eps10-gpu-sameproto3shadow-batch32-r1-20260409`
    - 使用同样的 `batch_size = 32`
    - 训练正常推进到 `step 90`
    - 写出 `checkpoints/final_checkpoint.pth`
  - 当前 bridge 评估 run：
    - `Research/workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-sameproto3shadow-batch32-diagnostic-20260409`
    - 已写出 `summary.json`
    - 指标为：
      - `auc = 0.541199`
      - `asr = 0.515625`
      - `tpr@1%fpr = 0.0`
      - `tpr@0.1%fpr = 0.0`
    - 当前数据规模为：
      - `shadow_train_size = 768`
      - `target_eval_size = 256`
  - 当前判断：
    - `shadow-02-member` 的失败并非纯资产损坏，至少部分来自较高训练规模和 orphan 进程残留
    - `batch_size = 32` 是当前第一个能同时打通 `shadow-02` 与 `shadow-03` 的有效单变量修正
    - 当前 batch32 bridge 已经产出第一份 decisive artifact，但它仍是 `runtime-smoke` 级诊断结果，不应直接改 admitted 合同

当前因此已满足“单一 white-box GPU 问题已启动”的阶段要求，但还没有 bridge summary，admitted 结果暂不改写。

## 退出后的写回顺序

1. `summary.json`
2. `manifest / intake`
3. `unified-attack-defense-table.json`
4. `Local-API registry seed`
5. 说明文档
