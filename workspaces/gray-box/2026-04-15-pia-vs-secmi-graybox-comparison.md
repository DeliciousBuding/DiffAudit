# 2026-04-15 Gray-Box Comparison: PIA vs SecMI

## Current Inputs

- `PIA` baseline:
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-20260409-gpu-512-adaptive/summary.json`
- `PIA` defended:
  - `workspaces/gray-box/runs/pia-cifar10-runtime-mainline-dropout-defense-20260409-gpu-512-allsteps-adaptive/summary.json`
- `SecMI` subset scaling rung:
  - `workspaces/gray-box/runs/secmi-cifar10-gpu-4096-20260415-r1/summary.json`
- `SecMI` full split rung:
  - `workspaces/gray-box/runs/secmi-cifar10-gpu-full-stat-20260415-r2/summary.json`

## Metric Snapshot

| Line | Samples per split | AUC | ASR | TPR@1%FPR | Notes |
|------|-------------------|-----|-----|-----------|-------|
| `PIA` baseline | `512` | `0.841339` | `0.786133` | `0.058594` | current admitted gray-box baseline |
| `PIA` defended | `512` | `0.828075` | `0.767578` | `0.052734` | current admitted defended gray-box rung |
| `SecMI` stat | `4096` | `0.888575` | `0.829224` | `0.073486` | first larger-scale corroboration rung |
| `SecMI` NNS | `4096` | `0.932602` | `0.869240` | `0.157461` | sample-efficient classifier head |
| `SecMI` stat | `25000` | `0.885833` | `0.815400` | `0.093960` | full split local execution |
| `SecMI` NNS | `25000` | `0.946286` | `0.879275` | `0.400450` | strongest current gray-box ranking signal |

## Interpretation

- `PIA` remains the cleaner admitted mainline because its defense pairing and repeated-query wording are already integrated into the current research narrative.
- `SecMI` is no longer just a literature placeholder. The imported CIFAR-10 bundle now gives a full-split GPU result and independently confirms that the current DDPM checkpoint leaks membership signal.
- On the current asset line, `SecMI`'s `stat` head is only moderately above `PIA`, but the `NNS` head is materially stronger once sample size is large enough.
- The practical gray-box story is no longer "`PIA` alone". It is "`PIA` as the defended mainline + `SecMI` as an independent corroboration line with a stronger classifier head".

## Recommended Competition Wording

- 主讲灰盒攻击仍以 `PIA` 为主，因为它已经接入当前 defense narrative。
- 同时补一句：`SecMI` 在同一 CIFAR-10 资产线上完成了 full-split GPU 复核，`stat` 与 `NNS` 两个头都显示稳定成员泄露，其中 `NNS` 头达到更高 AUC。
- 不要把 `SecMI` 直接写成替代 `PIA` 的唯一主讲，而是写成“第二条独立确认线”。

## Next Step

1. 把这份比较压回 `mainline-audit` / 比赛材料摘要。
2. 若还继续烧 GPU，优先决定是补 `PIA` larger-scale comparability，还是把 `CLiD` 从 artifact summary 推到 direct asset run。
