# 2026-04-08 White-Box Attack-Defense Table

## 主基线

- dataset: `CIFAR-10`
- model family: `DDPM`
- attack mainline: `GSA`
- defense candidate: `W-1 = DPDM`

## 当前对比表

| track | attack | defense | comparator type | AUC | ASR | TPR@1%FPR | TPR@0.1%FPR | evidence level | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| white-box | `GSA 1k-3shadow` | `none` | paper-aligned runtime mainline | `0.97514` | `0.919` | `0.55` | `0.205` | `runtime-mainline` | current strongest local white-box attack result |
| white-box | `DPDM W-1` | `DPDM eps10 smoke checkpoint` | target-only comparator | `0.493652` | `0.585938` | `0.015625` | `0.0` | `runtime-smoke` | weaker than attack baseline; target-only only |
| white-box | `DPDM W-1` | `DPDM eps10 smoke checkpoints x3` | multi-shadow comparator | `0.493652` | `0.5` | `0.015625` | `0.0` | `runtime-smoke` | first defended multi-shadow comparator |
| white-box | `DPDM W-1` | `DPDM defended-target + smoke checkpoints x3` | defended-target multi-shadow comparator | `0.496338` | `0.5` | `0.015625` | `0.0` | `runtime-smoke` | current most aligned local W-1 comparator |
| white-box | `DPDM W-1` | `DPDM defended-target + defended-shadows strong-v2 x3` | defended-target multi-shadow comparator | `0.541199` | `0.550781` | `0.0` | `0.0` | `runtime-smoke` | current strongest local W-1 comparator after stronger target/shadow training |
| white-box | `DPDM W-1` | `DPDM defended-target + defended-shadows strong-v2 x3` | defended-target multi-shadow comparator at `max_samples=512` | `0.537201` | `0.530273` | `0.013672` | `0.0` | `runtime-smoke` | larger defended comparator; result stays far below `GSA` mainline |
| white-box | `DPDM W-1` | `DPDM defended-target + defended-shadows strong-v2 x4` | defended-target three-shadow comparator at `max_samples=512` | `0.462799` | `0.47168` | `0.0` | `0.0` | `runtime-smoke` | current closest local defended comparator to `GSA 1k-3shadow` structure |
| white-box | `DPDM W-1` | `DPDM defended-target + defended-shadows strong-v2 x4` | defended-target three-shadow full-scale comparator | `0.490813` | `0.496` | `0.006` | `0.0` | `runtime-smoke` | full-scale defended comparator; still far below `GSA` attack mainline |
| white-box | `DPDM W-1` | `DPDM defended-target + defended-shadows strong-v3 x4` | defended-target three-shadow comparator at `max_samples=128` | `0.537048` | `0.519531` | `0.0` | `0.0` | `runtime-smoke` | first successful strong-v3 GPU defended comparator |
| white-box | `DPDM W-1` | `DPDM defended-target + defended-shadows strong-v3 x4` | defended-target three-shadow comparator at `max_samples=256` | `0.522339` | `0.527344` | `0.003906` | `0.0` | `runtime-smoke` | first medium-scale stable strong-v3 GPU defended comparator |

## 当前解释口径

- `GSA 1k-3shadow` 是当前白盒攻击主证据线
- `W-1` 当前已经不是“只有 checkpoint”，而是有 target-only 和 multi-shadow 两版 comparator
- `W-1 strong-v2` 比 smoke 版略高，但仍显著弱于 `GSA` 主线，方向上继续支持白盒防御有效
- `W-1 strong-v2 max512` 在更大评估规模下仍维持同一趋势，说明当前防御结果不只是 `128` 级样本偶然现象
- `W-1 strong-v2 3-shadow max512` 进一步把 `AUC` 压到 `0.462799`，这是当前最接近 `GSA 1k-3shadow` 结构的 defended 结果
- `W-1 strong-v2 3-shadow full-scale` 回到 `AUC = 0.490813`，但仍远低于 `GSA` 主线，说明当前 defended 结果在更大评估规模下仍可辩护
- `W-1 strong-v3 3-shadow max128` 已经成功落盘，说明 `strong-v3` checkpoint 集合在 GPU 上可执行，但当前还没超过 `strong-v2` 最佳 defended 结果
- `W-1 strong-v3 3-shadow max256` 也已成功落盘，说明 `strong-v3` 不是偶发成功，而是已进入稳定的中规模 GPU defended 结果阶段
- 但当前 `W-1` 还不是最终 benchmark，因为结果仍是 `runtime-smoke` 级 comparator

## 关联产物

- attack baseline:
  - `workspaces/white-box/runs/gsa-runtime-mainline-20260408-cifar10-1k-3shadow/summary.json`
- target-only defense comparator:
  - `workspaces/white-box/runs/dpdm-w1-target-only-20260408/summary.json`
- multi-shadow defense comparator:
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-20260408/summary.json`
- defended-target multi-shadow comparator:
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-20260408/summary.json`
- defended-target + defended-shadow strong-v2 comparator:
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv2-20260408/summary.json`
- defended-target + defended-shadow strong-v2 comparator at `max_samples=512`:
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv2-max512-20260408/summary.json`
- defended-target + defended-shadow three-shadow strong-v2 comparator at `max_samples=512`:
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv2-3shadow-max512-20260408/summary.json`
- defended-target + defended-shadow three-shadow strong-v2 full-scale comparator:
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv2-3shadow-full-20260408/summary.json`
- defended-target + defended-shadow three-shadow strong-v3 comparator at `max_samples=128`:
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-max128-20260408/summary.json`
- defended-target + defended-shadow three-shadow strong-v3 comparator at `max_samples=256`:
  - `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-3shadow-max256-20260408/summary.json`

## 下一步

1. 保留这张表作为当前白盒可引用基线
2. `strong-v3` 已经有 `max128` 和 `max256` 两条可用 GPU defended comparator，下一步可以进入 `max512`
3. 后续统一总表直接引用这里的九条 defense 记录
