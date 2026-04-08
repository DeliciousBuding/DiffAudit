# 2026-04-08 Unified Attack-Defense Table

## 主基线

- dataset: `CIFAR-10`
- model family: `DDPM`
- black-box main evidence: `recon DDIM public-100 step30`
- gray-box main evidence: `PIA GPU512 baseline + provisional G-1`
- white-box main evidence: `GSA 1k-3shadow + W-1 strong-v3 full-scale`

## 当前统一总表

| track | attack | defense | comparator type | AUC | ASR | TPR@1%FPR | TPR@0.1%FPR | quality/cost | evidence level | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| black-box | `recon DDIM public-100 step30` | `none` | runtime-mainline public subset | `0.849` | `0.51` | `1.0` | `n/a` | `100 public samples; DDIM step30; cuda runtime` | `runtime-mainline` | current black-box primary evidence; semantics of target/shadow/member/non-member still constrained |
| gray-box | `PIA GPU512 baseline` | `none` | runtime-mainline baseline at `max_samples=512` | `0.841339` | `0.786133` | `0.058594` | `0.011719` | `single GPU; elapsed_seconds=171.214752` | `runtime-mainline` | current strongest admitted gray-box baseline rung |
| gray-box | `PIA GPU512 baseline` | `G-1 stochastic-dropout` | runtime-mainline defended pair at `max_samples=512` | `0.82938` | `0.769531` | `0.023438` | `0.009766` | `single GPU; elapsed_seconds=131.89636` | `runtime-mainline` | current `provisional G-1`; defense signal repeated at `128/256/512` |
| white-box | `GSA 1k-3shadow` | `none` | paper-aligned runtime mainline | `0.97514` | `0.919` | `0.55` | `0.205` | `1k target eval; 3 shadows; cuda` | `runtime-mainline` | current strongest white-box attack result |
| white-box | `DPDM W-1` | `strong-v3 full-scale` | defended-target three-shadow full-scale comparator | `0.488783` | `0.4985` | `0.009` | `0.0` | `target_eval_size=2000; shadow_train_size=6000` | `runtime-smoke` | current defended white-box main rung |
| white-box | `DPDM W-1` | `strong-v2 full-scale` | defended-target three-shadow full-scale comparator | `0.490813` | `0.496` | `0.006` | `0.0` | `target_eval_size=2000; shadow_train_size=6000` | `runtime-smoke` | reference defended rung retained for comparison |

## 当前解释口径

- 黑盒主证据是 `recon`，不是 `variation`
- `variation / Towards` 继续保留为 second black-box track，但因为只有 synthetic smoke，不进入这张主对比表
- 灰盒当前正式主讲线是 `PIA + provisional G-1`
- `SecMI` 当前状态是 `blocked baseline`，不进入这张主对比表
- 白盒攻击主结果是 `GSA 1k-3shadow`
- 白盒 defended 主结果固定为 `W-1 strong-v3 full-scale`

## 下一步

1. 如果 `PIA GPU512` repeat 再次维持同方向下降，可把灰盒写法从 `provisional G-1` 提升为更稳定的 defended mainline。
2. `SecMI` 只有在真实 `flagfile + checkpoint root` 到位后才重新进入灰盒执行面。
3. 黑盒下一步只收口 `recon / variation` 文档口径，不抢 GPU。
