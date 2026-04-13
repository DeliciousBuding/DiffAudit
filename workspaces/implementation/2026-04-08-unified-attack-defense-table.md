# 2026-04-08 Unified Attack-Defense Table

## 主基线

- dataset: `CIFAR-10`
- model family: `DDPM`
- black-box main evidence: `recon DDIM public-100 step30`
- gray-box main evidence: `PIA GPU512 adaptive-reviewed baseline + provisional G-1 (all_steps)`
- white-box main evidence: `GSA 1k-3shadow epoch300 rerun1 + W-1 strong-v3 full-scale`

## 机器读源约束

- 当前唯一权威的 cross-track admitted JSON 是：
  - `workspaces/implementation/artifacts/unified-attack-defense-table.json`
- 根下旧文件 `workspaces/implementation/unified-attack-defense-table.json` 不再作为系统权威读源

## 当前统一总表

| track | attack | defense | comparator type | AUC | ASR | TPR@1%FPR | TPR@0.1%FPR | quality/cost | evidence level | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| black-box | `recon DDIM public-100 step30` | `none` | runtime-mainline public subset | `0.849` | `0.51` | `1.0` | `n/a` | `100 public samples per split; DDIM step30; artifact-mainline ready; cuda runtime` | `runtime-mainline` | current black-box primary evidence; semantics of target/shadow/member/non-member still constrained |
| gray-box | `PIA GPU512 baseline` | `none` | runtime-mainline baseline at `max_samples=512` with adaptive review | `0.841339` | `0.786133` | `0.058594` | `0.011719` | `attack_num=30; interval=10; batch_size=8; 512 samples per split; single GPU serial; adaptive repeats=3; wall-clock=212.993833s` | `runtime-mainline` | current admitted gray-box baseline rung; adaptive repeated-query review leaves baseline unchanged, as expected |
| gray-box | `PIA GPU512 baseline` | `G-1 stochastic-dropout (all_steps)` | runtime-mainline defended pair at `max_samples=512` with adaptive review | `0.828075` | `0.767578` | `0.052734` | `0.009766` | `attack_num=30; interval=10; batch_size=8; 512 samples per split; single GPU serial; adaptive repeats=3; wall-clock=223.128438s; surrogate LPIPS=0.035881` | `runtime-mainline` | current `provisional G-1`; repeated-query review still preserves a meaningful drop from baseline, and `all_steps` remains stronger than `late_steps_only` |
| white-box | `GSA 1k-3shadow` | `none` | paper-aligned runtime mainline | `0.998192` | `0.9895` | `0.987` | `0.432` | `target_eval_size=2000; shadow_train_size=4200; 3 shadows; cuda` | `runtime-mainline` | epoch300 rerun1 promoted as the current strongest white-box attack result |
| white-box | `DPDM W-1` | `strong-v3 full-scale` | defended-target three-shadow full-scale comparator | `0.488783` | `0.4985` | `0.009` | `0.0` | `target_eval_size=2000; shadow_train_size=6000; classifier=logistic-regression-1d` | `runtime-smoke` | current defended white-box main rung |
| white-box | `DPDM W-1` | `strong-v2 full-scale` | defended-target three-shadow full-scale comparator | `0.490813` | `0.496` | `0.006` | `0.0` | `target_eval_size=2000; shadow_train_size=6000; classifier=logistic-regression-1d` | `runtime-smoke` | reference defended rung retained for comparison |

## 当前解释口径

- 黑盒主证据是 `recon`，不是 `variation`
- `variation / Towards` 继续保留为 second black-box track，但因为只有 synthetic smoke，不进入这张主对比表
- 灰盒当前正式主讲线是 `PIA + provisional G-1`
- `PIA` 当前 canonical summary 与 defended summary 已切到 `workspace-verified` 的 adaptive-reviewed GPU512 pair
- `SecMI` 当前状态是 `blocked baseline`，不进入这张主对比表
- 白盒攻击主结果是提升后的 `GSA 1k-3shadow epoch300 rerun1`
- 白盒 defended 主结果固定为 `W-1 strong-v3 full-scale`
- `quality/cost` 当前已开始以结构化字段落到 admitted JSON；灰盒目前记录的是 surrogate `FID / LPIPS`，不是 paper-grade image metrics
- 灰盒 `late_steps_only` 当前保留为质量优先的消融支线，不进入主对比表

## 下一步

1. `PIA GPU512` adaptive ablation 已完成；当前优先是写回状态页、manifest 和 system view，而不是继续扩大样本规模。
2. `SecMI` 只有在真实 `flagfile + checkpoint root` 到位后才重新进入灰盒执行面。
3. 黑盒下一步只收口 `recon / variation` 文档口径，不抢 GPU。
