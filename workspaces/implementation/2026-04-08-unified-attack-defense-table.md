# 2026-04-08 Unified Attack-Defense Table

## 主基线

- dataset: `CIFAR-10`
- model family: `DDPM`
- black-box main evidence: `recon DDIM public-100 step30`
- gray-box main evidence: `PIA GPU512 adaptive-reviewed baseline + provisional G-1 (all_steps)`；当前最强 challenger: `TMIA-DM late-window long_window`；当前最强 defended challenger: `TMIA-DM + temporal-striding(stride=2)`
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
| gray-box | `TMIA-DM late-window GPU256 (repeat-confirmed)` | `none` | challenger line on late timesteps `[80,100,120]` with two GPU ladders completed | `0.837814` | `0.787109` | `0.050781` | `0.015625` | `batch_size=8; 256 samples per split; single GPU serial; late-window [80,100,120]; repeat-confirmed at GPU128 and GPU256` | `runtime-challenger` | strongest active gray-box challenger; `PIA` still holds headline continuity, but `TMIA-DM` is now a real competitor rather than a literature-side branch |
| gray-box | `TMIA-DM late-window GPU256 (repeat-confirmed)` | `stochastic-dropout (all_steps)` | defended challenger line on the same late-timestep contract | `0.82164` | `0.765625` | `0.042969` | `0.019531` | `batch_size=8; 256 samples per split; single GPU serial; late-window [80,100,120]; defended repeat-confirmed at GPU128 and GPU256` | `runtime-challenger` | defended gray-box is no longer `PIA`-only; dropout weakens `TMIA-DM` but does not eliminate it |
| gray-box | `TMIA-DM late-window GPU256 (repeat-confirmed)` | `temporal-striding (stride=2)` | defended challenger line on the strided late-timestep contract `[80,120]` | `0.7173` | `0.662109` | `0.019531` | `0.011719` | `batch_size=8; 256 samples per split; single GPU serial; requested window [80,100,120]; effective window [80,120]; repeat-confirmed at cpu32, GPU128, and GPU256` | `runtime-challenger` | current strongest defended `TMIA-DM` branch; materially stronger than `TMIA + dropout`, but still challenger-specific rather than the admitted defended project headline |
| white-box | `GSA 1k-3shadow` | `none` | paper-aligned runtime mainline | `0.998192` | `0.9895` | `0.987` | `0.432` | `target_eval_size=2000; shadow_train_size=4200; 3 shadows; cuda` | `runtime-mainline` | epoch300 rerun1 promoted as the current strongest white-box attack result |
| white-box | `DPDM W-1` | `strong-v3 full-scale` | defended-target three-shadow full-scale comparator | `0.488783` | `0.4985` | `0.009` | `0.0` | `target_eval_size=2000; shadow_train_size=6000; classifier=logistic-regression-1d` | `runtime-smoke` | current defended white-box main rung |
| white-box | `DPDM W-1` | `strong-v2 full-scale` | defended-target three-shadow full-scale comparator | `0.490813` | `0.496` | `0.006` | `0.0` | `target_eval_size=2000; shadow_train_size=6000; classifier=logistic-regression-1d` | `runtime-smoke` | reference defended rung retained for comparison |

## 当前解释口径

- 黑盒主证据是 `recon`，不是 `variation`
- `variation / Towards` 继续保留为 second black-box track，但因为只有 synthetic smoke，不进入这张主对比表
- 灰盒当前正式主讲线是 `PIA + provisional G-1`，但已不再是 `PIA` 单线叙事
- `TMIA-DM late-window` 已升级为当前最强 gray-box challenger，并已进入 defended 口径
- `TMIA-DM + temporal-striding(stride=2)` 已取代 `TMIA-DM + dropout`，成为当前最强 defended gray-box challenger
- `PIA` 当前 canonical summary 与 defended summary 已切到 `workspace-verified` 的 adaptive-reviewed GPU512 pair
- `SecMI` 当前状态是独立 corroboration line，不再写成 blocked baseline；但它不是这张主对比表的当前 gray-box 重点
- 白盒攻击主结果是提升后的 `GSA 1k-3shadow epoch300 rerun1`
- 白盒 defended 主结果固定为 `W-1 strong-v3 full-scale`
- `quality/cost` 当前已开始以结构化字段落到 admitted JSON；灰盒目前记录的是 surrogate `FID / LPIPS`，不是 paper-grade image metrics
- 灰盒 `TMIA-DM late-window` 目前以 operating-point challenger 身份进入主对比表，而不是替换 `PIA` headline

## 下一步

1. 灰盒当前优先不是再证明 `TMIA-DM` 存在，而是把 `PIA headline + TMIA challenger + TMIA-specific defended branch` 这一结构稳定传递给更高层。
2. 下一条灰盒研究任务应优先做 defended operating-point/summary sync，而不是回退到 naive fusion 或再开一轮无假设 defense shortlist。
3. 黑盒与白盒仍按各自主线推进；这张表只负责把跨线主读法固定住。
