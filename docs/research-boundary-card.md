# 4C Research 口播边界卡片
1. recon：只需说明 `recon DDIM public-100 step30` 是 admitted 级黑箱风险主线，用 fine-tuned/controlled/public-subset/proxy-shadow-member 的 data 作为输入，强调它是正在生效的风险证明，避免说任何“已完成”或“终极”版本。
2. PIA baseline：提 workspace-verified + adaptive-reviewed 的流程，但要明确 baseline 暴露的是 `epsilon-trajectory consistency`，并一起引用 `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR = 0.841339 / 0.786133 / 0.058594 / 0.011719`；同时强调 repeated-query review 只是 bounded adaptive 边界，checkpoint/source provenance 仍阻断任何 paper-aligned release。
3. PIA defended：围绕 `G-1 stochastic-dropout (all_steps)` 的临时 defended 线，说明它是 inference-time randomization 对 `epsilon-trajectory consistency` 的机制性削弱，并一起引用 `0.828075 / 0.767578 / 0.052734 / 0.009766` 这组四指标；不称为已复现、已公开或 validated privacy protection。
4. GSA + W-1：说清它们是 admitted white-box bridge 对 (`GSA 1k-3shadow` 攻击、`W-1 strong-v3 full-scale` 防御)，部署于 same-protocol/closed-frozen comparator 上，强调这不是 final benchmark 而只是 admitted bridge 阶段。
5. 禁区：绝不能讲 `SMP-LoRA` 是 admitted run、release GPU question 或 final attack/defense 报告，只能称其为 comparator-design exploratory packet，从 `2026-04-13` 起已转向 comparator hypothesis 讨论。
6. Replay artifacts 及 SMP-LoRA outputs：统一表述为 Local-API recon replay/demo-trace 材料，存于 `Research/experiments/...`，仅为 demo/debugging/trace context，不等同 admitted 证据。
7. SMP-LoRA 相关 optimizer/lr frontier、batch32 comparator 话题：放在“future gate/comparator hypothesis”里，表明还需单独 review，不与 admitted track 混用。
8. Conditional future-surface：只能说当前有 `SD1.5` local canary contract、first bounded `CFG` packet，以及 first actual runner-level hidden-guidance-jitter rerun = `negative but useful`；还要明确当前 `I-D` 没有 honest bounded successor lane。不能说 conditional diffusion 已被全面覆盖，不能说 hidden-guidance jitter 已是 release-grade defense，也不能说 low-FPR / adaptive robustness 已成立。
总结：核对 recon、PIA 与 GSA-W1 的 admitted 主线；其中 PIA 必须带 mechanistic wording、bounded adaptive 边界和 low-FPR 指标，SMP-LoRA 及 replay artifact 仅作 demo 辅助，未达 final benchmark。
