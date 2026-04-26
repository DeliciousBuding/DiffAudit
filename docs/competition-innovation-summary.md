# 4C 竞赛可讲的创新差异化亮点

1. **相比单点攻击工具，织起连贯证据链。** Recon track 不只是单条攻击脚本，而是将 `recon DDIM public-100 step30` 作为 admitted 主证据、`public-50 step10` 作为 best single metric reference，并在 `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json` 中保留整个 track 的 metadata，向 Leader 呈现风险证明流程而非孤立 payload。

2. **相比只做论文复现，走的是 workspace adaptive 走查而不是 blind copy。** PIA baseline 与 defended 均在 `workspace-verified + bounded repeated-query adaptive-reviewed` 框架下实施，当前主讲点不是“加了 dropout”，而是把 `PIA` 明确写成 `epsilon-trajectory consistency`，把 `stochastic-dropout(all_steps)` 写成 inference-time randomization 对该信号的机制性削弱；对外同时携带 `AUC / ASR / TPR@1%FPR / TPR@0.1%FPR`，并注明 adaptive 只到 bounded repeated-query review、`paper-aligned blocked by checkpoint/source provenance` 仍未解除。

3. **相比只做页面展示，交付的是 admitted 的 white-box bridge 对比。** GSA 1k-3shadow 和 W-1 strong-v3 full-scale 已形成 admitted pair，在 same-protocol bridge 上跑出了 `closed-frozen` comparator，并把 defended 结果写入 `unified-attack-defense-table.json`，不是平台上纯展示数据，而是可落到 Leader 口径的白盒防御对比。

4. **研究边界下仍能成立的亮点：SMP-LoRA 退而求比较器。** 虽然 `T06 optimizer/lr frontier` 被标为 `closed-mixed-no-go`、当前没有新的 GPU run，Research 端仍把 `baseline vs SMP-LoRA vs W-1 comparator` 当作 future gate，在 `2026-04-14-baseline-smp-lora-w1-comparator-admission-packet` 中提前准备比较器输出，表明我们能在 admitted 边界内稳步推进可复现对比而不是被动退回。

## Leader 短版摘要

我们把 recon/PIA/GSA-W1 证据链做成风险证明、mechanistic gray-box defense reading 和 white-box bridge，SMP-LoRA comparator 守住 admitted gate，防止单点攻击工具孤立且不是页面展示；其中 PIA 线现在要明确成“trajectory-consistency -> inference-time randomization”，并保留 low-FPR 与 provenance 边界。

高层口径速记：Recon/PIA/GSA 线索已经串成风险证明链，PIA 的 admitted 读法是 mechanistic + bounded adaptive + low-FPR contract，白盒则保留 bridge；当前仍没新 GPU-run，但材料可直接填在 4C 材料页里。
