# 4C 竞赛可讲的创新差异化亮点

1. **相比单点攻击工具，织起连贯证据链。** Recon track 不只是单条攻击脚本，而是将 `recon DDIM public-100 step30` 作为 admitted 主证据、`public-50 step10` 作为 best single metric reference，并在 `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json` 中保留整个 track 的 metadata，向 Leader 呈现风险证明流程而非孤立 payload。

2. **相比只做论文复现，走的是 workspace adaptive 走查而不是 blind copy。** PIA baseline 与 defended 均在 `workspace-verified + adaptive-reviewed` 框架下实施，已在同一个 `unified-attack-defense-table.json` 里记录 `stochastic-dropout` 与 `epsilon-trajectory consistency` 的验证细节，并注明 `paper-aligned blocked by checkpoint/source provenance`，说明我们保留了 adaptive reviewer 的判断和 provenance を卡住的边界。

3. **相比只做页面展示，交付的是 admitted 的 white-box bridge 对比。** GSA 1k-3shadow 和 W-1 strong-v3 full-scale 已形成 admitted pair，在 same-protocol bridge 上跑出了 `closed-frozen` comparator，并把 defended 结果写入 `unified-attack-defense-table.json`，不是平台上纯展示数据，而是可落到 Leader 口径的白盒防御对比。

4. **研究边界下仍能成立的亮点：SMP-LoRA 退而求比较器。** 虽然 `T06 optimizer/lr frontier` 被标为 `closed-mixed-no-go`、当前没有新的 GPU run，Research 端仍把 `baseline vs SMP-LoRA vs W-1 comparator` 当作 future gate，在 `2026-04-14-baseline-smp-lora-w1-comparator-admission-packet` 中提前准备比较器输出，表明我们能在 admitted 边界内稳步推进可复现对比而不是被动退回。

## Leader 短版摘要

我们把 recon/PIA/GSA-W1 证据链做成风险证明、workspace adaptive 复现和 bridge，SMP-LoRA comparator 守住 admitted gate，防止单点攻击工具孤立且不是页面展示，Leader 可填 4C 材料并说明 GPU-Run 仍未开。

高层口径速记：Recon/PIA/GSA 线索已经串成风险证明链，配套 workspace adaptive 复现与白盒 bridge，SMP-LoRA comparator 继续守住 admitted gate；当前仍没新 GPU-run，但材料可直接填在 4C 材料页里。
