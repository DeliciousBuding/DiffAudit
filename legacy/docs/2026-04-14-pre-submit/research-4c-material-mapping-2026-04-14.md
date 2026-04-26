# 研究口径对 4C 提交材料映射 — 2026-04-14

1. **Recon 主讲线**：根据 `leader-research-ready-summary.md` 和 `innovation-evidence-map.md`，`recon DDIM public-100 step30`（及其在 `Research/workspaces/implementation/artifacts/unified-attack-defense-table.json` 的 black-box row）是 AUC/ASR/TPR@1%FPR 等指标都经 Dual-Assessment 确认的 admitted risk 证据；4C 报告可直接引用该 artifact 和表格串联的 “recon risk exists” 结论。表述边界：必须注明这是 controlled/public subset 下的 admitted risk upper-bound 证据，不能扩展到任何 new GPU 质疑或宣称整个 recon 系统已完全防御。

2. **PIA 主讲线**：`competition-defense-qa.md`（涉及 GPU512 baseline + G-1 stochastic-dropout defended）与 `leader-research-ready-summary.md` 共同支撑 PIA 可 admissible 证据组合（baseline run、defended run、adaptive reviewer & provenance meta），4C 材料可按“workspace-verified/adaptive-reviewed defended pairing”介绍。边界：只谈 admitted baseline/defended evidence 和 provenance controls，避免将其叙述为对 checkpoint/source 之外的全局安全保障或“new GPU” 质疑的最终回答。

3. **GSA + W-1 主讲线**：同时引用 `competition-defense-qa.md` 和 `innovation-evidence-map.md` 中的 GSA 1k-3shadow + W-1 strong-v3 runs，这对 admitted bridge 提供白盒 comparator 和 defended contrast 引证（也记录在 unified attack-defense table）。4C 材料定位为 “GSA/W-1 admitted bridge + defended comparator” 逻辑，边界是在 closed-frozen comparator 上说明它提供跨线索对比，不能推广为整套 GSA/W-1 代表未来 SOTA。

4. **SMP-LoRA 仅 comparator**：虽然 SMP-LoRA 比较值被记录（`competition-defense-qa.md` 对 SMP-LoRA-GPU question gating 的讨论，`leader-research-ready-summary.md` 列出 comparator replay/editorial），但其在 4C 材料中只出现于 “comparator / future gate / cross-layer system” 的辅助段落，强调“no new GPU question” 和 “requires comparator/prep review”，决不能在主讲线里把 SMP-LoRA 作为 admitted track。
