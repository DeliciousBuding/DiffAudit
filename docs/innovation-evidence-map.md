# 创新点 → Admitted 证据 → 比赛可讲口径对照表

| 创新点 | 关联主线 | Admitted 证据 / 结果名 | 比赛可讲口径 | 不能怎么夸大 |
| --- | --- | --- | --- | --- |
| 以 `recon DDIM public-100 step30` 组成的黑盒风险证明链，替代孤立攻击脚本 | recon | `recon DDIM public-100 step30`（`unified-attack-defense-table.json` black-box row） | “recon 主线已经构建出 admitted 的风险证明链，证据记录在统一的 attack-defense table，可向 leader 报告这是黑盒下 minimal 权限的成员泄露风险” | “不要说达到了最终 exploit or SOTA，只能说黑盒行为空间下的 admitted risk upper bound” |
| 用 workspace-verified + adaptive-reviewed 流程验证 PIA baseline/defended，替代 paper-only 复现 | PIA | `PIA GPU512 baseline` / `PIA GPU512 defended`（gray-box rows） | “PIA baseline 与 defended 都在 admitted 的 workspace adaptive 复现框架下，表明我们有 repeated-query review 与 provenance controls 支撑 attack-defense 闭环” | “不能宣传是打开新 checkpoint 或改写论文结论；描述时只说 admitted baseline/defended 仍受 workspace review 约束” |
| 构建 GSA 1k-3shadow + W-1 strong-v3 的白盒 bridge，对比 admitted 攻防而不仅仅是展示 | GSA + W-1 | `GSA 1k-3shadow` attack + `W-1 strong-v3 full-scale` defense（white-box rows） | “GSA/W-1 形成 admitted white-box bridge，Leader 可说这是同协议下的 closed-frozen comparator，防御结果记录在 admitted table，便于与 defense-side 讨论白盒差异” | “不要夸它是最终 benchmark，强调这是 admitted comparator 的 risk proof 语义” |
| 在 admitted gate 内准备 cross-layer comparator，持续守住 SMP-LoRA + W-1 的 admitted 规划 | cross-layer system | `2026-04-14-baseline-smp-lora-w1-comparator-admission-packet`（future comparator notes） | “虽然没有新 GPU run，Research 端已经把 SMP-LoRA vs W-1 的 admitted comparator 当作 cross-layer gate，在材料包里补齐对比输出，说明我们保持 admitted boundary 内的 steady comparator work” | “不能讲成 SMP-LoRA 已经进入 main track 或拿到新训练，必须说明它仍属于 comparator/prep 状态，只能在 admitted gate 内推进” |

