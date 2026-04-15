# 4C 答辩问答卡

1. **这个项目的核心创新点是什么？**
   - 三条主线（recon/PIA/GSA+W-1）统一为 admitted 的风险证据链，并把所有分支证据拉回 `unified-attack-defense-table.json` 让 Leader 拿得出 4C 材料。

2. **为什么不是单篇论文复现？**
   - 每条主线都有自己独立的资产准入、provenance gate 和 admitted 合同，Research 角色是维持 three-track governance，不是跑完某篇 paper 就交差。

3. **recon 线怎么讲？**
   - 主证据固定为 `DDIM public-100 step30` 的 admitted artifact，主要解释它修正了 shadow_non_member 复用和 artifact-mainline 的对齐，replay/variation 只做证据复核。

4. **PIA 线怎么讲？**
   - 当前主讲口径是 `GPU512 baseline` 加上 `stochastic-dropout(all_steps)` defended pairing，已经在 admissible asset 上复现指标落差，checkpoint/source provenance 仍是 long-term blocker，Research 只补充文档和 replay 而不改 admitted claim。

5. **GSA+W-1 怎么讲？**
   - 口径写成 `GSA 1k-3shadow` 与 `W-1 strong-v3 full-scale` 这两组 admitted asset 的 defended comparator，Research 负责把指标写回 unified table 并保持 bridge 冻结，避免把解释变成新的执行线。

6. **为什么 SMP-LoRA 不进主讲？**
   - 最新 optimizer/lr frontier run stalled、指标波动大，系统只把 SMP-LoRA 当作 comparability/intake packet 继续守 gate，没真正释放 new GPU question。

7. **admitted 与 replay/debug 的边界怎么说？**
   - Admitted 需要 summary.json、artifact-mainline、bridge diagnostic 之类正式证据；replay/debug 只是验证这些 artifact（比如 recon reverify、SMP comparator replay）没有新 run 就不会改 admitted claim。

8. **指标可信性/复现性怎么回应？**
   - 有多次 replay：recon artifact reverify、PIA GPU128/256 adaptive pairing、GSA rerun1、W-1 batch32 bridge，指标来自 repeatable artifacts 与 unified table，口头就能报出 summary.json 路径。
