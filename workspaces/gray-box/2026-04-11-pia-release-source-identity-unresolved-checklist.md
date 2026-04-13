# 2026-04-11 PIA Release/Source Identity Unresolved Checklist

## Status Panel

- `owner`: `research_leader`
- `track`: `gray-box`
- `artifact_type`: `release-review checklist / cpu-only`
- `scope`: `release/source identity unresolved`
- `admission_effect`: `none`
- `paper_alignment_effect`: `none`
- `gpu_release`: `none`
- `queue_effect`: `none`
- `updated_at`: `2026-04-11 00:16 +08:00`

## Scope / Fixed Verdict

这份 checklist 只处理一个问题：

- `release/source identity unresolved`

它不处理：

- `split/protocol mismatch` 的闭环
- 攻防结果本身
- 质量 / 成本结论
- 下一条 GPU 候选审查

当前固定口径：

- `PIA provenance = remain long-term blocker`
- `workspace-verified + adaptive-reviewed + paper-aligned blocked by checkpoint/source provenance`
- `CPU-only`
- `active_gpu_question = none`
- `execution-layer status = no-go / not in current releasable queue`
- `gpu_release = none`

固定边界：

- `split/protocol mismatch = boundary only / review gate only`
- `PIA paper-aligned confirmation = document-level condition only, not a released queue item`
- `This checklist does not open queue, does not release GPU, and does not change no-go / hold verdicts.`

## Current Fixed Evidence

当前已固定、可继续引用的事实：

- `external/PIA` 当前只补到 `origin/main@0d7e08a5a07f44931692d52d54d0ce41aff8f54c` 级 repo identity
- README 当前只补到：
  - `DDPM/CIFAR10_train_ratio0.5.npz`
  - `MIA_efficient` OneDrive checkpoint 入口
- 本地 retained source bundle：
  - `workspaces/gray-box/assets/pia/sources/OneDrive_1_2026-4-7.zip`
  - `sha256 = B69310B92AF98B5AFF2D046747DA6650A915C0FC6A79291C999192D43CEF98E5`
- canonical checkpoint 与 zip 条目 `DDPM/ckpt_cifar10.pt` 字节一致
- `2026-04-09` strict gate 只能写成 historical clean snapshot
- `2026-04-10` strict redo 只能写成当前 repo dirty 的 hygiene signal

当前这些事实最多支撑：

- `repo/public-statement identity documented`
- `retained local source bundle`
- `checkpoint-to-bundle byte identity established`

当前这些事实不能支撑：

- `upstream release identity confirmed`
- `immutable release artifact confirmed`
- `paper-faithful release artifact confirmed`

## Immutable Evidence Still Missing

下列任一项缺失时，结论必须保持 `release/source identity unresolved`：

- [ ] 可审查的 upstream `file id / version / checksum / immutable release page / tag`
- [ ] 本地 zip 与上游发布对象的一一对应关系，而不是只与 README 入口松散对应
- [ ] 可引用的 immutable release record，而不是单纯 `OneDrive folder export`
- [ ] canonical checkpoint 身份从“本地 bundle 条目对应”提升到“上游发布物对应”
- [ ] 明确可审查的 release-level provenance，而不是只停留在 repo identity 或 public statement identity

## Acceptable Substitute Evidence

若上游没有 checksum / tag / file-id，只有满足团队接受的替代证据标准，才允许重启 release review：

- [ ] 替代证据必须可复核、可复述、可落到具体对象
- [ ] 替代证据必须把 bundle 身份锁到单一对象，不能只是口头说明
- [ ] 替代证据必须至少显式满足以下之一：
  - [ ] README / 仓库公开声明 + 当前 public head 身份 + 本地 retained bundle 条目字节一致，且能把 bundle 身份锁到单一发布对象
  - [ ] 维护者公开说明 / 固定页面 / 稳定下载对象，能把 retained bundle 身份锁到单一对象
- [ ] 明确记录：`origin/main` 对齐本身不够
- [ ] 明确记录：`retained local source bundle + byte identity` 本身不够
- [ ] 明确记录：若替代证据未达到预设阈值，结论仍是 `unresolved`，不得软放行
- [ ] 任何 substitute evidence 若要进入可用状态，必须单独记录：
  - [ ] `approval owner`
  - [ ] `approval artifact`
  - [ ] `approval date`
- [ ] 缺少上述任一审批留痕时，substitute evidence 只能写成 `candidate only / not yet approved`

## Strict-Review Hygiene

- [ ] `2026-04-09 strict gate passed` 只能写成 `historical clean snapshot`
- [ ] `2026-04-10 strict redo failed` 只能写成 `present-tense repo not clean`
- [ ] `strict redo failure` 不改变 blocker class，不清除 blocker
- [ ] 若未来要重启 strict release review，必须先清理 `external/PIA` 当前 pycache 漂移并重跑 strict gate
- [ ] 任何 present-tense clean claim 都必须基于新的 clean rerun，不能复用旧结论

## Carry-Forward Verdict / Forbidden Upgrade

继续固定：

- `release/source identity unresolved`
- `paper-aligned blocked by checkpoint/source provenance`
- `CPU-only`
- `active_gpu_question = none`

当前不允许写：

- `upstream release identity confirmed`
- `paper-faithful release artifact confirmed`
- `paper alignment blocker cleared`
- `PIA paper-aligned confirmation complete`
- 任何把 repo identity 偷换成 release identity 的表述
- 任何把本件写成 queue opening / GPU release / admission upgrade 的表述

## Review Gate

过 gate 前必须同时满足：

- [ ] 目标只写 `release/source identity unresolved`
- [ ] `split/protocol mismatch` 只出现在 `boundary / review gate / not in scope`
- [ ] `immutable evidence still missing` 与 `acceptable substitute evidence` 两栏都存在
- [ ] `CPU-only`、`active_gpu_question = none`、`gpu_release = none` 明确可见
- [ ] `PIA paper-aligned confirmation` 明确不是当前 queue item
- [ ] 没有任何一句把 blocker 写成已 cleared

## Exit Decision

本件结论只能二选一：

### A. Re-open paper-aligned review

只有在：

- immutable evidence 或 approved substitute evidence 达到预设阈值
- strict review hygiene 条件全部满足
- 并完成单独 release review

时，才允许从 `remain long-term blocker` 进入重审。

附加边界：

- 这只代表 `release/source identity` 分支可进入单独复审
- 不代表整体 provenance blocker 已清
- 不代表 `split/protocol mismatch` 已解决
- 不代表当前 queue 已打开
- 不代表 `PIA paper-aligned confirmation` 已成为当前 released queue item
- 不代表任何 GPU release / candidate promotion / admission upgrade

### B. Remain long-term blocker

只要上述条件任一未满足，就继续保持：

- `B = remain long-term blocker`
