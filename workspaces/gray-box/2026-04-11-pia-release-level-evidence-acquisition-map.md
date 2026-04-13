# 2026-04-11 PIA Release-Level Evidence Acquisition Map

## Status Panel

- `owner`: `总管理 Agent`
- `track`: `gray-box`
- `artifact_type`: `acquisition map / cpu-only`
- `scope`: `future release-level evidence targets for separate review`
- `admission_effect`: `none`
- `paper_alignment_effect`: `none`
- `gpu_release`: `none`
- `queue_effect`: `none`
- `updated_at`: `2026-04-11 01:19 +08:00`

## Scope / Fixed Verdict

本文档是 `release-level evidence acquisition map`，不是 `evidence packet`。

它只记录：

- 未来若要形成 `approval artifact`，需要补哪些 release-level evidence
- 哪些 source types 可以作为有效 acquisition target
- 哪些现有材料仍然不够

它不表示：

- 当前证据已到位
- 当前 review 已重开
- 当前 queue 已打开
- 当前 GPU 已释放
- 当前 admission 已升级

当前固定结论仍为：

- `release/source identity unresolved`
- `none approval-ready yet`
- `B = remain long-term blocker`

当前不变边界：

- `split/protocol mismatch = boundary only / review gate only`
- `CPU-only`
- `active_gpu_question = none`
- `execution-layer status = no-go / not in current releasable queue`
- `PIA paper-aligned confirmation = document-level condition only, not a released queue item`

## Current Confirmed Evidence

当前已确认、可继续复用的只包括：

- `repo/public-statement identity documented`
  - `external/PIA` 当前只补到 `origin/main@0d7e08a...` 级 repo identity
  - README 当前只补到 split 文件名与 `MIA_efficient` 入口
- `retained local source bundle`
  - `OneDrive_1_2026-4-7.zip`
  - bundle `sha256` 已固定
- `checkpoint-to-bundle byte identity established`
  - canonical checkpoint 与 zip 条目 `DDPM/ckpt_cifar10.pt` 字节一致
- `strict-review hygiene boundary documented`
  - `2026-04-09 strict gate passed = historical clean snapshot`
  - `2026-04-10 strict redo failed = present-tense repo not clean`

当前这些材料不支撑：

- `upstream release identity confirmed`
- `immutable release artifact confirmed`
- `paper-faithful release artifact confirmed`

## Missing Release-Level Evidence

未来若要进入真正的 approval artifact 组装，仍缺以下 release-level evidence：

1. 可审查的上游发布对象身份
   - `file id / object id / version / tag / immutable page / stable URL`
2. retained bundle 与单一上游发布对象的一一对应关系
3. 可引用的 immutable release record
   - 不能只是 `OneDrive folder export`
4. canonical checkpoint 到上游发布物的 identity lift
5. 可单独复核的 release-level provenance
6. 独立 approval 留痕
   - `approval owner`
   - `approval artifact`
   - `approval date`

## Future Acquisition Targets

未来只允许把下列对象视为有效 acquisition target：

### 1. Upstream Published Object

有效 source types：

- immutable release page
- stable download object
- 带 permalink 的 maintainer public statement

必须锁定的 object identifiers：

- `stable URL / page`
- `file_id / object_id / version / tag`
- 期望文件名
- `checksum`（若上游提供）

最低留痕字段：

- `collected_at`
- `collector`
- 采集方式
- 原始快照位置或快照 hash
- 对象标识
- 声明锚点

### 2. Retained Local Source Bundle

有效 source types：

- 本地 zip
- zip 条目清单
- bundle hash 记录

必须锁定的 object identifiers：

- `OneDrive_1_2026-4-7.zip` 的 `sha256`
- 成员路径 `DDPM/ckpt_cifar10.pt`
- 成员长度与时间戳

最低留痕字段：

- `local_path`
- `sha256`
- `member_path`
- `member_length`
- 提取/校验命令
- `verified_at`
- `verifier`

### 3. Canonical Runtime Object

有效 source types：

- canonical checkpoint
- manifest
- provenance record

必须锁定的 object identifiers：

- `checkpoint.pt` 的 `sha256`
- 对应 manifest / provenance 标识

最低留痕字段：

- `local_path`
- `sha256`
- 验证命令
- `verified_at`
- `verifier`
- 与 bundle member 的对应关系

### 4. Cross-Object Binding Record

有效 source types：

- object-to-bundle comparison memo
- review packet
- 独立 evidence sheet

必须锁定的 object identifiers：

- `upstream object id`
- `bundle sha256`
- `checkpoint sha256`

最低留痕字段：

- 绑定关系说明
- 比较方法
- 结论状态
- 剩余 caveat
- `reviewer`
- `reviewed_at`

### 5. Approval Record

有效 source types：

- 单独 approval memo
- decision record

必须锁定的 object identifiers：

- `approval artifact id / path`
- 被批准的 candidate id

最低留痕字段：

- `approval owner`
- `approval artifact`
- `approval date`
- 适用边界
- 明确写出：
  - 不改变 queue
  - 不释放 GPU
  - 不触发 admission upgrade

## Binding Chain

未来如果真要形成可审批结构，最小绑定链必须完整写成：

`upstream published object -> retained local bundle -> canonical checkpoint -> approved review record`

缺少其中任一层时，都不得写成：

- `approval-ready`
- `approved substitute evidence`
- `review reopened`

## Non-Evidence / Insufficient Materials

下列材料当前仍明确不够：

- `origin/main@0d7e08a...`
  - 只能证明 `repo identity`
  - 不能证明 `release identity`
- README 对 split 文件名与 `MIA_efficient` 入口的声明
  - 只能算 public-statement anchor
  - 不能单独锁定单一发布对象
- `retained local source bundle + checkpoint-to-bundle byte identity`
  - 只能证明本地链条
  - 不能证明该 bundle 就是上游 release object
- `2026-04-09 strict gate passed`
  - 只能算 `historical clean snapshot`
  - 不能复用成当前 clean
- `2026-04-10 strict redo failed`
  - 只能算 hygiene signal
  - 不能升级成 release/source evidence
- `ratio0.5` split 的本地形状、自洽性和代码消费路径
  - 只支撑 local split shape
  - 不直接放行 `release/source identity`
- `maintainer statement / fixed page / stable object`
  - 当前仍是规则占位
  - 不是实体 evidence packet

## Approval Artifact Assembly Rule

只有当未来出现新的 release-level evidence，并且同时满足以下条件时，才允许重新检查 approval readiness：

1. retained bundle 被锁到单一上游发布对象
2. 绑定链完整写到 canonical checkpoint
3. 独立 approval 留痕齐全
   - `approval owner`
   - `approval artifact`
   - `approval date`
4. boundary block 明确写出：
   - 不代表 provenance blocker 整体已清
   - 不代表 `split/protocol mismatch` 已解
   - 不打开 queue
   - 不释放 GPU
   - 不触发 admission upgrade

缺少任一项时，仍只能写成：

- `candidate only / not yet approved`

## No Review / Queue / GPU / Admission Effect

本文档不重开 review，不打开 queue，不释放 GPU，不触发 candidate promotion，不触发 admission upgrade。

当前固定：

- `active_gpu_question = none`
- `gpu_release = none`
- `execution-layer status = no-go / not in current releasable queue`
- `PIA paper-aligned confirmation` 继续不是当前 queue item

## Exit Decision

本轮唯一安全结论仍是：

- `none approval-ready yet`
- `B = remain long-term blocker`

本文档当前不允许被改写成：

- `upstream release identity confirmed`
- `immutable release artifact confirmed`
- `candidate readiness satisfied`
- `review reopened`
- `queue opened`
- `GPU released`
- `admission upgraded`

