# 2026-04-09 PIA Provenance Dossier（Checkpoint/Source 收口）

> 目标：把当前 `checkpoint/source provenance` blocker 从“口头口径”升级成**可审查 dossier**。  
> 边界：本 dossier 只收口 provenance（checkpoint/source/split/paper mapping）。它不替代攻防结果、信号解释或成本结论。

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 +08:00`
- `track`: `gray-box`
- `method`: `pia`
- `contract_key`: `gray-box/pia/cifar10-ddpm`
- `current_provenance_status`: `workspace-verified`
- `paper_alignment_status`: `blocked`
- `blocker_key`: `checkpoint/source provenance`

## 当前结论（必须对外一致）

- 当前已允许写：`workspace-verified`（并可携带 `provisional G-1` 叙事）。
- 当前仍禁止写：`paper-aligned`。
- 唯一 blocker：`checkpoint/source provenance` 未与论文/上游 release 声明形成可审查的一一对齐闭环。

当前主讲口径（仅作锚点，不在 dossier 内复述实验细节）：

- attack baseline: `pia-cifar10-runtime-mainline-20260409-gpu-512-adaptive`
- defended mainline: `pia-cifar10-runtime-mainline-dropout-defense-20260409-gpu-512-allsteps-adaptive`
- execution-layer status: `no-go / not in current releasable queue`

## 1) 当前已知证据（Evidence Pack）

以下证据均为工作区内可复核文件：

- canonical roots 与 summary 锚点（单点真相入口）：
  - `workspaces/gray-box/assets/pia/manifest.json`
- canonical provenance record（checkpoint/source/split 哈希固化）：
  - `workspaces/gray-box/assets/pia/PROVENANCE.md`
- 资产探针与 runtime preview（证明能加载）：
  - `workspaces/gray-box/2026-04-07-pia-real-asset-probe.md`
- runtime mainline（证明能跑）：
  - `workspaces/gray-box/2026-04-07-pia-runtime-mainline.md`
- next-run provenance gate（证明“输入可被机器可读固定 + `2026-04-09` historical clean snapshot 已留痕”）：
  - `workspaces/gray-box/2026-04-09-pia-provenance-gate.md`
  - `workspaces/gray-box/assets/pia/next-run-20260409-strict/manifest.json`
  - `workspaces/gray-box/assets/pia/next-run-20260409-strict/provenance.json`
- upstream identity supplement（补 `origin/main / README line anchors / split shape`）：
  - `workspaces/gray-box/2026-04-10-pia-provenance-upstream-identity-note.md`
- split / protocol delta supplement（补 `strict redo / code anchors / paper protocol delta`）：
  - `workspaces/gray-box/2026-04-10-pia-provenance-split-protocol-delta.md`
- 上游声明锚点（用于 paper protocol mapping）：
  - `external/PIA/README.md`

strict gate 的关键可复核字段（便于审查者快速定位）：

- `manifest_sha256 = 6ca897687d0e84a0d13b27c6ed4767a25a8eb24c3e676b115ac5658cdd7fff37`
- `external/PIA git.commit = 0d7e08a5a07f44931692d52d54d0ce41aff8f54c`
- `external/PIA git.dirty = false`

## 2) Checkpoint Lineage（谱系链）

### 2.1 谱系链（从上游声明到 canonical checkpoint）

1. 上游声明（repo 文案级）：`external/PIA/README.md` 提供 “Pretrained Checkpoint” 的 OneDrive 文件夹入口（`MIA_efficient`）。
2. 本地留存的来源包（source bundle）：
   - `workspaces/gray-box/assets/pia/sources/OneDrive_1_2026-4-7.zip`
3. 来源包内部 checkpoint 条目：
   - `OneDrive_1_2026-4-7.zip:DDPM/ckpt_cifar10.pt`
4. canonical checkpoint（当前运行使用）：
   - `workspaces/gray-box/assets/pia/checkpoints/cifar10_ddpm/checkpoint.pt`
   - 当前“转换”仅为路径重组/重命名；字节一致性已可审查（见 2.2）。

### 2.2 字节一致性与哈希（可审查事实）

已知且可复核：

- zip 内存在 `DDPM/ckpt_cifar10.pt`，其 `Length = 574274030`
- 对 zip 条目 `DDPM/ckpt_cifar10.pt` 计算 `sha256`：
  - `55C3F4545BC345522C402EF9C84627BB87BA53049C9541DDF46A6FEB5B0B283E`
- canonical checkpoint `checkpoint.pt` 的 `sha256`：
  - `55C3F4545BC345522C402EF9C84627BB87BA53049C9541DDF46A6FEB5B0B283E`

因此允许写成：

- “当前 canonical checkpoint 是从来源包条目 `DDPM/ckpt_cifar10.pt` 字节级复制得到（仅路径重组）。"

### 2.3 结论边界

上述谱系链解决的是：

- “本地 checkpoint 与本地来源包之间的对应关系”。

它尚未解决的是（仍导致 paper-aligned blocker）：

- “本地来源包是否可审查地证明来自论文作者的 paper-faithful release bundle（上游 release 版本/校验/ID 对齐）。”

## 3) Source/Hash（来源与哈希清单）

> 原则：只填“已知证据”；未知项必须 `待核`，但表格结构必须决策完整。

| 类别 | 路径 | sha256 | 角色/备注 |
| --- | --- | --- | --- |
| checkpoint（canonical） | `workspaces/gray-box/assets/pia/checkpoints/cifar10_ddpm/checkpoint.pt` | `55C3F4545BC345522C402EF9C84627BB87BA53049C9541DDF46A6FEB5B0B283E` | 当前运行使用 |
| source bundle（留存） | `workspaces/gray-box/assets/pia/sources/OneDrive_1_2026-4-7.zip` | `B69310B92AF98B5AFF2D046747DA6650A915C0FC6A79291C999192D43CEF98E5` | checkpoint 来源包留存 |
| dataset archive（留存） | `workspaces/gray-box/assets/pia/sources/cifar-10-python.tar.gz` | `6D958BE074577803D12ECDEFD02955F39262C83C16FE9348329D7FE0B5C001CE` | dataset 提取来源包 |
| member split（外部依赖） | `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz` | `ACA922ECEE25EF00DC6B6377EBAF7875DFCC77C2CDFE27C873B26A65134AA0C0` | 上游 external 依赖 |
| config（本地） | `tmp/configs/pia-cifar10-graybox-assets.local.yaml` | `57E99D0FF1834BD5F7FA64B6DE8F06EB29E13E4DC23EF3DDB905D41B6007878A` | strict gate 固定 |

## 4) Split Provenance（成员划分来源）

### 4.1 当前 split 与其固定方式

- 当前 split 文件固定为：
  - `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz`
  - `sha256 = ACA922ECEE25EF00DC6B6377EBAF7875DFCC77C2CDFE27C873B26A65134AA0C0`

strict gate 已将该文件与 split root tree hash 机器可读固定：

- `external/PIA/DDPM` `tree_sha256 = 31ef17bf5f399205eee80926bad4a382b2a9fe1a300e5fa85443a2b0559b3c71`

### 4.2 仍待闭环点

- `已补`：split shape 与代码消费路径
  - 当前 `ratio0.5` split 的 `50/50` 互斥结构，以及 `main.py / dataset_utils.py` 的 `mia_train_idxs / mia_eval_idxs` 消费路径已可审查
- `待核`：paper-faithful split / protocol 对齐
  - 当前仍缺“随机四次划分”的生成路径、种子与四组 split 资产固化
  - 当前仍缺“四个模型 + surrogate tau transfer”的 provenance 资产链
- `待核`：`external/PIA` 的 release/source 身份闭环。当前 `origin/main@0d7e08a` 与 README 行锚点已补到 note，但 tag/checksum/file-id 级对齐仍未成立。
- `待核`：strict-clean redo 的当前时态闭环
  - `2026-04-09` strict gate 是 clean historical snapshot
  - `2026-04-10` strict redo 已因 `repo_root dirty` 失败，未来若再做 strict review 必须先清理当前 pycache 漂移

## 5) Paper Protocol Mapping（协议映射，聚焦 provenance）

> 本节只做映射与缺口标注，不复述论文细节；凡未被工作区证据支撑的项一律 `待核`。

| 上游声明/论文语义要点 | 当前本地对应物 | 已知证据 | 仍缺口（blocker） |
| --- | --- | --- | --- |
| “预训练 checkpoint 可下载” | `workspaces/gray-box/assets/pia/sources/OneDrive_1_2026-4-7.zip` | `external/PIA/README.md` 提供入口；本地留存 zip+hash | `待核`：该 zip 与上游入口中的具体文件 ID/版本/校验对齐 |
| “DDPM CIFAR10 checkpoint 文件” | `OneDrive_1_2026-4-7.zip:DDPM/ckpt_cifar10.pt` 与 `.../checkpoint.pt` | 条目存在；条目 hash == canonical hash | `待核`：是否与论文实验所用 checkpoint 同一 release 版本 |
| “repo 提供 split” | `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz` | `external/PIA/README.md` 明确列出；strict gate 固定 hash | `待核`：split 的语义/生成路径与论文协议逐项对齐 |
| “论文中随机四次二分并训练四个模型做 τ transfer” | 当前仅有单个 `CIFAR10_train_ratio0.5.npz` 与单条 canonical 运行线 | paper report line anchors + local code anchors 已补齐 | `待核`：random-four-split generation / four-model provenance |

## 6) 未闭环点（必须明确，禁止口头漂移）

以下任一项未完成，都必须保持 `paper-aligned blocked by checkpoint/source provenance`：

- `待核`：上游 release 对齐证据
  - 本地 zip 是否来自 `external/PIA/README.md` 所指向的 release bundle（可审查字段：文件 ID/版本/校验 任一成立）。
- `待核`：上游 checksum/版本闭环策略
  - 若上游不提供 checksum：团队必须明确“允许什么替代审查证据”才能升 `paper-aligned`（否则永远无法升）。
- `待核`：`external/PIA` release/source 身份
  - strict gate 记录了 `git.commit` 与 `git.dirty=false`
  - `2026-04-10-pia-provenance-upstream-identity-note.md` 已补 `origin/main@0d7e08a` 与 README 行锚点
  - 但 tag/checksum/file-id 级对齐仍未成立
- `待核`：split 文件语义对齐
  - `.npz` 的成员定义/索引语义与论文协议逐项一致性说明。
- `待核`：strict-clean redo 的当前时态闭环
  - 当前 strict redo 已失败，未来若要重新激活 release-review，必须先清理 `external/PIA` 当前 pycache 漂移并重跑 strict gate。

## 7) 允许/禁止话术（必须执行）

允许写：

- `workspace-verified`
- `provisional G-1`
- `paper-aligned blocked by checkpoint/source provenance`

禁止写：

- `paper-aligned defense benchmark`
- `paper-faithful reproduction complete`
- `validated privacy win`
- 任何暗示 “checkpoint/source 已被论文 release 单独核准” 的表述

## 8) 退出条件（二选一）

> 这部分必须是 checklist；结论只能落在 A 或 B 之一，否则 blocker 永远口头化。

### A. 重新激活 `paper-aligned` 审查的条件（当前不是 active review path）

必须全部满足：

- [ ] 上游 release 对齐证据成立（至少一种团队认可的审查方式）：
  - [ ] 上游文件 ID/版本信息可审查
  - [ ] 且能与本地 zip `sha256 = B69310B...` 对齐（或有批准的替代审查证据）
- [ ] `external/PIA` upstream 身份闭环：
  - [ ] 记录 remote/tag/commit，并解释与 strict gate 记录 `git.commit` 的关系
- [ ] split provenance 闭环：
  - [ ] split 文件来源与语义对齐说明成立（同文件或同生成脚本+参数）
- [ ] 复核运行：
  - [ ] 重新跑 strict next-run gate 并留存新的 manifest/provenance
  - [ ] 重新跑一次 runtime mainline 作为最小复核（规模可最小化）

### B. 当前 closed default path：固化为长期 blocker

必须全部满足：

- [x] 明确决策：该线长期只允许 `workspace-verified`
- [x] 明确原因（至少一项）：上游 release 无校验、链接易失、license/分发限制、或其他
- [x] 固化口径：任何文档/状态页必须携带 `paper-aligned blocked by checkpoint/source provenance`

## 9) 决策记录

- `decision_date`: `2026-04-10`
- `decision_owner`: `research_leader`
- `chosen_path`: `B (remain long-term blocker)`
- `rationale`: `当前证据只足以支撑 workspace-verified，不足以支撑 paper-aligned`

## 10) Round-04 Closed Decision

本轮将 `PIA provenance dossier` 正式 closed 为：

- `B = remain long-term blocker`

这是当前唯一安全结论。

### 直接理由

1. 当前 dossier 自身仍把以下项列为 `待核`：
   - 上游 release 对齐
   - checksum / version policy
   - `external/PIA` upstream 身份闭环
   - split 语义对齐
2. `strict next-run provenance gate passed` 只证明：
   - 本地输入可被机器可读固定
   - `2026-04-09 historical snapshot` 在 strict gate 下是 clean
   - 本地 checkpoint 与本地来源包之间的字节一致性可审查
3. 上述证据不足以证明：
   - 导入 checkpoint source 已与论文作者 release bundle 一一对齐
   - 上游 release 版本 / checksum / split 语义已被单独核准

### 本轮固定口径

- 当前 `PIA` 灰盒主讲线已达到 `workspace-verified + adaptive-reviewed`
- 但仍不具备升级为 `paper-aligned` 的 provenance 审查闭环
- 后续统一口径保持：
  - `workspace-verified`
  - `paper-aligned blocked by checkpoint/source provenance`

### 不允许的越界写法

- `PIA has completed paper-aligned / paper-faithful reproduction`
- `strict gate + clean commit + local hash lock prove upstream release identity`
- `checkpoint / source / split have all been paper-verified`

### 对 Phase E 的直接影响

- `PIA paper-aligned confirmation` 不计入当前可释放 GPU 队列
- 文档层可以保留其条件性第 1 顺位
- 但执行层当前必须视为 `no-go`
- 只有在 provenance blocker 发生实质变化并完成单独 release review 后，才允许重新回到候选审查面
- 当前若讨论下一条 GPU 候选，只能先审查：
  - `Finding NeMo + local memorization + FB-Mem`
