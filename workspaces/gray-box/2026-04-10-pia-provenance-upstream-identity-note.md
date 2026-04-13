# 2026-04-10 PIA Provenance Upstream Identity Note

## Status Panel

- `owner`: `research_leader`
- `track`: `gray-box`
- `artifact_type`: `provenance supplement / cpu-only`
- `scope`: `release/source/split identity clarification`
- `admission_effect`: `none`
- `paper_alignment_effect`: `none`
- `updated_at`: `2026-04-10 22:13 +08:00`

## Purpose

这份 note 不把 `PIA` 升级为 `paper-aligned`。

它只负责把 provenance dossier 里仍然模糊的三类信息补成可审查事实：

1. `external/PIA` 当前 upstream git 身份
2. README 中关于 split 与 pretrained checkpoint 的公开声明锚点
3. 本地 split 文件的最小语义形状

## Fixed Evidence

### 1. 当前 upstream git 身份

- local repo:
  - `D:\Code\DiffAudit\Research\external\PIA`
- remote:
  - `origin = https://github.com/kong13661/PIA.git`
- local branch:
  - `main`
- local HEAD:
  - `0d7e08a5a07f44931692d52d54d0ce41aff8f54c`
- branch tracking:
  - `main -> origin/main`
- remote heads/tags check:
  - `origin/main = 0d7e08a5a07f44931692d52d54d0ce41aff8f54c`
  - 当前未观察到 tag 输出

这意味着：

- strict gate 里记录的 `git.commit = 0d7e08a...` 已经能与 public `origin/main` 当前 head 对齐
- 但这仍然只是 repo 身份对齐，不是 release/checksum 对齐

### 2. README 的公开声明锚点

`external/PIA/README.md` 中当前可复核的语句包括：

- line `11`
  - 声明 repo 提供 `DDPM/CIFAR10_train_ratio0.5.npz`
- line `101-102`
  - 声明 pretrained checkpoint 可从 `MIA_efficient` OneDrive 入口下载

当前 README 只给出：

- split 文件名
- OneDrive 文件夹入口

当前 README 没给出：

- release tag
- file checksum
- stable file id/version manifest

### 3. 本地 source bundle 与 checkpoint 条目

- source bundle:
  - `workspaces/gray-box/assets/pia/sources/OneDrive_1_2026-4-7.zip`
  - `sha256 = B69310B92AF98B5AFF2D046747DA6650A915C0FC6A79291C999192D43CEF98E5`
- checkpoint entry in zip:
  - `DDPM/ckpt_cifar10.pt`
  - `Length = 574274030`
  - `LastWriteTime = 2026-04-06T16:49:00+08:00`
- canonical checkpoint:
  - `workspaces/gray-box/assets/pia/checkpoints/cifar10_ddpm/checkpoint.pt`
  - `sha256 = 55C3F4545BC345522C402EF9C84627BB87BA53049C9541DDF46A6FEB5B0B283E`

当前允许继续维持的事实是：

- canonical checkpoint 与本地 source bundle 条目之间的字节级对应关系成立

### 4. split 文件的最小语义形状

对 `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz` 的本地检查结果：

- keys:
  - `mia_train_idxs`
  - `mia_eval_idxs`
  - `ratio`
- `mia_train_idxs`
  - `count = 25000`
  - `dtype = int64`
  - `min = 1`
  - `max = 49999`
- `mia_eval_idxs`
  - `count = 25000`
  - `dtype = int64`
  - `min = 0`
  - `max = 49996`
- set relation:
  - `overlap = 0`
  - `union = 50000`
- `ratio = 0.5`

当前这只足以说明：

- split 在本地表现为 CIFAR-10 train set 上的一个互斥 `50/50` 索引划分

当前它仍不足以说明：

- 这些索引一定与论文/作者使用的生成脚本、种子与协议语义完全一致

## Current Caveats

### 1. 现在的 `external/PIA` 工作树不是 clean

当前 `git status --short --branch` 观察到：

- `## main...origin/main`
- untracked:
  - `DDPM/__pycache__/components.cpython-311.pyc`
  - `DDPM/__pycache__/model.cpython-311.pyc`

这不推翻 `2026-04-09` strict gate 的历史结论。

它只说明：

- 如果未来要重新跑 strict gate，必须先清理这两个临时产物
- 当前不能把“此刻的本地 repo 状态”误写成仍然 clean

### 2. upstream identity 只补了 repo 身份，没有补 release 身份

当前已经补到：

- `origin/main`
- `HEAD commit`
- README 的 public statement

当前仍没补到：

- source bundle 的 stable file id/version
- upstream checksum
- release tag / immutable release page

### 3. split 语义只补到结构，不补到 paper-faithful

当前已经补到：

- key 名称
- 样本数量
- 无重叠
- `ratio = 0.5`

当前仍未补到：

- 生成脚本
- 采样种子
- 与论文 protocol text 的逐项对齐

## Decision Impact

这份 note 的作用是：

- 缩小 provenance 中“完全未知”的部分
- 把 `external/PIA upstream identity` 从纯口头项降成可审查项

它不改变当前 closed decision：

- `PIA provenance = remain long-term blocker`

当前 strongest claim 继续保持：

- `workspace-verified + adaptive-reviewed + paper-aligned blocked by checkpoint/source provenance`

## Recommended Carry-Forward Wording

允许写：

- `external/PIA repo identity is now documented against origin/main@0d7e08a`
- `README claims for split file and MIA_efficient checkpoint entry are now line-anchored`
- `split file shape is locally self-consistent as a 50/50 disjoint index partition`

不允许写：

- `upstream release identity is confirmed`
- `OneDrive source bundle is paper-verified`
- `split semantics are paper-faithful`
- `paper alignment blocker is cleared`

## Next Step

1. 把这份 note 纳入 provenance dossier 的 evidence pack
2. 把 `release/source identity` 与 `split semantics` 明确拆成剩余 blocker
3. 继续保持 `paper-aligned blocked by checkpoint/source provenance`
