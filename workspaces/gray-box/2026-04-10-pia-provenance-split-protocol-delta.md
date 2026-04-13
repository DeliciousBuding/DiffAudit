# 2026-04-10 PIA Provenance Split / Protocol Delta Note

## Status Panel

- `owner`: `research_leader`
- `track`: `gray-box`
- `artifact_type`: `provenance supplement / cpu-only`
- `scope`: `split semantics + paper protocol mapping + strict redo hygiene`
- `admission_effect`: `none`
- `paper_alignment_effect`: `none`
- `gpu_release`: `none`
- `updated_at`: `2026-04-10 23:05 +08:00`

## Purpose

这份 note 不把 `PIA` 升级为 `paper-aligned`。

它只补当前 provenance 中最值得收口的四个问题：

1. `2026-04-09` 的 strict gate 只能算历史 clean snapshot，不能继续被误写成“当前 repo 仍 clean”
2. 本地 source bundle 更像 `OneDrive folder export`，而不是带 immutable manifest 的 release artifact
3. 本地 `ratio0.5` split 的代码消费路径已经可以 line-anchor 到 `main.py / dataset_utils.py`
4. 论文里 `random-four-split + four-model tau transfer` 的协议面，与当前本地单固定 split 实现仍有明确差值

## 一、Round-29 Strict Redo

### Command

```powershell
powershell -ExecutionPolicy Bypass -File tools/pia_next_run/run.ps1 `
  --config tmp/configs/pia-cifar10-graybox-assets.local.yaml `
  --member-split-root external/PIA/DDPM `
  --repo-root external/PIA `
  --out-dir workspaces/gray-box/assets/pia/next-run-20260410-round29-strict-redo `
  --strict
```

### Outputs

- `workspaces/gray-box/assets/pia/next-run-20260410-round29-strict-redo/manifest.json`
- `workspaces/gray-box/assets/pia/next-run-20260410-round29-strict-redo/provenance.json`

### Result

- `OK = false`
- `validation.ok = false`
- `git.commit = 0d7e08a5a07f44931692d52d54d0ce41aff8f54c`
- `git.dirty = true`
- `validation.errors = ["repo_root is dirty"]`
- `manifest_sha256 = de0dd68004c9ffbe5a57605adacfbb73219099cdb8fb5effc9f9121406a4d889`

当前 dirty 的直接来源可复核为：

- `external/PIA/DDPM/__pycache__/components.cpython-311.pyc`
- `external/PIA/DDPM/__pycache__/model.cpython-311.pyc`

### Interpretation

这次 strict redo 的价值，是把下面两句话明确切开：

- `2026-04-09 strict gate passed under a clean repo snapshot`
- `2026-04-10 strict redo fails because the current repo is dirty`

因此允许写：

- `historical strict gate passed`
- `current strict redo is not clean and must not be reused as a present-tense clean claim`

不允许写：

- `strict provenance gate is still clean right now`
- `strict redo failure changes the blocker class or clears the blocker`

## 二、Source Bundle Shape

当前 source bundle：

- `workspaces/gray-box/assets/pia/sources/OneDrive_1_2026-4-7.zip`
- `sha256 = B69310B92AF98B5AFF2D046747DA6650A915C0FC6A79291C999192D43CEF98E5`

当前可直接观察到的顶层 checkpoint 条目只有四个：

1. `DDPM/ckpt_cifar10.pt`
2. `DDPM/ckpt_tini.pt`
3. `gradtts/libritts_grad.pt`
4. `gradtts/ljspeech_grad.pt`

当前没有观察到：

- immutable manifest
- stable file id / version record
- upstream checksum sheet
- release tag / version page

因此当前 source bundle 最多支撑：

- `retained local source bundle`

而不能支撑：

- `paper-faithful release artifact identity confirmed`

## 三、Split Semantics 的代码锚点

### README 声明

`external/PIA/README.md` 当前公开声明：

- line `11`
  - 提供 `DDPM/CIFAR10_train_ratio0.5.npz`
- line `23-38`
  - DDPM attack path 的输入是 `checkpoint / dataset / attack_num / interval`
- line `101-102`
  - pretrained checkpoint 从 `MIA_efficient` OneDrive 入口下载

### 本地代码消费路径

`external/PIA/DDPM/main.py`：

- line `107-109`
  - `np.load('./{FLAGS.dataset.upper()}_train_ratio0.5.npz')`
  - 训练只消费 `mia_train_idxs`

`external/PIA/DDPM/dataset_utils.py`：

- line `45-47`
  - 读取 `CIFAR10_train_ratio0.5.npz`
  - 取出 `mia_train_idxs` 与 `mia_eval_idxs`
- line `55-58`
  - `mia_train_idxs -> member_set`
  - `mia_eval_idxs -> nonmember_set`

### 当前本地 split 形状

当前 `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz` 已确认：

- `mia_train_idxs.count = 25000`
- `mia_eval_idxs.count = 25000`
- `overlap = 0`
- `union = 50000`
- `ratio = 0.5`

因此当前只允许写：

- `split shape aligned locally as a disjoint 50/50 partition`

## 四、Paper Protocol Mapping Delta

论文整理稿当前给出的 CIFAR10 相关协议锚点包括：

- line `252`
  - `Impact of t` 实验里写到“随机划分 CIFAR10 四次并比较各 partition”
- line `256`
  - `Determining the value of τ` 写到“随机二分四次，训练四个模型，选一个 surrogate，再用其 τ 攻击其他三个 victim”
- line `277`
  - sample-number 实验建立在“每个 split 一半训练、一半 hold-out”的结构上

当前本地 line 与论文协议的关系应拆成两层：

### 已对齐的部分

- 本地确实使用 `ratio = 0.5` 的互斥 train/hold-out split
- 本地代码确实显式区分了 `mia_train_idxs` 与 `mia_eval_idxs`

### 仍未对齐的部分

- 当前只看到一个固定 `CIFAR10_train_ratio0.5.npz`
- 当前没有把“随机四次划分”的生成路径、种子、四组 split 资产固定下来
- 当前没有把“四个模型 + surrogate tau transfer”对应的 provenance 资产链固定为 dossier evidence

因此这轮应把 blocker 从宽泛的：

- `split semantics still open`

收紧为：

- `split shape aligned locally`
- `paper-faithful random-four-split / four-model tau-transfer protocol still open`

## 五、当前允许 / 禁止话术

允许写：

- `PIA remains workspace-verified at the local asset level`
- `local split shape and code consumption are now line-anchored`
- `the current strict redo proves the repo is not strict-clean at present time`
- `paper protocol remains open at the random-four-split / four-model tau-transfer level`

禁止写：

- `split semantics are paper-faithful`
- `random-four-split protocol is verified`
- `strict redo failure resolves provenance`
- `source bundle release identity is confirmed`

## 六、Decision Impact

这份补件不改变当前 closed decision：

- `PIA provenance = remain long-term blocker`

但它把 blocker 的剩余部分收紧成更可审查的两类：

1. `release/source identity unresolved`
2. `CIFAR10 protocol mismatch vs local single-split implementation`

同时保留一个 hygiene 前置条件：

- 未来若要重启 strict review，必须先清理当前 `external/PIA` 工作树里的 pycache 漂移

当前 strongest claim 继续保持：

- `workspace-verified + adaptive-reviewed + paper-aligned blocked by checkpoint/source provenance`

## 七、Next Step

1. 把这份 note 纳入 provenance dossier 的 evidence pack
2. 在 dossier 中把 `split semantics` 进一步改写成 `split shape aligned locally but paper protocol still open`
3. 在 provenance 条件未发生实质变化前，继续保持：
   - `execution-layer status = no-go`
   - `gpu_release = none`
