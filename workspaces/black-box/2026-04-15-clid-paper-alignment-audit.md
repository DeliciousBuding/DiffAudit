# 2026-04-15 CLiD Paper-Alignment Audit

## Scope

This note audits the currently runnable local `CLiD` line against the downloaded supplementary bundle and the intended paper-facing protocol boundary.

Audit targets:

- `external/CLiD/`
- `Download/black-box/supplementary/clid-mia-supplementary/contents/CLID_MIA/`
- local bridge run artifacts under `workspaces/black-box/runs/clid-recon-clip-*`

## Executive Verdict

The current local `CLiD` line is runnable and useful, but it is still a local bridge, not a paper-faithful replication.

It should currently remain classified as:

- `workspace-verified local black-box corroboration`

It should **not** yet be presented as:

- full paper-faithful `CLiD` reproduction
- threshold-calibrated benchmark aligned with the released protocol

## Evidence

### 1. The local runnable script is a bridge, not the untouched upstream script

The current local rung script:

- `workspaces/black-box/runs/clid-recon-clip-target100-20260415-r1/mia_CLiD_clip_local.py`

differs materially from:

- `external/CLiD/mia_CLiD_clip.py`

The local bridge explicitly changes:

- `Use_data_model_name` to a local synthetic label;
- `diff_path` to a local SD1.5 snapshot;
- train/test dataset roots to exported local member/non-member folders;
- `unet` loading by injecting Recon LoRA attention processors;
- dataset loading logic to allow local `imagefolder`;
- output directory and batch size;
- subset sizing and attack branching.

This is enough to prove local executability, but it also proves the current line is a ported bridge.

### 2. The current model path was originally bound to user cache, not to a canonical staged root

The prepared local rung config recorded:

- base model:
  - `C:/Users/<user>/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/...`

That means the bridge initially depended on a user Hugging Face cache snapshot instead of a canonical repo-declared staged model root.

This has now been corrected at the asset-pointer level by:

- `configs/assets/staged-downloads.local.yaml`

but the runnable CLiD bridge itself has not yet been repointed.

### 3. Dataset protocol is not aligned with the released fine-tuning setting

The supplementary README describes a fine-tuning setting around:

- target/shadow models;
- released attack scripts;
- MS-COCO validation outputs;
- evaluation via `cal_clid_th.py` and `cal_clid_xgb.py`.

The current local line instead uses:

- Recon CelebA asset-family member/non-member samples;
- exported `dataset.pkl -> imagefolder + metadata.jsonl`;
- target-side local rung on `100 / 100` or smoke-scale subsets.

This is a valid local corroboration setup, but it is not the same protocol as the released CLiD real-setting path.

### 4. Variant coverage is narrower than the supplementary release

The local repo and the downloaded supplementary bundle are not byte-identical and do not expose the same file naming.

File-level difference:

- only in `external/CLiD/`:
  - `mia_CLiD_clip.py`
  - `mia_CLiD_impt.py`
  - `mia_loss_sd.py`
  - `LICENSE`
- only in downloaded supplementary:
  - `mia_Loss.py`
  - `mia_mydeN_impt.py`
  - `mia_mydeNoise.py`
  - `mia_SEC_PIA.py`

This suggests the current local copy is a cleaned or reorganized branch of the method family, not a strict frozen mirror of the downloaded bundle.

### 5. Current evidence is target-family corroboration, not calibrated paper-aligned evaluation

The current local note already states:

- both target-family local rungs are perfectly separated;
- no shadow-member rung is available from the current Recon asset bundle;
- the honest boundary is still local corroboration.

That means the current line proves signal portability, but not protocol alignment.

## Minimum Missing Pieces To Honestly Tighten The Boundary

These are the smallest next steps that would improve boundary quality without pretending to have completed the whole paper protocol.

### A. Repoint the base model to the staged local SD1.5 root

Switch the bridge from:

- user Hugging Face cache snapshot

to:

- `<DIFFAUDIT_ROOT>/Download/shared/weights/stable-diffusion-v1-5`

This is a hygiene and reproducibility fix, not a scientific upgrade, but it is mandatory.

### B. Freeze one explicit alignment target

Before changing more code, decide whether the next honest target is:

1. `paper-aligned local clip-only rung`, or
2. `full released-protocol reproduction`

Given current assets, the realistic near-term target is `paper-aligned local clip-only rung`.

### C. Port one released supplementary variant directly instead of only using the already-local branch

Use the downloaded supplementary root:

- `Download/black-box/supplementary/clid-mia-supplementary/contents/CLID_MIA`

and choose one concrete step:

- run or port `mia_mydeNoise.py` / `mia_mydeN_impt.py` semantics directly;
- or document exactly why `external/CLiD/mia_CLiD_clip.py` is the maintained local equivalent.

Without this, the boundary remains “our local CLiD-style bridge”, not “paper-aligned local CLiD”.

### D. Align evaluation output with released evaluators

The supplementary bundle provides:

- `cal_clid_th.py`
- `cal_clid_xgb.py`

The next honest upgrade is to ensure the local bridge emits outputs that can be consumed by one of these evaluators, or to document the exact incompatibility.

### E. Decide clip-only versus importance-clipping scope

The current smoke note already found that:

- `clid_impt` is blocked by missing token-importance metadata in the current Recon asset family.

So the honest near-term statement is:

- `clip-only local alignment is tractable now`
- `importance-clipping alignment remains blocked on metadata`

That should be written explicitly instead of leaving the method family boundary ambiguous.

## Recommended Next Upgrade Step

The best single next step is:

1. repoint the local CLiD bridge to the staged SD1.5 root;
2. keep the current CelebA local rung setup;
3. emit outputs in a form consumable by `cal_clid_th.py`;
4. decide whether that is enough to promote the line from:
   - `local corroboration`
   - to `paper-aligned local clip-only benchmark`

## Executed Upgrade Step

This note is no longer purely hypothetical.

A concrete paper-alignment hygiene step has now been executed via:

- `scripts/prepare_clid_local_bridge.py`
- prepared run root:
  - `workspaces/black-box/runs/clid-paper-align-asset-prep-20260415-r1`

What changed:

- the local CLiD bridge now reads staged shared assets from:
  - `configs/assets/staged-downloads.local.yaml`
- the localized runnable script now points to:
  - `<DIFFAUDIT_ROOT>/Download/shared/weights/stable-diffusion-v1-5`
- the prepared run records:
  - staged SD1.5 root
  - staged CLIP root
  - downloaded supplementary root
  - local Recon LoRA root

This is a real executed alignment step, but it is still a preparation-stage upgrade rather than a new benchmark verdict.

## Non-Goals For The Next Step

Do not treat the following as the immediate next paper-alignment step:

- another same-family target rung;
- another smoke on the same exported subsets;
- full importance-clipping support before metadata exists;
- a large GPU rerun before the boundary is cleaner.

## Promotion Decision

After the executed staged-path upgrade, the current line should still remain:

- `workspace-verified local black-box corroboration`

It should **not** yet be promoted to:

- `paper-aligned local benchmark`

Reason:

- the executed step improved reproducibility and boundary hygiene;
- it did not yet align evaluation with `cal_clid_th.py` / `cal_clid_xgb.py`;
- it did not yet add shadow-side or threshold-calibrated protocol evidence;
- it did not yet close the clip-only versus importance-clipping gap.

## Current Status Against Roadmap

- `P0-CL-1`: satisfied by this audit note
- `P0-CL-2`: satisfied by the minimum-missing-pieces section
- `P0-CL-3`: satisfied by the executed staged-path bridge preparation run
- `P0-CL-4`: satisfied by the promotion decision above
