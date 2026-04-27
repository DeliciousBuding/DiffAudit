# 2026-04-21 Research Storage Boundary Audit

## Question

Does the current `Research` repository actually follow one coherent storage boundary across `external/`, `third_party/`, `Download/`, `workspaces/*/assets/`, and `workspaces/*/runs/`, or are there real inconsistencies that should be treated as cleanup targets?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/README.md`
- `<DIFFAUDIT_ROOT>/Research/AGENTS.md`
- `<DIFFAUDIT_ROOT>/Research/docs/storage-boundary.md`
- `<DIFFAUDIT_ROOT>/Research/docs/README.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/README.md`
- live directory state under:
  - `<DIFFAUDIT_ROOT>/Research/external`
  - `<DIFFAUDIT_ROOT>/Research/third_party`
  - `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/assets`
  - `<DIFFAUDIT_ROOT>/Research/workspaces/white-box/assets`
  - `<DIFFAUDIT_ROOT>/Download`

## Findings

### 1. The intended boundary is now clear, but live state is still mixed

The intended rule is coherent:

- `external/` = upstream or exploratory code clones
- `third_party/` = minimal vendored code actually maintained in-repo
- `Download/` = raw downloaded datasets / weights / supplementary bundles
- `workspaces/<lane>/assets/` = lane-normalized admitted asset entrypoints
- `workspaces/<lane>/runs/` = evidence

But the live tree still contains boundary drift.

### 2. Real inconsistency: `external/SecMI` and `third_party/secmi` coexist

Current state:

- `external/SecMI` exists as a full upstream clone with `.git`
- `third_party/secmi` also exists as the minimal vendored subset actually referenced by current repo guidance

This is acceptable only if one is clearly exploratory and the other is clearly canonical. Right now that distinction is not visible enough from the tree itself.

### 3. Real inconsistency: `external/recon-assets` is asset-heavy, not code-heavy

Current state:

- `external/recon-assets` is about `4.82 GB`
- its children are asset bundles such as:
  - `ndss-2025-blackbox-membership-inference-fine-tuned-diffusion-models`
  - `public-kandinsky-pokemon`

This is the strongest boundary violation in the current repo layout:

- it lives under `external/`
- but it is clearly an asset bundle, not an upstream code repository

By the current storage rule, this belongs to a raw asset layer like `Download/` or to a lane asset gateway, not to `external/`.

### 4. Real inconsistency: `external/CLiD` mixes code and supplementary outputs

Current state:

- `external/CLiD` contains a git clone and executable upstream files
- but it also contains `inter_output/`, `poster/`, and `train_sh/`
- meanwhile `Download/black-box/supplementary/clid-mia-supplementary/...` already stores a CLiD supplementary bundle

So CLiD currently spans both:

- upstream code clone
- downloaded supplementary content

That split can be valid, but only if the repo clearly distinguishes:

- canonical upstream code root
- raw supplementary mirror
- lane-local normalized artifacts

Right now the tree shape still feels duplicated and under-explained.

### 5. Real inconsistency: `workspaces/README.md` is stale

It still says:

- experiment outputs go to `experiments/`

But current repo truth already uses:

- `workspaces/<lane>/runs/`

It also lags behind the current workspace set by omitting newer active directories such as:

- `cross-box/`
- `defense/`
- `intake/`
- `runtime/`

So one source of confusion is not only directory shape, but stale entry docs.

## Verdict

`positive`

The current storage policy is reasonable, but the live repo is not yet uniformly aligned to it.

The sharpest cleanup targets are:

1. clarify `external/SecMI` vs `third_party/secmi`
2. move or explicitly reclassify `external/recon-assets`
3. clarify `external/CLiD` vs `Download/.../clid-mia-supplementary`
4. repair stale entry docs like `workspaces/README.md`

## Immediate Cleanup Recommendation

Without doing risky bulk moves yet, the next honest cleanup pass should be:

1. keep `external/` for code clones only
2. move `external/recon-assets` out of `external/` into a raw asset location under `Download/black-box/`
3. keep `third_party/secmi` as the canonical in-repo vendored surface
4. leave `external/SecMI` only if it is explicitly treated as exploratory upstream reference
5. keep raw CLiD supplementary in `Download/`
6. keep lane-consumed manifests and admitted asset gateways under `workspaces/*/assets/`

## Canonical Evidence Anchor

- `<DIFFAUDIT_ROOT>/Research/workspaces/implementation/2026-04-21-research-storage-boundary-audit.md`
