# 2026-04-16 CLiD Threshold-Evaluator Compatibility Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `BB-3 / CLiD boundary-quality upgrade`
- `artifact_scope`: `target-side local clip bridge + released threshold evaluator`
- `device`: `cpu`
- `decision`: `evaluator-near but shadow-blocked`

## Question

Can the current local `CLiD` bridge already be treated as one minimum step closer to the released `cal_clid_th.py` evaluator, or is the boundary still too loose to tighten honestly?

## Executed Evidence

Released evaluator:

- `<DIFFAUDIT_ROOT>/Download/black-box/supplementary/clid-mia-supplementary/contents/CLID_MIA/cal_clid_th.py`

Current executed local rung:

- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/clid-recon-clip-target100-20260415-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/clid-recon-clip-target100-20260415-r1/outputs/Atk_clid_clip_M_local_recon_target_DATA_member_TRTE_train_MAXsmp_1_T_0415_033616.txt`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/clid-recon-clip-target100-20260415-r1/outputs/Atk_clid_clip_M_local_recon_target_DATA_member_TRTE_test_MAXsmp_1_T_0415_033616.txt`

Prepared staged-path bridge:

- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/clid-paper-align-asset-prep-20260415-r1/config.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/runs/clid-paper-align-asset-prep-20260415-r1/mia_CLiD_clip_local.py`

Prior audit:

- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-15-clid-paper-alignment-audit.md`

## Findings

### 1. The current local output matrix is evaluator-shaped

The current target-side rung writes two score files that are already close to the released `cal_clid_th.py` expectation:

- both files contain a first non-numeric header line that can be skipped;
- after skipping that line, both files parse cleanly into numeric matrices of shape `100 x 5`;
- this matches the evaluator's assumption that each line contains one first-column loss term plus several CLiD-condition columns.

### 2. The executed target-side rung still leaks the old cache-root boundary

The first header line of the current target-side local rung still records:

- `diff_path = C:/Users/<user>/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/...`

So even though staged-path preparation now exists separately, the already-executed local benchmark evidence still points at the user cache snapshot rather than the canonical staged SD1.5 root.

### 3. The released threshold evaluator still needs a shadow-side pair

`cal_clid_th.py` is not a pure target-only parser.

It expects four files:

1. shadow train
2. shadow test
3. target train
4. target test

The current local bridge evidence only provides the target-side pair.

So the result is:

- target-side file format is close enough to be called `evaluator-near`;
- but the full released threshold-evaluator path remains blocked because the shadow-side pair does not exist on the current Recon asset family.

### 4. Clip-only remains the only honest near-term boundary

Nothing in this check changes the existing scope decision:

- `clip-only local alignment` is still tractable;
- `importance-clipping` remains blocked on missing token-importance metadata;
- the current honest upgrade path is still boundary tightening, not paper-faithful promotion.

## Verdict

Current verdict:

- `evaluator-near but shadow-blocked`

Reason:

1. the current target-side local CLiD output is already close to the released threshold-evaluator input schema;
2. this is stronger than the earlier generic claim of “local bridge only”, because the numeric artifact shape is now explicitly checked;
3. but the executed evidence still carries a cache-root provenance leak, and the full released evaluator still cannot run honestly without shadow-side files;
4. therefore the line should tighten from generic local corroboration toward `evaluator-near local clip-only corroboration`, but not yet to `paper-aligned local benchmark`.

## Decision

Current boundary decision:

- `keep CLiD below paper-aligned local benchmark`
- `tighten wording to evaluator-near local clip-only corroboration`
- `treat shadow-side evaluator inputs as the next real blocker`

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: if CLiD is mentioned, the strongest honest wording is now `workspace-verified, evaluator-near local clip-only corroboration; full threshold-calibrated CLiD remains shadow-blocked`.
