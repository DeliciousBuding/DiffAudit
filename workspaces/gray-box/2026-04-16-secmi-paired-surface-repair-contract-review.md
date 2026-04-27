# 2026-04-16 SecMI Paired-Surface Repair Contract Review

## Question

After diagnosing the weak `SecMI 2048` paired packet as a drift-heavy artifact, what exact contract must a repaired paired-surface export follow before another `CDI` promotion decision or `2048` rerun is allowed?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-cdi-paired-surface-mismatch-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-cifar10-gpu-4096-20260415-r1/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-cifar10-gpu-full-stat-20260415-r2/summary.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/analysis.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/disagreement_analysis.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260416-r2/summary.json`
- `<DIFFAUDIT_ROOT>/Research/scripts/run_secmi_pia_disagreement.py`
- `<DIFFAUDIT_ROOT>/Research/src/diffaudit/attacks/secmi.py`
- `<DIFFAUDIT_ROOT>/Research/external/SecMI/mia_evals/dataset_utils.py`

## Review

### 1. The admitted `SecMI` scale story is stable under the mainline contract

- `4096 / 4096` official-style rung:
  - `stat AUC = 0.888575`
  - `t_sec = 100`
- `25000 / 25000` official-style rung:
  - `stat AUC = 0.885833`
  - `t_sec = 100`
- old strong paired `1024` packet:
  - `stat AUC = 0.884247`
  - `T_SEC = 100`

Interpretation:

- current evidence already supports a stable `SecMI stat` line at `1024`, `4096`, and full split
- therefore the repair target should be “return paired export to admitted contract,” not “reinterpret weak `2048` as new scale truth”

### 2. Split drift is not the blocker

- `external/SecMI/mia_evals/member_splits/CIFAR10_train_ratio0.5.npz`
- `external/PIA/DDPM/CIFAR10_train_ratio0.5.npz`

Current verification:

- arrays are equal for `mia_train_idxs`, `mia_eval_idxs`, and `ratio`
- file hashes are also identical

Interpretation:

- using the `PIA` split file versus the `SecMI` split file does not change the actual CIFAR-10 half-split content
- for honesty and provenance, repaired paired export should still prefer the admitted `SecMI` mainline root

### 3. Repair contract must lock the mainline `SecMI stat` surface

Required repaired contract:

1. `t_sec = 100`
2. `timestep = 10`
3. `batch_size = 64`
4. dataset root remains the canonical local CIFAR-10 root:
   - `workspaces/gray-box/assets/pia/datasets`
5. split root should be recorded and frozen to:
   - `external/SecMI/mia_evals/member_splits`
6. paired export should use the official-style `SecMI` attack path:
   - `get_intermediate_results(...)`
   - `execute_attack(..., type="stat")`
7. summary must record asset/runtime provenance explicitly, not only final metrics

### 4. Minimal implementation hardening is justified now

The paired export script should not keep a drift-prone default:

- `run_secmi_pia_disagreement.py` should default to `t_sec = 100`
- it should also add `external/SecMI` itself onto `sys.path`, not only `mia_evals`, so reruns do not depend on ad hoc shell env injection
- its summary should record the repaired contract explicitly

## Verdict

- `secmi_paired_surface_repair_contract_verdict = positive`
- repaired paired export must realign to the admitted `SecMI stat` contract before any new `CDI` promotion decision
- the honest next GPU question is now well-bounded:
  - rerun the `2048` paired export under the repaired contract

## Carry-Forward Rule

- `gpu_release = bounded`
- allowed next GPU task:
  - one repaired `SecMI 2048` paired-surface rerun under the frozen contract above
- do not open any larger-scale follow-up until that repaired rerun lands

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `platform_runtime_handoff = none`
- `competition_material_sync = none`
