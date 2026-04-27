# 2026-04-16 CDI Internal Canary Verdict

## Question

After freezing the first `CDI` collection contract, can the repo actually execute one bounded internal `CDI`-style canary on existing gray-box score artifacts without inventing a new training or copyright-claim surface?

## Inputs Reviewed

- `<DIFFAUDIT_ROOT>/Research/scripts/run_cdi_internal_canary.py`
- `<DIFFAUDIT_ROOT>/Research/tests/test_cdi_internal_canary.py`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/outputs/secmi_scores_1024.json`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/2026-04-16-cdi-feature-collection-surface-review.md`
- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/cdi-internal-canary-20260416-r1/audit_summary.json`

## Verification

Unit verification:

- `conda run -n diffaudit-research python -m unittest tests.test_cdi_internal_canary`
- result:
  - `Ran 5 tests ... OK`

Real canary execution:

- `conda run -n diffaudit-research python <DIFFAUDIT_ROOT>/Research/scripts/run_cdi_internal_canary.py --run-root <DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/cdi-internal-canary-20260416-r1 --secmi-scores <DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/secmi-pia-disagreement-20260415-r1/outputs/secmi_scores_1024.json --control-size 512 --test-size 512 --resamples 1 --seed 0`

## What Landed

The first bounded internal `CDI` canary now exists at:

- `<DIFFAUDIT_ROOT>/Research/workspaces/gray-box/runs/cdi-internal-canary-20260416-r1/audit_summary.json`

Emitted artifacts:

- `collections.json`
- `sample_scores.jsonl`
- `audit_summary.json`

Frozen shape:

- surface:
  - `CIFAR-10 DDPM shared-score contract`
- feature mode:
  - `SecMI stat only`
- collection split:
  - `P_ctrl = 512`
  - `P_test = 512`
  - `U_ctrl = 512`
  - `U_test = 512`

Observed statistic:

- `secmi_memberness_orientation = negated`
- `secmi_p_test_mean = -0.00021145`
- `secmi_u_test_mean = -0.00055238`
- `t_statistic = 24.47539`
- `p_value = 1.4084421823808848e-93`

## Verdict

- `internal_cdi_canary_verdict = positive but bounded`
- the repo can now execute one real internal `CDI`-shape canary on already-landed gray-box artifacts
- the first canary remains an internal audit-shape check, not an external copyright claim
- the orientation bug in raw `SecMI` scores is now explicitly normalized into a memberness direction before the Welch test

## Carry-Forward Rule

- keep `SecMI stat only` as the first-canary default
- keep paired `PIA + SecMI` as the next extension, not the initial default
- the highest-value active GPU follow-up remains a larger `PIA` shared-score surface refresh for that paired extension

## Handoff Decision

- `Research/ROADMAP.md`: update required
- `workspaces/gray-box/plan.md`: update required
- `root_sync = none`
- `platform_runtime_handoff = none`
- `competition_material_sync = none`

