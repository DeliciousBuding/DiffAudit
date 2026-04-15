# 2026-04-16 White-Box GSA2 Comparator Canary Progress

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `WB-2.1 / WB-2.2`
- `selected_path`: `GSA2 comparator`
- `track`: `white-box`
- `gpu_status`: `completed bounded canaries`

## Question

After `WB-1` removed the vague “gradient extraction may be broken” blocker and target-side `attack_method = 2` canaries succeeded, does the same direct `GSA2` extraction contract remain live on shadow-side admitted assets?

## Executed Evidence

Target-member direct extraction:

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-gradient-direct-attackmethod2-20260415-r2\target_member-gradients.pt`

Target-nonmember direct extraction:

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-gradient-direct-attackmethod2-20260415-r3\target_nonmember-gradients.pt`

Shadow-01-member direct extraction:

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-gradient-direct-attackmethod2-20260416-shadow01-member-r1\shadow01_member-gradients.pt`

Shared canary contract:

- upstream entrypoint:
  - `workspaces/white-box/external/GSA/DDPM/gen_l2_gradients_DDPM.py`
- admitted asset root:
  - `workspaces/white-box/assets/gsa-cifar10-1k-3shadow-epoch300-rerun1`
- `attack_method = 2`
- `sampling_frequency = 2`
- `ddpm_num_steps = 20`
- `prediction_type = epsilon`
- no forced `--dataset_name cifar10` on imagefolder-style assets
- output parent directory created before `torch.save(...)`

## Verdict

Current verdict:

- `positive progress`

Interpretation:

1. `GSA2 comparator` remains execution-eligible after moving beyond target-side only;
2. the direct extraction contract is now proven on:
   - target-member
   - target-nonmember
   - shadow-01-member
3. the remaining near-term gating step is no longer “does shadow-side work at all,” but “can the same contract be completed for a first shadow pair and then judged for comparator value?”

## Boundary

This is still **not** a formal second white-box line verdict.

What is now proven:

- target-side `attack_method = 2` extraction works on both member and non-member splits;
- shadow-side direct extraction also works on at least one admitted member split under the same contract.

What is not yet proven:

- first shadow pair completion (`shadow-01-nonmember`);
- bounded comparator quality versus admitted `GSA1`;
- promotion into a formal second white-box mainline.

## Next Best Move

1. run `shadow-01-nonmember` under the same direct extraction contract;
2. if that succeeds, assemble the first bounded `GSA2` comparator verdict instead of extending canaries indefinitely;
3. do not reopen `Finding NeMo` or `Local Mirror` while `GSA2` remains the only live bounded second-line branch.

## Handoff Note

- No immediate `Platform` or `Runtime` handoff is needed from this canary extension alone.
- Material-layer wording should still describe `GSA2` as `live comparator candidate`, not as an admitted second white-box benchmark.
