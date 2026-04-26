# 2026-04-16 White-Box GSA2 Bounded Comparator Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `WB-2.3`
- `selected_path`: `GSA2 comparator`
- `track`: `white-box`
- `comparator_status`: `completed`

## Question

After the target pair and first shadow pair were all extracted successfully under `attack_method = 2`, does `GSA2` deserve promotion as a real second white-box line, or should it be closed as non-promotable?

## Executed Evidence

Primary bounded comparator run:

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa2-bounded-comparator-shadow01-20260415-r1\summary.json`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa2-bounded-comparator-shadow01-20260415-r1\attack-output.txt`

Gradient artifacts consumed by the comparator:

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-gradient-direct-attackmethod2-20260415-r2\target_member-gradients.pt`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-gradient-direct-attackmethod2-20260415-r3\target_nonmember-gradients.pt`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-gradient-direct-attackmethod2-20260416-shadow01-member-r1\shadow01_member-gradients.pt`
- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-gradient-direct-attackmethod2-20260416-shadow01-nonmember-r1\shadow01_nonmember-gradients.pt`

Admitted mainline reference:

- `D:\Code\DiffAudit\Research\workspaces\white-box\runs\gsa-runtime-mainline-20260409-cifar10-1k-3shadow-epoch300-rerun1\summary.json`

## Metrics

Bounded `GSA2` comparator:

- `AUC = 0.922498`
- `ASR = 0.839`
- `TPR@1%FPR = 0.407`
- `TPR@0.1%FPR = 0.171`

Admitted `GSA1 1k-3shadow` mainline:

- `AUC = 0.998192`
- `ASR = 0.9895`
- `TPR@1%FPR = 0.987`
- `TPR@0.1%FPR = 0.432`

## Verdict

Current verdict:

- `positive secondary line`

Interpretation:

1. `GSA2` is a real, runnable, high-signal white-box attack variant on the admitted asset line, not a broken side branch.
2. Even with a bounded contract of one shadow pair plus reduced extraction budget (`ddpm_num_steps = 20`, `sampling_frequency = 2`), it remains materially above random and operationally meaningful.
3. However, it stays clearly below the admitted `GSA1 1k-3shadow` mainline on all recorded metrics.
4. Because it is still the same `GSA` family and not a genuinely different white-box family, it should be promoted only as a corroboration line, not as a replacement headline or a new distinct benchmark family.

The claim that this is a useful corroboration line rather than a new headline is an inference from the recorded metrics plus the narrower bounded contract.

## Boundary

What this verdict closes:

- `WB-2` no longer needs more canary-only work.
- `GSA2 comparator` is no longer in “maybe broken” status.

What this verdict does **not** claim:

- that `GSA2` beats admitted `GSA1`;
- that `GSA2` is a new white-box family;
- that this one-shadow bounded run is a paper-grade final benchmark.

## Carry-Forward Rule

1. keep `GSA1 1k-3shadow` as the admitted white-box headline;
2. keep `GSA2 bounded comparator` as a secondary corroboration line;
3. only reopen `GSA2` for more scale if there is a specific bounded reason, such as:
   - cost-vs-strength comparison against `GSA1`
   - multi-shadow expansion for the same `attack_method = 2` path
4. move the next white-box budget to `WB-3` or cross-box/system-consumable packaging instead of more blind canaries.

## Handoff Note

- `Platform` does not need a new surface immediately.
- `Runtime` may later expose `attack_variant = gsa2-bounded` if comparative white-box variants become a first-class concept.
- Material-layer wording can safely say that white-box risk is robust across both the admitted `GSA1` mainline and a bounded `GSA2` variant, but should not call `GSA2` the primary benchmark.
