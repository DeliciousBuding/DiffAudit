# 2026-04-16 Cross-box Agreement Analysis

## Task

- `X-4.1` cross-box agreement analysis

## Question

- After the recent black-box, gray-box, and white-box closures, what do the three boxes now genuinely agree on, where do they disagree, and does that change the next project-level move?

## Inputs Reviewed

- `docs/admitted-results-summary.md`
- `docs/leader-research-ready-summary.md`
- `docs/comprehensive-progress.md`
- `workspaces/implementation/2026-04-15-attack-defense-matrix.md`
- `workspaces/implementation/2026-04-16-crossbox-handoff-review.md`
- current `black-box / gray-box / white-box` workspace plans

## Agreement

The current three-box agreement is:

1. leakage risk is real at every access level
   - black-box already shows a stable risk proof
   - gray-box already shows a stable attack-defense story
   - white-box already shows a strong upper-bound attack plus a defended contrast
2. admitted headline remains unchanged
   - black-box: `recon DDIM public-100 step30`
   - gray-box: `PIA + stochastic-dropout(all_steps)`
   - white-box: `GSA + W-1 strong-v3 full-scale`
3. challenger/boundary layer is evolving faster than admitted headline
   - `TMIA late-window`
   - `TMIA + temporal-striding`
   - `CLiD evaluator-near`
   - `variation contract-ready blocked`
4. there is still no new GPU admission question forced by the latest closures
   - current high-value work is mostly decision-quality and candidate-generation work

## Disagreement

The current cross-box disagreement is:

1. the strongest story differs by access level
   - black-box is strongest as `risk exists`
   - gray-box is strongest as `attack + defense narrative`
   - white-box is strongest as `upper-bound / depth`
2. headline metrics and low-FPR behavior do not fully agree inside gray-box
   - `PIA` remains the safest headline on global attack-defense packaging
   - `TMIA-DM late-window` remains the stronger low-FPR challenger
3. defense maturity is asymmetric
   - black-box still lacks a landed defense line
   - gray-box has one admitted defense story plus a stronger defended challenger
   - white-box has one admitted defended family but no breadth
4. breadth and depth diverge across boxes
   - black-box has challenger diversity but weak defense maturity
   - gray-box has the best narrative balance
   - white-box has the strongest attack depth but weakest defense diversity

## Project-Level Reading

Current best project-level reading:

- keep the three-box narrative layered instead of forcing one universal “best method”
- treat `black-box = existence proof`, `gray-box = main story`, `white-box = depth/upper bound`
- keep admitted headline stable
- let challenger/boundary layer keep moving underneath it

## Verdict

- `positive`

This analysis changes project understanding enough to be worth recording: the repo is no longer mainly bottlenecked by “missing one more run,” but by keeping a layered, box-specific story honest while challenger/boundary evidence evolves underneath a stable admitted headline.

## Handoff Decision

- `Leader / materials`: yes, wording-only
  - emphasize the layered role split across black / gray / white
  - do not collapse them into one single “best method” story
- `Platform`: optional only
  - if future UI wants it, expose separate fields for `headline role` and `challenger role`
- `Runtime`: no sync needed
- `GPU`: no new GPU admission question is justified by this analysis alone

## Next Recommendation

1. do not reopen recently closed box-local micro-branches by reflex
2. next live work should bias to either:
   - candidate generation for a genuinely new family/defense, or
   - remaining repo-health cleanup such as `INF-2.3`
