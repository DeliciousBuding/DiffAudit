# 2026-04-16 Cross-box Handoff Review

## Task

- `X-4.4` visualization or decision-quality analysis that changes project understanding

## Question

- Do the latest gray-box, black-box, and white-box verdicts change project-level summary wording, system-consumable interpretation, or handoff requirements?

## Scope

- CPU-only review
- no new run
- no new schema unless a real higher-layer consumer now needs one

## Inputs Reviewed

- `docs/admitted-results-summary.md`
- `docs/leader-research-ready-summary.md`
- `docs/comprehensive-progress.md`
- `workspaces/implementation/2026-04-15-attack-defense-matrix.md`
- latest lane verdicts from gray-box / black-box / white-box on `2026-04-16`

## Verdict

- `positive` for summary-layer sync
- `no admitted headline change`

The current project truth is:

1. admitted headline remains unchanged
   - black-box: `recon DDIM public-100 step30`
   - gray-box: `PIA baseline + stochastic-dropout(all_steps)`
   - white-box: `GSA + W-1 strong-v3 full-scale`
2. challenger and boundary layer changed materially and must now be carried consistently
   - gray-box strongest attacker-side challenger remains `TMIA-DM late-window`
   - gray-box strongest defended challenger is now `TMIA + temporal-striding(stride=2)`, which supersedes `TMIA + dropout` as the preferred defended challenger reference
   - black-box `CLiD` wording should be fixed to `evaluator-near local clip-only corroboration`, not generic `local bridge` and not paper-aligned threshold evaluation
   - black-box `variation` should be fixed to `contract-ready blocked`, with explicit unblock contract:
     - `query_image_root + query images`
     - then `endpoint/proxy + budget + frozen parameters`
   - black-box `served-image-sanitization` is now an explicit `no-go`, not merely “not yet explored”
   - white-box defense breadth should be stated explicitly:
     - no second executable defended family exists in-repo beyond `W-1 = DPDM`

## Handoff Decision

- `Leader / materials`: yes
  - refresh summary wording and challenger-layer explanation
  - do not change admitted metrics or replace the main three-line table
- `Platform`: no immediate mandatory sync
  - existing admitted-table schema is still sufficient
  - optional future enhancement only: expose challenger queue fields more explicitly if UI wants non-admitted comparison cards
- `Runtime`: no immediate sync
  - no new runtime field or execution contract is required by this review
- `competition materials`: yes, wording-only sync
  - keep headline metrics fixed
  - add one sentence that gray-box defended challenger has shifted from `TMIA + dropout` to `TMIA + temporal-striding`
  - add one sentence that black-box mitigation has already rejected a first realistic serving-side sanitization idea

## Recommended Wording Guardrails

- Do say:
  - admitted headline is stable
  - challenger layer is evolving
  - black-box mitigation is still not landed, but the first realistic serving-side candidate has already been rejected
- Do not say:
  - `CLiD` is paper-faithful
  - `variation` is merely “blocked-assets” without the explicit reopen contract
  - white-box has multiple defended families already
  - `TMIA + dropout` is still the preferred defended challenger reference

## Next Action Recommendation

1. return to `GB-1` and push a second gray-box defense candidate that is genuinely different from dropout and temporal-striding
2. keep `X-4` open for future `agreement / calibration / portability` sub-directions, but treat the current handoff sync round as complete
