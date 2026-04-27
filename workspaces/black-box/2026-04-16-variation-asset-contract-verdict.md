# 2026-04-16 Variation Asset-Contract Verdict

## Status Panel

- `owner`: `ResearcherAgent`
- `task_scope`: `black-box variation recovery contract`
- `selected_track`: `variation / Towards`
- `device`: `cpu`
- `decision`: `contract-ready but still blocked`

## Question

Is `variation` still blocked in a vague “missing assets” sense, or has the repo now specified a concrete recovery contract that should govern any future unblock attempt?

## Executed Evidence

Current blocked note:

- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-09-variation-blocked.md`

Current recovery template:

- `<DIFFAUDIT_ROOT>/Research/workspaces/black-box/2026-04-09-variation-real-asset-template.md`
- `<DIFFAUDIT_ROOT>/Research/configs/attacks/variation_real_asset.template.yaml`

## Verdict

Current verdict:

- `contract-ready but still blocked`

Reason:

1. the repo no longer lacks clarity about what `variation` needs in order to resume;
2. the real blocker is now explicitly specified as a recovery contract rather than a vague “blocked-assets” label;
3. the first hard gate is still `query_image_root / query images`, not endpoint wiring or runner code;
4. after images exist, the next mandatory gates are:
   - real variation endpoint or equivalent proxy
   - explicit query budget
   - frozen `num_averages / distance_metric / threshold_strategy`

## Decision

Current decision:

- `keep variation as formal local secondary track`
- `do not reopen with endpoint-only claims`
- `only reopen after the contract fields are filled with real values`

Practical unblock rule:

1. `assets.dataset_root` must point to a real query image set;
2. that set should ideally expose `member/` and `nonmember/` splits, not just a flat directory;
3. `attack.parameters.endpoint` must be a real variation API or behaviorally equivalent proxy;
4. the run must carry a written query budget before it can be described as runtime-ready.

## Handoff Note

- `Platform`: no direct change needed.
- `Runtime`: no direct change needed.
- Materials: if `variation` is mentioned, the strongest honest wording is `formal local secondary track with a contract-ready unblock path, still blocked on real query images and endpoint/budget completion`.
