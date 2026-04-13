# 2026-04-09 Black-Box Template: Variation Real-Asset Recovery Gate

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-09 +08:00`
- `selected_track`: `variation / Towards`
- `current_state`: `real-API recovery template written`

## A. What Is Actually Missing

Current `variation` is not blocked by code first.

The first blocker is asset completeness:

- missing `query_image_root`
- missing real query images under that root

Secondary blockers that still remain after images exist:

- real variation endpoint or equivalent proxy
- query budget
- a fixed paper-aligned query protocol

So the recovery order must be:

1. query image set
2. endpoint / proxy
3. budget
4. protocol tuning

## B. Minimum Real-Asset Template

`variation` should only be promoted out of blocked state after all of the following exist:

### Query image root

Expected field:

- `assets.dataset_root`

Expected shape:

```text
<query_image_root>/
  member/
    *.png|jpg|jpeg|bmp|webp
  nonmember/
    *.png|jpg|jpeg|bmp|webp
```

If only one flat directory exists, it is enough for an asset probe but not enough for a paper-aligned evaluation claim.

### Endpoint

Expected field:

- `attack.parameters.endpoint`

Allowed current forms:

- real hosted variation API
- local proxy that is behaviorally equivalent to a variation API

Not enough:

- placeholder URL
- generic text-to-image endpoint with no image-to-image variation behavior

### Query budget

This must be written down explicitly before any real run is described as comparable:

- calls per image
- retry policy
- timeout policy
- whether rate limiting or paid quota applies

### Fixed attack parameters

Current config fields that must be fixed before a real run:

- `num_averages`
- `distance_metric`
- `threshold_strategy`

These should be frozen together with the endpoint and budget, not tuned ad hoc after the run.

## C. Recommended Local Config Skeleton

```yaml
task:
  name: variation-plan
  model_family: diffusion
  access_level: black_box
assets:
  dataset_id: variation-query-set-v1
  dataset_root: D:/path/to/variation-query-set
  model_id: variation-api-target
attack:
  method: variation
  num_samples: 0
  parameters:
    endpoint: https://your-variation-endpoint
    num_averages: 10
    distance_metric: l2
    threshold_strategy: threshold
report:
  output_dir: experiments/variation-real-asset-probe
```

Tracked repo template now exists at:

- `configs/attacks/variation_real_asset.template.yaml`

Notes:

- `num_samples` is not the paper claim by itself; it only controls how much the local runner tries to inspect.
- `num_averages` should be fixed to the actual protocol you want to defend in a report.

## D. Recovery Gate Judgment

The line may move from:

- `formal local secondary track + blocked real-API assets`

to:

- `asset-ready`

only after:

1. `query_image_root` exists
2. at least one concrete query image file is present
3. endpoint is real

The line may move toward:

- `runtime-ready` or `paper-aligned candidate`

only after:

1. member/non-member evaluation layout is explicit
2. query budget is written down
3. the chosen `num_averages / distance_metric / threshold_strategy` are frozen

## E. Immediate Next Step

1. Hand this template to anyone claiming they can unblock `variation`.
2. Do not reopen the line with only an endpoint and no query image set.
3. Keep `variation` as a black-box secondary track until this gate is satisfied.
