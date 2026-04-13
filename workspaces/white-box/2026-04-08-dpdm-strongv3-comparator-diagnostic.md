# 2026-04-08 White-Box Diagnostic: DPDM Strong-v3 Comparator

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 13:45:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `CPU diagnostic complete; GPU comparator still unstable`
- `evidence_level`: `runtime-smoke`

## A. What Was Tested

- GPU path:
  - `strong-v3` three-shadow full comparator was rerun multiple times
  - diagnostics were added through `progress.log` and more aggressive cache release
- CPU path:
  - same `strong-v3` target + shadow checkpoints
  - `device=cpu`
  - `sigma_points=4`
  - `max_samples=16`

## B. CPU Diagnostic Result

Workspace:

- `workspaces/white-box/runs/dpdm-w1-multi-shadow-comparator-targetmember-strongv3-cpu-diagnostic-20260408`

Metrics:

- `AUC = 0.523438`
- `ASR = 0.53125`
- `TPR@1%FPR = 0.0`

## C. Conclusion

- The `strong-v3` checkpoint set itself is readable and executable.
- The current blocker is specifically the GPU comparator runtime path.
- This narrows the problem from “asset corruption” to “GPU execution stability under multi-shadow full scoring”.

## D. Next Step

- keep the CPU diagnostic as evidence that the asset set is valid
- continue narrowing the GPU comparator instability before claiming a final `strong-v3` defended result
