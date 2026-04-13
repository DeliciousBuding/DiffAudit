# 2026-04-08 White-Box Follow-Up: DPDM Strong-v3 Training Rung

## Status Panel

- `owner`: `research_leader`
- `updated_at`: `2026-04-08 12:00:00 +08:00`
- `selected_defense`: `W-1 = DPDM`
- `current_state`: `target complete; shadow-02 complete; shadow-03 complete; shadow-01 pending`
- `gpu_usage`: `single GPU`
- `evidence_level`: `runtime-smoke`

## A. Training Profile

- `run_suffix = strong-v3`
- `epochs = 5`
- `loss.n_noise_samples = 2`

## B. Current Assets

- target checkpoint:
  - `external/DPDM/runs/dpdm-cifar10-targetmember-eps10-gpu-strong-v3/checkpoints/final_checkpoint.pth`
- shadow-02:
  - completed under `runs/dpdm-cifar10-shadow02-eps10-gpu-strong-v3`
- shadow-03:
  - completed under `runs/dpdm-cifar10-shadow03-eps10-gpu-strong-v3`
- shadow-01:
  - not started in this rung yet

## C. Execution Notes

- This rung originally exposed a sequencing bug: the watcher was waiting on the launcher PowerShell PID instead of the real training child PID.
- The bug has been corrected by introducing `scripts/launch_dpdm_target_and_shadows.ps1`.
- The mis-launched shadow process was stopped, and the watcher was rebound to the real target training PID before shadow execution resumed.

## D. Next Step

- train `shadow-01 strong-v3`
- then run the next defended comparator on the complete strong-v3 checkpoint set
